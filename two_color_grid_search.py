
from copy import deepcopy
from dxtbx.model.experiment_list import ExperimentList, Experiment
from dials.algorithms.indexing.basis_vector_search.real_space_grid_search import RealSpaceGridSearch
from scitbx import matrix
from dials.algorithms.indexing.basis_vector_search import optimise
from dials.algorithms.indexing.lattice_search import BasisVectorSearch
from dials.algorithms.indexing.symmetry import SymmetryHandler
import numpy as np
from dials.array_family import flex


def block_sum(rec_pts, ul, block_sz=50):
    """
    this function is used by the two_color_grid_search
    it helps compute functional values both fast
    and in a memory efficient way
    rec_pts, N x 3 array of reciprocal lattice vectors
    ul, M x 3 array of basis vectors on the hemisphere scaled
      by there magnitudes, for the grid search
    block_sz, how many RLPs to dot product at one time
      speed is gained using dot product however it consumes
      more memory by creating the intermediate N x M sized
      array...
    returns the M-sized array of Gildea-defined functional
      values for the grid search
    """
    block_rec_pts = np.array_split(rec_pts, max(1,rec_pts.shape[0]/block_sz))
    func_vals = np.zeros(ul.shape[0])
    for block_rlp in block_rec_pts:
        func_vals += np.sum(np.cos(2*np.pi*np.dot(block_rlp, ul.T)), axis=0)
    return func_vals


class TwoColorGridSearch(RealSpaceGridSearch):

    def __init__(self, spiral_seed=None, noise_scale=1.25, Nsp=250000, block_size=25,
                 verbose=False, *args, **kwargs):
        super(TwoColorGridSearch, self).__init__(*args, **kwargs)
        self.cell_dimensions = self._target_unit_cell.parameters()[:3]
        self.spiral_seed = spiral_seed
        self.noise_scale = noise_scale
        self.Nsp = Nsp
        self.verbose = verbose
        self.unique_cell_dimensions = set(self.cell_dimensions)
        self.block_size = block_size

    @property
    def search_directions(self):
        """Generator of the search directions (i.e. vectors with length 1)."""
        if self.verbose:
            print("Making search vecs")
        #if params.two_color.spiral_method is not None:
        np.random.seed(self.spiral_seed)
        #noise_scale = self.noise_params.two_color.spiral_method[0]
        #Nsp = int(params.two_color.spiral_method[1])
        if self.verbose:
            print("Number of search vectors: %i" %(self.Nsp * len(self.unique_cell_dimensions)))
        Js = np.arange(self.Nsp, 2*self.Nsp+1)
        #  Algorithm for spiraling points around a unit sphere
        #  We only want the second half
        J = 2*self.Nsp
        thetas = np.arccos((2*Js-1-J)/J)
        phis = np.sqrt(np.pi*J)*np.arcsin((2*Js-1-J)/J)
        x = np.sin(thetas)*np.cos(phis)
        y = np.sin(thetas)*np.sin(phis)
        z = np.cos(thetas)
        u_vecs = np.zeros((self.Nsp, 3))
        u_vecs[:, 0] = x[1:]
        u_vecs[:, 1] = y[1:]
        u_vecs[:, 2] = z[1:]

        return u_vecs

    def score_vectors(self, reciprocal_lattice_vectors):
        """Compute the functional for the given directions.

        Args:
            directions: An iterable of the search directions.
            reciprocal_lattice_vectors (scitbx.array_family.flex.vec3_double):
                The list of reciprocal lattice vectors.
        Returns:
            A tuple containing the list of search vectors and their scores.
        """
        # much faster to use numpy for massively over-sampled hemisphere..
        rec_pts = np.array([reciprocal_lattice_vectors[i] for i in range(len(reciprocal_lattice_vectors))])
        N_unique = len(self.unique_cell_dimensions)
        function_values = np.zeros(self.Nsp * N_unique)
        vectors = np.zeros((self.Nsp * N_unique, 3))
        u_vecs = self.search_directions
        for i, l in enumerate(self.unique_cell_dimensions):
            # create noise model on top of lattice lengths...
            if self.noise_scale > 0:
                vec_mag = np.random.normal(l, scale=self.noise_scale, size=u_vecs.shape[0])
                vec_mag = vec_mag[:, None]
            else:
                vec_mag = l

            ul = u_vecs * vec_mag
            func_slc = slice(i * self.Nsp, (i + 1) * self.Nsp)
            vectors[func_slc] = ul
            function_values[func_slc] = block_sum(rec_pts, ul, self.block_size)

        vectors = flex.vec3_double(list(map(list, vectors)))
        function_values = flex.double(function_values)

        return vectors, function_values


def two_color_grid_search(beam1, beam2, detector, reflections, experiments, params, verbose):
    basis_searcher = BasisVectorSearch(reflections, experiments, params)

    target_unit_cell = params.indexing.known_symmetry.unit_cell
    target_space_group = params.indexing.known_symmetry.space_group
    if target_space_group is not None:
        target_space_group = target_space_group.group()
    symmetry_handler = SymmetryHandler(
        unit_cell=target_unit_cell,
        space_group=target_space_group,
        max_delta=params.indexing.known_symmetry.max_delta)

    refls1 = deepcopy(reflections)
    refls2 = deepcopy(reflections)
    expList1 = ExperimentList()
    expList2 = ExperimentList()
    exp1 = Experiment()
    exp1.beam = beam1
    exp1.detector = detector
    exp2 = Experiment()
    exp2.beam = beam2
    exp2.detector = detector
    expList1.append(exp1)
    expList2.append(exp2)
    refls1.centroid_px_to_mm(expList1)
    refls2.centroid_px_to_mm(expList2)
    refls1.map_centroids_to_reciprocal_space(expList1)
    refls2.map_centroids_to_reciprocal_space(expList2)

    relp1 = refls1['rlp']
    relp2 = refls2['rlp']

    reciprocal_lattice_points = relp1.concatenate(relp2)

    _cell = params.indexing.known_symmetry.unit_cell

    #strat = RealSpaceGridSearch(
    #    max_cell=1.3 * max(_cell.parameters()[:3]),
    #    target_unit_cell=_cell)
    strat = TwoColorGridSearch(
        max_cell=1.3 * max(_cell.parameters()[:3]),
        target_unit_cell=_cell)

    candidate_basis_vectors, used = strat.find_basis_vectors(reciprocal_lattice_points)

    if params.indexing.two_color.optimize_initial_basis_vectors:
        if verbose:
            print("OPTIMIZE BASIS VECS")
        optimised_basis_vectors = optimise.optimise_basis_vectors(
            reciprocal_lattice_points.select(used),
            candidate_basis_vectors)
        candidate_basis_vectors = [matrix.col(v) for v in optimised_basis_vectors]


    candidate_orientation_matrices \
        = basis_searcher.find_candidate_orientation_matrices(
        candidate_basis_vectors)
    # max_combinations=params.basis_vector_combinations.max_try)
    candidate_orientation_matrices = [C for C in candidate_orientation_matrices]

    new_xtal_matrices = []
    for cm in candidate_orientation_matrices:
        new_crystal, cb_op_to_primitive = symmetry_handler.apply_symmetry(cm)
        if new_crystal is None:
            continue
        #new_crystal = new_crystal.change_basis(
        #    symmetry_handler.cb_op_primitive_inp)
        new_crystal = new_crystal.change_basis(cb_op_to_primitive)
        new_xtal_matrices.append(new_crystal)

    candidate_orientation_matrices = new_xtal_matrices

    if params.indexing.two_color.filter_by_mag is not None:
        if verbose:
            print("FILTERING BY MAG")
        FILTER_TOL = params.indexing.two_color.filter_by_mag  # e.g. 10,3 within 10 percent of params and 1 percent of ang
        target_uc = params.indexing.known_symmetry.unit_cell.parameters()
        good_mats = []
        for c in candidate_orientation_matrices:
            uc = c.get_unit_cell().parameters()
            comps = []
            for i in range(3):
                tol = 0.01 * FILTER_TOL[0] * target_uc[i]
                low = target_uc[i] - tol / 2.
                high = target_uc[i] + tol / 2
                comps.append(low < uc[i] < high)
            for i in range(3, 6):
                low = target_uc[i] - FILTER_TOL[1]
                high = target_uc[i] + FILTER_TOL[1]
                comps.append(low < uc[i] < high)
            if all(comps):
                if verbose:
                    print("matrix is ok:", uc)
                good_mats.append(c)
            else:
                if verbose:
                    print("matrix is NOT ok:", uc)
            if verbose:
                print("Filter kept %d / %d mats" % \
                      (len(good_mats), len(candidate_orientation_matrices)))
        candidate_orientation_matrices = good_mats

    return candidate_orientation_matrices

    # TODO: try to let simtbx choose the best model, see if it does better.
    #crystal_model, n_indexed = choose_best_orientation_matrix(
    #    candidate_orientation_matrices)
    #if crystal_model is not None:
    #    crystal_models = [crystal_model]
    #else:
    #    crystal_models = []
    #
    #candidate_orientation_matrices = crystal_models
    #
    #candidate_crystal_models = candidate_orientation_matrices
