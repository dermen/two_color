#!/usr/bin/env python
# -*- mode: python; coding: utf-8; indent-tabs-mode: nil; python-indent: 2 -*-
#
from __future__ import division
from dials.algorithms.indexing.stills_indexer import StillsIndexer
from dials.algorithms.indexing.lattice_search import BasisVectorSearch
from dials.algorithms.indexing.basis_vector_search.real_space_grid_search import RealSpaceGridSearch

import numpy as np
from copy import deepcopy
from scipy import constants as sci_cons
from six.moves import range
import math, copy
from scitbx import matrix
from scitbx.matrix import col
from dials.array_family import flex

# TODO consider fixing this , but it was hard coded to be off since I've seen this code
from dxtbx.model.experiment_list import Experiment, ExperimentList
from dials.algorithms.shoebox import MaskCode
from dials.algorithms.indexing.basis_vector_search import optimise

EV_CONV_FACTOR = round(1e10*sci_cons.h * sci_cons.c / sci_cons.electron_volt, 5)


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
  func_vals = np.zeros( ul.shape[0])
  for block_rlp in block_rec_pts:
    func_vals += np.sum( np.cos(2*np.pi*np.dot(block_rlp, ul.T)) ,axis=0)
  return func_vals


def index_reflections_detail(debug, experiments,
                             reflections,
                             detector,
                             reciprocal_lattice_points1,
                             reciprocal_lattice_points2,
                             d_min=None,
                             tolerance=0.3,
                             verbosity=0):
  ''' overwrites base class index_reflections function and assigns spots to
     their corresponding experiment (wavelength)'''

  # initialize each reflections miller index to 0,0,0
  reflections['miller_index'] = flex.miller_index(len(reflections), (0,0,0))

  # for two wavelengths
  assert len(experiments) == 3
  low_energy = 0   # 0th experiment is low-energy
  high_energy = 1  # 1st experiment is high-energ
  avg_energy = 2  # 2nd experiment is average energy (for spot overlaps)

  # code to check input orientation matrix
  # get predicted reflections based on basis vectors
  pred = False
  if pred ==True:
    experiments[0].crystal._ML_half_mosaicity_deg = .2
    experiments[0].crystal._ML_domain_size_ang = 1000
    predicted = flex.reflection_table.from_predictions_multi(experiments[0:2])
    predicted.as_pickle('test')

  inside_resolution_limit = flex.bool(len(reflections), True)
  if d_min is not None:
    d_spacings = 1/reflections['rlp'].norms()
    inside_resolution_limit &= (d_spacings > d_min)

  # boolean array, all yet-to-be spots that are bound by the resolution
  sel = inside_resolution_limit & (reflections['id'] == -1)
  # array of indices of the reflections
  isel = sel.iselection()
# I believe .select( isel) is same as .select( sel)
  #rlps0 = reciprocal_lattice_points1.select(isel)  # low-energy beam lp vectors calculated in two_color_grid_search
  #rlps1 = reciprocal_lattice_points2.select(isel)  # high-energy beam lps!
  #refs = reflections.select(isel)

  rlps0 = reciprocal_lattice_points1.select(sel)  # low-energy beam lp vectors calculated in two_color_grid_search
  rlps1 = reciprocal_lattice_points2.select(sel)  # high-energy beam lps!
  refs = reflections.select(sel)

  rlps = (rlps0, rlps1)  # put em in a tuple ?
  #rlp_norms = []
  hkl_ints = []
  norms = []
  diffs = []
  c1 = experiments.crystals()[0]
  assert( len(experiments.crystals()) == 1 )  # 3 beams but only 1 crystal!
  A = matrix.sqr(experiments.crystals()[0].get_A())
  A_inv = A.inverse()

  # confusing variable names, but for each set of R.L.P.s.
  # (one for the high and one for the low energy beam)
  # find the distance to the nearest integer hkl
  for rlp in range(len(rlps)):
    hkl_float = tuple(A_inv) * rlps[rlp]
    hkl_int = hkl_float.iround()
    differences = hkl_float - hkl_int.as_vec3_double()
    diffs.append(differences)
    norms.append(differences.norms())
    hkl_ints.append(hkl_int)

  n_rejects = 0
  for i_hkl in range(hkl_int.size()):
    n = flex.double([norms[j][i_hkl]
                     for j in range(len(rlps))])
    potential_hkls = [hkl_ints[j][i_hkl]
                      for j in range(len(rlps))]
    potential_rlps = [rlps[j][i_hkl]
                      for j in range(len(rlps))]
    if norms[0][i_hkl]>norms[1][i_hkl]:
      i_best_lattice = high_energy
      i_best_rlp = high_energy
    elif norms[0][i_hkl]<norms[1][i_hkl]:
      i_best_lattice = low_energy
      i_best_rlp = low_energy
    else:
      i_best_lattice = flex.min_index(n)
      i_best_rlp = flex.min_index(n)
    if n[i_best_lattice] > tolerance:
      n_rejects += 1
      continue
    miller_index = potential_hkls[i_best_lattice]
    reciprocal_lattice_points = potential_rlps[i_best_rlp]
    i_ref = isel[i_hkl]
    reflections['miller_index'][i_ref] = miller_index
    reflections['id'][i_ref] = i_best_lattice
    reflections['rlp'][i_ref] = reciprocal_lattice_points

  # if more than one spot can be assigned the same miller index then choose
  # the closest one
  miller_indices = reflections['miller_index'].select(isel)
  rlp_norms = reflections['rlp'].select(isel).norms()
  same=0
  for i_hkl, hkl in enumerate(miller_indices):
    if hkl == (0,0,0): continue
    iselection = (miller_indices == hkl).iselection()
    if len(iselection) > 1:
      for i in iselection:
        for j in iselection:
          if j <= i: continue
          crystal_i = reflections['id'][isel[i]]
          crystal_j = reflections['id'][isel[j]]
          if crystal_i != crystal_j:
            continue
          elif (crystal_i == -1 or crystal_j ==-1) or (crystal_i == -2 or crystal_j == -2):
            continue
          elif crystal_i ==2 or crystal_j ==2:
            continue
            #print hkl_ints[crystal_i][i], hkl_ints[crystal_j][j], crystal_i
          assert hkl_ints[crystal_j][j] == hkl_ints[crystal_i][i]
          same +=1
          if rlp_norms[i] < rlp_norms[j]:
            reflections['id'][isel[i]] = high_energy
            reflections['id'][isel[j]] = low_energy
          elif rlp_norms[j] < rlp_norms[i]:
            reflections['id'][isel[j]] = high_energy
            reflections['id'][isel[i]] = low_energy

  #calculate Bragg angles
  s0 = col(experiments[2].beam.get_s0())
  lambda_0 = experiments[0].beam.get_wavelength()
  lambda_1 = experiments[1].beam.get_wavelength()
  det_dist = experiments[0].detector[0].get_distance()
  px_size_mm = experiments[0].detector[0].get_pixel_size()[0]
  spot_px_coords=reflections['xyzobs.px.value'].select(isel)
  px_x,px_y,px_z = spot_px_coords.parts()
  res  = []
  for i in range(len(spot_px_coords)):
    res.append(detector[0].get_resolution_at_pixel(s0, (px_x[i], px_y[i])))
  # predicted spot distance  based on the resultion of the observed spot at either wavelength 1 or 2
  theta_1a = [math.asin(lambda_0/(2*res[i])) for i in range(len(res))]
  theta_2a = [math.asin(lambda_1/(2*res[i])) for i in range(len(res))]
  px_dist = [(math.tan(2*theta_1a[i])*det_dist-math.tan(2*theta_2a[i])*det_dist)/px_size_mm for i in range(len(spot_px_coords))]
  # first calculate distance from stop centroid to farthest valid pixel (determine max spot radius)
  # coords of farthest valid pixel
  # if the predicted spot distance at either wavelength is less than 2x distance described above than the spot is considered "overlapped" and assigned to experiment 2 at average wavelength

  valid = MaskCode.Valid | MaskCode.Foreground

  for i in range(len(refs)):
    if reflections['miller_index'][isel[i]]==(0,0,0): continue
    sb = reflections['shoebox'][isel[i]]
    bbox = sb.bbox
    mask = sb.mask
    centroid = col(reflections['xyzobs.px.value'][isel[i]][0:2])
    x1, x2, y1, y2, z1, z2 = bbox

    longest = 0
    for y in range(y1, y2):
      for x in range(x1, x2):
        if mask[z1,y-y1,x-x1] != valid:
          continue
        v = col([x,y])
        dist = (centroid -v).length()
        if dist > longest:
          longest = dist
    #print "Miller Index", reflections['miller_index'][i], "longest", longest,"predicted distance", px_dist_1[i]
    if 2*longest > px_dist[i]:
      avg_rlp0 = reflections['rlp'][isel[i]][0]*experiments[reflections['id'][isel[i]]].beam.get_wavelength()/experiments[2].beam.get_wavelength()
      avg_rlp1 = reflections['rlp'][isel[i]][1]*experiments[reflections['id'][isel[i]]].beam.get_wavelength()/experiments[2].beam.get_wavelength()
      avg_rlp2 = reflections['rlp'][isel[i]][2]*experiments[reflections['id'][isel[i]]].beam.get_wavelength()/experiments[2].beam.get_wavelength()
      reflections['id'][isel[i]] = avg_energy
      reflections['rlp'][isel[i]] = (avg_rlp0, avg_rlp1, avg_rlp2)

  # check for repeated hkl in experiment 2, and if experiment 2 has same hkl as experiment 0 or 1 the spot with the largest variance is assigned to experiment -2 and the remaining spot is assigned to experiment 2

  for i_hkl, hkl in enumerate(miller_indices):
    if hkl == (0,0,0): continue
    iselection = (miller_indices == hkl).iselection()
    if len(iselection) > 1:
      for i in iselection:
        for j in iselection:
          if j <= i: continue
          crystal_i = reflections['id'][isel[i]]
          crystal_j = reflections['id'][isel[j]]
          if (crystal_i == -1 or crystal_j ==-1) or (crystal_i == -2 or crystal_j == -2):
            continue
          # control to only filter for experient 2; duplicate miller indices in 0 and 1 are resolved above
          if (crystal_i == 1 and crystal_j == 0) or (crystal_i == 0 and crystal_j ==1):
            continue

          if (crystal_i == 2 or crystal_j == 2) and (reflections['xyzobs.px.variance'][isel[i]]<reflections['xyzobs.px.variance'][isel[j]]):
              reflections['id'][isel[j]] = -2
              reflections['id'][isel[i]] = avg_energy
          elif (crystal_i == 2 or crystal_j == 2) and (reflections['xyzobs.px.variance'][isel[i]]>reflections['xyzobs.px.variance'][isel[j]]):
              reflections['id'][isel[i]] = -2
              reflections['id'][isel[j]] = avg_energy
          if (crystal_i ==2 and crystal_j ==2) and (reflections['xyzobs.px.variance'][isel[i]]<reflections['xyzobs.px.variance'][isel[j]]):
              reflections['id'][isel[j]] = -2
              reflections['id'][isel[i]] = avg_energy
          elif (crystal_i ==2 and crystal_j ==2) and (reflections['xyzobs.px.variance'][isel[i]]>reflections['xyzobs.px.variance'][isel[j]]):
              reflections['id'][isel[i]] = -2
              reflections['id'][isel[j]] = avg_energy

  # check that each experiment list does not contain duplicate miller indices
  #exp_0 = reflections.select(reflections['id']==0)
  #exp_1 = reflections.select(reflections['id']==1)
  #exp_2 = reflections.select(reflections['id']==2)
  #assert len(exp_0['miller_index'])==len(set(exp_0['miller_index']))
  #assert len(exp_1['miller_index'])==len(set(exp_1['miller_index']))
  #assert len(exp_2['miller_index'])==len(set(exp_2['miller_index']))


class TwoColorIndexer(StillsIndexer):
  ''' class to calculate orientation matrix for 2 color diffraction images '''

  def __init__(self, reflections, experiments, params):

    assert(len(experiments) == 1), "I think this only works for length=1 experiments"
    self._det = experiments[0].detector

    beam = experiments[0].beam
    beam1 = copy.deepcopy(beam)
    beam2 = copy.deepcopy(beam)
    beam3 = copy.deepcopy(beam)
    wavelength1 = EV_CONV_FACTOR/params.indexing.two_color.low_energy
    wavelength2 = EV_CONV_FACTOR/params.indexing.two_color.high_energy
    wavelength3 = EV_CONV_FACTOR/params.indexing.two_color.avg_energy
    beam1.set_wavelength(wavelength1)
    beam2.set_wavelength(wavelength2)
    beam3.set_wavelength(wavelength3)
    self.beams = [beam1, beam2, beam3]
    self.debug = params.indexing.two_color.debug

    self.basis_searcher = BasisVectorSearch(reflections, experiments, params)
    super(TwoColorIndexer, self).__init__(reflections, experiments, params)

  def index(self):
    super(TwoColorIndexer, self).index()

    experiments2 = ExperimentList()
    indexed2 = flex.reflection_table()
    for e_number in range(len(self.refined_experiments)):
      experiments2.append(self.refined_experiments[e_number])
      ref_id = self.refined_reflections["id"]
      e_selection = flex.bool([r == e_number or r == 2 for r in ref_id])
      e_indexed = self.refined_reflections.select(e_selection)
      e_indexed['id'] = flex.int(len(e_indexed), e_number) # renumber all
      indexed2.extend(e_indexed)
      if e_number >=1: break
    self.refined_experiments = experiments2
    self.refined_reflections = indexed2

  def index_reflections(self, experiments, reflections, verbosity=0):
    '''
    if there are two or more experiments calls overloaded index_reflections
    This method should do everything that indexer_base.index_reflections does..
    which appears to be setting some kind of reflectio flags
    '''
    params_simple = self.params.index_assignment.simple
    index_reflections_detail(self.debug, experiments, reflections,
                             experiments[0].detector,
                             self.reciprocal_lattice_points1,
                             self.reciprocal_lattice_points2,
                             self.d_min,
                             tolerance=params_simple.hkl_tolerance,
                             verbosity=verbosity)

    for i in range(len(reflections)):
      if reflections['id'][i] == -1:
        reflections['id'][i] = 0

    reflections.set_flags(
      reflections['miller_index'] != (0,0,0), reflections.flags.indexed)

  def experiment_list_for_crystal(self, crystal):
    experiments = ExperimentList()
    for beam in self.beams:
      exp = Experiment()
      exp.beam=beam
      exp.detector=self._det
      exp.crystal=crystal
      experiments.append(exp)

    return experiments

  def find_lattices(self):
    '''assigns the crystal model and the beam(s) to the experiment list'''
    print("\n\nFINDING LATTTTSSSSSS\n")
    self.two_color_grid_search()
    crystal_models = self.candidate_crystal_models
    assert len(crystal_models) == 1
    # only return the experiments 0 and 1
    return self.experiment_list_for_crystal(crystal_models[0])

  def two_color_grid_search(self):
    '''creates candidate reciprocal lattice points based on two beams and performs
    2-D grid search based on maximizing the functional using N_UNIQUE_V candidate
    vectors (N_UNIQUE_V is usually 30 from Guildea paper)'''

    detector = self.experiments[0].detector
    refls1 = deepcopy(self.reflections)
    refls2 = deepcopy(self.reflections)
    expList1 = ExperimentList() 
    expList2 = ExperimentList() 
    exp1 = Experiment()
    exp1.beam=self.beams[0]
    exp1.detector=detector
    exp2 = Experiment()
    exp2.beam=self.beams[1]
    exp2.detector=detector
    expList1.append(exp1)
    expList2.append(exp2)
    refls1.centroid_px_to_mm(expList1)
    refls2.centroid_px_to_mm(expList2)
    refls1.map_centroids_to_reciprocal_space(expList1)
    refls2.map_centroids_to_reciprocal_space(expList2)

    self.reciprocal_lattice_points1 = refls1['rlp'].select(
          (refls1['id'] == -1))
    self.reciprocal_lattice_points2 = refls2['rlp'].select(
      (refls2['id'] == -1))

    self.reciprocal_lattice_points = self.reciprocal_lattice_points1.concatenate(self.reciprocal_lattice_points2)

    _cell = self.params.known_symmetry.unit_cell

    strat = RealSpaceGridSearch(
        1.3*max(_cell.parameters()[:3]),
        target_unit_cell=_cell)

    self.candidate_basis_vectors, used = strat.find_basis_vectors(self.reciprocal_lattice_points)

    if self.params.two_color.optimize_initial_basis_vectors:
        print("\n\n OPTIMIZE BASIS VECS\n\n")
        optimised_basis_vectors = optimise.optimise_basis_vectors(
            self.reciprocal_lattice_points.select(used),
            self.candidate_basis_vectors)
        self.candidate_basis_vectors = [matrix.col(v) for v in optimised_basis_vectors]

    #print "Indexing from %i reflections" %len(self.reciprocal_lattice_points)

    #def compute_functional(vector):
    #  '''computes functional for 2-D grid search'''
    #  two_pi_S_dot_v = 2 * math.pi * self.reciprocal_lattice_points.dot(vector)
    #  return flex.sum(flex.cos(two_pi_S_dot_v))

    #from rstbx.array_family import flex
    #from rstbx.dps_core import SimpleSamplerTool
    #assert self.target_symmetry_primitive is not None
    #assert self.target_symmetry_primitive.unit_cell() is not None
    #SST = SimpleSamplerTool(
    #  self.params.real_space_grid_search.characteristic_grid)
    #SST.construct_hemisphere_grid(SST.incr)
    #cell_dimensions = self.target_symmetry_primitive.unit_cell().parameters()[:3]
    #unique_cell_dimensions = set(cell_dimensions)

    #print("Making search vecs")
    #if self.params.two_color.spiral_method is not None:
    #    np.random.seed(self.params.two_color.spiral_seed)
    #    noise_scale = self.params.two_color.spiral_method[0]
    #    Nsp = int(self.params.two_color.spiral_method[1])
    #    print "Number of search vectors: %i" %(Nsp * len(unique_cell_dimensions))
    #    Js = np.arange( Nsp, 2*Nsp+1)
    #    #  Algorithm for spiraling points around a unit sphere
    #    #  We only want the second half
    #    J = 2*Nsp
    #    _thetas = np.arccos((2*Js-1-J)/J)
    #    _phis = np.sqrt(np.pi*J)*np.arcsin((2*Js-1-J)/J)
    #    _x = np.sin(_thetas)*np.cos(_phis)
    #    _y = np.sin(_thetas)*np.sin(_phis)
    #    _z = np.cos(_thetas)
    #    u_vecs = np.zeros( (Nsp,3))
    #    u_vecs[:,0] = _x[1:]
    #    u_vecs[:,1] = _y[1:]
    #    u_vecs[:,2] = _z[1:]

    #    rec_pts = np.array([self.reciprocal_lattice_points[i] for i in range(len(self.reciprocal_lattice_points))])
    #    N_unique = len(unique_cell_dimensions)

    #    # much faster to use numpy for massively over-sampled hemisphere..
    #    function_values = np.zeros( Nsp*N_unique)
    #    vectors = np.zeros( (Nsp*N_unique, 3) )
    #    for i, l in enumerate(unique_cell_dimensions):
    #      # create noise model on top of lattice lengths...
    #      if noise_scale > 0:
    #        vec_mag = np.random.normal( l, scale=noise_scale, size=u_vecs.shape[0] )
    #        vec_mag = vec_mag[:,None]
    #      else:
    #        vec_mag = l

    #      ul = u_vecs * vec_mag
    #      func_slc = slice( i*Nsp, (i+1)*Nsp)
    #      vectors[func_slc] = ul
    #      # function_values[func_slc] = np.sum( np.cos( 2*np.pi*np.dot(rec_pts, ul.T) ),
    #      #                            axis=0)
    #      function_values[func_slc] = block_sum(rec_pts, ul, self.params.two_color.block_size)

    #    del u_vecs, vec_mag
    #    order = np.argsort(function_values)[::-1]  # sort function values, largest values first
    #    function_values = function_values[order]
    #    vectors = vectors[order]

    #else:  # fall back on original flex method
    #    vectors = flex.vec3_double()
    #    function_values = flex.double()
    #    print "Number of search vectors: %i" % (   len(SST.angles)* len(unique_cell_dimensions))
    #    for i, direction in enumerate(SST.angles):
    #      for l in unique_cell_dimensions:
    #        v = matrix.col(direction.dvec) * l
    #        f = compute_functional(v.elems)
    #        vectors.append(v.elems)
    #        function_values.append(f)
    #    perm = flex.sort_permutation(function_values, reverse=True)
    #    vectors = vectors.select(perm)
    #    function_values = function_values.select(perm)

    #print("made search vecs")

    #unique_vectors = []
    #i = 0
    #while len(unique_vectors) < self.params.two_color.n_unique_v:
    #  v = matrix.col(vectors[i])
    #  is_unique = True
    #  if i > 0:
    #    for v_u in unique_vectors:
    #      if v.length() < v_u.length():
    #        if is_approximate_integer_multiple(v, v_u):
    #          is_unique = False
    #          break
    #      elif is_approximate_integer_multiple(v_u, v):
    #        is_unique = False
    #        break
    #  if is_unique:
    #    unique_vectors.append(v)
    #  i += 1
    #print ("chose unique basis vecs")
    #if self.params.debug:
    #  for i in range(self.params.two_color.n_unique_v):
    #    v = matrix.col(vectors[i])
    #    print v.elems, v.length(), function_values[i]

    #basis_vectors = [v.elems for v in unique_vectors]
    #self.candidate_basis_vectors = basis_vectors

    #if self.params.optimise_initial_basis_vectors:
    #  self.params.optimize_initial_basis_vectors = False
    #  # TODO: verify this reference to self.reciprocal_lattice_points is correct
    #  optimised_basis_vectors = optimise_basis_vectors(
    #    self.reciprocal_lattice_points, basis_vectors)
    #  optimised_function_values = flex.double([
    #    compute_functional(v) for v in optimised_basis_vectors])

    #  perm = flex.sort_permutation(optimised_function_values, reverse=True)
    #  optimised_basis_vectors = optimised_basis_vectors.select(perm)
    #  optimised_function_values = optimised_function_values.select(perm)

    #  unique_vectors = [matrix.col(v) for v in optimised_basis_vectors]

    #print "Number of unique vectors: %i" %len(unique_vectors)

    #if self.params.debug:
    #  for i in range(len(unique_vectors)):
    #    print compute_functional(unique_vectors[i].elems), unique_vectors[i].length(), unique_vectors[i].elems
    #    print

    #self.candidate_basis_vectors = unique_vectors

    #if self.params.debug:
    #  self.debug_show_candidate_basis_vectors()
    #if self.params.debug_plots:
    #  self.debug_plot_candidate_basis_vectors()

    #del vectors, function_values
    candidate_orientation_matrices \
      = self.basis_searcher.find_candidate_orientation_matrices(
        self.candidate_basis_vectors)
        # max_combinations=self.params.basis_vector_combinations.max_try)
    candidate_orientation_matrices = [C for C in candidate_orientation_matrices]
    if self.params.two_color.filter_by_mag is not None:
      print("\n\n FILTERING BY MAG\n\n")
      FILTER_TOL = self.params.two_color.filter_by_mag # e.g. 10,3 within 10 percent of params and 1 percent of ang
      target_uc = self.params.known_symmetry.unit_cell.parameters()
      good_mats = []
      for c in candidate_orientation_matrices:
        uc = c.get_unit_cell().parameters()
        comps = []
        for i in range(3):
          tol = 0.01* FILTER_TOL[0] * target_uc[i]
          low = target_uc[i] - tol/2.
          high = target_uc[i] + tol/2
          comps.append(low < uc[i] < high )
        for i in range(3,6):
          low = target_uc[i] - FILTER_TOL[1]
          high = target_uc[i] + FILTER_TOL[1]
          comps.append( low < uc[i] < high )
        if all( comps):
          print("matrix is ok:", uc)
          good_mats.append(c)
        else:
          print("matrix is NOT ok:", uc)
        print("\nFilter kept %d / %d mats" % \
              (len(good_mats), len(candidate_orientation_matrices)))
      candidate_orientation_matrices = good_mats

    # TODO: try to let simtbx choose the best model, see if it does better. 
    crystal_model, n_indexed = self.choose_best_orientation_matrix(
      candidate_orientation_matrices)
    if crystal_model is not None:
      crystal_models = [crystal_model]
    else:
      crystal_models = []

    candidate_orientation_matrices = crystal_models

    self.candidate_crystal_models = candidate_orientation_matrices


