#!/usr/bin/env libtbx.python
from argparse import ArgumentParser

parser = ArgumentParser("Index data images from LD91 two color experiment")
parser.add_argument("--f", type=str, required=True, help="Input image filename")
args = parser.parse_args()

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    has_mpi = True
except ImportError:
    rank = 0
    size = 1
    has_mpi = False

import sys
from cctbx.crystal import symmetry
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as strong_phil_scope
from dxtbx.model.experiment_list import ExperimentList, Experiment
from two_color.two_color_indexer import TwoColorIndexer
from two_color.two_color_phil import params as index_params
from IPython import embed
import dxtbx
from cxid9114.parameters import ENERGY_HIGH, ENERGY_LOW, ENERGY_CONV, WAVELEN_HIGH, WAVELEN_LOW

KNOWN_SYMM = symmetry("79,79,38,90,90,90", "P43212")

params = strong_phil_scope.extract()
params.spotfinder.threshold.algorithm = "dispersion"
params.spotfinder.filter.min_spot_size = 2
params.spotfinder.threshold.dispersion.global_threshold = 70
params.spotfinder.threshold.dispersion.kernel_size = 4, 4
params.spotfinder.threshold.dispersion.gain = 28
params.spotfinder.threshold.dispersion.sigma_strong = 2.25
params.spotfinder.threshold.dispersion.sigma_background = 6
params.spotfinder.force_2d = True

index_params.indexing.known_symmetry.space_group = KNOWN_SYMM.space_group_info()
index_params.indexing.known_symmetry.unit_cell = KNOWN_SYMM.unit_cell()
index_params.indexing.basis_vector_combinations.max_refine = 1
index_params.indexing.known_symmetry.absolute_angle_tolerance = 5.0
index_params.indexing.known_symmetry.relative_length_tolerance = 0.3
index_params.indexing.two_color.high_energy = ENERGY_HIGH
index_params.indexing.two_color.low_energy = ENERGY_LOW
index_params.indexing.two_color.avg_energy = ENERGY_LOW * .5 + ENERGY_HIGH * .5
index_params.indexing.two_color.filter_by_mag = 5, 3
index_params.indexing.two_color.optimize_initial_basis_vectors = True

KNOWN_SYMM = symmetry("79,79,38,90,90,90", "P43212")
if rank == 0:
    print("Loading the input file")
    sys.stdout.flush()
loader = dxtbx.load(args.f)
if rank == 0:
    print("Moving imageset to experiment list")
    sys.stdout.flush()
ISET = loader.get_imageset(loader.get_image_file())
DET = loader.get_detector()
BEAM = loader.get_beam()

all_errors = []
models = {}

num_imgs = len(ISET)
num_imgs = 18
n_success = 0
n_fail = 0
for img_num in range(num_imgs):
    if img_num % size != rank:
        continue
    expList = ExperimentList()
    exp = Experiment()
    exp.detector = DET
    exp.beam = BEAM
    exp.imageset = ISET[img_num:img_num+1]
    expList.append(exp)
    strong_refls = flex.reflection_table.from_observations(experiments=expList, params=params)
    if rank == 0:
        print("Found %d refls on image %d /%d" % (len(strong_refls), img_num+1, num_imgs))
        sys.stdout.flush()

    orient = TwoColorIndexer(strong_refls, expList, index_params)
    try:
        orient.index()
        if rank == 0:
            print("Indexing SUCCESS on image %d / %d" % (img_num+1, num_imgs))
            sys.stdout.flush()
        n_success += 1
    except Exception as err:
        if rank == 0:
            print("Indexing failed on image %d / %d" % (img_num+1, num_imgs))
            sys.stdout.flush()
        from IPython import embed
        embed()
        all_errors.append(err)
        models[img_num] = None
        n_fail += 1
        continue

    models[img_num] = orient.candidate_crystal_models

print("Rank %d : %d successes and %d failures" % (rank, n_success, n_fail))
sys.stdout.flush()
all_ranks_errors = comm.reduce(all_errors)
if rank==0:
    embed()
if has_mpi:
    comm.Barrier()