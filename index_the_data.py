#!/usr/bin/env libtbx.python
from __future__ import print_function
from argparse import ArgumentParser


parser = ArgumentParser("Index data images from LD91 two color experiment")
parser.add_argument("--f", type=str, required=True, help="Input image filename")
parser.add_argument("--n", type=int, default=None, help="process this many images total")
parser.add_argument("--expinput", action="store_true", help="whether the input is an expList")
parser.add_argument("--globalthresh", default=70, type=float)
parser.add_argument("--gain", default=28, type=float)
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

from copy import deepcopy
import os
import json
from cctbx.crystal import symmetry
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as strong_phil_scope
from dxtbx.model.experiment_list import ExperimentList, Experiment, ExperimentListFactory
from two_color.two_color_phil import params as index_params
import dxtbx
from cxid9114.parameters import ENERGY_HIGH, ENERGY_LOW, ENERGY_CONV, WAVELEN_HIGH, WAVELEN_LOW
from two_color.two_color_grid_search import two_color_grid_search

KNOWN_SYMM = symmetry("79,79,38,90,90,90", "P43212")

params = strong_phil_scope.extract()
params.spotfinder.threshold.algorithm = "dispersion"
params.spotfinder.filter.min_spot_size = 2
params.spotfinder.threshold.dispersion.global_threshold = args.globalthresh  # default 70
params.spotfinder.threshold.dispersion.kernel_size = 4, 4
params.spotfinder.threshold.dispersion.gain = args.gain  # default 28
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
    print("Loading the input file", flush=True)

if args.expinput:
    expListInput = ExperimentListFactory.from_json_file(args.f)
    ISET = expListInput.imagesets()[0]
    DET = expListInput.detectors()[0]
    BEAM = expListInput.beams()[0]
else:
    loader = dxtbx.load(args.f)
    ISET = loader.get_imageset(loader.get_image_file())
    # NOTE: puttin in the thin CSPAD
    DET = loader.get_detector()
    BEAM = loader.get_beam()

all_errors = []
models = {}

beam1 = deepcopy(BEAM)
beam2 = deepcopy(BEAM)
beam1.set_wavelength(WAVELEN_LOW)
beam2.set_wavelength(WAVELEN_HIGH)

num_imgs = len(ISET)
if args.n is not None:
    if num_imgs < args.n:
        if rank == 0:
            print("Trying to process too many images (max in file=%d)\n I will process %d." % (num_imgs, num_imgs))
    num_imgs = min(args.n, num_imgs)
n_success = 0
n_fail = 0

output_refls = flex.reflection_table()
dirname = os.path.dirname(args.f)
basename = os.path.basename(args.f)
basename = os.path.splitext(basename)[0]
refl_dirname = os.path.join(dirname, "refls")
if rank==0:
    if not os.path.exists(refl_dirname):
        os.makedirs(refl_dirname)
if has_mpi:
    comm.Barrier()

refl_name_template = "Rank%d_strong_%s.refl"
refl_name = os.path.join(refl_dirname, refl_name_template % (rank, basename))

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
    strong_refls["img_num"] = flex.double(len(strong_refls), img_num)
    output_refls.extend(strong_refls)
    if rank == 0:
        print("Found %d refls on image %d /%d" % (len(strong_refls), img_num+1, num_imgs), flush=True)
    try:
        INDEXED_LATTICES = two_color_grid_search(
            beam1, beam2, DET, strong_refls, expList, index_params, verbose=True)
        if rank == 0:
            print("Indexing SUCCESS on image %d / %d" % (img_num+1, num_imgs), flush=True)
        n_success += 1
    except Exception as err:
        if rank == 0:
            print("Indexing failed on image %d / %d" % (img_num+1, num_imgs), flush=True)
        all_errors.append(err)
        models[img_num] = None
        n_fail += 1
        continue

    models[img_num] = INDEXED_LATTICES

output_refls.as_file(refl_name)
if rank == 0:
    print("Wrote per rank strong refl file %s" % refl_name, flush=True)

comm.Barrier()
print("Rank %d : %d successes and %d failures" % (rank, n_success, n_fail), flush=True)

if has_mpi:
    all_errors = comm.reduce(all_errors)
if rank == 0:
    print("Encountered the following errors:", flush=True)
    print(set(all_errors), flush=True)

for i in models.keys():
    Cs = models[i]
    if Cs is None:
        continue
    models[i] = [C.get_A() for C in Cs]

modeldir = os.path.join(dirname, "models")
if rank==0:
    if not os.path.exists(modeldir):
        os.makedirs(modeldir)
if has_mpi:
    comm.Barrier()
modelname = os.path.join(modeldir, "Rank%d_models_%s.json" % (rank, basename))
with open(modelname, "w") as out:
    json.dump(models, out)
if rank == 0:
    print("Wrote model file %s" % modelname, flush=True)

comm.Barrier()
if rank == 0:
    print("Combining reflection tables... ", flush=True)
    master_table = flex.reflection_table()
    for i in range(size):
        print("Concat refls %d/%d" % (i+1, size))
        refl_name = os.path.join(refl_dirname, refl_name_template % (i, basename))
        refls = flex.reflection_table.from_file(refl_name)
        master_table.extend(refls)
    master_refl_name = os.path.join(dirname, "strong_%s.refl" % basename)
    print("Wrote master refl file %s" % master_refl_name, flush=True)
    master_table.as_file(master_refl_name)
