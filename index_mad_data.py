#!/usr/bin/env libtbx.python
from __future__ import print_function
from argparse import ArgumentParser

parser = ArgumentParser("Index data images from LD91 two color experiment")
parser.add_argument("--outdir",type=str, required=True) 
parser.add_argument("--glob", type=str, help="input image files", required=True)
parser.add_argument("--globalthresh", default=70, type=float)
parser.add_argument("--gain", default=28, type=float)
parser.add_argument("--n", default=None, type=int)
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

import glob
from copy import deepcopy
import os
from two_color import utils as two_color_utils
import json
import numpy as np
from IPython import embed
from cctbx.crystal import symmetry
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as strong_phil_scope
from dxtbx.model.experiment_list import ExperimentList, Experiment, ExperimentListFactory
from two_color.two_color_phil import params as index_params
import dxtbx
from cxid9114.parameters import ENERGY_HIGH, ENERGY_LOW, ENERGY_CONV, WAVELEN_HIGH, WAVELEN_LOW
from two_color.two_color_grid_search import two_color_grid_search

KNOWN_SYMM = symmetry("79.1,79.1,38.4,90,90,90", "P43212")

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

dirname = args.outdir
if rank==0:
    if not os.path.exists(dirname):
        os.makedirs(dirname)
comm.Barrier()

img_fnames = glob.glob(args.glob)
if rank==0:
    if not img_fnames:
        print("Found no filenames")

img_fnames_for_rank = np.array_split(img_fnames, size)[rank]

n_success = 0
n_fail = 0

num_imgs = len(img_fnames_for_rank)
if args.n is not None:
    num_imgs = args.n
    img_fnames_for_rank = img_fnames_for_rank[:num_imgs]

for img_num, fname in enumerate(img_fnames_for_rank):
    loader = dxtbx.load(fname)
    ISET = loader.get_imageset(loader.get_image_file())
    # NOTE: puttin in the thin CSPAD
    DET = loader.get_detector(0)
    BEAM = loader.get_beam(0)

    all_errors = []

    beam1 = deepcopy(BEAM)
    beam2 = deepcopy(BEAM)
    beam1.set_wavelength(WAVELEN_LOW)
    beam2.set_wavelength(WAVELEN_HIGH)

    assert len(ISET)==1

    basename = os.path.basename(fname)
    basename = os.path.splitext(basename)[0]
    if has_mpi:
        comm.Barrier()

    refl_name_template = "%s_strong.refl"
    refl_name = os.path.join(dirname, refl_name_template % (basename))
    
    expt_name_template = "%s_lattices.expt"
    expt_name = os.path.join(dirname, expt_name_template % (basename))

    expList = ExperimentList()
    exp = Experiment()
    exp.detector = DET
    exp.beam = BEAM
    exp.imageset = ISET
    expList.append(exp)
    
    strong_refls = flex.reflection_table.from_observations(experiments=expList, params=params)
    if rank == 0:
        print("Found %d refls on image %d /%d" % (len(strong_refls), img_num+1, num_imgs), flush=True)
    try:
        INDEXED_LATTICES = two_color_grid_search(
            beam1, beam2, DET, strong_refls, expList, index_params, verbose=True)
        if rank == 0:
            print("Indexing SUCCESS on image %d / %d" % (img_num+1, num_imgs),flush=True)
        n_success += 1
    except Exception as err:
        if rank == 0:
            print("Indexing failed on image %d / %d" % (img_num+1, num_imgs), flush=True)
        all_errors.append(err)
        n_fail += 1
        continue
   
    expList = ExperimentList()
    for C in INDEXED_LATTICES:
        exp = Experiment()
        exp.detector = DET
        exp.beam = BEAM
        exp.imageset = ISET
        exp.crystal = C
        expList.append(exp)
    
    best_crystal, n_pred_strong_overlap = two_color_utils.choose_best_crystal(expList, strong_refls)
    a,_,c,_,_,_ = best_crystal.get_unit_cell().parameters()
    print("Rank %d: best crystal: a=%.3f c=%.3f, N predict strong overlap = %d " % (rank, a,c,n_pred_strong_overlap))

    expList = ExperimentList()
    exp = Experiment()
    exp.detector = DET
    exp.beam = BEAM
    exp.imageset = ISET
    exp.crystal = best_crystal
    expList.append(exp)

    expList.as_json(expt_name)
    strong_refls.as_file(refl_name)
    if rank == 0:
        print("Wrote per rank strong refl file %s and exp file %s" % (refl_name, expt_name), flush=True)

comm.Barrier()
print("Rank %d : %d successes and %d failures" % (rank, n_success, n_fail), flush=True)

if has_mpi:
    all_errors = comm.reduce(all_errors)
if rank == 0:
    print("Encountered the following errors:", flush=True)
    print(set(all_errors), flush=True)
