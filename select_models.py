# coding: utf-8

from argparse import ArgumentParser

parser = ArgumentParser("make the predictions from the models")
parser.add_argument("--setuponly", action="store_true")
parser.add_argument('--f', type=str, required=True, help="input image file")
parser.add_argument("--Fobs", type=str, required=True, help="input mtz file")
parser.add_argument("--pershotucell", action="store_true")
parser.add_argument("--pershotncells", action="store_true")
parser.add_argument("--adu2photon", type=float, required=True)
parser.add_argument("--correctbackground", action="store_true",
                    help="multiple tilt plane by polarization correction when computing likelihood")
parser.add_argument("--bgoffsetonly", action="store_true")
parser.add_argument("--bgoffsetpositive", action="store_true")
parser.add_argument("--curvatures", action='store_true')
parser.add_argument("--plot", action='store_true')
parser.add_argument("--detdist", action='store_true', help='perturb then refine the detdist')
parser.add_argument("--ncells", action='store_true', help='perturb then refine the ncells')
parser.add_argument("--bg", action='store_true', help='refine bg planes... ')
parser.add_argument("--spotscale", action='store_true')
parser.add_argument("--bmatrix", action='store_true')
parser.add_argument("--umatrix", action='store_true')
parser.add_argument("--fcell", action='store_true')
parser.add_argument("--maxcalls", type=int, default=None)
parser.add_argument("--norescale", action="store_false")
parser.add_argument("--fcellsigmascale", type=float, default=None)
parser.add_argument("--loadomegakahnfile", type=str, default=None)
parser.add_argument("--saveomegakahnfile", type=str, default=None)
parser.add_argument("--shuffleshots", type=int, default=None, help="seed for shuffling shots")
parser.add_argument("--nshots", type=int, default=1, help="number of shots to process")
args = parser.parse_args()


import os
from simtbx.diffBragg import utils
from cxid9114.parameters import WAVELEN_HIGH, WAVELEN_LOW
from simtbx.diffBragg.sim_data import SimData
from simtbx.diffBragg.refiners.crystal_systems import TetragonalManager
import h5py
import json
import numpy as np
import pylab as plt
from IPython import embed
from dxtbx.model import Crystal
from simtbx.diffBragg.nanoBragg_crystal import  nanoBragg_crystal
from simtbx.diffBragg.nanoBragg_beam import nanoBragg_beam
from cxid9114.sf.estimate_Fobs_twocolor import complete_fobs
from cxid9114.parameters import ENERGY_CONV
import dxtbx



def load_crystal_from_hdf5data(h5dataset):
    A = h5dataset["a_real"][()]
    B = h5dataset["b_real"][()]
    C = h5dataset["c_real"][()]
    crystal_model = Crystal(A,B,C,SYMBOL)
    return crystal_model


# GLOBALS
SYMBOL = "P43212"

n_ranks = 6

datadir = os.environ.get("DD")
jsondir = os.path.join(datadir, 'agreement_data')
hdf5dir = os.path.join(datadir, 'hdf5data')

# load the input file for beam and detector
input_file = args.f
loader = dxtbx.load(input_file)
DET = loader.get_detector(0)
BEAM = loader.get_beam(0)
basename = os.path.basename(input_file)
basename = os.path.splitext(basename)[0]


models_for_refinement = []
for i_rank in range(n_ranks):
    json_fname = "/Users/dermen/two_color_testing/agreement_data/Rank%d_%s.json" % (i_rank, basename)
    h5_fname = "/Users/dermen/two_color_testing/hdf5data/Rank%d_%s.hdf5" % (i_rank, basename)

    json_path = os.path.join(jsondir, json_fname)
    h5_path = os.path.join(hdf5dir, h5_fname)

    J = json.load(open(json_path, 'r'))
    diffs = J["com_diffs"]  # prediction deviation data
    shot_keys = list(diffs.keys())
    for shot in shot_keys:
        models = diffs[shot]
        model_keys = list(models.keys())
        nbest = 0
        best = None
        for mod in model_keys:
            n = len(models[mod])
            if n > nbest:
                best = mod
                nbest = n
            #print ("rank %d, shot %s, model %s is best: Number indexed=%d" % (i_rank, shot, mod, n))
        if nbest < 7:
            continue
        print("rank %d, shot %s, model %s is best: Number indexed=%d" % (i_rank, shot, best, nbest))
        model_tuple = {"h5_path": h5_path, "shot":shot, "mod":best,"nindexed":nbest}
        models_for_refinement .append(model_tuple)

# sort by number of indexed in descending order
model_for_refinement = sorted(models_for_refinement, key=lambda x: -x["nindexed"])


if args.shuffleshots is not None:
    import random
    random.seed(args.shuffleshots)
    random.shuffle(model_for_refinement)

N_SHOTS = args.nshots

shot_ucell_managers ={}
shot_rois ={}
shot_nanoBragg_rois ={}
shot_roi_imgs ={}
shot_spectra ={}
shot_crystal_GTs = None
shot_crystal_models ={}
shot_xrel ={}
shot_yrel ={}
shot_abc_inits ={}
shot_asu ={}
shot_hkl ={}
shot_panel_ids ={}
nspot_per_shot = {}
shot_originZ_init = {}


for i_shot in range(N_SHOTS):
    model = model_for_refinement[i_shot]
    h5_path = model["h5_path"]
    model_id = model["mod"]
    shot_id = model["shot"]
    h5_handle = h5py.File(h5_path, "r")
    h5data = h5_handle["Image%s/Model%s" % (shot_id, model_id)]

    spectrum_weights = h5_handle["Image%s/spectrum_weights" % shot_id][()]
    spectrum_energies = h5_handle["spectrum_energies"][()]
    spectrum_wavelens = ENERGY_CONV / spectrum_energies

    crystal = load_crystal_from_hdf5data(h5data)
    spot_roi = h5data["bbox_roi"][()]
    bbox_panel_ids = h5data["bbox_panel"][()]
    padded_images = h5data["padded_roi_images"]
    HKLi = h5data["Hi"][()]

    # miller indices
    Hi_asu = utils.map_hkl_list(HKLi, anomalous_flag=True, symbol=SYMBOL)

    # unit cell manager
    a,_,c,_,_,_ = crystal.get_unit_cell().parameters()
    UcellMan = TetragonalManager(a=a, c=c)

    # make the sim_data instance to be used by refiner

    nbcryst = nanoBragg_crystal()
    nbcryst.dxtbx_crystal = crystal
    nbcryst.thick_mm = 0.1
    nbcryst.Ncells_abc = 20, 20, 20  # guess of Ncells abc

    nbbeam = nanoBragg_beam()
    nbbeam.size_mm = 0.000886226925452758  # NOTE its a circular beam whoops
    nbbeam.unit_s0 = BEAM.get_unit_s0()
    nbbeam.spectrum = [(WAVELEN_LOW, .5e12), (WAVELEN_HIGH, .5e12)]
    full_spectrum = list(zip(spectrum_wavelens, spectrum_weights))

    SIM = SimData()
    SIM.detector = DET

    SIM.crystal = nbcryst
    SIM.beam = nbbeam
    SIM.instantiate_diffBragg(oversample=2)
    SIM.D.nopolar = False
    SIM.D.default_F = 0
    SIM.D.progress_meter = False

    # store solid angle and polarization term for each pixel
    if not args.loadomegakahnfile:
        correction_terms = {}
        SIM.D.only_save_omega_kahn = True
        for pid in range(len(DET)):
            print(pid)
            SIM.D.update_dxtbx_geoms(DET, BEAM, pid)
            SIM.D.add_diffBragg_spots()
            omega_kahn = SIM.D.raw_pixels.as_numpy_array()
            correction_terms[pid] = omega_kahn
            SIM.D.raw_pixels *= 0
            SIM.D.raw_pixels_roi *= 0
        SIM.D.only_save_omega_kahn = False
        SIM.D.update_dxtbx_geoms(DET, BEAM, 0)

        if args.saveomegakahnfile is not None:
            with h5py.File(args.saveomegakahnfile, "w") as hh:
                hh.create_dataset("correction_data", data=[correction_terms[pid] for pid in range(len(DET))])
                hh.create_dataset("correction_pid", data=range(len(DET)))
    else:
        correction_terms = {}
        with h5py.File(args.loadomegakahnfile,"r") as hh:
            pids = hh["correction_pid"][()]
            correction_data = hh["correction_data"][()]

        correction_terms = {}
        for i_pid, pid in enumerate(pids):
            correction_terms[pid] = correction_data[i_pid]

    # TODO: the following need to be added to the refiner init function..
    nspot = len(spot_roi)
    nspot_per_shot[i_shot] = nspot

    nanoBragg_rois = []  # special nanoBragg format
    xrel, yrel, roi_imgs = [], [], []
    tilt_abc = []
    for i_roi, (x1, x2, y1, y2) in enumerate(spot_roi):
        nanoBragg_rois.append(((int(x1), int(x2)-1), (int(y1), int(y2)-1)))
        yr, xr = np.indices((y2 - y1, x2 - x1))
        xrel.append(xr)
        yrel.append(yr)
        padded_image = padded_images[i_roi]
        image = padded_image[:y2-y1, :x2-x1]
        image_in_photons = image / args.adu2photon
        roi_imgs.append(image_in_photons)
        pid = bbox_panel_ids[i_roi]
        if args.correctbackground:
            bg_correction = correction_terms[pid][:y2-y1, :x2-x1]
            tilt_offset_init = np.median(image_in_photons/bg_correction)
        else:
            tilt_offset_init = np.median(image_in_photons)
        tilt_abc.append([0, 0, tilt_offset_init])  # set a flat plane initially for each roi

    shot_ucell_managers[i_shot] = UcellMan
    shot_rois[i_shot] = spot_roi
    shot_nanoBragg_rois[i_shot] = nanoBragg_rois
    shot_roi_imgs[i_shot] = roi_imgs
    shot_spectra[i_shot] = full_spectrum  #SIM.beam.spectrum
    shot_crystal_models[i_shot] = SIM.crystal.dxtbx_crystal
    shot_xrel[i_shot] = xrel
    shot_yrel[i_shot] = yrel
    shot_abc_inits[i_shot] = tilt_abc
    shot_asu[i_shot] = Hi_asu  # TODO Im weird fix me
    shot_hkl[i_shot] = list(HKLi)  # TODO Im weird fix me
    shot_panel_ids[i_shot] = bbox_panel_ids
    shot_originZ_init[i_shot] = DET[0].get_origin()[2]


Hi_all_ranks, Hi_asu_all_ranks = [], []
for i in range(N_SHOTS):
    Hi_all_ranks += shot_hkl[i]
    Hi_asu_all_ranks += shot_asu[i]

# NOTE what to do here for miller array unit cell, doesnt matter ?
miller_array = complete_fobs(Hi_all_ranks, args.Fobs, (79, 79, 38, 90, 90, 90), SYMBOL)
SIM.crystal.miller_array = miller_array
SIM.update_Fhkl_tuple()

# this will map the measured miller indices to their index in the LBFGS parameter array self.x
idx_from_asu = {h: i for i, h in enumerate(set(Hi_asu_all_ranks))}
# we will need the inverse map during refinement to update the miller array in diffBragg, so we cache it here
asu_from_idx = {i: h for i, h in enumerate(set(Hi_asu_all_ranks))}

# always local parameters: rotations, spot scales, tilt coeffs
nrotation_param = 3*N_SHOTS
nscale_param = 1*N_SHOTS
ntilt_param = 0  # note: tilt means tilt plane
for i_shot in range(N_SHOTS):
    ntilt_param += 3 * nspot_per_shot[i_shot]

# unit cell parameters
nucell_param = len(shot_ucell_managers[0].variables)
n_pershot_ucell_param = 0
n_global_ucell_param = nucell_param
if args.pershotucell:
    n_pershot_ucell_param += nucell_param*N_SHOTS
    n_global_ucell_param = 0

# mosaic domain parameter m
n_ncell_param = 1
n_pershot_m_param = 0
n_global_m_param = n_ncell_param
if args.pershotncells:
    n_pershot_m_param = 1*N_SHOTS
    n_global_m_param = 0

ndetz_param = N_SHOTS # 1 per shot, though not necessarily refined
n_local_unknowns = nrotation_param + nscale_param + ntilt_param + ndetz_param + n_pershot_ucell_param + n_pershot_m_param

nfcell_param = len(idx_from_asu)
ngain_param = 1

n_global_unknowns = nfcell_param + ngain_param + n_global_m_param + n_global_ucell_param
n_total_unknowns = n_local_unknowns + n_global_unknowns


from simtbx.diffBragg.refiners.global_refiner import FatRefiner

omega_kahn = None
if args.correctbackground:
    omega_kahn = correction_terms

RUC = FatRefiner(
    n_total_params=n_total_unknowns,
    n_local_params=n_local_unknowns,
    n_global_params=n_global_unknowns,
    local_idx_start=0,
    shot_ucell_managers=shot_ucell_managers,
    shot_rois=shot_roi_imgs,
    shot_nanoBragg_rois=shot_nanoBragg_rois,
    shot_roi_imgs=shot_roi_imgs,
    shot_spectra=shot_spectra,
    shot_crystal_GTs=shot_crystal_GTs,
    shot_crystal_models=shot_crystal_models,
    shot_xrel=shot_xrel,
    shot_yrel=shot_yrel,
    shot_abc_inits=shot_abc_inits,
    shot_asu=shot_asu,
    global_param_idx_start=n_local_unknowns,
    shot_panel_ids=shot_panel_ids,
    log_of_init_crystal_scales=None,
    all_crystal_scales=None,
    global_ncells=not args.pershotncells,
    global_ucell=not args.pershotucell,
    global_originZ=False,
    shot_originZ_init=shot_originZ_init,
    sgsymbol=SYMBOL,
    omega_kahn=omega_kahn)

print("POK)))")
###
# MAKE A FRESH SIM INSTANCE
###


######


SIM.D.spot_scale = 0.0046

RUC.idx_from_asu = idx_from_asu
RUC.asu_from_idx = asu_from_idx
RUC.index_of_displayed_image = 0

RUC.refine_background_planes = args.bg
RUC.refine_Umatrix = args.umatrix
RUC.refine_Bmatrix = args.bmatrix
RUC.refine_ncells = args.ncells
RUC.refine_crystal_scale = args.spotscale
RUC.refine_Fcell = args.fcell
RUC.refine_detdist = args.detdist
RUC.refine_gain_fac = False
RUC.rescale_params = args.norescale

if args.maxcalls is not None:
    RUC.max_calls = args.maxcalls
else:
    RUC.max_calls = 1000
RUC.trad_conv_eps = 1e-7
RUC.trad_conv = True
RUC.trial_id = 0

RUC.plot_stride = 4
RUC.plot_spot_stride = 10
RUC.plot_residuals = False
RUC.plot_images = args.plot
RUC.setup_plots()

RUC.refine_rotZ = True
RUC.request_diag_once = False
RUC.S = SIM
if not args.curvatures:
    RUC.S.D.compute_curvatures = False
RUC.has_pre_cached_roi_data = True
RUC.S.D.update_oversample_during_refinement = True
RUC.use_curvatures = False
RUC.use_curvatures_threshold = 10
RUC.bg_offset_positive = args.bgoffsetpositive
RUC.bg_offset_only = args.bgoffsetonly
RUC.calc_curvatures = args.curvatures
RUC.poisson_only = False
RUC.verbose = True
RUC.big_dump = True
RUC.testing_mode = False
RUC.compute_image_model_correlation = True

RUC.spot_scale_init = [1]*N_SHOTS
RUC.m_init = 50  #Ncells_abc2[0]
RUC.ucell_inits = UcellMan.variables

#RUC.S.D.update_oversample_during_refinement = False  # todo: decide
#Fobs = RUC.S.crystal.miller_array_high_symmetry
#RUC.Fref = miller_array_GT
#dmax, dmin = Fobs.d_max_min()
#dmax, dmin = max(all_reso), min(all_reso)
#RUC.binner_dmax = dmax + 1e-6
#RUC.binner_dmin = dmin - 1e-6
#RUC.binner_nbin = 10
#RUC.scale_r1 = True

RUC.merge_stat_frequency = 10
RUC.print_resolution_bins = False
if args.fcellsigmascale is not None:
    RUC.fcell_sigma_scale = args.fcellsigmascale

RUC.run(setup_only=args.setuponly)
print("pop")
if RUC.hit_break_to_use_curvatures:
    RUC.num_positive_curvatures = 0
    RUC.use_curvatures = True
    RUC.run(setup=False)
