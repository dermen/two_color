from __future__ import print_function

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
import time
from two_color import utils
from IPython import embed
from cxid9114.sim import sim_utils

from cxid9114.sf import struct_fact_special
import os
import numpy as np
from scipy import constants
from copy import deepcopy

from cctbx import sgtbx, miller
from cctbx.crystal import symmetry
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as strong_phil_scope
from dials.algorithms.indexing.compare_orientation_matrices import rotation_matrix_differences
import dxtbx
from dxtbx.model.experiment_list import ExperimentList, Experiment, ExperimentListFactory
from dxtbx.model.beam import BeamFactory
from dxtbx.model.crystal import CrystalFactory
from dxtbx.model.detector import DetectorFactory
from scitbx.matrix import sqr, col
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype

from cxid9114.helpers import compare_with_ground_truth

from two_color.utils import get_two_color_rois
from two_color.two_color_grid_search import two_color_grid_search
from two_color.two_color_phil import params as index_params
from cxid9114.utils import open_flex



num_imgs = 18

######################
# LOAD JUNGFRAU MODEL
######################

print("Load the big azz detector")
expList_for_DET = ExperimentListFactory.from_json_file(
    "/Users/dermen/idx-run_000795.JF07T32V01_master_00000_integrated.expt",
    check_format=False)
DETECTOR = expList_for_DET.detectors()[0]
image_shape = (len(DETECTOR),) + DETECTOR[0].get_image_size()

#############
# BEAM MODEL
#############
beam_descr={'direction': (-0.0, -0.0, 1.0),
 'wavelength': 1.3037,
 'divergence': 0.0,
 'sigma_divergence': 0.0,
 'polarization_normal': (0.0, 1.0, 0.0),
 'polarization_fraction': 0.999,
 'flux': 1e12,
 'transmission': 1.0}

# two color experiment, two energies 100 eV apart
ENERGYLOW = 9510
ENERGYHIGH = 9610
ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt
WAVELENLOW = ENERGY_CONV/ENERGYLOW
WAVELENHIGH = ENERGY_CONV/ENERGYHIGH
BEAM = BeamFactory.from_dict(beam_descr)

#sf_path = os.path.dirname(struct_fact_special.__file__)
#
#FampLOW = struct_fact_special.sfgen(WAVELENLOW,
#    os.path.join(sf_path, "4bs7.pdb"),
#    yb_scatter_name=os.path.join(sf_path, "../sf/scanned_fp_fdp.tsv"))
#FampHIGH = struct_fact_special.sfgen(WAVELENHIGH,
#    os.path.join(sf_path, "4bs7.pdb"),
#    yb_scatter_name=os.path.join(sf_path, "../sf/scanned_fp_fdp.tsv"))
#FampLOW = FampLOW.as_amplitude_array()
#FampHIGH = FampHIGH.as_amplitude_array()

from iotbx.reflection_file_reader import any_reflection_file
FampLOW = any_reflection_file("5wp2-sf.cif").as_miller_arrays()[0]
#FampLOW = FampLOW.resolution_filter(d_max=33, d_min=1.4)
assert(FampLOW.is_xray_amplitude_array())
FampHigh = None

a = FampLOW.unit_cell().parameters()[0]  # 77
c = FampLOW.unit_cell().parameters()[2]  # 263

# make a single pattern
if rank == 0:
    print("Begin the big azz simulation", flush=True)

outname = "/Users/dermen/one_color_testing/eiger_h5_rank%d.h5" % rank

# pick out how many images per file ...
imgs_per = {}
for r in range(size):
    imgs_per[r] = 0
for i in range(num_imgs):
    if i % size != rank:
        continue
    imgs_per[rank] += 1

# make them images, allocate a hdf5 file per rank
with utils.H5AttributeGeomWriter(outname, image_shape=image_shape,
                                 num_images=imgs_per[rank], detector=DETECTOR, beam=BEAM) as writer:

    for img_num in range(num_imgs):
        if img_num % size != rank:
            continue
        ###############################
        # MAKE THE RANDOM CRYSTAL ORI
        ###############################

        np.random.seed(3142019 + img_num)
        # make random rotation about principle axes
        x = col((-1, 0, 0))
        y = col((0, -1, 0))
        z = col((0, 0, -1))
        rx, ry, rz = np.random.uniform(-180, 180, 3)
        RX = x.axis_and_angle_as_r3_rotation_matrix(rx, deg=True)
        RY = y.axis_and_angle_as_r3_rotation_matrix(ry, deg=True)
        RZ = z.axis_and_angle_as_r3_rotation_matrix(rz, deg=True)
        M = RX*RY*RZ
        real_a = M*col((a, -.5*a, 0))
        real_b = M*col((0, np.sqrt(3)*.5*a, 0))
        real_c = M*col((0, 0, c))

        # dxtbx crystal description
        cryst_descr = {'__id__': 'crystal',
                      'real_space_a': real_a.elems,
                      'real_space_b': real_b.elems,
                      'real_space_c': real_c.elems,
                      'space_group_hall_symbol': ' P 4nw 2abw'}

        CRYSTAL = CrystalFactory.from_dict(cryst_descr)

        t = time.time()
        #simsAB = sim_utils.sim_colors(
        #    CRYSTAL, DETECTOR, BEAM, [FampLOW, FampHigh],
        #    [ENERGYLOW, ENERGYHIGH],
        #    [.5e11, .5e11], pids=None, profile="gauss", cuda=False, oversample=0,
        #    Ncells_abc=(20, 20, 20), mos_dom=1, mos_spread=0,
        #    exposure_s=1, beamsize_mm=0.001, device_Id=0,
        #    amorphous_sample_thick_mm=0.100, add_water=True,
        #    show_params=False, accumulate=False, crystal_size_mm=0.01, printout_pix=None,
        #    time_panels=True)

        simsAB = sim_utils.sim_colors(
            CRYSTAL, DETECTOR, BEAM, [FampLOW],
            [ENERGYLOW],
            [1e12], pids=None, profile="gauss", cuda=False, oversample=0,
            Ncells_abc=(20, 20, 20), mos_dom=1, mos_spread=0,
            exposure_s=1, beamsize_mm=0.001, device_Id=0,
            amorphous_sample_thick_mm=0.200, add_water=True,
            show_params=False, accumulate=False, crystal_size_mm=0.01, printout_pix=None,
            time_panels=True)

        tsim = time.time()-t

        two_color_panels = np.array(simsAB[0])  # + np.array(simsAB[1])

        for pidx in range(len(DETECTOR)):
            SIM = nanoBragg(detector=DETECTOR, beam=BEAM, panel_id=pidx)
            SIM.beamsize_mm = 0.001
            SIM.exposure_s = 1
            SIM.flux = 1e12
            SIM.adc_offset_adu = 0
            # SIM.detector_psf_kernel_radius_pixels = 5
            # SIM.detector_psf_type = shapetype.Unknown  # for CSPAD
            SIM.detector_psf_fwhm_mm = 0
            SIM.quantum_gain = 1
            SIM.raw_pixels = flex.double(two_color_panels[pidx].ravel())
            SIM.add_noise()
            two_color_panels[pidx] = SIM.raw_pixels.as_numpy_array().reshape(two_color_panels[pidx].shape)
            SIM.free_all()
            del SIM

        writer.add_image(two_color_panels)

        #np.savez("/Users/dermen/two_color_testing/jung/jung_twocol_bs7_py3_img%d" % img_num,
        #         img=two_color_panels, det=DETECTOR.to_dict(), beam=BEAM.to_dict())
        if rank == 0:
            print("Done with shot %d / %d , time: %.4fs" % (img_num+1,num_imgs, tsim), flush=True)

        #loader = dxtbx.load(image_filename)
        #imageset = loader.get_imageset(filenames=[image_filename])
        #exp = Experiment()
        #exp.imageset = imageset
        #exp.crystal = CRYSTAL
        #exp.detector = loader.get_detector()
        #exp.beam = loader.get_beam()
        #expList = ExperimentList()
        #expList.append(exp)

