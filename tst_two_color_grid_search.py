

import numpy as np
from scipy import constants
from copy import deepcopy

from cctbx import sgtbx, miller
from cctbx.crystal import symmetry
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope as strong_phil_scope
from dials.algorithms.indexing.compare_orientation_matrices import rotation_matrix_differences
import dxtbx
from dxtbx.model.experiment_list import ExperimentList, Experiment
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

np.random.seed(3142019)
# make random rotation about principle axes
x = col((-1, 0, 0))
y = col((0, -1, 0))
z = col((0, 0, -1))
rx, ry, rz = np.random.uniform(-180, 180, 3)
RX = x.axis_and_angle_as_r3_rotation_matrix(rx, deg=True)
RY = y.axis_and_angle_as_r3_rotation_matrix(ry, deg=True)
RZ = z.axis_and_angle_as_r3_rotation_matrix(rz, deg=True)
M = RX*RY*RZ
real_a = M*col((79, 0, 0))
real_b = M*col((0, 79, 0))
real_c = M*col((0, 0, 38))

# two color experiment, two energies 100 eV apart
ENERGYLOW = 8944
ENERGYHIGH = 9034
ENERGY_CONV = 1e10*constants.c*constants.h / constants.electron_volt
WAVELENLOW = ENERGY_CONV/ENERGYLOW
WAVELENHIGH = ENERGY_CONV/ENERGYHIGH

# dxtbx beam model description
beam_descr = {'direction': (0.0, 0.0, 1.0),
             'divergence': 0.0,
             'flux': 1e12,
             'polarization_fraction': 1.,
             'polarization_normal': (0.0, 1.0, 0.0),
             'sigma_divergence': 0.0,
             'transmission': 1.0,
             'wavelength': 1.4}

# dxtbx crystal description
cryst_descr = {'__id__': 'crystal',
              'real_space_a': real_a.elems,
              'real_space_b': real_b.elems,
              'real_space_c': real_c.elems,
              'space_group_hall_symbol': ' P 4nw 2abw'}

# monolithic camera description
detdist = 100.
pixsize = 0.1
im_shape = 1536, 1536
det_descr = {'panels':
               [{'fast_axis': (1.0, 0.0, 0.0),
                 'slow_axis': (0.0, -1.0, 0.0),
                 'gain': 1.0,
                 'identifier': '',
                 'image_size': im_shape,
                 'mask': [],
                 'material': '',
                 'mu': 0.0,
                 'name': 'Panel',
                 'origin': (-im_shape[0]*pixsize/2., im_shape[1]*pixsize/2., -detdist),
                 'pedestal': 0.0,
                 'pixel_size': (pixsize, pixsize),
                 'px_mm_strategy': {'type': 'SimplePxMmStrategy'},
                 'raw_image_offset': (0, 0),
                 'thickness': 0.0,
                 'trusted_range': (0.0, 1e6),
                 'type': ''}]}

x1, x2, y1, y2 = 827, 847, 809, 829

BEAM = BeamFactory.from_dict(beam_descr)
DETECTOR = DetectorFactory.from_dict(det_descr)
CRYSTAL = CrystalFactory.from_dict(cryst_descr)

# make a dummie HKL table with constant HKL intensity
# this is just to make spots
DEFAULT_F = 1e3
symbol = CRYSTAL.get_space_group().info().type().lookup_symbol()  # this is just P43212
sgi = sgtbx.space_group_info(symbol)
symm = symmetry(unit_cell=CRYSTAL.get_unit_cell(), space_group_info=sgi)
miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.6, d_max=999)
Famp = flex.double(np.ones(len(miller_set.indices())) * DEFAULT_F)
Famp = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()

imgs = []
Ncells_abc = 21, 21, 21
oversmaple = 2
print("Doing simulation at wavelength %.4f" % WAVELENLOW)
BEAM.set_wavelength(WAVELENLOW)
SIM = nanoBragg(DETECTOR, BEAM, panel_id=0)
SIM.Ncells_abc = Ncells_abc
SIM.Fhkl = Famp
SIM.Amatrix = sqr(CRYSTAL.get_A()).transpose()
SIM.interpolate = 0
SIM.oversample = oversmaple
SIM.xtal_shape = shapetype.Gauss
SIM.add_nanoBragg_spots()
raw_pix = SIM.raw_pixels
imgs.append(raw_pix.as_numpy_array())
# reset for next wavelength ..
SIM.free_all()

print("Doing simulation at wavelength %.4f" % WAVELENHIGH)
BEAM.set_wavelength(WAVELENHIGH)
SIM = nanoBragg(DETECTOR, BEAM, panel_id=0)
SIM.Ncells_abc = Ncells_abc
SIM.Fhkl = Famp
SIM.Amatrix = sqr(CRYSTAL.get_A()).transpose()
SIM.oversample = oversmaple
SIM.interpolate = 0
SIM.xtal_shape = shapetype.Gauss
SIM.add_nanoBragg_spots()
raw_pix2 = SIM.raw_pixels
imgs.append(raw_pix2.as_numpy_array())

# make two color simulator and store as image file
SIM.raw_pixels = raw_pix + raw_pix2

SIM.adc_offset_adu = 10
SIM.detector_psf_fwhm_mm = 0
SIM.quantum_gain = 1
SIM.readout_noise_adu = 3
print("Adding noise")
SIM.add_noise()

image_filename = "two_color_image_000001.cbf"
print("Saving two color image to file %s" % image_filename)
#SIM.to_smv_format_py(image_filename)
SIM.to_cbf(image_filename)

loader = dxtbx.load(image_filename)
imageset = loader.get_imageset(filenames=[image_filename])
exp = Experiment()
exp.imageset = imageset
exp.crystal = CRYSTAL
exp.detector = loader.get_detector()
exp.beam = loader.get_beam()
expList = ExperimentList()
expList.append(exp)

# sanity test: create a new nanoBragg instance and instantiate with this detector and beam from ythe loader
# and then verify show_params() hasnt changed from above ...
origin_before_save = SIM.dials_origin_mm
test = nanoBragg(loader.get_detector(), loader.get_beam(), panel_id=0)
origin_after_save = test.dials_origin_mm
test.free_all()
assert(np.allclose(origin_before_save, origin_after_save))

params = strong_phil_scope.extract()
params.spotfinder.threshold.algorithm = "dispersion"
params.spotfinder.filter.min_spot_size = 4
params.spotfinder.threshold.dispersion.global_threshold = 50
params.spotfinder.threshold.dispersion.kernel_size = 5, 5
strong_refls = flex.reflection_table.from_observations(experiments=expList, params=params)

print("Found %d refls" % len(strong_refls))

print("Begin the indexing")
index_params.indexing.known_symmetry.space_group = CRYSTAL.get_space_group().info()
index_params.indexing.known_symmetry.unit_cell = CRYSTAL.get_unit_cell()
index_params.indexing.known_symmetry.absolute_angle_tolerance = 5.0
index_params.indexing.known_symmetry.relative_length_tolerance = 0.3
index_params.indexing.basis_vector_combinations.max_refine = 1
index_params.indexing.two_color.high_energy = ENERGYHIGH
index_params.indexing.two_color.low_energy = ENERGYLOW
index_params.indexing.two_color.avg_energy = ENERGYLOW * .5 + ENERGYHIGH * .5
index_params.indexing.two_color.filter_by_mag = 5, 3
index_params.indexing.two_color.optimize_initial_basis_vectors = True

beam1 = deepcopy(BEAM)
beam2 = deepcopy(BEAM)
beam1.set_wavelength(WAVELENLOW)
beam2.set_wavelength(WAVELENHIGH)
INDEXED_LATTICES = two_color_grid_search(
    beam1, beam2, DETECTOR, strong_refls, expList, index_params, verbose=True)

INDEXED_LATTICE = INDEXED_LATTICES[0]
print(rotation_matrix_differences([CRYSTAL, INDEXED_LATTICE]))



misori = compare_with_ground_truth(
    real_a.elems, real_b.elems, real_c.elems,
    INDEXED_LATTICES, symbol="P43212")
print(misori)

out = get_two_color_rois(INDEXED_LATTICE, DETECTOR, BEAM)
exit()
#############################################

## assign indices
#El = ExperimentListFactory.from_filenames([image_filename])
## El = ExperimentListFactory.from_json_file(El_json, check_format=False)
#
##iset = El.imagesets()[0]
#mos_spread = 0
#Ncells_abc = 7, 7, 7
#mos_doms = 1
#profile = "gauss"
#beamsize = 0.1
#exposure_s = 1
#spectrum = [(WAVELENLOW, 5e11),
#            (WAVELENHIGH, 5e11)]
#total_flux = np.sum(spectrum)
#xtal_size = 0.0005
#
## TODO multi panel
## Make a strong spot mask that is used to fit tilting planes
##defaultF = 1e5
##symm = symmetry(unit_cell=INDEXED_LATTICE.get_unit_cell(),
##                space_group_info=INDEXED_LATTICE.get_space_group().info())
##miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
##Fampz = flex.double(np.ones(len(miller_set.indices())) * defaultF)
##Fampz = miller.array(miller_set=miller_set, data=Famp).set_observation_type_xray_amplitude()
##FF = [Fampz, None]
#FF = [SIM.Fhkl, None]
#energies = [ENERGYLOW, ENERGYHIGH]
#FLUX = [total_flux*.5, total_flux*.5]
#
##INDEXED_LATTICE = CRYSTAL
#
#BEAM = El.beams()[0]
#DET = El.detectors()[0]
#beams = []
#device_Id = 0
#simsAB = sim_utils.sim_colors(
#    INDEXED_LATTICE, DET, BEAM, FF,
#    energies,
#    FLUX, pids=None, profile=profile, cuda=False, oversample=1,
#    Ncells_abc=Ncells_abc, mos_dom=mos_doms, mos_spread=mos_spread,
#    exposure_s=exposure_s, beamsize_mm=beamsize, device_Id=device_Id,
#    show_params=False, accumulate=False, crystal_size_mm=xtal_size)
#
#refls_at_colors = []
#for i_en, en in enumerate(energies):
#    beam = deepcopy(BEAM)
#    beam.set_wavelength(ENERGY_CONV / en)
#    R = prediction_utils.refls_from_sims(simsAB[i_en], DET, beam, thresh=1e-2)
#    refls_at_colors.append(R)
#    beams.append(beam)
#
## this gets the integration shoeboxes, not to be confused with strong spot bound boxes
#out = prediction_utils.get_prediction_boxes(
#    refls_at_colors,
#    DET, beams, INDEXED_LATTICE, delta_q=0.0575,
#    ret_patches=True, ec='w', fc='none', lw=1)
#
#Hi, bbox_roi, bbox_panel_ids, bbox_masks, patches = out

#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################

# simulate the result
print("sim indexed lattice wavelen 1")
BEAM.set_wavelength(WAVELENLOW)
SIM = nanoBragg(DETECTOR, BEAM, panel_id=0)
SIM.Ncells_abc = Ncells_abc
SIM.Fhkl = Famp
SIM.Amatrix = sqr(INDEXED_LATTICE.get_A()).transpose()
SIM.oversample = oversmaple
SIM.interpolate = 0
SIM.xtal_shape = shapetype.Gauss
SIM.add_nanoBragg_spots()
raw_pix = SIM.raw_pixels
SIM.free_all()

print("sim indexed lattice wavelen 2")
BEAM.set_wavelength(WAVELENHIGH)
SIM = nanoBragg(DETECTOR, BEAM, panel_id=0)
SIM.Ncells_abc = Ncells_abc
SIM.Fhkl = Famp
SIM.Amatrix = sqr(INDEXED_LATTICE.get_A()).transpose()
SIM.oversample = oversmaple
SIM.interpolate = 0
SIM.xtal_shape = shapetype.Gauss
SIM.add_nanoBragg_spots()
SIM.raw_pixels += raw_pix
SIM.adc_offset_adu = 10
SIM.detector_psf_fwhm_mm = 0
SIM.quantum_gain = 1
SIM.readout_noise_adu = 3
print("Adding noise")
SIM.add_noise()

image_filename = "two_color_image_000002.cbf"
print("Saving second two color image to file %s" % image_filename)
SIM.to_cbf(image_filename)
SIM.free_all()

print("OK!")
########
#
####################
#
#
################
