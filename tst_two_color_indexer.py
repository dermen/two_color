

import numpy as np
from scipy import constants
from IPython import embed

from cctbx import sgtbx, miller
from cctbx.crystal import symmetry
from dials.array_family import flex
from dials.command_line.find_spots import phil_scope
from dials.command_line.stills_process import phil_scope as stills_proc_phil
from dials.algorithms.indexing.compare_orientation_matrices import rotation_matrix_differences
import dxtbx
from dxtbx.model.experiment_list import ExperimentList, Experiment
from dxtbx.model.beam import BeamFactory
from dxtbx.model.crystal import CrystalFactory
from dxtbx.model.detector import DetectorFactory, Detector, Panel
from scitbx.matrix import sqr, col, rec
from simtbx.nanoBragg import nanoBragg
from simtbx.nanoBragg import shapetype

from two_color.two_color_indexer import TwoColorIndexer
from two_color.two_color_phil import two_color_phil_scope

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

image_filename = "two_color_image_000001.img"
print("Saving two color image to file %s" % image_filename)
SIM.to_smv_format_py(image_filename)

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

params = phil_scope.extract()
params.spotfinder.threshold.algorithm = "dispersion"
params.spotfinder.filter.min_spot_size = 2
strong_refls = flex.reflection_table.from_observations(experiments=expList, params=params)

print("Found %d refls" % len(strong_refls))

print ("Begin the indexing")
stills_proc_phil.adopt_scope(two_color_phil_scope)
index_params = stills_proc_phil.extract()
index_params.refinement.parameterisation.beam.fix = "all"
index_params.refinement.parameterisation.detector.fix = "all"
index_params.indexing.refinement_protocol.n_macro_cycles = 1
index_params.indexing.refinement_protocol.mode = None
index_params.indexing.known_symmetry.space_group = CRYSTAL.get_space_group().info()
index_params.indexing.known_symmetry.unit_cell = CRYSTAL.get_unit_cell()
index_params.indexing.debug = True
index_params.indexing.basis_vector_combinations.max_refine = 20
index_params.indexing.known_symmetry.absolute_angle_tolerance = 5.0
index_params.indexing.known_symmetry.relative_length_tolerance = 0.3
index_params.indexing.two_color.high_energy = ENERGYHIGH
index_params.indexing.two_color.low_energy = ENERGYLOW
index_params.indexing.two_color.avg_energy = ENERGYLOW * .5 + ENERGYHIGH * .5
index_params.indexing.stills.refine_all_candidates = False

orient = TwoColorIndexer(strong_refls, expList, index_params)
orient.index()

rotation_matrix_differences([CRYSTAL, orient.refined_experiments.crystals()[0]])
rotation_matrix_differences([CRYSTAL, orient.refined_experiments.crystals()[0]])
print(rotation_matrix_differences([CRYSTAL, orient.refined_experiments.crystals()[0]]))

INDEXED_LATTICE = orient.refined_experiments.crystals()[0]

# simulate the result
print("sim indexed lattice wavelen 1")
BEAM.set_wavelength(WAVELENLOW)
SIM = nanoBragg(DETECTOR, BEAM, panel_id=0)
SIM.Ncells_abc = Ncells_abc
SIM.Fhkl = Famp
SIM.Amatrix = sqr(INDEXED_LATTICE.get_A()).transpose()
SIM.oversample = oversmaple
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
SIM.xtal_shape = shapetype.Gauss
SIM.add_nanoBragg_spots()
SIM.raw_pixels += raw_pix
SIM.adc_offset_adu = 10
SIM.detector_psf_fwhm_mm = 0
SIM.quantum_gain = 1
SIM.readout_noise_adu = 3
print("Adding noise")
SIM.add_noise()

image_filename = "two_color_image_000002.img"
print("Saving second two color image to file %s" % image_filename)
SIM.to_smv_format_py(image_filename)
SIM.free_all()
