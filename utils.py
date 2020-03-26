# assign indices
import numpy as np
from cctbx.crystal import symmetry
from cctbx import miller
from cctbx.array_family import flex
from copy import deepcopy
from cxid9114.prediction import prediction_utils
from cxid9114.sim import sim_utils
from cxid9114.parameters import ENERGY_CONV, ENERGY_LOW, ENERGY_HIGH
import json
import h5py


class H5AttributeGeomWriter:

    def __init__(self, filename, image_shape, num_images, detector, beam, dtype=None, compression_args={}):

        self.file_handle = h5py.File(filename, 'w')
        self.beam = beam
        self.detector = detector
        if dtype is None:
            dtype = np.float64
        dset_shape = (num_images,) + tuple(image_shape)
        self.image_dset = self.file_handle.create_dataset(
            "images", shape=dset_shape,
            dtype=dtype, **compression_args)

        self._write_geom()

        self._counter = 0

    def add_image(self, image):
        if self._counter >= self.image_dset.shape[0]:  # TODO update dset size feature, which is possible
            raise IndexError("Maximum number of images is %d" % (self.image_dset.shape[0]))
        self.image_dset[self._counter] = image
        self._counter += 1

    def _write_geom(self):
        self.image_dset.attrs["dxtbx_beam_string"] = json.dumps(self.beam.to_dict())
        self.image_dset.attrs["dxtbx_detector_string"] = json.dumps(self.detector.to_dict())

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_handle.close()

    def __enter__(self):
        return self

    def close_file(self):
        self.file_handle.close()


def write_hdf5_attribute_file(dset, detector, beam):
    dset.attrs["dxtbx_beam_string"] = json.dumps(beam)
    dset.attrs["dxtbx_detector_string"] = json.dumps(detector)


def get_two_color_rois(crystal, detector, beam, thresh=1e-2,
                       mos_spread=0, mos_doms=1, Ncells_abc=(7, 7, 7),
                       beamsize=0.1, total_flux=1e12, profile="gauss",
                       xtal_size=0.0005, defaultF=1e3, cuda=False,
                       delta_q=0.0575, ret_patches=False):

    # TODO multi panel
    symm = symmetry(unit_cell=crystal.get_unit_cell(),
                    space_group_info=crystal.get_space_group().info())
    miller_set = symm.build_miller_set(anomalous_flag=True, d_min=1.5, d_max=999)
    Fampz = flex.double(np.ones(len(miller_set.indices())) * defaultF)
    Fampz = miller.array(miller_set=miller_set, data=Fampz).set_observation_type_xray_amplitude()
    FF = [Fampz, None]
    energies = [ENERGY_LOW, ENERGY_HIGH]
    FLUX = [total_flux*.5, total_flux*.5]

    #INDEXED_LATTICE = CRYSTAL

    beams = []
    device_Id = 0
    simsAB = sim_utils.sim_colors(
        crystal, detector, beam, FF,
        energies,
        FLUX, pids=None, profile=profile, cuda=cuda, oversample=1,
        Ncells_abc=Ncells_abc, mos_dom=mos_doms, mos_spread=mos_spread,
        exposure_s=1, beamsize_mm=beamsize, device_Id=device_Id,
        show_params=False, accumulate=False, crystal_size_mm=xtal_size)

    refls_at_colors = []
    for i_en, en in enumerate(energies):
        beam_at_color = deepcopy(beam)
        beam_at_color.set_wavelength(ENERGY_CONV / en)
        R = prediction_utils.refls_from_sims(simsAB[i_en], detector, beam_at_color, thresh=thresh)
        refls_at_colors.append(R)
        beams.append(beam)

    # this gets the integration shoeboxes, not to be confused with strong spot bound boxes
    out = prediction_utils.get_prediction_boxes(
        refls_at_colors,
        detector, beams, crystal, delta_q=delta_q,
        ret_patches=ret_patches, ec='w', fc='none', lw=1)

    # NOTE
    # Hi, bbox_roi, bbox_panel_ids, bbox_masks, patches = out
    # if ret_patches is False:
    #   Hi, bbox_roi, bbox_panel_ids, bbox_masks = out
    return out


def real_abc_from_Amat(Amat, from_nanoBragg=False):
    from scitbx.matrix import sqr
    A = sqr(Amat).inverse()
    if not from_nanoBragg:
        A = A.transpose()
    A = A.elems
    real_a = A[0], A[3], A[6]
    real_b = A[1], A[4], A[7]
    real_c = A[2], A[5], A[8]
    return real_a, real_b, real_c
