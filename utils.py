# assign indices
import numpy as np
from cctbx.crystal import symmetry
from cctbx import miller
from cctbx.array_family import flex
from copy import deepcopy
from scipy.spatial import cKDTree
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
                       delta_q=0.0575, ret_patches=False, device_Id=0):

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


def get_strong_rois(panel, refls, bb=7, min_dist=10, verbose=False):
    """
    GET ROI for two color strong spots
    panel, 2D pixel array (single panel of a detector)
    refls, the strong refls on the panel
    bb, the half width and half height of the ROIs
    min_dist, strong refls within this number of pixels will be merged as one
    """
    assert len(refls) > 0
    x,y,_ = map(lambda x: np.array(x)-0.5,prediction_utils.xyz_from_refl(refls) )
    tree = cKDTree(list(zip(x,y)))
    pairs = tree.query_pairs(r=min_dist, output_type='ndarray')
    npts = len(x)
    not_in_pairs = [i for i in range(npts) if i not in list(pairs.ravel())]
    new_points = [tree.data[i] for i in not_in_pairs]
    new_points += [np.mean([ tree.data[i] , tree.data[j]], axis=0) for i,j in pairs]
    if verbose:
        print("There were %d points, After merging there are %d points" % (npts, len(new_points)))
    xnew, ynew = np.array(new_points).T
    sdim, fdim = panel.shape
    rois = []
    for i,j in zip(xnew, ynew):
        i1 = int( max(i-bb,0))
        i2 = int(min(i+bb, fdim))
        j1 = int( max(j-bb,0))
        j2 = int( min(j+bb,sdim))
        rois.append( ((i1,i2),(j1,j2)) )
    return rois


def num_overlap_predicted_and_strong(Exp, refls,panels=None, thresh=1, 
        min_above_thresh=2, mosaic_param=7, defaultF=1e6,flux=1e12):
    if panels is None:
        iset = Exp.imageset 
        panels = np.array([p.as_numpy_array() for p in iset.get_raw_data(0)])
    DET = Exp.detector
    BEAM = Exp.beam
    CRYSTAL = Exp.crystal

    Rpp = prediction_utils.refls_by_panelname(refls)
    roi_per_pan = {}
    pids = []
    counts_per_pan = {}
    for ii, pid in enumerate(Rpp):
        panel = panels[pid]
        counts_per_pan[ii] = np.zeros_like(panel)
        refls = Rpp[pid]
        pids.append(pid)
        rois = get_strong_rois(panel=panel, refls=refls)
        roi_per_pan[ii] = rois
        for (i1,i2),(j1,j2) in rois:
            counts_per_pan[ii][j1:j2, i1:i2] += 1

    en_low, en_high = ENERGY_LOW, ENERGY_HIGH
    m = mosaic_param
    prediction_sims = sim_utils.sim_colors(crystal=CRYSTAL, detector=DET, beam=BEAM, 
        fluxes=[flux,flux], energies=[en_low, en_high], fcalcs=[defaultF,defaultF], 
        Ncells_abc=(m,m,m),mos_dom=1, mos_spread=0, roi_pp=roi_per_pan,pids=pids, 
        Gauss=True, exposure_s=1, counts_pp=counts_per_pan, accumulate=True)

    n_pred_at_strong = 0
    for ii, pid in enumerate(pids):
        rois = roi_per_pan[ii]
        sim_panel = prediction_sims[pid]
        assert sim_panel is not None
        for (i1,i2), (j1,j2) in rois:
            roi_sim = sim_panel[j1:j2, i1:i2]
            if (roi_sim > thresh).sum() > min_above_thresh:
                n_pred_at_strong += 1
    return n_pred_at_strong


def choose_best_crystal(expList, refls):
    seen_ucells = []
    best_crystal = None
    best_n = None
    raw = expList.imagesets()[0].get_raw_data(0)
    panels = np.array([p.as_numpy_array() for p in raw])
    for E in expList:
        n = num_overlap_predicted_and_strong(Exp=E, refls=refls, panels=panels)
        this_ucell = E.crystal.get_unit_cell()
        if not seen_ucells:
            seen_ucells.append(this_ucell)
        else:
            already_seen = any([uc.is_similar_to(this_ucell) for uc in seen_ucells])
            if not already_seen:
                seen_ucells.append(this_ucell)
            else:
                continue
        if best_crystal is None:
            best_crystal = E.crystal
            best_n = n
        else:
            if n > best_n:
                best_crystal = E.crystal
                best_n = n

    return best_crystal, best_n

