from __future__ import absolute_import, division

import numpy as np
import h5py
from scipy.signal import savgol_filter

import os
homedir = os.path.expanduser("~")
searchdir = os.path.join(homedir, ".dxtbx")
import sys
sys.path.append(searchdir)
from FormatHDF5AttributeGeometry  import FormatHDF5AttributeGeometry
from dials.array_family import flex
from dxtbx.model import Beam
from dxtbx.format.FormatHDF5 import FormatHDF5
from dxtbx.format.FormatStill import FormatStill

PPPG_ARGS = {'Nhigh': 100.0,
             'Nlow': 100.0,
             'high_x1': -5.0,
             'high_x2': 5.0,
             'inplace': True,
             'low_x1': -5.0,
             'low_x2': 5.0,
             'polyorder': 3,
             'verbose': False,
             'window_length': 51}


class FormatHDF5AttributeGeometryLD91(FormatHDF5AttributeGeometry):
    """
    Class for reading D9114 simulated monolithic cspad data
    """
    @staticmethod
    def understand(image_file):
        try:
            img_handle = h5py.File(image_file, "r")
            keys = img_handle.keys()
        except (IOError, AttributeError) as err:
            return False
        if "images" not in keys:
            return False
        images = img_handle["images"]
        if "dxtbx_detector_string" not in images.attrs:
            return False
        if "dxtbx_beam_string" not in images.attrs:
            return False
        if "gain" not in keys:
            return False
        if "dark" not in keys:
            return False
        if "mask" not in keys:
            return False
        return True

    def _start(self):
        super(FormatHDF5AttributeGeometryLD91, self)._start()
        self._is_low_gain = self._handle["gain"][()]
        self._pedestal = self._handle["dark"][()]
        self._mask = self._handle["mask"][()]
        self._image_dset = self._handle["images"]
        self._low_gain_val = 6.85  # TODO : make this optional by using a dataset in the hdf5 file

    def _correct_raw_data(self, index):
        self.panels = self._image_dset[index].astype(np.float64)  # 32x185x388 psana-style cspad array
        self._correct_panels()  # ap

    def get_raw_data(self, index=0):
        self._correct_raw_data(index)
        return psana_data_to_aaron64_data(self.panels, as_flex=True)

    def _apply_mask(self):
        self.panels *= self._mask

    def _correct_panels(self):
        self.panels -= self._pedestal
        self._apply_mask()
        pppg(self.panels,
             self._is_low_gain,
             self._mask,
             **PPPG_ARGS)
        self.panels[self._is_low_gain] = self.panels[self._is_low_gain]*self._low_gain_val


def pppg(shot_, gain, mask=None, window_length=101, polyorder=5,
        low_x1=-10, low_x2 = 10, high_x1=-20, high_x2=20, Nhigh=1000,
         Nlow=500, verbose=False, before_and_after=False,
         inplace=False):

    if not inplace:
        shot = shot_.copy()
    else:
        shot = shot_
    if mask is not None:
        is_low = gain*mask
        is_high = (~gain)*mask
    else:
        is_low = gain
        is_high = (~gain)

    low_gain_pid = np.where([ np.any( is_low[i] ) for i in range(32)])[0]
    high_gain_pid = np.where([ np.any( is_high[i] ) for i in range(32)])[0]

    bins_low = np.linspace(low_x1, low_x2, Nlow)
    bins_high = np.linspace(high_x1,high_x2,Nhigh)

    xdata_low = bins_low[1:]*.5 + bins_low[:-1]*.5
    xdata_high = bins_high[1:]*.5 + bins_high[:-1]*.5

    if before_and_after:
        before_low = []
        after_low = []
        before_high = []
        after_high = []

    common_mode_shifts = {}
    for i_pan in low_gain_pid:
        pixels = shot[i_pan][ is_low[i_pan] ]
        Npix = is_low[i_pan].sum()
        pix_hist = np.histogram( pixels, bins=bins_low, density=True)[0]
        smoothed_hist = savgol_filter( pix_hist, window_length=window_length,
                                    mode='constant',polyorder=polyorder)
        pk_val = np.argmax(smoothed_hist)
        shift = xdata_low[pk_val]
        common_mode_shifts[(i_pan, 'low')] = shift
        if verbose:
            print("shifted panel %d by %.4f" % (i_pan, shift))
        if before_and_after:
            before_low.append(pix_hist)
            pix_hist_shifted = np.histogram(pixels-shift, bins=bins_low, density=True)[0]
            after_low.append(pix_hist_shifted)
    for i_pan in high_gain_pid:
        pixels = shot[i_pan][is_high[i_pan]]
        Npix = is_high[i_pan].sum()
        pix_hist = np.histogram(pixels, bins=bins_high, density=True)[0]
        smoothed_hist = savgol_filter(pix_hist, window_length=window_length,mode='constant', polyorder=polyorder)
        pk_val=np.argmax(smoothed_hist)
        shift = xdata_high[pk_val]
        common_mode_shifts[(i_pan, 'high')] = shift
        if verbose:
            print("shifted panel %d by %.4f"%(i_pan,shift))
        if before_and_after:
            before_high.append( pix_hist)
            pix_hist_shifted = np.histogram( pixels-shift, bins=bins_high, density=True)[0]
            after_high.append( pix_hist_shifted)

    for (i_pan,which), shift in common_mode_shifts.items():
        if which =='low':
            shot[i_pan][ is_low[i_pan]] = shot[i_pan][ is_low[i_pan]] - shift
        if which == 'high':
            shot[i_pan][ is_high[i_pan]] = shot[i_pan][ is_high[i_pan]] - shift
    if verbose:
        print("Mean shift: %.4f"%(np.mean(common_mode_shifts.values())))
    if inplace:
        return
    elif before_and_after:
        return xdata_low, before_low, after_low, xdata_high, before_high, after_high, shot
    else:
        return shot


def psana_data_to_aaron64_data(data, as_flex=False):
    """
    :param data:  32 x 185 x 388 cspad data
    :return: 64 x 185 x 194 cspad data
    """
    asics = []
    # check if necessary to convert to float 64
    dtype = data.dtype
    if as_flex and dtype != np.float64:
        dtype = np.float64
    for split_asic in [(asic[:, :194], asic[:, 194:]) for asic in data]:
        for sub_asic in split_asic:  # 185x194 arrays
            if as_flex:
                sub_asic = np.ascontiguousarray(sub_asic, dtype=dtype)  # ensure contiguous arrays for flex
                sub_asic = flex.double(sub_asic)  # flex data beith double
            asics.append(sub_asic)
    if as_flex:
        asics = tuple(asics)
    return asics


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(FormatHDF5AttributeGeometryLD91.understand(arg))
