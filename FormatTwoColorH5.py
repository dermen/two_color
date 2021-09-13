from __future__ import absolute_import, division

import numpy as np
import h5py

from dxtbx.format.Format import Format
from dxtbx.format.FormatStill import FormatStill
from dxtbx.format.FormatHDF5 import FormatHDF5
from cxid9114.utils import pppg

from cxid9114.twocol.geom import DET,BEAM
from cxid9114 import utils

PPPG_ARGS = {'Nhigh': 100.0,
             'Nlow': 100.0,
             'high_x1': -5.0,
             'high_x2': 5.0,
             'inplace': True,
             'low_x1': -5.0,
             'low_x2': 5.0,
             'plot_details': False,
             'plot_metric': False,
             'polyorder': 3,
             'verbose': False,
             'window_length': 51}


class FormatTwoColorH5(FormatHDF5, FormatStill):
    """
    Class for reading D9114 simulated monolithic cspad data
    """
    @staticmethod
    def understand(image_file):
        try:
            img_handle = h5py.File(image_file,"r")
            keys = img_handle.keys()
        except (IOError, AttributeError) as err:
            return False
        if "imagesLD91" not in keys:
            print("no image")
            return False
        if "gain" not in keys:
            print("no gain")
            return False
        if "dark" not in keys:
            print("no dark")
            return False
        if "mask" not in keys:
            print("no mask")
            return False
        return True

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatStill.__init__(self, image_file, **kwargs)
        self._handle = h5py.File(image_file, "r")
        # load image components
        self._raw_panels = self._handle["imagesLD91"]
        self._is_low_gain = self._handle["gain"][()]
        self._pedestal = self._handle["dark"][()]
        self._mask = self._handle["mask"][()]
        self._low_gain_val = 6.85  # TODO : make this optional
        # load geometry components 
        self._geometry_define()

    def _correct_panels(self):
        self.panels -= self._pedestal
        self._apply_mask()
        pppg(self.panels,
             self._is_low_gain,
             self._mask,
             **PPPG_ARGS)
        self.panels[self._is_low_gain] = self.panels[self._is_low_gain]*self._low_gain_val

    def _geometry_define(self):
        self._cctbx_detector = self._detector_factory.from_dict(DET)
        self._cctbx_beam = self._beam_factory.from_dict(BEAM)

    def get_num_images(self):
        return self._raw_panels.shape[0]

    def _correct_raw_data(self, index):
        self.panels = self._raw_panels[index].astype(np.float64)  # 32x185x388 psana-style cspad array
        self._correct_panels()  # applies dark cal, common mode, and gain, in that order..

    def get_raw_data(self, index=0):
        self._correct_raw_data(index)
        return utils.psana_data_to_aaron64_data(self.panels, as_flex=True)

    def _apply_mask(self):
        self.panels *= self._mask

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_image_file(self, index=None):
        return Format.get_image_file(self)

    def get_detector(self, index=None):
        return self._cctbx_detector

    def get_beam(self, index=None):
        return self._cctbx_beam


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(FormatTwoColorH5.understand(arg))
