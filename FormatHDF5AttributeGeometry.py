from __future__ import absolute_import, division

import numpy as np
import ast
import h5py

from dxtbx.format.FormatHDF5 import FormatHDF5
from dials.array_family import flex
from dxtbx.format.FormatStill import FormatStill


class FormatHDF5AttributeGeometry(FormatHDF5, FormatStill):
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
        if "gain" in keys:
            return False
        return True

    def __init__(self, image_file, **kwargs):
        from dxtbx import IncorrectFormatError
        if not self.understand(image_file):
            raise IncorrectFormatError(self, image_file)
        FormatStill.__init__(self, image_file, **kwargs)
        self._handle = h5py.File(image_file, "r")
        self._image_dset = self._handle["images"]
        self._geometry_define()

    def _geometry_define(self):
        det_str = self._image_dset.attrs["dxtbx_detector_string"]
        beam_str = self._image_dset.attrs["dxtbx_beam_string"]
        self._cctbx_detector = self._detector_factory.from_dict(ast.literal_eval(det_str))
        self._cctbx_beam = self._beam_factory.from_dict(ast.literal_eval(beam_str))

    def get_num_images(self):
        return self._image_dset.shape[0]

    def get_raw_data(self, index=0):
        self.panels = self._image_dset[index]
        if self.panels.dtype == np.float64:
            flex_data = [flex.double(p) for p in self._image_dset[index]]
        else:
            flex_data = [flex.double(p.astype(np.float64)) for p in self._image_dset[index]]
        return tuple(flex_data)

    def get_detectorbase(self, index=None):
        raise NotImplementedError

    def get_detector(self, index=None):
        return self._cctbx_detector

    def get_beam(self, index=None):
        return self._cctbx_beam


if __name__ == '__main__':
    import sys
    for arg in sys.argv[1:]:
        print(FormatHDF5AttributeGeometry.understand(arg))
