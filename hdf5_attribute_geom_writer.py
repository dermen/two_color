
import json
import h5py
import numpy as np


class H5AttributeGeomWriter:

    def __init__(self, filename, image_shape, num_images, detector, beam, dtype=None, compression_args=None):
        """
        Simple class for writing dxtbx compatible HDF5 files
        :param filename:  input file path
        :param image_shape: shape of a single image (Npanel x Nfast x Nslow)
        :param num_images: how many images will you be writing to the file
        :param detector: dxtbx detector model
        :param beam: dxtbx beam model
        :param dtype: datatype for storage
        :param compression_args: compression arguments for h5py, lzf is performant and simple
            if you only plan to read file in python
        """
        if compression_args is None:
            compression_args = {}

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


def add_spectra_to_h5_attribute_geom_file(filepath, central_wavelengths=None, energies=None, weights=None, overwrite=False,
                                          compression_args=None):
    if compression_args is None:
        compression_args = {}
    energies_name = "spectrum_energies"
    weights_name = "spectrum_weights"
    central_wavelen_name = "central_wavelengths"

    if energies is not None and weights is not None : #central_wavelengths is not None:
        Nener = len(energies)
        Nweights = len(weights)
        if Nener != Nweights:
            raise ValueError("Weights (len %d) and energies (len %d) should be of same length" % (Nweights, Nener))

    elif central_wavelengths is None:
        print("Nothing to do. Need both energies and weights, or else need central_wavelengths")
        return

    with h5py.File(filepath, "r+") as h5:
        Nimages = h5["images"].shape[0]

        if energies is not None and weights is not None:
            if Nimages != len(weights):
                raise ValueError("Images (len %d) should be same length as weights and energies" % Nimages)
            _check_for_overwrite(h5, energies_name, overwrite)
            _check_for_overwrite(h5, weights_name, overwrite)
            h5.create_dataset(energies_name, data=energies, **compression_args)
            h5.create_dataset(weights_name, data=weights, **compression_args)

        elif central_wavelengths is not None:
            if Nimages != len(central_wavelengths):
                raise ValueError("Images (len %d) should be same length as weights and energies" % Nimages)

            _check_for_overwrite(h5, central_wavelen_name, overwrite)
            h5.create_dataset(energies_name, data=central_wavelengths, **compression_args)


def _check_for_overwrite(h5_handle, key, overwrite):
    keys = list(h5_handle.keys())
    if key in keys:
        if overwrite:
            del h5_handle[key]
        else:
            raise IOError("Key %s already exists in %f, please set overwrite=True" % (key, h5_handle.filename))
