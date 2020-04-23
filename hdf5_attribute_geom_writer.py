
import json
import h5py
import numpy as np


class H5AttributeGeomWriter:

    def __init__(self, filename, image_shape, num_images, detector, beam, dtype=None, 
        compression_args=None, detector_and_beam_are_dicts=False):
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
            Examples:
              compression_args={"compression": "lzf"}  # Python only
              comression_args = {"compression": "gzip", "compression_opts":9}
        :param detector_and_beam_are_dicts:
        """
        if compression_args is None:
            compression_args = {}

        self.file_handle = h5py.File(filename, 'w')
        self.beam = beam
        self.detector = detector
        self.detector_and_beam_are_dicts = detector_and_beam_are_dicts
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
        beam = self.beam
        det = self.detector
        if not self.detector_and_beam_are_dicts:
            beam = beam.to_dict()
            det =det.to_dict()

        self.image_dset.attrs["dxtbx_beam_string"] = json.dumps(beam)
        self.image_dset.attrs["dxtbx_detector_string"] = json.dumps(det)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_handle.close()

    def __enter__(self):
        return self

    def close_file(self):
        self.file_handle.close()


def add_spectra_to_file(
        filepath, central_wavelengths=None, 
        energies=None, weights=None, overwrite=False,
        compression_args=None):
    """
    Write spectra to an HDF5AttributeGeometry file
    
    filepath: path to exising file create with the above geometry writer
    central_wavelengths: array of wavelengths; length is Nshots
    energies: 2D array of energies, Nshots elements, where each element is
        a list of Nenergy energy channels
    weights: same as energies
    overwrite: boolean flag to overwrite exisiting spectra
    compression_args: dictionary kwargs for h5py dataset.create_dataset
        Examples:
          compression_args={"compression": "lzf"}  # Python only
          comression_args = {"compression": "gzip", "compression_opts":9}
    """
    
    if compression_args is None:
        compression_args = {}
    
    energies_name = "spectrum_energies"
    weights_name = "spectrum_weights"
    central_wavelen_name = "central_wavelengths"

    if energies is not None and weights is not None : #central_wavelengths is not None:
        Nener = len(energies)
        Nweights = len(weights)
        energies_ = np.vstack(energies)
        weights_ = np.vstack(weights)
        assert len(energies_.shape)==2, "Energies should be 2 dimensional array"
        assert len(weights_.shape)==2, "Weights should be 2 dimensional array"
        assert energies_.shape==weights_.shape, "Weights and energies are not the same 2D shape"

    elif central_wavelengths is None:
        print("Nothing to do. Need both energies and weights, or else need central_wavelengths")
        return

    with h5py.File(filepath, "r+") as h5:
        Nimages = h5["images"].shape[0]

        if energies is not None and weights is not None:
            if Nimages != len(weights):
                raise ValueError("Images (len %d) should be same length as weights and energies" 
                                % Nimages)
            _check_for_overwrite(h5, energies_name, overwrite)
            _check_for_overwrite(h5, weights_name, overwrite)
            h5.create_dataset(energies_name, data=energies_, **compression_args)
            h5.create_dataset(weights_name, data=weights_, **compression_args)

        elif central_wavelengths is not None:
            if Nimages != len(central_wavelengths):
                raise ValueError("Images (len %d) should be same length as weights and energies" 
                                % Nimages)

            _check_for_overwrite(h5, central_wavelen_name, overwrite)
            h5.create_dataset(energies_name, data=central_wavelengths, **compression_args)


def _check_for_overwrite(h5_handle, key, overwrite):
    """ checks if key already exisits and deletes if overwrite is True"""
    keys = list(h5_handle.keys())
    if key in keys:
        if overwrite:
            del h5_handle[key]
        else:
            raise IOError("Key %s already exists in %s, please set overwrite=True" 
                        % (key, h5_handle.filename))
