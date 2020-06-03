
import h5py

h5s = [h5py.File("twocol_run120_job%d.hdf5"%x,"r") for x in range(12)]
Nimg_per = [h["images"].shape[0] for h in h5s]
Nimg_tot = sum(Nimg_per)

h = h5s[0]
img_sh = h["images"].shape
Nenergy = h["spectrum_weights"].shape[1]
master_img_sh = (Nimg_tot, img_sh[1], img_sh[2], img_sh[3])
master_weights_sh = (Nimg_tot, Nenergy)
master_energies_sh = (Nimg_tot, Nenergy)
layout_img = h5py.VirtualLayout( shape=master_img_sh, dtype=h5s[0]['images'].dtype)
layout_energies = h5py.VirtualLayout( shape=master_energies_sh, dtype=h5s[0]['spectrum_energies'].dtype)
layout_weights = h5py.VirtualLayout( shape=master_weights_sh, dtype=h5s[0]['spectrum_weights'].dtype)
start = 0
for h in h5s:
    print(h.filename)
    imgs = h['images']
    energies = h["spectrum_energies"]
    weights = h["spectrum_weights"]
    N = imgs.shape[0]
    stop = start + N
    layout_img[start:stop] = h5py.VirtualSource(h.filename, "images", shape=imgs.shape)
    layout_energies[start:stop] = h5py.VirtualSource(h.filename, "images", shape=energies.shape)
    layout_weights[start:stop] = h5py.VirtualSource(h.filename, "images", shape=weights.shape)
    start = stop

with h5py.File("twocol_master_run120.hdf5", "w") as hout:
    other_keys = ["dark","mask","gain"]
    h = h5s[0]
    for key in other_keys:
        hout.create_dataset(key, data=h[key][()])
    img_dset = hout.create_virtual_dataset("images", layout_img) #, fillvalue=-5)
    hout.create_virtual_dataset("spectrum_weights", layout_energies) #, fillvalue=-5)
    hout.create_virtual_dataset("spectrum_energies", layout_weights) #, fillvalue=-5)
    det_str = h['images'].attrs['dxtbx_detector_string']
    beam_str = h['images'].attrs['dxtbx_beam_string']
    img_dset.attrs["dxtbx_detector_string"] = det_str
    img_dset.attrs["dxtbx_beam_string"] = beam_str 

