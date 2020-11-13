
from argparse import ArgumentParser

parser = ArgumentParser("mae the predictions from the models")
parser.add_argument('--f', type=str, required=True, help="input image file")
parser.add_argument('--ngpu', type=int, default=1, help="number of GPUs")
parser.add_argument('--plot', action="store_true")
parser.add_argument('--cuda', action="store_true")

args = parser.parse_args()

from pylab import *

import json
import h5py
from dxtbx.model import Crystal
import dxtbx
import os
from two_color import utils

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.rank
    size = comm.size
    has_mpi = True
except ImportError:
    rank = 0
    size = 1
    has_mpi = False


fig = figure()
ax = gca()

if rank == 0:
    print("Loading image file", flush=True)
input_file = args.f
rootdir = os.path.dirname(input_file)
basename = os.path.basename(input_file)
basename = os.path.splitext(basename)[0]
loader = dxtbx.load(input_file)
DET = loader.get_detector(0)
iset = loader.get_imageset(loader.get_image_file())

if rank == 0:
    print("Load json model file", flush=True)
modeldir = os.path.join(rootdir, "models")
if not os.path.exists(modeldir):
    os.makedirs(modeldir)
J = json.load(open("%s/Rank%d_models_%s.json" % (modeldir, rank, basename), "r"))
keys = list(J.keys())

hdf5dir = os.path.join(rootdir, "hdf5data")
if not os.path.exists(hdf5dir):
    os.makedirs(hdf5dir)

hdf5name = os.path.join(hdf5dir, "Rank%d_%s.hdf5" % (rank, basename))
with h5py.File(hdf5name, "w") as output_file:
    for i_image, k in enumerate(keys):
        img_num = int(k)

        BEAM = loader.get_beam(img_num)
        spectrum_weights = BEAM.get_spectrum_weights().as_numpy_array()
        if i_image == 0:
            spectrum_energies = BEAM.get_spectrum_energies().as_numpy_array()
            energies = output_file.create_dataset("spectrum_energies", data=spectrum_energies)
        output_file.create_dataset("Image%d/spectrum_weights" % img_num, data=spectrum_weights)

        imgdir = os.path.join(rootdir, "model_images/%s/Rank%d/Image%d" % (basename, rank, img_num))
        if not os.path.exists(imgdir):
            os.makedirs(imgdir)

        models = J[k]
        if not models:
            continue  # NOTE not sure why this happens
        seen_unit_cells = []
        for i_model, Amat in enumerate(models):

            modeldir = os.path.join(imgdir, "model%d" % i_model)

            if not os.path.exists(modeldir):
                os.makedirs(modeldir)

            a, b, c = utils.real_abc_from_Amat(Amat)
            C = Crystal(a, b, c, "P43212")

            this_ucell = C.get_unit_cell()
            if not seen_unit_cells:
                seen_unit_cells.append(this_ucell)
            else:
                already_seen = any([uc.is_similar_to(this_ucell) for uc in seen_unit_cells])
                if not already_seen:
                    seen_unit_cells.append(this_ucell)
                else:
                    continue

            imgs = [p.as_numpy_array() for p in iset[img_num:img_num+1].get_raw_data(0)]
            imgs = np.array(imgs)

            devId = np.random.randint(0, args.ngpu)
            out = utils.get_two_color_rois(C, DET, BEAM, ret_patches=True, cuda=args.cuda,device_Id=devId )

            Hi, bbox_roi, bbox_panel_ids, bbox_masks, patches = out
            output_file.create_dataset("Image%d/Model%d/Hi" % (img_num, i_model), data=Hi)
            output_file.create_dataset("Image%d/Model%d/bbox_roi" % (img_num, i_model), data=bbox_roi)
            output_file.create_dataset("Image%d/Model%d/bbox_panel" % (img_num, i_model), data=bbox_panel_ids)
            output_file.create_dataset("Image%d/Model%d/a_real" % (img_num, i_model), data=a)
            output_file.create_dataset("Image%d/Model%d/b_real" % (img_num, i_model), data=b)
            output_file.create_dataset("Image%d/Model%d/c_real" % (img_num, i_model), data=c)

            # store the ROI images
            roi_pix = [imgs[pid][y1:y2, x1:x2] for pid, (x1, x2, y1, y2)
                       in zip(bbox_panel_ids, bbox_roi)]
            max_Ydim = max([roi.shape[0] for roi in roi_pix])
            max_Xdim = max([roi.shape[1] for roi in roi_pix])
            padded_roi = [np.pad(roi, ((0, max_Ydim-roi.shape[0]), (0, max_Xdim-roi.shape[1])), mode='constant') for roi in roi_pix]
            output_file.create_dataset("Image%d/Model%d/padded_roi_images" % (img_num, i_model), data=padded_roi)

            if args.plot:
                u_pids = set(bbox_panel_ids)

                n_pan = len(u_pids)

                m = imgs[imgs > 0].mean()
                s = imgs[imgs > 0].std()
                vmin = m-s
                vmax=m+3*s
                for i_pid, pid in enumerate(u_pids):
                    ax.clear()
                    if i_pid % 5 == 0:
                        if rank == 0:
                            print("Plotting Image %d/%d, model %d/%d pid %d/%d"
                                  % (i_image+1, len(keys), i_model+1, len(models), i_pid+1, n_pan), flush=True)
                    figname = os.path.join(modeldir, "panel%02d.png" % pid)
                    ax.imshow(imgs[pid], vmin=vmin, vmax=vmax)
                    pid_pos = np.where(np.array(bbox_panel_ids) == pid)[0]
                    patches_on_panel = patches[pid]
                    for p in patches_on_panel:
                        ax.add_patch(p)
                    savefig(figname)
            else:
                print("Rank %d: Done with Image %d/%d, model %d/%d"
                      % (rank, i_image + 1, len(keys), i_model + 1, len(models)), flush=True)


