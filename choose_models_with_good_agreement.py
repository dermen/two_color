
from argparse import ArgumentParser

parser = ArgumentParser("mae the predictions from the models")
parser.add_argument('--f', type=str, required=True, help="input image file")
args = parser.parse_args()

from IPython import embed
from scipy.spatial import cKDTree
import h5py
from dials.array_family import flex
import dxtbx
import numpy as np
import matplotlib.path as mplPath
import json
import os
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

PLOT = False
if rank == 0:
    import pylab as plt
    plt.figure()
    ax = plt.gca()
print("Loading the master image file")
input_file = args.f
basename = os.path.basename(input_file)
basename = os.path.splitext(basename)[0]
loader = dxtbx.load(input_file)
det = loader.get_detector(0)
beam = loader.get_beam(0)

from dxtbx.model.experiment_list import ExperimentList, Experiment
exp = Experiment()
exp.detector = det
exp.beam = beam
El = ExperimentList()
El.append(exp)


print("Loading the strong spot reflections")
# load the strong spots
R = flex.reflection_table.from_file('/Users/dermen/two_color_testing/strong_%s.refl' % basename)

print("Loading the output hdf5 file")
H = h5py.File("/Users/dermen/two_color_testing/hdf5data/Rank%d_%s.hdf5" % (rank, basename), "r")
img_keys = [k for k in H.keys() if k.startswith("Image")]
img_numbers = [int(k.split("Image")[1]) for k in img_keys]

com_diffs = {}
azi_diffs = {}
all_panels = {}
all_com_bbox = {}
all_com_strong = {}
all_num_inside = {}
closest_offset = {}
for i_img, img_num in enumerate(img_numbers):
    com_diffs[img_num] = {}
    azi_diffs[img_num] = {}
    all_panels[img_num] = {}
    all_com_bbox[img_num] = {}
    all_com_strong[img_num] = {}
    all_num_inside[img_num] = {}
    closest_offset[img_num] = {}

    model_keys = [k for k in H["Image%d" % img_num].keys() if k.startswith("Model")]
    model_numbers = [int(k.split("Model")[1]) for k in model_keys]
    for model_num in model_numbers:
        print("Calculating agreement for image %d/%d, model %d/%d"
                % (i_img+1, len(img_numbers), model_num+1, len(model_numbers)))
        com_diffs[img_num][model_num] = []
        azi_diffs[img_num][model_num] = []
        all_panels[img_num][model_num] = []
        all_com_bbox[img_num][model_num] = []
        all_com_strong[img_num][model_num] = []
        all_num_inside[img_num][model_num] = []
        closest_offset[img_num][model_num] = []

        Rimg = R.select(R['img_num'] == img_num)
        Rimg.centroid_px_to_mm(El)
        Rimg.map_centroids_to_reciprocal_space(El)

        #predicted_refls.centroid_px_to_mm(El)
        nstrong = len(Rimg)
        strong_spot_azimuthals = []
        all_s1 = []
        for i_r in range(nstrong):
            r = Rimg[i_r]
            node = det[int(r['panel'])]
            orig = np.array(node.get_origin())
            fast = np.array(node.get_fast_axis())
            slow = np.array(node.get_slow_axis())
            pixsize = node.get_pixel_size()[0]
            i_fs, i_ss, _ = r['xyzobs.px.value']
            i_fs = i_fs - 0.5
            i_ss = i_ss - 0.5
            s1 = node.get_pixel_lab_coord((i_fs, i_ss))
            all_s1.append(s1)
            #s1 = orig + i_fs*fast*pixsize + i_ss*slow*pixsize
            #dist = node.get_distance()
            #s1_norm = np.linalg.norm(s1)
            #two_theta = np.arccos( dist/s1_norm)
            azimuthal = np.arctan2(s1[1], s1[0])
            strong_spot_azimuthals.append(azimuthal * 180 / np.pi)
        Rimg['azimuthal'] = flex.double(strong_spot_azimuthals)

        # make strong spots from each panel easily accessible
        strong_by_panel = {}
        u_strong_pids = np.unique(Rimg["panel"].as_numpy_array())
        for pid in u_strong_pids:
            Rimg_pid = Rimg.select(Rimg["panel"] == int(pid))
            xstrong, ystrong, _ = map(np.array, zip(*list(Rimg_pid['xyzobs.px.value'])))
            xstrong -= 0.5
            ystrong -= 0.5
            strong_by_panel[pid] = {}
            strong_by_panel[pid]["x"] = xstrong
            strong_by_panel[pid]["y"] = ystrong
            strong_by_panel[pid]["azi"] = Rimg_pid["azimuthal"]
            strong_tree = cKDTree(list(zip(xstrong, ystrong)))
            strong_by_panel[pid]["tree"] = strong_tree

        # load the bboxes and check which strong spots are contained inside ...
        x1, x2, y1, y2 = zip(*H["Image%d/Model%d/bbox_roi" % (img_num, model_num)][()])
        bbox_pids = H["Image%d/Model%d/bbox_panel" % (img_num, model_num)][()]
        #
        for i_bbox, (i1, i2, j1, j2) in enumerate(zip(x1, x2, y1, y2)):
            verts = (i1, j1), (i2, j1), (i1, j2), (i2, j2)
            bbox_com = .5 * (i1 + i2), .5 * (j1 + j2)
            #= np.array([i1 + i2, j1 + j2]) * .5
            bbox_path = mplPath.Path(verts)
            pid = bbox_pids[i_bbox]
            if pid not in strong_by_panel:
                continue
            # find the lab coord
            node = det[int(pid)]
            orig = np.array(node.get_origin())
            fast = np.array(node.get_fast_axis())
            slow = np.array(node.get_slow_axis())
            pixsize = node.get_pixel_size()[0]

            bbox_com_s1 = node.get_pixel_lab_coord((bbox_com[0], bbox_com[1]))  # orig + bbox_com[0] * fast * pixsize + bbox_com[1] * slow * pixsize
            #i_fs, i_ss, _ = r['xyzobs.px.value']
            #i_fs = i_fs - 0.5
            #i_ss = i_ss - 0.5

            xstrong, ystrong = strong_by_panel[pid]["x"], strong_by_panel[pid]["y"]
            #strong_tree = strong_by_panel[pid]["tree"]
            azi_strong = strong_by_panel[pid]["azi"]  # azimuthal angle of each strong spot
            is_in_bbox = bbox_path.contains_points(list(zip(xstrong, ystrong)))
            num_inside = sum(is_in_bbox)
            if num_inside > 2:
                continue
            elif num_inside in [1, 2]:
                strong_tree = cKDTree(list(zip(xstrong[is_in_bbox], ystrong[is_in_bbox])))
                close_strongs = strong_tree.query_ball_point(bbox_com, r=5)
                if not close_strongs:
                    min_offset_lab = None
                else:
                    offsets = np.array(list(zip(xstrong[is_in_bbox], ystrong[is_in_bbox])))[close_strongs] - bbox_com
                    dists = np.sum(offsets**2, axis=1)
                    closest = np.argmin(dists)
                    min_offset = list(offsets[closest])
                    # 3D position of closest spot
                    min_x, min_y = xstrong[is_in_bbox][closest], ystrong[is_in_bbox][closest]
                    min_s1 = node.get_pixel_lab_coord((min_x, min_y)) #orig + min_x * fast * pixsize + min_y * slow * pixsize
                    min_offset_lab = list(np.array(min_s1[:2]) - np.array(bbox_com_s1[:2]))

                    if rank == 0:
                        if PLOT and img_num == 18 and model_num == 0:
                            img = H["Image%d/Model%d/padded_roi_images" % (img_num, model_num)][i_bbox]
                            ax.clear()
                            ax.imshow(img, vmax=400)
                            x, y = np.array(list(zip(xstrong[is_in_bbox], ystrong[is_in_bbox])))[close_strongs][closest]
                            ax.plot(x - i1, y - j1, 'o', color='Deeppink', ms=11, mfc='none', mew=1.5)
                            for x, y in zip(xstrong[is_in_bbox], ystrong[is_in_bbox]):
                                ax.plot(x - i1, y - j1, 'x', color='r', ms=11)
                            ax.plot(bbox_com[0] - i1, bbox_com[1] - j1, '*', color='w', ms=11, mfc='none', mew=1.5)
                            ax.add_patch(plt.Rectangle(xy=(-0.5, -0.5), width=i2 - i1, height=j2 - j1, ec='w', fc='none'))
                            plt.savefig("some_images/agreement_test_Image%d_model%d_bbox%d.png" % (img_num, model_num, i_bbox))

                strong_com = np.mean(list(zip(xstrong[is_in_bbox], ystrong[is_in_bbox])), axis=0)
                strong_com_x, strong_com_y = strong_com
                strong_com_s1 = node.get_pixel_lab_coord((strong_com_x, strong_com_y))  #+ strong_com_x * fast * pixsize + strong_com_y * slow * pixsize

                com_diff = np.array(strong_com_s1[:2]) - np.array(bbox_com_s1[:2])
                if num_inside == 2:
                    azi1, azi2 = np.where(is_in_bbox)[0]
                    azi_diff = abs(azi_strong[azi1] - azi_strong[azi2])
                    azi_diffs[img_num][model_num].append(azi_diff)
                else:
                    azi_diffs[img_num][model_num].append(0)

                com_diffs[img_num][model_num].append(list(com_diff))
                all_com_strong[img_num][model_num].append(list(strong_com))
                all_com_bbox[img_num][model_num].append(list(bbox_com))
                all_panels[img_num][model_num].append(pid)
                all_num_inside[img_num][model_num].append(num_inside)
                closest_offset[img_num][model_num].append(min_offset_lab)

# save results
outname = "/Users/dermen/two_color_testing/agreement_data/Rank%d_%s.json" % (rank, basename)
with open(outname, 'w') as out:
    all_output = {"com_diffs": com_diffs, "azi_diffs": azi_diffs, "closest": closest_offset}
    json.dump(all_output, out)
