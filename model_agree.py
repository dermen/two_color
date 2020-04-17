from argparse import ArgumentParser
import os
parser = ArgumentParser()
parser.add_argument("--n", default=-1, type=int)
parser.add_argument('--f', type=str, required=True, help="input image file")
parser.add_argument('--minnum', type=int, default=7)
args = parser.parse_args()

input_file = args.f
basename = os.path.basename(input_file)
basename = os.path.splitext(basename)[0]

from pylab import *
import json

num_ranks = 6
all_x, all_y, all_az = [],[],[]
#basename = "Rank%d_twocol_run120_job0_py3_thincspad.json"
count = 0
all_strongs =0
broke = False
for r in range(num_ranks):
    print("process rank %d " %r)
    json_f = "/Users/dermen/two_color_testing/agreement_data/Rank%d_%s.json" % (r, basename)
    J = json.load(open(json_f, "r"))
    shots = J["com_diffs"].keys()
    com_data = J["com_diffs"]
    for s in shots:
        mods = list(com_data[s].keys())
        nums = [len(com_data[s][m]) for m in mods if com_data[s][m]]
        if not nums:
            print("Not any!")
            continue
        if not any([n > args.minnum for n in nums]):
            print("All have fewer than %d strong indexed ref!" % args.minnum)
            continue

        mod = mods[np.argmax(nums)]

        #if len(mods) != 1:

        #    continue
        #mod = list(mods)[0]
        #for mod in mods:
        com_diffs = com_data[s][mod]
        #com_diffs = J["closest"][s][mod]
        com_diffs = [c for c in com_diffs if c is not None]
        if not com_diffs:
            print("All none...")
            continue
        all_strongs += len(com_diffs)
        xdiff, ydiff = zip(*com_diffs)
        azi_diffs = J["azi_diffs"][s][mod]
        all_x += list(xdiff)
        all_y += list(ydiff)
        all_az += azi_diffs
        count += 1

print(count, all_strongs)

style.use('ggplot')
figure(1)
hexbin(all_x, all_y)
xlabel("x prediction offset (mm)")
xlabel("y prediction offset (mm)")
cbar = colorbar()
cbar.ax.set_title("# of images")

figure(2)
x = np.array(all_x)
y = np.array(all_y)
np.sqrt(x**2 + y**2)
d = np.sqrt(x**2 + y**2)
hist(d, bins='auto')
xlabel("prediction offset magnitude in millimeters")
ylabel("bincount")
ylabel("number of images")

figure(3)
hist(x, bins='auto', histtype='step', lw=2, label='x, median=%.4f mm, sig=%.4f' % (np.median(x), np.std(x)))
hist(y, bins='auto', histtype='step', lw=2, label="y, median=%.4f mm, sig=%.4f" % (np.median(y), np.std(y)))
xlabel("deviation (millimeters)")
ylabel("number of images")
legend()
show()

