from pylab import *
import json

num_ranks = 6
all_x, all_y, all_az = [],[],[]
basename ="Rank%d_twocol_run120_job0_py3_thincspad.json"
for r in range(num_ranks):
    print("process rank %d " %r)
    json_f = "/Users/dermen/two_color_testing/agreement_data/%s" % basename
    J = json.load(open(json_f % r, "r"))
    shots = J["com_diffs"].keys()
    for s in shots:
        mod = J["com_diffs"][s].keys()
        if len(mod) != 1:
            continue
        mod = list(mod)[0]
        #com_diffs = J["com_diffs"][s][mod]
        com_diffs = J["closest"][s][mod]
        com_diffs = [c for c in com_diffs if c is not None]
        if not com_diffs:
            continue
        xdiff, ydiff = zip(*com_diffs)
        azi_diffs = J["azi_diffs"][s][mod]
        all_x += list(xdiff)
        all_y += list(ydiff)
        all_az += azi_diffs
        

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
hist(x, bins='auto', histtype='step', lw=2, label='x, median=%.3f mm' % np.median(x))
hist(y, bins='auto', histtype='step', lw=2, label="y, median=%.3f mm" % np.median(y))
xlabel("deviation (millimeters)")
ylabel("number of images")
legend()
show()

