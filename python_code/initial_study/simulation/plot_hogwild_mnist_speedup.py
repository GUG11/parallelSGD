import numpy as np
from misc import utils
import matplotlib.pyplot as plt
import matplotlib
import os, re
import glob

def parseData(contents):
    lines = contents.split('\n')
    epoches, times, losses = [], [], []
    count = 0
    for l in lines:
        count += 1
        data = filter(None, l.split(' '))
        if count < 3 or len(data) != 3:
            continue
        epoches.append(int(data[0]))
        times.append(float(data[1]))
        losses.append(float(data[2]))
    return epoches, times, losses


if __name__ == '__main__':
    matplotlib.rcParams.update({'font.size': 30})
    n= 5000
    Ps = [1,2,4,8]
    schemes = ['random', 'corr']
    nit = 100000
    filedir = os.path.join('..', 'results', 'real_data', 'MNIST', 'hogwild')
    fig = plt.figure(num=1, figsize=(20, 12))
    ax = fig.add_subplot(1,1,1)

    for scheme in schemes:
        end_it = []
        speed_up = []
        for P in Ps:
            filepattern = '%s_n%d_T%d_ths%d_gamma*' % (scheme, n, nit, P)
            filepath = os.path.join(filedir, filepattern)
            filepathc = glob.glob(filepath)
            if len(filepathc) == 0:
                raise Exception("File %s does not exist\n" % filepath)
            # gammas, ys = [], []
            for filepath in filepathc:
                with open(filepath, 'r') as ins:
                    print('reading %s' % filepath)
                    contents = ins.read()
                epoches, times, losses = parseData(contents)
                x = np.array(epoches)
                y = np.array(losses)
                for i in xrange(len(y)):
                    if y[i] > 0 and y[0] / y[i] > 5:
                        end_it.append(sum(times[:i+1]))
                        print('P:%d, scheme:%s, end epoch:%d, time:%f' % (P, scheme, epoches[i], end_it[-1]))
                        break
                # ys.append(y)
            # ys = np.array(ys)
            # yssum = np.sum(ys, axis=1)
            # opt = np.argmin(yssum)
        for i in xrange(len(end_it)):
            speed_up.append(float(end_it[0]) / end_it[i] * Ps[i])
        ax.plot(Ps, speed_up, label=scheme, lw=2)

    utils.set_axis(ax, xlabel='iterations', ylabel='speed up', xticks=None, yticks=None, xlim=None, fontsize=30)
    plt.tight_layout()
    savefile = 'speedup_n%d_T%d.svg' % (n, nit)
    save_dir = os.path.join(filedir, savefile)
    fig.savefig(save_dir)
    savefile = 'speedup_n%d_T%d.pdf' % (n, nit)
    save_dir = os.path.join(filedir, savefile)
    fig.savefig(save_dir)
    savefile = 'speedup_n%d_T%d.jpg' % (n, nit)
    save_dir = os.path.join(filedir, savefile)
    fig.savefig(save_dir)
    plt.cla()
    print('succeed saving %s' % save_dir)
