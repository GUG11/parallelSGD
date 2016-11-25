"""
Brute force enumeration on parallel minibatch SGD
n = {1000,5000,10000}  number of samples
d = {100, 200, 1000} dimension
B = {100, 200, 500, 1000} batch size
P = {2,4,8,16} number of cores
numIters = 100000
learning rate = binary search on [1e-5, 5]
"""
from subprocess import call
import os
import glob

if __name__ == '__main__':
    ns = [1000, 5000, 10000]
    ds = [100, 200, 1000]
    Ps = [2, 4, 8]
    schemes = ['random', 'corr']
    nit = 100000
    lo, hi, tol = 1e-5, 0.01, 0.1
    filedir = os.path.join('..','results','simulations', 'Gaussian', 'hogwild')
    for scheme in schemes:
        for n in ns:
            for d in ds:
                for P in Ps:
                    filepattern = '%s_n%d_d%d_T%d_gamma*_ths%d' % (scheme, n, d, nit, P)
                    filepath = os.path.join(filedir, filepattern)
                    filepathc = glob.glob(filepath)
                    if len(filepathc) > 0:
                        print(filepathc, ' exists')
                    else:
                        lo, hi = 1e-5, 5
                        ret = 1
                        while 0.1 < hi - lo or ret != 0:
                            mid = (lo + hi) / 2
                            cmd = ['./hogwild_gaussian', str(n), str(d), str(P), str(mid), str(nit), '100', '100', scheme]
                            print(' '.join(cmd))
                            ret = call(cmd)
                            if ret == 0:
                                print('\n\nlearning rate %f converges!\n' % mid)
                                lo = mid
                            else:
                                print('\n\nlearning rate %f diverges!\n' % mid)
                                hi = mid
