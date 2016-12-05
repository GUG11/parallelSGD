"""
Brute force enumeration on parallel minibatch SGD
n = {1000,5000,10000}  number of samples
d = {100, 200, 1000} dimension
B = {100, 200, 500, 1000} batch size
P = {2,4,8,16} number of cores
numIters = 100000
learning rate = binary decay on [1e-5, 5]
"""
from subprocess import call
import os
import glob

if __name__ == '__main__':
    ns = [1000, 5000, 10000]
    ds = [100, 200, 1000]
    Bs = [100, 200, 500, 1000]
    Ps = [2, 4, 8, 16]
    schemes = ['random', 'corr']
    nit = 100000
    lo, hi, tol = 0, 0.01, 1e-5
    filedir = os.path.join('..','results','simulations', 'Gaussian', 'minibatch')
    for scheme in schemes:
        for n in ns:
            for d in ds:
                for B in Bs:
                    for P in Ps:
                        hi = 0.01
                        ret = 1
                        while tol < hi - lo:
                            mid = (lo + hi) / 2

                            filepattern = '%s_n%d_d%d_B%d_T%d_gamma%f_ths%d' % (scheme, n, d, B, nit, mid, P)
                            filepath = os.path.join(filedir, filepattern)
                            if os.path.isfile(filepath):
                                continue
                            cmd = ['./par_minibatch_gaussian', str(n), str(d), str(P), str(mid), str(nit), str(B), scheme]
                            print(' '.join(cmd))
                            ret = call(cmd)
                            if ret == 0:
                                print('\n\nlearning rate %f converges!\n' % mid)
                                # lo = mid
                            else:
                                print('\n\nlearning rate %f diverges!\n' % mid)
                            hi = mid
