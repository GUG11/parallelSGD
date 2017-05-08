"""
Brute force enumeration on hogwild MNIST
n = {1000,5000,10000}  number of samples
d = {100, 200, 1000} dimension
B = {100, 200, 500, 1000} batch size
P = {2,4,8,16} number of cores
numIters = 100000
learning rate = binary decay on [1e-5, 5]

update 12/07 add another dimension (sparse rate)
s = [0.1, 0.2, 0.3, 0.5, 1]
"""
from subprocess import call
import os, time
import glob

if __name__ == '__main__':
    ns = [500, 1000, 2000, 5000, 10000]
    Ps = [1, 2, 4, 8]
    schemes = ['random', 'corr']
    nit = 100000
    lo, hi, tol = 0, 0.01, 0.01
    relax = 0.8

    filedir = os.path.join('..', 'results', 'real_data', 'MNIST', 'hogwild')
    mnist_dir = os.path.join('..', 'data', 'MNIST')
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    for n in ns:
        for P in Ps:
            filepattern = '*_n%d_T%d_ths%d_gamma*' % (n, nit, P)
            filepath = os.path.join(filedir, filepattern)
            filepathc = glob.glob(filepath)
            if len(filepathc) > 0:
                print("File %s already exists\n" % filepath)
                time.sleep(1)
                continue

            ret = 1
            while ret != 0:
                for scheme in schemes:
                    lo, hi = 0, 1

                    # if os.path.isfile(filepath):
                    #    continue
                    while tol < hi - lo or ret != 0:
                        print('Search in [%f, %f)' % (lo, hi))
                        mid = (lo + hi) / 2
                        cmd = ['./hogwild_mnist', mnist_dir, str(n), str(P), str(mid), str(nit), '100', '100', scheme]
                        print(' '.join(cmd))
                        ret = call(cmd)
                        if ret == 0:
                            print('\n\nlearning rate %f converges!\n' % mid)
                            lo = mid
                        else:
                            print('\n\nlearning rate %f diverges!\n' % mid)
                            hi = mid
                    if scheme == 'random':
                        gamma_r = lo
                    else:
                        gamma_c = lo
                gamma0 = min(gamma_c, gamma_r)
                gamma = gamma0 * relax
                print('maximum step length searched: random:%f, corr:%f, choose the minimum:%f, relaxation factor:%f, final step length:%f\n' % (gamma_r, gamma_c, gamma0, relax, gamma))
                for scheme in schemes:
                    cmd = ['./hogwild_mnist', mnist_dir, str(n), str(P), str(gamma), str(nit), '100', '100', scheme, 'save']
                    print(' '.join(cmd))
                    time.sleep(1)
                    ret = call(cmd)
                    if ret != 0:
                        print('Final step length %f even diverges! repeat this step!!!' % gamma)
                        del_file = os.path.join(filedir, filename)
                        if os.path.exists(del_file) and scheme == 'corr':
                            print('delete file %s' % filename)
                            os.remove(del_file)
                        time.sleep(1.5)
                        break
                    elif scheme == 'random':
                        filename = '%s_n%d_T%d_ths%d_gamma%f' % (scheme, n, nit, P, gamma)

