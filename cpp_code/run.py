from subprocess import call
import os, argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exe_name', help='executable program name', type=str)
    parser.add_argument('-n', '--numsamples', help='number of samples', type=int, default=1000)
    parser.add_argument('-d', '--dimension', help='dimension', type=int, default=100)
    parser.add_argument('-nthreads', '--numthreads', help='number of threads', type=int, default=1)
    parser.add_argument('-lr', '--learningrate', help='learning rate', type=float, default=0.0005)
    parser.add_argument('-nit', '--numiterations', help='number of iterations', type=int, default=10000)
    parser.add_argument('-tp', '--printperiod', help='print the loss every Tp epoches', type=int, default=100)
    parser.add_argument('-tl', '--logperiod', help='log the loss every Tl epoches', type=int, default=100)
    args = parser.parse_args()
    cmd = [args.exe_name, str(args.numsamples), str(args.dimension), str(args.numthreads), str(args.learningrate), str(args.numiterations), str(args.printperiod), str(args.logperiod)]
    print(' '.join(cmd))
    call(cmd)
