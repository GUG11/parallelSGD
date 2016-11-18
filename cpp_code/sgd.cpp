#include "sgd.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>

#include <unistd.h>
#include <sys/types.h>

void serialSGD(Learner& learner, const arma::mat& X, const arma::mat& y, SGDProfile& sgdProfile, double learningRate, int numIters, const LogSettings& logsettings, std::vector<int> S) {
    int n = 0, d = 0, st = 0;
    std::clock_t tStart, tEnd;
    if (S.empty()) S.resize(X.n_cols);
    for (int i = 0; i < int(X.n_cols); i++) S[i] = i;
    n = S.size();
    d = X.n_rows;
    sgdProfile.T = logsettings.log_period;
    std::printf("Serial SGD: size (%d, %d)\n", n, d);
    // starting SGD
    for (int t = 0; t < numIters; t++) {
        tStart = std::clock();
        st = S[rand() % n];
        learner.update(X.col(st), y.col(st), learningRate);
        tEnd = std::clock();
        sgdProfile.times.push_back(double(tEnd - tStart) / CLOCKS_PER_SEC);
        if (t % sgdProfile.T == 0) sgdProfile.objs.push_back(learner.computeLoss(X, y));
        if (t % logsettings.print_period == 0) 
            printf("Pid: %d, epoch: %d, data index: %d, obj= %f, time= %f\n", \
                getpid(), t, st, sgdProfile.objs[t / sgdProfile.T], 
                std::accumulate(sgdProfile.times.begin() + t - logsettings.print_period, sgdProfile.times.begin() + t, 0.0));
    }
}

