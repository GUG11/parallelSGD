#include "sgd.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <thread>
#include <functional>
#include <unistd.h>
#include <sys/types.h>

int next_tid = 0;

void serialSGD(Learner* learner, const arma::mat& X, const arma::mat& y, SGDProfile& sgdProfile, double learningRate, int numIters, const LogSettings& logsettings, std::vector<int> S) {
    int n = 0, d = X.n_rows, st = 0;
    int id = next_tid++;
    std::clock_t tStart, tEnd;
    if (S.empty()) {
        S.resize(X.n_cols);
        for (int i = 0; i < int(X.n_cols); i++) S[i] = i;
    }
    n = S.size();
    sgdProfile.T = logsettings.log_period;
    std::printf("Serial SGD: size (%d, %d)\n", n, d);
    // starting SGD
    for (int t = 0; t < numIters; t++) {
        tStart = std::clock();
        st = S[rand() % n];
        learner->update(X.col(st), y.col(st), learningRate);
        tEnd = std::clock();
        sgdProfile.times.push_back(double(tEnd - tStart) / CLOCKS_PER_SEC);
        if (t % sgdProfile.T == 0) sgdProfile.objs.push_back(learner->computeLoss(X, y));
        if (t % logsettings.print_period == 0) 
            printf("Tid: %d, epoch: %d, data index: %d, obj= %f, time= %f\n", \
                id, t, st, sgdProfile.objs[t / sgdProfile.T], 
                std::accumulate(sgdProfile.times.begin() + t - logsettings.print_period, sgdProfile.times.begin() + t, 0.0));
    }
    printf("Thread %d finished.\n", id);
}

void parallelSGD(std::vector<Learner*>& learners, const arma::mat& X, const arma::mat& y, const std::vector<std::vector<int>>& dataPartition, std::vector<SGDProfile>& sgdProfile, double learningRate, int numIters, const LogSettings& logsettings) {
    int n = X.n_cols, d = X.n_rows, numThreads = learners.size();
    arma::mat w(1, d, arma::fill::zeros);
    std::printf("Parallel SGD: size (%d, %d), threads:%d\n", n, d, numThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
        threads.push_back(std::thread(serialSGD, learners[i], std::ref(X), std::ref(y), std::ref(sgdProfile[i]), learningRate, numIters, std::ref(logsettings), dataPartition[i]));
    }
    printf("Synchronizing all threads...\n");
    for (auto& th: threads) th.join();
    // aggregate the results
    for (auto& learner: learners) w += learner->getWeight();
    w /= numThreads;
    for (auto& learner: learners) learner->setWeight(w);
}

void parallelMinibatchSGD(std::vector<Learner*>& learners, const arma::mat& X, const arma::mat& y, Partition& partitionMethod, std::vector<SGDProfile>& sgdProfile, double learningRate, int numIters, const LogSettings& logsettings) {
}
