#include "sgd.h"
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <algorithm>
#include <functional>
#include <unistd.h>
#include <thread>
#include <sys/types.h>
#include <stdexcept>
#include <atomic>

std::atomic_int next_tid(0);
std::atomic_bool diverge(false);

void serialSGD(Learner* learner, const arma::mat& X, const arma::mat& y, SGDProfile* sgdProfile, double learningRate, int numIters, const LogSettings& logsettings, std::vector<int> S) {
    int n = 0, d = X.n_rows, st = 0;
    int id = next_tid++;
    double elapsedTime = 0, curLoss = 0;
    std::clock_t tStart, tEnd;
    if (S.empty()) {
        S.resize(X.n_cols);
        for (int i = 0; i < int(X.n_cols); i++) S[i] = i;
    }
    n = S.size();
    sgdProfile->T = logsettings.log_period;
    std::printf("Serial SGD: size (%d, %d), num of iterations:%d\n", n, d, numIters);
    // starting SGD
    for (int t = 0; t < numIters; t++) {
        if (diverge) {
            printf("Tid: %d terminates due to divergence!\n", id);
            break;
        }
        tStart = std::clock();
        st = S[rand() % n];
        learner->update(X.col(st), y.col(st), learningRate);
        tEnd = std::clock();
        // record the runtime and loss function
        elapsedTime = double(tEnd - tStart) / CLOCKS_PER_SEC;
        if ((t + 1) % sgdProfile->T == 0) 
            curLoss = learner->computeLoss(X, y);
        // critical area
        sgdProfile->profile_lock.lock();
        sgdProfile->times.push_back(elapsedTime);
        if ((t + 1) % sgdProfile->T == 0) sgdProfile->objs.push_back(curLoss);
        sgdProfile->profile_lock.unlock();
        // check divergence
        if (!sgdProfile->objs.empty() && sgdProfile->objs.back() > 1.1 * sgdProfile->objs[0]) {
            diverge = true;
            printf("Tid %d: Divergence, loss is more than 1.1 times of the initial!\n", id);
        } else if (!sgdProfile->objs.empty() && std::isnan(sgdProfile->objs.back())) {
            diverge = true;
            printf("Tid %d: Divergence, NAN exists in the loss function!\n", id);
        }
        // print 
        if ((t + 1) % logsettings.print_period == 0) 
            printf("Tid: %d, epoch: %d, data index: %d, obj= %f, time= %f\n", \
                id, t + 1, st, sgdProfile->objs[t / sgdProfile->T], 
                std::accumulate(sgdProfile->times.begin() + t - logsettings.print_period, sgdProfile->times.begin() + t, 0.0));
    }
    printf("Thread %d finished.\n", id);
}

void parallelSGD(std::vector<Learner*>& learners, const arma::mat& X, const arma::mat& y, const std::vector<std::vector<int>>& dataPartition, std::vector<SGDProfile>& sgdProfile, double learningRate, int numIters, const LogSettings& logsettings) {
    int n = X.n_cols, d = X.n_rows, numThreads = learners.size();
    arma::mat w(1, d, arma::fill::zeros);
    std::printf("Parallel SGD: size (%d, %d), threads:%d\n", n, d, numThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
        threads.push_back(std::thread(serialSGD, learners[i], std::ref(X), std::ref(y), &sgdProfile[i], learningRate, numIters / numThreads, std::ref(logsettings), dataPartition[i]));
    }
    printf("Synchronizing all threads...\n");
    for (auto& th: threads) th.join();
    // aggregate the results
    for (auto& learner: learners) w += learner->getWeight();
    w /= numThreads;
    if (diverge) {
        diverge = false;
        throw std::runtime_error("Divergence from some workers.\n");
    }
    for (auto& learner: learners) learner->setWeight(w);
}

void parallelMinibatchSGD(std::vector<Learner*>& learners, const arma::mat& X, const arma::mat& y, Partition& partitionMethod, SGDProfile& sgdProfile, double learningRate, int numIters, int batchSize) {
    std::vector<std::vector<int>> dataPartition;
    arma::mat subMat;
    Correlation corr;
    LogSettings logsettings(batchSize, batchSize);
    std::clock_t tStart, tEnd;
    int numThreads = learners.size(), T = numIters / batchSize;
    std::vector<SGDProfile> sgdProfiles(numThreads);
    int n = X.n_cols;
    sgdProfile.T = 1;
    std::vector<arma::uword> randomSamples, randomShuffles(n);
    if (batchSize > n) throw std::runtime_error("Batch size is larger than data size.\n");
    for (int i = 0; i < n; i++) randomShuffles[i] = i; 
    printf("Compute correlation matrix.\n");
    xcorr(X, X, corr);
    for (int t = 0; t < T; t++) {
        // random sample
        printf("Outer loop %d: Sample batch=%d at random.\n", t, batchSize);
        random_shuffle(randomShuffles.begin(), randomShuffles.end());
        randomSamples.assign(randomShuffles.begin(), randomShuffles.begin() + batchSize);
        subMat = arma::abs(corr.ncc.submat(arma::uvec(randomSamples), arma::uvec(randomSamples))); 
        // partition
        printf("Parition\n");
        partitionMethod.partition(subMat, numThreads, dataPartition);
        /*PartMetrics pmetrics_c(subMat, dataPartition);
        pmetrics_c.printMetrics();*/
        for (int p = 0; p < numThreads; p++) {
            for (int i = 0; i < int(dataPartition[p].size()); i++) {
                dataPartition[p][i] = randomSamples[dataPartition[p][i]];
            }
        }
        printf("Training\n");
        tStart = std::clock();
        parallelSGD(learners, X, y, dataPartition, sgdProfiles, learningRate, batchSize, logsettings);
        tEnd = std::clock();
        sgdProfile.times.push_back(double(tEnd - tStart) / CLOCKS_PER_SEC);
        sgdProfile.objs.push_back(learners[0]->computeLoss(X, y));
        //if (sgdProfile.objs.back() > 10.0 * sgdProfile.objs[0]) throw std::runtime_error("Divergence: loss function is more than 10 times of the initial.\n");
        printf("Epoch: %d, obj= %f, time= %f\n", t, sgdProfile.objs[t], sgdProfile.times[t]);
    }
}


void hogwild(Learner* learner, const arma::mat& X, const arma::mat& y, Partition& partitionMethod, SGDProfile& sgdProfile, double learningRate, int numIters, const LogSettings& logsettings, int numThreads) {
    std::vector<std::vector<int>> dataPartition;
    Correlation corr;
    int n = X.n_cols, d = X.n_rows;
    printf("Compute correlation matrix.\n");
    xcorr(X, X, corr);
    printf("Parition\n");
    partitionMethod.partition(arma::abs(corr.ncc), numThreads, dataPartition);
    std::printf("Hogwild: size (%d, %d), threads:%d\n", n, d, numThreads);
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; i++) {
        threads.push_back(std::thread(serialSGD, learner, std::ref(X), std::ref(y), &sgdProfile, learningRate, numIters / numThreads, std::ref(logsettings), dataPartition[i]));
    }
    printf("Synchronizing all threads...\n");
    for (auto& th: threads) th.join();
    if (diverge) {
        diverge = false;
        throw std::runtime_error("Divergence from some workers.\n");
    }
}
