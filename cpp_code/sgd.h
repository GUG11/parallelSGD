#pragma once

#include <vector>
#include <armadillo>    // column-major (d x N)
#include "loss_function.h"

struct SGDProfile {  // tracking SGD time and objective function
    int T;      // record objective function every T epoches
    std::vector<double> objs;   // objective functions
    std::vector<double> times;   // time consumption in each epoch
};

struct LogSettings {
    int print_period;   // print every print_period epoches
    int log_period;   // T in SGDProfile
    LogSettings(int print_T=100, int log_T=1) : print_period(print_T), log_period(log_T) {}
};

/**
 * S: set of indexes used (vector)
 * sgdProfile: output
 * return
 * */
void serialSGD(Learner& learner, const arma::mat& X, const arma::mat& y, SGDProfile& sgdProfile, double learningRate=0.1, int numIters=10000, const LogSettings& logsettings=LogSettings(), std::vector<int> S={});
