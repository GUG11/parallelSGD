/* parallel minibatch SGD */

#include "sgd.h"
#include "data_part.h"
#include "io.h"
#include <stdexcept>
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::fprintf(stderr, "Usage: par_minibatch_gaussian [n] [d] [num_threads] [learningRate] [num_iters] [print_period] [log_period] [partition_method]\n"); 
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]), d = atoi(argv[2]), num_threads = atoi(argv[3]);
    double learningRate = atof(argv[4]);
    int numIters = atoi(argv[5]);
    LogSettings logsettings(atoi(argv[6]), atoi(argv[7]));
    std::string partMethod = argv[8];
    arma::mat X(d, n, arma::fill::randn);
    arma::mat w(1, d, arma::fill::randu);
    arma::mat y = w * X;
    SGDProfile sgdProfile;
    LeastSquare learner(arma::mat(1, d, arma::fill::zeros));;
    BalancedMinCutParition bmcPart;
    RandomPartition rPart;
    
    try {
        if (partMethod == "random") 
            hogwild(&learner, X, y, rPart, sgdProfile, learningRate, numIters, logsettings, num_threads);
        else if (partMethod == "corr")
            hogwild(&learner, X, y, bmcPart, sgdProfile, learningRate, numIters, logsettings, num_threads);
        printf("Finally loss: %f\n", learner.computeLoss(X, y));
    } catch (std::exception& e) {
        std::cout << "Catch exception " << e.what();
        exit(EXIT_FAILURE);
    }
    try {
        writeSGDProfile("../results/simulations/Gaussian/hogwild", partMethod + "_n" + std::to_string(n) + "_d" + std::to_string(d) + "_T" + std::to_string(numIters) + "_gamma" + std::to_string(learningRate) + "_ths" + std::to_string(num_threads), sgdProfile);
    } catch (std::exception& e) {
        std::cout << "Catch exception " << e.what();
        exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
    return 0;
}
