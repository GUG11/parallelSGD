/* HOGWILD SGD */

#include "sgd.h"
#include "data_part.h"
#include "io.h"
#include "simu_data.h"
#include <stdexcept>
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc < 10) {
        std::fprintf(stderr, "Usage: par_minibatch_gaussian [n] [d] [s] [num_threads] [learningRate] [num_iters] [print_period] [log_period] [partition_method] [save(opt)]\n");
        exit(EXIT_FAILURE);
    }
    srand(0);   // fix the random seed
    int n = atoi(argv[1]), d = atoi(argv[2]), num_threads = atoi(argv[4]);
    double sparse_rate = atof(argv[3]);
    double learningRate = atof(argv[5]);
    int numIters = atoi(argv[6]);
    LogSettings logsettings(atoi(argv[7]), atoi(argv[8]));
    std::string partMethod = argv[9];
    bool save = 10 < argc;
    arma::mat X = generateGaussian(n, d, num_threads, sparse_rate);
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
    if (save) {
        std::string filedir = "../results/simulations/Gaussian" + std::to_string(sparse_rate).substr(0,3) + "/hogwild";
        std::string filename = partMethod + "_n" + std::to_string(n) + "_d" + std::to_string(d) + "_T" + std::to_string(numIters) + "_ths" + std::to_string(num_threads) + "_gamma" + std::to_string(learningRate);
        try {
            writeSGDProfile(filedir, filename, sgdProfile);
        } catch (std::exception& e) {
            std::cout << "Catch exception " << e.what();
            exit(EXIT_FAILURE);
        }
    }
    exit(EXIT_SUCCESS);
    return 0;
}
