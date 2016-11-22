/* parallel minibatch SGD */

#include "sgd.h"
#include "data_part.h"
#include "io.h"
#include <stdexcept>
#include <iostream>
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::fprintf(stderr, "Usage: par_minibatch_gaussian [n] [d] [num_threads] [learningRate] [num_iters] [batchsize] [partition_method]\n"); 
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]), d = atoi(argv[2]), num_threads = atoi(argv[3]);
    double learningRate = atof(argv[4]);
    int numIters = atoi(argv[5]), batchSize = atoi(argv[6]);
    std::string partMethod = argv[7];
    arma::mat X(d, n, arma::fill::randn);
    arma::mat w(1, d, arma::fill::randu);
    arma::mat y = w * X;
    SGDProfile sgdProfile;
    std::vector<Learner*> learners(num_threads);
    BalancedMinCutParition bmcPart;
    RandomPartition rPart;
    for (int i = 0; i < num_threads; i++)
        learners[i] = new LeastSquare(arma::mat(1, d, arma::fill::zeros));
    
    try {
        if (partMethod == "random") 
            parallelMinibatchSGD(learners, X, y, rPart, sgdProfile, learningRate, numIters,  batchSize);
        else if (partMethod == "corr")
            parallelMinibatchSGD(learners, X, y, bmcPart, sgdProfile, learningRate, numIters,  batchSize);
        printf("Finally loss: %f\n", learners[0]->computeLoss(X, y));
    } catch (std::exception& e) {
        std::cout << "Catch exception " << e.what();
        exit(EXIT_FAILURE);
    }
    
    writeSGDProfile("../results/simulations/Gaussian/minibatch", partMethod + "_n" + std::to_string(n) + "_d" + std::to_string(d) + "_B" + std::to_string(batchSize) + "_T" + std::to_string(numIters) + "_gamma" + std::to_string(learningRate) + "_ths" + std::to_string(num_threads), sgdProfile);
    for (auto& learner: learners) delete learner;
    exit(EXIT_SUCCESS);
    return 0;
}
