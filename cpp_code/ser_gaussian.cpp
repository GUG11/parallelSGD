/* Initial test program. serial SGD */

#include "sgd.h"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: par_random_gaussian [n] [d] [learningRate] [num_iters]\n"); 
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]), d = atoi(argv[2]);
    double learningRate = atof(argv[3]);
    int numIters = atoi(argv[4]);
    arma::mat X(d, n, arma::fill::randn);
    arma::mat w(1, d, arma::fill::randu);
    arma::mat y = w * X;
    SGDProfile sgdProfile;
    LeastSquare learner(arma::mat(1, d, arma::fill::zeros));
    serialSGD(&learner, X, y, &sgdProfile, learningRate, numIters);
    return 0;
}
