/* Initial test program. serial SGD */

#include "sgd.h"

int main() {
    int n = 1000, d = 100;
    arma::mat X(d, n, arma::fill::randn);
    arma::mat w(1, d, arma::fill::randu);
    arma::mat y = w * X;
    SGDProfile sgdProfile;
    LeastSquare learner(arma::mat(1, d, arma::fill::zeros));
    double learningRate = 0.0005;
    int numIters = 10000;
    serialSGD(learner, X, y, sgdProfile, learningRate, numIters);
    return 0;
}
