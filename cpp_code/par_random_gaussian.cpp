/* parallel SGD random partition */

#include "sgd.h"
#include "data_part.h"
#include <cstdlib>

int main(int argc, char* argv[]) {
    if (argc != 8) {
        std::fprintf(stderr, "Usage: par_random_gaussian [n] [d] [num_threads] [learningRate] [num_iters] [print_period] [log_period]\n"); 
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]), d = atoi(argv[2]), num_threads = atoi(argv[3]);
    double learningRate = atof(argv[4]);
    int numIters = atof(argv[5]);
    LogSettings logsettings(atoi(argv[6]), atoi(argv[7]));
    arma::mat X(d, n, arma::fill::randn);
    arma::mat w(1, d, arma::fill::randu);
    arma::mat y = w * X;
    std::vector<SGDProfile> sgdProfiles(num_threads);
    std::vector<Learner*> learners(num_threads);
    for (int i = 0; i < num_threads; i++)
        learners[i] = new LeastSquare(arma::mat(1, d, arma::fill::zeros));
    std::vector<std::vector<int>> dataPartition;
    
    randomPartition(n, num_threads, dataPartition);
    parallelSGD(learners, X, y, dataPartition, sgdProfiles, learningRate, numIters, logsettings);

    printf("Finally loss: %f\n", learners[0]->computeLoss(X, y));
    for (auto& learner: learners) delete learner;
    return 0;
}
