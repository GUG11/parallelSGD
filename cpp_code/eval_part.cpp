#include "sgd.h"
#include "data_part.h"
#include "io.h"
#include <string>
#include <cstdlib>
#include "simu_data.h"

int main(int argc, char* argv[]) {
    if (argc != 5) {
        std::fprintf(stderr, "Usage: eval_part [n] [d] [k] [s]\n"); 
        exit(EXIT_FAILURE);
    }
    int n = atoi(argv[1]), d = atoi(argv[2]), k = atoi(argv[3]);
    double sparse_rate = atof(argv[4]);
    arma::mat X = generateGaussian(n, d, k, sparse_rate);
    BalancedMinCutParition bmcPart;
    Correlation corr;
    RandomPartition rPart;
    std::vector<std::vector<int>> dataPartition_r, dataPartition_c;
    printf("Compute correlation\n");
    xcorr(X, X, corr);   
    printf("Partition\n");
    arma::mat edgeMat = arma::abs(corr.ncc);
    rPart.partition(edgeMat, k, dataPartition_r);
    bmcPart.partition(edgeMat, k, dataPartition_c);  
    PartMetrics pmetrics_c(edgeMat, dataPartition_c), pmetrics_r(edgeMat, dataPartition_r);
    printf("Correlation min cut\n");
    pmetrics_c.printMetrics();
    printf("Random partition\n");
    pmetrics_r.printMetrics();
    // save results
    std::string fileDir = "../results/simulations/Gaussian/partitions";
    std::string filePattern = "_n" + std::to_string(n) + "_d" + std::to_string(d) + "_k" + std::to_string(k) + "_s" + std::to_string(sparse_rate) + ".csv";
    std::string fileName1 = "random" + filePattern;
    std::string fileName2 = "corr" + filePattern;
    pmetrics_r.aveWeights.save(fileDir + '/' + fileName1, arma::arma_ascii);
    pmetrics_c.aveWeights.save(fileDir + '/' + fileName2, arma::arma_ascii);
    return 0;
}
