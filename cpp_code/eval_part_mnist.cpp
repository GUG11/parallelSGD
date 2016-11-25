#include <armadillo>
#include "sgd.h"
#include "io.h"
#include "metrics.h"
#include <cmath>
#include <cstdlib>
#include <algorithm>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::fprintf(stderr, "Usage: eval_part_mnist [n] [k]\n"); 
        exit(EXIT_FAILURE);
    }
    std::string filedir = "../data/MNIST";
    std::string filenameTrainingImages = "train-images-idx3-ubyte";
    std::string filenameTrainingLabels = "train-labels-idx1-ubyte";
    arma::mat trainingImages, X, labels;
    arma::mat trainingLabels;
    readMnistImages(filedir, filenameTrainingImages, trainingImages);
    readMnistLabels(filedir, filenameTrainingLabels, trainingLabels);
    int n = std::min(atoi(argv[1]), int(trainingImages.n_cols)), k = atoi(argv[2]);
    // int d = trainingImages.n_rows, c = round(trainingLabels.max()) + 1;
    X = trainingImages.cols(0, n-1);
    labels = trainingLabels.cols(0, n-1);

    BalancedMinCutParition bmcPart;
    Correlation corr;
    RandomPartition rPart;
    std::vector<std::vector<int>> dataPartition_r, dataPartition_c;
    printf("Compute correlation\n");
    xcorr(X, X, corr);   
    printf("Partition\n");
    arma::mat edgeMat = arma::abs(corr.corr);
    rPart.partition(edgeMat, k, dataPartition_r);
    bmcPart.partition(edgeMat, k, dataPartition_c);  
    PartMetrics pmetrics_c(edgeMat, dataPartition_c), pmetrics_r(edgeMat, dataPartition_r);
    printf("Correlation min cut\n");
    pmetrics_c.printMetrics();
    printf("Random partition\n");
    pmetrics_r.printMetrics();
    // save results
    std::string fileDir = "../results/real_data/MNIST/partitions";
    std::string filePattern = "_n" + std::to_string(n) + "_k" + std::to_string(k) + ".csv";
    std::string fileName1 = "random" + filePattern;
    std::string fileName2 = "corr" + filePattern;
    pmetrics_r.aveWeights.save(fileDir + '/' + fileName1, arma::arma_ascii);
    pmetrics_c.aveWeights.save(fileDir + '/' + fileName2, arma::arma_ascii);
    return 0;
}
