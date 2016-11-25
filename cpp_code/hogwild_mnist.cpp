#include "sgd.h"
#include "data_part.h"
#include "io.h"
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include "metrics.h"

int main(int argc, char* argv[]) {
    if (argc != 9) {
        std::fprintf(stderr, "Usage: [hogwild_mnist mnist_dir] [n] [num_threads] [learningRate] [num_iters] [print_period] [log_period] [partition_method]\n"); 
        exit(EXIT_FAILURE);
    }
    std::string filedir = argv[1];
    std::string filenameTrainingImages = "train-images-idx3-ubyte";
    std::string filenameTrainingLabels = "train-labels-idx1-ubyte";
    std::string filenameTestingImages = "t10k-images-idx3-ubyte";
    std::string filenameTestingLabels = "t10k-labels-idx1-ubyte";
    arma::mat trainingImages, testingImages, images, labels;
    arma::mat trainingLabels, predictedTrainingLabels, testingLabels, predictedTestingLabels;
    readMnistImages(filedir, filenameTrainingImages, trainingImages);
    readMnistLabels(filedir, filenameTrainingLabels, trainingLabels);
    readMnistImages(filedir, filenameTestingImages, testingImages);
    readMnistLabels(filedir, filenameTestingLabels, testingLabels);
    int n = std::min(int(trainingImages.n_cols), atoi(argv[2])), d = trainingImages.n_rows, c = round(trainingLabels.max()) + 1;
    int num_threads=atoi(argv[3]), numIters=atoi(argv[5]);
    double learningRate=atof(argv[4]);
    LogSettings logsettings(atoi(argv[6]), atoi(argv[7]));
    std::string partMethod = argv[8];
    images = trainingImages.cols(0, n-1);
    labels = trainingLabels.cols(0, n-1);
    SGDProfile sgdProfile;
    Softmax learner(arma::mat(c, d, arma::fill::zeros));
    BalancedMinCutParition bmcPart;
    RandomPartition rPart;
    try {
        if (partMethod == "random") 
            hogwild(&learner, images, labels, rPart, sgdProfile, learningRate, numIters, logsettings, num_threads);
        else if (partMethod == "corr")
            hogwild(&learner, images, labels, bmcPart, sgdProfile, learningRate, numIters, logsettings, num_threads);
        predictedTrainingLabels = learner.predict(images);
        predictedTestingLabels = learner.predict(testingImages);
        printf("Final training accuracy: %f\n", accuracy(predictedTrainingLabels, labels));
        printf("Final testing accuracy: %f\n", accuracy(predictedTestingLabels, testingLabels));
    } catch (std::exception& e) {
        std::cout << "Catch exception " << e.what();
        exit(EXIT_FAILURE);
    }
    try {
        writeSGDProfile("../results/real_data/MNIST/hogwild", partMethod + "_n" + std::to_string(n) + "_T" + std::to_string(numIters) + "_gamma" + std::to_string(learningRate) + "_ths" + std::to_string(num_threads), sgdProfile);
    } catch (std::exception& e) {
        std::cout << "Catch exception " << e.what();
        exit(EXIT_FAILURE);
    }
    exit(EXIT_SUCCESS);
    return 0;
}
