#include <armadillo>
#include "sgd.h"
#include "io.h"
#include "metrics.h"

int main() {
    std::string filedir = "../data/MNIST";
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
    int n = trainingImages.n_cols, d = trainingImages.n_rows, c = round(trainingLabels.max()) + 1;
    images = trainingImages.cols(0, n-1);
    labels = trainingLabels.cols(0, n-1);
    double learningRate=0.01, numIters=500000;
    LogSettings logsettings(2000, 2000);
    SGDProfile sgdProfile;
    Softmax learner(arma::mat(c, d, arma::fill::zeros));
    serialSGD(&learner, images, labels, &sgdProfile, learningRate, numIters, logsettings);
    predictedTrainingLabels = learner.predict(images);
    predictedTestingLabels = learner.predict(testingImages);
    printf("Final training accuracy: %f\n", accuracy(predictedTrainingLabels, labels));
    printf("Final testing accuracy: %f\n", accuracy(predictedTestingLabels, testingLabels));
    return 0;
}
