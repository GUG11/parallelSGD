#pragma once
#include <string>
#include "sgd.h"

bool writeSGDProfile(const std::string& filedir, const std::string& filename, const SGDProfile& sgdProfile);
bool writeAccuracy(const std::string& filedir, const std::string& filename, std::string& scheme, int training_number, int num_threads, double training_accuracy, double testing_accuracy);

void readMnistImages(const std::string& filepath, arma::mat& images);
void readMnistLabels(const std::string& filepath, arma::mat& labels);
void readMnistImages(const std::string& filedir, const std::string& filename, arma::mat& images);
void readMnistLabels(const std::string& filedir, const std::string& filename, arma::mat& labels);
