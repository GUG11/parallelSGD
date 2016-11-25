#pragma once
#include <string>
#include "sgd.h"

bool writeSGDProfile(const std::string& filedir, const std::string& filename, const SGDProfile& sgdProfile);

void readMnistImages(const std::string& filepath, arma::mat& images);
void readMnistLabels(const std::string& filepath, arma::mat& labels);
void readMnistImages(const std::string& filedir, const std::string& filename, arma::mat& images);
void readMnistLabels(const std::string& filedir, const std::string& filename, arma::mat& labels);
