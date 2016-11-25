/**
 * Evaluation the prediction from a learner
 * */
#pragma once

#include <armadillo>

double euclideanDistance(arma::mat& pred, arma::mat& y);    // 1 x n
double accuracy(arma::mat& pred, arma::mat& y);
