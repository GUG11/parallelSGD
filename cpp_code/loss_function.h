#pragma once

#include <armadillo>    // column-major (d x N)
#include <string>

arma::mat softmax(const arma::mat& z);
arma::mat oneHotEncode(const arma::mat& y);
arma::mat oneHotDecode(const arma::mat& OHX);

class Learner {
protected: arma::mat w;
    std::string type;   // regressor, classifier
public:
    Learner(const arma::mat& w0);
    virtual double computeLoss(const arma::mat& X, const arma::mat& y) = 0;
    virtual arma::mat computeGrad(const arma::mat& X, const arma::mat& y) = 0;
    void update(const arma::mat& X, arma::mat& y, double learningRate);
    virtual arma::mat predict(const arma::mat& X) = 0;
};
