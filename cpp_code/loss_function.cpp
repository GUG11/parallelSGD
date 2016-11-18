#include "loss_function.h"
#include <cmath>

const double eps = 1e-20;

arma::mat sigmoid(const arma::mat& x) {
    arma::mat s(size(x));
    for (size_t i = 0; i < x.n_elem; i++) {
        s[i] = 1. / (1. + std::exp(x[i]));
    }
    return s;
}
    

Learner::Learner(const arma::mat& w0) {
    w = w0;
    type = "";   
}

arma::mat Learner::getWeight() const {
    return w;
}

void Learner::update(const arma::mat& X, const arma::mat& y, double learningRate) {
    arma::mat grad = computeGrad(X, y);
    w -= learningRate * grad;
}

LeastSquare::LeastSquare(const arma::mat& w0) : Learner(w0) {
    type = "regressor";
}

double LeastSquare::computeLoss(const arma::mat& X/*dxn*/, const arma::mat& y/*1xn*/) {
    arma::mat e/*1xn*/ = predict(X) - y;  
    double loss = arma::dot(e, e);
    return loss;
}

arma::mat LeastSquare::computeGrad(const arma::mat& X, const arma::mat& y) {
    arma::mat grad/*1xd*/ = 2 * (predict(X) - y) * X.t();
    return grad;
}

arma::mat LeastSquare::predict(const arma::mat& X) {
    arma::mat y/*1xn*/ = w/*1xd*/ * X/*dxn*/;
    return y;
}


Logistic::Logistic(const arma::mat& w0) : Learner(w0) {
    type = "classifier";
}

double Logistic::computeLoss(const arma::mat& X, const arma::mat& y){
    double loss = 0;
    arma::mat h = sigmoid(w * X);
    for (size_t i = 0; i < y.n_elem; i++) {
        loss -= y[i] * std::log(std::max(h[i], eps)) + \
                (1 - y[i]) * std::log(std::max(1 - h[i], eps));
    }
    return loss;
}

arma::mat Logistic::computeGrad(const arma::mat& X, const arma::mat& y) {
    arma::mat h = sigmoid(w * X);
    arma::mat grad = (h - y) * X.t();   // 1xn nxd => 1xd
    return grad;
}

arma::mat Logistic::predict(const arma::mat& X) {
    arma::mat h = sigmoid(w * X);
    arma::mat pred(size(h));   
    for (size_t i = 0; i < h.n_elem; i++) pred[i] = h[i] > 0.5 ? 1. : 0.; 
    return pred;
}
