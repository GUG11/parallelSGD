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

arma::mat oneHotEncode(const arma::mat& y, int c) {
    int n = y.n_cols;
    arma::mat OHX(c, n, arma::fill::zeros);
    for (int i = 0; i < n; i++) OHX.at(int(round(y[i])), i) = 1;
    return OHX;
}

arma::mat oneHotDecode(const arma::mat& OHX) {
    int n = OHX.n_cols;
    arma::mat y(1, n);
    for (int i = 0; i < n; i++) {
        arma::uword index;
        OHX.col(i).max(index);
        y[i] = index;
    }
    return y;
}

arma::mat softmax(const arma::mat& z) {  // w(c x d)  X (dxn) => (cxn)
    arma::mat expz = arma::exp(z);   // cxn
    arma::mat sumExpz = arma::sum(expz);   // 1xn
    arma::mat sm(z.n_rows, z.n_cols);
    for (arma::uword i = 0; i < z.n_rows; i++) {
        for (arma::uword j = 0; j < z.n_cols; j++) {
            sm.at(i, j) = expz.at(i, j) / sumExpz[j];
        }
    }
    return sm;   // cxn
}

Learner::Learner(const arma::mat& w0) {
    w = w0;   // 1xd
    type = "";   
}

arma::mat Learner::getWeight() const {
    return w;
}

void Learner::setWeight(const arma::mat& w) {
    this->w = w;
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
    double loss = 0, loss0 = 0;
    arma::mat h = sigmoid(w * X);
    for (size_t i = 0; i < y.n_elem; i++) {
        loss0 = y[i] * std::log(h[i]) + (1 - y[i]) * std::log(1 - h[i]);
        if (!std::isnan(loss0) && !std::isinf(loss0)) loss -= loss0;
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


Softmax::Softmax(const arma::mat& w0) : Learner(w0) {  // c x d
    type = "classifier";
}

arma::mat Softmax::computeProbability(const arma::mat& X) {  // d x n
    arma::mat scores = w * X;
    arma::mat prob = softmax(scores);
    return prob;
}

double Softmax::computeLoss(const arma::mat& X, const arma::mat& y) {
    arma::mat OHX = oneHotEncode(y, w.n_rows);      // c x n
    arma::mat prob = computeProbability(X);  // c x n
    arma::mat entropy(w.n_rows, X.n_cols); 
    double loss = 0, entropy0 = 0;
    for (int i = 0; i < int(entropy.n_elem); i++) {
        entropy0 = OHX[i] * std::log(prob[i]);
        if (std::isnan(entropy0) || std::isinf(entropy0)) continue;
        loss -= entropy0; 
    }
    return loss;
}

arma::mat Softmax::computeGrad(const arma::mat& X, const arma::mat& y) {
    arma::mat OHX = oneHotEncode(y, w.n_rows);     // c x n
    arma::mat prob = computeProbability(X);  // c x n
    arma::mat grad = (prob - OHX) * X.t();
    return grad;
}

arma::mat Softmax::predict(const arma::mat& X) {
    arma::mat prob = computeProbability(X);
    arma::mat y = oneHotDecode(prob);
    return y;
}
