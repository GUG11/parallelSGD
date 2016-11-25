#include "metrics.h"
#include <cmath>

double euclideanDistance(arma::mat& pred, arma::mat& y) {
    arma::mat e/*1xn*/ = pred - y;  
    double dist = arma::dot(e, e);
    return dist;
}

double accuracy(arma::mat& pred, arma::mat& y) {
    int count = 0, n = pred.n_elem;
    double acc = 0;
    for (int i = 0; i < n; i++) {
        count += int(round(pred[i])) == int(round(y[i]));
    }
    acc = double(count) / n;
    return acc;
}
