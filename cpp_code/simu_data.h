#pragma once

#include <armadillo>

arma::mat generateGaussian(int n, int d);
arma::mat generateGaussian(int n, int d, int num_groups, double sparse_rate);
