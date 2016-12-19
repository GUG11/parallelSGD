#include "simu_data.h"
#include <algorithm>

arma::mat generateGaussian(int n, int d) {
    arma::mat X(d, n, arma::fill::randn);
    return X;
}

/**
 * Divide the n data samples into num_groups, with each group a certain sparse mask
 * */
arma::mat generateGaussian(int n, int d, int num_groups, double sparse_rate) {
    arma::mat X(d, n, arma::fill::randn);
    std::vector<int> group_size(num_groups, n / num_groups), intervals(num_groups + 1);
    int num_ones = std::min(d, std::max(1, int(d * sparse_rate)));
    arma::mat basic_mask = arma::join_cols(arma::mat(num_ones, 1, arma::fill::ones), arma::mat(d - num_ones, 1, arma::fill::zeros)), shuffled_mask = basic_mask;
    for (int i = 0; i < num_groups; i++) group_size[i] += i < (n % num_groups);
    for (int i = 0; i < num_groups; i++) intervals[i+1] = intervals[i] + group_size[i]; 
    for (int i = 0; i < num_groups; i++) {
        shuffled_mask = arma::shuffle(shuffled_mask);
        for (int j = intervals[i]; j < intervals[i+1]; j++) {
            for (int k = 0; k < d; k++) {
                X(k, j) *= shuffled_mask[k];
            }
        }
    }
    return X;
}
