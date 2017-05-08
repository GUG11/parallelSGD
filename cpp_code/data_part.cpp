#include "data_part.h"
#include <algorithm>
#include <cassert>


void xcorr(const arma::mat& X, const arma::mat& Y, Correlation& correlation) { // X (d x n); Y (d x m)
    correlation.corr = X.t() * Y;
    arma::mat xnorm(X.n_cols, 1), ynorm(1, Y.n_cols), xyNorm;
    for (int i = 0; i < int(X.n_cols); i++) xnorm[i] = arma::norm(X.col(i));
    for (int j = 0; j < int(Y.n_cols); j++) ynorm[j] = arma::norm(Y.col(j));
    xyNorm = xnorm * ynorm;
    correlation.ncc = correlation.corr / xyNorm;
}

PartMetrics::PartMetrics(const arma::mat& edgeMat, const std::vector<std::vector<int>>& dataPartition) {
        int k = dataPartition.size();
        weights.zeros(k, k);
        numEdges.zeros(k, k);
        aveWeights.zeros(k, k);
        for (int i = 0; i < k; i++) {
            for (int j = 0; j < k; j++) {
                // edges between group i and j
                for (auto& u : dataPartition[i]) {
                    for (auto& v : dataPartition[j]) {
                        if (i == j && u <= v) continue;  // remove duplicates
                        weights.at(i, j) += edgeMat.at(u, v);
                        numEdges.at(i, j)++;
                    }
                }
                aveWeights.at(i, j) = weights.at(i, j) / numEdges.at(i, j);
            }
        }
    }

void PartMetrics::printMetrics() {
    weights.print("Weight Sum\n");
    numEdges.print("Num edges\n");
    aveWeights.print("Average weight\n");
}

void PartMetrics::average(double& avg_intra, double& avg_inter) {
    int k = aveWeights.n_rows;
    avg_intra = 0;
    avg_inter = 0;
    for (int i = 0; i < k; i++) {
        avg_intra += aveWeights.at(i, i);
        for (int j = i + 1; j < k; j++) {
            avg_inter += aveWeights.at(i, j);
        }
    }
    avg_intra /= k;
    avg_inter /= k * (k-1) / 2;
}

void RandomPartition::partition(const arma::mat& edgeMat, int P, std::vector<std::vector<int>>& dataPartition) {
    assert(edgeMat.n_rows==edgeMat.n_cols);
    partition(edgeMat.n_rows, P, dataPartition);
}

void RandomPartition::partition(int n, int P, std::vector<std::vector<int>>& dataPartition) {
    std::vector<int> seq(n);
    for (int i = 0; i < n; i++) seq[i] = i;
    partition(seq, P, dataPartition);
}

void RandomPartition::partition(std::vector<int> indexes, int P, std::vector<std::vector<int>>& dataPartition) {
    int n = indexes.size();
    dataPartition.assign(P, {});
    std::random_shuffle(indexes.begin(), indexes.end());
    for (int i = 0, k = 0; i < n; i++, k = (k+1) % P) dataPartition[k].push_back(indexes[i]);
}


// Greedy Algorithm for k-way Graph paritition Sachin Jain 1998
void BalancedMinCutParition::update(const arma::mat& edgeMat, int v, int addSet) {
    int n = edgeMat.n_rows, k = diff.size();
    added[v] = true;
    for (int p = 0; p < k; p++) {
        for (int i = 0; i < n; i++) {
            if (!added[i]) {
                if (p == addSet)  diff[p][i] -= edgeMat.at(i, v);
                else diff[p][i] += edgeMat.at(i, v);
            }
        }
    }
}

void BalancedMinCutParition::computeMinVal() {
    for (int p = 0; p < k; p++) {
        minvals[p].second = -1;
        for (int i = 0; i < n; i++) {
            if (!added[i] && (minvals[p].second == -1 || diff[p][i] < minvals[p].first)) {
                minvals[p].first = diff[p][i];
                minvals[p].second = i;
            }
        }
    }
}

int BalancedMinCutParition::chooseAddSet(const std::vector<std::vector<int>>& dataPartition) {
    int totalSize = n - numLeft;
    int addSet = -1;
    double mminval = 0;
    for (int p = 0; p < k; p++) {
        if (int(dataPartition[p].size()) * k <= totalSize && (addSet < 0 || minvals[p].first < mminval)) {
            addSet = p;
            mminval = minvals[p].first;
        }
    }
    return addSet;
}

void BalancedMinCutParition::partition(const arma::mat& edgeMat, int P, std::vector<std::vector<int>>& dataPartition) {
    int addSet = 0;
    n = edgeMat.n_rows;
    numLeft = n;
    k = P;
    added.assign(n, false);
    diff.assign(P, std::vector<double>(n, 0));
    minvals.resize(P);
    dataPartition.assign(P, {});
    // set seeds
    update(edgeMat, 0, 0);
    dataPartition[0].push_back(0);
    numLeft -= 1;
    // greedy add
    while (numLeft > 0) {
        computeMinVal();
        addSet = chooseAddSet(dataPartition);
        // add the node
        dataPartition[addSet].push_back(minvals[addSet].second);
        update(edgeMat, minvals[addSet].second, addSet);
        numLeft--;
    }
}
