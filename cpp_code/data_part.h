#pragma once

#include <vector>
#include <unordered_map>
#include <armadillo>
#include <lemon/list_graph.h>
#include <utility>

struct Correlation {
    arma::mat corr;
    arma::mat ncc;   // cross correlation
};

struct PartMetrics {
    arma::mat weights;
    arma::imat numEdges;
    arma::mat aveWeights;

    PartMetrics(const arma::mat& edgeMat, const std::vector<std::vector<int>>& dataPartition);    
    void printMetrics();
    void average(double& avg_intra, double& avg_inter);
};

void xcorr(const arma::mat& X, const arma::mat& Y, Correlation& correlation);   // X (d x n); Y (d x m)

/* parition rules */
class Partition {
public:
    virtual void partition(const arma::mat& edgeMat, int P, std::vector<std::vector<int>>& dataPartition) = 0;
};

class RandomPartition : public Partition {
public:
    virtual void partition(const arma::mat& edgeMat, int P, std::vector<std::vector<int>>& dataPartition);
    void partition(int n, int P, std::vector<std::vector<int>>& dataPartition);
    void partition(std::vector<int> indexes, int P, std::vector<std::vector<int>>& dataPartition);
};


class BalancedMinCutParition : public Partition {
protected: 
    int k;
    int n;
    int numLeft;
    std::vector<bool> added;
    std::vector<std::vector<double>> diff;
    std::vector<std::pair<double, int>> minvals;

    void update(const arma::mat& edgeMat, int v, int addSet);
    void computeMinVal();
    int chooseAddSet(const std::vector<std::vector<int>>& dataPartition);
public:
    virtual void partition(const arma::mat& edgeMat, int P, std::vector<std::vector<int>>& dataPartition);
};
