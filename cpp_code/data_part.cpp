#include "data_part.h"
#include <algorithm>

void randomPartition(int n, int P, std::vector<std::vector<int>>& dataPartition) {
    std::vector<int> seq(n);
    dataPartition.assign(P, {});
    for (int i = 0; i < n; i++) seq[i] = i;
    std::random_shuffle(seq.begin(), seq.end());      
    for (int i = 0, k = 0; i < n; i++, k = (k+1) % P) dataPartition[k].push_back(seq[i]);
}
