#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <vector>
#include <limits>
#include <cmath>

class HungarianAlgorithm {
public:
    HungarianAlgorithm() = default;
    ~HungarianAlgorithm() = default;

    double Solve(
        const std::vector<std::vector<double>>& DistMatrix,
        std::vector<int>& Assignment
    );
};

#endif // HUNGARIAN_H