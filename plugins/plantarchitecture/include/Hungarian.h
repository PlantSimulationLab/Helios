/** \file "Hungarian.h" Header file for the Hungarian (Kuhn-Munkres) assignment algorithm.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef HUNGARIAN_H
#define HUNGARIAN_H

#include <cmath>
#include <limits>
#include <vector>

class HungarianAlgorithm {
public:
    HungarianAlgorithm() = default;
    ~HungarianAlgorithm() = default;

    double Solve(const std::vector<std::vector<double>> &DistMatrix, std::vector<int> &Assignment);
};

#endif // HUNGARIAN_H
