/** \file "Hungarian.h" Header file for the Hungarian (Kuhn-Munkres) assignment algorithm.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

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
