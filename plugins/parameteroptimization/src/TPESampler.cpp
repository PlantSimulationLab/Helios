/** \file "TPESampler.cpp" Internal C++ port of the Tree-structured Parzen Estimator sampler from Optuna v5.0.0.

Copyright (C) 2016-2025 Brian Bailey

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

SPDX-License-Identifier: LGPL-2.1-or-later

--------------------------------------------------------------------------------
This file is a C++ port of optuna/samplers/_tpe/sampler.py, parzen_estimator.py,
probability_distributions.py and _truncnorm.py from Optuna v5.0.0
(https://github.com/optuna/optuna), which are distributed under the following license:

MIT License

Copyright (c) 2018 Preferred Networks, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

--------------------------------------------------------------------------------
Optuna's _truncnorm.py, the source of the truncated normal routines in this file,
contains code from the SciPy project, which is distributed under the following license:

Copyright (c) 2001-2002 Enthought, Inc. 2003-2022, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

--------------------------------------------------------------------------------
numpyArgsort() in this file reproduces the generic quicksort of NumPy's np.argsort
(https://numpy.org), which is distributed under the following license:

Copyright (c) 2005-2025, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "TPESampler.h"
#include "global.h"
#include <cstdint>
#include <numeric>

namespace helios_tpe {

    namespace {
        const double pi = 3.14159265358979323846;
        const double sqrt_two = std::sqrt(2.0);
        const double norm_pdf_log_constant = std::log(std::sqrt(2.0 * pi));
        const double ndtri_exp_approximation_constant = std::sqrt(3.0) / pi;
        const double prior_weight = 1.0;
        const double half_step = 0.5; //!< Half of the unit step of INTEGER dimensions

        // np.logaddexp
        double logSum(double log_p, double log_q) {
            if (log_p == log_q) {
                return log_p + std::log(2.0);
            }
            const double difference = log_p - log_q;
            return difference > 0.0 ? log_p + std::log1p(std::exp(-difference)) : log_q + std::log1p(std::exp(difference));
        }

        double logDiff(double log_p, double log_q) {
            return log_p + std::log1p(-std::exp(log_q - log_p));
        }

        // Heapsort of indices[first, first + count), the depth-limit fallback of numpyArgsort()
        void numpyHeapsort(const std::vector<double> &values, std::vector<size_t> &indices, size_t first, size_t count) {
            const auto at = [&indices, first](size_t position) -> size_t & { return indices[first + position - 1]; }; // Positions count from 1
            const auto siftDown = [&values, &at](size_t start, size_t end, size_t moving_index) {
                size_t hole = start;
                size_t child = 2 * start;
                while (child <= end) {
                    if (child < end && values[at(child)] < values[at(child + 1)]) {
                        child++;
                    }
                    if (!(values[moving_index] < values[at(child)])) {
                        break;
                    }
                    at(hole) = at(child);
                    hole = child;
                    child *= 2;
                }
                at(hole) = moving_index;
            };

            for (size_t parent = count / 2; parent > 0; parent--) {
                siftDown(parent, count, at(parent));
            }
            for (size_t end = count; end > 1; end--) {
                const size_t moving_index = at(end);
                at(end) = at(1);
                siftDown(1, end - 1, moving_index);
            }
        }

        double logGaussMassLeftTail(double a, double b) {
            return logDiff(logNdtr(b), logNdtr(a));
        }

        // Standard normal CDF of x * sqrt(2), using erfc to retain accuracy in the tails
        double ndtrFromScaledArgument(double x) {
            const double half_width = 1.0 / sqrt_two;
            if (x < -half_width) {
                return 0.5 * std::erfc(-x);
            }
            if (x < half_width) {
                return 0.5 + 0.5 * std::erf(x);
            }
            return 1.0 - 0.5 * std::erfc(x);
        }
    } // namespace

    UniformStream::UniformStream(unsigned int seed) : engine_(seed) {
    }

    double UniformStream::next() {
        const std::uint32_t upper_bits = static_cast<std::uint32_t>(engine_()) >> 5;
        const std::uint32_t lower_bits = static_cast<std::uint32_t>(engine_()) >> 6;
        return (upper_bits * 67108864.0 + lower_bits) / 9007199254740992.0;
    }

    double ndtr(double a) {
        return 0.5 + 0.5 * std::erf(a / sqrt_two);
    }

    double logNdtr(double a) {
        const double x = a / sqrt_two;
        if (a > 6.0) {
            return -ndtrFromScaledArgument(-x);
        }
        if (a > -20.0) {
            return std::log(ndtrFromScaledArgument(x));
        }

        // Asymptotic expansion of the far left tail
        const double log_lhs = -0.5 * a * a - std::log(-a) - 0.5 * std::log(2.0 * pi);
        double last_total = 0.0;
        double right_hand_side = 1.0;
        double numerator = 1.0;
        double denominator_factor = 1.0;
        const double denominator_constant = 1.0 / (a * a);
        double sign = 1.0;
        int term = 0;
        while (std::abs(last_total - right_hand_side) > std::numeric_limits<double>::epsilon()) {
            term++;
            last_total = right_hand_side;
            sign = -sign;
            denominator_factor *= denominator_constant;
            numerator *= 2.0 * term - 1.0;
            right_hand_side += sign * numerator * denominator_factor;
        }
        return log_lhs + std::log(right_hand_side);
    }

    double logGaussMass(double a, double b) {
        // The right tail is inaccurate, so symmetry is used to work only in the left tail
        if (b <= 0.0) {
            return logGaussMassLeftTail(a, b);
        }
        if (a > 0.0) {
            return logGaussMassLeftTail(-b, -a);
        }
        return std::log1p(-ndtr(a) - ndtr(-b));
    }

    double ndtriExp(double y) {
        // Upstream documents these limits but gives NaN at them. A sample whose quantile rounds to the end of the
        // distribution reaches them, and rounding can also make y slightly positive.
        if (y >= 0.0) {
            return std::numeric_limits<double>::infinity();
        }
        if (std::isinf(y)) {
            return -std::numeric_limits<double>::infinity();
        }

        // Newton's method, flipping the sign of y close to zero for numerical stability
        const bool flipped = y > -1e-2;
        const double z = flipped ? std::log(-std::expm1(y)) : y;
        double x = z < -5.0 ? -std::sqrt(-2.0 * (z + norm_pdf_log_constant)) : -ndtri_exp_approximation_constant * std::log(std::expm1(-z));

        for (int iteration = 0; iteration < 100; iteration++) {
            const double log_ndtr_x = logNdtr(x);
            const double log_norm_pdf_x = -0.5 * x * x - norm_pdf_log_constant;
            const double step = (log_ndtr_x - z) * std::exp(log_ndtr_x - log_norm_pdf_x);
            x -= step;
            if (std::abs(step) < 1e-8 * std::abs(x)) {
                break;
            }
        }
        return flipped ? -x : x;
    }

    double truncnormPpf(double q, double a, double b) {
        if (a == b) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (q == 0.0) {
            return a;
        }
        if (q == 1.0) {
            return b;
        }

        const double log_mass = logGaussMass(a, b);
        double value;
        if (a < 0.0) {
            value = ndtriExp(logSum(logNdtr(a), std::log(q) + log_mass));
        } else {
            // log_ndtr is more stable in the left tail, so the side is flipped for a >= 0
            value = -ndtriExp(logSum(logNdtr(-b), std::log1p(-q) + log_mass));
        }

        // Rounding can overshoot the bounds, and the inverse is infinite at the numerical limits of ndtriExp()
        return std::min(std::max(value, a), b);
    }

    double truncnormLogPdf(double x, double a, double b, double loc, double scale) {
        const double standardized = (x - loc) / scale;
        if (a == b) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (standardized < a || standardized > b) {
            return -std::numeric_limits<double>::infinity();
        }
        return -0.5 * standardized * standardized - norm_pdf_log_constant - logGaussMass(a, b) - std::log(scale);
    }

    std::vector<size_t> numpyArgsort(const std::vector<double> &values) {
        struct Range {
            size_t first; // Inclusive
            size_t last; // Inclusive
            int depth_budget;
        };

        std::vector<size_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), size_t(0));
        if (values.empty()) {
            return indices;
        }

        const auto valueAt = [&values, &indices](size_t position) { return values[indices[position]]; };
        const auto swapIfLess = [&indices, &valueAt](size_t left, size_t right) {
            if (valueAt(left) < valueAt(right)) {
                std::swap(indices[left], indices[right]);
            }
        };

        int size_log2 = 0;
        for (size_t remaining = values.size(); remaining > 1; remaining /= 2) {
            size_log2++;
        }

        std::vector<Range> pending;
        Range range{0, values.size() - 1, 2 * size_log2};
        while (true) {
            if (range.depth_budget < 0) {
                numpyHeapsort(values, indices, range.first, range.last - range.first + 1);
            } else {
                // Partition around a median of three, continuing with the smaller side
                while (range.last - range.first >= 16) {
                    const size_t middle = range.first + (range.last - range.first) / 2;
                    swapIfLess(middle, range.first);
                    swapIfLess(range.last, middle);
                    swapIfLess(middle, range.first);

                    const double pivot = valueAt(middle);
                    size_t left = range.first;
                    size_t right = range.last - 1;
                    std::swap(indices[middle], indices[right]);
                    while (true) {
                        do {
                            left++;
                        } while (valueAt(left) < pivot);
                        do {
                            right--;
                        } while (pivot < valueAt(right));
                        if (left >= right) {
                            break;
                        }
                        std::swap(indices[left], indices[right]);
                    }
                    std::swap(indices[left], indices[range.last - 1]);

                    range.depth_budget--;
                    if (left - range.first < range.last - left) {
                        pending.push_back({left + 1, range.last, range.depth_budget});
                        range.last = left - 1;
                    } else {
                        pending.push_back({range.first, left - 1, range.depth_budget});
                        range.first = left + 1;
                    }
                }

                // Insertion sort of what remains
                for (size_t position = range.first + 1; position <= range.last; position++) {
                    const size_t moving_index = indices[position];
                    const double moving_value = values[moving_index];
                    size_t hole = position;
                    while (hole > range.first && moving_value < valueAt(hole - 1)) {
                        indices[hole] = indices[hole - 1];
                        hole--;
                    }
                    indices[hole] = moving_index;
                }
            }

            if (pending.empty()) {
                return indices;
            }
            range = pending.back();
            pending.pop_back();
        }
    }

    size_t defaultGamma(size_t n_trials) {
        return std::min(static_cast<size_t>(std::ceil(0.1 * static_cast<double>(n_trials))), static_cast<size_t>(25));
    }

    std::vector<double> defaultWeights(size_t n_observations) {
        if (n_observations < 25) {
            return std::vector<double>(n_observations, 1.0);
        }

        std::vector<double> weights(n_observations, 1.0);
        const size_t n_ramp = n_observations - 25;
        const double start = 1.0 / static_cast<double>(n_observations);
        if (n_ramp == 1) {
            weights[0] = start;
        } else if (n_ramp > 1) {
            const double step = (1.0 - start) / static_cast<double>(n_ramp - 1);
            for (size_t i = 0; i < n_ramp - 1; i++) {
                weights[i] = static_cast<double>(i) * step + start;
            }
        }
        return weights;
    }

    ParzenEstimator::ParzenEstimator(const std::vector<std::vector<double>> &observations, const std::vector<SearchDimension> &dimensions) {

        weights_ = defaultWeights(observations.size());
        weights_.push_back(observations.empty() ? 1.0 : prior_weight);
        const double weight_sum = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        log_weights_.reserve(weights_.size());
        weights_cdf_.reserve(weights_.size());
        double cumulative_weight = 0.0;
        for (double &weight: weights_) {
            weight /= weight_sum;
            log_weights_.push_back(std::log(weight));
            cumulative_weight += weight;
            weights_cdf_.push_back(cumulative_weight);
        }
        for (double &cumulative: weights_cdf_) {
            cumulative /= cumulative_weight;
        }

        kernels_.resize(dimensions.size());
        for (size_t d = 0; d < dimensions.size(); d++) {
            kernels_[d].dimension = dimensions[d];
            if (dimensions[d].kind == DimensionKind::CATEGORICAL) {
                fitCategorical(kernels_[d], observations, d);
            } else {
                fitNumerical(kernels_[d], observations, d);
            }
        }
    }

    void ParzenEstimator::fitNumerical(DimensionKernels &kernels, const std::vector<std::vector<double>> &observations, size_t dimension_index) const {

        const bool is_integer = kernels.dimension.kind == DimensionKind::INTEGER;
        const double low = kernels.dimension.low - (is_integer ? half_step : 0.0);
        const double high = kernels.dimension.high + (is_integer ? half_step : 0.0);
        kernels.adapted_low = low;
        kernels.adapted_high = high;

        const size_t n_observations = observations.size();
        kernels.mu.resize(n_observations + 1);
        kernels.sigma.resize(n_observations + 1);
        for (size_t i = 0; i < n_observations; i++) {
            kernels.mu[i] = observations[i][dimension_index];
        }

        // Each kernel's width is the larger gap to its neighbors, the bounds counting as neighbors
        // The order among tied observations decides which kernel gets which width, and so affects the search
        const std::vector<size_t> sorted_indices = numpyArgsort(std::vector<double>(kernels.mu.begin(), kernels.mu.begin() + static_cast<std::ptrdiff_t>(n_observations)));

        std::vector<double> sorted_mus_with_endpoints(n_observations + 2);
        sorted_mus_with_endpoints.front() = low;
        sorted_mus_with_endpoints.back() = high;
        for (size_t i = 0; i < n_observations; i++) {
            sorted_mus_with_endpoints[i + 1] = kernels.mu[sorted_indices[i]];
        }

        const double max_sigma = high - low;
        const double n_kernels = static_cast<double>(n_observations + 1);
        const double min_sigma = (high - low) / std::min(100.0, 1.0 + n_kernels); // "magic clip"
        for (size_t i = 0; i < n_observations; i++) {
            double gap;
            if (n_observations >= 2 && i == 0) {
                gap = sorted_mus_with_endpoints[2] - sorted_mus_with_endpoints[1]; // The bounds do not count for the outermost kernels when there are others
            } else if (n_observations >= 2 && i == n_observations - 1) {
                gap = sorted_mus_with_endpoints[n_observations] - sorted_mus_with_endpoints[n_observations - 1];
            } else {
                gap = std::max(sorted_mus_with_endpoints[i + 1] - sorted_mus_with_endpoints[i], sorted_mus_with_endpoints[i + 2] - sorted_mus_with_endpoints[i + 1]);
            }
            kernels.sigma[sorted_indices[i]] = std::min(std::max(gap, min_sigma), max_sigma);
        }

        // Prior kernel
        kernels.mu[n_observations] = 0.5 * (low + high);
        kernels.sigma[n_observations] = high - low;

        const size_t kernel_count = n_observations + 1;
        kernels.log_sigma.resize(kernel_count);
        kernels.truncation_a.resize(kernel_count);
        kernels.truncation_b.resize(kernel_count);
        kernels.log_mass.resize(kernel_count);
        for (size_t k = 0; k < kernel_count; k++) {
            kernels.log_sigma[k] = std::log(kernels.sigma[k]);
            kernels.truncation_a[k] = (low - kernels.mu[k]) / kernels.sigma[k];
            kernels.truncation_b[k] = (high - kernels.mu[k]) / kernels.sigma[k];
            kernels.log_mass[k] = logGaussMass(kernels.truncation_a[k], kernels.truncation_b[k]);
        }
    }

    void ParzenEstimator::fitCategorical(DimensionKernels &kernels, const std::vector<std::vector<double>> &observations, size_t dimension_index) const {

        const size_t n_choices = kernels.dimension.n_choices;
        if (observations.empty()) {
            kernels.category_weights.assign(1, std::vector<double>(n_choices, 1.0 / static_cast<double>(n_choices)));
        } else {
            const size_t n_kernels = observations.size() + 1;
            kernels.category_weights.assign(n_kernels, std::vector<double>(n_choices, prior_weight / static_cast<double>(n_kernels)));
            for (size_t i = 0; i < observations.size(); i++) {
                kernels.category_weights[i][static_cast<size_t>(observations[i][dimension_index])] += 1.0;
            }
            for (std::vector<double> &row: kernels.category_weights) {
                const double row_sum = std::accumulate(row.begin(), row.end(), 0.0);
                for (double &weight: row) {
                    weight /= row_sum;
                }
            }
        }

        kernels.log_category_weights = kernels.category_weights;
        for (std::vector<double> &row: kernels.log_category_weights) {
            for (double &weight: row) {
                weight = std::log(weight);
            }
        }
    }

    std::vector<std::vector<double>> ParzenEstimator::sample(UniformStream &stream, size_t size) const {

        // The order of the random draws matches upstream: kernel selection, then one quantile per
        // sample for each categorical dimension, then all samples of each numerical dimension in turn.
        std::vector<size_t> active_kernels(size);
        for (size_t &active: active_kernels) {
            const double quantile = stream.next();
            active = static_cast<size_t>(std::upper_bound(weights_cdf_.begin(), weights_cdf_.end(), quantile) - weights_cdf_.begin());
        }

        std::vector<std::vector<double>> samples(size, std::vector<double>(kernels_.size()));

        for (size_t d = 0; d < kernels_.size(); d++) {
            const DimensionKernels &kernels = kernels_[d];
            if (kernels.dimension.kind != DimensionKind::CATEGORICAL) {
                continue;
            }
            for (size_t i = 0; i < size; i++) {
                const double quantile = stream.next();
                const std::vector<double> &row = kernels.category_weights[active_kernels[i]];
                size_t category = 0;
                double cumulative = row[0];
                while (category + 1 < row.size() && cumulative < quantile) {
                    category++;
                    cumulative += row[category];
                }
                samples[i][d] = static_cast<double>(category);
            }
        }

        for (size_t d = 0; d < kernels_.size(); d++) {
            const DimensionKernels &kernels = kernels_[d];
            if (kernels.dimension.kind == DimensionKind::CATEGORICAL) {
                continue;
            }
            for (size_t i = 0; i < size; i++) {
                const size_t k = active_kernels[i];
                const double quantile = stream.next();
                double value = truncnormPpf(quantile, kernels.truncation_a[k], kernels.truncation_b[k]) * kernels.sigma[k] + kernels.mu[k];
                if (kernels.dimension.kind == DimensionKind::INTEGER) {
                    // np.round rounds halves to even, as does nearbyint in the default rounding mode
                    value = kernels.dimension.low + std::nearbyint(value - kernels.dimension.low);
                    value = std::min(std::max(value, kernels.dimension.low), kernels.dimension.high);
                }
                samples[i][d] = value;
            }
        }

        return samples;
    }

    std::vector<double> ParzenEstimator::logPdf(const std::vector<std::vector<double>> &samples) const {

        std::vector<double> log_densities(samples.size());
        const size_t n_kernels = weights_.size();
        std::vector<double> weighted_log_pdf(n_kernels);

        for (size_t i = 0; i < samples.size(); i++) {
            weighted_log_pdf = log_weights_;

            for (size_t d = 0; d < kernels_.size(); d++) {
                const DimensionKernels &kernels = kernels_[d];
                const double x = samples[i][d];

                if (kernels.dimension.kind == DimensionKind::CATEGORICAL) {
                    const size_t category = static_cast<size_t>(x);
                    for (size_t k = 0; k < n_kernels; k++) {
                        weighted_log_pdf[k] += kernels.log_category_weights[k][category];
                    }
                } else if (kernels.dimension.kind == DimensionKind::INTEGER) {
                    for (size_t k = 0; k < n_kernels; k++) {
                        const double left = (x - half_step - kernels.mu[k]) / kernels.sigma[k];
                        const double right = (x + half_step - kernels.mu[k]) / kernels.sigma[k];
                        weighted_log_pdf[k] += logGaussMass(left, right) - kernels.log_mass[k];
                    }
                } else {
                    for (size_t k = 0; k < n_kernels; k++) {
                        const double standardized = (x - kernels.mu[k]) / kernels.sigma[k];
                        if (standardized < kernels.truncation_a[k] || standardized > kernels.truncation_b[k]) {
                            weighted_log_pdf[k] = -std::numeric_limits<double>::infinity();
                        } else {
                            weighted_log_pdf[k] += -0.5 * standardized * standardized - norm_pdf_log_constant - kernels.log_mass[k] - kernels.log_sigma[k];
                        }
                    }
                }
            }

            // log-sum-exp, avoiding (-inf) - (-inf) when the probability is zero
            double max_log_pdf = *std::max_element(weighted_log_pdf.begin(), weighted_log_pdf.end());
            if (std::isinf(max_log_pdf) && max_log_pdf < 0.0) {
                max_log_pdf = 0.0;
            }
            double sum = 0.0;
            for (const double value: weighted_log_pdf) {
                sum += std::exp(value - max_log_pdf);
            }
            log_densities[i] = std::log(sum) + max_log_pdf;
        }

        return log_densities;
    }

    size_t selectBestCandidate(const std::vector<double> &log_likelihood_below, const std::vector<double> &log_likelihood_above) {
        size_t best_candidate = 0;
        double best_acquisition = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < log_likelihood_below.size(); i++) {
            const double acquisition = log_likelihood_below[i] - log_likelihood_above[i];
            if (acquisition > best_acquisition) {
                best_acquisition = acquisition;
                best_candidate = i;
            }
        }
        return best_candidate;
    }

    // Upstream seeds its two random streams identically, which is kept so that a given seed reproduces upstream's trials.
    TPESampler::TPESampler(std::vector<SearchDimension> dimensions, size_t n_startup_trials, size_t n_ei_candidates, unsigned int seed) :
        dimensions_(std::move(dimensions)), n_startup_trials_(n_startup_trials), n_ei_candidates_(n_ei_candidates), model_stream_(seed), random_stream_(seed) {
    }

    std::vector<double> TPESampler::ask() {
        return points_.size() < n_startup_trials_ ? sampleRandom() : sampleModel();
    }

    void TPESampler::tell(const std::vector<double> &point, double value) {
        points_.push_back(point);
        values_.push_back(value);
    }

    std::vector<double> TPESampler::sampleRandom() {
        std::vector<double> point(dimensions_.size());
        for (size_t d = 0; d < dimensions_.size(); d++) {
            const SearchDimension &dimension = dimensions_[d];
            if (dimension.kind == DimensionKind::CATEGORICAL) {
                // One-hot encoding: one uniform per category, of which the highest wins
                size_t best_category = 0;
                double best_score = -1.0;
                for (size_t category = 0; category < dimension.n_choices; category++) {
                    const double score = random_stream_.next();
                    if (score > best_score) {
                        best_score = score;
                        best_category = category;
                    }
                }
                point[d] = static_cast<double>(best_category);
            } else if (dimension.kind == DimensionKind::INTEGER) {
                const double low = dimension.low - half_step;
                const double high = dimension.high + half_step;
                const double value = low + (high - low) * random_stream_.next();
                const double rounded = std::floor(value - dimension.low + 0.5) + dimension.low;
                point[d] = std::min(std::max(rounded, dimension.low), dimension.high);
            } else {
                const double value = dimension.low + (dimension.high - dimension.low) * random_stream_.next();
                const double largest_below_high = std::nextafter(dimension.high, dimension.high - 1.0);
                point[d] = std::min(value, largest_below_high);
            }
        }
        return point;
    }

    std::vector<double> TPESampler::sampleModel() {

        // Split the trials into the best n_below and the rest. Both groups stay in trial order, as
        // defaultWeights() favors recent observations.
        const size_t n_below = std::min(defaultGamma(points_.size()), points_.size());
        std::vector<size_t> ranked(points_.size());
        std::iota(ranked.begin(), ranked.end(), size_t(0));
        std::stable_sort(ranked.begin(), ranked.end(), [this](size_t left, size_t right) { return values_[left] < values_[right]; });
        std::vector<bool> is_below(points_.size(), false);
        for (size_t i = 0; i < n_below; i++) {
            is_below[ranked[i]] = true;
        }

        std::vector<std::vector<double>> below;
        std::vector<std::vector<double>> above;
        below.reserve(n_below);
        above.reserve(points_.size() - n_below);
        for (size_t i = 0; i < points_.size(); i++) {
            (is_below[i] ? below : above).push_back(points_[i]);
        }

        const ParzenEstimator estimator_below(below, dimensions_);
        const ParzenEstimator estimator_above(above, dimensions_);

        const std::vector<std::vector<double>> candidates = estimator_below.sample(model_stream_, n_ei_candidates_);
        const std::vector<double> log_likelihood_below = estimator_below.logPdf(candidates);
        const std::vector<double> log_likelihood_above = estimator_above.logPdf(candidates);

        return candidates[selectBestCandidate(log_likelihood_below, log_likelihood_above)];
    }

} // namespace helios_tpe
