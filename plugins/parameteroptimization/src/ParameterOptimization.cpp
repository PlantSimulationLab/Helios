/** \file "ParameterOptimization.cpp" Primary source file for the Parameter Optimization plug-in.

Copyright (C) 2016-2025 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "ParameterOptimization.h"
#include <numeric>

using namespace std;
using namespace helios;

// Round value to nearest integer if parameter type is INTEGER
inline float roundIfInteger(float val, ParameterType type) {
    return (type == ParameterType::INTEGER) ? std::round(val) : val;
}

// Compute display frequency for progress output based on current iteration and total iterations.
// Early iterations are shown 1:1; longer runs are downsampled to avoid flooding the console.
static int computeDisplayFrequency(int current_iter, int max_iterations) {
    if (current_iter <= 50)
        return 1;
    if (max_iterations > 5000)
        return 100;
    if (max_iterations > 2000)
        return 50;
    if (max_iterations > 500)
        return 10;
    if (max_iterations > 100)
        return 5;
    return 1;
}

// Validate parameters shared across all optimization algorithms.
static void validateParameters(const ParametersToOptimize &parameters) {
    for (const auto &[name, param]: parameters) {
        if (param.min > param.max) {
            helios_runtime_error("ERROR (ParameterOptimization): Parameter '" + name + "' has min (" + std::to_string(param.min) + ") > max (" + std::to_string(param.max) + ").");
        }
        if (std::isnan(param.min) || std::isnan(param.max) || std::isnan(param.value)) {
            helios_runtime_error("ERROR (ParameterOptimization): Parameter '" + name + "' contains NaN values.");
        }
        if (param.type == ParameterType::CATEGORICAL) {
            if (param.categories.empty()) {
                helios_runtime_error("ERROR (ParameterOptimization): CATEGORICAL parameter '" + name + "' has an empty categories list.");
            }
        } else if (param.type == ParameterType::INTEGER) {
            if (std::round(param.min) != param.min || std::round(param.max) != param.max) {
                helios_runtime_error("ERROR (ParameterOptimization): INTEGER parameter '" + name + "' requires integer-valued min and max.");
            }
        }
    }
}

// Compact fitness graph with adaptive downsampling
static void displayOptimizationGraph(const std::vector<float> &fitness_history, int current_iter, int max_iterations, double elapsed_time, bool is_first_call) {
    const int graph_width = 50;
    const int graph_height = 8;

    if (fitness_history.empty())
        return;

    // Auto-scale Y axis
    float min_fitness = *std::min_element(fitness_history.begin(), fitness_history.end());
    float max_fitness = *std::max_element(fitness_history.begin(), fitness_history.end());
    float range = max_fitness - min_fitness;
    if (range < 1e-6f)
        range = 1.0f;

    // Adaptive downsampling: start fine, then coarsen based on progress
    int iters_per_symbol = computeDisplayFrequency(current_iter, max_iterations);

    // Build compact graph
    std::vector<std::string> graph(graph_height, std::string(graph_width, ' '));

    // Collect downsampled data points
    int n_points = fitness_history.size();
    std::vector<float> symbols;
    for (int i = 0; i < n_points; i += iters_per_symbol) {
        // Find best (minimum) fitness in this bucket
        float bucket_min = fitness_history[i];
        for (int j = i; j < std::min(i + iters_per_symbol, n_points); j++) {
            bucket_min = std::min(bucket_min, fitness_history[j]);
        }
        symbols.push_back(bucket_min);
    }

    int num_symbols = symbols.size();
    if (num_symbols == 0)
        return;

    // Normalize y-axis: show relative improvement (0 = worst so far, 1 = best so far)
    // This always uses the full graph height
    float worst_fitness = max_fitness;
    float best_fitness = min_fitness;
    float improvement_range = worst_fitness - best_fitness;
    if (improvement_range < 1e-6f)
        improvement_range = 1.0f;

    // Plot symbols evenly distributed across x-axis
    for (int sym_idx = 0; sym_idx < num_symbols; sym_idx++) {
        // Spread symbols evenly across full width
        int x = (sym_idx * graph_width) / std::max(1, num_symbols - 1);
        if (x >= graph_width)
            x = graph_width - 1;

        // Normalize: 0 = worst (max fitness), 1 = best (min fitness)
        float improvement = (worst_fitness - symbols[sym_idx]) / improvement_range;
        int y = graph_height - 1 - static_cast<int>(improvement * (graph_height - 1));
        y = std::max(0, std::min(graph_height - 1, y));

        graph[y][x] = '*';
    }

    // Calculate ETA
    double avg_time_per_iter = elapsed_time / current_iter;
    double eta = avg_time_per_iter * (max_iterations - current_iter);

    // Print compact graph
    std::cout << "\nIter " << current_iter << "/" << max_iterations << " | Best: " << std::scientific << std::setprecision(2) << min_fitness << " | 1 sym = " << iters_per_symbol << " iters\n";
    std::cout << "Time: " << std::fixed << std::setprecision(1) << elapsed_time << "s"
              << " | ETA: " << std::setprecision(1) << eta << "s"
              << " | Total: " << std::setprecision(1) << (elapsed_time + eta) << "s\n";

    for (int y = 0; y < graph_height; y++) {
        std::cout << graph[y] << "\n";
    }
    std::cout << std::string(graph_width, '-') << "\n";
    std::cout << "0" << std::string(graph_width - std::to_string(current_iter).length(), ' ') << current_iter << "\n";
    std::cout << std::flush;
}

// Hash function for ParametersToOptimize to enable caching
namespace {
    struct ParametersHash {
        std::size_t operator()(const ParametersToOptimize &params) const {
            // Use commutative XOR so iteration order of unordered_map doesn't matter
            std::size_t hash = 0;
            for (const auto &[name, param]: params) {
                std::size_t element_hash = std::hash<std::string>{}(name);
                float hash_val = roundIfInteger(param.value, param.type);
                element_hash ^= std::hash<float>{}(hash_val) + 0x9e3779b9 + (element_hash << 6) + (element_hash >> 2);
                hash ^= element_hash;
            }
            return hash;
        }
    };

    struct ParametersEqual {
        bool operator()(const ParametersToOptimize &a, const ParametersToOptimize &b) const {
            if (a.size() != b.size())
                return false;
            for (const auto &[name, param]: a) {
                auto it = b.find(name);
                if (it == b.end())
                    return false;
                float a_val = roundIfInteger(param.value, param.type);
                float b_val = roundIfInteger(it->second.value, it->second.type);
                if (std::abs(a_val - b_val) > 1e-9f)
                    return false;
            }
            return true;
        }
    };

    // Simple matrix operations for GP
    std::vector<float> matVecMul(const std::vector<std::vector<float>> &mat, const std::vector<float> &vec) {
        std::vector<float> result(mat.size(), 0.0f);
        for (size_t i = 0; i < mat.size(); ++i) {
            for (size_t j = 0; j < vec.size(); ++j) {
                result[i] += mat[i][j] * vec[j];
            }
        }
        return result;
    }

    // Cholesky decomposition (in-place, lower triangular)
    bool choleskyDecompose(std::vector<std::vector<float>> &mat) {
        size_t n = mat.size();
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < j; ++k) {
                    sum += mat[i][k] * mat[j][k];
                }
                if (i == j) {
                    float val = mat[i][i] - sum;
                    if (val <= 0.0f)
                        return false; // Not positive definite
                    mat[i][j] = std::sqrt(val);
                } else {
                    mat[i][j] = (mat[i][j] - sum) / mat[j][j];
                }
            }
        }
        return true;
    }

    // Solve L*y = b where L is lower triangular (forward substitution)
    std::vector<float> solveTriangular(const std::vector<std::vector<float>> &L, const std::vector<float> &b, bool lower = true) {
        size_t n = b.size();
        std::vector<float> y(n);
        if (lower) {
            for (size_t i = 0; i < n; ++i) {
                float sum = 0.0f;
                for (size_t j = 0; j < i; ++j) {
                    sum += L[i][j] * y[j];
                }
                y[i] = (b[i] - sum) / L[i][i];
            }
        } else {
            for (int i = n - 1; i >= 0; --i) {
                float sum = 0.0f;
                for (size_t j = i + 1; j < n; ++j) {
                    sum += L[i][j] * y[j];
                }
                y[i] = (b[i] - sum) / L[i][i];
            }
        }
        return y;
    }

    // Jacobi eigendecomposition for symmetric matrices.
    // On return, eigenvalues contains the eigenvalues and eigenvectors has
    // eigenvectors as columns (eigenvectors[row][col] where col is the eigenvector index).
    void jacobiEigen(std::vector<std::vector<float>> A, std::vector<float> &eigenvalues, std::vector<std::vector<float>> &eigenvectors) {
        size_t n = A.size();
        eigenvectors.assign(n, std::vector<float>(n, 0.0f));
        for (size_t i = 0; i < n; ++i)
            eigenvectors[i][i] = 1.0f;

        const int max_sweeps = 100;
        for (int sweep = 0; sweep < max_sweeps; ++sweep) {
            // Check convergence: sum of squares of off-diagonal elements
            float off_diag = 0.0f;
            for (size_t i = 0; i < n; ++i)
                for (size_t j = i + 1; j < n; ++j)
                    off_diag += A[i][j] * A[i][j];
            if (off_diag < 1e-12f)
                break;

            for (size_t p = 0; p < n; ++p) {
                for (size_t q = p + 1; q < n; ++q) {
                    if (std::abs(A[p][q]) < 1e-12f)
                        continue;

                    float t, c, s;
                    if (std::abs(A[p][p] - A[q][q]) < 1e-12f) {
                        t = 1.0f;
                    } else {
                        float tau = (A[q][q] - A[p][p]) / (2.0f * A[p][q]);
                        t = (tau >= 0 ? 1.0f : -1.0f) / (std::abs(tau) + std::sqrt(1.0f + tau * tau));
                    }
                    c = 1.0f / std::sqrt(1.0f + t * t);
                    s = t * c;

                    // Rotate A
                    float A_pp = A[p][p], A_qq = A[q][q], A_pq = A[p][q];
                    A[p][p] = c * c * A_pp - 2 * s * c * A_pq + s * s * A_qq;
                    A[q][q] = s * s * A_pp + 2 * s * c * A_pq + c * c * A_qq;
                    A[p][q] = 0.0f;
                    A[q][p] = 0.0f;

                    for (size_t i = 0; i < n; ++i) {
                        if (i == p || i == q)
                            continue;
                        float A_ip = A[i][p], A_iq = A[i][q];
                        A[i][p] = c * A_ip - s * A_iq;
                        A[p][i] = A[i][p];
                        A[i][q] = s * A_ip + c * A_iq;
                        A[q][i] = A[i][q];
                    }

                    // Accumulate eigenvectors
                    for (size_t i = 0; i < n; ++i) {
                        float V_ip = eigenvectors[i][p], V_iq = eigenvectors[i][q];
                        eigenvectors[i][p] = c * V_ip - s * V_iq;
                        eigenvectors[i][q] = s * V_ip + c * V_iq;
                    }
                }
            }
        }
        eigenvalues.resize(n);
        for (size_t i = 0; i < n; ++i)
            eigenvalues[i] = A[i][i];
    }

    // PCA state computed from a population, used for BLXPCA crossover.
    struct PCAState {
        std::vector<float> mean;
        std::vector<std::vector<float>> eigenvectors; // columns are eigenvectors
        std::vector<float> eigenvalues;
        std::vector<std::string> param_names; // ordered names of numeric params in PCA
        bool valid = false;

        // Forward: y = diag(1/sqrt(lambda)) * P^T * (x - mean)
        std::vector<float> transform(const ParametersToOptimize &params) const {
            size_t n = param_names.size();
            std::vector<float> centered(n);
            for (size_t i = 0; i < n; ++i)
                centered[i] = params.at(param_names[i]).value - mean[i];

            std::vector<float> result(n, 0.0f);
            for (size_t i = 0; i < n; ++i) {
                float dot = 0.0f;
                for (size_t j = 0; j < n; ++j)
                    dot += eigenvectors[j][i] * centered[j]; // P^T row i = column i of P
                float scale = (eigenvalues[i] > 1e-10f) ? 1.0f / std::sqrt(eigenvalues[i]) : 0.0f;
                result[i] = dot * scale;
            }
            return result;
        }

        // Inverse: x = P * diag(sqrt(lambda)) * y + mean
        std::vector<float> inverseTransform(const std::vector<float> &y) const {
            size_t n = param_names.size();
            std::vector<float> scaled(n);
            for (size_t i = 0; i < n; ++i)
                scaled[i] = y[i] * std::sqrt(std::max(eigenvalues[i], 1e-10f));

            std::vector<float> result(n, 0.0f);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    result[i] += eigenvectors[i][j] * scaled[j];

            for (size_t i = 0; i < n; ++i)
                result[i] += mean[i];
            return result;
        }
    };

    // Compute PCA from the numeric (FLOAT/INTEGER) parameters of a population.
    PCAState computePCA(const std::vector<ParametersToOptimize> &population, const ParametersToOptimize &template_params) {
        PCAState pca;

        // Collect sorted names of numeric params
        for (const auto &[name, param]: template_params) {
            if (param.type != ParameterType::CATEGORICAL)
                pca.param_names.push_back(name);
        }
        std::sort(pca.param_names.begin(), pca.param_names.end());
        size_t n = pca.param_names.size();
        size_t m = population.size();

        if (n == 0 || m < 2)
            return pca;

        // Extract values (m individuals × n dimensions)
        std::vector<std::vector<float>> data(m, std::vector<float>(n));
        for (size_t j = 0; j < m; ++j)
            for (size_t i = 0; i < n; ++i)
                data[j][i] = population[j].at(pca.param_names[i]).value;

        // Mean
        pca.mean.assign(n, 0.0f);
        for (size_t j = 0; j < m; ++j)
            for (size_t i = 0; i < n; ++i)
                pca.mean[i] += data[j][i];
        for (size_t i = 0; i < n; ++i)
            pca.mean[i] /= m;

        // Center
        for (size_t j = 0; j < m; ++j)
            for (size_t i = 0; i < n; ++i)
                data[j][i] -= pca.mean[i];

        // Covariance: S = X^T X / (m-1)
        std::vector<std::vector<float>> cov(n, std::vector<float>(n, 0.0f));
        for (size_t j = 0; j < m; ++j)
            for (size_t i1 = 0; i1 < n; ++i1)
                for (size_t i2 = i1; i2 < n; ++i2)
                    cov[i1][i2] += data[j][i1] * data[j][i2];
        for (size_t i1 = 0; i1 < n; ++i1)
            for (size_t i2 = i1; i2 < n; ++i2) {
                cov[i1][i2] /= (m - 1);
                cov[i2][i1] = cov[i1][i2];
            }

        // Eigendecompose
        jacobiEigen(cov, pca.eigenvalues, pca.eigenvectors);
        pca.valid = true;
        return pca;
    }

    // Gaussian Process with RBF kernel
    class GaussianProcess {
    public:
        GaussianProcess(size_t n_dims, float length_scale_base = 1.0f, float signal_variance = 1.0f, float noise_variance = 1e-4f) : n_dims_(n_dims), signal_variance_(signal_variance), noise_variance_(noise_variance) {
            // Scale lengthscale with sqrt(dimensionality) as per modern BO practice
            // Prior: ℓ ~ LogNormal(μ + log(d)/2, σ)
            // We use the median: ℓ = exp(μ) * sqrt(d), with μ = 0 by default
            length_scale_ = length_scale_base * std::sqrt(static_cast<float>(n_dims));
        }

        // Set hyperparameters (for optimization)
        void setHyperparameters(float length_scale, float signal_variance, float noise_variance) {
            length_scale_ = length_scale;
            signal_variance_ = signal_variance;
            noise_variance_ = noise_variance;
        }

        // Get current hyperparameters
        std::vector<float> getHyperparameters() const {
            return {length_scale_, signal_variance_, noise_variance_};
        }

        // Compute negative log marginal likelihood (for hyperparameter optimization)
        // NLML = 0.5 * y^T * K^{-1} * y + 0.5 * log|K| + (n/2) * log(2π)
        float computeNLML(const std::vector<std::vector<float>> &X, const std::vector<float> &y) const {
            size_t n = X.size();
            if (n == 0)
                return 1e10f;

            // Build kernel matrix K
            std::vector<std::vector<float>> K(n, std::vector<float>(n));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    float k = kernel(X[i], X[j]);
                    K[i][j] = k;
                    K[j][i] = k;
                }
                K[i][i] += noise_variance_; // Add noise to diagonal
            }

            // Cholesky decomposition K = L * L^T
            std::vector<std::vector<float>> L = K;
            if (!choleskyDecompose(L)) {
                // Not positive definite - return high penalty
                return 1e10f;
            }

            // Compute data fit term: 0.5 * y^T * K^{-1} * y = 0.5 * ||L^{-1} * y||²
            std::vector<float> Ly_inv = solveTriangular(L, y, true);
            float data_fit = 0.0f;
            for (float val: Ly_inv) {
                data_fit += val * val;
            }
            data_fit *= 0.5f;

            // Compute complexity penalty: 0.5 * log|K| = sum(log(diag(L)))
            float log_det = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                log_det += std::log(std::max(L[i][i], 1e-10f));
            }

            // Compute normalization constant: (n/2) * log(2π)
            float normalization = 0.5f * n * std::log(2.0f * M_PI);

            return data_fit + log_det + normalization;
        }

        void fit(const std::vector<std::vector<float>> &X, const std::vector<float> &y) {
            X_ = X;
            y_ = y;

            size_t n = X.size();
            K_ = std::vector<std::vector<float>>(n, std::vector<float>(n));

            // Compute kernel matrix
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    float k = kernel(X[i], X[j]);
                    K_[i][j] = k;
                    K_[j][i] = k;
                }
                K_[i][i] += noise_variance_; // Add noise to diagonal
            }

            // Cholesky decomposition
            L_ = K_;
            if (!choleskyDecompose(L_)) {
                // If decomposition fails, add more noise to diagonal
                for (size_t i = 0; i < n; ++i) {
                    K_[i][i] += 1e-3f;
                }
                L_ = K_;
                choleskyDecompose(L_);
            }

            // Compute alpha = K^-1 * y = L^-T * L^-1 * y
            std::vector<float> Ly_inv = solveTriangular(L_, y_, true);

            // Transpose L for upper triangular solve
            std::vector<std::vector<float>> LT(n, std::vector<float>(n, 0.0f));
            for (size_t i = 0; i < n; ++i) {
                for (size_t j = 0; j <= i; ++j) {
                    LT[j][i] = L_[i][j];
                }
            }
            alpha_ = solveTriangular(LT, Ly_inv, false);
        }

        std::pair<float, float> predict(const std::vector<float> &x) const {
            size_t n = X_.size();
            std::vector<float> k_star(n);

            for (size_t i = 0; i < n; ++i) {
                k_star[i] = kernel(x, X_[i]);
            }

            // Mean prediction: k_star^T * alpha
            float mean = 0.0f;
            for (size_t i = 0; i < n; ++i) {
                mean += k_star[i] * alpha_[i];
            }

            // Variance: k(x,x) - k_star^T * K^-1 * k_star
            float k_xx = kernel(x, x);
            std::vector<float> v = solveTriangular(L_, k_star, true);
            float variance = k_xx;
            for (size_t i = 0; i < n; ++i) {
                variance -= v[i] * v[i];
            }
            variance = std::max(0.0f, variance); // Numerical stability

            return {mean, std::sqrt(variance)};
        }

    private:
        float kernel(const std::vector<float> &x1, const std::vector<float> &x2) const {
            float dist_sq = 0.0f;
            for (size_t i = 0; i < x1.size(); ++i) {
                float diff = x1[i] - x2[i];
                dist_sq += diff * diff;
            }
            return signal_variance_ * std::exp(-dist_sq / (2.0f * length_scale_ * length_scale_));
        }

        size_t n_dims_;
        float length_scale_;
        float signal_variance_;
        float noise_variance_;
        std::vector<std::vector<float>> X_;
        std::vector<float> y_;
        std::vector<std::vector<float>> K_;
        std::vector<std::vector<float>> L_;
        std::vector<float> alpha_;
    };
} // namespace

ParametersToOptimize readParametersFromFile(const std::string &filename) {
    ParametersToOptimize params;
    // Resolve file path using project-based resolution
    std::filesystem::path resolved_path = resolveProjectFile(filename);
    std::string resolved_filename = resolved_path.string();

    std::ifstream infile(resolved_filename);
    if (!infile.is_open()) {
        helios_runtime_error("ERROR (ParameterOptimization::readParametersFromFile): Failed to open parameter file '" + filename + "'.");
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty())
            continue;

        std::istringstream ss(line);
        std::string name;
        float value, min, max;

        if (std::getline(ss, name, ',') && ss >> value && ss.ignore(1) && ss >> min && ss.ignore(1) && ss >> max) {
            params[name] = ParameterToOptimize{value, min, max};
        }
    }

    return params;
}

// Helper functions for Bayesian Optimization
namespace {
    // Convert ParametersToOptimize to normalized vector [0,1]
    std::vector<float> paramsToVector(const ParametersToOptimize &params, const std::vector<std::string> &param_names) {
        std::vector<float> vec(param_names.size());
        for (size_t i = 0; i < param_names.size(); ++i) {
            const auto &p = params.at(param_names[i]);
            vec[i] = (p.value - p.min) / (p.max - p.min); // Normalize to [0,1]
        }
        return vec;
    }

    // Convert normalized vector [0,1] to ParametersToOptimize
    ParametersToOptimize vectorToParams(const std::vector<float> &vec, const ParametersToOptimize &template_params, const std::vector<std::string> &param_names) {
        ParametersToOptimize params = template_params;
        for (size_t i = 0; i < param_names.size(); ++i) {
            auto &p = params.at(param_names[i]);
            p.value = p.min + vec[i] * (p.max - p.min);
            p.value = std::min(std::max(p.value, p.min), p.max); // Clamp to bounds
        }
        return params;
    }

    // Latin Hypercube Sampling for initial points
    std::vector<std::vector<float>> latinHypercubeSampling(size_t n_samples, size_t n_dims, std::mt19937 &rng) {
        std::vector<std::vector<float>> samples(n_samples, std::vector<float>(n_dims));
        std::uniform_real_distribution<float> uni01(0.0f, 1.0f);

        for (size_t d = 0; d < n_dims; ++d) {
            std::vector<size_t> indices(n_samples);
            for (size_t i = 0; i < n_samples; ++i)
                indices[i] = i;
            std::shuffle(indices.begin(), indices.end(), rng);

            for (size_t i = 0; i < n_samples; ++i) {
                float segment = 1.0f / n_samples;
                samples[i][d] = (indices[i] + uni01(rng)) * segment;
            }
        }
        return samples;
    }

    // Optimize GP hyperparameters using random search
    // Returns optimal [length_scale, signal_variance, noise_variance]
    std::vector<float> optimizeGPHyperparameters(const std::vector<std::vector<float>> &X, const std::vector<float> &y, size_t n_dims, std::mt19937 &rng, size_t n_trials = 50, const std::vector<float> &warm_start = {}) {

        // Search ranges (log-scale for better coverage)
        std::uniform_real_distribution<float> log_length_scale(std::log(0.01f), std::log(3.0f));
        std::uniform_real_distribution<float> log_signal_var(std::log(0.1f), std::log(10.0f));
        std::uniform_real_distribution<float> log_noise_var(std::log(1e-6f), std::log(0.1f));

        float best_nlml = std::numeric_limits<float>::max();
        std::vector<float> best_params = {
                1.0f * std::sqrt(static_cast<float>(n_dims)), // Default length_scale
                1.0f, // Default signal_variance
                1e-4f // Default noise_variance
        };

        // Try default first
        GaussianProcess gp_test(n_dims);
        float default_nlml = gp_test.computeNLML(X, y);
        best_nlml = default_nlml;

        // Try warm-start hyperparameters from previous iteration (if provided)
        if (warm_start.size() == 3) {
            GaussianProcess gp_warm(n_dims);
            gp_warm.setHyperparameters(warm_start[0], warm_start[1], warm_start[2]);
            float warm_nlml = gp_warm.computeNLML(X, y);
            if (warm_nlml < best_nlml) {
                best_nlml = warm_nlml;
                best_params = warm_start;
            }
        }

        // Random search
        for (size_t trial = 0; trial < n_trials; ++trial) {
            float length_scale = std::exp(log_length_scale(rng)) * std::sqrt(static_cast<float>(n_dims));
            float signal_var = std::exp(log_signal_var(rng));
            float noise_var = std::exp(log_noise_var(rng));

            GaussianProcess gp_trial(n_dims);
            gp_trial.setHyperparameters(length_scale, signal_var, noise_var);
            float nlml = gp_trial.computeNLML(X, y);

            if (nlml < best_nlml) {
                best_nlml = nlml;
                best_params = {length_scale, signal_var, noise_var};
            }
        }

        return best_params;
    }
} // namespace


static ParameterOptimization::Result evaluatePopulation(const std::vector<ParametersToOptimize> &pop, const std::function<float(const ParametersToOptimize &)> &simulation) {

    ParameterOptimization::Result best;
    best.fitness = std::numeric_limits<float>::max();
    for (const auto &ind: pop) {
        float fit = simulation(ind);
        if (fit < best.fitness) {
            best.parameters = ind;
            best.fitness = fit;
        }
    }
    return best;
}

static ParameterOptimization::Result runBayesianOptimization(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters, const BayesianOptimization &bo, bool print_progress,
                                                             const std::string &write_progress_to_file, std::ofstream &outProgress, bool message_flag) {

    std::mt19937 rng(std::random_device{}());
    helios::Timer timer;
    timer.tic();

    validateParameters(parameters);

    // Get parameter names in consistent order
    std::vector<std::string> param_names;
    for (const auto &[name, _]: parameters) {
        param_names.push_back(name);
    }
    std::sort(param_names.begin(), param_names.end());
    size_t n_dims = param_names.size();

    // Determine initial samples
    size_t n_initial = bo.initial_samples > 0 ? bo.initial_samples : std::max<size_t>(2 * n_dims, 10);
    n_initial = std::min(n_initial, bo.max_evaluations);

    // Print starting message
    if (print_progress) {
        std::cout << "\n========================================\n";
        std::cout << "Beginning Bayesian Optimization\n";
        std::cout << "========================================\n";
        std::cout << "Parameters to optimize: " << n_dims << "\n";
        std::cout << "Max iterations: " << bo.max_evaluations << "\n";
        std::cout << "Initial samples: " << n_initial << "\n";
        std::cout << "UCB kappa (exploration): " << bo.ucb_kappa << "\n";
        std::cout << "Acquisition samples: " << bo.acquisition_samples << "\n";
        std::cout << "========================================\n\n";
    }

    // Generate initial samples using Latin Hypercube Sampling
    auto initial_points = latinHypercubeSampling(n_initial, n_dims, rng);

    // Evaluate initial points
    std::vector<std::vector<float>> X;
    std::vector<float> y;
    std::vector<float> fitness_history; // Track all best fitness values for graphing
    ParameterOptimization::Result best;
    best.fitness = std::numeric_limits<float>::max();

    for (size_t i = 0; i < n_initial; ++i) {
        ParametersToOptimize params = vectorToParams(initial_points[i], parameters, param_names);
        float fitness = simulation(params);

        X.push_back(initial_points[i]);
        y.push_back(fitness);

        if (fitness < best.fitness) {
            best.fitness = fitness;
            best.parameters = params;
        }

        fitness_history.push_back(best.fitness);

        int display_freq = computeDisplayFrequency(i + 1, bo.max_evaluations);

        // Display graph at appropriate frequency or on first/last
        if (print_progress && (i == 0 || (i + 1) % display_freq == 0 || i + 1 == n_initial)) {
            double elapsed_time = timer.toc("mute");
            displayOptimizationGraph(fitness_history, i + 1, bo.max_evaluations, elapsed_time, i == 0);
        }
    }

    // Bayesian Optimization loop
    std::vector<float> prev_hyperparams; // Warm-start for hyperparameter optimization
    for (size_t iter = n_initial; iter < bo.max_evaluations; ++iter) {
        // Limit GP training data if needed
        if (X.size() > bo.max_gp_samples) {
            // Keep best samples
            std::vector<size_t> indices(X.size());
            for (size_t i = 0; i < X.size(); ++i)
                indices[i] = i;
            std::partial_sort(indices.begin(), indices.begin() + bo.max_gp_samples, indices.end(), [&](size_t a, size_t b) { return y[a] < y[b]; });

            std::vector<std::vector<float>> X_new(bo.max_gp_samples);
            std::vector<float> y_new(bo.max_gp_samples);
            for (size_t i = 0; i < bo.max_gp_samples; ++i) {
                X_new[i] = X[indices[i]];
                y_new[i] = y[indices[i]];
            }
            X = X_new;
            y = y_new;
        }

        // Recompute normalization statistics from current data each iteration
        float y_mean = 0.0f, y_std = 0.0f;
        for (float val: y)
            y_mean += val;
        y_mean /= y.size();
        for (float val: y)
            y_std += (val - y_mean) * (val - y_mean);
        y_std = std::sqrt(y_std / y.size());
        if (y_std < 1e-6f)
            y_std = 1.0f;

        std::vector<float> y_normalized(y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            y_normalized[i] = (y[i] - y_mean) / y_std;
        }

        // Optimize GP hyperparameters — full search periodically, reuse previous otherwise
        if (prev_hyperparams.empty() || (iter - n_initial) % 20 == 0) {
            prev_hyperparams = optimizeGPHyperparameters(X, y_normalized, n_dims, rng, 50, prev_hyperparams);
        }

        // Fit Gaussian Process with optimized hyperparameters
        GaussianProcess gp(n_dims);
        gp.setHyperparameters(prev_hyperparams[0], prev_hyperparams[1], prev_hyperparams[2]);
        gp.fit(X, y_normalized);

        // Optimize acquisition function (UCB)
        std::uniform_real_distribution<float> uni01(0.0f, 1.0f);
        std::vector<float> best_candidate;
        float best_acquisition = -std::numeric_limits<float>::max();

        for (size_t i = 0; i < bo.acquisition_samples; ++i) {
            std::vector<float> candidate(n_dims);
            for (size_t d = 0; d < n_dims; ++d) {
                candidate[d] = uni01(rng);
            }

            auto [mean, std] = gp.predict(candidate);
            // Convert back to original scale (approximately)
            mean = mean * y_std + y_mean;
            std = std * y_std;

            // UCB for minimization: lower mean is better, higher uncertainty is better
            // acq = -mean + kappa * std (higher is better)
            float acq = -mean + bo.ucb_kappa * std;

            if (acq > best_acquisition) {
                best_acquisition = acq;
                best_candidate = candidate;
            }
        }

        // Evaluate best candidate
        ParametersToOptimize next_params = vectorToParams(best_candidate, parameters, param_names);
        float fitness = simulation(next_params);

        X.push_back(best_candidate);
        y.push_back(fitness);

        if (fitness < best.fitness) {
            best.fitness = fitness;
            best.parameters = next_params;
        }

        fitness_history.push_back(best.fitness);

        int display_freq = computeDisplayFrequency(iter + 1, bo.max_evaluations);

        // Display graph at appropriate frequency or on last iteration
        if (print_progress && ((iter + 1) % display_freq == 0 || iter + 1 == bo.max_evaluations)) {
            double elapsed_time = timer.toc("mute");
            displayOptimizationGraph(fitness_history, iter + 1, bo.max_evaluations, elapsed_time, false);
        }

        // Write to progress file every iteration (not just on improvement)
        if (!write_progress_to_file.empty() && outProgress.is_open()) {
            for (const auto &[name, param]: best.parameters) {
                outProgress << iter + 1 << "," << best.fitness << "," << name << "," << param.value << "," << param.min << "," << param.max << "\n";
            }
        }
    }

    // Print final summary
    if (print_progress) {
        double total_time = timer.toc("mute");
        std::cout << "\n========================================\n";
        std::cout << "Bayesian Optimization Complete\n";
        std::cout << "========================================\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
        std::cout << "Best fitness: " << std::setprecision(6) << best.fitness << "\n";
        std::cout << "\nBest parameters:\n";
        for (const auto &[name, param]: best.parameters) {
            std::cout << "  " << name << ": " << param.value << "\n";
        }
        std::cout << "========================================\n";
    }

    return best;
}

// Genetic Algorithm implementation
static ParameterOptimization::Result runGeneticAlgorithm(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters, const GeneticAlgorithm &ga, bool print_progress,
                                                         const std::string &write_progress_to_file, const std::string &write_result_to_file, const std::string &read_input_from_file, std::ofstream &outProgress, bool message_flag) {

    // Genetic Algorithm
    helios::Timer timer;
    timer.tic();

    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> uni01(0.f, 1.f);

    validateParameters(parameters);

    // Fitness cache to avoid redundant evaluations
    std::unordered_map<ParametersToOptimize, float, ParametersHash, ParametersEqual> fitness_cache;
    size_t cache_hits = 0;
    size_t total_queries = 0;

    // Cached simulation wrapper
    auto cached_simulation = [&](const ParametersToOptimize &params) -> float {
        total_queries++;
        auto it = fitness_cache.find(params);
        if (it != fitness_cache.end()) {
            cache_hits++;
            return it->second;
        }
        float fitness = simulation(params);
        fitness_cache[params] = fitness;
        return fitness;
    };

    size_t num_generations = ga.generations;
    size_t popsize = std::max<size_t>(2, ga.population_size);

    // Extract mutation rate from the mutation variant
    float mutation_rate = std::visit([](const auto &m) { return m.rate; }, ga.mutation);

    // Print starting message
    if (print_progress) {
        std::cout << "\n========================================\n";
        std::cout << "Beginning Genetic Algorithm Optimization\n";
        std::cout << "========================================\n";
        std::cout << "Parameters to optimize: " << parameters.size() << "\n";
        std::cout << "Generations: " << num_generations << "\n";
        std::cout << "Population size: " << popsize << "\n";
        std::cout << "Mutation rate: " << mutation_rate << "\n";
        std::cout << "Crossover rate: " << ga.crossover_rate << "\n";
        std::cout << "Elitism rate: " << ga.elitism_rate << "\n";
        std::cout << "========================================\n\n";
    }
    std::vector<ParametersToOptimize> population(popsize);
    for (auto &ind: population) {
        ind = parameters;
        for (auto &[name, param]: ind) {
            if (param.type == ParameterType::CATEGORICAL) {
                std::uniform_int_distribution<size_t> cat_dist(0, param.categories.size() - 1);
                param.value = param.categories[cat_dist(rng)];
            } else {
                param.value = param.min + (param.max - param.min) * uni01(rng);
                param.value = roundIfInteger(param.value, param.type);
            }
        }
    }

    if (!read_input_from_file.empty()) {
        ParametersToOptimize best_from_file = readParametersFromFile(read_input_from_file);
        population[0] = best_from_file; // seed with known best
    }

    ParameterOptimization::Result best = evaluatePopulation(population, cached_simulation);

    std::vector<double> generation_times;
    helios::Timer gen_timer;

    PCAState pca; // Persists across generations for BLXPCA and HybridMutation

    // Determine PCA update needs
    bool crossover_needs_pca = std::holds_alternative<BLXPCACrossover>(ga.crossover);
    bool mutation_needs_pca = std::holds_alternative<HybridMutation>(ga.mutation);
    bool needs_pca = crossover_needs_pca || mutation_needs_pca;
    size_t pca_interval = SIZE_MAX;
    if (crossover_needs_pca)
        pca_interval = std::min(pca_interval, std::get<BLXPCACrossover>(ga.crossover).pca_update_interval);
    if (mutation_needs_pca)
        pca_interval = std::min(pca_interval, std::get<HybridMutation>(ga.mutation).pca_update_interval);

    for (size_t g = 0; g < num_generations; g++) {
        gen_timer.tic();
        std::vector<ParametersToOptimize> new_pop(popsize);

        // Pre-compute fitness for entire population (pure lookup, no side effects in comparators)
        std::vector<float> pop_fitness(popsize);
        for (size_t i = 0; i < popsize; ++i) {
            pop_fitness[i] = cached_simulation(population[i]);
        }

        // Elitism: keep top individuals
        size_t elitism_count = std::max<size_t>(1, popsize * ga.elitism_rate);
        std::vector<size_t> elite_indices(popsize);
        for (size_t i = 0; i < popsize; ++i)
            elite_indices[i] = i;
        std::partial_sort(elite_indices.begin(), elite_indices.begin() + elitism_count, elite_indices.end(), [&](size_t a, size_t b) { return pop_fitness[a] < pop_fitness[b]; });
        for (size_t e = 0; e < elitism_count; ++e) {
            new_pop[e] = population[elite_indices[e]];
        }

        std::uniform_int_distribution<size_t> dist(0, popsize - 1);

        // Compute PCA periodically if needed
        if (needs_pca && g % pca_interval == 0) {
            pca = computePCA(population, parameters);
        }

        for (size_t i = elitism_count; i < popsize; i++) {
            // Selection, Tournament (uses pre-computed fitness)
            auto tournamentSelect = [&](int k = 3) -> const ParametersToOptimize & {
                size_t bestIdx = dist(rng);
                for (int j = 1; j < k; ++j) {
                    size_t challengerIdx = dist(rng);
                    if (pop_fitness[challengerIdx] < pop_fitness[bestIdx]) {
                        bestIdx = challengerIdx;
                    }
                }
                return population[bestIdx];
            };

            const auto &parent1 = tournamentSelect();
            const auto &parent2 = tournamentSelect();

            ParametersToOptimize child = parameters;

            // Categorical parameters: always per-gene crossover (unaffected by crossover type)
            for (auto &[name, param]: child) {
                if (param.type == ParameterType::CATEGORICAL) {
                    const auto &p1 = parent1.at(name);
                    param.value = p1.value;
                    if (uni01(rng) < ga.crossover_rate) {
                        param.value = (uni01(rng) < 0.5f) ? p1.value : parent2.at(name).value;
                    }
                    if (uni01(rng) < mutation_rate) {
                        std::uniform_int_distribution<size_t> cat_dist(0, param.categories.size() - 1);
                        param.value = param.categories[cat_dist(rng)];
                    }
                }
            }

            // Phase 1: Crossover for numeric parameters
            std::visit(
                    [&](const auto &crossover_op) {
                        using T = std::decay_t<decltype(crossover_op)>;
                        if constexpr (std::is_same_v<T, BLXPCACrossover>) {
                            // BLXPCA: transform parents to PCA space, blend, transform back
                            if (pca.valid) {
                                auto z1 = pca.transform(parent1);
                                auto z2 = pca.transform(parent2);
                                size_t n_pca = pca.param_names.size();

                                std::vector<float> z_child(n_pca);
                                bool do_crossover = uni01(rng) < ga.crossover_rate;
                                for (size_t d = 0; d < n_pca; ++d) {
                                    if (do_crossover) {
                                        float alpha = crossover_op.alpha;
                                        float minval = std::min(z1[d], z2[d]);
                                        float maxval = std::max(z1[d], z2[d]);
                                        float range = maxval - minval;
                                        z_child[d] = minval - alpha * range + uni01(rng) * (1.f + 2.f * alpha) * range;
                                    } else {
                                        z_child[d] = z1[d];
                                    }
                                }

                                // Transform back to original space
                                auto vals = pca.inverseTransform(z_child);
                                for (size_t d = 0; d < n_pca; ++d) {
                                    child.at(pca.param_names[d]).value = vals[d];
                                }
                            } else {
                                // Fallback to BLX-alpha if PCA not valid
                                if (g == 0 && i == elitism_count && print_progress) {
                                    std::cout << "WARNING (ParameterOptimization): PCA computation failed; BLXPCA crossover falling back to standard BLX-alpha.\n";
                                }
                                for (auto &[name, param]: child) {
                                    if (param.type == ParameterType::CATEGORICAL)
                                        continue;
                                    const auto &p1 = parent1.at(name);
                                    const auto &p2 = parent2.at(name);

                                    float val = p1.value;
                                    if (uni01(rng) < ga.crossover_rate) {
                                        float alpha = crossover_op.alpha;
                                        float minval = std::min(p1.value, p2.value);
                                        float maxval = std::max(p1.value, p2.value);
                                        float range = maxval - minval;
                                        val = minval - alpha * range + uni01(rng) * (1.f + 2.f * alpha) * range;
                                    }
                                    param.value = val;
                                }
                            }
                        } else {
                            // BLXAlphaCrossover: per-gene crossover for numeric params
                            for (auto &[name, param]: child) {
                                if (param.type == ParameterType::CATEGORICAL)
                                    continue;
                                const auto &p1 = parent1.at(name);
                                const auto &p2 = parent2.at(name);

                                float val = p1.value;
                                if (uni01(rng) < ga.crossover_rate) {
                                    float alpha = crossover_op.alpha;
                                    float minval = std::min(p1.value, p2.value);
                                    float maxval = std::max(p1.value, p2.value);
                                    float range = maxval - minval;
                                    val = minval - alpha * range + uni01(rng) * (1.f + 2.f * alpha) * range;
                                }
                                param.value = val;
                            }
                        }
                    },
                    ga.crossover);

            // Phase 2: Mutation for numeric parameters
            std::visit(
                    [&](const auto &mutation_op) {
                        using T = std::decay_t<decltype(mutation_op)>;
                        if constexpr (std::is_same_v<T, PerGeneMutation>) {
                            // Per-gene Gaussian mutation
                            for (auto &[name, param]: child) {
                                if (param.type == ParameterType::CATEGORICAL)
                                    continue;
                                if (uni01(rng) < mutation_op.rate) {
                                    std::normal_distribution<float> gauss(0.f, 0.1f * (param.max - param.min));
                                    param.value += gauss(rng);
                                }
                            }
                        } else if constexpr (std::is_same_v<T, IsotropicMutation>) {
                            // Isotropic: single gate, mutate all genes together
                            bool do_mutate = uni01(rng) < mutation_op.rate;
                            if (do_mutate) {
                                for (auto &[name, param]: child) {
                                    if (param.type == ParameterType::CATEGORICAL)
                                        continue;
                                    std::normal_distribution<float> gauss(0.f, 0.1f * (param.max - param.min));
                                    param.value += gauss(rng);
                                }
                            }
                        } else {
                            // HybridMutation
                            bool do_mutate = uni01(rng) < mutation_op.rate;
                            if (do_mutate && pca.valid) {
                                size_t n_pca = pca.param_names.size();
                                // Roll sub-strategy
                                float strategy_roll = uni01(rng);
                                if (strategy_roll < mutation_op.pca_gaussian_prob) {
                                    // PCA-Gaussian
                                    auto z_child = pca.transform(child);
                                    for (size_t d = 0; d < n_pca; ++d) {
                                        std::normal_distribution<float> gauss(0.f, mutation_op.sigma_pca);
                                        z_child[d] += gauss(rng);
                                    }
                                    auto vals = pca.inverseTransform(z_child);
                                    for (size_t d = 0; d < n_pca; ++d) {
                                        child.at(pca.param_names[d]).value = vals[d];
                                    }
                                } else if (strategy_roll < mutation_op.pca_gaussian_prob + mutation_op.pca_cauchy_prob) {
                                    // PCA-Cauchy (heavy-tailed)
                                    auto z_child = pca.transform(child);
                                    std::cauchy_distribution<float> cauchy(0.f, mutation_op.gamma_cauchy);
                                    for (size_t d = 0; d < n_pca; ++d) {
                                        z_child[d] += cauchy(rng);
                                    }
                                    auto vals = pca.inverseTransform(z_child);
                                    for (size_t d = 0; d < n_pca; ++d) {
                                        child.at(pca.param_names[d]).value = vals[d];
                                    }
                                } else {
                                    // Random Direction
                                    std::vector<float> direction(n_pca);
                                    float norm = 0.0f;
                                    for (size_t d = 0; d < n_pca; ++d) {
                                        std::normal_distribution<float> gauss(0.f, 1.f);
                                        direction[d] = gauss(rng);
                                        norm += direction[d] * direction[d];
                                    }
                                    norm = std::sqrt(norm);
                                    for (size_t d = 0; d < n_pca; ++d)
                                        direction[d] /= norm;

                                    // Scale by half-normal step size × avg parameter range
                                    float avg_range = 0.0f;
                                    for (const auto &pname: pca.param_names) {
                                        const auto &p = child.at(pname);
                                        avg_range += (p.max - p.min);
                                    }
                                    avg_range /= pca.param_names.size();

                                    std::normal_distribution<float> half_normal(0.f, 1.f);
                                    float step_size = std::abs(half_normal(rng)) * mutation_op.sigma_random * avg_range;

                                    // Apply
                                    for (size_t d = 0; d < n_pca; ++d) {
                                        child.at(pca.param_names[d]).value += step_size * direction[d];
                                    }
                                }
                            } else if (do_mutate && !pca.valid) {
                                // Fallback to per-gene mutation when PCA not available
                                if (g == 0 && i == elitism_count && print_progress) {
                                    std::cout << "WARNING (ParameterOptimization): PCA computation failed; HybridMutation falling back to per-gene mutation.\n";
                                }
                                for (auto &[name, param]: child) {
                                    if (param.type == ParameterType::CATEGORICAL)
                                        continue;
                                    std::normal_distribution<float> gauss(0.f, 0.1f * (param.max - param.min));
                                    param.value += gauss(rng);
                                }
                            }
                        }
                    },
                    ga.mutation);

            // Phase 3: Clamp and round all numeric params
            for (auto &[name, param]: child) {
                if (param.type != ParameterType::CATEGORICAL) {
                    param.value = std::min(std::max(param.value, param.min), param.max);
                    param.value = roundIfInteger(param.value, param.type);
                }
            }

            new_pop[i] = child;
        }

        population = std::move(new_pop);
        ParameterOptimization::Result generation_best = evaluatePopulation(population, cached_simulation);

        // Record generation time
        double gen_time = gen_timer.toc("mute");
        generation_times.push_back(gen_time);

        if (generation_best.fitness <= best.fitness) {
            best = generation_best;
            if (print_progress) {
                double elapsed_time = timer.toc("mute");

                // Calculate average time per generation and estimate time remaining
                double avg_gen_time = 0.0;
                for (double t: generation_times) {
                    avg_gen_time += t;
                }
                avg_gen_time /= generation_times.size();

                size_t remaining_gens = num_generations - g - 1;
                double estimated_remaining = avg_gen_time * remaining_gens;

                // Print progress bar
                int bar_width = 30;
                float progress = static_cast<float>(g + 1) / num_generations;
                int pos = static_cast<int>(bar_width * progress);

                std::cout << "\rGen [";
                for (int i = 0; i < bar_width; ++i) {
                    if (i < pos)
                        std::cout << "=";
                    else if (i == pos)
                        std::cout << ">";
                    else
                        std::cout << " ";
                }
                std::cout << "] " << static_cast<int>(progress * 100.0) << "% (" << (g + 1) << "/" << num_generations << ") | "
                          << "Best: " << std::fixed << std::setprecision(4) << best.fitness << " | "
                          << "Time: " << std::setprecision(1) << elapsed_time << "s | "
                          << "ETA: " << estimated_remaining << "s";
                std::cout << std::flush;

                // Print parameters on new lines when best improves
                std::cout << "\n";
                for (const auto &[name, param]: best.parameters) {
                    std::cout << "  " << name << ": " << param.value << "\n";
                }
            }
            if (!write_progress_to_file.empty()) {
                if (outProgress.is_open()) {
                    for (const auto &[name, param]: best.parameters) {
                        outProgress << g << "," << best.fitness << "," << name << "," << param.value << "," << param.min << "," << param.max << "\n";
                    }
                    if (message_flag) {
                        std::cout << "Best parameters progress written to: " << write_progress_to_file << "\n";
                    }
                } else {
                    helios_runtime_error("ERROR (ParameterOptimization): Failed to write to progress file '" + write_progress_to_file + "'.");
                }
            }
        }
    }

    if (!write_result_to_file.empty()) {
        std::ofstream out(write_result_to_file);
        if (out.is_open()) {
            out << "parameter,value,min,max\n";
            for (const auto &[name, param]: best.parameters) {
                out << name << "," << param.value << "," << param.min << "," << param.max << "\n";
            }
            out.close();
            if (message_flag) {
                std::cout << "Best parameters written to: " << write_result_to_file << "\n";
            }
        } else {
            helios_runtime_error("ERROR (ParameterOptimization): Failed to write to result file '" + write_result_to_file + "'.");
        }
    }

    if (!write_progress_to_file.empty()) {
        outProgress.close();
    }

    // Print final summary
    if (print_progress) {
        double total_time = timer.toc("mute");
        size_t actual_evaluations = total_queries - cache_hits;
        float cache_hit_rate = total_queries > 0 ? (100.0f * cache_hits / total_queries) : 0.0f;

        std::cout << "\n========================================\n";
        std::cout << "Genetic Algorithm Optimization Complete\n";
        std::cout << "========================================\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
        std::cout << "Best fitness: " << std::setprecision(6) << best.fitness << "\n";
        std::cout << "Fitness evaluations: " << actual_evaluations << " (saved " << cache_hits << " via caching, " << std::setprecision(1) << cache_hit_rate << "% hit rate)\n";
        std::cout << "\nBest parameters:\n";
        for (const auto &[name, param]: best.parameters) {
            std::cout << "  " << name << ": " << param.value << "\n";
        }
        std::cout << "========================================\n";
    }

    return best;
}

// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) implementation
// Based on Hansen & Ostermeier 2001, Hansen 2016 tutorial
static ParameterOptimization::Result runCMAES(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters, const CMAES &cmaes, bool print_progress, const std::string &write_progress_to_file,
                                              const std::string &write_result_to_file, std::ofstream &outProgress, bool message_flag) {

    std::mt19937 rng(std::random_device{}());
    helios::Timer timer;
    timer.tic();

    validateParameters(parameters);

    // Get parameter names in consistent order
    std::vector<std::string> param_names;
    for (const auto &[name, _]: parameters) {
        param_names.push_back(name);
    }
    std::sort(param_names.begin(), param_names.end());
    size_t n = param_names.size();

    // CMA-ES Strategy Parameters (following Hansen)
    size_t lambda = cmaes.lambda > 0 ? cmaes.lambda : (4 + static_cast<size_t>(floor(3.0 * log(static_cast<double>(n)))));
    size_t mu = lambda / 2; // Number of parents

    // Recombination weights
    std::vector<float> weights(mu);
    float sum_weights = 0.0f;
    for (size_t i = 0; i < mu; ++i) {
        weights[i] = log(mu + 0.5f) - log(i + 1.0f);
        sum_weights += weights[i];
    }
    for (size_t i = 0; i < mu; ++i) {
        weights[i] /= sum_weights;
    }

    float mu_eff = 1.0f / std::inner_product(weights.begin(), weights.end(), weights.begin(), 0.0f);

    // Adaptation parameters
    float c_c = (4.0f + mu_eff / n) / (n + 4.0f + 2.0f * mu_eff / n);
    float c_s = (mu_eff + 2.0f) / (n + mu_eff + 5.0f);
    float c_1 = 2.0f / ((n + 1.3f) * (n + 1.3f) + mu_eff);
    float c_mu = std::min(1.0f - c_1, 2.0f * (mu_eff - 2.0f + 1.0f / mu_eff) / ((n + 2.0f) * (n + 2.0f) + mu_eff));
    float d_s = 1.0f + 2.0f * std::max(0.0f, sqrt((mu_eff - 1.0f) / (n + 1.0f)) - 1.0f) + c_s;
    float chi_n = sqrt(n) * (1.0f - 1.0f / (4.0f * n) + 1.0f / (21.0f * n * n));

    // Initialize state
    std::vector<float> mean(n, 0.5f); // Start at center [0,1]
    float sigma = cmaes.sigma;
    std::vector<float> p_c(n, 0.0f);
    std::vector<float> p_s(n, 0.0f);
    std::vector<std::vector<float>> C(n, std::vector<float>(n, 0.0f));
    for (size_t i = 0; i < n; ++i)
        C[i][i] = 1.0f;

    // Print starting message
    if (print_progress) {
        std::cout << "\n========================================\n";
        std::cout << "Beginning CMA-ES Optimization\n";
        std::cout << "========================================\n";
        std::cout << "Parameters to optimize: " << n << "\n";
        std::cout << "Max iterations: " << cmaes.max_evaluations << "\n";
        std::cout << "Population size (λ): " << lambda << "\n";
        std::cout << "Parents (μ): " << mu << "\n";
        std::cout << "Initial step-size (σ): " << sigma << "\n";
        std::cout << "========================================\n\n";
    }

    ParameterOptimization::Result best;
    best.fitness = std::numeric_limits<float>::max();
    std::vector<float> fitness_history;
    std::normal_distribution<float> norm(0.0f, 1.0f);

    size_t gen = 0;
    size_t total_evals = 0;

    while (total_evals < cmaes.max_evaluations) {
        // Sample lambda offspring
        std::vector<std::vector<float>> offspring(lambda, std::vector<float>(n));
        std::vector<std::vector<float>> y_samples(lambda, std::vector<float>(n)); // Store y = L*z
        std::vector<float> fitnesses(lambda);

        // Compute C^(1/2) using Cholesky decomposition: C = L * L^T, so C^(1/2) = L
        std::vector<std::vector<float>> L = C;
        bool chol_success = choleskyDecompose(L);

        if (!chol_success) {
            // Add regularization to diagonal if Cholesky fails
            for (size_t i = 0; i < n; ++i) {
                C[i][i] += 1e-6f;
            }
            L = C;
            choleskyDecompose(L);
        }

        for (size_t k = 0; k < lambda && total_evals < cmaes.max_evaluations; ++k) {
            // Sample z ~ N(0,I)
            std::vector<float> z(n);
            for (size_t i = 0; i < n; ++i) {
                z[i] = norm(rng);
            }

            // y = C^(1/2) * z = L * z
            for (size_t i = 0; i < n; ++i) {
                y_samples[k][i] = 0.0f;
                for (size_t j = 0; j <= i; ++j) {
                    y_samples[k][i] += L[i][j] * z[j];
                }
            }

            // x = mean + sigma * y
            for (size_t i = 0; i < n; ++i) {
                offspring[k][i] = mean[i] + sigma * y_samples[k][i];
                offspring[k][i] = std::max(0.0f, std::min(1.0f, offspring[k][i]));
            }

            // Evaluate
            ParametersToOptimize params = vectorToParams(offspring[k], parameters, param_names);
            fitnesses[k] = simulation(params);
            total_evals++;

            if (fitnesses[k] < best.fitness) {
                best.fitness = fitnesses[k];
                best.parameters = params;
            }
        }

        fitness_history.push_back(best.fitness);

        // Display graph
        int display_freq = computeDisplayFrequency(gen + 1, cmaes.max_evaluations);
        if (print_progress && (gen == 0 || (gen + 1) % display_freq == 0 || total_evals >= cmaes.max_evaluations)) {
            double elapsed_time = timer.toc("mute");
            displayOptimizationGraph(fitness_history, total_evals, cmaes.max_evaluations, elapsed_time, gen == 0);
        }

        // Write progress
        if (!write_progress_to_file.empty() && outProgress.is_open()) {
            for (const auto &[name, param]: best.parameters) {
                outProgress << gen << "," << best.fitness << "," << name << "," << param.value << "," << param.min << "," << param.max << "\n";
            }
        }

        if (total_evals >= cmaes.max_evaluations)
            break;

        // Sort by fitness
        std::vector<size_t> indices(lambda);
        for (size_t i = 0; i < lambda; ++i)
            indices[i] = i;
        std::partial_sort(indices.begin(), indices.begin() + mu, indices.end(), [&](size_t a, size_t b) { return fitnesses[a] < fitnesses[b]; });

        // Compute effective displacement vectors from clamped offspring
        // This ensures consistency: the covariance adapts to the actual evaluated positions
        std::vector<std::vector<float>> y_effective(lambda, std::vector<float>(n));
        std::vector<float> old_mean = mean;
        for (size_t k = 0; k < lambda; ++k) {
            for (size_t i = 0; i < n; ++i) {
                y_effective[k][i] = (offspring[k][i] - old_mean[i]) / sigma;
            }
        }

        // Update mean using clamped offspring (the positions we actually evaluated)
        for (size_t i = 0; i < n; ++i) {
            mean[i] = 0.0f;
            for (size_t k = 0; k < mu; ++k) {
                mean[i] += weights[k] * offspring[indices[k]][i];
            }
        }

        // Compute mean step in search space: (mean_new - mean_old) / sigma
        std::vector<float> mean_step(n);
        for (size_t i = 0; i < n; ++i) {
            mean_step[i] = (mean[i] - old_mean[i]) / sigma;
        }

        // Update evolution path for sigma (p_s)
        // C = L * L^T, so L^(-1) applied to the mean step gives the isotropic step
        std::vector<float> p_s_step = solveTriangular(L, mean_step, true);

        float coeff_s = sqrt(c_s * (2.0f - c_s) * mu_eff);
        for (size_t i = 0; i < n; ++i) {
            p_s[i] = (1.0f - c_s) * p_s[i] + coeff_s * p_s_step[i];
        }

        float norm_p_s = sqrt(std::inner_product(p_s.begin(), p_s.end(), p_s.begin(), 0.0f));

        // Update evolution path for covariance (p_c)
        float h_s = (norm_p_s / sqrt(1.0f - pow(1.0f - c_s, 2.0f * (gen + 1.0f)))) < (1.4f + 2.0f / (n + 1.0f)) * chi_n ? 1.0f : 0.0f;

        float coeff_c = sqrt(c_c * (2.0f - c_c) * mu_eff);
        for (size_t i = 0; i < n; ++i) {
            p_c[i] = (1.0f - c_c) * p_c[i] + h_s * coeff_c * mean_step[i];
        }

        // Update C (rank-one + rank-mu)
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                float val = (1.0f - c_1 - c_mu) * C[i][j];

                // Rank-one
                val += c_1 * p_c[i] * p_c[j];

                // Rank-mu: use effective displacement vectors from clamped offspring
                for (size_t k = 0; k < mu; ++k) {
                    val += c_mu * weights[k] * y_effective[indices[k]][i] * y_effective[indices[k]][j];
                }

                C[i][j] = val;
                C[j][i] = val;
            }
        }

        // Update sigma
        sigma *= exp((c_s / d_s) * (norm_p_s / chi_n - 1.0f));
        sigma = std::min(10.0f, std::max(1e-10f, sigma)); // Bounds

        gen++;
    }

    // Write final result to file if requested
    if (!write_result_to_file.empty()) {
        std::ofstream out(write_result_to_file);
        if (out.is_open()) {
            out << "parameter,value,min,max\n";
            for (const auto &[name, param]: best.parameters) {
                out << name << "," << param.value << "," << param.min << "," << param.max << "\n";
            }
            out.close();
            if (message_flag) {
                std::cout << "Best parameters written to: " << write_result_to_file << "\n";
            }
        } else {
            helios_runtime_error("ERROR (ParameterOptimization): Failed to write to result file '" + write_result_to_file + "'.");
        }
    }

    // Print final summary
    if (print_progress) {
        double total_time = timer.toc("mute");
        std::cout << "\n========================================\n";
        std::cout << "CMA-ES Optimization Complete\n";
        std::cout << "========================================\n";
        std::cout << "Total time: " << std::fixed << std::setprecision(2) << total_time << " s\n";
        std::cout << "Generations: " << gen << "\n";
        std::cout << "Evaluations: " << total_evals << "\n";
        std::cout << "Best fitness: " << std::setprecision(6) << best.fitness << "\n";
        std::cout << "Final step-size (σ): " << std::setprecision(4) << sigma << "\n";
        std::cout << "\nBest parameters:\n";
        for (const auto &[name, param]: best.parameters) {
            std::cout << "  " << name << ": " << param.value << "\n";
        }
        std::cout << "========================================\n";
    }

    return best;
}

// Preset factory methods for GeneticAlgorithm
GeneticAlgorithm GeneticAlgorithm::explore() {
    GeneticAlgorithm ga;
    ga.generations = 200;
    ga.population_size = 40;
    ga.crossover_rate = 0.9f;
    ga.elitism_rate = 0.02f;
    ga.crossover = BLXPCACrossover{};
    HybridMutation hm;
    hm.rate = 0.20f;
    hm.pca_gaussian_prob = 0.50f;
    hm.pca_cauchy_prob = 0.35f;
    ga.mutation = hm;
    return ga;
}

GeneticAlgorithm GeneticAlgorithm::exploit() {
    GeneticAlgorithm ga;
    ga.generations = 100;
    ga.population_size = 30;
    ga.crossover_rate = 0.7f;
    ga.elitism_rate = 0.15f;
    ga.crossover = BLXPCACrossover{};
    PerGeneMutation pm;
    pm.rate = 0.05f;
    ga.mutation = pm;
    return ga;
}

// Preset factory methods for BayesianOptimization
BayesianOptimization BayesianOptimization::explore() {
    BayesianOptimization bo;
    bo.max_evaluations = 200;
    bo.ucb_kappa = 4.0f;
    bo.acquisition_samples = 2000;
    return bo;
}

BayesianOptimization BayesianOptimization::exploit() {
    BayesianOptimization bo;
    bo.max_evaluations = 100;
    bo.ucb_kappa = 0.5f;
    bo.acquisition_samples = 2000;
    return bo;
}

// Preset factory methods for CMAES
CMAES CMAES::explore() {
    CMAES cmaes;
    cmaes.max_evaluations = 300;
    cmaes.sigma = 0.5f;
    return cmaes;
}

CMAES CMAES::exploit() {
    CMAES cmaes;
    cmaes.max_evaluations = 100;
    cmaes.sigma = 0.1f;
    return cmaes;
}

// setAlgorithm implementations
void ParameterOptimization::setAlgorithm(const GeneticAlgorithm &algorithm) {
    algorithm_ = algorithm;
}

void ParameterOptimization::setAlgorithm(const BayesianOptimization &algorithm) {
    algorithm_ = algorithm;
}

void ParameterOptimization::setAlgorithm(const CMAES &algorithm) {
    algorithm_ = algorithm;
}

// Main run dispatcher
ParameterOptimization::Result ParameterOptimization::run(std::function<float(const ParametersToOptimize &)> simulation, const ParametersToOptimize &parameters) {

    // Validate output file paths before starting optimization
    if (!write_result_to_file.empty()) {
        std::string result_path = write_result_to_file;
        if (!validateOutputPath(result_path, {".csv", ".txt"})) {
            helios_runtime_error("ERROR (ParameterOptimization::run): Invalid output path for write_result_to_file '" + write_result_to_file + "'. Check that the directory exists and you have write permission.");
        }
    }
    if (!write_progress_to_file.empty()) {
        std::string progress_path = write_progress_to_file;
        if (!validateOutputPath(progress_path, {".csv", ".txt"})) {
            helios_runtime_error("ERROR (ParameterOptimization::run): Invalid output path for write_progress_to_file '" + write_progress_to_file + "'. Check that the directory exists and you have write permission.");
        }
    }

    std::ofstream outProgress;
    if (!write_progress_to_file.empty()) {
        outProgress = std::ofstream(write_progress_to_file);
        outProgress.clear();
        outProgress << "generation,fitness,parameter,value,min,max\n";
    }

    return std::visit(
            [&](const auto &algo) -> Result {
                using T = std::decay_t<decltype(algo)>;
                if constexpr (std::is_same_v<T, BayesianOptimization>) {
                    return runBayesianOptimization(simulation, parameters, algo, print_progress, write_progress_to_file, outProgress, message_flag);
                } else if constexpr (std::is_same_v<T, CMAES>) {
                    return runCMAES(simulation, parameters, algo, print_progress, write_progress_to_file, write_result_to_file, outProgress, message_flag);
                } else {
                    return runGeneticAlgorithm(simulation, parameters, algo, print_progress, write_progress_to_file, write_result_to_file, read_input_from_file, outProgress, message_flag);
                }
            },
            algorithm_);
}

int ParameterOptimization::selfTestVisualized(const std::string &output_dir) {
    std::cout << "Running visualization tests for optimization algorithms..." << std::endl;
    std::cout << "Output directory: " << output_dir << std::endl << std::endl;

    // Test functions
    struct TestFunction {
        std::string name;
        std::function<float(float, float)> func;
        std::pair<float, float> bounds;
        std::pair<float, float> optimum;
    };

    std::vector<TestFunction> test_functions = {// Sphere function: f(x,y) = x^2 + y^2, minimum at (0,0)
                                                {"sphere", [](float x, float y) { return x * x + y * y; }, {-5.0f, 5.0f}, {0.0f, 0.0f}},

                                                // Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, minimum at (1,1)
                                                {"rosenbrock",
                                                 [](float x, float y) {
                                                     float term1 = (1.0f - x);
                                                     float term2 = (y - x * x);
                                                     return term1 * term1 + 100.0f * term2 * term2;
                                                 },
                                                 {-2.0f, 2.0f},
                                                 {1.0f, 1.0f}}};

    // Algorithm configurations
    struct AlgoConfig {
        std::string name;
        std::function<void(ParameterOptimization &, const std::variant<GeneticAlgorithm, BayesianOptimization, CMAES> &)> setup;
        std::variant<GeneticAlgorithm, BayesianOptimization, CMAES> algo_variant;
    };

    std::vector<AlgoConfig> algorithms;

    // GA configuration
    {
        GeneticAlgorithm ga;
        ga.generations = 200;
        ga.population_size = 30;
        ga.crossover_rate = 0.8f;
        ga.crossover = BLXPCACrossover{};
        IsotropicMutation im;
        ga.mutation = im;

        algorithms.push_back({"GA", [](ParameterOptimization &opt, const auto &variant) { opt.setAlgorithm(std::get<GeneticAlgorithm>(variant)); }, ga});
    }

    // BO configuration
    {
        BayesianOptimization bo;
        bo.max_evaluations = 300;
        bo.initial_samples = 15;
        bo.ucb_kappa = 2.5f;
        bo.acquisition_samples = 200;

        algorithms.push_back({"BO", [](ParameterOptimization &opt, const auto &variant) { opt.setAlgorithm(std::get<BayesianOptimization>(variant)); }, bo});
    }

    // CMA-ES configuration
    {
        CMAES cmaes;
        cmaes.max_evaluations = 3000;
        cmaes.sigma = 0.5f;

        algorithms.push_back({"CMAES", [](ParameterOptimization &opt, const auto &variant) { opt.setAlgorithm(std::get<CMAES>(variant)); }, cmaes});
    }

    // Run tests writing both population evaluations and best-per-generation
    for (const auto &test_func: test_functions) {
        std::cout << "Testing function: " << test_func.name << std::endl;

        for (const auto &algo_config: algorithms) {
            std::cout << "  Running " << algo_config.name << "..." << std::endl;

            // Setup parameters
            ParametersToOptimize params = {{"x", {0.0f, test_func.bounds.first, test_func.bounds.second}}, {"y", {0.0f, test_func.bounds.first, test_func.bounds.second}}};

            // Output filenames
            std::string pop_filename = output_dir + "/" + algo_config.name + "_" + test_func.name + "_population.csv";
            std::string best_filename = output_dir + "/" + algo_config.name + "_" + test_func.name + "_best.csv";

            // Track all evaluations through fitness function wrapper (for population scatter)
            std::vector<std::tuple<int, float, float, float>> eval_history;
            int eval_count = 0;

            auto fitness_func = [&](const ParametersToOptimize &p) {
                float x = p.at("x").value;
                float y = p.at("y").value;
                float fitness = test_func.func(x, y);
                eval_history.push_back({eval_count++, x, y, fitness});
                return fitness;
            };

            // Run with write_progress_to_file for correct best-per-generation tracking
            ParameterOptimization optimizer;
            optimizer.print_progress = false;
            optimizer.write_progress_to_file = best_filename;
            algo_config.setup(optimizer, algo_config.algo_variant);
            auto result = optimizer.run(fitness_func, params);

            // Write population history (all evaluations) to file
            size_t pop_size = 1;
            if (auto *ga_ptr = std::get_if<GeneticAlgorithm>(&algo_config.algo_variant)) {
                pop_size = ga_ptr->population_size;
            } else if (std::holds_alternative<CMAES>(algo_config.algo_variant)) {
                pop_size = 4 + static_cast<size_t>(floor(3.0 * log(2.0)));
            }

            std::ofstream outfile(pop_filename);
            outfile << "iteration,individual_id,x,y,fitness\n";
            for (size_t i = 0; i < eval_history.size(); ++i) {
                auto &[id, x, y, fit] = eval_history[i];
                int iteration = i / pop_size;
                int individual = i % pop_size;
                outfile << iteration << "," << individual << "," << x << "," << y << "," << fit << "\n";
            }
            outfile.close();

            std::cout << "    Best fitness: " << result.fitness << " at (" << result.parameters.at("x").value << ", " << result.parameters.at("y").value << ")" << std::endl;
            std::cout << "    Written " << eval_history.size() << " evaluations to " << pop_filename << std::endl;
            std::cout << "    Written best-per-generation to " << best_filename << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "Visualization test complete. Output files written to: " << output_dir << std::endl;
    std::cout << "Population CSV: iteration,individual_id,x,y,fitness" << std::endl;
    std::cout << "Best CSV: generation,fitness,parameter,value,min,max" << std::endl;

    return 0;
}
