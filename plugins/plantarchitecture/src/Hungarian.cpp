#include "Hungarian.h"

double HungarianAlgorithm::Solve(
    const std::vector<std::vector<double>>& DistMatrix,
    std::vector<int>& Assignment
) {
    int n = static_cast<int>(DistMatrix.size());
    if (n == 0) {
        Assignment.clear();
        return 0.0;
    }
    int m = static_cast<int>(DistMatrix[0].size());
    int dim = std::max(n, m);

    // Build a square cost matrix 'a', filling missing entries with a large cost
    std::vector<std::vector<double>> a(dim, std::vector<double>(dim, std::numeric_limits<double>::max()));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            double v = DistMatrix[i][j];
            a[i][j] = std::isfinite(v) ? v : (std::numeric_limits<double>::max() * 0.5);
        }
    }

    // u, v are the dual potentials; p, way are the matching helpers
    std::vector<double> u(dim+1, 0.0), v2(dim+1, 0.0);
    std::vector<int> p(dim+1, 0), way(dim+1, 0);

    // Main loop: one row per iteration
    for (int i = 1; i <= dim; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(dim+1, std::numeric_limits<double>::max());
        std::vector<bool> used(dim+1, false);

        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = std::numeric_limits<double>::max();

            // Try to improve the matching by looking at all free columns
            for (int j = 1; j <= dim; ++j) {
                if (used[j]) continue;
                double cur = a[i0-1][j-1] - u[i0] - v2[j];
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }

            // Update potentials
            for (int j = 0; j <= dim; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v2[j]    -= delta;
                } else {
                    minv[j] -= delta;
                }
            }

            j0 = j1;
        } while (p[j0] != 0);

        // Now invert the path to grow the matching
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    // Build the result
    Assignment.assign(n, -1);
    for (int j = 1; j <= dim; ++j) {
        if (p[j] <= n && j <= m) {
            Assignment[p[j]-1] = j-1;
        }
    }

    // Compute the actual cost
    double cost = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = Assignment[i];
        if (j >= 0 && j < m) {
            cost += DistMatrix[i][j];
        }
    }
    return cost;
}