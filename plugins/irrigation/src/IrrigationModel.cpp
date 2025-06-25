/** \file "IrrigationModel.cpp" Primary source file for irrigation plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "IrrigationModel.h"

using namespace helios;

IrrigationModel::IrrigationModel(Context *context) { this->context = context; }

// ─────────────── Local helpers ──────────────────────────────────────────── //
namespace {
    constexpr double NODE_TOL = 1.0e-6; // coordinate merge tolerance (m)

    void trim(std::string &s) {
        auto is_ws = [](unsigned char c) { return std::isspace(c); };
        while (!s.empty() && is_ws(s.back()))
            s.pop_back();
        auto it = std::find_if_not(s.begin(), s.end(), is_ws);
        s.erase(s.begin(), it);
    }

    std::map<int, std::string> pairsToMap(const std::vector<std::pair<int, std::string>> &pairs) {
        std::map<int, std::string> m;
        for (const auto &p: pairs)
            m[p.first] = p.second;
        return m;
    }
} // anonymous namespace

// ─────────────── Node management ───────────────────────────────────────── //
int IrrigationModel::getOrCreateNode(double x, double y) {
    for (std::size_t i = 0; i < nodes.size(); ++i)
        if (std::fabs(nodes[i].x - x) < NODE_TOL && std::fabs(nodes[i].y - y) < NODE_TOL)
            return int(i);

    nodes.push_back({x, y, 0.0, false});
    return int(nodes.size() - 1);
}

// ─────────────── Pipe helper ───────────────────────────────────────────── //
int IrrigationModel::addPipe(int n1, int n2, double L, double d, double kminor) {
    if (n1 == n2)
        return 1; // degenerate
    pipes.push_back({n1, n2, L, d, kminor});
    return 0;
}

// ─────────────── Entity: LINE ──────────────────────────────────────────── //
int IrrigationModel::parseLineEntity(const std::map<int, std::string> &ent) {
    if (!(ent.count(10) && ent.count(20) && ent.count(11) && ent.count(21)))
        return 1;

    double x1 = std::stod(ent.at(10)), y1 = std::stod(ent.at(20));
    double x2 = std::stod(ent.at(11)), y2 = std::stod(ent.at(21));
    double d = ent.count(40) ? std::stod(ent.at(40)) : 0.05;
    if (d > 1.0)
        d /= 1000.0;

    int n1 = getOrCreateNode(x1, y1);
    int n2 = getOrCreateNode(x2, y2);
    return addPipe(n1, n2, std::hypot(x2 - x1, y2 - y1), d);
}

// ─────────────── Entity: LWPOLYLINE ─────────────────────────────────────── //
int IrrigationModel::parseLWPolylineEntity(const std::vector<std::pair<int, std::string>> &ent) {
    std::vector<std::pair<double, double>> verts;
    double curX = 0.0, diameter = 0.05;
    bool haveX = false, closed = false;

    for (const auto &kv: ent) {
        int code = kv.first;
        if (code == 10) {
            curX = std::stod(kv.second);
            haveX = true;
        } else if (code == 20) {
            if (haveX) {
                verts.emplace_back(curX, std::stod(kv.second));
                haveX = false;
            }
        } else if (code == 40)
            diameter = std::stod(kv.second);
        else if (code == 70)
            closed = (std::stoi(kv.second) & 1) != 0;
    }
    if (verts.size() < 2)
        return 1;
    if (diameter > 1.0)
        diameter /= 1000.0;

    for (std::size_t i = 1; i < verts.size(); ++i) {
        int n1 = getOrCreateNode(verts[i - 1].first, verts[i - 1].second);
        int n2 = getOrCreateNode(verts[i].first, verts[i].second);
        addPipe(n1, n2, std::hypot(verts[i].first - verts[i - 1].first, verts[i].second - verts[i - 1].second), diameter);
    }
    if (closed && verts.size() > 2) {
        int n1 = getOrCreateNode(verts.back().first, verts.back().second);
        int n2 = getOrCreateNode(verts.front().first, verts.front().second);
        addPipe(n1, n2, std::hypot(verts.back().first - verts.front().first, verts.back().second - verts.front().second), diameter);
    }
    return 0;
}

// ─────────────── Entity: (old) POLYLINE + VERTEX ―──────────────────────── //
int IrrigationModel::parsePolylineEntity(const std::vector<std::pair<int, std::string>> &header, const std::vector<std::vector<std::pair<int, std::string>>> &vertices) {
    bool closed = false;
    double diameter = 0.05;
    for (const auto &kv: header) {
        if (kv.first == 70)
            closed = (std::stoi(kv.second) & 1) != 0;
        if (kv.first == 40)
            diameter = std::stod(kv.second);
    }
    if (diameter > 1.0)
        diameter /= 1000.0;

    std::vector<std::pair<double, double>> verts;
    for (const auto &vtx: vertices) {
        std::map<int, std::string> v = pairsToMap(vtx);
        if (v.count(10) && v.count(20))
            verts.emplace_back(std::stod(v.at(10)), std::stod(v.at(20)));
    }
    if (verts.size() < 2)
        return 1;

    for (std::size_t i = 1; i < verts.size(); ++i) {
        int n1 = getOrCreateNode(verts[i - 1].first, verts[i - 1].second);
        int n2 = getOrCreateNode(verts[i].first, verts[i].second);
        addPipe(n1, n2, std::hypot(verts[i].first - verts[i - 1].first, verts[i].second - verts[i - 1].second), diameter);
    }
    if (closed && verts.size() > 2) {
        int n1 = getOrCreateNode(verts.back().first, verts.back().second);
        int n2 = getOrCreateNode(verts.front().first, verts.front().second);
        addPipe(n1, n2, std::hypot(verts.back().first - verts.front().first, verts.back().second - verts.front().second), diameter);
    }
    return 0;
}

// ─────────────── DXF reader ────────────────────────────────────────────── //
int IrrigationModel::readDXF(const std::string &filename) {
    std::ifstream in(filename);
    if (!in.is_open())
        helios_runtime_error("ERROR (IrrigationModel::readDXF): Cannot open '" + filename + "'");

    std::vector<std::string> raw;
    std::string line;
    while (std::getline(in, line)) {
        trim(line);
        raw.push_back(line);
    }
    in.close();

    std::size_t i = 0;
    while (i + 1 < raw.size()) {
        if (std::stoi(raw[i]) != 0) {
            i += 2;
            continue;
        }
        std::string tag = raw[i + 1];
        i += 2;

        if (tag == "LINE") {
            std::vector<std::pair<int, std::string>> ent{{0, "LINE"}};
            while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
                ent.emplace_back(std::stoi(raw[i]), raw[i + 1]);
                i += 2;
            }
            parseLineEntity(pairsToMap(ent));
        } else if (tag == "LWPOLYLINE") {
            std::vector<std::pair<int, std::string>> ent{{0, "LWPOLYLINE"}};
            while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
                ent.emplace_back(std::stoi(raw[i]), raw[i + 1]);
                i += 2;
            }
            parseLWPolylineEntity(ent);
        } else if (tag == "POLYLINE") {
            std::vector<std::pair<int, std::string>> head{{0, "POLYLINE"}};
            while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
                head.emplace_back(std::stoi(raw[i]), raw[i + 1]);
                i += 2;
            }
            std::vector<std::vector<std::pair<int, std::string>>> verts;
            while (i + 1 < raw.size()) {
                if (std::stoi(raw[i]) != 0) {
                    i += 2;
                    continue;
                }
                std::string subt = raw[i + 1];
                if (subt == "VERTEX") {
                    std::vector<std::pair<int, std::string>> v{{0, "VERTEX"}};
                    i += 2;
                    while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
                        v.emplace_back(std::stoi(raw[i]), raw[i + 1]);
                        i += 2;
                    }
                    verts.push_back(std::move(v));
                } else if (subt == "SEQEND") {
                    i += 2;
                    break;
                } else {
                    break;
                }
            }
            parsePolylineEntity(head, verts);
        }
        /* ignore all other entity types */
    }

    // fix first node at 50 psi (placeholder until layer-based BCs are added)
    if (!nodes.empty()) {
        nodes.front().fixed = true;
        nodes.front().pressure = 50.0;
    }

    checkConnectivity(); // throw if any node is unreachable
    return 0;
}

// ─────────────── Connectivity check ────────────────────────────────────── //
void IrrigationModel::checkConnectivity() const {
    if (nodes.empty())
        return;

    std::vector<char> seen(nodes.size(), 0);
    std::queue<int> q;
    q.push(0);
    seen[0] = 1; // node 0 is fixed source

    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (const auto &p: pipes) {
            int v = (p.n1 == u) ? p.n2 : (p.n2 == u) ? p.n1 : -1;
            if (v >= 0 && !seen[v]) {
                seen[v] = 1;
                q.push(v);
            }
        }
    }
    for (std::size_t i = 0; i < nodes.size(); ++i)
        if (!seen[i])
            helios_runtime_error("ERROR (IrrigationModel): Node " + std::to_string(i) +
                                 " is disconnected from the fixed-pressure "
                                 "reference.");
}

// ─────────────── Linear solver (Gauss-Jordan) ─────────────────────────── //
std::vector<double> IrrigationModel::solveLinear(std::vector<std::vector<double>> A, std::vector<double> b) const {
    const int n = int(A.size());
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i + 1; j < n; ++j)
            if (std::fabs(A[j][i]) > std::fabs(A[pivot][i]))
                pivot = j;

        std::swap(A[i], A[pivot]);
        std::swap(b[i], b[pivot]);
        double diag = A[i][i];
        if (std::fabs(diag) < 1e-12)
            throw std::runtime_error("Singular hydraulic matrix");
        for (int j = i; j < n; ++j)
            A[i][j] /= diag;
        b[i] /= diag;

        for (int k = 0; k < n; ++k)
            if (k != i) {
                double f = A[k][i];
                for (int j = i; j < n; ++j)
                    A[k][j] -= f * A[i][j];
                b[k] -= f * b[i];
            }
    }
    return b;
}

// ─────────────── Solve pressures ───────────────────────────────────────── //
double IrrigationModel::pipeResistance(const IrrigationPipe &p) {
    // Darcy–Weisbach with Blasius f = 0.316 Re^-0.25 at Re≈1e5
    const double nu = 1.0e-6; // kinematic viscosity (m²/s)
    const double rho = 1000.0; // density (kg/m³)
    const double g = 9.80665;
    const double Q = 1.0e-4; // design flow (m³/s) – stub
    const double A = M_PI * p.diameter * p.diameter / 4.0;
    const double v = Q / A;
    const double Re = v * p.diameter / nu;
    const double f = 0.3164 / std::pow(Re, 0.25);
    return 32.0 * f * rho * p.length / (M_PI * M_PI * std::pow(p.diameter, 5) * g);
}

int IrrigationModel::solve() {
    const std::size_t n = nodes.size();
    std::vector<int> mapIndex(n, -1);
    int unknowns = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (!nodes[i].fixed)
            mapIndex[i] = unknowns++;

    std::vector<std::vector<double>> A(unknowns, std::vector<double>(unknowns, 0.0));
    std::vector<double> b(unknowns, 0.0);

    for (const auto &p: pipes) {
        double R = pipeResistance(p);
        int i = p.n1, j = p.n2;
        int ai = mapIndex[i], aj = mapIndex[j];

        if (ai >= 0) {
            A[ai][ai] += 1.0 / R;
            if (aj >= 0)
                A[ai][aj] -= 1.0 / R;
            else
                b[ai] += nodes[j].pressure / R;
        }
        if (aj >= 0) {
            A[aj][aj] += 1.0 / R;
            if (ai >= 0)
                A[aj][ai] -= 1.0 / R;
            else
                b[aj] += nodes[i].pressure / R;
        }
    }

    std::vector<double> x = solveLinear(A, b);
    for (std::size_t i = 0; i < n; ++i)
        if (mapIndex[i] >= 0)
            nodes[i].pressure = x[mapIndex[i]];
    return 0;
}

// ─────────────── DXF writer (unchanged except for const tweaks) ───────── //
int IrrigationModel::writeDXF(const std::string &filename) const {
    std::ofstream out(filename);
    if (!out.is_open())
        helios_runtime_error("ERROR (IrrigationModel::writeDXF): Cannot open '" + filename + "' for writing.");

    std::cout << "Writing DXF file '" << filename << "'..." << std::flush;

    out << "0\nSECTION\n2\nENTITIES\n";
    for (const auto &p: pipes) {
        const auto &n1 = nodes[p.n1];
        const auto &n2 = nodes[p.n2];
        out << "0\nLINE\n8\n0\n10\n" << n1.x << "\n20\n" << n1.y << "\n11\n" << n2.x << "\n21\n" << n2.y << "\n";
    }
    for (const auto &node: nodes)
        out << "0\nTEXT\n8\n0\n10\n" << node.x << "\n20\n" << node.y << "\n40\n0.2\n1\n" << node.pressure << "\n";

    out << "0\nENDSEC\n0\nEOF\n";
    std::cout << "done.\n";
    return 0;
}

int IrrigationModel::selfTest() {
    std::cout << "Running irrigation plug-in self-test..." << std::endl;
    readDXF("../plugins/irrigation/doc/simple_network.dxf");
    solve();
    writeDXF("../plugins/irrigation/doc/simple_network_out.dxf");
    for (size_t i = 0; i < nodes.size(); ++i) {
        std::cout << "Node " << i << " pressure: " << nodes[i].pressure << std::endl;
    }
    return 0;
}
