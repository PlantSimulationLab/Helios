/** \file "IrrigationModel.h" Primary header file for irrigation plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef IRRIGATIONMODEL
#define IRRIGATIONMODEL

#include "Context.h"

// ─────────────── Network primitives ──────────────────────────────────────── //
struct IrrigationNode {
    double x{}, y{}; //!< plan-view coordinates (m)
    double pressure{0.0}; //!< hydraulic head (psi)
    bool fixed{false}; //!< true ⇔ boundary condition
};

struct IrrigationPipe {
    int n1{-1}, n2{-1}; //!< node indices
    double length{0.0}; //!< centre-to-centre length (m)
    double diameter{0.05}; //!< internal diameter (m)
    double kminor{1.0}; //!< minor-loss coefficient (elbows, tees…)
};

// ─────────────────────── Main class ─────────────────────────────────────── //
class IrrigationModel {
public:
    explicit IrrigationModel(helios::Context *context);

    int readDXF(const std::string &filename); //!< import network
    int solve(); //!< compute pressures
    int writeDXF(const std::string &filename) const;
    int selfTest();

private:
    // ――― utilities ――― //
    int getOrCreateNode(double x, double y);
    int addPipe(int n1, int n2, double length, double diameter, double kminor = 1.0);

    static double pipeResistance(const IrrigationPipe &p);
    void checkConnectivity() const;

    // ――― DXF helpers ――― //
    int parseLineEntity(const std::map<int, std::string> &entity);
    int parseLWPolylineEntity(const std::vector<std::pair<int, std::string>> &);
    int parsePolylineEntity(const std::vector<std::pair<int, std::string>> &, const std::vector<std::vector<std::pair<int, std::string>>> &);

    // ――― linear solver ――― //
    std::vector<double> solveLinear(std::vector<std::vector<double>> A, std::vector<double> b) const;

    // ――― data ――― //
    helios::Context *context{nullptr};
    std::vector<IrrigationNode> nodes;
    std::vector<IrrigationPipe> pipes;
};

#endif
