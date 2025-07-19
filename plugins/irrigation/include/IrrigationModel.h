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
#include <vector>
#include <cmath>
#include <string>


#include <map>
#include <unordered_map>

struct Position {
    double x, y;
};

// Using std::map with node IDs
struct Node {
    int id;
    std::string type; //junction, barb, emitter
    Position position;
    double pressure;
    bool is_fixed;
};

struct Link {
    int from;
    int to;
    double length;
    double diameter;
    std::string type; //lateral, submain, main, barbToemitter
    //double resistance;
};

class IrrigationModel {
public:
    std::unordered_map<int, Node> nodes;  // NodeID -> Node
    std::vector<Link> links;              // Pipe connections

    // Create a sprinkler system
    void createSprinklerSystem(double Pw, double fieldLength, double fieldWidth,
                             double sprinklerSpacing, double lineSpacing,
                             const std::string& connectionType);

    // Helper functions
    double calculateEmitterFlow(double Pw) const;
    void printNetwork() const;

private:
    static constexpr double INCH_TO_METER = 0.0254;
};

/*
struct IrrigationPipe {
    int    n1{-1}, n2{-1}; //!< node indices
    double length{0.0};    //!< centre-to-centre length [m]
    double diameter{0.05}; //!< internal diameter [m]
    double kminor{1.0};    //!< minor-loss coefficient
    double resistance{0.0};
};
*/

// ─────────────────────── Main class ───────────────────────────────────── //
/*class IrrigationModel {
public:
    explicit IrrigationModel(helios::Context* ctx);

    int readDXF (const std::string& filename); //!< import geometry
    int solve();                            //!< compute pressures
    int writeDXF(const std::string& filename) const;

    int selfTest();

private:
    // ――― utilities ――― //
    int  getOrCreateNode(double x, double y);
    int  addPipe(int n1, int n2, double L,
                 double diameter, double kminor = 1.0);

    static double pipeResistance(const IrrigationPipe& p);
    void checkConnectivity() const;

    // ――― DXF helpers ――― //
    int parseLineEntity     (const std::map<int, std::string>& ent);
    int parseLWPolylineEntity(const std::vector<std::pair<int, std::string>>&);
    int parsePolylineEntity (const std::vector<std::pair<int, std::string>>&,
                             const std::vector<std::vector<
                                 std::pair<int, std::string>>>&);

    // ――― internal solver ――― //
    std::vector<double> solveLinear(std::vector<std::vector<double>> A,
                                    std::vector<double>              b) const;

    // ――― data ――― //
    helios::Context*               context{nullptr};
    std::vector<IrrigationNode>    nodes;
    std::vector<IrrigationPipe>    pipes;
};*/
//
// class IrrigationModel{
// public:
//     explicit IrrigationModel(helios::Context* context);
//
//     // Core functionality
//     void createBasicSystem(float field_length, float field_width);
//     void solveHydraulics();
//
//     // Getters
//     const std::vector<IrrigationNode>& getNodes() const { return nodes; }
//     const std::vector<IrrigationPipe>& getPipes() const { return pipes; }
//
// private:
//     helios::Context* context;
//     std::vector<IrrigationNode> nodes;
//     std::vector<IrrigationPipe> pipes;
//
//     void calculatePipeResistances();
// };


#endif
