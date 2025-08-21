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

#ifndef IRRIGATIONMODEL_H
#define IRRIGATIONMODEL_H

#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <stdexcept>  // For std::runtime_error

class SprinklerSystem;

struct Position {
    double x;
    double y;

    // distance calculation
    double distanceTo(const Position& other) const {
        return std::hypot(x - other.x, y - other.y);
    }
};




struct Node {
    int id;
    std::string type;
    Position position;
    double pressure;
    bool is_fixed;
    std::vector<int> neighbors;
};

struct Link {
    int from;
    int to;
    double diameter;
    double length;
    std::string type;

    // Helper method
    std::string toString() const {
        return type + " from " + std::to_string(from) +
               " to " + std::to_string(to) +
               " (L=" + std::to_string(length) + "m)";
    }
};

struct HydraulicResults {
    std::vector<double> nodalPressures;
    std::vector<double> flowRates;
    bool converged = false;
    int iterations = 0;
};

enum class SubmainPosition {
    NORTH,
    SOUTH,
    MIDDLE
};


class IrrigationModel {
public:
    static constexpr double INCH_TO_METER = 0.0254;
    static constexpr double FEET_TO_METER = 0.3048;
    std::unordered_map<int, Node> nodes;
    std::vector<Link> links;

    // Main system creation

    void createCompleteSystem(double Pw, double fieldLength, double fieldWidth,
                       double sprinklerSpacing, double lineSpacing,
                       const std::string& connectionType,
                       SubmainPosition submainPos = SubmainPosition::NORTH);

    int getWaterSourceId() const { return waterSourceId; }
    int getNextNodeId() const;
    // Print network in visualization format
    void printNetwork() const;
    HydraulicResults calculateHydraulics(const std::string& nozzleType, double Qspecified, double Pw);
    //diagonistic test on the irrigation system
    void validateHydraulicSystem() const;
    // Get system summary as string
    std::string getSystemSummary() const;
    double calculateEmitterFlow(const std::string& nozzleType, double pressure);


private:

    int waterSourceId = -1;  // Tracks water source node ID

    // Helper methods
    void validateParameters(double fieldLength, double fieldWidth,
                          double sprinklerSpacing, double lineSpacing) const;

    void createSprinklerSystem(double fieldLength, double fieldWidth,
                             double sprinklerSpacing, double lineSpacing,
                             const std::string& connectionType);

    Position calculateWaterSourcePosition(double fieldLength, double fieldWidth,
                                        const std::string& lateralDirection) const;

   // void addSubmainAndWaterSource(double fieldLength, double fieldWidth,
    //                                         const std::string& lateralDirection);

    void addSubmainAndWaterSource(double fieldLength, double fieldWidth,
                                             const std::string& lateralDirection,
                                             SubmainPosition submainPosition);

    void buildNeighborLists();
    double calculateResistance(double Re, double Wbar, const Link& link, int iter);
    const Link* findLink(int from, int to) const;

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
