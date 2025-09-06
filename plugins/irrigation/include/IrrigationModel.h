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
    // Get system summary as string
    std::string getSystemSummary() const;
    double calculateEmitterFlow(const std::string& nozzleType, double pressure);

    // for irregular shapes

    void setBoundaryPolygon(const std::vector<Position>& polygon);
    void createIrregularSystem(double Pw, const std::vector<Position>& boundary,
                                 double sprinklerSpacing, double lineSpacing,
                                 const std::string& connectionType,
                                 SubmainPosition submainPos);
    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest(int argc = 0, char** argv = nullptr);


private:

    int waterSourceId = -1;  // Tracks water source node ID

    // Helper methods
    std::vector<Position> boundaryPolygon; // for irregular shapes

    // Helper methods for polygon operations
    bool isPointInsidePolygon(const Position& point) const;

    Position findPolygonEntryPoint(const Position& outsidePoint, const Position& insidePoint) const;
    bool lineSegmentIntersection(const Position& p1, const Position& p2,
                               const Position& p3, const Position& p4,
                               Position& result) const;
    int findClosestSprinklerJunction(const Position& referencePoint) const;

    // Irregular system methods
    Position calculateOptimalSubmainPosition(SubmainPosition position,
                                                            double fieldLength, double fieldWidth) const;
    std::vector<int> createSubmainAlongPath(const Position& startPos,
                                                       const std::string& direction,
                                                       double fieldLength, double fieldWidth);
    void connectSubmainToSprinklers(int submainNodeId);
 //   void connectSubmainToSprinkler(int submainNodeId, int sprinklerId);
    void connectSubmainToLateral(int submainNodeId, const Link* lateral);
    void connectSubmainToLaterals(int submainNodeId, const std::string& lateralDirection);
    double getLateralRowKey(const Link& lateral, const std::string& lateralDirection) const;

    double distanceToPolygon(const Position& point) const;
    double pointToSegmentDistance(const Position& point,
                                                 const Position& segStart,
                                                 const Position& segEnd) const; //helper
    Position closestPointOnSegment(const Position& point,
                                                   const Position& segStart,
                                                   const Position& segEnd) const;
    void createAndConnectWaterSource(const std::vector<int>& submainNodeIds);
    Position calculateOptimalWaterSourcePosition(const std::vector<int>& submainNodeIds) const;
    void connectWaterSourceToNearestSubmain(int waterSourceId, const std::vector<int>& submainNodeIds);
    Position calculateOutwardDirection(const Position& point) const;

    void addSubmainAndWaterSourceIrregular(double fieldLength, double fieldWidth,
                                                          const std::string& lateralDirection,
                                                          SubmainPosition submainPosition);

    std::vector<int> createHorizontalSubmain(const Position& startPos, double fieldLength);
    std::vector<int> createVerticalSubmain(const Position& startPos, double fieldWidth);

    void createSprinklerSystem(double fieldLength, double fieldWidth,
                             double sprinklerSpacing, double lineSpacing,
                             const std::string& connectionType);


    void createSprinklerSystemGeneral(double fieldLength, double fieldWidth,
                                              double sprinklerSpacing, double lineSpacing,
                                              const std::string& connectionType);

    Position calculateWaterSourcePosition(double fieldLength, double fieldWidth,
                                        const std::string& lateralDirection) const;

    void validateParameters(double fieldLength, double fieldWidth,
                  double sprinklerSpacing, double lineSpacing) const; //include into the createCompleteSystem()
    void validateHydraulicSystem() const;
    void validateSubmainConnectivity() const;
    void validateMinimumSprinklersPerRow() const;
    void validateCompleteSprinklerUnits() const;
    void  addMissingSprinklerUnits(double rowX, const std::vector<int>& existingSprinklers, int count);
    void ensureMinimumSprinklersPerRow();
    void ensureAllRowsConnected();
    Position calculateLateralMidpoint(const Link* lateral) const;
    bool hasCompleteSprinklerUnit(int junctionId) const;

    const Link* findOptimalLateralConnection(const std::vector<const Link*>& laterals,
                                                         const Position& submainPos) const;
    void splitLateralAndConnect(const Link* lateral, int connectionNodeId,
                                           int submainNodeId, const Position& connectionPoint);
    Position findOptimalConnectionPointOnLateral(const Link* lateral, const Position& submainPos) const;
    // to ensure that there are at least two microsprinkler assemblies in each row

    bool hasEmitterConnection(int barbId) const;

   // void addSubmainAndWaterSource(double fieldLength, double fieldWidth,
    //                                         const std::string& lateralDirection);

    void addSubmainAndWaterSource(double fieldLength, double fieldWidth,
                                             const std::string& lateralDirection,
                                             SubmainPosition submainPosition);

    void buildNeighborLists();
    double calculateResistance(double Re, double Wbar, const Link& link, int iter);
    const Link* findLink(int from, int to) const;

    //adding new functions
    void connectSubmainToLateral(int submainNodeId);
    const Link* findClosestLateral(const std::vector<const Link*>& laterals,
                                               const Position& submainPos) const;
    void connectSubmainToLateralSegment(int submainNodeId, const Link* lateral);
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
