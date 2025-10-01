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
#include <map>
#include <string>
#include <unordered_map>
#include <cmath>
#include <stdexcept>  // For std::runtime_error

const double INCH_TO_METER = 0.0254;
const double FEET_TO_METER = 0.3048;

struct ComponentSpecs {
    double diameter;        // in meters
    double length;          // in meters
    std::string material;
};

struct SprinklerAssembly {
    std::string name;
    ComponentSpecs barbToEmitter;
    ComponentSpecs lateralToBarb;
    std:: string emitterType;
    double emitter_x;  // flow power
    double emitter_k; // flow coefficient
    std:: string barbType;
};


class SprinklerConfigLibrary {

    std::unordered_map<std::string, SprinklerAssembly> sprinklerLibrary;

public:
    SprinklerConfigLibrary();

    void registerSprinklerAssembly(const SprinklerAssembly& type);
    bool hasSprinklerAssembly(const std::string& typeName) const;
    const SprinklerAssembly& getSprinklerType(const std::string& typeName) const;
    std::vector<std::string> getAvailableTypes() const;

    // Pre-configured sprinkler types
    static SprinklerAssembly create_NPC_Nelson_flat();
    static SprinklerAssembly create_NPC_Toro_flat();
    static SprinklerAssembly create_NPC_Toro_sharp();
    static SprinklerAssembly create_PC_Nelson_flat();
    static SprinklerAssembly create_PC_Toro_flat();
    static SprinklerAssembly create_PC_Toro_sharp();

};

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
    double flow; //include flow rate if the node is an emitter
    std::vector<int> neighbors;
    int zoneID = 0; // default one zone
};

struct Link {
    int from;
    int to;
    double diameter;
    double length;
    std::string type;
    double flow; //m3/s
    int zoneID = 0;
    //velocity

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
    std::vector<double> emitterFlows;
    bool converged = false;
    int iterations = 0;
};

enum class SubmainPosition {
    NORTH,
    SOUTH,
    MIDDLE
};

enum class ZoningMode {
    X_BASED,
    Y_BASED,
    XY_BASED,
    AUTO_BASED
};


class IrrigationModel {
public:
    std::unordered_map<int, Node> nodes;
    std::vector<Link> links;

    // Main system creation
    IrrigationModel() : sprinklerLibrary() {} // calls SprinklerConfigLibrary constructor
    int getWaterSourceId() const;
    int getNextNodeId() const;
    // Print network in visualization format
    void printNetwork() const;
    HydraulicResults calculateHydraulics(bool doPreSize, const std::string& sprinklerAssemblyType,
                                                     double Qspecified, double Pw,
                                                     double V_main, double V_lateral);
    std::string getSystemSummary() const;
    double calculateEmitterFlow(const std::string& sprinklerAssemblyType, double pressure);
    void setBoundaryPolygon(const std::vector<Position>& polygon);
    void createIrregularSystem(double Pw, const std::vector<Position>& boundary,
                                 double sprinklerSpacing, double lineSpacing,
                                 const std::string& connectionType,
                                 const std::string& sprinklerConfig,
                                 SubmainPosition submainPos);
    void preSizePipes(double V_main = 1.5, double V_lateral = 2.0);
    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest(int argc = 0, char** argv = nullptr);


private:

    int waterSourceId = -1;  // Tracks water source node ID

    // Helper methods
    std::vector<Position> boundaryPolygon; // for irregular shapes
    SprinklerConfigLibrary sprinklerLibrary;

    // Helper methods for polygon operations
    bool isPointInsidePolygon(const Position& point) const;

    // Irregular system methods
    Position calculateOptimalSubmainPosition(SubmainPosition position,
                                                            double fieldLength, double fieldWidth) const;
    double getLateralRowKey(const Link& lateral, const std::string& lateralDirection) const;

    double distanceToPolygon(const Position& point) const;
    double pointToSegmentDistance(const Position& point,
                                                 const Position& segStart,
                                                 const Position& segEnd) const; //helper
    void createAndConnectWaterSource(const std::vector<int>& submainNodeIds);
    void addSubmainAndWaterSourceIrregular(double fieldLength, double fieldWidth,
                                                          const std::string& lateralDirection,
                                                          SubmainPosition submainPosition);

    std::vector<int> createHorizontalSubmain(const Position& startPos, double fieldLength);
    std::vector<int> createVerticalSubmain(const Position& startPos, double fieldWidth);
    void createSprinklerSystemGeneral(double fieldLength, double fieldWidth,
                                              double sprinklerSpacing, double lineSpacing,
                                              const std::string& connectionType,
                                              const std::string& sprinklerAssembly);

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

    void splitLateralAndConnect(const Link* lateral, int submainNodeId, std::map<double, std::vector<size_t>>& rowLaterals, const std::string& lateralDirection);
    Position findOptimalConnectionPointOnLateral(const Link* lateral, const Position& submainPos) const;
    // to ensure that there are at least two microsprinkler assemblies in each row
    bool hasEmitterConnection(int barbId) const;
    void addSubmainAndWaterSource(double fieldLength, double fieldWidth,
                                             const std::string& lateralDirection,
                                             SubmainPosition submainPosition);

    void buildNeighborLists();
    static double calculateResistance(double Re, double Wbar, double Kf_barb, const Link& link, int iter);
   // double minorLoss_kf(double Re, const std::string& name = "NPC_flat_barb");
    double minorLoss_kf(const double Re, const std::string& sprinklerAssemblyType);
    const Link* findLink(int from, int to) const;
    Link* findLink(int from, int to); //for updating flow


    //adding new functions
    Position calculatePolygonBasedSubmainPosition(SubmainPosition position,
                                                             double fieldLength, double fieldWidth) const;

    Position calculateLineIntersection(const Position& p1, const Position& p2,
                                                  const Position& p3, const Position& p4) const;

    void connectSubmainToLaterals(const std::vector<int>& submainNodeIds,
                                             const std::string& lateralDirection);

    const Link* findOptimalLateralForConnection(
        const std::map<double, std::vector<size_t>>& rowLaterals, // map of rowKey -> lateral indices
        const Position& submainPos,
        double expectedRowKey,          // Ensure same row
        const std::string& lateralDirection) const;

    Position calculateSubmainLateralIntersection(const Position& submainPos,
                                                            const Position& lateralStart,
                                                            const Position& lateralEnd,
                                                            const std::string& lateralDirection) const;
    void validateSubmainConnections(const std::vector<int>& submainNodeIds) const;
    int findNearestSubmainNode(const Position& point,
                                          const std::vector<int>& submainNodeIds,
                                          double expectedRowKey,
                                          const std::string& lateralDirection,
                                          double tolerance = 1.0) const;
    double calculateLateralCentrality(const Link* lateral,
                                                 const std::vector<const Link*>& rowLaterals) const;
    bool validateLink(const Link& link) const;

    double calculateLateralLength(const Link* lateral) const;
    double computeLinkFlows(int nodeId, int parentId,
                        const std::unordered_map<int, std::vector<int>>& adj);
    // recursion function to calculate flows upstream
    //double getFlowUpstream(int nodeId, int parentId);
    // help functions for delineating irrigation zones
    void assignZones(int numZones, ZoningMode mode, int zonesX = 1, int zonesY = 1);
    void addSubmainAndValve(int zoneId);
   // void activateZone(int zoneId, double Qspecified);

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
