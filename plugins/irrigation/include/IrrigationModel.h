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

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept> // For std::runtime_error
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
    std::string emitterType;
    double emitter_x;  // flow power
    double emitter_k; // flow coefficient
    double stakeHeight;
    std::string barbType;
};


class SprinklerConfigLibrary {

    std::unordered_map<std::string, SprinklerAssembly> sprinklerLibrary;

public:

    SprinklerConfigLibrary();

    void registerSprinklerAssembly(const SprinklerAssembly& type);
    bool hasSprinklerAssembly(const std::string& typeName) const;
    const SprinklerAssembly& getSprinklerType(const std::string& typeName) const;
    std::vector<std::string> getAvailableTypes() const;

    // sprinkler types stored based on experimental data
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
    double z; //elevation

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
    int zoneID = 0; //
    bool isActive = false;  // node active state for hydraulic calculation later
    bool isValveOpen = false; // this is for zone_valve nodes only
};

struct Link {
    int from;
    int to;
    double diameter;
    double length;
    std::string type;
    double flow; //m3/s
    int zoneID = 0;

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


struct PressureLossResult {
    int fromNode;
    int toNode;
    double pressureLoss;
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
    size_t getCacheSize() const;

    // Main system creation
    IrrigationModel() : sprinklerLibrary() {} // calls SprinklerConfigLibrary constructor
    int getWaterSourceId() const;
    int getNextNodeId() const;
    const std::unordered_map<int, Node>& getNodes() const;
    const std::vector<Link>& getLinks() const;
    std::vector<int> getNodesByType(const std::string& type) const;
    std::vector<Link> getLinksByType(const std::string& type) const;

    // Print network in visualization format
    void printNetwork() const;
    HydraulicResults calculateHydraulics(bool doPreSize, const std::string& sprinklerAssemblyType,
                                                     double Qspecified, double Pw,
                                                     double V_main, double V_lateral);
    std::string getSystemSummary() const;
    double calculateEmitterFlow(const std::string& sprinklerAssemblyType,
                                double pressure,
                                bool updateEmitterNodes = true);
    void setBoundaryPolygon(const std::vector<Position>& polygon);

    void preSizePipes(double V_main = 1.5, double V_lateral = 2.0);
    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest(int argc = 0, char** argv = nullptr);
    std::vector<PressureLossResult> getBarbToEmitterPressureLosses() const;
    std::vector<PressureLossResult> getLateralToBarbPressureLosses() const;
    void printPressureLossAnalysis(const IrrigationModel& system);
    void writePressureLossesToFile(const IrrigationModel& system, const std::string& filename);

    double getPressureDifference(int nodeId1, int nodeId2) const;
    std::vector<std::pair<int, int>> findConnectedNodePairs(const std::string& type1, const std::string& type2) const;
    bool connectionExists(int node1, int node2) const;
    void createVirtualConnection(int node1, int node2,
                                               const ComponentSpecs& specs,
                                               const std::string& connectionType,
                                               const std::string& assemblyType);
    int countNodesByType(const std::string& type) const;
    int getMaxNodeId() const;
    void createAssembliesFromIrricad(const std::string& sprinklerAssemblyType);
    // void addWaterSource(double fieldLength, double fieldWidth,
    //                                const std::string& lateralDirection,
    //                                SubmainPosition submainPosition);
    bool loadFromTextFile(const std::string& filename);

    void validateHydraulicSystem() const;
    void writeMatrixToFile(const std::vector<std::vector<double>>& A,
                                          const std::vector<double>& RHS,
                                          const std::vector<int>& orderedNodeIds,
                                          const std::string& filename);

    ////////////////////////////////////////////////////////////////////////////////
    // Multi-zone functionality
    void assignZones(
        int numZones,
        const std::vector<std::vector<Position>>& zoneBoundaries,
        double Pw,
        double sprinklerSpacing,
        double lineSpacing,
        const std::string& connectionType,
        const std::string& sprinklerConfig,
        SubmainPosition submainPos
    );

    // Zone query functions
    std::vector<int> getNodesByZone(int zoneID) const;
    std::vector<size_t> getLinksByZone(int zoneID) const;


    //functions to help simulate any combination of active zones
    // HydraulicResults calculateHydraulicsByZone(int activeZoneID,
    //                                          bool doPreSize,
    //                                          const std::string& sprinklerAssemblyType,
    //                                          double Qspecified,
    //                                          double Pw,
    //                                          double V_main,
    //                                          double V_lateral);

    // Helper methods
    // using cached BFS using valve control
    void openZoneValve(int zoneID);
    void closeZoneValve(int zoneID);
    bool isZoneValveOpen(int zoneID) const;
    void setZoneValveState(int zoneID, bool open);
    // fast call for changing zone_valve state
    void activateAllZones();
    void deactivateAllZones();
    void activateSingleZone(int zoneID);
    void activateZones(const std::vector<int>& zoneIDs);
    HydraulicResults calculateHydraulicsMultiZoneOptimized(
        bool doPreSize,
        const std::string& sprinklerAssemblyType,
        double Qspecified,
        double Pw,
        double V_main,
        double V_lateral);
    void initialize();
    void setActiveNodeMethod(bool useFastMethod);
    void checkUnassignedNodes();

    // function to generate the system curve (GPM vs head in feet)
    std::vector<std::pair<double, double>> generateSystemCurve(
            const std::vector<double>& emitterRequiredPressure,
            const std::string& sprinklerAssemblyType,
            double V_main,
            double V_lateral,
            int referenceNodeId = -1,
            double staticHead = 0.0);

    void writeSystemCurveToCsvWithFit(const std::string& filename,
                                      const std::vector<std::pair<double, double>>& curve) const;
    static std::tuple<double, double, double> fitCurveQuadratic(
            const std::vector<std::pair<double, double>>& curve);


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
    // void addSubmainAndWaterSourceIrregular(double fieldLength, double fieldWidth,
    //                                                       const std::string& lateralDirection,
    //                                                       SubmainPosition submainPosition);

    std::vector<int> createHorizontalSubmain(const Position& startPos, double fieldLength,
                                             const std::vector<Position>& boundary, double lineSpacing, int zoneID);
    std::vector<int> createVerticalSubmain(const Position& startPos, double fieldWidth,
                                         const std::vector<Position>& boundary, double lineSpacing, int zoneID);
    // std::vector<int> createHorizontalSubmain(const Position& startPos, double fieldLength);
    // std::vector<int> createVerticalSubmain(const Position& startPos, double fieldWidth);

    void createSprinklerSystemGeneral(double fieldLength, double fieldWidth,
                                              double sprinklerSpacing, double lineSpacing,
                                              const std::string& connectionType,
                                              const std::string& sprinklerConfig, double minX, double minY, int zoneID);

    // Position calculateWaterSourcePosition(double fieldLength, double fieldWidth,
    //                                     const std::string& lateralDirection) const;

    void validateParameters(double fieldLength, double fieldWidth,
                  double sprinklerSpacing, double lineSpacing) const; //include into the createCompleteSystem()
    // void validateHydraulicSystem() const;
    void validateSubmainConnectivity() const;
    void validateMinimumSprinklersPerRow() const;
    void validateCompleteSprinklerUnits() const;
    void addMissingSprinklerUnits(double rowX, const std::vector<int>& existingSprinklers, int count, const std::string& sprinklerConfig);
    void ensureMinimumSprinklersPerRow(const std::string& sprinklerConfig);
    void ensureAllRowsConnected();
    Position calculateLateralMidpoint(const Link* lateral) const;
    bool hasCompleteSprinklerUnit(int junctionId) const;

    void splitLateralAndConnect(const Link* lateral, int submainNodeId, std::map<double, std::vector<size_t>>& rowLaterals,
        const std::string& lateralDirection, int zoneID);
    Position findOptimalConnectionPointOnLateral(const Link* lateral, const Position& submainPos) const;
    // to ensure that there are at least two microsprinkler assemblies in each row
    bool hasEmitterConnection(int barbId) const;

    void buildNeighborLists();
    static double calculateResistance(double Re, double Wbar, double Kf_barb, const Link& link, int iter);
    double minorLoss_kf(const double Re, const std::string& sprinklerAssemblyType);
    const Link* findLink(int from, int to) const;
    Link* findLink(int from, int to); //for updating flow

    //adding new functions
    Position calculatePolygonBasedSubmainPosition(SubmainPosition position,
                                                             double fieldLength, double fieldWidth) const;

    void connectSubmainToLaterals(const std::vector<int>& submainNodeIds, const std::string& lateralDirection, int zoneID);

    const Link* findOptimalLateralForConnection(
        const std::map<double, std::vector<size_t>>& rowLaterals, // map of rowKey -> lateral indices
        const Position& submainPos,
        double expectedRowKey,          // Ensure same row
        const std::string& lateralDirection) const;

    void validateSubmainConnections(const std::vector<int>& submainNodeIds) const;

    bool validateLink(const Link& link) const;

   // double calculateLateralLength(const Link* lateral) const;
    double computeLinkFlows(int nodeId, int parentId,
                        const std::unordered_map<int, std::vector<int>>& adj,
                        std::unordered_set<int>& visited); // recursion function to calculate flows upstream

    /////// functions to assist irricad layout re creation
    void recreateLateralConnections();
    void removeExistingLateralConnections();
    void createLateralSegments();
    void connectLateralsToSubmain();
    bool shouldRemoveLateralNeighbor(int nodeId1, int nodeId2);
    void createLateralConnection(int startNode, int endNode, double distance);
    int findClosestLateralToSubmain(const std::vector<int>& lateralNodes,
                                               const std::vector<int>& submainNodes);
    int findClosestSubmainNode(int lateralNode, const std::vector<int>& submainNodes);
    void createLateralToSubmainConnection(int lateralNode, int submainNode, double distance);
    std::vector<int> getSubmainNodeIdsFromLinks();


    /////////////////////////////////////////////////////////////////////////////////////////
    ///
    // Zone-specific system creation
    void createIrregularSystemForZone(
        const std::vector<Position>& boundary,
        double sprinklerSpacing,
        double lineSpacing,
        const std::string& connectionType,
        const std::string& sprinklerConfig,
        SubmainPosition submainPos,
        int zoneID
    );
    Position calculateOptimalSubmainPositionForZone(
        SubmainPosition position, double minX, double minY,
        double fieldLength, double fieldWidth);

    // Zone submain management
    void addSubmainForZone(double fieldLength, double fieldWidth,
                                          const std::string& lateralDirection,
                                          SubmainPosition submainPosition,
                                          int zoneID,
                                          double lateralSpacing);

    // Valve and connection management
    void addZoneValveAndConnect(int zoneID, const std::vector<Position>& zoneBoundary);
    Position calculateZoneCentroid(const std::vector<Position>& boundary) const;
    int findZoneSubmainJunction(int zoneID) const;
    int createZoneSubmainJunction(int zoneID, const std::vector<Position>& zoneBoundary);

    // newly added functions
    Position calculateOptimalValvePosition(const Position& submainPos,
                                      const std::vector<Position>& zoneBoundary);

    void connectValveToSubmainWith90Degree(int valveId, int submainId, int zoneID);

    void connectWaterSourceToValveWith90Degree(int valveId, int zoneID);


    //Helper functions for updated assignZone function
    void create90DegreeConnection(int fromNodeId, int toNodeId, int zoneID);
    void optimizeZoneValveConnections();
    void cleanupMainlineNetwork();


    // adding function to track irrigation zone active state
    std::unordered_map<int, bool> zoneActiveState; // zoneID -> isActive
    std::unordered_map<int, double> zoneDemand;    // zoneID -> total flow demand


    // functions to do BFS for checking node active state & option to do cached optimization
    std::unordered_map<std::uint32_t, std::unordered_set<int>> activeNodeCache;
    uint32_t currentValveConfigHash = 0;

    // Pre-computed valve-to-nodes mapping
    std::unordered_map<int, std::unordered_set<int>> valveToNodes;

    // Helper methods
    void bfsCollectNodes(int startId, int targetZoneID, std::unordered_set<int>& result);
    uint32_t computeValveConfigHash() const;
    void ensureMainlineConnectivity(std::unordered_set<int>& activeNodes);
    void propagateMainlineActivation();
    const std::unordered_set<int>& getActiveNodesCached();
    void updateActiveNodesFast();
    void updateActiveNodesByTreeTraversal();
    void bfsFromValve(int valveId);
    void buildValveToNodesMapping();


    void clearActiveNodeCache();
    void printActiveNodeStats() const;


};

#endif
