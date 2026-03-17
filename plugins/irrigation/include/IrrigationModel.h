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

//! Physical properties of a pipe component.
struct ComponentSpecs {
    double diameter;        // in meters
    double length;          // in meters
    std::string material;
};

//! Configuration for a complete sprinkler unit, including pipe fittings and emitter hydraulic properties.
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


//! Library of predefined sprinkler assembly configurations.
class SprinklerConfigLibrary {

    std::unordered_map<std::string, SprinklerAssembly> sprinklerLibrary;

public:

    //! Constructor that initializes the library with predefined sprinkler assembly types.
    SprinklerConfigLibrary();

    //! Register a custom sprinkler assembly in the library.
    void registerSprinklerAssembly(const SprinklerAssembly& type);
    //! Check if a sprinkler assembly type is registered in the library.
    bool hasSprinklerAssembly(const std::string& typeName) const;
    //! Get a sprinkler assembly by type name.
    const SprinklerAssembly& getSprinklerType(const std::string& typeName) const;
    //! Get a list of all registered sprinkler assembly type names.
    std::vector<std::string> getAvailableTypes() const;

    // sprinkler types stored based on experimental data
    static SprinklerAssembly create_NPC_Nelson_flat();
    static SprinklerAssembly create_NPC_Toro_flat();
    static SprinklerAssembly create_NPC_Toro_sharp();
    static SprinklerAssembly create_PC_Nelson_flat();
    static SprinklerAssembly create_PC_Toro_flat();
    static SprinklerAssembly create_PC_Toro_sharp();

};

//! 3D spatial coordinate in the irrigation network.
struct Position {
    double x;
    double y;
    double z; //elevation

    //! Compute the 2D Euclidean distance to another position.
    double distanceTo(const Position& other) const {
        return std::hypot(x - other.x, y - other.y);
    }
};


//! A point in the hydraulic network.
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

//! A pipe segment connecting two nodes.
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

//! Output from the hydraulic solver.
struct HydraulicResults {
    std::vector<double> nodalPressures;
    std::vector<double> flowRates;
    std::vector<double> emitterFlows;
    bool converged = false;
    int iterations = 0;
};


//! Pressure loss between two nodes.
struct PressureLossResult {
    int fromNode;
    int toNode;
    double pressureLoss;
};



//! Placement of the submain distribution line within an irrigation zone.
enum class SubmainPosition {
    NORTH,  //!< Place the submain at the north (top) edge
    SOUTH,  //!< Place the submain at the south (bottom) edge
    MIDDLE  //!< Place the submain through the center
};

enum class ZoningMode {
    X_BASED,
    Y_BASED,
    XY_BASED,
    AUTO_BASED
};


//! Irrigation system hydraulic model.
class IrrigationModel {
public:
    std::unordered_map<int, Node> nodes;
    std::vector<Link> links;
    //! Return the memory usage of internal caches.
    size_t getCacheSize() const;

    //! Default constructor. Initializes the sprinkler configuration library with predefined assembly types.
    IrrigationModel() : sprinklerLibrary(), nextNodeId(1) {}
    //! Get the water source node ID.
    int getWaterSourceId() const;
    //! Get the next available node ID.
    int getNextNodeId();
    //! Get all nodes in the system.
    const std::unordered_map<int, Node>& getNodes() const;
    //! Get all links in the system.
    const std::vector<Link>& getLinks() const;
    //! Get node IDs filtered by type string.
    std::vector<int> getNodesByType(const std::string& type) const;
    //! Get links filtered by type string.
    std::vector<Link> getLinksByType(const std::string& type) const;

    //! Disable command-line output messages from this plug-in.
    void disableMessages();

    //! Enable command-line output messages from this plug-in.
    void enableMessages();

    //! Check if command-line output messages are enabled.
    bool isMessageEnabled() const { return message_flag; }

    //! Print the network in a visualization format.
    void printNetwork() const;
    //! Run the single-zone hydraulic solver.
    HydraulicResults calculateHydraulics(bool doPreSize, const std::string& sprinklerAssemblyType,
                                                     double Qspecified, double Pw,
                                                     double V_main, double V_lateral);
    //! Get a text summary of the network.
    std::string getSystemSummary() const;
    //! Calculate the emitter discharge rate based on sprinkler assembly type and operating pressure.
    /**
     * \param[in] sprinklerAssemblyType Name of the sprinkler assembly from the library.
     * \param[in] pressure Operating pressure (psi).
     * \param[in] updateEmitterNodes If true (default), writes the computed flow to all emitter nodes.
     * \return Emitter flow rate (m3/s).
     */
    double calculateEmitterFlow(const std::string& sprinklerAssemblyType,
                                double pressure,
                                bool updateEmitterNodes = true);
    //! Set an irregular field boundary polygon.
    void setBoundaryPolygon(const std::vector<Position>& polygon);

    //! Automatically size pipe diameters based on target maximum velocities.
    /**
     * \param[in] V_main Target maximum mainline velocity (m/s). Default: 1.5.
     * \param[in] V_lateral Target maximum lateral velocity (m/s). Default: 2.0.
     */
    void preSizePipes(double V_main = 1.5, double V_lateral = 2.0);
    //! Self-test
    /**
     * \return 0 if test was successful, 1 if test failed.
     */
    static int selfTest(int argc = 0, char** argv = nullptr);
    //! Get pressure drops across all barb-to-emitter connections.
    std::vector<PressureLossResult> getBarbToEmitterPressureLosses() const;
    //! Get pressure drops across all lateral-to-barb connections.
    std::vector<PressureLossResult> getLateralToBarbPressureLosses() const;
    //! Print a formatted pressure loss analysis to the console.
    void printPressureLossAnalysis(const IrrigationModel& system);
    //! Write the pressure loss analysis to a text file.
    void writePressureLossesToFile(const IrrigationModel& system, const std::string& filename);

    //! Get the pressure difference between two nodes.
    double getPressureDifference(int nodeId1, int nodeId2) const;
    //! Find pairs of connected nodes matching the specified types.
    std::vector<std::pair<int, int>> findConnectedNodePairs(const std::string& type1, const std::string& type2) const;
    //! Check if a connection exists between two nodes.
    bool connectionExists(int node1, int node2) const;
    //! Create a virtual connection between two nodes.
    void createVirtualConnection(int node1, int node2,
                                               const ComponentSpecs& specs,
                                               const std::string& connectionType,
                                               const std::string& assemblyType);
    //! Count the number of nodes of a given type.
    int countNodesByType(const std::string& type) const;
    //! Get the maximum node ID in the system.
    int getMaxNodeId() const;
    //! Recreate sprinkler assemblies from an Irricad layout.
    void createAssembliesFromIrricad(const std::string& sprinklerAssemblyType);
    //! Load an irrigation system from a text file.
    bool loadFromTextFile(const std::string& filename);

    //! Validate the hydraulic system (water source, connectivity, link integrity).
    void validateHydraulicSystem() const;
    //! Write the linear system matrix to a file for debugging.
    void writeMatrixToFile(const std::vector<std::vector<double>>& A,
                                          const std::vector<double>& RHS,
                                          const std::vector<int>& orderedNodeIds,
                                          const std::string& filename);

    ////////////////////////////////////////////////////////////////////////////////
    // Multi-zone functionality

    //! Create a complete multi-zone irrigation system from zone boundary polygons.
    /**
     * \param[in] numZones Number of irrigation zones to create.
     * \param[in] zoneBoundaries Boundary polygon vertices for each zone.
     * \param[in] Pw Water source pressure (psi).
     * \param[in] sprinklerSpacing Spacing between sprinklers along a lateral line (m).
     * \param[in] lineSpacing Spacing between lateral lines (m).
     * \param[in] connectionType "vertical" or "horizontal" lateral orientation.
     * \param[in] sprinklerConfig Name of the sprinkler assembly from the library.
     * \param[in] submainPos Submain placement within each zone.
     */
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

    //! Get node IDs belonging to a specific zone.
    std::vector<int> getNodesByZone(int zoneID) const;
    //! Get link indices belonging to a specific zone.
    std::vector<size_t> getLinksByZone(int zoneID) const;

    //! Open the valve for a specific zone.
    void openZoneValve(int zoneID);
    //! Close the valve for a specific zone.
    void closeZoneValve(int zoneID);
    //! Query whether a zone's valve is currently open.
    bool isZoneValveOpen(int zoneID) const;
    //! Set the valve state for a specific zone.
    void setZoneValveState(int zoneID, bool open);
    //! Open all zone valves.
    void activateAllZones();
    //! Close all zone valves.
    void deactivateAllZones();
    //! Close all valves, then open only the specified zone.
    void activateSingleZone(int zoneID);
    //! Close all valves, then open the specified zones.
    void activateZones(const std::vector<int>& zoneIDs);
    //! Run the multi-zone hydraulic solver.
    /**
     * \param[in] doPreSize If true, runs preSizePipes() before solving.
     * \param[in] sprinklerAssemblyType Sprinkler assembly name (used for minor loss calculations).
     * \param[in] Qspecified Specified emitter flow rate (m3/s).
     * \param[in] Pw Water source pressure (psi).
     * \param[in] V_main Maximum mainline velocity for pipe sizing (m/s).
     * \param[in] V_lateral Maximum lateral velocity for pipe sizing (m/s).
     * \return HydraulicResults struct with pressures, flows, and convergence status.
     */
    HydraulicResults calculateHydraulicsMultiZoneOptimized(
        bool doPreSize,
        const std::string& sprinklerAssemblyType,
        double Qspecified,
        double Pw,
        double V_main,
        double V_lateral);
    //! Build the valve-to-nodes mapping. Must be called after assignZones().
    void initialize();
    //! Choose between fast cached method and tree traversal for active node determination.
    void setActiveNodeMethod(bool useFastMethod);
    //! Check for nodes with unassigned zone IDs and print warnings.
    void checkUnassignedNodes();

    //! Generate system curve data points (GPM vs. head in feet) for pump selection.
    /**
     * \param[in] emitterRequiredPressure Emitter operating pressures to evaluate (psi).
     * \param[in] sprinklerAssemblyType Sprinkler assembly name.
     * \param[in] V_main Maximum mainline velocity for pipe sizing (m/s).
     * \param[in] V_lateral Maximum lateral velocity for pipe sizing (m/s).
     * \param[in] referenceNodeId Optional node ID for analysis (default: -1 for automatic).
     * \param[in] staticHead Optional static head adjustment (default: 0.0).
     * \return Vector of (GPM, head in feet) data points.
     */
    std::vector<std::pair<double, double>> generateSystemCurve(
            const std::vector<double>& emitterRequiredPressure,
            const std::string& sprinklerAssemblyType,
            double V_main,
            double V_lateral,
            int referenceNodeId = -1,
            double staticHead = 0.0);

    //! Write the system curve to a CSV file with a quadratic fit.
    void writeSystemCurveToCsvWithFit(const std::string& filename,
                                      const std::vector<std::pair<double, double>>& curve) const;
    //! Fit a quadratic curve to system curve data.
    static std::tuple<double, double, double> fitCurveQuadratic(
            const std::vector<std::pair<double, double>>& curve);


private:

    int waterSourceId = -1;  // Tracks water source node ID
    int nextNodeId = 1;      // Tracked counter for O(1) node ID generation
    bool message_flag = true; // Controls command-line output messages

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
    double computeLinkFlows(int sourceId,
                        const std::unordered_map<int, std::vector<int>>& adj,
                        std::unordered_map<uint64_t, Link*>& linkMap); // iterative post-order flow accumulation

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
