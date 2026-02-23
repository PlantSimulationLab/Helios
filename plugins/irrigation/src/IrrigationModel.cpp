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
#include <iostream>
#include <algorithm>
#include <cmath>
// #include <Eigen/Dense>  // For matrix operations
#include <unordered_set>
#include <queue>
#include <iomanip>

#include <algorithm>
#include <climits>

#include "global.h"

// Add parameter validation
void IrrigationModel::validateParameters(double fieldLength, double fieldWidth,
                                       double sprinklerSpacing, double lineSpacing) const {
    if (fieldLength <= 0 || fieldWidth <= 0) {
        helios::helios_runtime_error("ERROR (IrrigationModel::validateParameters): Field dimensions must be positive");
    }
    if (sprinklerSpacing <= 0 || lineSpacing <= 0) {
        helios::helios_runtime_error("ERROR (IrrigationModel::validateParameters): Spacing values must be positive");
    }
    if (sprinklerSpacing > fieldLength || lineSpacing > fieldWidth) {
        helios::helios_runtime_error("ERROR (IrrigationModel::validateParameters): Spacing values cannot exceed field dimensions");
    }
}

// Add system summary method
std::string IrrigationModel::getSystemSummary() const {
    std::string summary;
    summary += "\nSystem Summary:\n";
    summary += "Total nodes: " + std::to_string(nodes.size()) + "\n";
    summary += "Total links: " + std::to_string(links.size()) + "\n";

    // Node counts
    std::unordered_map<std::string, int> nodeCounts;
    for (const auto& [id, node] : nodes) {
        nodeCounts[node.type]++;
    }

    summary += "\nNode Types:\n";
    for (const auto& [type, count] : nodeCounts) {
        summary += "- " + type + ": " + std::to_string(count) + "\n";
    }

    // Link counts
    std::unordered_map<std::string, int> linkCounts;
    for (const auto& link : links) {
        linkCounts[link.type]++;
    }

    summary += "\nPipe Types:\n";
    for (const auto& [type, count] : linkCounts) {
        summary += "- " + type + ": " + std::to_string(count) + "\n";
    }

    summary += "-----------------------\n";
    return summary;
}



// Position IrrigationModel::calculateWaterSourcePosition(double fieldLength, double fieldWidth,
//                                                      const std::string& lateralDirection) const {
//     if (lateralDirection == "vertical") {
//         return {
//             //fieldLength / 3.0 + 5.0,  // x
//             18.4112,
//             fieldWidth - fieldWidth / 3.0 - 0.5  // y
//         };
//     } else {
//         return {
//             fieldLength / 2.0,  // x
//             0.0  // y
//         };
//     }
// }



void IrrigationModel::createSprinklerSystemGeneral(double fieldLength, double fieldWidth,
                                          double sprinklerSpacing, double lineSpacing,
                                          const std::string& connectionType,
                                          const std::string& sprinklerConfig, double minX, double minY, int zoneID) {
    std::cout<<"Sprinkler Config: " <<sprinklerConfig<<"\n";
    const SprinklerAssembly& config = sprinklerLibrary.getSprinklerType(sprinklerConfig);
    double stake_height = config.stakeHeight;

    const int num_laterals = static_cast<int>(std::ceil(fieldLength / lineSpacing)) + 1;
    const int num_sprinklers_perline = static_cast<int>(std::ceil(fieldWidth / sprinklerSpacing)) + 1;
    int nodeId = getNextNodeId(); //assign unique node number regardless of zones
  //  int nodeId = 1; //to repeat the same node numbers by different zone number
    const double barbOffset = 0;
    const double emitterOffset = 0;

    // Grid to track valid junction nodes for lateral connections
    std::vector<std::vector<int>> junctionGrid(num_laterals, std::vector<int>(num_sprinklers_perline, -1));

    for (int i = 0; i < num_laterals; ++i) {
        for (int j = 0; j < num_sprinklers_perline; ++j) {
            double x = minX + i * lineSpacing;
            double y = minY + j * sprinklerSpacing;

            Position testPoint{x, y};

            if (!boundaryPolygon.empty() && !isPointInsidePolygon(testPoint)) {
                continue;
            }

            // Create junction node for sprinkler assembly connection
            nodes[nodeId] = {nodeId, "lateral_sprinkler_jn", {x, y}, 0.0, false, 0.0, {}, zoneID};
            int junctionId = nodeId++;
            junctionGrid[i][j] = junctionId;

            // Create barb and emitter nodes (existing code)
            double barbX = x + barbOffset * cos(M_PI/4);
            double barbY = y + barbOffset * sin(M_PI/4);

            nodes[nodeId] = {nodeId, "barb", {barbX, barbY}, 0.0, false, 0.0, {}, zoneID};
            int barbId = nodeId++;

            double emitterX = barbX + emitterOffset * cos(M_PI/4);
            double emitterY = barbY + emitterOffset * sin(M_PI/4);
            nodes[nodeId] = {nodeId, "emitter", {emitterX, emitterY, stake_height}, 0.0, false, 0.0, {},  zoneID};
            int emitterId = nodeId++;

            // Connect barb to emitter
            links.push_back({
                barbId, emitterId,
                config.barbToEmitter.diameter, // 0.154* INCH_TO_METER,
                config.barbToEmitter.length, //30 * INCH_TO_METER,
                "barbToemitter",
                0.0,
                zoneID
            });

            // Connect junction to barb
            links.push_back({
                junctionId, barbId,
                config.lateralToBarb.diameter, //0.12 * INCH_TO_METER,
                config.lateralToBarb.length, // 0.5 * INCH_TO_METER,
                "lateralTobarb",
                0.0,
                zoneID
            });
        }
    }

    // Connect laterals - only connect if both endpoints are valid and inside polygon
    for (int i = 0; i < num_laterals; ++i) {
        for (int j = 0; j < num_sprinklers_perline; ++j) {
            if (junctionGrid[i][j] == -1) continue;

            // Check if neighbor exists and is inside polygon
            if (connectionType == "vertical" && j < num_sprinklers_perline - 1 &&
                junctionGrid[i][j+1] != -1) {
                links.push_back({
                    junctionGrid[i][j], junctionGrid[i][j+1],
                    0.67 * INCH_TO_METER,
                    sprinklerSpacing,
                    "lateral",
                    0.0,
                    zoneID
                });
            }
            else if (connectionType == "horizontal" && i < num_laterals - 1 &&
                     junctionGrid[i+1][j] != -1) {
                links.push_back({
                    junctionGrid[i][j], junctionGrid[i+1][j],
                    0.67 * INCH_TO_METER,
                    sprinklerSpacing,
                    "lateral",
                    0.0,
                    zoneID
                });
            }
        }
    }
    ensureMinimumSprinklersPerRow(sprinklerConfig);

    // Validate the sprinkler system
     validateMinimumSprinklersPerRow();
}


void IrrigationModel::ensureMinimumSprinklersPerRow(const std::string& sprinklerConfig) {
    // arrange sprinkler junctions by lateral row
    std::map<double, std::vector<int>> sprinklersByLateralRow;

    for (const auto& [id, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn" && isPointInsidePolygon(node.position)) {
            double lateralRow = std::round(node.position.x * 100.0) / 100.0;
            sprinklersByLateralRow[lateralRow].push_back(id);
        }
    }

    // add missing sprinkler units to rows with less than 2
    for (auto& [row, sprinklerIds] : sprinklersByLateralRow) {
        if (sprinklerIds.size() < 2) {
            std::cout << "Adding missing sprinkler units to row at x = " << row << std::endl;
            addMissingSprinklerUnits(row, sprinklerIds, 2 - sprinklerIds.size(), sprinklerConfig);
        }
    }
}

void IrrigationModel::addMissingSprinklerUnits(double rowX, const std::vector<int>& existingSprinklers, int count, const std::string& sprinklerConfig) {

    const SprinklerAssembly& config = sprinklerLibrary.getSprinklerType(sprinklerConfig);
    double stake_height = config.stakeHeight;
    if (existingSprinklers.empty()) return;

    // get y of existing sprinklers to determine spacing
    std::vector<double> existingYPositions;
    for (int id : existingSprinklers) {
        existingYPositions.push_back(nodes.at(id).position.y);
    }
    std::sort(existingYPositions.begin(), existingYPositions.end());

    //  optimal positions for new sprinklers
    std::vector<double> newYPositions;

    if (existingYPositions.size() == 1) {
        // Only one sprinkler, add another with typical spacing
        double spacing = 16.0; // Default spacing in feet
        newYPositions.push_back(existingYPositions[0] + spacing);
    } else {

        double maxGap = 0.0;         //  largest gap between existing sprinklers
        size_t gapIndex = 0;

        for (size_t i = 1; i < existingYPositions.size(); ++i) {
            double gap = existingYPositions[i] - existingYPositions[i-1];
            if (gap > maxGap) {
                maxGap = gap;
                gapIndex = i;
            }
        }

        // Add new sprinkler in the largest gap
        if (maxGap > 20.0) { // Only if gap is large enough
            double newY = existingYPositions[gapIndex-1] + (maxGap / 2.0);
            newYPositions.push_back(newY);
        }
    }

    // Create new sprinkler units
    for (double yPos : newYPositions) {
        if (count <= 0) break;

        Position newPos{rowX, yPos};

        int junctionId = getNextNodeId();
        nodes[junctionId] = {junctionId, "lateral_sprinkler_jn", newPos, 0.0, false};

        double barbX = newPos.x + 0.0 * cos(M_PI/4);
        double barbY = newPos.y + 0.0 * sin(M_PI/4);
        int barbId = getNextNodeId();
        nodes[barbId] = {barbId, "barb", {barbX, barbY}, 0.0, false};

        double emitterX = barbX + 0.0 * cos(M_PI/4);
        double emitterY = barbY + 0.0 * sin(M_PI/4);
        int emitterId = getNextNodeId();
        nodes[emitterId] = {emitterId, "emitter", {emitterX, emitterY, stake_height}, 0.0, false};

        // Connect
        links.push_back({junctionId, barbId, 0.12 * INCH_TO_METER, 0.5 * INCH_TO_METER, "lateralTobarb"});
        links.push_back({barbId, emitterId, 0.12 * INCH_TO_METER, 30 * INCH_TO_METER, "barbToemitter"});

        count--;
    }
}


int IrrigationModel::getNextNodeId() const {
    if (nodes.empty()) return 1;
    return std::max_element(nodes.begin(), nodes.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; })->first + 1;
}


std::vector<PressureLossResult> IrrigationModel::getBarbToEmitterPressureLosses() const {


    std::vector<PressureLossResult> results;

    std::vector<std::pair<int, int>> connectedPairs = findConnectedNodePairs("barb", "emitter");
    for (const auto& pair : connectedPairs) {
        int barbNodeId = pair.first;
        int emitterNodeId = pair.second;
        double pressureLoss = std::abs(getPressureDifference(barbNodeId, emitterNodeId));

        results.push_back({barbNodeId, emitterNodeId, pressureLoss});
    }

    return results;
}

std::vector<PressureLossResult> IrrigationModel::getLateralToBarbPressureLosses() const {
    std::vector<PressureLossResult> results;
    std::vector<std::pair<int, int>> connectedPairs = findConnectedNodePairs("lateral_sprinkler_jn", "barb");
    for (const auto& pair : connectedPairs) {
        int barbNodeId = pair.first;
        int emitterNodeId = pair.second;
        double pressureLoss = std::abs(getPressureDifference(barbNodeId, emitterNodeId));

        results.push_back({barbNodeId, emitterNodeId, pressureLoss});
    }

    return results;
}


double IrrigationModel::getPressureDifference(int nodeId1, int nodeId2) const {
    // Check if nodes exist
    if (nodes.find(nodeId1) == nodes.end() || nodes.find(nodeId2) == nodes.end()) {
        helios::helios_runtime_error("ERROR (IrrigationModel::getPressureDifference(): One or both node IDs not found: " +
                                   std::to_string(nodeId1) + ", " + std::to_string(nodeId2));
    }
    double pressure1 = nodes.at(nodeId1).pressure;
    double pressure2 = nodes.at(nodeId2).pressure;

    return pressure1 - pressure2;
}

std::vector<std::pair<int, int>> IrrigationModel::findConnectedNodePairs(const std::string& type1, const std::string& type2) const {
    std::vector<std::pair<int, int>> connectedPairs;

    // iterate through all nodes of type1 and check their neighbors
    for (const auto& [nodeId, node] : nodes) {
        if (node.type == type1) {
            // Check all neighbors of this node
            for (int neighborId : node.neighbors) {
                const Node& neighbor = nodes.at(neighborId);
                if (neighbor.type == type2) {
                    connectedPairs.emplace_back(nodeId, neighborId);
                }
            }
        }
    }

    return connectedPairs;
}

void IrrigationModel::createAssembliesFromIrricad(const std::string& sprinklerAssemblyType) {
    waterSourceId = getMaxNodeId();
    Position waterSourcePos = nodes[waterSourceId].position;
    nodes[waterSourceId] = {
        waterSourceId,
        "waterSource",
        waterSourcePos,
         0.0, //check if the pressure is set correctly for water source
        true
    };

    const SprinklerAssembly& config = sprinklerLibrary.getSprinklerType(sprinklerAssemblyType);
    int nextNodeId = getMaxNodeId() + 1;

    // Group emitters by lateral line Y-coordinate
    std::map<double, std::vector<std::pair<int, Node>>> lateralLines;

    for (const auto& [nodeId, node] : nodes) {
        if (node.type == "emitter") {
            lateralLines[node.position.y].emplace_back(nodeId, node);
        }
    }

    // Store the first junction of each lateral line for water source connection
    std::vector<int> firstLateralJunctions;

    for (auto& [yCoord, emitters] : lateralLines) {
        std::sort(emitters.begin(), emitters.end(),
                 [](const auto& a, const auto& b) {
                     return a.second.position.x < b.second.position.x;
                 });

        // First, create all lateral junctions and barb assemblies for this lateral line
        std::vector<int> lateralJunctionIds;

        for (const auto& [emitterId, emitter] : emitters) {
            int lateralJunctionId = nextNodeId++;
            nodes[lateralJunctionId] = {
                lateralJunctionId,
                "lateral_sprinkler_jn",
                {emitter.position.x, emitter.position.y},
                0.0,
                false
            };
            lateralJunctionIds.push_back(lateralJunctionId);

            int barbId = nextNodeId++;
            nodes[barbId] = {
                barbId,
                "barb",
                {emitter.position.x, emitter.position.y - 0.1},
                0.0,
                false
            };

            // Sprinkler assembly connections
       createVirtualConnection(lateralJunctionId, barbId, config.lateralToBarb,
                             "lateralTobarb", sprinklerAssemblyType);
       createVirtualConnection(barbId, emitterId, config.barbToEmitter,
                             "barbToemitter", sprinklerAssemblyType);
        }

        // create continuous lateral pipe connections between all lateral junctions
        // for (size_t i = 1; i < lateralJunctionIds.size(); i++) {
        //     int prevJunctionId = lateralJunctionIds[i-1];
        //     int currJunctionId = lateralJunctionIds[i];
        //
        //     double distance = nodes[prevJunctionId].position.distanceTo(
        //         nodes[currJunctionId].position);
        //
        //     // createVirtualConnection(prevJunctionId, currJunctionId,
        //     //                       {0.75 * INCH_TO_METER, distance, "PVC"},
        //     //                       "lateral", "");
        //
        //     std::cout << "Created lateral connection: " << prevJunctionId
        //               << " -> " << currJunctionId << " (distance: " << distance << "m)" << std::endl;
        // }

        // Store the first junction for water source connection
        // if (!lateralJunctionIds.empty()) {
        //     firstLateralJunctions.push_back(lateralJunctionIds[0]);
        // }
    }

    recreateLateralConnections();
    // double fieldLength = 0.0;
    // double fieldWidth = 0.0;
    // calculateFieldDimensions(fieldLength, fieldWidth);
    //
    // std::cout << "Calculated field dimensions: " << fieldLength << "m x " << fieldWidth << "m" << std::endl;
    //
    // // Add water source to the system
    // addWaterSource(fieldLength, fieldWidth, "vertical", SubmainPosition::MIDDLE);

    validateHydraulicSystem();
}

// void IrrigationModel::calculateFieldDimensions(double& length, double& width) {
//     if (nodes.empty()) {
//         length = 0.0;
//         width = 0.0;
//         return;
//     }
//
//     double minX = std::numeric_limits<double>::max();
//     double maxX = std::numeric_limits<double>::lowest();
//     double minY = std::numeric_limits<double>::max();
//     double maxY = std::numeric_limits<double>::lowest();
//
//     // Calculate bounding box from all nodes
//     for (const auto& [nodeId, node] : nodes) {
//         minX = std::min(minX, node.position.x);
//         maxX = std::max(maxX, node.position.x);
//         minY = std::min(minY, node.position.y);
//         maxY = std::max(maxY, node.position.y);
//     }
//
//     length = maxX - minX;
//     width = maxY - minY;
//
//     std::cout << "Field bounds: X[" << minX << " - " << maxX << "], Y["
//               << minY << " - " << maxY << "]" << std::endl;
//     std::cout << "Calculated dimensions: " << length << "m x " << width << "m" << std::endl;
// }

// void IrrigationModel::addWaterSource(double fieldLength, double fieldWidth,
//                                    const std::string& lateralDirection,
//                                    SubmainPosition submainPosition) {
//
//     // Get the next available node ID (this will be the last node)
//     int waterSourceId = getNextNodeId();
//     Position waterSourcePos;
//
//     // Calculate submain position based on field dimensions
//     double submainY = 0.0;
//     double submainX = 0.0;
//
//     if (lateralDirection == "vertical") {
//         // Position logic for vertical laterals
//         waterSourcePos = {fieldLength / 2.0 - 2.0, fieldWidth - fieldWidth / 3.0 + 2.0};
//
//         // Adjust based on submain position
//         switch (submainPosition) {
//             case SubmainPosition::NORTH:
//                 waterSourcePos.y = fieldWidth - 1.0;
//                 submainY = fieldWidth;
//                 break;
//             case SubmainPosition::SOUTH:
//                 waterSourcePos.y = 1.0;
//                 submainY = 0.0;
//                 break;
//             case SubmainPosition::MIDDLE:
//                 waterSourcePos.y = fieldWidth - fieldWidth / 3.0 + 2.0;
//                 submainY = fieldWidth - fieldWidth / 3.0 - 0.5;
//                 break;
//         }
//     } else {
//         // Position for horizontal laterals
//         waterSourcePos = {fieldLength / 2.0, 0.0};
//         submainX = fieldLength / 2.0;
//     }
//
//     // Create the water source node
//     nodes[waterSourceId] = {
//         waterSourceId,
//         "waterSource",
//         waterSourcePos,
//         0.0,
//         true
//     };
//
//     std::cout << "Created water source at ID: " << waterSourceId
//               << " position: (" << waterSourcePos.x << ", " << waterSourcePos.y << ")" << std::endl;
//
//     // Find closest submain LINK to water source
//     int closestLinkIndex = -1;
//     double minDistance = std::numeric_limits<double>::max();
//     Position closestPoint;
//
//     for (size_t i = 0; i < links.size(); ++i) {
//         const auto& link = links[i];
//         if (link.type != "submain") continue;
//
//         const auto& nodeA = nodes[link.from];
//         const auto& nodeB = nodes[link.to];
//
//         // Check if both endpoints are submain_junction nodes
//         if (nodeA.type != "submain_junction" || nodeB.type != "submain_junction") {
//             continue;
//         }
//
//         // Check if water source X position is within this submain segment's range
//         double minX = std::min(nodeA.position.x, nodeB.position.x);
//         double maxX = std::max(nodeA.position.x, nodeB.position.x);
//
//         if (waterSourcePos.x >= minX && waterSourcePos.x <= maxX) {
//             // Calculate closest point on the segment to water source
//             Position vecAB = {nodeB.position.x - nodeA.position.x, nodeB.position.y - nodeA.position.y};
//             Position vecAW = {waterSourcePos.x - nodeA.position.x, waterSourcePos.y - nodeA.position.y};
//             double dotProduct = vecAB.x * vecAW.x + vecAB.y * vecAW.y;
//             double lengthSquared = vecAB.x * vecAB.x + vecAB.y * vecAB.y;
//
//             // Avoid division by zero
//             if (lengthSquared < 1e-12) continue;
//
//             double t = std::max(0.0, std::min(1.0, dotProduct / lengthSquared));
//
//             Position pointOnSegment = {
//                 nodeA.position.x + t * vecAB.x,
//                 nodeA.position.y + t * vecAB.y
//             };
//
//             double dist = waterSourcePos.distanceTo(pointOnSegment);
//             if (dist < minDistance) {
//                 minDistance = dist;
//                 closestLinkIndex = i;
//                 closestPoint = pointOnSegment;
//             }
//         }
//     }
//
//     if (closestLinkIndex != -1) {
//         // Create new junction on the submain
//         int newJunctionId = getNextNodeId();
//         nodes[newJunctionId] = {
//             newJunctionId,
//             "submain_junction",
//             closestPoint,
//             0.0,
//             false
//         };
//
//         // Splitting the closest submain link
//         const Link& oldLink = links[closestLinkIndex];
//         double length1 = nodes[oldLink.from].position.distanceTo(closestPoint);
//         double length2 = nodes[oldLink.to].position.distanceTo(closestPoint);
//
//         links.push_back({
//             oldLink.from,
//             newJunctionId,
//             oldLink.diameter,
//             length1,
//             "submain"
//         });
//         links.push_back({
//             newJunctionId,
//             oldLink.to,
//             oldLink.diameter,
//             length2,
//             "submain"
//         });
//         links.erase(links.begin() + closestLinkIndex);
//
//         // Connect new submain junction to water source
//         double mainlineLength = waterSourcePos.distanceTo(closestPoint);
//         links.push_back({
//             newJunctionId,
//             waterSourceId,
//             5.0 * INCH_TO_METER,
//             mainlineLength,
//             "mainline"
//         });
//
//         std::cout << "Connected water source to submain via new junction " << newJunctionId
//                   << " (mainline length: " << mainlineLength << "m)" << std::endl;
//     } else {
//         // Fallback: connect to closest existing submain junction node
//         int closestSubmainId = -1;
//         double minDist = std::numeric_limits<double>::max();
//         for (const auto& [id, node] : nodes) {
//             if (node.type == "submain_junction") {
//                 double dist = node.position.distanceTo(waterSourcePos);
//                 if (dist < minDist) {
//                     minDist = dist;
//                     closestSubmainId = id;
//                 }
//             }
//         }
//
//         if (closestSubmainId != -1) {
//             // Connect water source to closest existing submain junction
//             links.push_back({
//                 closestSubmainId,
//                 waterSourceId,
//                 5.0 * INCH_TO_METER,
//                 minDist,
//                 "mainline"
//             });
//             std::cout << "Connected water source to existing submain junction " << closestSubmainId
//                       << " (distance: " << minDist << "m)" << std::endl;
//         } else {
//             // Create a new submain junction at calculated position
//             int newSubmainId = getNextNodeId();
//             Position newSubmainPos = (lateralDirection == "vertical") ?
//                 Position{fieldLength / 2.0, submainY} :
//                 Position{submainX, fieldWidth / 2.0};
//
//             nodes[newSubmainId] = {
//                 newSubmainId,
//                 "submain_junction",
//                 newSubmainPos,
//                 0.0,
//                 false
//             };
//
//             // Connect new submain junction to water source
//             double mainlineLength = waterSourcePos.distanceTo(newSubmainPos);
//             links.push_back({
//                 newSubmainId,
//                 waterSourceId,
//                 5.0 * INCH_TO_METER,
//                 mainlineLength,
//                 "mainline"
//             });
//
//             std::cout << "Created new submain junction " << newSubmainId
//                       << " and connected to water source (mainline length: " << mainlineLength << "m)" << std::endl;
//         }
//     }
// }

int IrrigationModel::getMaxNodeId() const {
    int maxId = 0;
    for (const auto& [nodeId, node] : nodes) {
        if (nodeId > maxId) maxId = nodeId;
    }
    return maxId;
}

int IrrigationModel::countNodesByType(const std::string& type) const {
    int count = 0;
    for (const auto& [nodeId, node] : nodes) {
        if (node.type == type) count++;
    }
    return count;
}

void IrrigationModel::createVirtualConnection(int node1, int node2,
                                           const ComponentSpecs& specs,
                                           const std::string& connectionType,
                                           const std::string& assemblyType) {
    links.push_back({
        node1, node2,
        specs.diameter,
        specs.length,
        connectionType,
        true  // virtual flag
    });

    // Update neighbors
    nodes[node1].neighbors.push_back(node2);
    nodes[node2].neighbors.push_back(node1);
}

bool IrrigationModel::connectionExists(int node1, int node2) const {
    for (const auto& link : links) {
        if ((link.from == node1 && link.to == node2) ||
            (link.from == node2 && link.to == node1)) {
            return true;
            }
    }
    return false;
}


/////////////////////////////////
void IrrigationModel::recreateLateralConnections() { //function for re-creating Irricad systems
    // Remove all existing lateral connections
    removeExistingLateralConnections();

    // Create lateral segments between lateral_sprinkler_jn nodes
    createLateralSegments();

    // Create lateral-to-submain connections
    // connectLateralsToSubmain();
    std::vector<int> submainNodeIds = getSubmainNodeIdsFromLinks();
    // for (int i = 177 ; i < 192; i++) {
    //     std::cout << "submainID: " << submainNodeIds[i] << std::endl;
    // }
    connectSubmainToLaterals(submainNodeIds, "vertical", 1);

    std::cout << "Recreated lateral connections successfully." << std::endl;
}

std::vector<int> IrrigationModel::getSubmainNodeIdsFromLinks() {
    std::vector<int> submainNodeIds;
    std::set<int> uniqueNodes; // Use set to avoid duplicates

    // Extract all node IDs from submain links excluding valve nodes
    for (const auto& link : links) {
        if (link.type.find("submain") != std::string::npos ||
            link.type == "submain") {

            // Check if start node is not a valve before adding
            if (nodes.count(link.from) && nodes.at(link.from).type != "waterSource" && nodes.at(link.from).type != "submain_junction" ) { //may need to ignore the valve type also
                uniqueNodes.insert(link.from);
            }

            // Check if end node is not a valve before adding
            if (nodes.count(link.to) && nodes.at(link.to).type != "waterSource" && nodes.at(link.to).type != "submain_junction" ) {
                uniqueNodes.insert(link.to);
            }
            }
    }

    // Convert set to vector
    submainNodeIds.assign(uniqueNodes.begin(), uniqueNodes.end());

    std::cout << "Found " << submainNodeIds.size() << " submain nodes from links (excluding valves)." << std::endl;

    // update submain junction name
    for (int nodeId : submainNodeIds) {
        nodes[nodeId].type = "submain_junction";
        if (nodes.count(nodeId)) {
            const auto& node = nodes.at(nodeId);
            // std::cout << "  Submain node " << nodeId << " (" << node.type
            //           << ") at (" << node.position.x << ", " << node.position.y << ")" << std::endl;
        }
    }

    return submainNodeIds;
}

void IrrigationModel::removeExistingLateralConnections() {
    // Remove links with "lateral" type from the LINKS section
    auto it = links.begin();
    while (it != links.end()) {
        if (it->type.find("lateral") != std::string::npos &&
            it->type != "lateralTobarb" &&
            it->type != "lateralToBarb") {
            std::cout << "Removing lateral connection: " << it->from << " -> " << it->to << std::endl;
            it = links.erase(it);
        } else {
            ++it;
        }
    }

    // remove from node neighbors
    for (auto& [nodeId, node] : nodes) {
        auto neighborIt = node.neighbors.begin();
        while (neighborIt != node.neighbors.end()) {
            // Check if this neighbor connection should be a lateral
            if (shouldRemoveLateralNeighbor(nodeId, *neighborIt)) {
                neighborIt = node.neighbors.erase(neighborIt);
            } else {
                ++neighborIt;
            }
        }
    }
}

bool IrrigationModel::shouldRemoveLateralNeighbor(int nodeId1, int nodeId2) {
    const Node& node1 = nodes.at(nodeId1);
    const Node& node2 = nodes.at(nodeId2);

    // Remove if both are lateral_sprinkler_jn or junction types that form lateral lines
    if (node1.type == "junction" && node2.type == "junction" &&
         std::abs(node1.position.y - node2.position.y) < 0.1) { // Same Y-coordinate line?
        return true;
    }

    return false;
}

void IrrigationModel::createLateralSegments() {
    // Group lateral_sprinkler_jn nodes by x-coordinate (This ONLY works for vertical orientation)
    // change to y-coordinate for horizontal orientation
    std::map<double, std::vector<int>> lateralLines;

    for (const auto& [nodeId, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn" || node.type == "end_junction") {
            lateralLines[node.position.x].push_back(nodeId);
        }
    }

    // Create lateral segments for each lateral line
    for (auto& [yCoord, lateralNodes] : lateralLines) {
        // Sort by X-coordinate
        std::sort(lateralNodes.begin(), lateralNodes.end(),
                 [this](int a, int b) {
                     return nodes[a].position.y < nodes[b].position.y;
                 });

        // Create lateral segments between consecutive lateral_sprinkler_jn nodes
        for (size_t i = 1; i < lateralNodes.size(); i++) {
            int prevNodeId = lateralNodes[i-1];
            int currNodeId = lateralNodes[i];

            double distance = nodes[prevNodeId].position.distanceTo(nodes[currNodeId].position);

            // Create lateral segment
            createLateralConnection(prevNodeId, currNodeId, distance);

            std::cout << "Created lateral segment: " << prevNodeId << " -> " << currNodeId
                      << " (distance: " << distance << "m)" << std::endl;
        }
    }
}

void IrrigationModel::createLateralConnection(int startNode, int endNode, double distance) {
    // Create the lateral connection
    links.push_back({
        startNode,
        endNode,
        0.05,  // diameter from your data
        distance,
        "lateral",
        false
    });

    // Update neighbors
    nodes[startNode].neighbors.push_back(endNode);
    nodes[endNode].neighbors.push_back(startNode);
}
void IrrigationModel::connectLateralsToSubmain() {
    // Find all junction nodes (submain junctions at Y=167.616)
    std::vector<int> submainJunctions;
    for (const auto& [nodeId, node] : nodes) {
        if (node.type == "junction" && std::abs(node.position.y - 167.616) < 0.1) {
            submainJunctions.push_back(nodeId);
        }
    }

    if (submainJunctions.empty()) {
        std::cout << "Warning: No submain junction nodes found at Y=167.616." << std::endl;
        return;
    }

    // Group lateral_sprinkler_jn nodes by Y-coordinate (lateral lines)
    std::map<double, std::vector<int>> lateralLines;
    for (const auto& [nodeId, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn") {
            lateralLines[node.position.y].push_back(nodeId);
        }
    }

    std::cout << "Found " << lateralLines.size() << " lateral lines." << std::endl;

    int connectionsCreated = 0;

    // For each lateral line, find the lateral_sprinkler_jn closest to ANY submain junction
    for (auto& [yCoord, lateralNodes] : lateralLines) {
        if (lateralNodes.empty()) continue;

        // Find the lateral_sprinkler_jn in this line that is closest to any submain junction
        int closestLateralNode = findClosestLateralToSubmain(lateralNodes, submainJunctions);
        int closestSubmainNode = findClosestSubmainNode(closestLateralNode, submainJunctions);

        if (closestLateralNode != -1 && closestSubmainNode != -1) {
            double distance = nodes[closestLateralNode].position.distanceTo(
                nodes[closestSubmainNode].position);

            // Create lateral-to-submain connection
            createLateralToSubmainConnection(closestLateralNode, closestSubmainNode, distance);
            connectionsCreated++;

            std::cout << "Connected lateral line (Y=" << yCoord << ") - lateral " << closestLateralNode
                      << " to submain " << closestSubmainNode << " (distance: " << distance << "m)" << std::endl;
        }
    }

    std::cout << "Created " << connectionsCreated << " lateral-to-submain connections." << std::endl;
}

int IrrigationModel::findClosestLateralToSubmain(const std::vector<int>& lateralNodes,
                                               const std::vector<int>& submainNodes) {
    if (lateralNodes.empty() || submainNodes.empty()) return -1;

    int closestLateral = lateralNodes[0];
    double minDistance = std::numeric_limits<double>::max();

    // Find which lateral node in this line is closest to ANY submain node
    for (int lateralNode : lateralNodes) {
        for (int submainNode : submainNodes) {
            double distance = nodes[lateralNode].position.distanceTo(nodes[submainNode].position);
            if (distance < minDistance) {
                minDistance = distance;
                closestLateral = lateralNode;
            }
        }
    }

    return closestLateral;
}

int IrrigationModel::findClosestSubmainNode(int lateralNode, const std::vector<int>& submainNodes) {
    if (submainNodes.empty()) return -1;

    int closestSubmain = submainNodes[0];
    double minDistance = std::numeric_limits<double>::max();

    for (int submainNode : submainNodes) {
        double distance = nodes[lateralNode].position.distanceTo(nodes[submainNode].position);
        if (distance < minDistance) {
            minDistance = distance;
            closestSubmain = submainNode;
        }
    }

    return closestSubmain;
}

void IrrigationModel::createLateralToSubmainConnection(int lateralNode, int submainNode, double distance) {
    // Check if connection already exists
    if (connectionExists(lateralNode, submainNode)) {
        std::cout << "Connection already exists between lateral " << lateralNode
                  << " and submain " << submainNode << std::endl;
        return;
    }

    // Create lateral-to-submain connection
    links.push_back({
        lateralNode,
        submainNode,
        0.75 * INCH_TO_METER,  // diameter from your data
        distance,
        "lateralTosubmain",
        false
    });

    // Update neighbors
    nodes[lateralNode].neighbors.push_back(submainNode);
    nodes[submainNode].neighbors.push_back(lateralNode);
}



/////////////////////////////////

void IrrigationModel::printPressureLossAnalysis(const IrrigationModel& system) {
    // Barb to Emitter analysis
    auto barbEmitterPairs = system.findConnectedNodePairs("barb", "emitter");

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "BARB TO EMITTER PRESSURE LOSS ANALYSIS\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Total connections found: " << barbEmitterPairs.size() << "\n\n";

    if (barbEmitterPairs.empty()) {
        std::cout << "No barb-to-emitter connections found in the system.\n";
        return;
    }

    std::vector<double> losses;
    int outOfRangeCount = 0;

    std::cout << "DETAILED CONNECTION ANALYSIS:\n";
    std::cout << std::string(60, '-') << "\n";

    for (size_t i = 0; i < barbEmitterPairs.size(); ++i) {
        int barbId = barbEmitterPairs[i].first;
        int emitterId = barbEmitterPairs[i].second;
        double loss = std::abs(system.getPressureDifference(barbId, emitterId));
        losses.push_back(loss);

        // Get node details
        const auto& barbNode = system.nodes.at(barbId);
        const auto& emitterNode = system.nodes.at(emitterId);

        std::cout << "Connection " << (i + 1) << ":\n";
        std::cout << "  Barb Node " << barbId << " → Emitter Node " << emitterId << "\n";
        std::cout << "  Pressure Loss: " << loss << " psi";

        // Range validation
        bool inRange = (loss >= 0.8 && loss <= 1.5);
        if (!inRange) {
            std::cout << " [OUT OF RANGE: 0.8-1.5 psi]";
            outOfRangeCount++;
        }
        std::cout << "\n";

        std::cout << "  Barb Pressure: " << barbNode.pressure << " psi";
        std::cout << "  |  Emitter Pressure: " << emitterNode.pressure << " psi";
        std::cout << "  |  Delta: " << (barbNode.pressure - emitterNode.pressure) << " psi\n";
        std::cout << "  Positions: Barb(" << barbNode.position.x << "," << barbNode.position.y << ")";
        std::cout << " → Emitter(" << emitterNode.position.x << "," << emitterNode.position.y << ")\n";
        std::cout << std::string(40, '-') << "\n";
    }

    // Statistics
    double minLoss = *std::min_element(losses.begin(), losses.end());
    double maxLoss = *std::max_element(losses.begin(), losses.end());
    double avgLoss = std::accumulate(losses.begin(), losses.end(), 0.0) / losses.size();

    std::cout << "\nSTATISTICAL SUMMARY:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << "Minimum loss: " << minLoss << " psi\n";
    std::cout << "Maximum loss: " << maxLoss << " psi\n";
    std::cout << "Average loss: " << avgLoss << " psi\n";
    std::cout << "Connections within range: " << (losses.size() - outOfRangeCount) << "/" << losses.size() << "\n";
    std::cout << "Out of range: " << outOfRangeCount << "/" << losses.size() << "\n";

    if (avgLoss >= 0.8 && avgLoss <= 1.5 && outOfRangeCount == 0) {
        std::cout << "✓ ALL connections are within expected range (0.8-1.5 psi)!\n";
    } else {
        std::cout << "✗ Some connections are outside expected range!\n";
    }
}


void IrrigationModel::writePressureLossesToFile(const IrrigationModel& system, const std::string& filename) {
    std::ofstream outFile(filename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }

    // Write header with metadata
    outFile << "# Pressure Loss Comparison Data\n";
  //  outFile << "# System: " << system.getSystemSummary() << "\n";
    outFile << "# Format: ConnectionType,FromNode,ToNode,PressureLoss,FromPressure,ToPressure\n\n";

    // Barb to Emitter section
    outFile << "BARB_TO_EMITTER_START\n";
    auto barbEmitterPairs = system.findConnectedNodePairs("barb", "emitter");

    for (const auto& pair : barbEmitterPairs) {
        int barbId = pair.first;
        int emitterId = pair.second;
        double loss = std::abs(system.getPressureDifference(barbId, emitterId));

        const auto& barbNode = system.nodes.at(barbId);
        const auto& emitterNode = system.nodes.at(emitterId);

        outFile << "barbToemitter,"
                << barbId << ","
                << emitterId << ","
                << loss << ","
                << barbNode.pressure << ","
                << emitterNode.pressure << "\n";
    }
    outFile << "BARB_TO_EMITTER_END\n\n";

    // Lateral to Barb section
    outFile << "LATERAL_TO_BARB_START\n";
    auto lateralBarbPairs = system.findConnectedNodePairs("lateral_sprinkler_jn", "barb");

    for (const auto& pair : lateralBarbPairs) {
        int lateralId = pair.first;
        int barbId = pair.second;
        double loss = std::abs(system.getPressureDifference(lateralId, barbId));

        const auto& lateralNode = system.nodes.at(lateralId);
        const auto& barbNode = system.nodes.at(barbId);

        outFile << "lateralTobarb,"
                << lateralId << ","
                << barbId << ","
                << loss << ","
                << lateralNode.pressure << ","
                << barbNode.pressure << "\n";
    }
    outFile << "LATERAL_TO_BARB_END\n";

    outFile.close();
    std::cout << "Pressure losses written to: " << filename << "\n";
}

// Also add for lateral to barb
void printLateralToBarbAnalysis(const IrrigationModel& system) {
    auto lateralBarbPairs = system.findConnectedNodePairs("lateral_sprinkler_jn", "barb");

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "LATERAL TO BARB PRESSURE LOSS ANALYSIS\n";
    std::cout << std::string(60, '=') << "\n";

}



bool IrrigationModel::isPointInsidePolygon(const Position& point) const {
    if (boundaryPolygon.size() < 3) return true;

    bool inside = false;
    size_t n = boundaryPolygon.size();

    for (size_t i = 0, j = n - 1; i < n; j = i++) {
        if (((boundaryPolygon[i].y > point.y) != (boundaryPolygon[j].y > point.y)) &&
            (point.x < (boundaryPolygon[j].x - boundaryPolygon[i].x) *
             (point.y - boundaryPolygon[i].y) /
             (boundaryPolygon[j].y - boundaryPolygon[i].y) + boundaryPolygon[i].x)) {
            inside = !inside;
             }
    }

    return inside;
}


void IrrigationModel::setBoundaryPolygon(const std::vector<Position>& polygon) {
    boundaryPolygon = polygon;

}


const Link* IrrigationModel::findOptimalLateralForConnection(
    const std::map<double, std::vector<size_t>>& rowLaterals, // map of rowKey -> lateral indices
    const Position& submainPos,
    double expectedRowKey,
    const std::string& lateralDirection) const
{
    // Find the row in the map
    auto it = rowLaterals.find(expectedRowKey);
    if (it == rowLaterals.end() || it->second.empty()) {
        std::cout << "No laterals found in row " << expectedRowKey << std::endl;
        return nullptr;
    }

    const std::vector<size_t>& lateralIndices = it->second;
    const Link* optimalLateral = nullptr;
    double bestScore = std::numeric_limits<double>::max();

    // Iterate only over laterals in this row
    for (size_t idx : lateralIndices) {
        const Link& lateral = links[idx];  // access the lateral from main links array

        // Optional to verify lateral is indeed in the correct row
        double lateralRowKey = getLateralRowKey(lateral, lateralDirection);
        if (std::abs(lateralRowKey - expectedRowKey) > 0.1) continue;

        // get optimal connection point using midpoint
        Position connectionPoint = calculateLateralMidpoint(&lateral);

        // get distance to submain node
        double distance = submainPos.distanceTo(connectionPoint);

        if (distance < bestScore) {
            bestScore = distance;
            optimalLateral = &lateral;
        }

        // printing lateral info
        const auto& fromNode = nodes.at(lateral.from);
        const auto& toNode = nodes.at(lateral.to);
        // std::cout << "  Considering lateral " << lateral.from << "-" << lateral.to
        //           << " from (" << fromNode.position.x << "," << fromNode.position.y << ")"
        //           << " to (" << toNode.position.x << "," << toNode.position.y << ")"
        //           << ", Distance: " << distance << std::endl;
    }

    if (optimalLateral) {
        const auto& fromNode = nodes.at(optimalLateral->from);
        const auto& toNode = nodes.at(optimalLateral->to);
        // std::cout << "✅ Selected optimal lateral: " << optimalLateral->from << "-"
        //           << optimalLateral->to
        //           << " from (" << fromNode.position.x << "," << fromNode.position.y << ")"
        //           << " to (" << toNode.position.x << "," << toNode.position.y << ")"
        //           << ", Distance: " << bestScore << std::endl;        // std::cout << "✅ Selected optimal lateral: " << optimalLateral->from << "-"
        //           << optimalLateral->to
        //           << " from (" << fromNode.position.x << "," << fromNode.position.y << ")"
        //           << " to (" << toNode.position.x << "," << toNode.position.y << ")"
        //           << ", Distance: " << bestScore << std::endl;
    }

    return optimalLateral;
}


// double IrrigationModel::calculateLateralLength(const Link* lateral) const {
//     const auto& fromNode = nodes.at(lateral->from);
//     const auto& toNode = nodes.at(lateral->to);
//     return fromNode.position.distanceTo(toNode.position);
// }

Position IrrigationModel::calculateLateralMidpoint(const Link* lateral) const {
    const auto& fromNode = nodes.at(lateral->from);
    const auto& toNode = nodes.at(lateral->to);

    return {
        (fromNode.position.x + toNode.position.x) / 2.0,
        (fromNode.position.y + toNode.position.y) / 2.0
    };
}


Position IrrigationModel::findOptimalConnectionPointOnLateral(const Link* lateral,
                                                            const Position& submainPos) const {
    const auto& fromNode = nodes.at(lateral->from);
    const auto& toNode = nodes.at(lateral->to);

    // find point on lateral that is the closest to the submain
    Position vecAB = {toNode.position.x - fromNode.position.x,
                     toNode.position.y - fromNode.position.y};
    Position vecAP = {submainPos.x - fromNode.position.x,
                     submainPos.y - fromNode.position.y};

    double dotProduct = vecAB.x * vecAP.x + vecAB.y * vecAP.y;
    double lengthSquared = vecAB.x * vecAB.x + vecAB.y * vecAB.y;

    // avoid division by zero
    if (lengthSquared < 1e-12) {
        return fromNode.position; // Lateral has zero length
    }

    double t = std::max(0.0, std::min(1.0, dotProduct / lengthSquared));

    return {
        fromNode.position.x + t * vecAB.x,
        fromNode.position.y + t * vecAB.y
    };
}

// void IrrigationModel::connectSubmainToLaterals(
//     const std::vector<int>& submainNodeIds,
//     const std::string& lateralDirection)
// {
//     //  Group laterals by their row position
//     std::map<double, std::vector<size_t>> lateralsByRow;
//
//     for (size_t i = 0; i < links.size(); ++i) {
//         const auto& link = links[i];
//         if (link.type != "lateral") continue;
//
//         if (!validateLink(link)) {
//             std::cerr << "Warning: Invalid lateral link " << link.from
//                       << "-" << link.to << std::endl;
//             continue;
//         }
//
//         double rowKey = getLateralRowKey(link, lateralDirection);
//         lateralsByRow[rowKey].push_back(i);
//     }
//
//     // Track connected rows
//     std::set<double> connectedRows;
//
//     //  Loop through each submain node
//     for (int submainNodeId : submainNodeIds) {
//         const auto& submainPos = nodes.at(submainNodeId).position;
//
//         for (const auto& [rowKey, lateralIndices] : lateralsByRow) {
//             if (lateralIndices.empty()) continue;
//             if (connectedRows.find(rowKey) != connectedRows.end()) continue;
//
//             std::cout << "\n Processing Row Key: " << rowKey
//                       << " Laterals in this row: " << lateralIndices.size() << std::endl;
//
//             //  Find the optimal lateral in this row
//             const Link* optimalLateral = findOptimalLateralForConnection(
//                 lateralsByRow,
//                 submainPos,
//                 rowKey,
//                 lateralDirection
//             );
//
//             if (!optimalLateral) {
//             //    std::cerr << "No optimal lateral found for row: " << rowKey << std::endl;
//                 helios::helios_runtime_error("ERROR (IrrigationModel::connectSubmainToLaterals) No optimal lateral found for row");
//                 continue;
//             }
//
//             // Copy the lateral before splitting since splitLateralAndConnect
//             // erases from links and would invalidate any pointers into it.
//             Link optimalLateralCopy = *optimalLateral;
//
//             // split lateral and connect to submain
//             splitLateralAndConnect(&optimalLateralCopy, submainNodeId, lateralsByRow, lateralDirection);
//
//             connectedRows.insert(rowKey);  // Mark this row as connected
//
//             const auto& fromNode = nodes.at(optimalLateralCopy.from);
//             const auto& toNode = nodes.at(optimalLateralCopy.to);
//             // std::cout << " Connected lateral " << optimalLateral->from << "-"
//             //           << optimalLateral->to << " to submain node " << submainNodeId << std::endl;
//             break;
//         }
//     }
// }

void IrrigationModel::connectSubmainToLaterals(const std::vector<int>& submainNodeIds, const std::string& lateralDirection, int zoneID)
{
   // std::cout << " DEBUG connectSubmainToLaterals for zone " << zoneID << " " << std::endl;

    // Group laterals by their row position(only original laterals), not lateralToSubmain
    std::map<double, std::vector<size_t>> lateralsByRow;

    for (size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];

        // Only process the original lateral pipes, not lateralToSubmain connections
        if (link.type != "lateral" || link.zoneID != zoneID) continue;

        if (!validateLink(link)) {
            std::cerr << "Warning: Invalid lateral link " << link.from
                      << "-" << link.to << " in zone " << zoneID << std::endl;
            continue;
        }

        double rowKey = getLateralRowKey(link, lateralDirection);
        lateralsByRow[rowKey].push_back(i);
      //  std::cout << "DEBUG - Added ORIGINAL lateral " << i << " to row " << rowKey << std::endl;
    }

    // std::cout << "DEBUG - Zone " << zoneID << " has " << lateralsByRow.size()
    //           << " ORIGINAL lateral rows to connect" << std::endl;

    // Debug: Show only original lateral rows
    // for (const auto& [rowKey, lateralIndices] : lateralsByRow) {
    //     std::cout << "DEBUG - ORIGINAL Row " << rowKey << " has " << lateralIndices.size() << " laterals" << std::endl;
    //     for (size_t idx : lateralIndices) {
    //         const auto& link = links[idx];
    //         std::cout << "    ORIGINAL Lateral " << link.from << "-" << link.to
    //                   << " type: " << link.type << std::endl;
    //     }
    // }

    // Track connected rows
    std::set<double> connectedRows;

    // Loop through each submain node
    for (int submainNodeId : submainNodeIds) {
        // Verify this submain node belongs to the correct zone
        if (!nodes.count(submainNodeId) || nodes.at(submainNodeId).zoneID != zoneID) {
            continue;
        }

        const auto& submainPos = nodes.at(submainNodeId).position;

        // std::cout << "DEBUG - Processing submain node " << submainNodeId
        //           << " at (" << submainPos.x << ", " << submainPos.y << ")" << std::endl;

        for (const auto& [rowKey, lateralIndices] : lateralsByRow) {
            if (lateralIndices.empty()) continue;
            if (connectedRows.find(rowKey) != connectedRows.end()) {
              //  std::cout << "DEBUG - ORIGINAL Row " << rowKey << " already connected" << std::endl;
                continue;
            }

            // std::cout << "DEBUG - Trying to connect ORIGINAL row " << rowKey
            //           << " with " << lateralIndices.size() << " laterals to submain "
            //           << submainNodeId << std::endl;

            // Find the optimal lateral in this row to ensure it only considers original laterals
            const Link* optimalLateral = findOptimalLateralForConnection(
                lateralsByRow,
                submainPos,
                rowKey,
                lateralDirection
            );

            if (!optimalLateral) {
             //   std::cout << "DEBUG - No optimal lateral found for ORIGINAL row " << rowKey << std::endl;
                continue;
            }

            // Copy the lateral before splitting since splitLateralAndConnect
            // erases from links and would invalidate any pointers into it.
            Link optimalLateralCopy = *optimalLateral;

            // Verify this is an ORIGINAL lateral, not a lateralToSubmain
            if (optimalLateralCopy.type != "lateral") {
                // std::cout << "ERROR: Optimal lateral is not an original lateral! Type: "
                //           << optimalLateral->type << std::endl;
                continue;
            }

            // std::cout << "DEBUG - Found optimal ORIGINAL lateral: " << optimalLateral->from
            //           << "-" << optimalLateral->to << std::endl;

            // Split lateral and connect to submain
            splitLateralAndConnect(&optimalLateralCopy, submainNodeId, lateralsByRow, lateralDirection, zoneID);

            connectedRows.insert(rowKey);
            // std::cout << "DEBUG - SUCCESS: Connected ORIGINAL row " << rowKey << " to submain "
            //           << submainNodeId << std::endl;
            break;
        }
    }

    // connection report
    // std::cout << "=== CONNECTION SUMMARY for zone " << zoneID << " ===" << std::endl;
    // std::cout << "Total ORIGINAL lateral rows: " << lateralsByRow.size() << std::endl;
    // std::cout << "Connected ORIGINAL rows: " << connectedRows.size() << std::endl;

    // List connected and unconnected rows
    for (const auto& [rowKey, lateralIndices] : lateralsByRow) {
        if (connectedRows.find(rowKey) != connectedRows.end()) {
         //   std::cout << "  CONNECTED: Row " << rowKey << std::endl;
        } else {
            std::cout << "  UNCONNECTED: Row " << rowKey << std::endl;
        }
    }
}


//nodes in the same x/y are assigned to the same lateral row
double IrrigationModel::getLateralRowKey(const Link& lateral, const std::string& lateralDirection) const {
    const auto& fromNode = nodes.at(lateral.from);
    const auto& toNode = nodes.at(lateral.to);

    if (lateralDirection == "vertical") {
        // Use average X position for better accuracy
        double avgX = (fromNode.position.x + toNode.position.x) / 2.0;
        return std::round(avgX * 100.0) / 100.0;
    } else {
        double avgY = (fromNode.position.y + toNode.position.y) / 2.0;
        return std::round(avgY * 100.0) / 100.0;
    }
}


bool IrrigationModel::validateLink(const Link& link) const {
    return nodes.count(link.from) > 0 && nodes.count(link.to) > 0;
}



// add zoneID for assignZones
void IrrigationModel::splitLateralAndConnect(
    const Link* lateral,
    int submainNodeId,
    std::map<double, std::vector<size_t>>& rowLaterals,
    const std::string& lateralDirection,
    int zoneID)
{
    // std::cout << "splitLateralAndConnect called for lateral "
    //         << lateral->from << "-" << lateral->to << " to submain " << submainNodeId << std::endl;
    // Find the index of the original lateral in links
    size_t lateralIndex = SIZE_MAX;
    for (size_t i = 0; i < links.size(); ++i) {
        const auto& l = links[i];
        if (l.from == lateral->from && l.to == lateral->to &&
            l.diameter == lateral->diameter &&
            std::abs(l.length - lateral->length) < 1e-6)
        {
            lateralIndex = i;
            break;
        }
    }

    if (lateralIndex == SIZE_MAX) {
        std::cerr << "ERROR: Could not find the original lateral to split: "
                  << lateral->from << "-" << lateral->to << " in zone " << zoneID << std::endl;
        return;
    }

    Link originalLateral = links[lateralIndex];
    const auto& fromNode = nodes.at(originalLateral.from);
    const auto& toNode = nodes.at(originalLateral.to);
    const auto& submainPos = nodes.at(submainNodeId).position;

    // Remove lateral index from rowLaterals
    double originalRowKey = getLateralRowKey(originalLateral, lateralDirection);
    auto& rowVector = rowLaterals[originalRowKey];
    rowVector.erase(std::remove(rowVector.begin(), rowVector.end(), lateralIndex), rowVector.end());

    // Now erase the original lateral using indexing
    links.erase(links.begin() + lateralIndex);

    // Adjust rowLaterals indices for all remaining laterals
    for (auto& [key, vec] : rowLaterals) {
        for (auto& idx : vec) {
            if (idx > lateralIndex) idx--;
        }
    }

    // Calculate lengths of new segments
    double length1 = fromNode.position.distanceTo(submainPos);
    double length2 = submainPos.distanceTo(toNode.position);
    // Reconnect the two new lateral segments WITH ZONE ID
    links.push_back({
        originalLateral.from,
        submainNodeId,
        originalLateral.diameter,
        length1,
        "lateralToSubmain",
        0.0,
        zoneID  // Set zone ID
    });

    links.push_back({
        submainNodeId,
        originalLateral.to,
        originalLateral.diameter,
        length2,
        "lateralToSubmain",
        0.0,
        zoneID  // Set zone ID
    });

    // Update rowLaterals with new segment indices
    size_t indexNew1 = links.size() - 2;
    size_t indexNew2 = links.size() - 1;

    double rowKey1 = getLateralRowKey(links[indexNew1], lateralDirection);
    double rowKey2 = getLateralRowKey(links[indexNew2], lateralDirection);

    rowLaterals[rowKey1].push_back(indexNew1);
    rowLaterals[rowKey2].push_back(indexNew2);

    // std::cout << "DEBUG - Split lateral " << originalLateral.from << "-" << originalLateral.to
    //           << " in zone " << zoneID << " into " << originalLateral.from << "-" << submainNodeId
    //           << " and " << submainNodeId << "-" << originalLateral.to << std::endl;
}


void IrrigationModel::validateSubmainConnections(const std::vector<int>& submainNodeIds) const {
    std::set<int> connectedLaterals;

    // Find all connected laterals
    for (const auto& link : links) {
        if (link.type == "lateralToSubmain") {
            // Trace back to find which lateral this connection serves
            for (const auto& lateralLink : links) {
                if (lateralLink.from == link.to || lateralLink.to == link.to) {
                    connectedLaterals.insert(lateralLink.from);
                    connectedLaterals.insert(lateralLink.to);
                }
            }
        }
    }

    // Check for unconnected laterals
    for (const auto& link : links) {
        if (link.type == "lateral") {
            if (connectedLaterals.find(link.from) == connectedLaterals.end() &&
                connectedLaterals.find(link.to) == connectedLaterals.end()) {
                std::cerr << "Warning: Lateral " << link.from << "-" << link.to
                          << " is not connected to submain" << std::endl;
                }
        }
    }
}


std::vector<int> IrrigationModel::createHorizontalSubmain(const Position& startPos, double fieldLength, const std::vector<Position>& boundary, double lineSpacing, int zoneID) {
    std::vector<int> submainNodeIds;

    // Calculate the actual X bounds from the boundary
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();

    for (const auto& point : boundary) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
    }

    // For horizontal submain, create nodes along the Y-position of startPos, but spanning the X-range of the boundary

    // Calculate number of nodes based on field length and typical spacing
    int numNodes = std::max(2, static_cast<int>(fieldLength/ lineSpacing) + 1);
    //std::cout << "Creating " << numNodes << " submain nodes" << std::endl;

    for (int i = 0; i < numNodes; ++i) {
        Position nodePos;

        // Position along X-axis from minX to maxX
        nodePos.x = minX + i * (lineSpacing );

        // Y-position from startPos for submain line
        nodePos.y = startPos.y;

        // Z-position
        nodePos.z = startPos.z;

       // std::cout << "Creating submain node at (" << nodePos.x << ", " << nodePos.y << ")" << std::endl;

        int nodeId = getNextNodeId();
        nodes[nodeId] = {
            nodeId,
            "submain_junction",
            nodePos,
            0.0,
            false,
            0.0,
            {}, // neighbors
            zoneID
        };
        submainNodeIds.push_back(nodeId);

        // Connect to previous submain node
        if (i > 0) {
            double length = nodes[submainNodeIds[i-1]].position.distanceTo(nodePos);
            links.push_back({
                submainNodeIds[i-1],
                nodeId,
                3.0 * INCH_TO_METER,
                length,
                "submain",
                0.0,
                zoneID
            });
            // std::cout << " Connected node " << submainNodeIds[i-1] << " to " << nodeId
            //           << " (length: " << length << "m)" << std::endl;
        }
    }

    // std::cout << "Successfully created horizontal submain with " << submainNodeIds.size()
    //           << " nodes from X=" << minX << " to X=" << maxX << std::endl;
    /////           Print all created node positions
    // std::cout << "Submain node positions:" << std::endl;
    // for (int nodeId : submainNodeIds) {
    //     const auto& node = nodes[nodeId];
    //     std::cout << "  Node " << nodeId << ": (" << node.position.x << ", " << node.position.y << ")" << std::endl;
    // }

    return submainNodeIds;
}


std::vector<int> IrrigationModel::createVerticalSubmain(const Position& startPos, double fieldWidth,
                                     const std::vector<Position>& boundary, double lineSpacing, int zoneID) {
    std::vector<int> submainNodeIds;

    // Calculate Y bounds from the zone boundary
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& point : boundary) {
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    if (!std::isfinite(minY) || !std::isfinite(maxY) || minY >= maxY) {
        return submainNodeIds;
    }

    // check if spacing doesnt divide evenly, it may go beyond the boundary point
    int numNodes = std::max(2, static_cast<int>(fieldWidth / lineSpacing) + 1);

    for (int i = 0; i < numNodes; ++i) {
        Position nodePos;
        nodePos.x = startPos.x;
        nodePos.y = minY + i * lineSpacing;
        nodePos.z = startPos.z;

        int nodeId = getNextNodeId();
        nodes[nodeId] = {
            nodeId,
            "submain_junction",
            nodePos,
            0.0,
            false,
            0.0,
            {}, // neighbors
            zoneID
        };
        submainNodeIds.push_back(nodeId);

        if (i > 0) {
            double length = nodes[submainNodeIds[i - 1]].position.distanceTo(nodePos);
            links.push_back({
                submainNodeIds[i - 1],
                nodeId,
                3.0 * INCH_TO_METER,
                length,
                "submain",
                0.0,
                zoneID
            });
        }
    }

    return submainNodeIds;
}



void IrrigationModel::validateSubmainConnectivity() const {
    if (boundaryPolygon.empty()) return;

    // Check that every lateral row has at least one connection to submain
    std::set<double> lateralRows;
    std::set<double> connectedRows;

    // Identify all lateral rows
    for (const auto& link : links) {
        if (link.type == "lateral") {
            const auto& fromNode = nodes.at(link.from);
            double lateralRow = std::round(fromNode.position.x * 100.0) / 100.0;
            lateralRows.insert(lateralRow);
        }
    }

    // Identify connected rows
    for (const auto& link : links) {
        if (link.type == "submainToLateral") {
            // Find which lateral row this connection serves
            const auto& connectionNode = nodes.at(link.to);
            if (connectionNode.type == "submain_junction") {
                // Find the lateral connected to this junction
                for (const auto& lateralLink : links) {
                    if ((lateralLink.from == link.to || lateralLink.to == link.to) &&
                        lateralLink.type == "lateral") {
                        const auto& lateralNode = nodes.at(
                            lateralLink.from == link.to ? lateralLink.to : lateralLink.from
                        );
                        double lateralRow = std::round(lateralNode.position.x * 100.0) / 100.0;
                        connectedRows.insert(lateralRow);
                        break;
                    }
                }
            }
        }
    }

    // Report unconnected rows
    std::set<double> unconnectedRows;
    std::set_difference(lateralRows.begin(), lateralRows.end(),
                       connectedRows.begin(), connectedRows.end(),
                       std::inserter(unconnectedRows, unconnectedRows.begin()));

    if (!unconnectedRows.empty()) {
        std::cerr << "Warning: " << unconnectedRows.size()
                  << " lateral rows are not connected to submain:" << std::endl;
        for (double row : unconnectedRows) {
            std::cerr << "  - Lateral at x = " << row << std::endl;
        }
    }
}

Position IrrigationModel::calculateOptimalSubmainPosition(SubmainPosition position,
                                                        double fieldLength, double fieldWidth) const {
    // Find all lateral rows (for vertical laterals)
    std::set<double> rowCenters;

    for (const auto& [id, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn" && isPointInsidePolygon(node.position)) {
            // For vertical laterals, group by x-coordinate (rows)
            rowCenters.insert(node.position.x);
        }
    }

    if (rowCenters.empty()) {
        //polygon-based placement
        return calculatePolygonBasedSubmainPosition(position, fieldLength, fieldWidth);
    }

    // Calculate average row position
    double avgRowPosition = 0.0;
    for (double row : rowCenters) {
        avgRowPosition += row;
    }
    avgRowPosition /= rowCenters.size();

    // gets Y-position based on requested placement
    double yPos;
    switch (position) {
        case SubmainPosition::NORTH:
            yPos = fieldWidth + 2.0;
            break;
        case SubmainPosition::SOUTH:
            yPos = -2.0;
            break;
        case SubmainPosition::MIDDLE:
        default:
            // Find average Y position of sprinklers
            double avgY = 0.0;
            int count = 0;
            for (const auto& [id, node] : nodes) {
                if (node.type == "lateral_sprinkler_jn" && isPointInsidePolygon(node.position)) {
                    avgY += node.position.y;
                    count++;
                }
            }
            if (count > 0) {
                yPos = avgY / count; // Offset above average Y
            } else {
                yPos = fieldWidth / 2.0; // can add offset 2.0
            }
            break;
    }

    return {avgRowPosition, yPos};
}

Position IrrigationModel::calculatePolygonBasedSubmainPosition(SubmainPosition position,
                                                             double fieldLength, double fieldWidth) const {
    // Find the longest edge of the polygon (original logic as fallback)
    double maxLength = 0;
    Position bestPosition;
    size_t n = boundaryPolygon.size();

    for (size_t i = 0; i < n; i++) {
        size_t j = (i + 1) % n;
        const Position& p1 = boundaryPolygon[i];
        const Position& p2 = boundaryPolygon[j];

        double length = p1.distanceTo(p2);
        if (length > maxLength) {
            maxLength = length;
            Position midpoint = {(p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0};

            double dx = p2.x - p1.x;
            double dy = p2.y - p1.y;
            double norm = std::sqrt(dx*dx + dy*dy);

            if (norm > 1e-12) {
                double perpX = -dy / norm;
                double perpY = dx / norm;
                bestPosition = {midpoint.x + perpX * 2.0, midpoint.y + perpY * 2.0};
            }
        }
    }

    // Adjust based on requested position
    switch (position) {
        case SubmainPosition::NORTH:
            return {bestPosition.x, fieldWidth + 2.0};
        case SubmainPosition::SOUTH:
            return {bestPosition.x, -2.0};
        case SubmainPosition::MIDDLE:
        default:
            return bestPosition;
    }
}

void IrrigationModel::ensureAllRowsConnected() {
    // Find all lateral rows
    std::set<double> lateralRows;
    for (const auto& link : links) {
        if (link.type == "lateral") {
            const auto& fromNode = nodes.at(link.from);
            lateralRows.insert(std::round(fromNode.position.x * 100.0) / 100.0);
        }
    }

    // Find connected rows
    std::set<double> connectedRows;
    for (const auto& link : links) {
        if (link.type == "submainToLateral") {
            const auto& connectionNode = nodes.at(link.to);
            if (connectionNode.type == "submain_junction") {
                for (const auto& lateralLink : links) {
                    if ((lateralLink.from == link.to || lateralLink.to == link.to) &&
                        lateralLink.type == "lateral") {
                        const auto& lateralNode = nodes.at(
                            lateralLink.from == link.to ? lateralLink.to : lateralLink.from
                        );
                        connectedRows.insert(std::round(lateralNode.position.x * 100.0) / 100.0);
                    }
                }
            }
        }
    }

    // Connect unconnected rows
    for (double row : lateralRows) {
        if (connectedRows.find(row) != connectedRows.end()) continue;

        // Find a lateral in this row
        const Link* targetLateral = nullptr;
        for (const auto& link : links) {
            if (link.type == "lateral") {
                const auto& fromNode = nodes.at(link.from);
                if (std::abs(fromNode.position.x - row) < 0.1) {
                    targetLateral = &link;
                    break;
                }
            }
        }

        if (targetLateral == nullptr) continue;

        // Find nearest submain node
        int nearestSubmainId = -1;
        double minDistance = std::numeric_limits<double>::max();
        Position lateralMidpoint = calculateLateralMidpoint(targetLateral);

        for (const auto& [id, node] : nodes) {
            if (node.type == "submain_junction") {
                double distance = lateralMidpoint.distanceTo(node.position);
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestSubmainId = id;
                }
            }
        }

        if (nearestSubmainId != -1) {
//            connectSubmainToLateral(nearestSubmainId, targetLateral);
            std::cout << "Added connection for unconnected row at x = " << row << std::endl;
        }
    }
}


void IrrigationModel::createAndConnectWaterSource(const std::vector<int>& submainNodeIds) {
    if (submainNodeIds.empty()) return;

    // Place water source at the beginning or end of submain based on proximity to boundary
    Position waterSourcePos;

    // Check which end of submain is closer to the polygon boundary
    Position firstNodePos = nodes[submainNodeIds.front()].position;
    Position lastNodePos = nodes[submainNodeIds.back()].position;

    double firstDistToBoundary = distanceToPolygon(firstNodePos);
    double lastDistToBoundary = distanceToPolygon(lastNodePos);

    if (firstDistToBoundary < lastDistToBoundary) {
        // Place near first node
        waterSourcePos = firstNodePos;
        waterSourcePos.x -= 5.0; // Offset 5m to be adjusted
    } else {
        // Place near last node
        waterSourcePos = lastNodePos;
        waterSourcePos.x += 5.0; //need to be adjusted
    }

    waterSourceId = getNextNodeId();
    nodes[waterSourceId] = {
        waterSourceId, "waterSource", waterSourcePos, 0.0, true
    };

    // Connect to nearest submain node
    double minDistance = std::numeric_limits<double>::max();
    int nearestSubmainId = -1;

    for (int submainId : submainNodeIds) {
        double distance = waterSourcePos.distanceTo(nodes[submainId].position);
        if (distance < minDistance) {
            minDistance = distance;
            nearestSubmainId = submainId;
        }
    }

    if (nearestSubmainId != -1) {
        links.push_back({
            waterSourceId, nearestSubmainId,
            5.0 * INCH_TO_METER,
            minDistance,
            "mainline"
        });
    }
}

double IrrigationModel::distanceToPolygon(const Position& point) const {
    if (boundaryPolygon.empty()) return 0.0;

    double minDistance = std::numeric_limits<double>::max();
    size_t n = boundaryPolygon.size();

    for (size_t i = 0; i < n; i++) {
        size_t j = (i + 1) % n;
        const Position& p1 = boundaryPolygon[i];
        const Position& p2 = boundaryPolygon[j];

        // Calculate distance from point to line segment
        double distance = pointToSegmentDistance(point, p1, p2);
        if (distance < minDistance) {
            minDistance = distance;
        }
    }

    return minDistance;
}

double IrrigationModel::pointToSegmentDistance(const Position& point,
                                             const Position& segStart,
                                             const Position& segEnd) const {
    Position segVec = {segEnd.x - segStart.x, segEnd.y - segStart.y};
    Position pointVec = {point.x - segStart.x, point.y - segStart.y};

    double segLengthSquared = segVec.x * segVec.x + segVec.y * segVec.y;
    if (segLengthSquared == 0.0) {
        return std::sqrt(pointVec.x * pointVec.x + pointVec.y * pointVec.y);
    }

    double t = std::max(0.0, std::min(1.0,
        (pointVec.x * segVec.x + pointVec.y * segVec.y) / segLengthSquared));

    Position projection = {
        segStart.x + t * segVec.x,
        segStart.y + t * segVec.y
    };

    return std::sqrt(
        (point.x - projection.x) * (point.x - projection.x) +
        (point.y - projection.y) * (point.y - projection.y)
    );
}


// Position IrrigationModel::calculateOptimalWaterSourcePosition(const std::vector<int>& submainNodeIds) const {
//     if (submainNodeIds.empty()) return {0, 0};
//
//     // Strategy 1: Find the submain node closest to polygon boundary for easy access
//     int bestSubmainId = -1;
//     double minBoundaryDistance = std::numeric_limits<double>::max();
//
//     for (int submainId : submainNodeIds) {
//         double distance = distanceToPolygon(nodes.at(submainId).position);
//         if (distance < minBoundaryDistance) {
//             minBoundaryDistance = distance;
//             bestSubmainId = submainId;
//         }
//     }
//
//     // Strategy 2: Use the first submain node as fallback
//     if (bestSubmainId == -1) {
//         bestSubmainId = submainNodeIds[0];
//     }
//
//     const auto& submainPos = nodes.at(bestSubmainId).position;
//
//     // Place water source outside the polygon, near the selected submain node
//     Position outwardDir = calculateOutwardDirection(submainPos);
//
//     // Normalize the outward direction
//     double dirLength = std::sqrt(outwardDir.x * outwardDir.x + outwardDir.y * outwardDir.y);
//     if (dirLength > 1e-12) {
//         outwardDir.x /= dirLength;
//         outwardDir.y /= dirLength;
//     } else {
//         outwardDir = {-1.0, 0.0}; // Default: left direction
//     }
//
//     // Place water source 5-10 meters outside the polygon
//     double offsetDistance = 7.5; // meters
//     return {
//         submainPos.x + outwardDir.x * offsetDistance,
//         submainPos.y + outwardDir.y * offsetDistance
//     };
// }


void IrrigationModel::validateMinimumSprinklersPerRow() const {
    if (boundaryPolygon.empty()) return;
    std::map<double, std::vector<int>> sprinklersByLateralRow;

    for (const auto& [id, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn" && isPointInsidePolygon(node.position)) {
            double lateralRow = std::round(node.position.x * 100.0) / 100.0;
            sprinklersByLateralRow[lateralRow].push_back(id);
        }
    }

    // check if each row has at least 2 sprinkler junctions
    for (const auto& [row, sprinklerIds] : sprinklersByLateralRow) {
        if (sprinklerIds.size() < 2) {
            helios::helios_runtime_error("ERROR (IrrigationModel::validateMinimumSprinklersPerRow()) Minimum sprinkler numbers not met");

        }
    }
   validateCompleteSprinklerUnits();     // validate complete sprinkler units (junction, barb & emitter)
}

void IrrigationModel::validateCompleteSprinklerUnits() const {
    int incompleteUnits = 0;

    for (const auto& [id, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn") {
            // Check if this junction has complete sprinkler unit
            if (!hasCompleteSprinklerUnit(id)) {
                incompleteUnits++;
                helios::helios_runtime_error("ERROR (IrrigationModel::validateCompleteSprinklerUnits()) Incomplete sprinkler unit at junction"+ std::to_string(id));
            }
        }
    }

    if (incompleteUnits > 0) {
        helios::helios_runtime_error("ERROR (IrrigationModel::validateCompleteSprinklerUnits()) "+ std::to_string(incompleteUnits)
            + " incomplete sprinkler units found");
    }
}

bool IrrigationModel::hasCompleteSprinklerUnit(int junctionId) const {
    // complete unit: junction -> barb -> emitter
    bool hasBarb = false;
    bool hasEmitter = false;

    // Check connections from this junction
    for (const auto& link : links) {
        if (link.from == junctionId) {
            auto it = nodes.find(link.to);
            if (it != nodes.end()) {
                if (it->second.type == "barb") {
                    hasBarb = true;
                    // Check if barb has emitter connection
                    if (hasEmitterConnection(link.to)) {
                        hasEmitter = true;
                    }
                }
            }
        }
    }

    return hasBarb && hasEmitter;
}

bool IrrigationModel::hasEmitterConnection(int barbId) const {
    for (const auto& link : links) {
        if (link.from == barbId) {
            auto it = nodes.find(link.to);
            if (it != nodes.end() && it->second.type == "emitter") {
                return true;
            }
        }
    }
    return false;
}


void IrrigationModel::printNetwork() const {
    std::cout << "NODES_START\n";
    for (const auto& [id, node] : nodes) {
        std::cout << id << " "
                  << node.position.x << " "
                  << node.position.y << " "
                  << node.type << " "
                  << node.pressure << " "
                  << node.is_fixed << "\n";
    }
    std::cout << "NODES_END\nLINKS_START\n";
    for (const auto& link : links) {
        std::cout << link.from << " "
                  << link.to << " "
                  << link.diameter << " "
                  << link.length << " "
                  << link.type << "\n";
    }
    std::cout << "LINKS_END\n";
}

// it currently selects the pipe size for submain and main
void IrrigationModel::preSizePipes(double V_main, double V_lateral) {
    std::vector<double> availableSizes = {
        0.025, 0.032, 0.040, 0.050, 0.063,
        0.075, 0.090, 0.110, 0.125, 0.140,
        0.160, 0.200, 0.225, 0.250, 0.315, 0.355, 0.400,
        0.450, 0.500, 0.550, 0.600, 0.750, 0.900
    }; // commercially available pipes in meters

    buildNeighborLists();

    //  reachable set using BFS by zone
    int sourceId = getWaterSourceId();
    if (sourceId == -1 || !nodes.count(sourceId)) {

        std::cerr << "Warning: preSizePipes skipped (no valid water source)." << std::endl;
        return;
    }

    std::unordered_set<int> zoneIds;
    for (const auto& [id, node] : nodes) {
        if (node.zoneID > 0) zoneIds.insert(node.zoneID);
    }

    std::unordered_set<int> reachable;
    if (zoneIds.empty()) {
        bfsCollectNodes(sourceId, 0, reachable);
    } else {
        for (int zoneId : zoneIds) {
            bfsCollectNodes(sourceId, zoneId, reachable);
        }
    }

    // Build adjacency only on reachable nodes
    std::unordered_map<int, std::vector<int>> adj;
    for (auto& link : links) {
        link.flow = 0.0;
        if (reachable.count(link.from) == 0 || reachable.count(link.to) == 0) continue;
        adj[link.from].push_back(link.to);
        adj[link.to].push_back(link.from);
    }

    if (adj.empty()) {
        helios::helios_runtime_error("ERROR (IrrigationModel::preSizePipes()) preSizePipes found no reachable active graph");
        return;
    }

    std::unordered_set<int> visited;
    computeLinkFlows(sourceId, -1, adj, visited);
    for (auto& link : links) {
        double Vmax = 0.0;
        const double oldDiameter = link.diameter;

        if (link.type == "mainline_to_zone") {
            Vmax = V_main;      // mainline velocity
        }
        else if (link.type == "submain") {
            Vmax = V_main;      // or  separate V_submain
        }
        // else if (link.type == "lateral") {
        //     Vmax = V_lateral;
        // } //skip re-sizing lateral pipes
        else {
            std::cout << link.type << std::endl;
            continue;  // skip unknown pipe types
        }

        double Q = std::abs(link.flow);
        if (Q <= 0.0 || Vmax <= 0.0) continue;

        double Dreq = std::sqrt(4.0 * Q / (M_PI * Vmax)); //selecting pipe based on maximum allowable velocity

        auto it = std::find_if(availableSizes.begin(), availableSizes.end(),
                               [&](double d) { return d >= Dreq; });
        //look for the smallest available diameter >= to Dreq
        if (it != availableSizes.end()) {
            link.diameter = *it;
        } else {
            link.diameter = availableSizes.back();
        }

        std::cout << "Presize compare [" << link.type << "] "
                  << link.from << "->" << link.to
                  << " flow=" << Q << " m3/s" << " before D=" << oldDiameter << " m" << " after D=" << link.diameter << " m"
                  << std::endl;
    }
}

double IrrigationModel::computeLinkFlows(int nodeId, int parentId,
                        const std::unordered_map<int, std::vector<int>>& adj,
                        std::unordered_set<int>& visited) {
    if (visited.count(nodeId)) return 0.0;
    visited.insert(nodeId);

    double totalFlow = 0.0;

    // If this node is an emitter, add its demand
    if (nodes[nodeId].type == "emitter") {
        totalFlow += nodes[nodeId].flow;  // or currentSources[idx]
    }

    auto itAdj = adj.find(nodeId);
    if (itAdj == adj.end()) return totalFlow;

    // Recurse into neighbors
    for (int childId : itAdj->second) {
        if (childId == parentId) continue;  // avoid backtracking

        double childFlow = computeLinkFlows(childId, nodeId, adj, visited);

        // Assign flow to the link between nodeId (childId)
        Link* lnk = findLink(nodeId, childId);
        if (lnk) lnk->flow = childFlow;

        totalFlow += childFlow;
    }

    return totalFlow;
}


Link* IrrigationModel::findLink(int from, int to) {
    for (auto& l : links) {
        if ((l.from == from && l.to == to) ||
            (l.from == to && l.to == from)) {
            return &l;  // non-const pointer
            }
    }
    return nullptr;
}


// //linear solver without using external package
// old solver for dense matrix

// HydraulicResults IrrigationModel::calculateHydraulics(bool doPreSize, const std::string& sprinklerAssemblyType,
//                                                      double Qspecified, double Pw,
//                                                      double V_main, double V_lateral) {
//
//     //Check for lateral_sprinkler_jn to barb connections
//     std::vector<std::pair<int, int>> lateralToBarbPairs = findConnectedNodePairs("barb", "emitter");
//
//     if (lateralToBarbPairs.empty()) {
//         std::cout << "No lateral_sprinkler_jn to barb connections found. Creating virtual connections..." << std::endl;
//      //   createVirtualConnections(sprinklerAssemblyType);
//         std::cout << "Created virtual connections..." << std::endl;
//     }
//
//     // const SprinklerAssembly & config = sprinklerLibrary.getSprinklerType(sprinklerAssemblyType);
//     // std::string barbType = config.barbType;
//     // std::string nozzleType = config.emitterType;
//
//
//     // Constants
//     const double rho = 997.0;     // kg/m3
//     const double mu = 8.90e-04;   // Pas
//     const double err_max = 1e-3;
//     const int max_iter = 100;
//
//     buildNeighborLists();
//     const int numNodes = nodes.size();
//     if (numNodes == 0) return HydraulicResults();
//
//     // Create ordered list and mapping
//     std::vector<int> orderedNodeIds;
//
//
//     for (const auto& [id, node] : nodes) orderedNodeIds.push_back(id);
//     // check if the nodes need to be ordered
//     std::sort(orderedNodeIds.begin(), orderedNodeIds.end()); // for irregularShape
//
//     std::unordered_map<int, int> nodeIndexMap;
//     for (int i = 0; i < orderedNodeIds.size(); ++i) {
//         nodeIndexMap[orderedNodeIds.at(i)] = i;  // 0-based indexing
//     }
//
//    //  node and index numbers in nodeIndexMap
//     std::cout << "Node Index Map:\n";
//     for (const auto& [id, node] : nodeIndexMap) {
//         std::cout << "Node " << id << " -> index " << node << "\n";
//     }
//
//     int waterSourceId = getWaterSourceId();
//         //orderedNodeIds.back();
//     int waterSourceIndex = nodeIndexMap.at(waterSourceId);     // water source fixed on the last row
//
//     //  matrices initialization
//     std::vector<std::vector<double>> A(numNodes, std::vector<double>(numNodes, 0.0));
//     std::vector<double> RHS(numNodes, 0.0);
//     std::vector<double> nodalPressure(numNodes, 0.0);
//     std::vector<double> nodalPressure_old(numNodes, 0.0);
//
//     //std::cout << "number of nodes " << numNodes << std::endl;
//     // set water source for once
//
//     A[waterSourceIndex][waterSourceIndex] = 1.0;
//     RHS[waterSourceIndex] = Pw * 6894.76;
//
//     // Initialize current sources
//   //  std::vector<double> currentSources(numNodes, 0.0);
//     for (auto& [id, node] : nodes) {
//         int idx = nodeIndexMap[id];
//         if (node.type == "emitter" && idx != waterSourceIndex) {
//          //   currentSources[idx] = Qspecified;
//             node.flow = Qspecified;
//         }
//     }
//
//     // Flow variables
//     std::unordered_map<int, std::vector<double>> Re, W_bar, vol_rate, R, delta_P;
//     for (const auto& [id, node] : nodes) {
//         int size = node.neighbors.size();
//         Re[id] = std::vector<double>(size, 0.0);
//         W_bar[id] = std::vector<double>(size, 0.0);
//         vol_rate[id] = std::vector<double>(size, 0.0);
//         R[id] = std::vector<double>(size, 0.0);
//         delta_P[id] = std::vector<double>(size, 0.0);
//     }
//
//     double err = 1e6;
//     int iter = 1;
//
//     while (std::abs(err) > err_max && iter < max_iter) {
//         std::cout << "\n=== Iteration " << iter << " ===\n";
//
//         // Build matrix A (water source row remains A[ws][ws] = 1.0)
//         for (int orderedIndex = 0; orderedIndex < orderedNodeIds.size(); ++orderedIndex) {
//             int id = orderedNodeIds.at(orderedIndex);
//             const auto& node = nodes[id];
//
//             std::cout << "\n orderedIndex" << orderedIndex << " id " << id << "\n";
//
//            // skipping isolated nodes, starts with node 0
//             if (node.neighbors.empty()) {
//                 A.at(orderedIndex).at(orderedIndex) = 1.0;  // Set diagonal to 1
//                // A.at(orderedIndex).at(orderedIndex+1) = -1.0; // diagonal is downstream the previous node
//                 RHS.at(orderedIndex) = 0;              // Set RHS to 0
//                // continue;
//             }
//
//             if (orderedIndex == waterSourceIndex) continue;  // Skip water source
//             std::cout << "\n watersourceId " << waterSourceId << "\n";
//
//             // Reset row
//           //  std::fill(A[orderedIndex].begin(), A[orderedIndex].end(), 0.0);
//             A[orderedIndex][orderedIndex] = 0.0;
//
//
//             for (size_t j = 0; j < node.neighbors.size(); ++j) {
//                 int neighborId = node.neighbors[j];
//                 std::cout << neighborId << "\n";
//
//                 if (!nodeIndexMap.count(neighborId)) continue;
//
//                 int neighborIdx = nodeIndexMap[neighborId];
//                 const Link* link = findLink(id, neighborId);
//                 if (!link) continue;
//                 double Kf_barb = minorLoss_kf(Re[id][j], sprinklerAssemblyType);
//                 double Rval = 0.0;
//
//                 Rval = calculateResistance(Re[id][j], W_bar[id][j],Kf_barb, *link, iter);
//                 std::cout << "velocity: "<< W_bar[id][j] << "\n";
//                 std::cout << "other Rval: "<< "Rval " << Rval << "\n";
//
//                 R[id][j] = Rval;
//
//                 //Rval = std::max(Rval, 1e-12);
//
//
//                 A[orderedIndex][orderedIndex] -= 1.0 / Rval;
//                 A[orderedIndex][neighborIdx] = 1.0 / Rval;
//             }
//         }
//
//         // Update RHS with elevation effect
//         for (int n = 0; n < numNodes; ++n) {
//             if (n == waterSourceIndex) continue;
//
//             int nodeId = orderedNodeIds[n];
//             RHS[n] = nodes[nodeId].flow;
//
//             //    RHS[n] = currentSources[n];
//             std::cout << RHS[n] << "\n";
//             // Add elevation contribution to RHS
//             // For each neighbor connection, add elevation difference term
//             const auto& node = nodes[nodeId];
//             double elevation_contribution = 0.0;
//
//             for (size_t j = 0; j < node.neighbors.size(); ++j) {
//                 int neighborId = node.neighbors[j];
//                 if (!nodeIndexMap.count(neighborId)) continue;
//
//                 const Link* link = findLink(nodeId, neighborId);
//                 if (!link) continue;
//
//                 const auto& neighborNode = nodes[neighborId];
//                 double elevation_diff = node.position.z - neighborNode.position.z;
//
//                 // elevation effect rho*g*h in Pa
//                 double elevation_pressure_diff = rho * 9.81 * elevation_diff;
//
//                 // Distribute elevation contribution
//                 // include rho*g*h in pressure difference calculation?
//                 elevation_contribution += elevation_pressure_diff / R[nodeId][j];
//             }
//
//             RHS[n] += elevation_contribution;
//         }
//
//         //  output
//         std::cout << "Matrix A (water source at " << waterSourceIndex << "):\n";
//         for (int i = 0; i < numNodes; i++) {
//             std::cout << "Row " << i << ": ";
//             for (int j = 0; j < numNodes; j++) {
//                 std::cout << std::setw(12) << std::setprecision(6) << A[i][j] << " ";
//             }
//             std::cout << " | RHS: " << RHS[i] << std::endl;
//         }
//
//         // Solve linear system
//         for (int gs_iter = 0; gs_iter < 100; ++gs_iter) {
//             for (int i = 0; i < numNodes; ++i) {
//                 //if (i == waterSourceIndex) continue;
//
//                //check if row is valid
//                 if (std::abs(A[i][i]) < 1e-12) {
//                     nodalPressure[i] = 0.0;
//                     continue;
//                 }
//
//                 double sum = RHS[i];
//                 for (int j = 0; j < numNodes; ++j) {
//                     if (i != j) sum -= A[i][j] * nodalPressure[j]; // Aij*Xj)
//                 }
//
//                 double new_pressure = sum / A[i][i]; // Xi = (bi - sum( Aij*Xj)) / Aii
//                 if (std::isnan(new_pressure) || std::isinf(new_pressure)) {
//                     std::cerr << "NaN pressure at node " << i << ", using old value\n";
//                     new_pressure = nodalPressure_old[i];
//                 }
//                 nodalPressure[i] = new_pressure;
//             }
//         }
//         // Eigen::MatrixXd A_eigen(numNodes, numNodes);
//         // Eigen::VectorXd RHS_eigen(numNodes);
//         //
//         // // Copy your A and RHS to Eigen matrices
//         // for (int i = 0; i < numNodes; ++i) {
//         //     for (int j = 0; j < numNodes; ++j) {
//         //         A_eigen(i, j) = A[i][j];
//         //     }
//         //     RHS_eigen(i) = RHS[i];
//         // }
//         //
//         // // Solve using Eigen (similar to MATLAB's \)
//         // Eigen::VectorXd nodalPressure_eigen = A_eigen.colPivHouseholderQr().solve(RHS_eigen);
//         //
//         //
//         // for (int i = 0; i < numNodes; ++i) {
//         //     nodalPressure[i] = nodalPressure_eigen(i);
//         // }
//
//         // Update flow variables with safety checks
//         for (const auto& [id, node] : nodes) {
//             int idx = nodeIndexMap[id];
//
//             for (size_t j = 0; j < node.neighbors.size(); ++j) {
//                 int neighborId = node.neighbors[j];
//                 if (!nodeIndexMap.count(neighborId)) continue;
//
//                 int neighborIdx = nodeIndexMap[neighborId];
//                 Link* link = findLink(id, neighborId);
//                 if (!link) continue;
//
//                 delta_P[id][j] = nodalPressure[idx] - nodalPressure[neighborIdx];
//
//                 //  Avoid division by zero
//                 double denominator = R[id][j] * (M_PI/4.0) * pow(link->diameter, 2);
//                 if (std::abs(denominator) < 1e-12) {
//                     W_bar[id][j] = 0.0;
//                     std::cout << "denominator: " << denominator << std::endl;
//                 } else {
//                     W_bar[id][j] = std::abs(delta_P[id][j]) / denominator;
//                     std::cout << "denominator: " << denominator << std::endl;
//
//                     std::cout<<"velocity:" << W_bar[id][j]<<"\n";
//                 }
//
//                 vol_rate[id][j] = W_bar[id][j] * (M_PI/4.0) * pow(link->diameter, 2);
//                 link->flow = vol_rate[id][j];
//                 Re[id][j] = std::abs(W_bar[id][j]) * link->diameter * rho / mu;
//                 if (link->type == "lateralTobarb") {
//                     std::cout <<"flow_rate" << vol_rate[id][j] <<" Re:" << Re[id][j] << "\n";
//
//                 }
//                 // Debug NaN values
//                 if (std::isnan(W_bar[id][j])) {
//                     std::cerr << "NaN W_bar at node " << id << "->" << neighborId
//                               << ": R=" << R[id][j] << ", dP=" << delta_P[id][j] << "\n";
//                 }
//             }
//         }
//
//         // Update emitter flows with safety
//         for (auto& [id, node] : nodes) {
//             int idx = nodeIndexMap[id];
//             if (node.type == "emitter" && idx != waterSourceIndex) {
//                 double new_flow = calculateEmitterFlow(sprinklerAssemblyType, nodalPressure[idx], false);
//                 if (std::isnan(new_flow) || std::isinf(new_flow)) {
//                     std::cerr << "NaN flow at emitter " << id << ", pressure: " << nodalPressure[idx] << "\n";
//                     new_flow = 0.0;
//                 }
//                 node.flow = new_flow;
//               //  currentSources[idx] = new_flow;
//             }
//         }
//
//         // Error calculation
//         if (iter>1) {
//             double norm_diff = 0.0, norm_old = 0.0;
//             for (int i = 0; i < numNodes; ++i) {
//                 if (i == waterSourceIndex) continue;
//                 double diff = nodalPressure[i] - nodalPressure_old[i];
//                 norm_diff += diff * diff;
//                 norm_old += nodalPressure_old[i] * nodalPressure_old[i];
//             }
//
//             err = (norm_old > 1e-12) ? std::sqrt(norm_diff / norm_old) : std::sqrt(norm_diff); //relative error
//            //err = std::sqrt(norm_diff/ numNodes);//RMS error
//         }
//
//         std::cout << "Error: " << err << std::endl;
//
//         if (std::isnan(err)) break;
//         nodalPressure_old = nodalPressure;
//         iter++;
//     }
//
//     // Prepare results
//     HydraulicResults results;
//     results.nodalPressures.resize(numNodes);
//     results.flowRates.resize(links.size());
//     results.converged = (err <= err_max);
//     results.iterations = iter;
//
//     // Convert pressures to psi
//     for (int i = 0; i < numNodes; ++i) {
//         results.nodalPressures[i] = nodalPressure[i] / 6894.76;
//     }
//
//     // update nodal pressure
//     for (const auto& [id, node] : nodes) {
//         if (nodeIndexMap.count(id)) {
//             int matrixIndex = nodeIndexMap[id];
//             nodes[id].pressure = results.nodalPressures[matrixIndex];
//         }
//     }
//
//     // link flows
//     for (size_t i = 0; i < links.size(); ++i) {
//         const auto& link = links[i];
//         int from = link.from;
//         int to = link.to;
//
//         if (!nodeIndexMap.count(from) || !nodeIndexMap.count(to)) {
//             results.flowRates[i] = 0.0;
//             continue;
//         }
//
//         auto it = std::find(nodes[from].neighbors.begin(), nodes[from].neighbors.end(), to);
//         if (it != nodes[from].neighbors.end()) {
//             size_t idx = std::distance(nodes[from].neighbors.begin(), it);
//             results.flowRates[i] = vol_rate[from][idx];
//         } else {
//             results.flowRates[i] = 0.0;
//         }
//     }
//     // Save emitter flows
//     // results.emitterFlows.clear();
//     for (const auto& [id, node] : nodes) {
//         if (node.type == "emitter") {
//             results.emitterFlows[id] = node.flow;
//         }
//     }
//
//     if (!results.converged) {
//       //  std::cerr << "Warning: Solver did not converge after " << iter << " iterations\n";
//         results.converged = false;
//         helios::helios_runtime_error("ERROR (IrrigationModel::calculateHydraulics): Solver did not converge after " + std::to_string(iter) + " iterations\n");
//     }
//
//
//     writeMatrixToFile(A, RHS, orderedNodeIds, "matrix_debug.txt");
//
//     return results;
// }

void IrrigationModel::writeMatrixToFile(const std::vector<std::vector<double>>& A,
                                      const std::vector<double>& RHS,
                                      const std::vector<int>& orderedNodeIds,
                                      const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return;
    }

    const int numNodes = orderedNodeIds.size();

    file << "=== HYDRAULIC SYSTEM MATRIX ===" << std::endl;
    file << "Matrix A (" << numNodes << "x" << numNodes << ") with RHS vector" << std::endl;
    file << "Format: Row_Node -> [Col_Node_Values] | RHS" << std::endl;
    file << "=================================" << std::endl << std::endl;

    // Write column headers
    file << std::setw(12) << "Row\\Col";
    for (int j = 0; j < numNodes; ++j) {
        file << std::setw(12) << orderedNodeIds[j];
    }
    file << " | " << std::setw(12) << "RHS" << std::endl;

    // Write separator line
    file << std::setw(12) << "--------";
    for (int j = 0; j < numNodes; ++j) {
        file << std::setw(12) << "---";
    }
    file << "-+-" << std::setw(12) << "---" << std::endl;

    // Write matrix rows
    for (int i = 0; i < numNodes; ++i) {
        int rowNodeId = orderedNodeIds[i];
        const auto& rowNode = nodes.at(rowNodeId);

        // Row header with node info
        file << std::setw(6) << rowNodeId << "(" << std::setw(8) << rowNode.type << ")";

        // Matrix values
        for (int j = 0; j < numNodes; ++j) {
            file << std::setw(12) << std::scientific << std::setprecision(3) << A[i][j];
        }

        // RHS value
        file << " | " << std::setw(12) << std::scientific << std::setprecision(3) << RHS[i];

        // Additional node info
        file << "  // Neighbors: ";
        for (int neighbor : rowNode.neighbors) {
            file << neighbor << " ";
        }
        if (rowNode.neighbors.empty()) {
            file << "ISOLATED";
        }

        file << std::endl;
    }

    // Add summary information
    file << std::endl << "=== MATRIX SUMMARY ===" << std::endl;
    file << "Total nodes: " << numNodes << std::endl;
    file << "Water source node: " << getWaterSourceId() << std::endl;

    // Count non-zero entries
    int nonZeroCount = 0;
    double maxVal = 0.0, minVal = 0.0;
    for (int i = 0; i < numNodes; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            if (std::abs(A[i][j]) > 1e-12) {
                nonZeroCount++;
                maxVal = std::max(maxVal, std::abs(A[i][j]));
                if (std::abs(A[i][j]) > 1e-12) {
                    minVal = (minVal == 0.0) ? std::abs(A[i][j]) : std::min(minVal, std::abs(A[i][j]));
                }
            }
        }
    }

    file << "Non-zero entries: " << nonZeroCount << "/" << (numNodes * numNodes)
         << " (" << (100.0 * nonZeroCount / (numNodes * numNodes)) << "%)" << std::endl;
    file << "Max |A_ij|: " << maxVal << std::endl;
    file << "Min non-zero |A_ij|: " << minVal << std::endl;

    file.close();
    std::cout << "Matrix written to: " << filename << std::endl;
}


const Link* IrrigationModel::findLink(int from, int to) const {
    for (const auto& l : links) {
        if ((l.from == from && l.to == to) || (l.to == from && l.from == to)) {
            return &l;
        }
    }
    return nullptr;
}

int IrrigationModel::getWaterSourceId() const { return waterSourceId; }

/* functions for delineating irrigation zones
 *
 */

// void IrrigationModel::assignZones(
//     int numZones,
//     const std::vector<std::vector<Position>>& zoneBoundaries,
//     double Pw,
//     double sprinklerSpacing,
//     double lineSpacing,
//     const std::string& connectionType,
//     const std::string& sprinklerConfig,
//     SubmainPosition submainPos)
// {
//     if (zoneBoundaries.size() != static_cast<size_t>(numZones)) {
//         std::cerr << "Error: Number of boundaries (" << zoneBoundaries.size()
//                   << ") does not match numZones (" << numZones << ")." << std::endl;
//         return;
//     }
//
//     // Clear existing system
//     nodes.clear();
//     links.clear();
//     boundaryPolygon.clear();
//     waterSourceId = -1;
//
//     // Create the main water source first (shared by all zones)
//     waterSourceId = getNextNodeId();
//     Position waterSourcePos = {-5.0, 0.0, 0.0}; // You might want to calculate a better position
//
//     nodes[waterSourceId] = {
//         waterSourceId,
//         "waterSource",
//         waterSourcePos,
//         Pw, // Set the water source pressure
//         true
//     };
//
//     // Store base counts before creating zones
//     size_t baseNodeCount = nodes.size();
//     size_t baseLinkCount = links.size();
//
//     for (int z = 0; z < numZones; ++z) {
//         std::cout << "Creating zone " << (z + 1) << "..." << std::endl;
//
//         // Store current counts before creating this zone
//         size_t zoneStartNodeCount = nodes.size();
//         size_t zoneStartLinkCount = links.size();
//
//         // Generate the irrigation system for this zone
//         createIrregularSystemForZone(Pw, zoneBoundaries[z], sprinklerSpacing, lineSpacing,
//                                    connectionType, sprinklerConfig, submainPos, z + 1);
//
//
//
//         // Add zone valve and connect to main water source
//         addZoneValveAndConnect(z + 1, zoneBoundaries[z]);
//     //     std::cout << "Zone " << (z + 1) << " created with "
//     //               << (nodes.size() - zoneStartNodeCount) << " nodes and "
//     //               << (links.size() - zoneStartLinkCount) << " links" << std::endl;
//      }
//
//   //  validateHydraulicSystem();
// }


void IrrigationModel::createIrregularSystemForZone(
    const std::vector<Position>& boundary,
    double sprinklerSpacing,
    double lineSpacing,
    const std::string& connectionType,
    const std::string& sprinklerConfig,
    SubmainPosition submainPos,
    int zoneID)
{
    // Set boundary for this zone
    setBoundaryPolygon(boundary);


    // std::cout << "Boundary points: " << boundary.size() << std::endl;
    // for (const auto& p : boundary) {
    //     std::cout << "  (" << p.x << ", " << p.y << ")" << std::endl;
    // }

    // Validate boundary
    if (boundary.size() < 3) {
        helios::helios_runtime_error("ERROR (IrrigationModel::createIrregularSystemForZone): Invalid boundary size");
    }

    if (connectionType != "vertical" && connectionType != "horizontal") {
        helios::helios_runtime_error("ERROR (IrrigationModel::createIrregularSystemForZone): Invalid lateral connection orientation");
    }

    // Calculate bounding box for grid generation
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& point : boundary) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    double fieldLength = maxX - minX;
    double fieldWidth = maxY - minY;
    // std::cout<<"fieldLength = "<<fieldLength<<std::endl;
    // std::cout<<"fieldWidth = "<<fieldWidth<<std::endl;

    // Create sprinkler system within the boundary
    createSprinklerSystemGeneral(fieldLength, fieldWidth, sprinklerSpacing, lineSpacing,
                               connectionType, sprinklerConfig,
                               minX, minY, zoneID);

    // Add submain for this zone (no water source connection yet)
    addSubmainForZone(fieldLength, fieldWidth, connectionType, submainPos, zoneID, lineSpacing);

    // Validate
    validateParameters(fieldLength, fieldWidth, sprinklerSpacing, lineSpacing);
}

void IrrigationModel::addSubmainForZone(double fieldLength, double fieldWidth,
                                      const std::string& lateralDirection,
                                      SubmainPosition submainPosition,
                                      int zoneID,
                                      double lateralSpacing)
{
    // Calculate the actual bounds from the boundary polygon for this zone
    double minX = std::numeric_limits<double>::max();
    double minY = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& point : boundaryPolygon) {
        minX = std::min(minX, point.x);
        minY = std::min(minY, point.y);
        maxX = std::max(maxX, point.x);
        maxY = std::max(maxY, point.y);
    }

    std::cout << "Zone " << zoneID << " bounds: X[" << minX << " - " << maxX
              << "], Y[" << minY << " - " << maxY << "]" << std::endl;

    // Calculate submain position based on the actual zone bounds
    Position submainLinePos = calculateOptimalSubmainPositionForZone(submainPosition, minX, minY, fieldLength, fieldWidth);

    std::cout << "Calculated submain start position: ("
              << submainLinePos.x << ", " << submainLinePos.y << ")" << std::endl;

    std::vector<int> submainNodeIds;

    if (lateralDirection == "vertical") {
        // Use existing createHorizontalSubmain but pass the correct starting position
        // will create nodes relative to the position we give it
        submainNodeIds = createHorizontalSubmain(submainLinePos, fieldLength, boundaryPolygon, lateralSpacing, zoneID);
    } else {
        submainNodeIds = createVerticalSubmain(submainLinePos, fieldWidth, boundaryPolygon, lateralSpacing, zoneID);
    }

    // Connect laterals to submain within this zone
    connectSubmainToLaterals(submainNodeIds, lateralDirection, zoneID);

    // Assign zone ID to submain nodes
    for (int nodeId : submainNodeIds) {
        if (nodes.count(nodeId)) {
            nodes[nodeId].zoneID = zoneID;
        }
    }


}


Position IrrigationModel::calculateOptimalSubmainPositionForZone(
    SubmainPosition position, double minX, double minY,
    double fieldLength, double fieldWidth)
{
    double submainX = 0.0;
    double submainY = 0.0;

    switch (position) {
        case SubmainPosition::NORTH:
            submainY = minY + fieldWidth + 2.0; // Above the zone
            submainX = minX; //  start at left edge of the zone
            break;
        case SubmainPosition::SOUTH:
            submainY = minY - 2.0; // Below the zone
            submainX = minX; //  left edge
            break;
        case SubmainPosition::MIDDLE:
        default:
            submainY = minY + fieldWidth / 2.0;
            submainX = minX; // left edge
            break;
    }

  //  std::cout << "Submain position for zone: (" << submainX << ", " << submainY << ")" << std::endl;

    return {submainX, submainY};
}

// void IrrigationModel::addZoneValveAndConnect(int zoneID, const std::vector<Position>& zoneBoundary)
// {
//     // Find the submain junction in this zone FIRST
//     int zoneSubmainId = findZoneSubmainJunction(zoneID);
//     std::cout << "zoneSubmainId: " << zoneSubmainId << std::endl;
//
//     if (zoneSubmainId == -1) {
//         std::cerr << "Warning: No submain junction found in zone " << zoneID
//                   << ". Creating one manually." << std::endl;
//         zoneSubmainId = createZoneSubmainJunction(zoneID, zoneBoundary);
//     }
//
//     // Position valve near the submain junction but offset for proper routing
//     Position submainPos = nodes[zoneSubmainId].position;
//     Position valvePos = calculateOptimalValvePosition(submainPos, zoneBoundary);
//
//     std::cout << "Valve position calculated: (" << valvePos.x << ", " << valvePos.y << ")" << std::endl;
//
//     // Create zone valve
//     int valveId = getNextNodeId();
//     std::cout << "Valve ID: " << valveId << std::endl;
//
//     nodes[valveId] = {
//         valveId,
//         "zone_valve",
//         valvePos,
//         0.0, // Pressure will be calculated
//         false,
//         0.0,
//         {},
//         zoneID
//     };
//
//     // Connect valve to zone submain with 90-degree routing if needed
//     connectValveToSubmainWith90Degree(valveId, zoneSubmainId, zoneID);
//
//     // Connect water source to valve with 90-degree routing
//     connectWaterSourceToValveWith90Degree(valveId, zoneID);
//
//     std::cout << "Zone " << zoneID << ": Added valve " << valveId
//               << " connected to submain " << zoneSubmainId
//               << " and water source " << waterSourceId << std::endl;
// }

Position IrrigationModel::calculateOptimalValvePosition(const Position& submainPos,
                                                       const std::vector<Position>& zoneBoundary)
{
    // Place valve near submain but outside the main irrigation area
    // positions should allow straight-line connections

    // zone bounds
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::lowest();
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::lowest();

    for (const auto& point : zoneBoundary) {
        minX = std::min(minX, point.x);
        maxX = std::max(maxX, point.x);
        minY = std::min(minY, point.y);
        maxY = std::max(maxY, point.y);
    }

    // Try different positions around the submain
    std::vector<Position> candidatePositions;

    // Position offsets in meters
    double offset = -(maxX-minX)/2; // offset left/right of the submain

    // Candidate positions: left, right, above, below the submain
 //   candidatePositions.push_back({submainPos.x - offset, submainPos.y, 0.0});      // Left
    candidatePositions.push_back({submainPos.x + offset, submainPos.y, 0.0});      // Right
  //  candidatePositions.push_back({submainPos.x, submainPos.y - offset, 0.0});      // Below
  //  candidatePositions.push_back({submainPos.x, submainPos.y + offset, 0.0});      // Above

    // Also try corners for better routing
 //   candidatePositions.push_back({submainPos.x - offset, submainPos.y - offset, 0.0}); // Bottom-left
    // candidatePositions.push_back({submainPos.x + offset, submainPos.y - offset, 0.0}); // Bottom-right
    // candidatePositions.push_back({submainPos.x - offset, submainPos.y + offset, 0.0}); // Top-left
    // candidatePositions.push_back({submainPos.x + offset, submainPos.y + offset, 0.0}); // Top-right

    // Score candidates based on:
    // 1. Distance to water source (shorter)
    // 2. Clear path to water source (not crossing irrigation area)
    // 3. Position relative to zone boundary (prefer outside or edge)

    Position waterSourcePos = nodes[waterSourceId].position;
    Position bestPosition = candidatePositions[0];
    double bestScore = std::numeric_limits<double>::lowest();

    for (const auto& candidate : candidatePositions) {
        double score = 0.0;

        // Prefer positions that allow straight lines to water source
        bool straightLineX = std::abs(candidate.x - waterSourcePos.x) < 1.0;
        bool straightLineY = std::abs(candidate.y - waterSourcePos.y) < 1.0;

        if (straightLineX || straightLineY) {
            score += 10.0; // Bonus for alignment
        }

        // Prefer positions near zone boundary for easier access
        double distToBoundary = std::min({
            candidate.x - minX, maxX - candidate.x,
            candidate.y - minY, maxY - candidate.y
        });

        if (distToBoundary < 2.0) {
            score += 5.0; // Bonus for boundary proximity
        }

        // Small penalty for distance (we want reasonably close but not necessarily closest)
        double distanceToSubmain = candidate.distanceTo(submainPos);
        score -= distanceToSubmain * 0.1;

        if (score > bestScore) {
            bestScore = score;
            bestPosition = candidate;
        }
    }

    return bestPosition;
}

void IrrigationModel::connectValveToSubmainWith90Degree(int valveId, int submainId, int zoneID)
{
    Position valvePos = nodes[valveId].position;
    Position submainPos = nodes[submainId].position;

    // For direct connection (if aligned)
    if (std::abs(valvePos.x - submainPos.x) < 0.1 || std::abs(valvePos.y - submainPos.y) < 0.1) {
        // Straight line connection
        double length = valvePos.distanceTo(submainPos);
        links.push_back({
            valveId,
            submainId,
            3.0 * INCH_TO_METER,
            length,
            "zone_valve",
            false,
            zoneID
        });

        // Update neighbors
        nodes[valveId].neighbors.push_back(submainId);
        nodes[submainId].neighbors.push_back(valveId);
    } else {
        // Create 90-degree connection with intermediate point
        int intermediateId = getNextNodeId();
        Position intermediatePos;

        // Choose whether to go horizontal then vertical or vertical then horizontal
        // Based on which requires less total length
        double option1Length = std::abs(valvePos.x - submainPos.x) + std::abs(valvePos.y - submainPos.y);
        double option2Length = std::abs(valvePos.y - submainPos.y) + std::abs(valvePos.x - submainPos.x);

        if (option1Length <= option2Length) {
            // Horizontal then vertical
            intermediatePos = {submainPos.x, valvePos.y, 0.0};
        } else {
            // Vertical then horizontal
            intermediatePos = {valvePos.x, submainPos.y, 0.0};
        }

        // Create intermediate node
        nodes[intermediateId] = {
            intermediateId,
            "valve_connection_junction",
            intermediatePos,
            0.0,
            false,
            0.0,
            {},
            zoneID
        };

        // Create first segment (valve to intermediate)
        double length1 = valvePos.distanceTo(intermediatePos);
        links.push_back({
            valveId,
            intermediateId,
            3.0 * INCH_TO_METER,
            length1,
            "zone_valve",
            false,
            zoneID
        });

        // Create second segment (intermediate to submain)
        double length2 = intermediatePos.distanceTo(submainPos);
        links.push_back({
            intermediateId,
            submainId,
            3.0 * INCH_TO_METER,
            length2,
            "zone_valve",
            false,
            zoneID
        });

        // Update neighbors
        nodes[valveId].neighbors.push_back(intermediateId);
        nodes[intermediateId].neighbors.push_back(valveId);
        nodes[intermediateId].neighbors.push_back(submainId);
        nodes[submainId].neighbors.push_back(intermediateId);

        std::cout << "Created 90-degree valve connection via intermediate node " << intermediateId << std::endl;
    }
}

// void IrrigationModel::connectWaterSourceToValveWith90Degree(int valveId, int zoneID)
// {
//     Position valvePos = nodes[valveId].position;
//     Position sourcePos = nodes[waterSourceId].position;
//
//     // For direct connection (if aligned)
//     if (std::abs(valvePos.x - sourcePos.x) < 0.1 || std::abs(valvePos.y - sourcePos.y) < 0.1) {
//         // Straight line connection
//         double length = sourcePos.distanceTo(valvePos);
//         links.push_back({
//             waterSourceId,
//             valveId,
//             5.0 * INCH_TO_METER, // Mainline diameter
//             length,
//             "mainline_to_zone",
//             false,
//             zoneID
//         });
//
//         // Update neighbors
//         nodes[waterSourceId].neighbors.push_back(valveId);
//         nodes[valveId].neighbors.push_back(waterSourceId);
//     } else {
//         // Create 90-degree connection with intermediate point(s)
//         // Try to route around the irrigation area
//
//         std::vector<Position> route = calculate90DegreeRoute(sourcePos, valvePos, zoneID);
//
//         int previousNodeId = waterSourceId;
//
//         for (size_t i = 0; i < route.size(); ++i) {
//             int intermediateId = getNextNodeId();
//
//             nodes[intermediateId] = {
//                 intermediateId,
//                 "mainline_junction",
//                 route[i],
//                 0.0,
//                 false,
//                 0.0,
//                 {},
//                 zoneID
//             };
//
//             double segmentLength = nodes[previousNodeId].position.distanceTo(route[i]);
//             links.push_back({
//                 previousNodeId,
//                 intermediateId,
//                 5.0 * INCH_TO_METER,
//                 segmentLength,
//                 "mainline_to_zone",
//                 false,
//                 zoneID
//             });
//
//             // Update neighbors
//             nodes[previousNodeId].neighbors.push_back(intermediateId);
//             nodes[intermediateId].neighbors.push_back(previousNodeId);
//
//             previousNodeId = intermediateId;
//         }
//
//         // Final connection to valve
//         double finalLength = nodes[previousNodeId].position.distanceTo(valvePos);
//         links.push_back({
//             previousNodeId,
//             valveId,
//             5.0 * INCH_TO_METER,
//             finalLength,
//             "mainline_to_zone",
//             false,
//             zoneID
//         });
//
//         nodes[previousNodeId].neighbors.push_back(valveId);
//         nodes[valveId].neighbors.push_back(previousNodeId);
//
//         std::cout << "Created 90-degree mainline connection with " << route.size() << " intermediate points" << std::endl;
//     }
// }


// connect water source to valve updated using nearest neighbor

void IrrigationModel::connectWaterSourceToValveWith90Degree(int valveId, int zoneID)
{
    Position valvePos = nodes[valveId].position;
    Position sourcePos = nodes[waterSourceId].position;

    // Find all existing mainline junctions and zone valves
    std::vector<int> existingMainlineNodes;
    existingMainlineNodes.push_back(waterSourceId); // Start with water source

    for (const auto& [id, node] : nodes) {
        if (node.type == "mainline_junction") {
            existingMainlineNodes.push_back(id);
        }
    }

    // Find the closest existing mainline node to connect to
    int closestNodeId = -1;
    double minDistance = std::numeric_limits<double>::max();
    Position closestNodePos;

    for (int existingNodeId : existingMainlineNodes) {
        // Don't connect to yourself
        if (existingNodeId == valveId) continue;

        Position existingPos = nodes[existingNodeId].position;
        double distance = valvePos.distanceTo(existingPos);

        // Prefer connecting to water source if it's reasonably close
        if (existingNodeId == waterSourceId) {
            distance *= 0.9; // Slight bias toward water source
        }

        if (distance < minDistance) {
            minDistance = distance;
            closestNodeId = existingNodeId;
            closestNodePos = existingPos;
        }
    }

    if (closestNodeId == -1) {
        // Fallback: connect directly to water source
        closestNodeId = waterSourceId;
        closestNodePos = sourcePos;
        minDistance = valvePos.distanceTo(sourcePos);
    }

    std::cout << "Connecting zone valve " << valveId << " to node " << closestNodeId
              << " (type: " << nodes[closestNodeId].type
              << ", distance: " << minDistance << "m)" << std::endl;

    // Check if alignment allows straight connection
    bool isAlignedX = std::abs(valvePos.x - closestNodePos.x) < 0.1;
    bool isAlignedY = std::abs(valvePos.y - closestNodePos.y) < 0.1;

    if (isAlignedX || isAlignedY) {
        // Straight line connection
        double length = valvePos.distanceTo(closestNodePos);
        links.push_back({
            closestNodeId,
            valveId,
            5.0 * INCH_TO_METER, // Mainline diameter
            length,
            "mainline_to_zone",
            false,
            zoneID
        });

        // Update neighbors
        nodes[closestNodeId].neighbors.push_back(valveId);
        nodes[valveId].neighbors.push_back(closestNodeId);

        std::cout << "Created straight connection from zone valve " << valveId
                  << " to node " << closestNodeId << " (length: " << length << "m)" << std::endl;
    } else {
        // Create 90-degree connection
        create90DegreeConnection(closestNodeId, valveId, zoneID);
    }
}

void IrrigationModel::create90DegreeConnection(int fromNodeId, int toNodeId, int zoneID)
{
    Position fromPos = nodes[fromNodeId].position;
    Position toPos = nodes[toNodeId].position;

    // Determine the optimal 90-degree routing
    // Option 1: Horizontal then vertical
    Position option1Point = {toPos.x, fromPos.y, 0.0};
    double option1Length = fromPos.distanceTo(option1Point) + option1Point.distanceTo(toPos);

    // Option 2: Vertical then horizontal
    Position option2Point = {fromPos.x, toPos.y, 0.0};
    double option2Length = fromPos.distanceTo(option2Point) + option2Point.distanceTo(toPos);

    // Choose the shorter route
    Position cornerPoint;
    if (option1Length <= option2Length) {
        cornerPoint = option1Point; // Horizontal then vertical
    } else {
        cornerPoint = option2Point; // Vertical then horizontal
    }

    // Check if corner point is too close to either endpoint
    if (fromPos.distanceTo(cornerPoint) < 0.5 || toPos.distanceTo(cornerPoint) < 0.5) {
        // If too close, connect directly
        double directLength = fromPos.distanceTo(toPos);
        links.push_back({
            fromNodeId,
            toNodeId,
            5.0 * INCH_TO_METER,
            directLength,
            "mainline_to_zone",
            false,
            zoneID
        });

        nodes[fromNodeId].neighbors.push_back(toNodeId);
        nodes[toNodeId].neighbors.push_back(fromNodeId);

        std::cout << "Created direct connection (corner too close)" << std::endl;
        return;
    }

    // Create intermediate corner node
    int cornerNodeId = getNextNodeId();
    nodes[cornerNodeId] = {
        cornerNodeId,
        "mainline_junction",
        cornerPoint,
        0.0,
        false,
        0.0,
        {},
        zoneID
    };

    // Create first segment
    double length1 = fromPos.distanceTo(cornerPoint);
    links.push_back({
        fromNodeId,
        cornerNodeId,
        5.0 * INCH_TO_METER,
        length1,
        "mainline_to_zone",
        false,
        zoneID
    });

    // Create second segment
    double length2 = cornerPoint.distanceTo(toPos);
    links.push_back({
        cornerNodeId,
        toNodeId,
        5.0 * INCH_TO_METER,
        length2,
        "mainline_to_zone",
        false,
        zoneID
    });

    // Update neighbors
    nodes[fromNodeId].neighbors.push_back(cornerNodeId);
    nodes[cornerNodeId].neighbors.push_back(fromNodeId);
    nodes[cornerNodeId].neighbors.push_back(toNodeId);
    nodes[toNodeId].neighbors.push_back(cornerNodeId);

    std::cout << "Created 90-degree connection via corner node " << cornerNodeId
              << " (total length: " << (length1 + length2) << "m)" << std::endl;
}



// Modified addZoneValveAndConnect to use optimized connections
void IrrigationModel::addZoneValveAndConnect(int zoneID, const std::vector<Position>& zoneBoundary)
{
    // Find the submain junction in this zone
    int zoneSubmainId = findZoneSubmainJunction(zoneID);

    if (zoneSubmainId == -1) {
        std::cerr << "No submain junction found in zone " << zoneID
                  << ". Creating one manually." << std::endl;
        zoneSubmainId = createZoneSubmainJunction(zoneID, zoneBoundary);
    }

    // Position valve near the submain junction
    Position submainPos = nodes[zoneSubmainId].position;
    Position valvePos = calculateOptimalValvePosition(submainPos, zoneBoundary);

    // Create zone valve
    int valveId = getNextNodeId();

    nodes[valveId] = {
        valveId,
        "zone_valve",
        valvePos,
        0.0,
        false,
        0.0,
        {},
        zoneID,
        false,    // isActive = false initially
        false     // isValveOpen = false initially
    };

    // Connect valve to zone submain
    connectValveToSubmainWith90Degree(valveId, zoneSubmainId, zoneID);

    // Connect valve to mainline network
    connectWaterSourceToValveWith90Degree(valveId, zoneID);

    std::cout << "Zone " << zoneID << ": Added valve " << valveId
              << " connected to submain " << zoneSubmainId << std::endl;

    // Rebuild valve mapping if this is a new valve
    if (valveToNodes.find(valveId) == valveToNodes.end()) {
        buildValveToNodesMapping();
    }
}

// Add this to assignZones function after creating all zones
void IrrigationModel::assignZones(
    int numZones,
    const std::vector<std::vector<Position>>& zoneBoundaries,
    double Pw,
    double sprinklerSpacing,
    double lineSpacing,
    const std::string& connectionType,
    const std::string& sprinklerConfig,
    SubmainPosition submainPos)
{
    if (zoneBoundaries.size() != static_cast<size_t>(numZones)) {
        helios::helios_runtime_error("ERROR (IrrigationModel::assignZones) Number of boundaries " + std::to_string(zoneBoundaries.size())
                 + " does not match numZones " + std::to_string(numZones));
        return;
    }

    // Clear existing system
    nodes.clear();
    links.clear();
    boundaryPolygon.clear();
    waterSourceId = -1;

    // Create the main water source (assume one system flow for all zones)
    waterSourceId = getNextNodeId();
    Position waterSourcePos = {-5.0, -10.0, 0.0}; // Position water source at a central location

    nodes[waterSourceId] = {
        waterSourceId,
        "waterSource",
        waterSourcePos,
        Pw,
        true
    };

    for (int z = 0; z < numZones; ++z) {
        std::cout << "Creating zone " << (z + 1) << "..." << std::endl;

        // Generate the irrigation system for this zone
        createIrregularSystemForZone(zoneBoundaries[z], sprinklerSpacing, lineSpacing,
                                   connectionType, sprinklerConfig, submainPos, z + 1);

        // Add zone valve and connect to main water source
        addZoneValveAndConnect(z + 1, zoneBoundaries[z]);

    }
    buildNeighborLists();

}



/////////////////////////////////////////////////////////////////////////////


Position IrrigationModel::calculateZoneCentroid(const std::vector<Position>& boundary) const
{
    if (boundary.empty()) {
        return {0.0, 0.0, 0.0};
    }

    double sumX = 0.0, sumY = 0.0;
    for (const auto& point : boundary) {
        sumX += point.x;
        sumY += point.y;
    }

    return {
        sumX / boundary.size(),
        sumY / boundary.size(),
        0.0
    };
}

int IrrigationModel::findZoneSubmainJunction(int zoneID) const
{
    // Look for submain junctions in the specified zone
    for (const auto& [id, node] : nodes) {
        if (node.zoneID == zoneID && node.type == "submain_junction") {
            return id;
        }
    }

    // If no submain junction found, try to find any junction in the zone
    for (const auto& [id, node] : nodes) {
        if (node.zoneID == zoneID &&
            (node.type.find("junction") != std::string::npos ||
             node.type.find("submain") != std::string::npos)) {
            return id;
        }
    }

    return -1;
}

int IrrigationModel::createZoneSubmainJunction(int zoneID, const std::vector<Position>& zoneBoundary)
{
    // Create a submain junction at the centroid of the zone
    Position junctionPos = calculateZoneCentroid(zoneBoundary);
    int junctionId = getNextNodeId();

    nodes[junctionId] = {
        junctionId,
        "submain_junction",
        junctionPos,
        0.0,
        false,
        0.0,
        {},
        zoneID
    };

    return junctionId;
}

const std::unordered_map<int, Node>& IrrigationModel::getNodes() const {
    return nodes;
}

const std::vector<Link>& IrrigationModel::getLinks() const {
    return links;
}


std::vector<int> IrrigationModel::getNodesByType(const std::string& type) const {
    std::vector<int> result;
    for (const auto& [id, node] : nodes) {
        if (node.type == type) {
            result.push_back(id);
        }
    }
    return result;
}

std::vector<Link> IrrigationModel::getLinksByType(const std::string& type) const {
    std::vector<Link> result;
    for (const auto& link : links) {
        if (link.type == type) {
            result.push_back(link);
        }
    }
    return result;
}

// Helper function to get nodes by zone
std::vector<int> IrrigationModel::getNodesByZone(int zoneID) const
{
    std::vector<int> zoneNodes;
    for (const auto& [id, node] : nodes) {
        if (node.zoneID == zoneID) {
            zoneNodes.push_back(id);
        }
    }
    return zoneNodes;
}

// Helper function to get links by zone
std::vector<size_t> IrrigationModel::getLinksByZone(int zoneID) const
{
    std::vector<size_t> zoneLinks;
    for (size_t i = 0; i < links.size(); ++i) {
        if (links[i].zoneID == zoneID) {
            zoneLinks.push_back(i);
        }
    }
    return zoneLinks;
}


void IrrigationModel::validateHydraulicSystem() const {
    // Water source validation
    if (waterSourceId == -1) {
        helios::helios_runtime_error("ERROR (IrrigationModel::validateHydraulicSystem) No water source defined");
    }

    if (!nodes.count(waterSourceId)) {
        helios::helios_runtime_error("ERROR (IrrigationModel::validateHydraulicSystem) Water source node doesn't exist");
    }

    // connectivity check, traversing through the network
    std::unordered_set<int> connectedNodes;
    std::queue<int> nodesToProcess;
    nodesToProcess.push(waterSourceId);
    connectedNodes.insert(waterSourceId);

    while (!nodesToProcess.empty()) {
        int current = nodesToProcess.front();
        nodesToProcess.pop();

        // Check all links connected to this node (both directions)
        for (const auto& link : links) {
            if (link.from == current && !connectedNodes.count(link.to)) {
                connectedNodes.insert(link.to);
                nodesToProcess.push(link.to);
            }
            if (link.to == current && !connectedNodes.count(link.from)) {
                connectedNodes.insert(link.from);
                nodesToProcess.push(link.from);
            }
        }
    }

    // Report disconnected nodes
    size_t disconnectedCount = nodes.size() - connectedNodes.size();
    if (disconnectedCount > 0) {
        helios::helios_runtime_error("ERROR (IrrigationModel::validateHydraulicSystem) Nodes are disconnected from water source");
        // List disconnected nodes
        for (const auto& [id, node] : nodes) {
            if (!connectedNodes.count(id)) {
                std::cerr << "  - Node " << id << " at ("
                         << node.position.x << ", "
                         << node.position.y << ")" << std::endl;
            }
        }
    }

    // Link validation
    for (const auto& link : links) {
        if (link.diameter <= 0) {
            helios::helios_runtime_error("ERROR (IrrigationModel::validateHydraulicSystem) Invalid diameter (<= 0) in link between nodes " +
                                   std::to_string(link.from) + " and " +
                                   std::to_string(link.to));
        }
        if (link.length <= 0) {
            helios::helios_runtime_error("ERROR (IrrigationModel::validateParameters) Invalid length (<= 0) in link between nodes " +
                                   std::to_string(link.from) + " and " +
                                   std::to_string(link.to));
        }
    }

    // printing system summary
    std::cout << "System validation complete:\n"
              << "  - Total nodes: " << nodes.size() << "\n"
              << "  - Connected nodes: " << connectedNodes.size() << "\n"
              << "  - Total links: " << links.size() << "\n"
              << "  - Water source pressure: "
              << nodes.at(waterSourceId).pressure << " psi" << std::endl;
}


void IrrigationModel::buildNeighborLists() {
    // Clear existing neighbors
    for (auto& [id, node] : nodes) {
        node.neighbors.clear();
    }

    // Build neighbor lists
    for (const auto& link : links) {
        nodes[link.from].neighbors.push_back(link.to);
        nodes[link.to].neighbors.push_back(link.from);
    }
}

double IrrigationModel::calculateResistance(double Re, double Wbar, double Kf_barb, const Link& link, int iter) {
    const double rho = 997.0;
    const double mu = 8.90e-04;
    double R = 0.0;

    // depends on laminar or turbulent flow
    if (Re < 2000 || iter == 1) {
        R = mu * link.length * 128.0 / (M_PI * pow(link.diameter, 4));
    //    std::cout << "it is laminar flow" << std::endl;
    } else {
        R = 0.6328 * pow(Wbar, 0.75) * pow(mu, 0.25) * link.length * pow(rho, 0.75) /
            (M_PI * pow(link.diameter, 3.25));
      //  std::cout << "it is turbulent flow" << std::endl;
      //  std::cout << "link type: " << link.type << "\n";

        // using minor loss equation if the link is lateralToBarb
        if (link.type == "lateralTobarb" && Kf_barb > 0) {
       //     std::cout << "calling lateralTobarb" << std::endl;
            R =  2.0 * Wbar * Kf_barb * rho / (M_PI * pow(link.diameter, 2));
        }
    }


    // Safety check
    if (R <= 0 || std::isinf(R) || std::isnan(R)) {
        std::cerr << "Invalid resistance calculated: " << R
                  << " for link " << link.toString()
                  << " with Re=" << Re << std::endl;
        R = 1.0; // fallback value
        helios::helios_runtime_error("ERROR (IrrigationModel::calculateResistance): Invalid resistance");
    }

    return R;
}

double IrrigationModel::minorLoss_kf(const double Re, const std::string& sprinklerAssemblyType) {
    double kf;
    const SprinklerAssembly &config = sprinklerLibrary.getSprinklerType(sprinklerAssemblyType);
    std::string barbType = config.barbType;

    if (barbType == "NA") {
        kf = 0;
    } else if (barbType == "PC_Nelson_flat_barb") {
        kf = 1.78;
    } else if (barbType == "PC_Toro_flat_barb") {
        kf = 1.88;
    } else if (barbType == "PC_Toro_sharp_barb") {
        kf = 2.86;
    } else if (barbType  == "NPC_Nelson_flat_barb") {
   //    std::cout<<"reading the minor loss of NelsonFlat" <<std::endl;

        kf = 1.73;
    } else if (barbType  == "NPC_Toro_flat_barb") {
        kf = 1.48; //1.50 for half tube
    } else if (barbType  == "NPC_Toro_sharp_barb") {
    //    std::cout<<"reading the minor loss of ToroSharp" <<std::endl;
        kf = 3.12;
    } else if (barbType  == "Standard elbow, 45 deg") { //need to add a function for valve
        if (Re <= 50) {
            kf = 10;
        } else if (Re > 50 && Re <= 100) {
            kf = 6;
        } else if (Re > 100 && Re <= 200) {
            kf = 1.5;
        } else if (Re > 200 && Re <= 400) {
            kf = 0.8;
        } else { // re > 400
            kf = 0.45;
        }
    } else if (barbType  == "Standard elbow, 90 deg") {
        if (Re <= 50) {
            kf = 17;
        } else if (Re > 50 && Re <= 100) {
            kf = 7;
        } else if (Re > 100 && Re <= 200) {
            kf = 2.5;
        } else if (Re > 200 && Re <= 400) {
            kf = 1.2;
        } else { // re > 400
            kf = 0.85;
        }
    } else if (barbType  == "Check valve, Swing") {
        if (Re <= 50) {
            kf = 55;
        } else if (Re > 50 && Re <= 100) {
            kf = 17;
        } else if (Re > 100 && Re <= 200) {
            kf = 9;
        } else if (Re > 200 && Re <= 400) {
            kf = 5.8;
        } else { // re > 400
            kf = 3.2;
        }
    } else {
        // other unknown names
        std::cerr << "Warning: Unknown component name '" << barbType  << "'. Returning 0." << std::endl;
        helios::helios_runtime_error("ERROR (IrrigationModel::minorLoss_kf): Unknown minor loss");
        kf = 0;
    }

    return kf;
}

double IrrigationModel::calculateEmitterFlow(const std::string& sprinklerAssemblyType,
                                             double pressure,
                                             bool updateEmitterNodes) {

    const SprinklerAssembly & config = sprinklerLibrary.getSprinklerType(sprinklerAssemblyType);
    std::string nozzleType = config.emitterType;

    // Convert pressure from Pa to psi
    double pressure_psi = pressure ;

    double computedFlow = 0.0;

    if (nozzleType == "PC") {
        // Linear relationship for PC nozzle (GPH to m3/s)
        computedFlow = std::max(0.0, std::min(1.6049e-5, (0.9 * pressure_psi * 1.052e-6)));
    } else if (nozzleType == "NPC") {
        // Non-linear relationship for NPC nozzle
        // const double x = 0.477; //fan jet J2 orange 0.53 0.477;
        // const double k = 3.317; //1.76;
        computedFlow = config.emitter_k * pow(pressure_psi, config.emitter_x)*1.052e-6; //get coefficients directly from library
    } else {
        helios::helios_runtime_error("ERROR (IrrigationModel::calculateEmitterFlow): Unknown nozzle type");
        return 0.0;
    }

    // update all emitter nodes so preSizePipes can roll up branch flows
    if (updateEmitterNodes) {
        for (auto& [id, node] : nodes) {
            if (node.type == "emitter") {
                node.flow = computedFlow;
            }
        }
    }

    return computedFlow;
}

SprinklerConfigLibrary::SprinklerConfigLibrary() {
    // Register assemblies to the library
    registerSprinklerAssembly(create_NPC_Nelson_flat());
    registerSprinklerAssembly(create_NPC_Toro_flat());
    registerSprinklerAssembly(create_NPC_Toro_sharp());
    registerSprinklerAssembly(create_PC_Nelson_flat());
    registerSprinklerAssembly(create_PC_Toro_flat());
    registerSprinklerAssembly(create_PC_Toro_sharp());

}

void SprinklerConfigLibrary::registerSprinklerAssembly(const SprinklerAssembly& type) {
    sprinklerLibrary[type.name] = type;
}

bool SprinklerConfigLibrary::hasSprinklerAssembly(const std::string& typeName) const {
    return sprinklerLibrary.find(typeName) != sprinklerLibrary.end();
}

const SprinklerAssembly& SprinklerConfigLibrary::getSprinklerType(const std::string& typeName) const {
    auto it = sprinklerLibrary.find(typeName);
    if (it == sprinklerLibrary.end()){
        helios::helios_runtime_error("ERROR (SprinklerConfidLibrary::getSprinklerType): Sprinkler not found");
    }
    return it->second;
}

std::vector<std::string> SprinklerConfigLibrary::getAvailableTypes() const {
    std::vector<std::string> types;
    for (const auto& pair : sprinklerLibrary) {
        types.push_back(pair.first);
    }
    return types;
}

// add sprinkler types default or custom
SprinklerAssembly SprinklerConfigLibrary:: create_NPC_Nelson_flat() {
    SprinklerAssembly type;
    type.name = "NPC_Nelson_flat";
    type.lateralToBarb = {0.12 * INCH_TO_METER, 0.75* INCH_TO_METER, "PVC"};
    type.barbToEmitter = { 0.154 * INCH_TO_METER, 30*INCH_TO_METER,"PVC"};
    type.emitterType = "NPC";
    type.emitter_x = 0.477;
    type.emitter_k = 3.317;
    type.stakeHeight = 0.1; //10cm above ground
    type.barbType =  "NPC_Nelson_flat_barb";
    return type;
}

SprinklerAssembly SprinklerConfigLibrary:: create_NPC_Toro_flat() {
    SprinklerAssembly type;
    type.name = "NPC_Toro_flat";
    type.lateralToBarb = {0.104* INCH_TO_METER, 0.74*INCH_TO_METER, "PVC"};
    type.barbToEmitter = { 0.154 * INCH_TO_METER, 30*INCH_TO_METER,"PVC"};
    type.emitterType = "NPC";
    type.emitter_x = 0.477;
    type.emitter_k = 3.317;
    type.stakeHeight = 0.1; //10cm above ground
    type.barbType = "NPC_Toro_flat_barb";
    return type;
}

SprinklerAssembly SprinklerConfigLibrary:: create_NPC_Toro_sharp() {
    SprinklerAssembly type;
    type.name = "NPC_Toro_sharp";
    type.lateralToBarb = {0.118* INCH_TO_METER, 0.74* INCH_TO_METER, "PVC"};
    type.barbToEmitter = { 0.154 * INCH_TO_METER, 30*INCH_TO_METER,"PVC"};
    type.emitterType = "NPC";
    type.emitter_x = 0.477;
    type.emitter_k = 3.317;
    type.stakeHeight = 0.1; //10cm above ground
    type.barbType =  "NPC_Toro_sharp_barb";
    return type;
}

SprinklerAssembly SprinklerConfigLibrary:: create_PC_Nelson_flat() {
    SprinklerAssembly type;
    type.name = "PC_Nelson_flat";
    type.lateralToBarb = {0.12 * INCH_TO_METER, 0.74* INCH_TO_METER, "PVC"};
    type.barbToEmitter = { 0.154 * INCH_TO_METER, 30*INCH_TO_METER,"PVC"};
    type.emitterType = "PC";
    type.emitter_x = 0.0;
    type.emitter_k = 3.317;
    type.stakeHeight = 0.1; //10cm above ground
    type.barbType =  "PC_Nelson_flat_barb";
    return type;
}

SprinklerAssembly SprinklerConfigLibrary:: create_PC_Toro_flat() {
    SprinklerAssembly type;
    type.name = "PC_Toro_flat";
    type.lateralToBarb = {0.104* INCH_TO_METER, 0.74* INCH_TO_METER, "PVC"};
    type.barbToEmitter = { 0.154 * INCH_TO_METER, 30*INCH_TO_METER,"PVC"};
    type.emitterType = "PC";
    type.emitter_x = 0.0;
    type.emitter_k = 3.317;
    type.stakeHeight = 0.1; //10cm above ground
    type.barbType = "PC_Toro_flat_barb";
    return type;
}

SprinklerAssembly SprinklerConfigLibrary:: create_PC_Toro_sharp() {
    SprinklerAssembly type;
    type.name = "PC_Toro_sharp";
    type.lateralToBarb = {0.118* INCH_TO_METER, 0.74* INCH_TO_METER, "PVC"};
    type.barbToEmitter = { 0.154 * INCH_TO_METER, 30*INCH_TO_METER,"PVC"};
    type.emitterType = "PC";
    type.emitter_x = 0.0;
    type.emitter_k = 3.317;
    type.stakeHeight = 0.1; //10cm above ground
    type.barbType =  "PC_Toro_sharp_barb";
    return type;
}


// adjusted to start the nodeID from 1
bool IrrigationModel::loadFromTextFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        helios::helios_runtime_error("IrrigationModel::loadFromTextFile(): fail to open IRRICAD file");
        return false;
    }

    std::string line;
    bool readingNodes = false;
    bool readingLinks = false;
    int nodeCount = 0;
    int linkCount = 0;
    const int ID_OFFSET = 0; // Shift all IDs by 1

    while (std::getline(file, line)) {
        // Skip empty lines and comments
        if (line.empty() || line[0] == '#') continue;

        // Check section headers
        if (line == "NODES_START") {
            readingNodes = true;
            readingLinks = false;
            continue;
        } else if (line == "NODES_END") {
            readingNodes = false;
            continue;
        } else if (line == "LINKS_START") {
            readingLinks = true;
            readingNodes = false;
            continue;
        } else if (line == "LINKS_END") {
            readingLinks = false;
            continue;
        }

        if (readingNodes) {
            std::istringstream iss(line);
            Node node;
            int originalId;

            // Format: id x y type pressure is_fixed flow elevation
            if (iss >> originalId >> node.position.x >> node.position.y >> node.position.z >> node.type
                >> node.pressure >> node.is_fixed >> node.flow) {

                // Apply offset to node ID
                node.id = originalId + ID_OFFSET;
                nodes[node.id] = node;
                nodeCount++;
            } else {
                std::cerr << "Warning: Could not parse node line: " << line << std::endl;
            }
        } else if (readingLinks) {
            std::istringstream iss(line);
            Link link;
            int originalFrom, originalTo;

            // Format: from to diameter length type flow
            if (iss >> originalFrom >> originalTo >> link.diameter >> link.length
                >> link.type >> link.flow) {

                // Apply offset to link endpoints
                link.from = originalFrom + ID_OFFSET;
                link.to = originalTo + ID_OFFSET;

                links.push_back(link);
                linkCount++;
            } else {
                std::cerr << "Warning: Could not parse link line: " << line << std::endl;
            }
        }
    }

    // Build neighbor lists
    buildNeighborLists();

    std::cout << "Successfully loaded text file: " << nodeCount << " nodes, "
              << linkCount << " links" << std::endl;
    std::cout << "Node IDs shifted from 0-based to 1-based indexing" << std::endl;

    return true;
}


//functions to help simulate any combination of active zones
void IrrigationModel::openZoneValve(int zoneID)
{
    bool changed = false;
    for (auto& [id, node] : nodes) {
        if (node.type == "zone_valve" && node.zoneID == zoneID) {
            if (!node.isValveOpen) {
                node.isValveOpen = true;
                changed = true;
                std::cout << "Opened valve for zone " << zoneID << " (node " << id << ")" << std::endl;
            }
        }
    }

    if (changed) {
        updateActiveNodesFast();
    } else {
        std::cout << "Valve for zone " << zoneID << " was already open or not found" << std::endl;
    }
}

void IrrigationModel::closeZoneValve(int zoneID)
{
    bool changed = false;
    for (auto& [id, node] : nodes) {
        if (node.type == "zone_valve" && node.zoneID == zoneID) {
            if (node.isValveOpen) {
                node.isValveOpen = false;
                changed = true;
                std::cout << "Closed valve for zone " << zoneID << " (node " << id << ")" << std::endl;
            }
        }
    }

    if (changed) {
        updateActiveNodesFast();
    } else {
        std::cout << "Valve for zone " << zoneID << " was already closed or not found" << std::endl;
    }
}

void IrrigationModel::setZoneValveState(int zoneID, bool open)
{
    if (open) {
        openZoneValve(zoneID);
    } else {
        closeZoneValve(zoneID);
    }
}

bool IrrigationModel::isZoneValveOpen(int zoneID) const
{
    for (const auto& [id, node] : nodes) {
        if (node.type == "zone_valve" && node.zoneID == zoneID) {
            return node.isValveOpen;
        }
    }
    return false;
}


void IrrigationModel::buildValveToNodesMapping()
{
    std::cout << "\n building zone map" << std::endl;

    valveToNodes.clear();

    // find all zone valves
    std::vector<int> zoneValves;
    for (const auto& [id, node] : nodes) {
        if (node.type == "zone_valve") {
            zoneValves.push_back(id);
        }
    }

    // For each valve, find all nodes in its zone
    for (int valveId : zoneValves) {
        int zoneID = nodes[valveId].zoneID;

        std::unordered_set<int> allNodesInZone;

        // Find by zone ID (make sure to get all nodes in the zone)
        for (const auto& [id, node] : nodes) {
            if (node.zoneID == zoneID) {
                allNodesInZone.insert(id);
            }
        }

        // ensure connectivity via bfs
        std::unordered_set<int> connectedNodes;
        bfsCollectNodes(valveId, zoneID, connectedNodes);

        // Combine both sets
        allNodesInZone.insert(connectedNodes.begin(), connectedNodes.end());

        valveToNodes[valveId] = allNodesInZone;

        std::cout << "Valve " << valveId << " (Zone " << zoneID
                  << ") can reach " << allNodesInZone.size()
                  << " nodes in its zone" << std::endl;
    }

   // std::cout << "  zone map completed\n" << std::endl;
}

void IrrigationModel::bfsCollectNodes(int startId, int targetZoneID,
                                            std::unordered_set<int>& result)
{
    std::queue<int> q;
    std::unordered_set<int> visited;
    q.push(startId);
    visited.insert(startId);

    while (!q.empty()) {
        int current = q.front();
        q.pop();

        result.insert(current);

        const Node& currentNode = nodes.at(current);

        for (int neighbor : currentNode.neighbors) {
            if (visited.count(neighbor)) continue;

            const Node& neighborNode = nodes.at(neighbor);

            // Allow traversal if neighbor is in the same zone or it's infrastructure (water source/mainline)
            // or current is infrastructure
            if (neighborNode.zoneID == targetZoneID ||
                currentNode.type == "zone_valve" ||
                neighborNode.type == "waterSource" ||
                neighborNode.type == "mainline_junction" ||
                currentNode.type == "waterSource" ||
                currentNode.type == "mainline_junction") {

                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
}

void IrrigationModel::checkUnassignedNodes()
{
    std::cout << "\n Check unassigned nodes " << std::endl;

    int unassignedCount = 0;
    int irrigationUnassigned = 0;

    for (const auto& [id, node] : nodes) {
        if (node.zoneID == 0) {
            unassignedCount++;
            if (node.type == "lateral_sprinkler_jn" || node.type == "barb" || node.type == "emitter") {
                irrigationUnassigned++;
                std::cout << "  Unassigned irrigation node: " << id
                          << " (" << node.type << ")" << std::endl;
                }
        }
    }

    std::cout << "Total unassigned nodes: " << unassignedCount << std::endl;
    std::cout << "Unassigned irrigation nodes: " << irrigationUnassigned << std::endl;

    if (irrigationUnassigned > 0) {
        std::cout << "\n Problem: Irrigation nodes exist but have zoneID = 0" << std::endl;
        std::cout << "Nodes not included in any zone's active set!" << std::endl;
    }
}

std::vector<std::pair<double, double>> IrrigationModel::generateSystemCurve(
        const std::vector<double>& emitterRequiredPressure,
        const std::string& sprinklerAssemblyType,
        double V_main,
        double V_lateral,
        int referenceNodeId,
        double staticHead) {

    std::vector<std::pair<double, double>> systemCurvePoints;
    if (emitterRequiredPressure.empty()) return systemCurvePoints;

    // // Ensure water source ID is set
    // if (waterSourceId == -1) {
    //     for (const auto& [id, node] : nodes) {
    //         if (node.type == "waterSource") {
    //             waterSourceId = id;
    //             break;
    //         }
    //     }
    // }
    if (waterSourceId == -1) {
        helios::helios_runtime_error("ERROR (IrrigationModel::generateSystemCurve): No waterSource found");
        return systemCurvePoints;
    }

    // set a reference node if not provided farthest active emitter from water source
    if (referenceNodeId == -1) {
        double maxDist = -1.0;
        int bestId = -1;
        const auto& wsPos = nodes.at(waterSourceId).position;
        for (const auto& [id, node] : nodes) {
            if (node.type != "emitter") continue;
            if (!node.isActive) continue;
            double dist = wsPos.distanceTo(node.position);
            if (dist > maxDist) {
                maxDist = dist;
                bestId = id;
            }
        }
        referenceNodeId = bestId;
    }

    if (referenceNodeId == -1 || !nodes.count(referenceNodeId)) {
        std::cerr << "Warning: No valid reference node found for system curve." << std::endl;
        return systemCurvePoints;
    }

    for (double emitterPressure : emitterRequiredPressure) {
        if (emitterPressure <= 0.0) continue;

        const double Qspecified =  calculateEmitterFlow(sprinklerAssemblyType,emitterPressure);
        const double Pw = 0.0;
        preSizePipes(1.5, 2.0);
        HydraulicResults results = calculateHydraulicsMultiZoneOptimized(
            false, sprinklerAssemblyType, Qspecified, Pw, V_main, V_lateral);

        double totalFlow = 0.0;
        for (double q : results.emitterFlows) {
            totalFlow += q;
        }

        // Pressure loss from water source to reference (psi) and convert to feet, add static head (feet)
        double totalHead = (-nodes[referenceNodeId].pressure + emitterPressure)*2.31 + staticHead;

        // Convert flow to GPM
        const double totalFlowGpm = totalFlow * 15850.323141489; // m3/s -> GPM
        systemCurvePoints.emplace_back(totalFlowGpm, totalHead);
    }

    return systemCurvePoints;
}


void IrrigationModel::writeSystemCurveToCsvWithFit(
    const std::string& filename,
    const std::vector<std::pair<double, double>>& curve) const
{
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing\n";
        return;
    }

    auto [a, b, c] = fitCurveQuadratic(curve);
    out << "# Head = a + b*GPM + c*GPM^2\n";
    out << "# a=" << a << ", b=" << b << ", c=" << c << "\n";
    out << "gpm,head_ft\n";
    for (const auto& point : curve) {
        out << point.first << "," << point.second << "\n";
    }
}

std::tuple<double, double, double> IrrigationModel::fitCurveQuadratic(
    const std::vector<std::pair<double, double>>& curve)
{
    // Fit Head = a + b*GPM + c*GPM^2 using least squares.
    // Returns coefficients  If insufficient data, returns zeros.
    const size_t n = curve.size();
    if (n < 3) {
        helios::helios_runtime_error("ERROR (IrrigationModel::fitCurveQuadratic): Need at least 3 points to fit quadratic system curve");
        return {0.0, 0.0, 0.0};
    }

    double Sx = 0.0, Sx2 = 0.0, Sx3 = 0.0, Sx4 = 0.0;
    double Sy = 0.0, Sxy = 0.0, Sx2y = 0.0;

    for (const auto& [x, y] : curve) {
        const double x2 = x * x;
        Sx += x;
        Sx2 += x2;
        Sx3 += x2 * x;
        Sx4 += x2 * x2;
        Sy += y;
        Sxy += x * y;
        Sx2y += x2 * y;
    }

    // Solve normal equations:
    // [ n   Sx   Sx2 ] [a] = [ Sy  ]
    // [ Sx  Sx2  Sx3 ] [b] = [ Sxy ]
    // [ Sx2 Sx3  Sx4 ] [c] = [ Sx2y]
    const double A00 = static_cast<double>(n), A01 = Sx,  A02 = Sx2;
    const double A10 = Sx,  A11 = Sx2, A12 = Sx3;
    const double A20 = Sx2, A21 = Sx3, A22 = Sx4;

    const double B0 = Sy, B1 = Sxy, B2 = Sx2y;

    const double det =
        A00 * (A11 * A22 - A12 * A21) -
        A01 * (A10 * A22 - A12 * A20) +
        A02 * (A10 * A21 - A11 * A20);

    if (std::abs(det) < 1e-12) {
        helios::helios_runtime_error("ERROR (IrrigationModel::fitCurveQuadratic): Warning: Quadratic fit is ill-conditioned.");
        return {0.0, 0.0, 0.0};
    }

    const double detA =
        B0 * (A11 * A22 - A12 * A21) -
        A01 * (B1 * A22 - A12 * B2) +
        A02 * (B1 * A21 - A11 * B2);

    const double detB =
        A00 * (B1 * A22 - A12 * B2) -
        B0 * (A10 * A22 - A12 * A20) +
        A02 * (A10 * B2 - B1 * A20);

    const double detC =
        A00 * (A11 * B2 - B1 * A21) -
        A01 * (A10 * B2 - B1 * A20) +
        B0 * (A10 * A21 - A11 * A20);

    const double a = detA / det;
    const double b = detB / det;
    const double c = detC / det;

    return {a, b, c};
}


uint32_t IrrigationModel::computeValveConfigHash() const
{
    uint32_t hash = 0;
    int valveIndex = 0;

    std::vector<int> valveIds;     // Sort valves by ID for consistent hashing

    for (const auto& [id, node] : nodes) {
        if (node.type == "zone_valve") {
            valveIds.push_back(id);
        }
    }
    std::sort(valveIds.begin(), valveIds.end());

    // Create hash from sorted valve states
    for (int valveId : valveIds) {
        const Node& valve = nodes.at(valveId);
        // Each valve contributes 2 bits to the hash
        hash = (hash << 2) | (valve.isValveOpen ? 0b11 : 0b00);
        valveIndex++;

        if (valveIndex >= 16) break; // Limit to 16 valves for 32-bit hash
    }

    // Ensure hash is never 0 (0 means no configuration
    if (hash == 0 && !valveIds.empty()) {
        hash = 1;
    }

    return hash;
}

const std::unordered_set<int>& IrrigationModel::getActiveNodesCached()
{
    uint32_t configHash = computeValveConfigHash();

    // Show current valve states
    // std::cout << " Computing Active Nodes" << std::endl;
    // std::cout << "Valve configuration hash: " << configHash << std::endl;
    // std::cout << "Current valve states:" << std::endl;

    std::vector<std::pair<int, bool>> valveStates;
    for (const auto& [id, node] : nodes) {
        if (node.type == "zone_valve") {
            valveStates.emplace_back(node.zoneID, node.isValveOpen);
            std::cout << "  Zone " << node.zoneID << " (valve " << id
                      << "): " << (node.isValveOpen ? "OPEN" : "CLOSED") << std::endl;
        }
    }

    // Cache lookup
    auto it = activeNodeCache.find(configHash);
    if (it != activeNodeCache.end()) {
        // std::cout << "Cache hit. Using cached result with "
        //           << it->second.size() << " active nodes" << std::endl;
        currentValveConfigHash = configHash;
        return it->second;
    }

    //std::cout << "Cache miss. Computing active nodes." << std::endl;

    // Compute new active node set
    std::unordered_set<int> activeNodes;

    // Always include water source
    if (waterSourceId != -1) {
        activeNodes.insert(waterSourceId);
        nodes[waterSourceId].isActive = true;
    } else {
        // Find water source if not set
        for (const auto& [id, node] : nodes) {
            if (node.type == "waterSource") {
                waterSourceId = id;
                activeNodes.insert(id);
                nodes[id].isActive = true;
                break;
            }
        }
    }

    // Add nodes from open valves using pre-computed mapping
    for (const auto& [valveId, valveNodes] : valveToNodes) {
        if (nodes[valveId].isValveOpen) {
            activeNodes.insert(valveNodes.begin(), valveNodes.end());
        }
    }

    // Ensure connectivity between active zones via mainline
   // ensureMainlineConnectivity(activeNodes);

    // Update node active states
    for (auto& [id, node] : nodes) {
        node.isActive = (activeNodes.find(id) != activeNodes.end());
    }

    // Cache the result
    activeNodeCache[configHash] = activeNodes;
    currentValveConfigHash = configHash;

    // std::cout << "Computed active nodes for hash " << configHash
    //           << ": " << activeNodes.size() << " nodes active" << std::endl;

    return activeNodeCache[configHash];
}

void IrrigationModel::ensureMainlineConnectivity(std::unordered_set<int>& activeNodes)
{
    bool changed = true;

    while (changed) {
        changed = false;

        for (const auto& link : links) {
            bool fromActive = (activeNodes.find(link.from) != activeNodes.end());
            bool toActive = (activeNodes.find(link.to) != activeNodes.end());

            // If link is mainline and one end is active, activate the other end
            if (link.type.find("mainline_to_zone") != std::string::npos) {

                if (fromActive && !toActive) {
                    // Check if to-node is a valve (don't activate through closed valves)
                    if (nodes[link.to].type == "zone_valve" && !nodes[link.to].isValveOpen) {
                        continue;
                    }
                    activeNodes.insert(link.to);
                    changed = true;
                } else if (!fromActive && toActive) {
                    // Check if from-node is a valve
                    if (nodes[link.from].type == "zone_valve" && !nodes[link.from].isValveOpen) {
                        continue;
                    }
                    activeNodes.insert(link.from);
                    changed = true;
                }
            }
        }
    }
}

void IrrigationModel::updateActiveNodesFast()
{
    // Use cached computation
    const std::unordered_set<int>& activeNodes = getActiveNodesCached();

    // Update all nodes (already done in getActiveNodesCached, but ensure)
    for (auto& [id, node] : nodes) {
        node.isActive = (activeNodes.find(id) != activeNodes.end());
    }
}

void IrrigationModel::updateActiveNodesByTreeTraversal()
{
    // Reset all nodes to inactive
    for (auto& [id, node] : nodes) {
        node.isActive = false;
    }

    // Mark water source as always active
    if (waterSourceId != -1) {
        nodes[waterSourceId].isActive = true;
    } else {
        // Find water source
        for (auto& [id, node] : nodes) {
            if (node.type == "waterSource") {
                waterSourceId = id;
                node.isActive = true;
                break;
            }
        }
    }

    // Start BFS from each open zone valve
    for (const auto& [id, node] : nodes) {
        if (node.type == "zone_valve" && node.isValveOpen) {
            bfsFromValve(id);
        }
    }

    // Ensure mainline connectivity
    propagateMainlineActivation();
}

void IrrigationModel::bfsFromValve(int valveId)
{
    std::queue<int> nodeQueue;
    std::unordered_set<int> visited;

    nodeQueue.push(valveId);
    visited.insert(valveId);
    nodes[valveId].isActive = true;

    while (!nodeQueue.empty()) {
        int currentId = nodeQueue.front();
        nodeQueue.pop();

        // Traverse to neighbors (except through closed valves)
        for (int neighborId : nodes[currentId].neighbors) {
            // Skip if already visited
            if (visited.count(neighborId)) continue;

            // Don't traverse through closed valves (except water source direction)
            if (nodes[neighborId].type == "zone_valve" && !nodes[neighborId].isValveOpen) {
                continue;
            }

            visited.insert(neighborId);
            nodeQueue.push(neighborId);
            nodes[neighborId].isActive = true;
        }
    }
}

void IrrigationModel::propagateMainlineActivation()
{
    // Ensure mainline junctions between active zones are active
    // This is a second pass to connect isolated active zones via mainline
    bool changed = true;

    while (changed) {
        changed = false;

        for (const auto& link : links) {
            if (link.type.find("mainline_to_zone") != std::string::npos ||
                link.type == "zone_valve") {

                bool fromActive = nodes[link.from].isActive;
                bool toActive = nodes[link.to].isActive;

                // If one end is active and the other isn't a closed valve, activate it
                if (fromActive && !toActive) {
                    // Don't activate through closed valves
                    if (nodes[link.to].type == "zone_valve" && !nodes[link.to].isValveOpen) {
                        continue;
                    }
                    nodes[link.to].isActive = true;
                    changed = true;
                } else if (!fromActive && toActive) {
                    if (nodes[link.from].type == "zone_valve" && !nodes[link.from].isValveOpen) {
                        continue;
                    }
                    nodes[link.from].isActive = true;
                    changed = true;
                }
            }
        }
    }
}

HydraulicResults IrrigationModel::calculateHydraulicsMultiZoneOptimized(
    bool doPreSize,
    const std::string& sprinklerAssemblyType,
    double Qspecified,
    double Pw,
    double V_main,
    double V_lateral)
{
    // call active nodes from cache (for activated zones)
    const std::unordered_set<int>& activeNodeIds = getActiveNodesCached();

    if (activeNodeIds.empty()) {
        helios::helios_runtime_error("ERROR (IrrigationModel::calculateHydraulicsMultiZoneOptimized): No active nodes in the system.");
        return HydraulicResults();
    }

    // call active links
    std::vector<Link*> activeLinks;
    activeLinks.reserve(links.size());
    for (auto& link : links) {
        if (activeNodeIds.find(link.from) != activeNodeIds.end() &&
            activeNodeIds.find(link.to) != activeNodeIds.end()) {
            activeLinks.push_back(&link);
        }
    }

    // std::cout << "Found " << activeNodeIds.size() << " active nodes and "
    //           << activeLinks.size() << " active links" << std::endl;

    // ordered node list and mapping
    std::vector<int> orderedNodeIds(activeNodeIds.begin(), activeNodeIds.end());
    std::sort(orderedNodeIds.begin(), orderedNodeIds.end());

    std::unordered_map<int, int> nodeIndexMap;
    for (int i = 0; i < orderedNodeIds.size(); ++i) {
        nodeIndexMap[orderedNodeIds[i]] = i;
    }

    // create adjacency list for active nodes
    std::unordered_map<int, std::vector<int>> activeNeighbors;
    for (const Link* link : activeLinks) {
        activeNeighbors[link->from].push_back(link->to);
        activeNeighbors[link->to].push_back(link->from);
    }



    // Constants
    const double rho = 997.0;
    const double mu = 8.90e-04;
    const double err_max = 1e-3;
    const int max_iter = 1000;

    const int numNodes = orderedNodeIds.size();

    // water source index
    int waterSourceIndex = -1;
    if (nodeIndexMap.find(waterSourceId) != nodeIndexMap.end()) {
        waterSourceIndex = nodeIndexMap[waterSourceId];
    } else {
        helios::helios_runtime_error("ERROR (IrrigationModel::calculateHydraulicsMultiZoneOptimized): Water source not in active nodes.");
        return HydraulicResults();
    }

    // std::cout << "Water source index = " << waterSourceIndex
    //           << " (node ID = " << waterSourceId << ")" << std::endl;

    // Build compressed sparse row structure
    std::vector<int> rowPtr(numNodes + 1, 0);
    std::vector<std::vector<int>> neighborIdxList(numNodes);
    std::vector<std::vector<Link*>> neighborLinkList(numNodes);
    std::vector<std::vector<double>> neighborResistance(numNodes);

    // Link lookup map
    auto makeKey = [](int a, int b) -> uint64_t {
        int lo = std::min(a, b);
        int hi = std::max(a, b);
        return (static_cast<uint64_t>(lo) << 32) | static_cast<uint32_t>(hi);
    };

    std::unordered_map<uint64_t, Link*> linkByKey;
    linkByKey.reserve(activeLinks.size() * 2);
    for (Link* link : activeLinks) {
        linkByKey[makeKey(link->from, link->to)] = link;
    }

    // Build neighbor lists and count non-zeros
    for (int i = 0; i < numNodes; ++i) {
        int nodeId = orderedNodeIds[i];
        if (activeNeighbors.count(nodeId)) {
            for (int neighborId : activeNeighbors[nodeId]) {
                auto it = nodeIndexMap.find(neighborId);
                if (it == nodeIndexMap.end()) continue;

                uint64_t key = makeKey(nodeId, neighborId);
                auto linkIt = linkByKey.find(key);
                if (linkIt == linkByKey.end()) continue;

                neighborIdxList[i].push_back(it->second);
                neighborLinkList[i].push_back(linkIt->second);
                neighborResistance[i].push_back(0.0); // Will be filled later
            }
        }
        // Each node has diagonal  & one entry per neighbor
        rowPtr[i + 1] = rowPtr[i] + 1 + static_cast<int>(neighborIdxList[i].size());
    }

    // Allocate CSR arrays
    std::vector<int> colIdx(rowPtr.back());
    std::vector<double> values(rowPtr.back(), 0.0);
    std::vector<double> RHS(numNodes, 0.0);

    // Fill column indices
    for (int i = 0; i < numNodes; ++i) {
        int base = rowPtr[i];
        colIdx[base] = i; // Diagonal

        for (size_t k = 0; k < neighborIdxList[i].size(); ++k) {
            colIdx[base + 1 + static_cast<int>(k)] = neighborIdxList[i][k];
        }
    }

    // Initialize flow variables
    std::vector<double> nodalPressure(numNodes, 0.0);
    std::vector<double> nodalPressure_old(numNodes, 0.0);

    // Set initial emitter flows
    double waterSourcePressurePa = Pw * 6894.76;
    for (int nodeId : orderedNodeIds) {
        Node& node = nodes[nodeId];
        if (node.type == "emitter" && node.isActive) {
            node.flow = Qspecified;
        }
    }

    // Main iteration loop
    double err = 1e6;
    int iter = 1;

    // set initial pressure guess for a start
    for (int i = 0; i < numNodes; ++i) {
        if (i == waterSourceIndex) {
            nodalPressure[i] = waterSourcePressurePa;
        } else {
            // Linear decay from source
            nodalPressure[i] = waterSourcePressurePa * 0.5; //check this decay
        }
    }

    while (std::abs(err) > err_max && iter < max_iter) {
       // std::cout << "Iteration " << iter << ", current error: " << err << std::endl;

        // Reset matrix and RHS
        std::fill(values.begin(), values.end(), 0.0);
        std::fill(RHS.begin(), RHS.end(), 0.0);

        // Build matrix and RHS
        for (int i = 0; i < numNodes; ++i) {
            int nodeId = orderedNodeIds[i];

            if (i == waterSourceIndex) {
                // Fixed pressure boundary condition
                int base = rowPtr[i];
                values[base] = 1.0;  // A[i][i] = 1
                RHS[i] = waterSourcePressurePa;  // b[i] = Pw
                continue;
            }

            int base = rowPtr[i];
            double diag = 0.0;

            if (neighborIdxList[i].empty()) {
                // Isolated node - no flow condition
                values[base] = 1.0;  // Pi = 0
                RHS[i] = 0.0;
                continue;
            }

            // For each neighbor
            for (size_t k = 0; k < neighborIdxList[i].size(); ++k) {
                int j = neighborIdxList[i][k];
                Link* link = neighborLinkList[i][k];

                // Calculate current flow velocity for this link
                double deltaP = nodalPressure[i] - nodalPressure[j];
                double W_bar_ij = 0.0;

                // Estimate current velocity (use previous iteration or initial guess)
                if (iter == 1) {
                    // Initial guess based on pressure difference
                    W_bar_ij = std::abs(deltaP) / (rho * 9.81 * link->length);
                } else {
                    // Use velocity from previous iteration
                    W_bar_ij = std::abs(link->flow) / (M_PI/4.0 * link->diameter * link->diameter);
                }

                // Reynolds number
                double Re_ij = std::abs(W_bar_ij) * link->diameter * rho / mu;

                // call minor loss coefficient
                double Kf_barb = minorLoss_kf(Re_ij, sprinklerAssemblyType);

                // Calculate resistance term
                double Rval = calculateResistance(Re_ij, W_bar_ij, Kf_barb, *link, iter);
                neighborResistance[i][k] = Rval;

                if (Rval > 1e-12) {
                    double invR = 1.0 / Rval;

                    // Diagonal contribution
                    diag += invR;

                    // Off-diagonal contribution
                    values[base + 1 + static_cast<int>(k)] = -invR;

                    // Elevation contributions to RHS
                    const auto& node_i = nodes[nodeId];
                    const auto& node_j = nodes[orderedNodeIds[j]];
                    double elevation_diff = node_i.position.z - node_j.position.z;
                    double elevation_pressure = rho * 9.81 * elevation_diff;

                    // Distribute elevation effect (half to each node)
                    RHS[i] += elevation_pressure * invR * 0.5;
                }
            }

            // Set diagonal
            values[base] = diag;

            // Flow demand (negative for outflow, positive for inflow)
            if (nodes[nodeId].type == "emitter") {
                RHS[i] -= nodes[nodeId].flow;  // Outflow is negative source
            }
        }

        // Solve linear system with Gauss-seidel
        std::vector<double> x = nodalPressure;

        // run Gauss-Seidel iterations
        const int gs_max_iter = 1000;
        const double gs_tol = 1e-6;

        for (int gs_iter = 0; gs_iter < gs_max_iter; ++gs_iter) {
            double max_diff = 0.0;

            for (int i = 0; i < numNodes; ++i) {
                if (i == waterSourceIndex) {
                    x[i] = waterSourcePressurePa;  // Fixed
                    continue;
                }

                double sum = RHS[i];
                double diag = values[rowPtr[i]];

                // Subtract contributions from neighbors
                for (int idx = rowPtr[i] + 1; idx < rowPtr[i + 1]; ++idx) {
                    int j = colIdx[idx];
                    sum -= values[idx] * x[j];
                }

                if (std::abs(diag) > 1e-12) {
                    double x_new = sum / diag;
                    max_diff = std::max(max_diff, std::abs(x_new - x[i]));
                    x[i] = x_new;
                }
            }

            if (max_diff < gs_tol) {
                std::cout << "  GS converged in " << gs_iter << " iterations" << std::endl;
                break;
            }
        }

        nodalPressure = x;

        // Update link flows based on new pressures
        for (int i = 0; i < numNodes; ++i) {
            for (size_t k = 0; k < neighborIdxList[i].size(); ++k) {
                int j = neighborIdxList[i][k];
                Link* link = neighborLinkList[i][k];

                double deltaP = nodalPressure[i] - nodalPressure[j];
                double Rval = neighborResistance[i][k];

                if (Rval > 1e-12) {
                    double Q = deltaP / Rval;
                    link->flow = std::abs(Q);
                }
            }
        }

        // Update emitter flows based on new pressures
        for (int i = 0; i < numNodes; ++i) {
            int nodeId = orderedNodeIds[i];
            Node& node = nodes[nodeId];

            if (node.type == "emitter" && node.isActive) {

                double pressure_psi = nodalPressure[i] / 6894.76;
                double new_flow = Qspecified;
                if (Pw > 0) {
                    new_flow = calculateEmitterFlow (sprinklerAssemblyType, pressure_psi,false); //default is True and enables global update

                    if (std::isnan(new_flow) || std::isinf(new_flow)) {
                        new_flow = 0.0;
                    }
                    node.flow = new_flow;
                }
            }
        }

        // Error calculation
        if (iter > 1) {
            err = 0.0;
            for (int i = 0; i < numNodes; ++i) {
                if (i == waterSourceIndex) continue;
                double diff = nodalPressure[i] - nodalPressure_old[i];
                err += diff * diff;
            }
            err = std::sqrt(err / (numNodes - 1));
          //  std::cout << "  Pressure change RMS: " << err << " Pa" << std::endl;
        }

        nodalPressure_old = nodalPressure;
        iter++;

        // Early convergence check
        if (err < err_max) {
            std::cout << "Converged at iteration " << iter << std::endl;
            break;
        }
    }

    // check pressure distribution
    std::cout << "\n Final pressure distribution" << std::endl;
    std::cout << "Node id | Type | Pressure (psi) | Flow (m3/s)" << std::endl;

    for (int i = 0; i < numNodes; ++i) {
        int nodeId = orderedNodeIds[i];
        const Node& node = nodes[nodeId];
        double pressure_psi = nodalPressure[i] / 6894.76;

        std::cout << std::setw(6) << nodeId << " | "
                  << std::setw(15) << node.type << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << pressure_psi << " | ";

        if (node.type == "emitter") {
            std::cout << std::scientific << std::setprecision(4) << node.flow;
        } else {
            std::cout << "N/A";
        }
        std::cout << std::endl;
    }

    // Check water source
    if (waterSourceIndex >= 0) {
        std::cout << "\nWater source: P = " << nodalPressure[waterSourceIndex]/6894.76
                  << " psi (target: " << Pw << " psi)" << std::endl;
    }

    // Prepare results
    HydraulicResults results;
    results.nodalPressures.resize(numNodes);
    results.flowRates.resize(activeLinks.size());
    results.converged = (err <= err_max);
    results.iterations = iter;

    // Convert pressures to psi and update nodes
    double totalEmitterFlow = 0.0;
    for (int i = 0; i < numNodes; ++i) {
        results.nodalPressures[i] = nodalPressure[i] / 6894.76;
        int nodeId = orderedNodeIds[i];
        nodes[nodeId].pressure = results.nodalPressures[i];

        // Collect emitter flows
        if (nodes[nodeId].type == "emitter" && nodes[nodeId].isActive) {
            results.emitterFlows.push_back(nodes[nodeId].flow);
            totalEmitterFlow += nodes[nodeId].flow;
        }
    }

    // Update link flows
    for (size_t i = 0; i < activeLinks.size(); ++i) {
        results.flowRates[i] = activeLinks[i]->flow;
    }

    std::cout << "\n Hydraulic result summary" << std::endl;
    std::cout << "Converged: " << (results.converged ? "YES" : "NO") << std::endl;
    std::cout << "Iterations: " << iter << std::endl;
    std::cout << "Final error: " << err << std::endl;
    std::cout << "Total emitter flow: " << totalEmitterFlow * 1000 * 3600 << " L/hr" << std::endl;
    std::cout << "Number of emitters: " << results.emitterFlows.size() << std::endl;

    if (!results.converged && iter >= max_iter) {
        helios::helios_runtime_error("ERROR (IrrigationModel::calculateHydraulicsMultiZoneOptimized): Solver did not converge in " +
            std::to_string(max_iter) + " iterations");
    }

    return results;
}

void IrrigationModel::clearActiveNodeCache()
{
    activeNodeCache.clear();
    currentValveConfigHash = 0;
    std::cout << "Active node cache cleared." << std::endl;
}

size_t IrrigationModel::getCacheSize() const
{
    return activeNodeCache.size();
}

void IrrigationModel::printActiveNodeStats() const
{
    std::cout << "\n Active Node Info" << std::endl;
    std::cout << "Cache size: " << activeNodeCache.size() << " configurations" << std::endl;
    std::cout << "Current valve configuration hash: " << currentValveConfigHash << std::endl;

    // Count active nodes in current configuration
    int activeCount = 0;
    for (const auto& [id, node] : nodes) {
        if (node.isActive) activeCount++;
    }
    std::cout << "Currently active nodes: " << activeCount << " / " << nodes.size() << std::endl;

    // Print valve states
    std::cout << "\nValve States:" << std::endl;
    for (const auto& [id, node] : nodes) {
        if (node.type == "zone_valve") {
            std::cout << "  Valve " << id << " (Zone " << node.zoneID
                      << "): " << (node.isValveOpen ? "OPEN" : "CLOSED") << std::endl;
        }
    }
}

void IrrigationModel::activateAllZones()
{
    std::cout << "Activating all zones..." << std::endl;
    for (auto& [id, node] : nodes) {
        if (node.type == "zone_valve") {
            node.isValveOpen = true;
        }
    }
    updateActiveNodesFast();
}

void IrrigationModel::deactivateAllZones()
{
    std::cout << "Deactivating all zones..." << std::endl;
    for (auto& [id, node] : nodes) {
        if (node.type == "zone_valve") {
            node.isValveOpen = false;
        }
    }
    updateActiveNodesFast();
}

void IrrigationModel::activateSingleZone(int zoneID)
{
    std::cout << "Activating single zone " << zoneID << "..." << std::endl;
    deactivateAllZones();
    openZoneValve(zoneID);
}

void IrrigationModel::activateZones(const std::vector<int>& zoneIDs)
{
    std::cout << "Activating zones: ";
    for (int zoneID : zoneIDs) {
        std::cout << zoneID << " ";
    }
    std::cout << std::endl;

    deactivateAllZones();
    for (int zoneID : zoneIDs) {
        openZoneValve(zoneID);
    }
}

void IrrigationModel::initialize()
{
    // valve-to-nodes mapping
    buildValveToNodesMapping();
    updateActiveNodesFast(); //default
}

// Provide both for flexibility
void IrrigationModel::setActiveNodeMethod(bool useFastMethod)
{
    if (useFastMethod) {
        updateActiveNodesFast();          // Use cached fast method
    } else {
        updateActiveNodesByTreeTraversal();         // Use simple tree traversal
    }
}



// // ─────────────── Global DXF-parsing constants ─────────────────────────── //
// namespace {
//     constexpr double NODE_TOL = 1.0e-6; // merge tolerance [m]
//     constexpr double MIN_SEG_LEN = 1.0; // skip segments < 1 m
//     constexpr bool ORTH_ONLY = true; // keep only h/v pipes
//     constexpr bool SKIP_CLOSED = true; // ignore closed polylines
//     const std::unordered_set<std::string> IGNORE_LAYERS = {"0"};
//
//     inline void trim(std::string &s) {
//         auto is_ws = [](unsigned char c) { return std::isspace(c); };
//         while (!s.empty() && is_ws(s.back()))
//             s.pop_back();
//         auto it = std::find_if_not(s.begin(), s.end(), is_ws);
//         s.erase(s.begin(), it);
//     }
//
//     std::map<int, std::string> pairsToMap(const std::vector<std::pair<int, std::string>> &vec) {
//         std::map<int, std::string> m;
//         for (const auto &kv: vec)
//             m[kv.first] = kv.second;
//         return m;
//     }
//
//     bool keepLayer(const std::string &layer) { return !IGNORE_LAYERS.count(layer); }
//
//     bool keepSegment(double x1, double y1, double x2, double y2) {
//         if (ORTH_ONLY) {
//             double dx = std::fabs(x2 - x1);
//             double dy = std::fabs(y2 - y1);
//             if (dx >= NODE_TOL && dy >= NODE_TOL)
//                 return false; // diagonal
//         }
//         return std::hypot(x2 - x1, y2 - y1) >= MIN_SEG_LEN;
//     }
// } // anonymous namespace
//
//
// // ─────────────── Node management ─────────────────────────────────────── //
// int IrrigationModel::getOrCreateNode(double x, double y) {
//     for (std::size_t i = 0; i < nodes.size(); ++i)
//         if (std::fabs(nodes[i].x - x) < NODE_TOL && std::fabs(nodes[i].y - y) < NODE_TOL)
//             return int(i);
//
//     nodes.push_back({x, y, 0.0, false});
//     return int(nodes.size() - 1);
// }
//
// // ─────────────── Pipe helper ─────────────────────────────────────────── //
// int IrrigationModel::addPipe(int n1, int n2, double L, double d, double kminor) {
//     if (n1 == n2)
//         return 1; // degenerate
//     pipes.push_back({n1, n2, L, d, kminor});
//     return 0;
// }
//
// // ─────────────── Entity: LINE ────────────────────────────────────────── //
// int IrrigationModel::parseLineEntity(const std::map<int, std::string> &ent) {
//     if (!(ent.count(10) && ent.count(20) && ent.count(11) && ent.count(21)))
//         return 1;
//
//     std::string layer = ent.count(8) ? ent.at(8) : "";
//     if (!keepLayer(layer))
//         return 0; // silently ignore
//
//     double x1 = std::stod(ent.at(10)), y1 = std::stod(ent.at(20));
//     double x2 = std::stod(ent.at(11)), y2 = std::stod(ent.at(21));
//     if (!keepSegment(x1, y1, x2, y2))
//         return 0;
//
//     double dia = ent.count(40) ? std::stod(ent.at(40)) : 0.05;
//     if (dia > 1.0)
//         dia /= 1000.0;
//
//     int n1 = getOrCreateNode(x1, y1);
//     int n2 = getOrCreateNode(x2, y2);
//     return addPipe(n1, n2, std::hypot(x2 - x1, y2 - y1), dia);
// }
//
// // ─────────────── Entity: LWPOLYLINE ─────────────────────────────────── //
// int IrrigationModel::parseLWPolylineEntity(const std::vector<std::pair<int, std::string>> &ent) {
//     std::vector<std::pair<double, double>> verts;
//     double curX = 0.0, dia = 0.05;
//     bool haveX = false, closed = false;
//     std::string layer;
//
//     for (const auto &kv: ent) {
//         int code = kv.first;
//         if (code == 8)
//             layer = kv.second;
//         else if (code == 10) {
//             curX = std::stod(kv.second);
//             haveX = true;
//         } else if (code == 20 && haveX) {
//             verts.emplace_back(curX, std::stod(kv.second));
//             haveX = false;
//         } else if (code == 40)
//             dia = std::stod(kv.second);
//         else if (code == 70)
//             closed = (std::stoi(kv.second) & 1) != 0;
//     }
//
//     if (!keepLayer(layer) || verts.size() < 2 || (closed && SKIP_CLOSED))
//         return 0;
//
//     if (dia > 1.0)
//         dia /= 1000.0;
//
//     for (std::size_t i = 1; i < verts.size(); ++i) {
//         auto [x1, y1] = verts[i - 1];
//         auto [x2, y2] = verts[i];
//         if (!keepSegment(x1, y1, x2, y2))
//             continue;
//         addPipe(getOrCreateNode(x1, y1), getOrCreateNode(x2, y2), std::hypot(x2 - x1, y2 - y1), dia);
//     }
//     if (closed && verts.size() > 2) {
//         auto [x1, y1] = verts.back();
//         auto [x2, y2] = verts.front();
//         if (keepSegment(x1, y1, x2, y2))
//             addPipe(getOrCreateNode(x1, y1), getOrCreateNode(x2, y2), std::hypot(x2 - x1, y2 - y1), dia);
//     }
//     return 0;
// }
//
// // ─────────────── Entity: POLYLINE + VERTEX ───────────────────────────── //
// int IrrigationModel::parsePolylineEntity(const std::vector<std::pair<int, std::string>> &header, const std::vector<std::vector<std::pair<int, std::string>>> &vertices) {
//     bool closed = false;
//     double dia = 0.05;
//     std::string layer;
//     for (const auto &kv: header) {
//         if (kv.first == 8)
//             layer = kv.second;
//         if (kv.first == 70)
//             closed = (std::stoi(kv.second) & 1) != 0;
//         if (kv.first == 40)
//             dia = std::stod(kv.second);
//     }
//
//     if (!keepLayer(layer) || (closed && SKIP_CLOSED))
//         return 0;
//     if (dia > 1.0)
//         dia /= 1000.0;
//
//     std::vector<std::pair<double, double>> verts;
//     for (const auto &vtx: vertices) {
//         auto v = pairsToMap(vtx);
//         if (v.count(10) && v.count(20))
//             verts.emplace_back(std::stod(v.at(10)), std::stod(v.at(20)));
//     }
//     if (verts.size() < 2)
//         return 0;
//
//     for (std::size_t i = 1; i < verts.size(); ++i) {
//         auto [x1, y1] = verts[i - 1];
//         auto [x2, y2] = verts[i];
//         if (!keepSegment(x1, y1, x2, y2))
//             continue;
//         addPipe(getOrCreateNode(x1, y1), getOrCreateNode(x2, y2), std::hypot(x2 - x1, y2 - y1), dia);
//     }
//     if (closed && verts.size() > 2) {
//         auto [x1, y1] = verts.back();
//         auto [x2, y2] = verts.front();
//         if (keepSegment(x1, y1, x2, y2))
//             addPipe(getOrCreateNode(x1, y1), getOrCreateNode(x2, y2), std::hypot(x2 - x1, y2 - y1), dia);
//     }
//     return 0;
// }
//
// // ─────────────── DXF reader (dispatch) ───────────────────────────────── //
// int IrrigationModel::readDXF(const std::string &filename) {
//     std::ifstream in(filename);
//     if (!in.is_open())
//         helios_runtime_error("ERROR(IrrigationModel::readDXF): cannot open \"" + filename + "\"");
//
//     std::vector<std::string> raw;
//     std::string ln;
//     while (std::getline(in, ln)) {
//         trim(ln);
//         raw.push_back(ln);
//     }
//     in.close();
//
//     std::size_t i = 0;
//     while (i + 1 < raw.size()) {
//         if (std::stoi(raw[i]) != 0) {
//             i += 2;
//             continue;
//         }
//         std::string tag = raw[i + 1];
//         i += 2;
//
//         if (tag == "LINE") {
//             std::vector<std::pair<int, std::string>> ent{{0, "LINE"}};
//             while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
//                 ent.emplace_back(std::stoi(raw[i]), raw[i + 1]);
//                 i += 2;
//             }
//             parseLineEntity(pairsToMap(ent));
//         } else if (tag == "LWPOLYLINE") {
//             std::vector<std::pair<int, std::string>> ent{{0, "LWPOLYLINE"}};
//             while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
//                 ent.emplace_back(std::stoi(raw[i]), raw[i + 1]);
//                 i += 2;
//             }
//             parseLWPolylineEntity(ent);
//         } else if (tag == "POLYLINE") {
//             std::vector<std::pair<int, std::string>> hdr{{0, "POLYLINE"}};
//             while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
//                 hdr.emplace_back(std::stoi(raw[i]), raw[i + 1]);
//                 i += 2;
//             }
//             std::vector<std::vector<std::pair<int, std::string>>> verts;
//             while (i + 1 < raw.size()) {
//                 if (std::stoi(raw[i]) != 0) {
//                     i += 2;
//                     continue;
//                 }
//                 std::string subt = raw[i + 1];
//                 if (subt == "VERTEX") {
//                     std::vector<std::pair<int, std::string>> v{{0, "VERTEX"}};
//                     i += 2;
//                     while (i + 1 < raw.size() && std::stoi(raw[i]) != 0) {
//                         v.emplace_back(std::stoi(raw[i]), raw[i + 1]);
//                         i += 2;
//                     }
//                     verts.emplace_back(std::move(v));
//                 } else if (subt == "SEQEND") {
//                     i += 2;
//                     break;
//                 } else {
//                     break;
//                 }
//             }
//             parsePolylineEntity(hdr, verts);
//         }
//     }
//
//     // ── pick the node nearest origin as supply (50 psi) ─────────────── //
//     int supply = -1;
//     double best = std::numeric_limits<double>::max();
//     for (std::size_t k = 0; k < nodes.size(); ++k) {
//         double r2 = nodes[k].x * nodes[k].x + nodes[k].y * nodes[k].y;
//         if (r2 < best) {
//             best = r2;
//             supply = int(k);
//         }
//     }
//     if (supply >= 0) {
//         nodes[supply].fixed = true;
//         nodes[supply].pressure = 50.0;
//     }
//
//     checkConnectivity();
//     return 0;
// }
//
// // ─────────────── Connectivity check ────────────────────────────────────── //
// void IrrigationModel::checkConnectivity() const {
//     if (nodes.empty())
//         return;
//
//     std::vector<char> seen(nodes.size(), 0);
//
//     std::queue<int> q;
//     for (std::size_t i = 0; i < nodes.size(); ++i) {
//         if (nodes[i].fixed) {
//             q.push(int(i));
//             seen[i] = 1;
//         }
//     }
//
//     while (!q.empty()) {
//         int u = q.front();
//         q.pop();
//         for (const auto &p: pipes) {
//             int v = (p.n1 == u) ? p.n2 : (p.n2 == u) ? p.n1 : -1;
//             if (v >= 0 && !seen[v]) {
//                 seen[v] = 1;
//                 q.push(v);
//             }
//         }
//     }
//     for (std::size_t i = 0; i < nodes.size(); ++i)
//         if (!seen[i])
//             helios_runtime_error("ERROR (IrrigationModel): Node " + std::to_string(i) +
//                                  " is disconnected from the fixed-pressure "
//                                  "reference.");
// }
//
// // ─────────────── Linear solver (Gauss-Jordan) ─────────────────────────── //
// std::vector<double> IrrigationModel::solveLinear(std::vector<std::vector<double>> A, std::vector<double> b) const {
//     const int n = int(A.size());
//     for (int i = 0; i < n; ++i) {
//         int pivot = i;
//         for (int j = i + 1; j < n; ++j)
//             if (std::fabs(A[j][i]) > std::fabs(A[pivot][i]))
//                 pivot = j;
//
//         std::swap(A[i], A[pivot]);
//         std::swap(b[i], b[pivot]);
//         double diag = A[i][i];
//         if (std::fabs(diag) < 1e-12)
//             throw std::runtime_error("Singular hydraulic matrix");
//         for (int j = i; j < n; ++j)
//             A[i][j] /= diag;
//         b[i] /= diag;
//
//         for (int k = 0; k < n; ++k)
//             if (k != i) {
//                 double f = A[k][i];
//                 for (int j = i; j < n; ++j)
//                     A[k][j] -= f * A[i][j];
//                 b[k] -= f * b[i];
//             }
//     }
//     return b;
// }
//
// // ─────────────── Solve pressures ───────────────────────────────────────── //
// double IrrigationModel::pipeResistance(const IrrigationPipe &p) {
//     // Darcy–Weisbach with Blasius f = 0.316 Re^-0.25 at Re≈1e5
//     const double nu = 1.0e-6; // kinematic viscosity (m²/s)
//     const double rho = 1000.0; // density (kg/m³)
//     const double g = 9.80665;
//     const double Q = 1.0e-4; // design flow (m³/s) – stub
//     const double A = M_PI * p.diameter * p.diameter / 4.0;
//     const double v = Q / A;
//     const double Re = v * p.diameter / nu;
//     const double f = 0.3164 / std::pow(Re, 0.25);
//     return 32.0 * f * rho * p.length / (M_PI * M_PI * std::pow(p.diameter, 5) * g);
// }
//
// int IrrigationModel::solve() {
//     const std::size_t n = nodes.size();
//     std::vector<int> mapIndex(n, -1);
//     int unknowns = 0;
//     for (std::size_t i = 0; i < n; ++i)
//         if (!nodes[i].fixed)
//             mapIndex[i] = unknowns++;
//
//     std::vector<std::vector<double>> A(unknowns, std::vector<double>(unknowns, 0.0));
//     std::vector<double> b(unknowns, 0.0);
//
//     for (const auto &p: pipes) {
//         double R = pipeResistance(p);
//         int i = p.n1, j = p.n2;
//         int ai = mapIndex[i], aj = mapIndex[j];
//
//         if (ai >= 0) {
//             A[ai][ai] += 1.0 / R;
//             if (aj >= 0)
//                 A[ai][aj] -= 1.0 / R;
//             else
//                 b[ai] += nodes[j].pressure / R;
//         }
//         if (aj >= 0) {
//             A[aj][aj] += 1.0 / R;
//             if (ai >= 0)
//                 A[aj][ai] -= 1.0 / R;
//             else
//                 b[aj] += nodes[i].pressure / R;
//         }
//     }
//
//     std::vector<double> x = solveLinear(A, b);
//     for (std::size_t i = 0; i < n; ++i)
//         if (mapIndex[i] >= 0)
//             nodes[i].pressure = x[mapIndex[i]];
//     return 0;
// }
//
// // ─────────────── DXF writer (unchanged except for const tweaks) ───────── //
// int IrrigationModel::writeDXF(const std::string& filename) const
// {
//     std::ofstream out(filename);
//     if (!out.is_open())
//         helios_runtime_error("ERROR(IrrigationModel::writeDXF): cannot open \""
//                              + filename + "\" for writing.");
//
//     std::cout << "Writing DXF \"" << filename << "\"… " << std::flush;
//
//     out << "0\nSECTION\n2\nENTITIES\n";
//
//     // ── pipes as LINE on layer PIPE_NET ─────────────────────────────── //
//     for (const auto& p : pipes) {
//         const auto& n1 = nodes[p.n1];
//         const auto& n2 = nodes[p.n2];
//         out << "0\nLINE\n8\nPIPE_NET\n62\n5\n"      // blue
//             << "10\n" << n1.x << "\n20\n" << n1.y
//             << "\n11\n" << n2.x << "\n21\n" << n2.y << "\n";
//     }
//
//     // ── node pressures as TEXT on layer PRESSURE ───────────────────── //
//     for (const auto& node : nodes) {
//         out << "0\nTEXT\n8\nPRESSURE\n62\n1\n"      // red
//             << "10\n" << node.x << "\n20\n" << node.y
//             << "\n40\n0.2\n1\n"                     // height 0.2; string follows
//             << std::fixed << std::setprecision(2) << node.pressure << "\n";
//     }
//
//     out << "0\nENDSEC\n0\nEOF\n";
//     std::cout << "done.\n";
//     return 0;
// }
//
// int IrrigationModel::selfTest() {
//     std::cout << "Running irrigation plug-in self-test..." << std::endl;
//     readDXF("../plugins/irrigation/doc/simple_network.dxf");
//     solve();
//     writeDXF("../plugins/irrigation/doc/simple_network_out.dxf");
//     for (size_t i = 0; i < nodes.size(); ++i) {
//         std::cout << "Node " << i << " pressure: " << nodes[i].pressure << std::endl;
//     }
//     return 0;
// }
