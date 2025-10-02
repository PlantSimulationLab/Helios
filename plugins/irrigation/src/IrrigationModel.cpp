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
#include <unordered_set>  // For std::unordered_set
#include <queue>          // For std::queue
#include <iomanip>  // for std::setw and std::setprecision

#include <algorithm> // For std::find

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
    summary += "========================\n";
    summary += "Total nodes: " + std::to_string(nodes.size()) + "\n";
    summary += "Total links: " + std::to_string(links.size()) + "\n";

    // Node type counts
    std::unordered_map<std::string, int> nodeCounts;
    for (const auto& [id, node] : nodes) {
        nodeCounts[node.type]++;
    }

    summary += "\nNode Types:\n";
    for (const auto& [type, count] : nodeCounts) {
        summary += "- " + type + ": " + std::to_string(count) + "\n";
    }

    // Link type counts
    std::unordered_map<std::string, int> linkCounts;
    for (const auto& link : links) {
        linkCounts[link.type]++;
    }

    summary += "\nPipe Types:\n";
    for (const auto& [type, count] : linkCounts) {
        summary += "- " + type + ": " + std::to_string(count) + "\n";
    }

    summary += "========================\n";
    return summary;
}



Position IrrigationModel::calculateWaterSourcePosition(double fieldLength, double fieldWidth,
                                                     const std::string& lateralDirection) const {
    if (lateralDirection == "vertical") {
        return {
            //fieldLength / 3.0 + 5.0,  // x
            18.4112,
            fieldWidth - fieldWidth / 3.0 - 0.5  // y
        };
    } else {
        return {
            fieldLength / 2.0,  // x
            0.0  // y
        };
    }
}

void IrrigationModel::createSprinklerSystemGeneral(double fieldLength, double fieldWidth,
                                          double sprinklerSpacing, double lineSpacing,
                                          const std::string& connectionType,
                                          const std::string& sprinklerConfig) {
    std::cout<<"Sprinkler Config: " <<sprinklerConfig<<"\n";
    const SprinklerAssembly& config = sprinklerLibrary.getSprinklerType(sprinklerConfig);
    double stake_height = config.stakeHeight;

    const int num_laterals = static_cast<int>(std::ceil(fieldLength / lineSpacing)) + 1;//x direction
    const int num_sprinklers_perline = static_cast<int>(std::ceil(fieldWidth /sprinklerSpacing)) + 1; // y direction
    int nodeId = 1;
    const double barbOffset = 0;
    const double emitterOffset = 0;

    // Grid to track valid junction nodes for lateral connections
    std::vector<std::vector<int>> junctionGrid(num_laterals, std::vector<int>(num_sprinklers_perline, -1));

    // Create sprinklers only inside polygon
    for (int i = 0; i < num_laterals; ++i) {
        for (int j = 0; j < num_sprinklers_perline; ++j) {
            double x = i * lineSpacing;
            double y = j * sprinklerSpacing;

            Position testPoint{x, y};

            // Skip points outside the boundary polygon
            if (!boundaryPolygon.empty() && !isPointInsidePolygon(testPoint)) {
                continue;
            }

            // Create junction node for sprinkler assembly connection
            nodes[nodeId] = {nodeId, "lateral_sprinkler_jn", {x, y}, 0.0, false};
            int junctionId = nodeId++;
            junctionGrid[i][j] = junctionId;

            // Create barb and emitter nodes (existing code)
            double barbX = x + barbOffset * cos(M_PI/4);
            double barbY = y + barbOffset * sin(M_PI/4);

            nodes[nodeId] = {nodeId, "barb", {barbX, barbY}, 0.0, false};
            int barbId = nodeId++;

            double emitterX = barbX + emitterOffset * cos(M_PI/4);
            double emitterY = barbY + emitterOffset * sin(M_PI/4);
            nodes[nodeId] = {nodeId, "emitter", {emitterX, emitterY, stake_height}, 0.0, false};
            int emitterId = nodeId++;

            // Connect barb to emitter
            links.push_back({
                barbId, emitterId,
                config.barbToEmitter.diameter, // 0.154* INCH_TO_METER,
                config.barbToEmitter.length, //30 * INCH_TO_METER,
                "barbToemitter",
                0.0
            });

            // Connect junction to barb
            links.push_back({
                junctionId, barbId,
                config.lateralToBarb.diameter, //0.12 * INCH_TO_METER,
                config.lateralToBarb.length, // 0.5 * INCH_TO_METER,
                "lateralTobarb",
                0.0
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
                    "lateral"
                });
            }
            else if (connectionType == "horizontal" && i < num_laterals - 1 &&
                     junctionGrid[i+1][j] != -1) {
                links.push_back({
                    junctionGrid[i][j], junctionGrid[i+1][j],
                    0.67 * INCH_TO_METER,
                    sprinklerSpacing,
                    "lateral"
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

    // Similar implementation as above but for lateral→barb
    // Expected range: 0.4-0.9 psi
}


void IrrigationModel::addSubmainAndWaterSource(double fieldLength, double fieldWidth,
                                             const std::string& lateralDirection,
                                             SubmainPosition submainPosition) {
    // Determine submain position based on both direction and placement option
    double submainY = 0.0;
    double submainX = 0.0;

    if (lateralDirection == "vertical") {
        // For vertical laterals, use position options
        switch (submainPosition) {
            case SubmainPosition::NORTH:
                submainY = fieldWidth;
                break;
            case SubmainPosition::SOUTH:
                submainY = 0.0;
                break;
            case SubmainPosition::MIDDLE:
                submainY = fieldWidth - fieldWidth / 3.0 - 0.5; // manual setting
                break;
        }
    } else {
        // For horizontal laterals, place submain at x midpoint
        submainX = fieldLength / 2.0 + 1.0;
    }

    // Find all lateral links that connect to the submain
    std::vector<int> connectingLaterals;
    std::vector<int> connectingNodes;

    if (submainPosition == SubmainPosition::MIDDLE) {
        // Use intersection method for middle position
        for (size_t i = 0; i < links.size(); ++i) {
            const auto& link = links[i];
            if (link.type == "lateral") {
                const auto& fromNode = nodes[link.from];
                const auto& toNode = nodes[link.to];

                if (lateralDirection == "vertical") {
                    // Check if this lateral intersects with submain Y position
                    if ((fromNode.position.y < submainY && toNode.position.y > submainY) ||
                        (fromNode.position.y > submainY && toNode.position.y < submainY)) {
                        connectingLaterals.push_back(i);
                    }
                } else {
                    // For horizontal laterals, check X intersection
                    if ((fromNode.position.x < submainX && toNode.position.x > submainX) ||
                        (fromNode.position.x > submainX && toNode.position.x < submainX)) {
                        connectingLaterals.push_back(i);
                    }
                }
            }
        }
    } else {
        // For north/south positions, find nodes at the edge
        for (const auto& [id, node] : nodes) {
            if (node.type == "lateral_sprinkler_jn") {
                if (lateralDirection == "vertical") {
                    if (std::abs(node.position.y - submainY) < 0.001) {
                        connectingNodes.push_back(id);
                    }
                } else {
                    if (std::abs(node.position.x - submainX) < 0.001) {
                        connectingNodes.push_back(id);
                    }
                }
            }
        }
    }

    // Create submain nodes and connections
    int submainNodeId = getNextNodeId();
    int submainLinkId = links.size();

    if (submainPosition == SubmainPosition::MIDDLE) {
        // Middle position - use intersection method
        for (size_t i = 0; i < connectingLaterals.size(); ++i) {
            int lateralIdx = connectingLaterals[i];
            auto& lateral = links[lateralIdx];

            // Calculate intersection point
            Position fromPos = nodes[lateral.from].position;
            Position toPos = nodes[lateral.to].position;
            Position intersectPos;

            if (lateralDirection == "vertical") {
                intersectPos = {fromPos.x, submainY};
            } else {
                intersectPos = {submainX, fromPos.y};
            }

            // Create new submain node
            nodes[submainNodeId] = {
                submainNodeId,
                "submain_junction",
                intersectPos,
                0.0,
                false,
                0.0
            };

            // Split the lateral into two segments
            int originalTo = lateral.to;

            // First segment: from original from to new submain node
            lateral.to = submainNodeId;
            lateral.length = fromPos.distanceTo(intersectPos);

            // Second segment: from submain node to original to
            int newLateralId = links.size();
            links.push_back({
                submainNodeId,
                originalTo,
                0.67 * INCH_TO_METER,
                toPos.distanceTo(intersectPos), //split half sum not add up sprinklerSpacing
                "lateral"
            });

            // Create connection from submain node to submain
            if (i < connectingLaterals.size() - 1) {
                int nextSubmainNodeId = submainNodeId + 1;

                // Get the NEXT lateral's intersection point
                auto& nextLateral = links[connectingLaterals[i+1]];
                Position nextFromPos = nodes[nextLateral.from].position;
                Position nextIntersectPos = {nextFromPos.x, submainY};  // For vertical

                // Calculate ACTUAL pipe length needed
                double spacing = (lateralDirection == "vertical") ?
                    std::abs(nextIntersectPos.x - intersectPos.x) :  // Horizontal distance
                    std::abs(nextIntersectPos.y - intersectPos.y);   // Vertical distance

                // Ensure minimum length
                spacing = std::max(spacing, 0.1);  // At least 10cm

                links.push_back({
                    submainNodeId,
                    nextSubmainNodeId,
                    4.0 * INCH_TO_METER,
                    spacing,
                    "submain"
                });
            }

            submainNodeId++;
        }
    } else {
        // North/South positions then connect edge nodes directly
        std::sort(connectingNodes.begin(), connectingNodes.end(), [this](int a, int b) {
            return nodes[a].position.x < nodes[b].position.x;
        });

        for (size_t i = 0; i < connectingNodes.size(); ++i) {
            Position pos = nodes[connectingNodes[i]].position;

            // Adjust position to be exactly on submain line
            if (lateralDirection == "vertical") {
                pos.y = submainY;
            } else {
                pos.x = submainX;
            }

            // Create submain connection node
            nodes[submainNodeId] = {
                submainNodeId,
                "submain_junction",
                pos,
                0.0,
                false,
                0.0
            };

            // Connect lateral to submain
            links.push_back({
                connectingNodes[i],
                submainNodeId,
                0.67 * INCH_TO_METER,
                nodes[connectingNodes[i]].position.distanceTo(pos),
                "lateralToSubmain"
            });

            // Connect to previous submain node if exists
            if (i > 0) {
                double dist = pos.distanceTo(nodes[submainNodeId-1].position);

                if (dist > 0) {
                    links.push_back({
                        submainNodeId - 1,
                        submainNodeId,
                        4.0 * INCH_TO_METER,
                        dist,
                        "submain"
                    });
                }
                std::cout << "Created submain node " << submainNodeId << " at (" << pos.x << "," << pos.y << ")\n";
            }

            submainNodeId++;
        }
    }

     // Create water source node (ALWAYS as the last node)
    waterSourceId = getNextNodeId();
    Position waterSourcePos;

    if (lateralDirection == "vertical") {
        // manual setting: water source position logic for vertical laterals
        waterSourcePos = {fieldLength /2-2, fieldWidth - fieldWidth / 3.0 +2};

        // Adjust based on submain position
        if (submainPosition == SubmainPosition::NORTH) {
            waterSourcePos.y = fieldWidth - 1.0;
        } else if (submainPosition == SubmainPosition::SOUTH) {
            waterSourcePos.y = 1.0;
        }
    } else {
        // position for horizontal laterals
        waterSourcePos = {fieldLength / 2.0, 0.0};
    }

    nodes[waterSourceId] = {
        waterSourceId,
        "waterSource",
        waterSourcePos,
        0.0,
        true
    };

    // Find closest submain LINK (not node) to water source
    int closestLinkIndex = -1;
    double minDistance = std::numeric_limits<double>::max();
    Position closestPoint;

    // Determine submain Y position for range checking
    if (lateralDirection == "vertical") {
        switch (submainPosition) {
            case SubmainPosition::NORTH:
                submainY = fieldWidth;
                break;
            case SubmainPosition::SOUTH:
                submainY = 0.0;
                break;
            case SubmainPosition::MIDDLE:
                submainY = fieldWidth - fieldWidth / 3.0 - 0.5;
                break;
        }
    }

    for (size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];
        if (link.type != "submain") continue;

        const auto& nodeA = nodes[link.from];
        const auto& nodeB = nodes[link.to];

        // Check if both endpoints are submain_junction nodes
        if (nodeA.type != "submain_junction" || nodeB.type != "submain_junction") {
            continue;
        }

        // Check if water source X position is within this submain segment's range
        double minX = std::min(nodeA.position.x, nodeB.position.x);
        double maxX = std::max(nodeA.position.x, nodeB.position.x);

        if (waterSourcePos.x >= minX && waterSourcePos.x <= maxX) {
            // Calculate closest point on the segment to water source
            Position vecAB = {nodeB.position.x - nodeA.position.x, nodeB.position.y - nodeA.position.y};
            Position vecAW = {waterSourcePos.x - nodeA.position.x, waterSourcePos.y - nodeA.position.y};
            double dotProduct = vecAB.x * vecAW.x + vecAB.y * vecAW.y;
            double lengthSquared = vecAB.x * vecAB.x + vecAB.y * vecAB.y;

            // Avoid division by zero
            if (lengthSquared < 1e-12) continue;

            double t = std::max(0.0, std::min(1.0, dotProduct / lengthSquared));

            Position pointOnSegment = {
                nodeA.position.x + t * vecAB.x,
                nodeA.position.y + t * vecAB.y
            };

            double dist = waterSourcePos.distanceTo(pointOnSegment);
            if (dist < minDistance) {
                minDistance = dist;
                closestLinkIndex = i;
                closestPoint = pointOnSegment;
            }
        }
    }

    if (closestLinkIndex != -1) {
        //create new junction on the submain
        int newJunctionId = getNextNodeId();
        nodes[newJunctionId] = {
            newJunctionId,
            "submain_junction",
            closestPoint,
            0.0,
            false
        };

        // splitting the closest submain link
        const Link& oldLink = links[closestLinkIndex];
        double length1 = nodes[oldLink.from].position.distanceTo(closestPoint);
        double length2 = nodes[oldLink.to].position.distanceTo(closestPoint);

        links.push_back({
            oldLink.from,
            newJunctionId,       // to: new submain junction
            oldLink.diameter,
            length1,
            "submain"
        });
        links.push_back({
            newJunctionId,
            oldLink.to,
            oldLink.diameter,
            length2,
            "submain"
        });
        links.erase(links.begin() + closestLinkIndex);         // Remove the old link and add two new submain segments

        // Connect new submain junction (second last) to water source (last)
        links.push_back({
            newJunctionId,
            waterSourceId,
            2.0 * INCH_TO_METER,
            waterSourcePos.distanceTo(closestPoint),
            "mainline"
        });
    } else {
        // Fallback: connect to closest existing SUBMAIN JUNCTION node only
        int closestSubmainId = -1;
        double minDist = std::numeric_limits<double>::max();
        for (const auto& [id, node] : nodes) {
            if (node.type == "submain_junction") {
                double dist = node.position.distanceTo(waterSourcePos);
                if (dist < minDist) {
                    minDist = dist;
                    closestSubmainId = id;
                }
            }
        }

        if (closestSubmainId != -1) {
            // Connect water source (last node) to closest existing submain junction
            links.push_back({
                closestSubmainId,
                waterSourceId,
                2.0 * INCH_TO_METER,
                minDist,
                "mainline"
            });
        } else {
            // Create a new submain junction (this will be second last node)
            int newSubmainId = getNextNodeId();
            Position newSubmainPos = (lateralDirection == "vertical") ?
                Position{fieldLength / 2.0, submainY} :
                Position{submainX, fieldWidth / 2.0};

            nodes[newSubmainId] = {
                newSubmainId,
                "submain_junction",
                newSubmainPos,
                0.0,
                false,
                0.0
            };

            // Connect new submain junction (second last) to water source (last)
            links.push_back({
                newSubmainId,          // from: new submain junction (second last)
                waterSourceId,         // to: water source (always last node)
                2.0 * INCH_TO_METER,
                waterSourcePos.distanceTo(newSubmainPos),
                "mainline"
            });
        }
    }
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

void IrrigationModel::createIrregularSystem(double Pw, const std::vector<Position>& boundary,
                                         double sprinklerSpacing, double lineSpacing,
                                         const std::string& connectionType,
                                         const std::string& sprinklerConfig,
                                         SubmainPosition submainPos) {
    // Clear existing system
    nodes.clear();
    links.clear();

    setBoundaryPolygon(boundary);

    if (Pw <= 0) {
        helios::helios_runtime_error("ERROR (IrrigationModel::createIrregularSystem(): Water source pressure must be positive");
     //   throw std::invalid_argument("Water source pressure must be positive. Got: " + std::to_string(Pw));
    }

    // Validate boundary area is reasonable
    if (boundary.size() < 3) {
        helios::helios_runtime_error("ERROR (IrrigationModel::createIrregularSystem(): invalid Boundary size");
    //    throw std::invalid_argument("Boundary must have at least 3 points");
    }

    if (connectionType != "vertical" && connectionType != "horizontal") {
        helios::helios_runtime_error("ERROR (IrrigationModel::createSprinklerSystemGeneral(): invalid lateral connection orientation");
     //   throw std::invalid_argument("Invalid lateral connection orientation: " + connectionType);

    }

    // bounding box for grid generation
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

    // Create sprinkler system within the boundary
    createSprinklerSystemGeneral(fieldLength, fieldWidth, sprinklerSpacing, lineSpacing, connectionType,  sprinklerConfig);

    // Add submain and water source (modified to handle irregular layouts)
    addSubmainAndWaterSourceIrregular(fieldLength, fieldWidth, connectionType, submainPos);

    // Validate and set water source pressure
    validateParameters(fieldLength, fieldWidth, sprinklerSpacing, lineSpacing);
    //validateHydraulicSystem();

    if (waterSourceId != -1) {
        nodes[waterSourceId].pressure = Pw;
        nodes[waterSourceId].is_fixed = true;
    }
}

void IrrigationModel::addSubmainAndWaterSourceIrregular(double fieldLength, double fieldWidth,
                                                      const std::string& lateralDirection,
                                                      SubmainPosition submainPosition) {
    // Create submain nodes
    // should it be placed at the center of the row, or center of the chosen lateral?
    Position submainLinePos = calculateOptimalSubmainPosition(submainPosition, fieldLength, fieldWidth);
    std::vector<int> submainNodeIds;

    if (lateralDirection == "vertical") {
      submainNodeIds = createHorizontalSubmain(submainLinePos, fieldLength);
    } else {
        submainNodeIds = createVerticalSubmain(submainLinePos, fieldWidth);
    }
    connectSubmainToLaterals(submainNodeIds, lateralDirection);

     // Create water source and final connections
     createAndConnectWaterSource(submainNodeIds);

    // Final validation
     // validateHydraulicSystem();
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
        std::cout << "  Considering lateral " << lateral.from << "-" << lateral.to
                  << " from (" << fromNode.position.x << "," << fromNode.position.y << ")"
                  << " to (" << toNode.position.x << "," << toNode.position.y << ")"
                  << ", Distance: " << distance << std::endl;
    }

    if (optimalLateral) {
        const auto& fromNode = nodes.at(optimalLateral->from);
        const auto& toNode = nodes.at(optimalLateral->to);
        std::cout << "✅ Selected optimal lateral: " << optimalLateral->from << "-"
                  << optimalLateral->to
                  << " from (" << fromNode.position.x << "," << fromNode.position.y << ")"
                  << " to (" << toNode.position.x << "," << toNode.position.y << ")"
                  << ", Distance: " << bestScore << std::endl;
    }

    return optimalLateral;
}


double IrrigationModel::calculateLateralLength(const Link* lateral) const {
    const auto& fromNode = nodes.at(lateral->from);
    const auto& toNode = nodes.at(lateral->to);
    return fromNode.position.distanceTo(toNode.position);
}

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

void IrrigationModel::connectSubmainToLaterals(
    const std::vector<int>& submainNodeIds,
    const std::string& lateralDirection)
{
    //  Group laterals by their row position
    std::map<double, std::vector<size_t>> lateralsByRow;

    for (size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];
        if (link.type != "lateral") continue;

        if (!validateLink(link)) {
            std::cerr << "Warning: Invalid lateral link " << link.from
                      << "-" << link.to << std::endl;
            continue;
        }

        double rowKey = getLateralRowKey(link, lateralDirection);
        lateralsByRow[rowKey].push_back(i);
    }

    // Track connected rows
    std::set<double> connectedRows;

    //  Loop through each submain node
    for (int submainNodeId : submainNodeIds) {
        const auto& submainPos = nodes.at(submainNodeId).position;

        for (const auto& [rowKey, lateralIndices] : lateralsByRow) {
            if (lateralIndices.empty()) continue;
            if (connectedRows.find(rowKey) != connectedRows.end()) continue;

            std::cout << "\n--- Processing Row Key: " << rowKey
                      << " --- Laterals in this row: " << lateralIndices.size() << std::endl;

            //  Find the optimal lateral in this row
            const Link* optimalLateral = findOptimalLateralForConnection(
                lateralsByRow,
                submainPos,
                rowKey,
                lateralDirection
            );

            if (!optimalLateral) {
                std::cout << " No optimal lateral found for row " << rowKey << std::endl;
                continue;
            }

            // split lateral and connect to submain
            splitLateralAndConnect(optimalLateral, submainNodeId, lateralsByRow, lateralDirection);

            connectedRows.insert(rowKey);  // Mark this row as connected

            const auto& fromNode = nodes.at(optimalLateral->from);
            const auto& toNode = nodes.at(optimalLateral->to);
            std::cout << " Connected lateral " << optimalLateral->from << "-"
                      << optimalLateral->to << " to submain node " << submainNodeId << std::endl;
            break;
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

int IrrigationModel::findNearestSubmainNode(const Position& point,
                                          const std::vector<int>& submainNodeIds,
                                          double expectedRowKey,
                                          const std::string& lateralDirection,
                                          double tolerance) const {
    int nearestId = -1;
    double minDistance = std::numeric_limits<double>::max();

    std::cout << "   Looking for submain in row " << expectedRowKey
              << " (tolerance: ±" << tolerance << ")" << std::endl;

    for (int submainId : submainNodeIds) {
        const auto& submainPos = nodes.at(submainId).position;

        double submainRowKey;
        if (lateralDirection == "vertical") {
            submainRowKey = std::round(submainPos.x * 100.0) / 100.0;
        } else {
            submainRowKey = std::round(submainPos.y * 100.0) / 100.0;
        }

        double rowDifference = std::abs(submainRowKey - expectedRowKey);

        if (rowDifference > tolerance) {
            std::cout << "     Submain " << submainId << " at (" << submainPos.x << "," << submainPos.y
                      << ") is in row " << submainRowKey << " - TOO FAR (diff: " << rowDifference << ")" << std::endl;
            continue;
        }

        double distance = point.distanceTo(submainPos);
        std::cout << "     Submain " << submainId << " at (" << submainPos.x << "," << submainPos.y
                  << ") in row " << submainRowKey << " - distance: " << distance << "m" << std::endl;

        if (distance < minDistance) {
            minDistance = distance;
            nearestId = submainId;
        }
    }

    if (nearestId != -1) {
        std::cout << "   Selected submain " << nearestId << " with distance " << minDistance << "m" << std::endl;
    } else {
        std::cout << "   No suitable submain found in row " << expectedRowKey << std::endl;
    }

    return nearestId;
}

double IrrigationModel::calculateLateralCentrality(const Link* lateral,
                                                 const std::vector<const Link*>& rowLaterals) const {
    // Calculate how central this lateral is within its row
    std::vector<double> positions;
    for (const Link* l : rowLaterals) {
        positions.push_back(calculateLateralMidpoint(l).y); // Use Y for vertical laterals
    }

    std::sort(positions.begin(), positions.end());
    double median = positions[positions.size() / 2];

    double currentPos = calculateLateralMidpoint(lateral).y;
    return std::abs(currentPos - median) / median;
}

bool IrrigationModel::validateLink(const Link& link) const {
    return nodes.count(link.from) > 0 && nodes.count(link.to) > 0;
}

Position IrrigationModel::calculateSubmainLateralIntersection(const Position& submainPos,
                                                            const Position& lateralStart,
                                                            const Position& lateralEnd,
                                                            const std::string& lateralDirection) const {
    if (lateralDirection == "vertical") {
        // Horizontal submain + vertical lateral: intersection at (submainX, lateralY)
        return {submainPos.x, lateralStart.y};
    } else {
        // Vertical submain + horizontal lateral: intersection at (lateralX, submainY)
        return {lateralStart.x, submainPos.y};
    }
}
void IrrigationModel::splitLateralAndConnect(
    const Link* lateral,
    int submainNodeId,
    std::map<double, std::vector<size_t>>& rowLaterals,
    const std::string& lateralDirection)
{
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
        std::cerr << " Could not find the original lateral to split: "
                  << lateral->from << "-" << lateral->to << std::endl;
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

    // now erase the original lateral using indexing
    links.erase(links.begin() + lateralIndex);

    // Adjust rowLaterals indices for all remaining laterals (must update to avoid empty indexing)
    for (auto& [key, vec] : rowLaterals) {
        for (auto& idx : vec) {
            if (idx > lateralIndex) idx--;
        }
    }

    // Calculate lengths of new segments
    double length1 = fromNode.position.distanceTo(submainPos);
    double length2 = submainPos.distanceTo(toNode.position);

    // reconnect the two new lateral segments
    links.push_back({originalLateral.from, submainNodeId,
                     originalLateral.diameter, length1, "lateralToSubmain"});
    links.push_back({submainNodeId, originalLateral.to,
                     originalLateral.diameter, length2, "lateralToSubmain"});

    // Update rowLaterals with new segment indices
    size_t indexNew1 = links.size() - 2;
    size_t indexNew2 = links.size() - 1;

    double rowKey1 = getLateralRowKey(links[indexNew1], lateralDirection);
    double rowKey2 = getLateralRowKey(links[indexNew2], lateralDirection);

    rowLaterals[rowKey1].push_back(indexNew1);
    rowLaterals[rowKey2].push_back(indexNew2);

    std::cout << "Split lateral " << originalLateral.from << "-" << originalLateral.to
              << " into " << originalLateral.from << "-" << submainNodeId
              << " and " << submainNodeId << "-" << originalLateral.to << std::endl;
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

std::vector<int> IrrigationModel::createHorizontalSubmain(const Position& startPos, double fieldLength) {
    std::vector<int> submainNodeIds;

    // get lateral pipe X-positions and store lateral links
    std::map<double, const Link*> lateralMap; // X-position -> Lateral link

    for (const auto& link : links) {
        if (link.type == "lateral") {
            const auto& fromNode = nodes.at(link.from);
            const auto& toNode = nodes.at(link.to);

            // Use midpoint of lateral pipe for better alignment
            double midpointX = (fromNode.position.x + toNode.position.x) / 2.0;
            lateralMap[midpointX] = &link;
        }
    }

    if (lateralMap.empty()) {
        // fall backt to make sure evenly spaced nodes
        int numNodes = std::max(2, static_cast<int>(fieldLength / (22.0*FEET_TO_METER))); //assign minimum of 3 nodes
        for (int i = 0; i < numNodes; ++i) {
            Position nodePos = startPos;
            nodePos.x += i * (fieldLength / (numNodes - 1));

            int nodeId = getNextNodeId();
            nodes[nodeId] = {nodeId, "submain_junction", nodePos, 0.0, false};
            submainNodeIds.push_back(nodeId);

            if (i > 0) {
                double length = nodes[submainNodeIds[i-1]].position.distanceTo(nodePos);
                links.push_back({
                    submainNodeIds[i-1], nodeId,
                    4.0 * INCH_TO_METER, length, "submain"
                });
            }
        }
    } else {
        // Create submain nodes aligned with lateral pipe positions
        std::vector<double> sortedLateralX;
        for (const auto& entry : lateralMap) {
            sortedLateralX.push_back(entry.first);
        }
        std::sort(sortedLateralX.begin(), sortedLateralX.end());

        for (size_t i = 0; i < sortedLateralX.size(); ++i) {
            Position nodePos = startPos;
            nodePos.x = sortedLateralX[i]; // Align with lateral pipe position

            int nodeId = getNextNodeId();
            nodes[nodeId] = {nodeId, "submain_junction", nodePos, 0.0, false};
            submainNodeIds.push_back(nodeId);

            // Connect to previous submain node
            if (i > 0) {
                double length = nodes[submainNodeIds[i-1]].position.distanceTo(nodePos);
                links.push_back({
                    submainNodeIds[i-1], nodeId,
                    4.0 * INCH_TO_METER, length, "submain"
                });
            }
      //updateLateralSubmainConnection(nodeId, lateralMap[sortedLateralX[i]]);
        }
    }

    return submainNodeIds;
}


Position IrrigationModel::calculateLineIntersection(const Position& p1, const Position& p2,
                                                  const Position& p3, const Position& p4) const {
    // Line-line intersection formula
    double denom = (p1.x - p2.x) * (p3.y - p4.y) - (p1.y - p2.y) * (p3.x - p4.x);

    if (std::abs(denom) < 1e-12) {
        // Lines are parallel, return midpoint
        return {(p3.x + p4.x) / 2.0, (p3.y + p4.y) / 2.0};
    }

    double t = ((p1.x - p3.x) * (p3.y - p4.y) - (p1.y - p3.y) * (p3.x - p4.x)) / denom;

    return {
        p1.x + t * (p2.x - p1.x),
        p1.y + t * (p2.y - p1.y)
    };
}


std::vector<int> IrrigationModel::createVerticalSubmain(const Position& startPos, double fieldWidth) {
    std::vector<int> submainNodeIds;

    // Collect lateral pipe Y-positions for horizontal laterals
    std::set<double> lateralYPositions;

    for (const auto& link : links) {
        if (link.type == "lateral") {
            const auto& fromNode = nodes.at(link.from);
            const auto& toNode = nodes.at(link.to);

            double midpointY = (fromNode.position.y + toNode.position.y) / 2.0;
            lateralYPositions.insert(midpointY);
        }
    }

    if (lateralYPositions.empty()) {
        // create evenly spaced nodes
        int numNodes = std::max(2, static_cast<int>(fieldWidth / (22.0* FEET_TO_METER))); //create a minimum of 2
        for (int i = 0; i < numNodes; ++i) {
            Position nodePos = startPos;
            nodePos.y += i * (fieldWidth / (numNodes - 1));

            int nodeId = getNextNodeId();
            nodes[nodeId] = {nodeId, "submain_junction", nodePos, 0.0, false};
            submainNodeIds.push_back(nodeId);

            if (i > 0) {
                double length = nodes[submainNodeIds[i-1]].position.distanceTo(nodePos);
                links.push_back({
                    submainNodeIds[i-1], nodeId,
                    4.0 * INCH_TO_METER, length, "submain"
                });
            }
        }
    } else {
        // Create submain nodes aligned with lateral pipe positions
        std::vector<double> sortedLateralY(lateralYPositions.begin(), lateralYPositions.end());

        for (size_t i = 0; i < sortedLateralY.size(); ++i) {
            Position nodePos = startPos;
            nodePos.y = sortedLateralY[i]; // Align with lateral position

            int nodeId = getNextNodeId();
            nodes[nodeId] = {nodeId, "submain_junction", nodePos, 0.0, false};
            submainNodeIds.push_back(nodeId);

            if (i > 0) {
                double length = nodes[submainNodeIds[i-1]].position.distanceTo(nodePos);
                links.push_back({
                    submainNodeIds[i-1], nodeId,
                    4.0 * INCH_TO_METER, length, "submain"
                });
            }
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

    // Organize sprinkler junctions by lateral row
    std::map<double, std::vector<int>> sprinklersByLateralRow;

    for (const auto& [id, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn" && isPointInsidePolygon(node.position)) {
            double lateralRow = std::round(node.position.x * 100.0) / 100.0;
            sprinklersByLateralRow[lateralRow].push_back(id);
        }
    }

    // Check each row has at least 2 sprinkler junctions
    for (const auto& [row, sprinklerIds] : sprinklersByLateralRow) {
        if (sprinklerIds.size() < 2) {
            helios::helios_runtime_error("ERROR (IrrigationModel::validateMinimumSprinklersPerRow()) Minimum sprinkler numbers not met");

        }
    }
   // ensureMinimumSprinklersPerRow();

    // Also validate complete sprinkler units (junction + barb + emitter)
   validateCompleteSprinklerUnits();
}

void IrrigationModel::validateCompleteSprinklerUnits() const {
    int incompleteUnits = 0;

    for (const auto& [id, node] : nodes) {
        if (node.type == "lateral_sprinkler_jn") {
            // Check if this junction has complete sprinkler unit
            if (!hasCompleteSprinklerUnit(id)) {
                incompleteUnits++;
                std::cerr << "Warning: Incomplete sprinkler unit at junction " << id << std::endl;
            }
        }
    }

    if (incompleteUnits > 0) {
        std::cerr << "Warning: " << incompleteUnits << " incomplete sprinkler units found." << std::endl;
    }
}

bool IrrigationModel::hasCompleteSprinklerUnit(int junctionId) const {
    // A complete unit has: junction -> barb -> emitter
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
        0.160, 0.200, 0.225, 0.250, 0.315, 0.355, 0.400
    }; // commercially available pipes in meters

    std::unordered_map<int, std::vector<int>> adj;
    for (const auto& link : links) {
        adj[link.from].push_back(link.to);
        adj[link.to].push_back(link.from);
    }

    // Start recursive flow calculation from water source
    int sourceId = getWaterSourceId();
    computeLinkFlows(sourceId, -1, adj);

    for (auto& link : links) {
        double Vmax = 0.0;

        if (link.type == "main") {
            Vmax = V_main;      // mainline rule
        }
        else if (link.type == "submain") {
            Vmax = V_main;      // same rule as main (or define separate V_submain)
        }
        // else if (link.type == "lateral") {
        //     Vmax = V_lateral;
        // } //skip re-sizing lateral pipes
        else {
            continue;  // skip unknown pipe types
        }
        double Q = link.flow;
        if (Q <= 0.0) continue;

        double Dreq = std::sqrt(4.0 * Q / (M_PI * Vmax));
        //selecting pipe based on maximum allowable velocity

        auto it = std::find_if(availableSizes.begin(), availableSizes.end(),
                               [&](double d) { return d >= Dreq; });
        //look for the smallest available diameter >= to Dreq
        if (it != availableSizes.end()) {
            link.diameter = *it;
        } else {
            link.diameter = availableSizes.back();
        }
    }
}

double IrrigationModel::computeLinkFlows(int nodeId, int parentId,
                        const std::unordered_map<int, std::vector<int>>& adj) {
    double totalFlow = 0.0;

    // If this node is an emitter, add its demand
    if (nodes[nodeId].type == "emitter") {
        totalFlow += nodes[nodeId].flow;  // or currentSources[idx]
    }

    // Recurse into neighbors (children)
    for (int childId : adj.at(nodeId)) {
        if (childId == parentId) continue;  // avoid backtracking

        double childFlow = computeLinkFlows(childId, nodeId, adj);

        // Assign flow to the link between nodeId ↔ childId
        Link* lnk = findLink(nodeId, childId);
        if (lnk) lnk->flow = childFlow;  // flow carried by this pipe

        totalFlow += childFlow;  // accumulate flows upstream
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


//linear solver without using external package

HydraulicResults IrrigationModel::calculateHydraulics(bool doPreSize, const std::string& sprinklerAssemblyType,
                                                     double Qspecified, double Pw,
                                                     double V_main, double V_lateral) {
    const SprinklerAssembly & config = sprinklerLibrary.getSprinklerType(sprinklerAssemblyType);
    std::string barbType = config.barbType;
    std::string nozzleType = config.emitterType;

    // Constants
    const double rho = 997.0;     // kg/m^3
    const double mu = 8.90e-04;   // Pa·s
    const double err_max = 1e-3;
    const int max_iter = 100;

    buildNeighborLists();
    const int numNodes = nodes.size();
    if (numNodes == 0) return HydraulicResults();

    // Create ordered list and mapping
    std::vector<int> orderedNodeIds;
    for (const auto& [id, node] : nodes) orderedNodeIds.push_back(id);
    std::sort(orderedNodeIds.begin(), orderedNodeIds.end());

    std::unordered_map<int, int> nodeIndexMap;
    for (int i = 0; i < orderedNodeIds.size(); ++i) {
        nodeIndexMap[orderedNodeIds[i]] = i;  // 0-based indexing
    }

   //  node and index numbers in nodeIndexMap
    std::cout << "Node Index Map:\n";
    for (const auto& [id, node] : nodeIndexMap) {
        std::cout << "Node " << id << " -> index " << node << "\n";
    }

    // water source fixed on the last row
    int waterSourceId = orderedNodeIds.back();
    int waterSourceIndex = nodeIndexMap[waterSourceId];

    // Initialize matrices
    std::vector<std::vector<double>> A(numNodes, std::vector<double>(numNodes, 0.0));
    std::vector<double> RHS(numNodes, 0.0);
    std::vector<double> nodalPressure(numNodes, 0.0);
    std::vector<double> nodalPressure_old(numNodes, 0.0);

    // set water source for once
    A[waterSourceIndex][waterSourceIndex] = 1.0;
    RHS[waterSourceIndex] = Pw * 6894.76;

    // Initialize current sources
  //  std::vector<double> currentSources(numNodes, 0.0);
    for (auto& [id, node] : nodes) {
        int idx = nodeIndexMap[id];
        if (node.type == "emitter" && idx != waterSourceIndex) {
         //   currentSources[idx] = Qspecified;
            node.flow = Qspecified;
        }
    }

    // Flow variables
    std::unordered_map<int, std::vector<double>> Re, W_bar, vol_rate, R, delta_P;
    for (const auto& [id, node] : nodes) {
        int size = node.neighbors.size();
        Re[id] = std::vector<double>(size, 0.0);
        W_bar[id] = std::vector<double>(size, 0.0);
        vol_rate[id] = std::vector<double>(size, 0.0);
        R[id] = std::vector<double>(size, 0.0);
        delta_P[id] = std::vector<double>(size, 0.0);
    }

    double err = 1e6;
    int iter = 1;

    while (std::abs(err) > err_max && iter < max_iter) {
        std::cout << "\n=== Iteration " << iter << " ===\n";

        // Build matrix A (water source row remains A[ws][ws] = 1.0)
        for (int orderedIndex = 0; orderedIndex < orderedNodeIds.size(); ++orderedIndex) {
            int id = orderedNodeIds[orderedIndex];
            const auto& node = nodes[id];

            std::cout << "\n orderedIndex" << orderedIndex << " id " << id << "\n";

            if (orderedIndex == waterSourceIndex) continue;  // Skip water source
            std::cout << "\n watersourceId " << waterSourceId << "\n";

            A[orderedIndex][orderedIndex] = 0.0;

            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                int neighborId = node.neighbors[j];
                std::cout << neighborId << "\n";

                if (!nodeIndexMap.count(neighborId)) continue;

                int neighborIdx = nodeIndexMap[neighborId];
                const Link* link = findLink(id, neighborId);
                if (!link) continue;
                double Kf_barb = minorLoss_kf(Re[id][j], barbType);
                double Rval = 0.0;
                // if (link->type == "lateralTobarb" && Kf_barb > 0) {
                //     Rval  = 2.0 * W_bar[id][j] * Kf_barb * rho / (M_PI * pow(link->diameter, 2));
                //     R[id][j] = Rval;
                //     std::cout << "velocity: "<< W_bar[id][j] << "\n";
                //
                //     std::cout << "diameter: "<< link->diameter << "\n";
                //    std::cout << "Kf"<< Kf_barb<< "Rval " << Rval << "\n";
                // }

                Rval = calculateResistance(Re[id][j], W_bar[id][j],Kf_barb, *link, iter);
                std::cout << "velocity: "<< W_bar[id][j] << "\n";
                std::cout << "other Rval: "<< "Rval " << Rval << "\n";

                R[id][j] = Rval;

                //Rval = std::max(Rval, 1e-12);


                A[orderedIndex][orderedIndex] -= 1.0 / Rval;
                A[orderedIndex][neighborIdx] = 1.0 / Rval;
            }
        }

        // Update RHS with elevation effect
        for (int n = 0; n < numNodes; ++n) {
            if (n == waterSourceIndex) continue;

            int nodeId = orderedNodeIds[n];
            RHS[n] = nodes[nodeId].flow;

            //    RHS[n] = currentSources[n];
            std::cout << RHS[n] << "\n";
            // Add elevation contribution to RHS
            // For each neighbor connection, add elevation difference term
            const auto& node = nodes[nodeId];
            double elevation_contribution = 0.0;

            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                int neighborId = node.neighbors[j];
                if (!nodeIndexMap.count(neighborId)) continue;

                const Link* link = findLink(nodeId, neighborId);
                if (!link) continue;

                const auto& neighborNode = nodes[neighborId];
                double elevation_diff = node.position.z - neighborNode.position.z;

                // elevation effect rho*g*h in Pa
                double elevation_pressure_diff = rho * 9.81 * elevation_diff;

                // Distribute elevation contribution
                // include rho*g*h in pressure difference calculation?
                elevation_contribution += elevation_pressure_diff / R[nodeId][j];
            }

            RHS[n] += elevation_contribution;
        }

        //  output
        std::cout << "Matrix A (water source at " << waterSourceIndex << "):\n";
        for (int i = 0; i < numNodes; i++) {
            std::cout << "Row " << i << ": ";
            for (int j = 0; j < numNodes; j++) {
                std::cout << std::setw(12) << std::setprecision(6) << A[i][j] << " ";
            }
            std::cout << " | RHS: " << RHS[i] << std::endl;
        }

        // Solve linear system
        for (int gs_iter = 0; gs_iter < 100; ++gs_iter) {
            for (int i = 0; i < numNodes; ++i) {
                //if (i == waterSourceIndex) continue;

               //check if row is valid
                if (std::abs(A[i][i]) < 1e-12) {
                    nodalPressure[i] = 0.0;
                    continue;
                }

                double sum = RHS[i];
                for (int j = 0; j < numNodes; ++j) {
                    if (i != j) sum -= A[i][j] * nodalPressure[j]; // Aij·Xj)
                }

                double new_pressure = sum / A[i][i]; // Xi = (bi - Σ Aij·Xj) / Aii
                if (std::isnan(new_pressure) || std::isinf(new_pressure)) {
                    std::cerr << "NaN pressure at node " << i << ", using old value\n";
                    new_pressure = nodalPressure_old[i];
                }
                nodalPressure[i] = new_pressure;
            }
        }
        // Eigen::MatrixXd A_eigen(numNodes, numNodes);
        // Eigen::VectorXd RHS_eigen(numNodes);
        //
        // // Copy your A and RHS to Eigen matrices
        // for (int i = 0; i < numNodes; ++i) {
        //     for (int j = 0; j < numNodes; ++j) {
        //         A_eigen(i, j) = A[i][j];
        //     }
        //     RHS_eigen(i) = RHS[i];
        // }
        //
        // // Solve using Eigen (similar to MATLAB's backslash)
        // Eigen::VectorXd nodalPressure_eigen = A_eigen.colPivHouseholderQr().solve(RHS_eigen);
        //
        //
        // for (int i = 0; i < numNodes; ++i) {
        //     nodalPressure[i] = nodalPressure_eigen(i);
        // }

        // Update flow variables with safety checks
        for (const auto& [id, node] : nodes) {
            int idx = nodeIndexMap[id];

            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                int neighborId = node.neighbors[j];
                if (!nodeIndexMap.count(neighborId)) continue;

                int neighborIdx = nodeIndexMap[neighborId];
                Link* link = findLink(id, neighborId);
                if (!link) continue;

                delta_P[id][j] = nodalPressure[idx] - nodalPressure[neighborIdx];

                //  Avoid division by zero
                double denominator = R[id][j] * (M_PI/4.0) * pow(link->diameter, 2);
                if (std::abs(denominator) < 1e-12) {
                    W_bar[id][j] = 0.0;
                    std::cout << "denominator: " << denominator << std::endl;
                } else {
                    W_bar[id][j] = std::abs(delta_P[id][j]) / denominator;
                    std::cout << "denominator: " << denominator << std::endl;

                    std::cout<<"velocity:" << W_bar[id][j]<<"\n";
                }

                vol_rate[id][j] = W_bar[id][j] * (M_PI/4.0) * pow(link->diameter, 2);
                link->flow = vol_rate[id][j];
                Re[id][j] = std::abs(W_bar[id][j]) * link->diameter * rho / mu;
                if (link->type == "lateralTobarb") {
                    std::cout <<"flow_rate" << vol_rate[id][j] <<" Re:" << Re[id][j] << "\n";

                }
                // Debug NaN values
                if (std::isnan(W_bar[id][j])) {
                    std::cerr << "NaN W_bar at node " << id << "->" << neighborId
                              << ": R=" << R[id][j] << ", dP=" << delta_P[id][j] << "\n";
                }
            }
        }

        // Update emitter flows with safety
        for (auto& [id, node] : nodes) {
            int idx = nodeIndexMap[id];
            if (node.type == "emitter" && idx != waterSourceIndex) {
                double new_flow = calculateEmitterFlow(sprinklerAssemblyType, nodalPressure[idx]);
                if (std::isnan(new_flow) || std::isinf(new_flow)) {
                    std::cerr << "NaN flow at emitter " << id << ", pressure: " << nodalPressure[idx] << "\n";
                    new_flow = 0.0;
                }
                node.flow = new_flow;
              //  currentSources[idx] = new_flow;
            }
        }

        // Error calculation
        if (iter>1) {
            double norm_diff = 0.0, norm_old = 0.0;
            for (int i = 0; i < numNodes; ++i) {
                if (i == waterSourceIndex) continue;
                double diff = nodalPressure[i] - nodalPressure_old[i];
                norm_diff += diff * diff;
                norm_old += nodalPressure_old[i] * nodalPressure_old[i];
            }

            err = (norm_old > 1e-12) ? std::sqrt(norm_diff / norm_old) : std::sqrt(norm_diff); //relative error
           //err = std::sqrt(norm_diff/ numNodes);//RMS error
        }

        std::cout << "Error: " << err << std::endl;

        if (std::isnan(err)) break;
        nodalPressure_old = nodalPressure;
        iter++;
    }

    // Prepare results
    HydraulicResults results;
    results.nodalPressures.resize(numNodes);
    results.flowRates.resize(links.size());
    results.converged = (err <= err_max);
    results.iterations = iter;

    // Convert pressures to psi
    for (int i = 0; i < numNodes; ++i) {
        results.nodalPressures[i] = nodalPressure[i] / 6894.76;
    }

    // update nodal pressure
    for (const auto& [id, node] : nodes) {
        if (nodeIndexMap.count(id)) {
            int matrixIndex = nodeIndexMap[id];
            nodes[id].pressure = results.nodalPressures[matrixIndex];
        }
    }

    // link flows
    for (size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];
        int from = link.from;
        int to = link.to;

        if (!nodeIndexMap.count(from) || !nodeIndexMap.count(to)) {
            results.flowRates[i] = 0.0;
            continue;
        }

        auto it = std::find(nodes[from].neighbors.begin(), nodes[from].neighbors.end(), to);
        if (it != nodes[from].neighbors.end()) {
            size_t idx = std::distance(nodes[from].neighbors.begin(), it);
            results.flowRates[i] = vol_rate[from][idx];
        } else {
            results.flowRates[i] = 0.0;
        }
    }
    // Save emitter flows
    // results.emitterFlows.clear();
    // for (const auto& [id, node] : nodes) {
    //     if (node.type == "emitter") {
    //         results.emitterFlows[id] = node.flow;
    //     }
    // }

    if (!results.converged) {
        std::cerr << "Warning: Solver did not converge after " << iter << " iterations\n";
        results.converged = false;

    }

    return results;
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

void IrrigationModel::assignZones(int numZones, ZoningMode mode, int zonesX, int zonesY) {
    struct LateralInfo {
        Link* link;
        double avgX;
        double avgY;
    };
    std::vector<LateralInfo> lateralLinks;

    // Step 1: Collect laterals
    for (auto& link : links) {
        if (link.type == "lateral") {
            const Node& n1 = nodes[link.from];
            const Node& n2 = nodes[link.to];
            double avgX = 0.5 * (n1.position.x + n2.position.x);
            double avgY = 0.5 * (n1.position.y + n2.position.y);
            lateralLinks.push_back({&link, avgX, avgY});
        }
    }

    int lateralCount = lateralLinks.size();
    if (lateralCount == 0 || numZones <= 0) return;

    // 🔎 Handle AUTO mode by switching to X or Y based
    if (mode == ZoningMode::AUTO_BASED) {
        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (auto& entry : lateralLinks) {
            minX = std::min(minX, entry.avgX);
            maxX = std::max(maxX, entry.avgX);
            minY = std::min(minY, entry.avgY);
            maxY = std::max(maxY, entry.avgY);
        }

        double width  = maxX - minX;
        double height = maxY - minY;

        if (width > height) {
            mode = ZoningMode::X_BASED;
        } else {
            mode = ZoningMode::Y_BASED;
        }
    }

    // 🔎 Existing zoning logic
    if (mode == ZoningMode::X_BASED) {
        std::sort(lateralLinks.begin(), lateralLinks.end(),
                  [](const LateralInfo& a, const LateralInfo& b) {
                      return a.avgX < b.avgX;
                  });

        int perZone = (lateralCount + numZones - 1) / numZones;
        int zone = 1, count = 0;

        for (auto& entry : lateralLinks) {
            Link* link = entry.link;
            link->zoneID = zone;
            nodes[link->from].zoneID = zone;
            nodes[link->to].zoneID   = zone;

            count++;
            if (count >= perZone && zone < numZones) {
                zone++;
                count = 0;
            }
        }
    }
    else if (mode == ZoningMode::Y_BASED) {
        std::sort(lateralLinks.begin(), lateralLinks.end(),
                  [](const LateralInfo& a, const LateralInfo& b) {
                      return a.avgY < b.avgY;
                  });

        int perZone = (lateralCount + numZones - 1) / numZones;
        int zone = 1, count = 0;

        for (auto& entry : lateralLinks) {
            Link* link = entry.link;
            link->zoneID = zone;
            nodes[link->from].zoneID = zone;
            nodes[link->to].zoneID   = zone;

            count++;
            if (count >= perZone && zone < numZones) {
                zone++;
                count = 0;
            }
        }
    }
    else if (mode == ZoningMode::XY_BASED) {
        double minX = std::numeric_limits<double>::max();
        double maxX = std::numeric_limits<double>::lowest();
        double minY = std::numeric_limits<double>::max();
        double maxY = std::numeric_limits<double>::lowest();

        for (auto& entry : lateralLinks) {
            minX = std::min(minX, entry.avgX);
            maxX = std::max(maxX, entry.avgX);
            minY = std::min(minY, entry.avgY);
            maxY = std::max(maxY, entry.avgY);
        }

        for (auto& entry : lateralLinks) {
            int xBin = std::min(int((entry.avgX - minX) / (maxX - minX + 1e-9) * zonesX), zonesX - 1);
            int yBin = std::min(int((entry.avgY - minY) / (maxY - minY + 1e-9) * zonesY), zonesY - 1);

            int zone = yBin * zonesX + xBin + 1;

            Link* link = entry.link;
            link->zoneID = zone;
            nodes[link->from].zoneID = zone;
            nodes[link->to].zoneID   = zone;
        }
    }

    // Step 3: Create submains and valves
    for (int z = 1; z <= numZones; ++z) {
        addSubmainAndValve(z);
    }
}

void IrrigationModel::addSubmainAndValve(int zoneId) {
    // add valve node
    int valveId = getNextNodeId();
    Node valveNode;
    valveNode.id = valveId;
    valveNode.type = "valve";
    const auto& node = nodes.at(waterSourceId);
    valveNode.position = node.position; // place near mainline
    valveNode.zoneID = zoneId;
    valveNode.pressure = 0.0;
    valveNode.is_fixed = false;
    valveNode.flow = 0.0;
    nodes[valveId] = valveNode;

    // connect mainline to valve
    Link valveLink;
 //   valveLink.from = mainlineSourceId; // assumed ID of global mainline source
    valveLink.to   = valveId;
    valveLink.type = "valve";
    valveLink.diameter = 5.0;
    valveLink.length = 0.5; // short stub length
    valveLink.zoneID = zoneId;
    links.push_back(valveLink);

    // creating submain node (junction for laterals in this zone)
    int submainId = getNextNodeId();
    Node submainNode;
    submainNode.id = submainId;
    submainNode.type = "submain";
  //  submainNode.position = estimateZoneCentroid(zoneId); // put submain near the zone
    submainNode.zoneID = zoneId;
    submainNode.pressure = 0.0;
    submainNode.is_fixed = false;
    submainNode.flow = 0.0;
    nodes[submainId] = submainNode;

    // Connect valve to submain
    Link submainLink;
    submainLink.from = valveId;
    submainLink.to   = submainId;
    submainLink.type = "submain";
    submainLink.diameter = 5.0;
    submainLink.length = nodes[valveId].position.distanceTo(submainNode.position);
    submainLink.zoneID = zoneId;
    links.push_back(submainLink);

    // Step 5: Re-attach laterals in this zone to the submain
    for (auto& link : links) {
        if (link.zoneID == zoneId && link.type == "lateral") {
            // ensure lateral connects to submain
            nodes[link.from].neighbors.push_back(submainId);
            nodes[link.to].neighbors.push_back(submainId);

            // Optionally insert a link submain -> lateral head
            // Link feeder;
            // feeder.from = submainId;
            // feeder.to   = link.from; // attach to first node of lateral
            // feeder.type = "feeder";
            // feeder.diameter = defaultFeederDiameter;
            // feeder.length = distance(nodes[submainId].position, nodes[link.from].position);
            // feeder.zoneId = zoneId;
            // links.push_back(feeder);
        }
    }

    std::cout << "Zone " << zoneId << ": added valve " << valveId
              << " and submain " << submainId << std::endl;
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
        std::cerr << "Warning: " << disconnectedCount
                 << " nodes are disconnected from the water source!" << std::endl;

        // List disconnected nodes
        for (const auto& [id, node] : nodes) {
            if (!connectedNodes.count(id)) {
                std::cerr << "  - Node " << id << " at ("
                         << node.position.x << ", "
                         << node.position.y << ")" << std::endl;
            }
        }
    }

    // Link property validation
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
        std::cout << "it is laminar flow" << std::endl;
    } else {
        R = 0.6328 * pow(Wbar, 0.75) * pow(mu, 0.25) * link.length * pow(rho, 0.75) /
            (M_PI * pow(link.diameter, 3.25));
        std::cout << "it is turbulent flow" << std::endl;
        std::cout << "link type: " << link.type << "\n";


    // using minor loss equation if the link is lateralToBarb
        if (link.type == "lateralTobarb" && Kf_barb > 0) {
            std::cout << "calling lateralTobarb" << std::endl;
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
        std::cout<<"reading the minor loss of NelsonFlat" <<std::endl;

        kf = 1.73;
    } else if (barbType  == "NPC_Toro_flat_barb") {
        kf = 1.48; //1.50 for half tube
    } else if (barbType  == "NPC_Toro_sharp_barb") {
        std::cout<<"reading the minor loss of ToroSharp" <<std::endl;
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

double IrrigationModel::calculateEmitterFlow(const std::string& sprinklerAssemblyType, double pressure) {
    const SprinklerAssembly & config = sprinklerLibrary.getSprinklerType(sprinklerAssemblyType);
    std::string nozzleType = config.emitterType;

    // Convert pressure from Pa to psi
    double pressure_psi = pressure / 6894.76;

    if (nozzleType == "PC") {
        // Linear relationship for PC nozzle (GPH to m3/s)
        return std::max(0.0, std::min(1.6049e-5, (0.9 * pressure_psi * 1.052e-6)));
    }
    else if (nozzleType == "NPC") {
        // Non-linear relationship for NPC nozzle
        // const double x = 0.477; //fan jet J2 orange 0.53 0.477;
        // const double k = 3.317; //1.76;
        return config.emitter_k * pow(pressure_psi, config.emitter_x)*1.052e-6; //get coefficients directly from library
    }
    else {
        helios::helios_runtime_error("ERROR (IrrigationModel::calculateEmitterFlow): Unknown nozzle type");
        return 0.0;
    }
}

// add to library after creating the item
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
    type.name = "NPC_Nelson_flat_barb";
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
    type.name = "NPC_Toro_flat_barb";
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
    type.name = "NPC_Toro_sharp_barb";
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
    type.name = "PC_Nelson_flat_barb";
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
    type.name = "PC_Toro_flat_barb";
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
    type.name = "PC_Toro_sharp_barb";
    type.lateralToBarb = {0.118* INCH_TO_METER, 0.74* INCH_TO_METER, "PVC"};
    type.barbToEmitter = { 0.154 * INCH_TO_METER, 30*INCH_TO_METER,"PVC"};
    type.emitterType = "PC";
    type.emitter_x = 0.0;
    type.emitter_k = 3.317;
    type.stakeHeight = 0.1; //10cm above ground
    type.barbType =  "PC_Toro_sharp_barb";
    return type;
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
