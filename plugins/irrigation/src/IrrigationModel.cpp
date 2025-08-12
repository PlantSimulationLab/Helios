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
#include <Eigen/Dense>  // For matrix operations
#include <unordered_set>  // For std::unordered_set
#include <queue>          // For std::queue
#include <algorithm>    // For std::find

// Add parameter validation
void IrrigationModel::validateParameters(double fieldLength, double fieldWidth,
                                       double sprinklerSpacing, double lineSpacing) const {
    if (fieldLength <= 0 || fieldWidth <= 0) {
        throw std::runtime_error("Field dimensions must be positive");
    }
    if (sprinklerSpacing <= 0 || lineSpacing <= 0) {
        throw std::runtime_error("Spacing values must be positive");
    }
    if (sprinklerSpacing > fieldLength || lineSpacing > fieldWidth) {
        throw std::runtime_error("Spacing values cannot exceed field dimensions");
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

// Modify createCompleteSystem to include validation

void IrrigationModel::createCompleteSystem(double Pw, double fieldLength, double fieldWidth,
                                        double sprinklerSpacing, double lineSpacing,
                                        const std::string& connectionType,
                                        SubmainPosition submainPos) {
    nodes.clear();
    links.clear();

    createSprinklerSystem(fieldLength, fieldWidth, sprinklerSpacing, lineSpacing, connectionType);
    addSubmainAndWaterSource(fieldLength, fieldWidth, connectionType, submainPos);

    // Set fixed pressure at water source
    if (waterSourceId != -1) {
        nodes[waterSourceId].pressure = Pw;
        nodes[waterSourceId].is_fixed = true;
    }
}


Position IrrigationModel::calculateWaterSourcePosition(double fieldLength, double fieldWidth,
                                                     const std::string& lateralDirection) const {
    if (lateralDirection == "vertical") {
        return {
            fieldLength / 3.0 + 5.0,  // x
            fieldWidth - fieldWidth / 3.0 - 0.5  // y
        };
    } else {
        return {
            fieldLength / 2.0,  // x
            0.0  // y
        };
    }
}

void IrrigationModel::createSprinklerSystem(double fieldLength, double fieldWidth,
                                          double sprinklerSpacing, double lineSpacing,
                                          const std::string& connectionType) {
    int num_laterals = static_cast<int>(std::ceil(fieldLength / sprinklerSpacing)) + 1;
    int num_sprinklers_perline = static_cast<int>(std::ceil(fieldWidth / lineSpacing)) + 1;

    int nodeId = 1;
    const double barbOffset = 0.3; // Distance from lateral to barb (meters)
    const double emitterOffset = 0.5; // Distance from barb to emitter (meters)

    for (int i = 0; i < num_laterals; ++i) {
        for (int j = 0; j < num_sprinklers_perline; ++j) {
            double x = i * sprinklerSpacing;
            double y = j * lineSpacing;

            // Create junction node for sprinkler assembly connection
            nodes[nodeId] = {nodeId, "lateral_sprinkler_jn", {x, y}, 0.0, false};
            int junctionId = nodeId++;

            // Calculate 45 degree offset position for barb
            double barbX = x + barbOffset * cos(M_PI/4);
            double barbY = y + barbOffset * sin(M_PI/4);

            // Create barb node
            nodes[nodeId] = {nodeId, "barb", {barbX, barbY}, 0.0, false};
            int barbId = nodeId++;

            // Calculate position for emitter (continuing 45° from barb)
            double emitterX = barbX + emitterOffset * cos(M_PI/4);
            double emitterY = barbY + emitterOffset * sin(M_PI/4);

            // Create emitter node
            nodes[nodeId] = {nodeId, "emitter", {emitterX, emitterY}, 0.0, false};
            int emitterId = nodeId++;

            // Connect barb to emitter
            links.push_back({
                barbId, emitterId,
                0.154 * INCH_TO_METER, // ~6mm diameter
                emitterOffset,         // Actual physical length
                "barbToemitter"
            });

            // Connect junction to barb
            links.push_back({
                junctionId, barbId,
                0.25 * INCH_TO_METER, // ~10mm diameter
                barbOffset,          // Actual physical length
                "lateralTobarb"
            });

            // Connect laterals (if not last in row/column)
            if (connectionType == "vertical" && j < num_sprinklers_perline - 1) {
                links.push_back({
                    junctionId, junctionId + 3,
                    0.67 * INCH_TO_METER, // ~17mm diameter
                    lineSpacing,         // Full spacing distance
                    "lateral"
                });
            }
            else if (connectionType == "horizontal" && i < num_laterals - 1) {
                links.push_back({
                    junctionId, junctionId + num_sprinklers_perline * 3,
                    0.67 * INCH_TO_METER, // ~17mm diameter
                    sprinklerSpacing,    // Full spacing distance
                    "lateral"
                });
            }
        }
    }
}

int IrrigationModel::getNextNodeId() const {
    if (nodes.empty()) return 1;
    return std::max_element(nodes.begin(), nodes.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; })->first + 1;
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
                false
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
                toPos.distanceTo(intersectPos),
                "lateral"
            });

            // Create connection from submain node to submain
            if (i < connectingLaterals.size() - 1) {
                int nextSubmainNodeId = submainNodeId + 1;
                double spacing = (lateralDirection == "vertical") ?
                    std::abs(nodes[nextSubmainNodeId].position.x - intersectPos.x) :
                    std::abs(nodes[nextSubmainNodeId].position.y - intersectPos.y);

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
        // North/South positions - connect edge nodes directly
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
                false
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
            }

            submainNodeId++;
        }
    }

    // Create water source node
    waterSourceId = getNextNodeId();
    Position waterSourcePos;

    if (lateralDirection == "vertical") {
        // manual setting: water source position logic for vertical laterals
        waterSourcePos = {fieldLength / 3.0 + 5.0, fieldWidth - fieldWidth / 3.0 - 0.5};

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

    // Find closest submain node to connect to
    int closestSubmainId = -1;
    double minDistance = std::numeric_limits<double>::max();

    for (const auto& [id, node] : nodes) {
        if (node.type == "submain_junction") {
            double dist = node.position.distanceTo(waterSourcePos);
            if (dist < minDistance) {
                minDistance = dist;
                closestSubmainId = id;
            }
        }
    }

    if (closestSubmainId != -1) {
        // Connect water source to closest submain node
        links.push_back({
            waterSourceId,
            closestSubmainId,
            2.0 * INCH_TO_METER,
            5.0 * FEET_TO_METER, // Fixed length
            "mainline"
        });
    } else {
        // Fallback: create a default connection if no submain nodes found
        int defaultSubmainId = getNextNodeId();
        Position defaultPos = (lateralDirection == "vertical") ?
            Position{fieldLength / 2.0, submainY} :
            Position{submainX, fieldWidth / 2.0};

        nodes[defaultSubmainId] = {
            defaultSubmainId,
            "submain_junction",
            defaultPos,
            0.0,
            false
        };

        links.push_back({
            waterSourceId,
            defaultSubmainId,
            2.0 * INCH_TO_METER,
            waterSourcePos.distanceTo(defaultPos),
            "mainline"
        });
    }
}

// void IrrigationModel::addSubmainAndWaterSource(double fieldLength, double fieldWidth,
//                                              const std::string& lateralDirection,
//                                              SubmainPosition submainPosition) {
//     // Determine submain Y position based on placement option
//     double submainY;
//     switch (submainPosition) {
//         case SubmainPosition::NORTH:
//             submainY = fieldWidth;
//             break;
//         case SubmainPosition::SOUTH:
//             submainY = 0.0;
//             break;
//         case SubmainPosition::MIDDLE:
//             submainY = fieldWidth / 2.0 + 2.5;
//             break;
//     }
//
//     // Find all lateral junctions that connect to submain
//     std::vector<int> junctionNodes;
//     for (const auto& [id, node] : nodes) {
//         if (node.type == "lateral_sprinkler_jn") {
//             if ((lateralDirection == "vertical" && std::abs(node.position.y - submainY) < 0.1) ||
//                 (lateralDirection == "horizontal" && std::abs(node.position.x) < 0.1)) {
//                 junctionNodes.push_back(id);
//             }
//         }
//     }
//
//     // If no junctions found (middle position might need special handling)
//     if (junctionNodes.empty() && submainPosition == SubmainPosition::MIDDLE) {
//         // For middle position, connect to all laterals with vertical links
//         for (const auto& [id, node] : nodes) {
//             if (node.type == "lateral_sprinkler_jn") {
//                 junctionNodes.push_back(id);
//             }
//         }
//     }
//
//     if (junctionNodes.empty()) {
//         int defaultJnId = getNextNodeId();
//         nodes[defaultJnId] = {defaultJnId, "lateral_sprinkler_jn", {0.0, submainY}, 0.0, false};
//         junctionNodes.push_back(defaultJnId);
//     }
//
//     // Create submain nodes and connection junctions
//     int submainStartId = getNextNodeId();
//     std::sort(junctionNodes.begin(), junctionNodes.end(), [this](int a, int b) {
//         return nodes[a].position.x < nodes[b].position.x;
//     });
//
//     for (size_t i = 0; i < junctionNodes.size(); ++i) {
//         Position pos = nodes[junctionNodes[i]].position;
//
//         // For middle position, adjust the y-coordinate to submain position
//         if (submainPosition == SubmainPosition::MIDDLE) {
//             pos.y = submainY;
//         }
//
//         // Create submain connection junction
//         int submainJnId = submainStartId + i*2;
//         nodes[submainJnId] = {submainJnId, "lateral_sub_jn", pos, 0.0, false};
//
//          // Create submain node (offset from junction)
//         // int submainId = submainStartId + i*2 + 1;
//         // Position submainPos = pos;
//         // submainPos.x += 0.5;
//         // nodes[submainId] = {submainId, "submain", submainPos, 0.0, false};
//
//         // Connect lateral junction to submain junction
//         if (submainPosition == SubmainPosition::MIDDLE) {
//             // For middle position, create vertical connections
//             pos.y = submainY;
//
//             // Calculate vertical distance with minimum length
//             double vertical_dist = std::abs(nodes[junctionNodes[i]].position.y - submainY);
//             if (vertical_dist < 0.1) vertical_dist = 0.5;
//
//             links.push_back({
//                 junctionNodes[i],
//                 static_cast<int>(submainStartId + i*2),
//                 0.67 * INCH_TO_METER,
//                 vertical_dist,
//                 "lateralToSubmain"
//             });
//         }
//         //
//         // Connect submain junction to submain node
//         // links.push_back({
//         //     submainJnId, submainId,
//         //     0.67 * INCH_TO_METER,
//         //     0.5,
//         //     "submainConnection"
//         // });
//
//         // Connect submain nodes in sequence
//         // if (i > 0) {
//         //     double dist = nodes[submainId].position.distanceTo(nodes[submainId-2].position);
//         //     if (dist <= 0) {  // Add validation
//         //         throw std::runtime_error("Zero distance between submain nodes " +
//         //                                std::to_string(submainId-2) + " and " +
//         //                                std::to_string(submainId));
//         //     }
//         //
//         //     links.push_back({
//         //         submainId - 2, submainId,
//         //         4.0 * INCH_TO_METER,
//         //         nodes[submainId].position.distanceTo(nodes[submainId-2].position),
//         //         "submain"
//         //     });
//         // }
//     }
//
//     // Create water source node
//     waterSourceId = getNextNodeId();
//     Position waterPos = nodes[submainStartId].position;
//     waterPos.x -= 5.0;
//
//     nodes[waterSourceId] = {waterSourceId, "waterSource", waterPos, 0.0, true};
//
//     // Connect water source to first submain junction
//     links.push_back({
//         waterSourceId, submainStartId,
//         2.0 * INCH_TO_METER,
//         nodes[waterSourceId].position.distanceTo(nodes[submainStartId].position),
//         "mainline"
//     });
//
// }

// void IrrigationModel::addSubmainAndWaterSource(double fieldLength, double fieldWidth,
//                                              const std::string& lateralDirection) {
//     // Create submain line position
//     double submainY = (lateralDirection == "vertical") ?
//         fieldWidth - fieldWidth/3.0 - 0.5 : 0.0;
//
//     // Find all lateral junctions that connect to submain
//     std::vector<int> junctionNodes;
//     for (const auto& [id, node] : nodes) {
//         if (node.type == "lateral_sprinkler_jn") {
//             if ((lateralDirection == "vertical" && std::abs(node.position.y - submainY) < 0.1) ||
//                 (lateralDirection == "horizontal" && std::abs(node.position.x) < 0.1)) {
//                 junctionNodes.push_back(id);
//             }
//         }
//     }
//
//     if (junctionNodes.empty()) {
//         int defaultJnId = getNextNodeId();
//         nodes[defaultJnId] = {defaultJnId, "lateral_sprinkler_jn", {0.0, submainY}, 0.0, false};
//         junctionNodes.push_back(defaultJnId);
//     }
//
//     // Create submain nodes and connection junctions
//     int submainStartId = getNextNodeId();
//     std::sort(junctionNodes.begin(), junctionNodes.end(), [this](int a, int b) {
//         return nodes[a].position.x < nodes[b].position.x;
//     });
//
//     for (size_t i = 0; i < junctionNodes.size(); ++i) {
//         Position pos = nodes[junctionNodes[i]].position;
//         pos.y = submainY;
//
//         // Create submain connection junction
//         int submainJnId = submainStartId + i*2;
//         nodes[submainJnId] = {submainJnId, "lateral_sub_jn", pos, 0.0, false};
//
//         // Create submain node (offset from junction)
//         int submainId = submainStartId + i*2 + 1;
//         Position submainPos = pos;
//         submainPos.x += 0.5;
//         nodes[submainId] = {submainId, "submain", submainPos, 0.0, false};
//
//         // Connect lateral junction to submain junction
//         links.push_back({
//             junctionNodes[i], submainJnId,
//             0.67 * INCH_TO_METER,
//             0.5,
//             "lateralToSubmain"
//         });
//
//         // Connect submain junction to submain node
//         links.push_back({
//             submainJnId, submainId,
//             0.67 * INCH_TO_METER,
//             0.5,
//             "submainConnection"
//         });
//
//         // Connect submain nodes in sequence
//         if (i > 0) {
//             links.push_back({
//                 submainId - 2, submainId,
//                 4.0 * INCH_TO_METER,
//                 nodes[submainId].position.distanceTo(nodes[submainId-2].position),
//                 "submain"
//             });
//         }
//     }
//
//     // Create water source node
//     waterSourceId = getNextNodeId();
//     Position waterPos = nodes[submainStartId].position;
//     waterPos.x -= 5.0;
//
//     nodes[waterSourceId] = {waterSourceId, "waterSource", waterPos, 0.0, true};
//
//     // Connect water source to first submain junction
//     links.push_back({
//         waterSourceId, submainStartId,
//         2.0 * INCH_TO_METER,
//         nodes[waterSourceId].position.distanceTo(nodes[submainStartId].position),
//         "mainline"
//     });
// }

double IrrigationModel::calculateEmitterFlow(double Pw) const {
    // Simplified flow calculation
    return 0.1 * std::sqrt(Pw); // Example formula
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

//linear solver without using external package

HydraulicResults IrrigationModel::calculateHydraulics(const std::string& nozzleType,
                                                   double Qspecified, double Pw) {
    // Constants
    const double rho = 997.0;     // kg/m^3
    const double mu = 8.90e-04;   // Pa·s
    const double err_max = 1e-3;
    const int max_iter = 50;      // Increased since our solver is less sophisticated

    // Initialize data structures
    buildNeighborLists();

    const int numNodes = nodes.size();
    if (numNodes == 0) return HydraulicResults();

    // Convert to 0-based indexing for easier matrix operations
    std::unordered_map<int, int> nodeIndexMap;
    int index = 0;
    for (const auto& [id, node] : nodes) {
        nodeIndexMap[id] = index++;
    }

    // Matrix and vector storage
    std::vector<std::vector<double>> A(numNodes, std::vector<double>(numNodes, 0.0));
    std::vector<double> RHS(numNodes, 0.0);
    std::vector<double> nodalPressure(numNodes, 0.0);
    std::vector<double> nodalPressure_old(numNodes, 0.0);

    // Set fixed pressure at water source
    if (waterSourceId != -1 && nodeIndexMap.count(waterSourceId)) {
        int wsIndex = nodeIndexMap[waterSourceId];
        A[wsIndex][wsIndex] = 1.0;
        RHS[wsIndex] = Pw * 6894.76;  // Convert psi to Pa
    }

    // Initialize current sources (emitters)
    std::vector<double> currentSources(numNodes, 0.0);
    for (const auto& [id, node] : nodes) {
        if (node.type == "emitter" && nodeIndexMap.count(id)) {
            currentSources[nodeIndexMap[id]] = Qspecified;
        }
    }

    // Initialize flow variables
    std::unordered_map<int, std::vector<double>> Re;
    std::unordered_map<int, std::vector<double>> W_bar;
    std::unordered_map<int, std::vector<double>> vol_rate;
    std::unordered_map<int, std::vector<double>> R;
    std::unordered_map<int, std::vector<double>> delta_P;

    for (const auto& [id, node] : nodes) {
        Re[id] = std::vector<double>(node.neighbors.size(), 0.0);
        W_bar[id] = std::vector<double>(node.neighbors.size(), 0.0);
        vol_rate[id] = std::vector<double>(node.neighbors.size(), 0.0);
        R[id] = std::vector<double>(node.neighbors.size(), 0.0);
        delta_P[id] = std::vector<double>(node.neighbors.size(), 0.0);
    }

    // Iterative solution
    double err = 1e6;
    int iter = 0;

    while (std::abs(err) > err_max && iter < max_iter) {
        // Build matrix A
        for (const auto& [id, node] : nodes) {
            if (id == waterSourceId) continue; // Skip fixed pressure node

            int i = nodeIndexMap[id];
            A[i][i] = 0.0;

            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                int neighborId = node.neighbors[j];

                // Find the link between node and neighbor
                const Link* link = nullptr;
                for (const auto& l : links) {
                    if ((l.from == id && l.to == neighborId) ||
                        (l.to == id && l.from == neighborId)) {
                        link = &l;
                        break;
                    }
                }

                if (!link || !nodeIndexMap.count(neighborId)) continue;

                // Calculate resistance
                R[id][j] = calculateResistance(Re[id][j], W_bar[id][j], *link, iter);

                // Update matrix coefficients
                A[i][i] -= 1.0 / R[id][j];
                int neighborIdx = nodeIndexMap[neighborId];
                A[i][neighborIdx] = 1.0 / R[id][j];
            }
        }


        // After building matrix A
        std::cout << "\nMatrix diagonal elements:\n";
        for (int i = 0; i < numNodes; ++i) {
            std::cout << "A[" << i << "][" << i << "] = " << A[i][i] << "\n";
        }

        // Check if water source row is properly set
        if (waterSourceId != -1) {
            int wsIdx = nodeIndexMap[waterSourceId];
            std::cout << "Water source row: " << wsIdx << " | A value: " << A[wsIdx][wsIdx]
                      << " | RHS: " << RHS[wsIdx] << "\n";
        }

        // Update RHS with current sources
        for (int i = 0; i < numNodes; ++i) {
            if (waterSourceId != -1 && i == nodeIndexMap[waterSourceId]) continue;
            RHS[i] = currentSources[i];
        }

        // Solve linear system using Gauss-Seidel iteration
        for (int gs_iter = 0; gs_iter < 100; ++gs_iter) {
            for (int i = 0; i < numNodes; ++i) {
                if (waterSourceId != -1 && i == nodeIndexMap[waterSourceId]) continue;

                double sum = RHS[i];
                for (int j = 0; j < numNodes; ++j) {
                    if (i != j) {
                        sum -= A[i][j] * nodalPressure[j];
                    }
                }
                nodalPressure[i] = sum / A[i][i];
            }
        }

        // Update flow variables
        bool isAllLaminar = true;
        for (const auto& [id, node] : nodes) {
            if (!nodeIndexMap.count(id)) continue;

            int idx = nodeIndexMap[id];
            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                int neighborId = node.neighbors[j];
                if (!nodeIndexMap.count(neighborId)) continue;

                // Find the link
                const Link* link = nullptr;
                for (const auto& l : links) {
                    if ((l.from == id && l.to == neighborId) ||
                        (l.to == id && l.from == neighborId)) {
                        link = &l;
                        break;
                    }
                }

                if (!link) continue;

                int neighborIdx = nodeIndexMap[neighborId];

                delta_P[id][j] = nodalPressure[idx] - nodalPressure[neighborIdx];
                W_bar[id][j] = std::abs(delta_P[id][j]) /
                              (R[id][j] * (M_PI/4.0) * pow(link->diameter, 2));

                vol_rate[id][j] = W_bar[id][j] * (M_PI/4.0) * pow(link->diameter, 2);
                Re[id][j] = std::abs(W_bar[id][j]) * link->diameter * rho / mu;

                if (Re[id][j] > 2000) {
                    isAllLaminar = false;
                }
            }
        }

        // Update emitter flows based on current pressure
        for (auto& [id, node] : nodes) {
            if (node.type == "emitter" && nodeIndexMap.count(id)) {
                currentSources[nodeIndexMap[id]] = calculateEmitterFlow(nozzleType, nodalPressure[nodeIndexMap[id]]);
            }
        }

        // Calculate error
        if (iter == 0) {
            err = 1e6; // Initial error
        } else {
            double norm_diff = 0.0;
            double norm_old = 0.0;
            for (int i = 0; i < numNodes; ++i) {
                norm_diff += std::pow(nodalPressure[i] - nodalPressure_old[i], 2);
                norm_old += std::pow(nodalPressure_old[i], 2);
            }
            err = std::sqrt(norm_diff) / std::sqrt(norm_old);
        }

        nodalPressure_old = nodalPressure;
        iter++;
    }

    // Prepare results
    HydraulicResults results;
    results.nodalPressures.resize(numNodes);
    results.flowRates.resize(links.size());

    // Convert nodal pressures from Pa to psi
    for (int i = 0; i < numNodes; ++i) {
        results.nodalPressures[i] = nodalPressure[i] / 6894.76;
    }

    // Store flow rates for each link
    for (size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];
        int from = link.from;
        int to = link.to;

        if (!nodeIndexMap.count(from) || !nodeIndexMap.count(to)) {
            results.flowRates[i] = 0.0;
            continue;
        }

        // find neighbor index
        auto it = std::find(nodes[from].neighbors.begin(), nodes[from].neighbors.end(), to);
        if (it != nodes[from].neighbors.end()) {
            size_t idx = std::distance(nodes[from].neighbors.begin(), it);
            results.flowRates[i] = vol_rate[from][idx];
        } else {
            results.flowRates[i] = 0.0;
        }
    }

    std::cout << "Iteration " << iter << " error: " << err << "\n";
    if (iter == max_iter - 1) {
        std::cerr << "Warning: Solver did not converge!" << std::endl;
    }

    return results;
}


//hydraulic results: pressure and flow rate
/*
HydraulicResults IrrigationModel::calculateHydraulics(const std::string& nozzleType,
                                                     double Qspecified, double Pw) {
    // Constants
    const double rho = 997.0;     // kg/m^3
    const double mu = 8.90e-04;   // Pa·s
    const double err_max = 1e-3;
    const int max_iter = 5;

    // Initialize data structures
    buildNeighborLists();

    const int numNodes = nodes.size();
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(numNodes, numNodes);
    Eigen::VectorXd RHS = Eigen::VectorXd::Zero(numNodes);
    Eigen::VectorXd nodalPressure = Eigen::VectorXd::Zero(numNodes);
    Eigen::VectorXd nodalPressure_old = Eigen::VectorXd::Zero(numNodes);

    // Set fixed pressure at water source
    A(numNodes-1, numNodes-1) = 1.0;
    RHS(numNodes-1) = Pw * 6894.76;  // Convert psi to Pa

    // Initialize current sources (emitters)
    std::vector<double> currentSources(numNodes, 0.0);
    for (const auto& [id, node] : nodes) {
        if (node.type == "emitter") {
            currentSources[id-1] = Qspecified; // Using 0-based index for Eigen
        }
    }

    // Initialize flow variables
    std::unordered_map<int, std::vector<double>> Re;
    std::unordered_map<int, std::vector<double>> W_bar;
    std::unordered_map<int, std::vector<double>> vol_rate;
    std::unordered_map<int, std::vector<double>> R;
    std::unordered_map<int, std::vector<double>> delta_P;

    for (const auto& [id, node] : nodes) {
        Re[id] = std::vector<double>(node.neighbors.size(), 0.0);
        W_bar[id] = std::vector<double>(node.neighbors.size(), 0.0);
        vol_rate[id] = std::vector<double>(node.neighbors.size(), 0.0);
        R[id] = std::vector<double>(node.neighbors.size(), 0.0);
        delta_P[id] = std::vector<double>(node.neighbors.size(), 0.0);
    }

    // Iterative solution
    double err = 1e6;
    int iter = 0;

    while (std::abs(err) > err_max && iter < max_iter) {
        // Build matrix A
        for (const auto& [id, node] : nodes) {
            if (id == waterSourceId) continue; // Skip fixed pressure node

            int i = id - 1; // Convert to 0-based index
            A(i, i) = 0.0;

            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                int neighborId = node.neighbors[j];

                // Find the link between node and neighbor
                const Link* link = nullptr;
                for (const auto& l : links) {
                    if ((l.from == id && l.to == neighborId) ||
                        (l.to == id && l.from == neighborId)) {
                        link = &l;
                        break;
                    }
                }

                if (!link) continue;

                // Calculate resistance
                R[id][j] = calculateResistance(Re[id][j], W_bar[id][j], *link, iter);

                // Update matrix coefficients
                A(i, i) -= 1.0 / R[id][j];
                int neighborIdx = neighborId - 1;
                A(i, neighborIdx) = 1.0 / R[id][j];
            }
        }

        // Update RHS with current sources
        for (int i = 0; i < numNodes-1; ++i) {
            RHS(i) = currentSources[i];
        }

        // Solve linear system
        nodalPressure = A.colPivHouseholderQr().solve(RHS);

        // Update flow variables
        bool isAllLaminar = true;
        for (const auto& [id, node] : nodes) {
            for (size_t j = 0; j < node.neighbors.size(); ++j) {
                int neighborId = node.neighbors[j];

                // Find the link
                const Link* link = nullptr;
                for (const auto& l : links) {
                    if ((l.from == id && l.to == neighborId) ||
                        (l.to == id && l.from == neighborId)) {
                        link = &l;
                        break;
                    }
                }

                if (!link) continue;

                int idx = id - 1;
                int neighborIdx = neighborId - 1;

                delta_P[id][j] = nodalPressure(idx) - nodalPressure(neighborIdx);
                W_bar[id][j] = std::abs(delta_P[id][j]) /
                              (R[id][j] * (M_PI/4.0) * pow(link->diameter, 2));

                vol_rate[id][j] = W_bar[id][j] * (M_PI/4.0) * pow(link->diameter, 2);
                Re[id][j] = std::abs(W_bar[id][j]) * link->diameter * rho / mu;

                if (Re[id][j] > 2000) {
                    isAllLaminar = false;
                }
            }
        }

        // Update emitter flows based on current pressure
        for (auto& [id, node] : nodes) {
            if (node.type == "emitter") {
                currentSources[id-1] = calculateEmitterFlow(nozzleType, nodalPressure(id-1));
            }
        }

        // Calculate error
        if (iter == 0) {
            err = 1e6; // Initial error
        } else {
            err = (nodalPressure - nodalPressure_old).norm() / nodalPressure_old.norm();
        }

        nodalPressure_old = nodalPressure;
        iter++;
    }

    // Prepare results
    HydraulicResults results;
    results.nodalPressures.resize(numNodes);
    results.flowRates.resize(links.size());

    // Convert nodal pressures from Pa to psi
    for (int i = 0; i < numNodes; ++i) {
        results.nodalPressures[i] = nodalPressure(i) / 6894.76;
    }

    // Store flow rates for each link
    for (size_t i = 0; i < links.size(); ++i) {
        const auto& link = links[i];
        int from = link.from;
        int to = link.to;

        // Find the index of the neighbor
        auto it = std::find(nodes[from].neighbors.begin(), nodes[from].neighbors.end(), to);
        if (it != nodes[from].neighbors.end()) {
            size_t idx = std::distance(nodes[from].neighbors.begin(), it);
            results.flowRates[i] = vol_rate[from][idx];
        } else {
            results.flowRates[i] = 0.0;
        }
    }

    return results;
}
*/

void IrrigationModel::validateHydraulicSystem() const {
    // 1. Check water source
    if (waterSourceId == -1) {
        throw std::runtime_error("No water source defined!");
    }

    if (!nodes.count(waterSourceId)) {
        throw std::runtime_error("Water source node doesn't exist!");
    }

    if (!nodes.at(waterSourceId).is_fixed) {
        throw std::runtime_error("Water source node not marked as fixed!");
    }

    // 2. Check connectivity
    std::unordered_set<int> connectedNodes;
    std::queue<int> toVisit;
    toVisit.push(waterSourceId);

    while (!toVisit.empty()) {
        int current = toVisit.front();
        toVisit.pop();

        if (connectedNodes.count(current)) continue;
        connectedNodes.insert(current);

        for (int neighbor : nodes.at(current).neighbors) {
            toVisit.push(neighbor);
        }
    }

    if (connectedNodes.size() != nodes.size()) {
        std::cerr << "Warning: " << (nodes.size() - connectedNodes.size())
                  << " nodes are disconnected from the water source!" << std::endl;
    }

    // 3. Check link properties
    for (const auto& link : links) {
        if (link.diameter <= 0) {
            throw std::runtime_error("Invalid diameter in link: " + link.toString());
        }
        if (link.length <= 0) {
            throw std::runtime_error("Invalid length in link: " + link.toString());
        }
    }

    std::cout << "Hydraulic system validation passed with " << nodes.size()
              << " nodes and " << links.size() << " links." << std::endl;
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

double IrrigationModel::calculateResistance(double Re, double Wbar, const Link& link, int iter) {
    const double rho = 997.0;
    const double mu = 8.90e-04;
    double R = 0.0;

    if (Re < 2000 || iter == 1) {
        R = mu * link.length * 128.0 / (M_PI * pow(link.diameter, 4));
    }
    else if (Re >= 2000 && Re < 1e5) {
        R = 0.6328 * pow(Wbar, 0.75) * pow(mu, 0.25) * link.length * pow(rho, 0.75) /
            (M_PI * pow(link.diameter, 3.25));
    }
    else {
        R = 0.6328 * pow(Wbar, 0.75) * pow(mu, 0.25) * link.length * pow(rho, 0.75) /
            (M_PI * pow(link.diameter, 3.25));
    }

    // Add validation
    if (R <= 0 || std::isinf(R) || std::isnan(R)) {
        std::cerr << "Invalid resistance calculated: " << R
                  << " for link " << link.toString()
                  << " with Re=" << Re << std::endl;
        R = 1.0; // Fallback value
    }

    return R;
}

double IrrigationModel::calculateEmitterFlow(const std::string& nozzleType, double pressure) {
    // Convert pressure from Pa to psi
    double pressure_psi = pressure / 6894.76;

    if (nozzleType == "PC") {
        // Linear relationship for PC nozzle (GPH to m³/s)
        return std::max(0.0, std::min(1.052e-6, (0.9 * pressure_psi * 1.052e-6)));
    }
    else if (nozzleType == "NPC") {
        // Non-linear relationship for NPC nozzle
        const double x = 0.477;
        const double k = 3.317;
        return k * pow(pressure_psi, x) * 1.052e-6;
    }
    else {
        throw std::invalid_argument("Unknown nozzle type: " + nozzleType);
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
