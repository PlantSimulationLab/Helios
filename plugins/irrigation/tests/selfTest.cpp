#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"
#include "global.h"
#include "IrrigationModel.h"

using namespace helios;

namespace helios {
    void helios_runtime_error(const std::string& message) {
        throw std::runtime_error(message);
    }
} // namespace helios

const float err_tol = 1e-3f; // Error tolerance for floating-point comparisons

// Helper: run a callable inside capture_cout + capture_cerr, returning whether it threw.
// This keeps all model output silent while allowing doctest assertions outside the capture scope.
template<typename Func>
bool run_silently(Func&& func) {
    bool threw = false;
    {
        capture_cout cap_out;
        capture_cerr cap_err;
        try {
            func();
        } catch (...) {
            threw = true;
        }
    }
    return threw;
}

DOCTEST_TEST_CASE("Basic System Creation") {
    IrrigationModel model;

    SUBCASE("Empty System") {
        auto summary = model.getSystemSummary();
        DOCTEST_CHECK(summary.find("Total nodes: 0") != std::string::npos);
        DOCTEST_CHECK(summary.find("Total links: 0") != std::string::npos);
    }

    SUBCASE("Complete System Creation") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK_FALSE(threw);

        auto summary = model.getSystemSummary();
        DOCTEST_CHECK(summary.find("waterSource") != std::string::npos);
        DOCTEST_CHECK(summary.find("lateral") != std::string::npos);
        DOCTEST_CHECK(model.getWaterSourceId() != -1);
    }
}

DOCTEST_TEST_CASE("Parameter Validation") {
    IrrigationModel model;

    SUBCASE("Valid Parameters") {
        std::vector<Position> rectangularBoundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK_FALSE(threw);
    }

    SUBCASE("Water Source Pressure Assignment") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 0.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK_FALSE(threw);
        int ws0 = model.getWaterSourceId();
        DOCTEST_CHECK(ws0 != -1);
        DOCTEST_CHECK(model.nodes.at(ws0).pressure == doctest::Approx(0.0));

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, -5.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK_FALSE(threw);
        int ws1 = model.getWaterSourceId();
        DOCTEST_CHECK(ws1 != -1);
        DOCTEST_CHECK(model.nodes.at(ws1).pressure == doctest::Approx(-5.0));
    }

    SUBCASE("Invalid Boundary") {
        std::vector<Position> emptyBoundary;
        std::vector<Position> singlePoint = {{0.0, 0.0}};
        std::vector<Position> twoPoints = {{0.0, 0.0}, {10.0, 0.0}};

        bool threw;
        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{emptyBoundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{singlePoint};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{twoPoints};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);
    }

    SUBCASE("Invalid Spacing Values") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        bool threw;
        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 0.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, -2.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);
    }

    SUBCASE("Spacing Exceeds Field Dimensions") {
        std::vector<Position> smallBoundary = {
            {0.0, 0.0}, {10.0, 0.0}, {10.0, 5.0}, {0.0, 5.0}
        };

        bool threw;
        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{smallBoundary};
            model.assignZones(1, zoneBoundaries, 25.0, 20.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{smallBoundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 10.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);
    }

    SUBCASE("Invalid Connection Type") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        bool threw;
        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "diagonal", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);
    }

    SUBCASE("Invalid Sprinkler Assembly Type") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        bool threw;
        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "Invalid_Sprinkler_Type", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK(threw);
    }
}

DOCTEST_TEST_CASE("Submain Position Variations") {
    IrrigationModel model;
    std::vector<Position> boundary = {
        {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
    };

    SUBCASE("Submain at Beginning") {
        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
        });
        DOCTEST_CHECK_FALSE(threw);
    }

    SUBCASE("Submain at Middle") {
        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK_FALSE(threw);
    }

    SUBCASE("Submain at South") {
        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH);
        });
        DOCTEST_CHECK_FALSE(threw);
    }
}

DOCTEST_TEST_CASE("System Properties for Irregular Systems") {
    IrrigationModel model;

    SUBCASE("Node and Link Counts") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });

        auto summary = model.getSystemSummary();

        DOCTEST_CHECK(summary.find("Total nodes: 0") == std::string::npos);
        DOCTEST_CHECK(summary.find("Total links: 0") == std::string::npos);
        DOCTEST_CHECK(summary.find("emitter") != std::string::npos);
        DOCTEST_CHECK(summary.find("barb") != std::string::npos);
        DOCTEST_CHECK(summary.find("lateral_sprinkler_jn") != std::string::npos);
    }

    SUBCASE("Different Sprinkler Types") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {80.0, 0.0}, {80.0, 40.0}, {0.0, 40.0}
        };

        bool threw;
        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 8.0, 4.0, "horizontal", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK_FALSE(threw);

        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 12.0, 6.0, "vertical", "NPC_Toro_flat", SubmainPosition::MIDDLE);
        });
        DOCTEST_CHECK_FALSE(threw);
    }
}

DOCTEST_TEST_CASE("Irregular System Creation") {
    IrrigationModel model;
    double lineSpacing = 22.0 * FEET_TO_METER;
    double sprinklerSpacing = 16.0 * FEET_TO_METER;
    double fieldWidth = 3 * sprinklerSpacing;
    double fieldLength = 3 * lineSpacing;

    std::vector<Position> rectangularBoundary = {
        {0, 0}, {0, fieldWidth}, {fieldLength, fieldWidth}, {fieldLength, 0}
    };

    std::vector<Position> irregularBoundary = {
        {0, 0}, {50, 0}, {75, 25}, {50, 50}, {25, 75}, {0, 50}
    };

    SUBCASE("Rectangular Boundary") {
        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary};
            model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
        });
        DOCTEST_CHECK_FALSE(threw);

        DOCTEST_CHECK(model.nodes.size() > 0);
        DOCTEST_CHECK(model.links.size() > 0);
        DOCTEST_CHECK(model.getWaterSourceId() != -1);
    }

    SUBCASE("Irregular Boundary") {
        bool threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{irregularBoundary};
            model.assignZones(1, zoneBoundaries, 25.0, 16.0, 16.0, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
        });
        DOCTEST_CHECK_FALSE(threw);

        DOCTEST_CHECK(model.nodes.size() > 0);
        DOCTEST_CHECK(model.links.size() > 0);
    }

    SUBCASE("Different Submain Positions") {
        SUBCASE("North Position") {
            bool threw = run_silently([&]{
                std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary};
                model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
            });
            DOCTEST_CHECK_FALSE(threw);
        }

        SUBCASE("South Position") {
            bool threw = run_silently([&]{
                std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary};
                model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH);
            });
            DOCTEST_CHECK_FALSE(threw);
        }

        SUBCASE("Middle Position") {
            bool threw = run_silently([&]{
                std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary};
                model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
            });
            DOCTEST_CHECK_FALSE(threw);
        }
    }
}

DOCTEST_TEST_CASE("Node and Link Types") {
    IrrigationModel model;
    double lineSpacing = 22.0 * FEET_TO_METER;
    double sprinklerSpacing = 16.0 * FEET_TO_METER;
    double fieldWidth = 3 * sprinklerSpacing;
    double fieldLength = 3 * lineSpacing;

    std::vector<Position> boundary = {
        {0, 0}, {0, fieldWidth}, {fieldLength, fieldWidth}, {fieldLength, 0}
    };

    run_silently([&]{
        std::vector<std::vector<Position>> zoneBoundaries{boundary};
        model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
    });

    SUBCASE("Node Type Counts") {
        int lateralJunctions = 0;
        int barbs = 0;
        int emitters = 0;
        int submainJunctions = 0;
        int waterSources = 0;

        for (const auto& [id, node] : model.nodes) {
            if (node.type == "lateral_sprinkler_jn") lateralJunctions++;
            else if (node.type == "barb") barbs++;
            else if (node.type == "emitter") emitters++;
            else if (node.type == "submain_junction") submainJunctions++;
            else if (node.type == "waterSource") waterSources++;
        }

        DOCTEST_CHECK(lateralJunctions > 0);
        DOCTEST_CHECK(barbs > 0);
        DOCTEST_CHECK(emitters > 0);
        DOCTEST_CHECK(submainJunctions > 0);
        DOCTEST_CHECK(lateralJunctions == barbs);
        DOCTEST_CHECK(barbs == emitters);
    }

    SUBCASE("Link Type Counts") {
        int laterals = 0;
        int barbToEmitter = 0;
        int lateralToBarb = 0;
        int submain = 0;
        int mainline = 0;

        for (const auto& link : model.links) {
            if (link.type == "lateral") laterals++;
            else if (link.type == "barbToemitter") barbToEmitter++;
            else if (link.type == "lateralTobarb") lateralToBarb++;
            else if (link.type == "submain") submain++;
            else if (link.type == "mainline") mainline++;
        }

        DOCTEST_CHECK(laterals > 0);
        DOCTEST_CHECK(barbToEmitter > 0);
        DOCTEST_CHECK(lateralToBarb > 0);
        DOCTEST_CHECK(submain > 0);
    }
}

DOCTEST_TEST_CASE("Water Source Configuration") {
    IrrigationModel model;
    double Pw = 25.0;
    double lineSpacing = 22.0 * FEET_TO_METER;
    double sprinklerSpacing = 16.0 * FEET_TO_METER;
    double fieldWidth = 3 * sprinklerSpacing;
    double fieldLength = 3 * lineSpacing;

    std::vector<Position> boundary = {
        {0, 0}, {0, fieldWidth}, {fieldLength, fieldWidth}, {fieldLength, 0}
    };

    run_silently([&]{
        std::vector<std::vector<Position>> zoneBoundaries{boundary};
        model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
    });

    SUBCASE("Water Source Properties") {
        int wsID = model.getWaterSourceId();
        REQUIRE(wsID != -1);

        const auto& wsNode = model.nodes.at(wsID);
        DOCTEST_CHECK(wsNode.type == "waterSource");
        DOCTEST_CHECK(wsNode.is_fixed == true);
        DOCTEST_CHECK(wsNode.pressure == doctest::Approx(Pw));

        bool isConnected = false;
        for (const auto& link : model.links) {
            if (link.from == wsID || link.to == wsID) {
                isConnected = true;
                break;
            }
        }
        DOCTEST_CHECK(isConnected == true);
    }
}

DOCTEST_TEST_CASE("Help Functions") {
    IrrigationModel model;

    SUBCASE("Next Node ID") {
        DOCTEST_CHECK(model.getNextNodeId() == 1);
        DOCTEST_CHECK(model.getNextNodeId() == 2);
        DOCTEST_CHECK(model.getNextNodeId() == 3);

        int id = model.getNextNodeId(); // returns 4
        model.nodes[id] = {id, "test", {0, 0}, 0.0, false};
        DOCTEST_CHECK(model.getNextNodeId() == 5);
    }

    SUBCASE("Distance Calculations") {
        Position p1 = {0, 0};
        Position p2 = {3, 4};

        DOCTEST_CHECK(p1.distanceTo(p2) == doctest::Approx(5.0));
    }
}

DOCTEST_TEST_CASE("Hydraulic System Validation") {
    IrrigationModel model;

    SUBCASE("Empty system throws (no water source)") {
        bool threw = run_silently([&]{ model.validateHydraulicSystem(); });
        DOCTEST_CHECK(threw);
    }

    SUBCASE("Valid generated system does not throw") {
        std::vector<Position> boundary = {
            {0.0, 0.0, 0.0},
            {0.0, 40.0, 0.0},
            {60.0, 40.0, 0.0},
            {60.0, 0.0, 0.0}
        };

        bool threw;
        threw = run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 16.0, 22.0, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
        });
        DOCTEST_CHECK_FALSE(threw);

        threw = run_silently([&]{ model.validateHydraulicSystem(); });
        DOCTEST_CHECK_FALSE(threw);
    }
}

DOCTEST_TEST_CASE("Unassigned Nodes Check") {
    IrrigationModel model;

    SUBCASE("Empty system check does not throw") {
        bool threw = run_silently([&]{ model.checkUnassignedNodes(); });
        DOCTEST_CHECK_FALSE(threw);
    }

    SUBCASE("System with zoneID 0 nodes check does not throw") {
        model.nodes[1] = {1, "lateral_sprinkler_jn", {0.0, 0.0, 0.0}, 0.0, false, 0.0};
        model.nodes[2] = {2, "barb", {0.1, 0.0, 0.0}, 0.0, false, 0.0};
        model.nodes[3] = {3, "emitter", {0.1, 0.0, 0.1}, 0.0, false, 0.0};

        bool threw = run_silently([&]{ model.checkUnassignedNodes(); });
        DOCTEST_CHECK_FALSE(threw);
    }
}

DOCTEST_TEST_CASE("Emitter Flow Calculation") {
    IrrigationModel model;

    SUBCASE("NPC flow increases with pressure") {
        const double q_low = model.calculateEmitterFlow("NPC_Nelson_flat", 10.0, false);
        const double q_high = model.calculateEmitterFlow("NPC_Nelson_flat", 30.0, false);

        DOCTEST_CHECK(q_low > 0.0);
        DOCTEST_CHECK(q_high > q_low);
    }

    SUBCASE("updateEmitterNodes writes emitter node flows") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {60.0, 0.0}, {60.0, 40.0}, {0.0, 40.0}
        };
        run_silently([&]{
            std::vector<std::vector<Position>> zoneBoundaries{boundary};
            model.assignZones(1, zoneBoundaries, 25.0, 10.0, 10.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE);
        });

        const double expected = model.calculateEmitterFlow("NPC_Nelson_flat", 20.0, true);

        bool foundEmitter = false;
        for (const auto& [id, node] : model.nodes) {
            if (node.type == "emitter") {
                foundEmitter = true;
                DOCTEST_CHECK(node.flow == doctest::Approx(expected));
            }
        }
        DOCTEST_CHECK(foundEmitter);
    }
}

DOCTEST_TEST_CASE("Multi-zone Optimized Hydraulics") {
    const double Pw = 30.0;
    const double sprinklerSpacing = 16.0 * FEET_TO_METER;
    const double lineSpacing = 22.0 * FEET_TO_METER;

    std::vector<Position> zone1 = {
        {0.0, 0.0, 0.0}, {30.0, 0.0, 0.0}, {30.0, 30.0, 0.0}, {0.0, 30.0, 0.0}
    };
    std::vector<Position> zone2 = {
        {30.0, 0.0, 0.0}, {60.0, 0.0, 0.0}, {60.0, 30.0, 0.0}, {30.0, 30.0, 0.0}
    };
    std::vector<Position> zone3 = {
        {60.0, 0.0, 0.0}, {90.0, 0.0, 0.0}, {90.0, 30.0, 0.0}, {60.0, 30.0, 0.0}
    };

    SUBCASE("Single zone") {
        IrrigationModel model;
        std::vector<std::vector<Position>> zones = {zone1};

        HydraulicResults results;
        bool threw = run_silently([&]{
            model.assignZones(1, zones, Pw, sprinklerSpacing, lineSpacing,
                              "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH);
            model.initialize();
            model.activateAllZones();

            const double Qspecified = model.calculateEmitterFlow("NPC_Nelson_flat", Pw, false);
            results = model.calculateHydraulicsMultiZoneOptimized(
                true, "NPC_Nelson_flat", Qspecified, Pw, 1.5, 2.0);
        });
        DOCTEST_CHECK_FALSE(threw);

        DOCTEST_CHECK(!results.nodalPressures.empty());
        DOCTEST_CHECK(!results.flowRates.empty());
        DOCTEST_CHECK(results.iterations > 0);
    }

    SUBCASE("Three zones") {
        IrrigationModel model;
        std::vector<std::vector<Position>> zones = {zone1, zone2, zone3};

        HydraulicResults results;
        bool threw = run_silently([&]{
            model.assignZones(3, zones, Pw, sprinklerSpacing, lineSpacing,
                              "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH);
            model.initialize();
            model.activateAllZones();

            const double Qspecified = model.calculateEmitterFlow("NPC_Nelson_flat", Pw, false);
            results = model.calculateHydraulicsMultiZoneOptimized(
                true, "NPC_Nelson_flat", Qspecified, Pw, 1.5, 2.0);
        });
        DOCTEST_CHECK_FALSE(threw);

        DOCTEST_CHECK(!results.nodalPressures.empty());
        DOCTEST_CHECK(!results.flowRates.empty());
        DOCTEST_CHECK(results.iterations > 0);
    }
}

DOCTEST_TEST_CASE("Hydraulic Calculations Optimized Multi-Zone") {
    IrrigationModel model;
    const double Pw = 25.0;
    const double lineSpacing = 22.0 * FEET_TO_METER;
    const double sprinklerSpacing = 16.0 * FEET_TO_METER;

    std::vector<Position> zone1 = {
        {0.0, 0.0, 0.0},
        {0.0, 3.0 * sprinklerSpacing, 0.0},
        {3.0 * lineSpacing, 3.0 * sprinklerSpacing, 0.0},
        {3.0 * lineSpacing, 0.0, 0.0}
    };
    std::vector<std::vector<Position>> zones = {zone1};

    run_silently([&]{
        model.assignZones(1, zones, Pw, sprinklerSpacing, lineSpacing,
                          "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH);
        model.initialize();
        model.activateAllZones();
    });

    SUBCASE("Emitter Flow Calculation") {
        const double flow = model.calculateEmitterFlow("NPC_Nelson_flat", Pw, false);
        DOCTEST_CHECK(flow > 0.0);

        const double flowHigher = model.calculateEmitterFlow("NPC_Nelson_flat", Pw + 10.0, false);
        DOCTEST_CHECK(flowHigher > flow);
    }

    SUBCASE("Optimized Multi-Zone Hydraulic Calculation") {
        const double Q_specified = model.calculateEmitterFlow("NPC_Nelson_flat", Pw, false);

        HydraulicResults results;
        bool threw = run_silently([&]{
            results = model.calculateHydraulicsMultiZoneOptimized(
                true, "NPC_Nelson_flat", Q_specified, Pw, 1.5, 2.0);
        });
        DOCTEST_CHECK_FALSE(threw);

        DOCTEST_CHECK(results.iterations > 0);
        DOCTEST_CHECK(!results.nodalPressures.empty());
        DOCTEST_CHECK(!results.flowRates.empty());
        DOCTEST_CHECK(results.converged == true);
    }
}

DOCTEST_TEST_CASE("Generate System Curve") {
    IrrigationModel model;
    const double Pw = 30.0;
    const double sprinklerSpacing = 16.0 * FEET_TO_METER;
    const double lineSpacing = 22.0 * FEET_TO_METER;

    std::vector<Position> zone1 = {
        {0.0, 0.0, 0.0},
        {0.0, 3.0 * sprinklerSpacing, 0.0},
        {3.0 * lineSpacing, 3.0 * sprinklerSpacing, 0.0},
        {3.0 * lineSpacing, 0.0, 0.0}
    };
    std::vector<std::vector<Position>> zones = {zone1};

    run_silently([&]{
        model.assignZones(1, zones, Pw, sprinklerSpacing, lineSpacing,
                          "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH);
        model.initialize();
        model.activateAllZones();
    });

    SUBCASE("Empty pressure vector returns empty curve") {
        std::vector<std::pair<double, double>> curve;
        bool threw = run_silently([&]{
            curve = model.generateSystemCurve({}, "NPC_Nelson_flat", 1.5, 2.0);
        });
        DOCTEST_CHECK_FALSE(threw);
        DOCTEST_CHECK(curve.empty());
    }

    SUBCASE("Generates finite GPM-head points") {
        const std::vector<double> pressuresPsi = {15.0, 20.0, 25.0};
        std::vector<std::pair<double, double>> curve;
        bool threw = run_silently([&]{
            curve = model.generateSystemCurve(pressuresPsi, "NPC_Nelson_flat", 1.5, 2.0);
        });
        DOCTEST_CHECK_FALSE(threw);

        DOCTEST_CHECK(curve.size() == pressuresPsi.size());
        for (const auto& p : curve) {
            DOCTEST_CHECK(std::isfinite(p.first));
            DOCTEST_CHECK(std::isfinite(p.second));
            DOCTEST_CHECK(p.first >= 0.0);  // GPM
        }
    }
}

int IrrigationModel::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
