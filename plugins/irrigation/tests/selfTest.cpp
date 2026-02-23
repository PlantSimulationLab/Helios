#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"
#include "IrrigationModel.h"

using namespace helios;

namespace helios {
    void helios_runtime_error(const std::string& message) {
        throw std::runtime_error(message);
    }
} // namespace helios

const float err_tol = 1e-3f; // Error tolerance for floating-point comparisons




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

        DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));

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

        DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
    }

    SUBCASE("Water Source Pressure Assignment") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 0.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
        int ws0 = model.getWaterSourceId();
        DOCTEST_CHECK(ws0 != -1);
        DOCTEST_CHECK(model.nodes.at(ws0).pressure == doctest::Approx(0.0));

        DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, -5.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
        int ws1 = model.getWaterSourceId();
        DOCTEST_CHECK(ws1 != -1);
        DOCTEST_CHECK(model.nodes.at(ws1).pressure == doctest::Approx(-5.0));
    }

    SUBCASE("Invalid Boundary") {
        std::vector<Position> emptyBoundary;
        std::vector<Position> singlePoint = {{0.0, 0.0}};
        std::vector<Position> twoPoints = {{0.0, 0.0}, {10.0, 0.0}};

        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{emptyBoundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{singlePoint}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{twoPoints}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
    }

    SUBCASE("Invalid Spacing Values") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 0.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, -2.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
    }

    SUBCASE("Spacing Exceeds Field Dimensions") {
        std::vector<Position> smallBoundary = {
            {0.0, 0.0}, {10.0, 0.0}, {10.0, 5.0}, {0.0, 5.0}
        };

        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{smallBoundary}; model.assignZones(1, zoneBoundaries, 25.0, 20.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{smallBoundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 10.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
    }

    SUBCASE("Invalid Connection Type") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "diagonal", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
    }

    SUBCASE("Invalid Sprinkler Assembly Type") {
        std::vector<Position> boundary = {
            {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
        };

        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "Invalid_Sprinkler_Type", SubmainPosition::MIDDLE); }()));
        DOCTEST_CHECK_THROWS(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 5.0, "vertical", "", SubmainPosition::MIDDLE); }()));
    }
}

  DOCTEST_TEST_CASE("Submain Position Variations") {
      IrrigationModel model;
      std::vector<Position> boundary = {
          {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
      };

      SUBCASE("Submain at Beginning") {
          DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH); }()));
      }

      SUBCASE("Submain at Middle") {
          DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
      }

      SUBCASE("Submain at South") {
          DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH); }()));
      }
  }

 DOCTEST_TEST_CASE("System Properties for Irregular Systems") {
     IrrigationModel model;

     SUBCASE("Node and Link Counts") {
         std::vector<Position> boundary = {
             {0.0, 0.0}, {100.0, 0.0}, {100.0, 50.0}, {0.0, 50.0}
         };

         ([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 5.0, 5.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }());

         auto summary = model.getSystemSummary();

         // Should have more than 0 nodes and links
         DOCTEST_CHECK(summary.find("Total nodes: 0") == std::string::npos);
         DOCTEST_CHECK(summary.find("Total links: 0") == std::string::npos);

         // Check for specific component types
         DOCTEST_CHECK(summary.find("emitter") != std::string::npos);
         DOCTEST_CHECK(summary.find("barb") != std::string::npos);
         DOCTEST_CHECK(summary.find("lateral_sprinkler_jn") != std::string::npos);
     }

     SUBCASE("Different Sprinkler Types") {
         std::vector<Position> boundary = {
             {0.0, 0.0}, {80.0, 0.0}, {80.0, 40.0}, {0.0, 40.0}
         };

         DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 8.0, 4.0, "horizontal", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
         DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 12.0, 6.0, "vertical", "NPC_Toro_flat", SubmainPosition::MIDDLE); }()));
     }
}

  DOCTEST_TEST_CASE("Irregular System Creation") {
         IrrigationModel model;
         double Pw = 25.0;
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
             DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary}; model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH); }()));

             DOCTEST_CHECK(model.nodes.size() > 0);
             DOCTEST_CHECK(model.links.size() > 0);
             DOCTEST_CHECK(model.getWaterSourceId() != -1);
         }

         SUBCASE("Irregular Boundary") {
             DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{irregularBoundary}; model.assignZones(1, zoneBoundaries, 25.0, 16.0, 16.0, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH); }()));

             DOCTEST_CHECK(model.nodes.size() > 0);
             DOCTEST_CHECK(model.links.size() > 0);
         }

         SUBCASE("Different Submain Positions") {
             SUBCASE("North Position") {
                 CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary}; model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH); }()));
             }

             SUBCASE("South Position") {
                 DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary}; model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH); }()));
             }

             DOCTEST_SUBCASE("Middle Position") {
                 DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{rectangularBoundary}; model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }()));
             }
         }
     }

  DOCTEST_TEST_CASE("Node and Link Types") {
         IrrigationModel model;
         double Pw = 25.0;
         double lineSpacing = 22.0 * FEET_TO_METER;
         double sprinklerSpacing = 16.0 * FEET_TO_METER;
         double fieldWidth = 3 * sprinklerSpacing;
         double fieldLength = 3 * lineSpacing;

         std::vector<Position> boundary = {
             {0, 0}, {0, fieldWidth}, {fieldLength, fieldWidth}, {fieldLength, 0}
         };

         ([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH); }());

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

      ([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH); }());

      SUBCASE("Water Source Properties") {
          int wsID = model.getWaterSourceId();
          REQUIRE(wsID != -1);

          const auto& wsNode = model.nodes.at(wsID);
          DOCTEST_CHECK(wsNode.type == "waterSource");
          DOCTEST_CHECK(wsNode.is_fixed == true);
          DOCTEST_CHECK(wsNode.pressure == doctest::Approx(Pw));

          // Water source should be connected to the system
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

         // Add nodes and check ID generation
         model.nodes[1] = {1, "test", {0, 0}, 0.0, false};
         DOCTEST_CHECK(model.getNextNodeId() == 2);

         model.nodes[5] = {5, "test", {0, 0}, 0.0, false};
         DOCTEST_CHECK(model.getNextNodeId() == 6);
     }

     SUBCASE("Distance Calculations") {
         Position p1 = {0, 0};
         Position p2 = {3, 4};
         Position p3 = {0, 5};
         Position p4 = {5, 0};

         DOCTEST_CHECK(p1.distanceTo(p2) == doctest::Approx(5.0));
       //  CHECK(model.pointToSegmentDistance(p1, p3, p4) == doctest::Approx(0.0));
     }
 }

DOCTEST_TEST_CASE("Hydraulic System Validation") {
    IrrigationModel model;

    SUBCASE("Empty system throws (no water source)") {
        DOCTEST_CHECK_THROWS(model.validateHydraulicSystem());
    }

    SUBCASE("Valid generated system does not throw") {
        std::vector<Position> boundary = {
            {0.0, 0.0, 0.0},
            {0.0, 40.0, 0.0},
            {60.0, 40.0, 0.0},
            {60.0, 0.0, 0.0}
        };

        DOCTEST_CHECK_NOTHROW(([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 16.0, 22.0, "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH); }()));
        DOCTEST_CHECK_NOTHROW(model.validateHydraulicSystem());
    }
}

DOCTEST_TEST_CASE("Unassigned Nodes Check") {
    IrrigationModel model;

    SUBCASE("Empty system check does not throw") {
        DOCTEST_CHECK_NOTHROW(model.checkUnassignedNodes());
    }

    SUBCASE("System with zoneID 0 nodes check does not throw") {
        model.nodes[1] = {1, "lateral_sprinkler_jn", {0.0, 0.0, 0.0}, 0.0, false, 0.0};
        model.nodes[2] = {2, "barb", {0.1, 0.0, 0.0}, 0.0, false, 0.0};
        model.nodes[3] = {3, "emitter", {0.1, 0.0, 0.1}, 0.0, false, 0.0};

        DOCTEST_CHECK_NOTHROW(model.checkUnassignedNodes());
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
        ([&]{ std::vector<std::vector<Position>> zoneBoundaries{boundary}; model.assignZones(1, zoneBoundaries, 25.0, 10.0, 10.0, "vertical", "NPC_Nelson_flat", SubmainPosition::MIDDLE); }());

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

        model.assignZones(1, zones, Pw, sprinklerSpacing, lineSpacing,
                          "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH);
        model.initialize();
        model.activateAllZones();

        const double Qspecified = model.calculateEmitterFlow("NPC_Nelson_flat", Pw, false);
        HydraulicResults results;
        DOCTEST_CHECK_NOTHROW(results = model.calculateHydraulicsMultiZoneOptimized(
            true, "NPC_Nelson_flat", Qspecified, Pw, 1.5, 2.0));

        DOCTEST_CHECK(!results.nodalPressures.empty());
        DOCTEST_CHECK(!results.flowRates.empty());
        DOCTEST_CHECK(results.iterations > 0);
    }

    SUBCASE("Three zones") {
        IrrigationModel model;
        std::vector<std::vector<Position>> zones = {zone1, zone2, zone3};

        model.assignZones(3, zones, Pw, sprinklerSpacing, lineSpacing,
                          "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH);
        model.initialize();
        model.activateAllZones();

        const double Qspecified = model.calculateEmitterFlow("NPC_Nelson_flat", Pw, false);
        HydraulicResults results;
        DOCTEST_CHECK_NOTHROW(results = model.calculateHydraulicsMultiZoneOptimized(
            true, "NPC_Nelson_flat", Qspecified, Pw, 1.5, 2.0));

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

    DOCTEST_CHECK_NOTHROW(model.assignZones(
        1, zones, Pw, sprinklerSpacing, lineSpacing,
        "vertical", "NPC_Nelson_flat", SubmainPosition::NORTH));

    model.initialize();
    model.activateAllZones();

    SUBCASE("Emitter Flow Calculation") {
        const double flow = model.calculateEmitterFlow("NPC_Nelson_flat", Pw,
false);
        DOCTEST_CHECK(flow > 0.0);

        const double flowHigher = model.calculateEmitterFlow("NPC_Nelson_flat", Pw + 10.0, false);
        DOCTEST_CHECK(flowHigher > flow);
    }

    SUBCASE("Optimized Multi-Zone Hydraulic Calculation") {
        const double Q_specified = model.calculateEmitterFlow("NPC_Nelson_flat", Pw, false);

        HydraulicResults results;
        DOCTEST_CHECK_NOTHROW(results = model.calculateHydraulicsMultiZoneOptimized(
            true, "NPC_Nelson_flat", Q_specified, Pw, 1.5, 2.0));

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

    DOCTEST_CHECK_NOTHROW(model.assignZones(
        1, zones, Pw, sprinklerSpacing, lineSpacing,
        "vertical", "NPC_Nelson_flat", SubmainPosition::SOUTH));

    model.initialize();
    model.activateAllZones();

    SUBCASE("Empty pressure vector returns empty curve") {
        std::vector<std::pair<double, double>> curve;
        DOCTEST_CHECK_NOTHROW(curve = model.generateSystemCurve(
            {}, "NPC_Nelson_flat", 1.5, 2.0));
        DOCTEST_CHECK(curve.empty());
    }

    SUBCASE("Generates finite GPM-head points") {
        const std::vector<double> pressuresPsi = {15.0, 20.0, 25.0};
        std::vector<std::pair<double, double>> curve;
        DOCTEST_CHECK_NOTHROW(curve = model.generateSystemCurve(
            pressuresPsi, "NPC_Nelson_flat", 1.5, 2.0));

        DOCTEST_CHECK(curve.size() == pressuresPsi.size());
        for (const auto& p : curve) {
            DOCTEST_CHECK(std::isfinite(p.first));
            DOCTEST_CHECK(std::isfinite(p.second));
            DOCTEST_CHECK(p.first >= 0.0);  // GPM
        }
    }
}








// private function
// DOCTEST_TEST_CASE("Water Source Position Calculation") {
//     IrrigationModel model;
//
//     SUBCASE("Vertical Laterals") {
//         auto pos = model.calculateWaterSourcePosition(100.0, 50.0, "vertical");
//         DOCTEST_CHECK(pos.x == doctest::Approx(100.0/3.0 + 5.0));
//         DOCTEST_CHECK(pos.y == doctest::Approx(50.0 - 50.0/3.0 - 0.5));
//     }
//
//     SUBCASE("Horizontal Laterals") {
//         auto pos = model.calculateWaterSourcePosition(100.0, 50.0, "horizontal");
//         DOCTEST_CHECK(pos.x == doctest::Approx(50.0));
//         DOCTEST_CHECK(pos.y == doctest::Approx(0.0));
//     }
// }


// DOCTEST_TEST_CASE("Hydraulic Calculations") {
//     IrrigationModel model;
//
//     SUBCASE("Empty System Hydraulics") {
//         // Test with no nodes (should handle gracefully)
//         auto results = model.calculateHydraulics("NPC", 10.0, 25.0);
//
//         DOCTEST_CHECK(results.converged == false);
//         DOCTEST_CHECK(results.iterations == 0);
//         DOCTEST_CHECK(results.nodalPressures.empty());
//         DOCTEST_CHECK(results.flowRates.empty());
//     }
//
//     SUBCASE("Small Complete System with NPC Nozzle") {
//         // Create a small complete system
//         model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);
//
//         auto results = model.calculateHydraulics("NPC", 15.0, 25.0);
//
//         // Basic validation
//         DOCTEST_CHECK(results.converged == true);
//         DOCTEST_CHECK(results.iterations > 0);
//         DOCTEST_CHECK_FALSE(results.nodalPressures.empty());
//         DOCTEST_CHECK_FALSE(results.flowRates.empty());
//
//         // Water source pressure should be maintained
//         bool has_water_source_pressure = false;
//         for (const auto& pressure : results.nodalPressures) {
//             if (pressure == doctest::Approx(25.0).epsilon(0.1)) {
//                 has_water_source_pressure = true;
//                 break;
//             }
//         }
//         DOCTEST_CHECK(has_water_source_pressure == true);
//
//         // All pressures should be reasonable
//         for (const auto& pressure : results.nodalPressures) {
//             DOCTEST_CHECK(pressure >= 0.0);
//             DOCTEST_CHECK(pressure <= 30.0); // Should be less than source pressure + margin
//         }
//     }
//
//     SUBCASE("Small Complete System with NPC Nozzle") {
//         model.createCompleteSystem(30.0, 60.0, 40.0, 12.0, 6.0, "vertical", SubmainPosition::SOUTH);
//
//         auto results = model.calculateHydraulics("NPC", 20.0, 30.0);
//
//         DOCTEST_CHECK(results.converged == true);
//         DOCTEST_CHECK(results.iterations > 0);
//
//         // NPC should have non-zero flows
//         bool has_non_zero_flow = false;
//         for (const auto& flow : results.flowRates) {
//             if (flow > 0.0) {
//                 has_non_zero_flow = true;
//                 break;
//             }
//         }
//         DOCTEST_CHECK(has_non_zero_flow == true);
//
//         // Check pressure gradient (should decrease from source)
//         bool pressure_decreases = false;
//         for (size_t i = 1; i < results.nodalPressures.size(); ++i) {
//             if (results.nodalPressures[i] < results.nodalPressures[0]) {
//                 pressure_decreases = true;
//                 break;
//             }
//         }
//         DOCTEST_CHECK(pressure_decreases == true);
//     }
//
//     SUBCASE("Different Connection Types") {
//         SUBCASE("Vertical Connection") {
//             model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);
//             auto results = model.calculateHydraulics("PC", 15.0, 25.0);
//             DOCTEST_CHECK(results.converged == true);
//         }
//
//         SUBCASE("Horizontal Connection") {
//             model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "horizontal", SubmainPosition::NORTH);
//             auto results = model.calculateHydraulics("PC", 15.0, 25.0);
//             DOCTEST_CHECK(results.converged == true);
//         }
//     }
//
//     SUBCASE("Different Submain Positions") {
//         SUBCASE("North Submain") {
//             model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);
//             auto results = model.calculateHydraulics("PC", 15.0, 25.0);
//             DOCTEST_CHECK(results.converged == true);
//         }
//
//         SUBCASE("Middle Submain") {
//             model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);
//             auto results = model.calculateHydraulics("PC", 15.0, 25.0);
//             DOCTEST_CHECK(results.converged == true);
//         }
//
//         SUBCASE("South Submain") {
//             model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::SOUTH);
//             auto results = model.calculateHydraulics("PC", 15.0, 25.0);
//             DOCTEST_CHECK(results.converged == true);
//         }
//     }
//
//     SUBCASE("Zero Pressure Input") {
//         model.createCompleteSystem(0.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);
//
//         auto results = model.calculateHydraulics("PC", 10.0, 0.0);
//
//         // With zero pressure, all flows should be zero or very small
//         for (const auto& flow : results.flowRates) {
//             DOCTEST_CHECK(flow >= 0.0);
//             DOCTEST_CHECK(flow <= 1e-6); // Essentially zero
//         }
//     }
//
//     SUBCASE("High Pressure System") {
//         model.createCompleteSystem(50.0, 100.0, 80.0, 15.0, 8.0, "vertical", SubmainPosition::MIDDLE);
//
//         auto results = model.calculateHydraulics("PC", 25.0, 50.0);
//
//         DOCTEST_CHECK(results.converged == true);
//
//         // High pressure should result in higher flows
//         double total_flow = 0.0;
//         for (const auto& flow : results.flowRates) {
//             total_flow += flow;
//         }
//         DOCTEST_CHECK(total_flow > 0.0);
//     }
//
//     SUBCASE("Convergence Validation") {
//         model.createCompleteSystem(30.0, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);
//
//         auto results = model.calculateHydraulics("PC", 10.0, 30.0);
//
//         // Should converge within reasonable iterations
//         DOCTEST_CHECK(results.converged == true);
//         DOCTEST_CHECK(results.iterations <= 100); // Should not take excessive iterations
//
//         // Converged results should have finite values
//         for (const auto& pressure : results.nodalPressures) {
//             DOCTEST_CHECK(std::isfinite(pressure));
//         }
//         for (const auto& flow : results.flowRates) {
//             DOCTEST_CHECK(std::isfinite(flow));
//         }
//     }
//
//     SUBCASE("Invalid Nozzle Type") {
//         model.createCompleteSystem(25.0, 50.0, 30.0, 10.0, 5.0, "vertical", SubmainPosition::NORTH);
//
//         // Should throw for invalid nozzle type
//         DOCTEST_CHECK_THROWS_AS(model.calculateHydraulics("INVALID", 10.0, 25.0), std::invalid_argument);
//     }
//
//     SUBCASE("Mass Conservation Check") {
//         model.createCompleteSystem(30.0, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);
//
//         auto results = model.calculateHydraulics("PC", 10.0, 30.0);
//
//         // Basic sanity check: total flow should be positive
//         double total_flow = 0.0;
//         for (const auto& flow : results.flowRates) {
//             total_flow += flow;
//         }
//
//         DOCTEST_CHECK(total_flow > 0.0);
//         DOCTEST_CHECK(total_flow < 1000.0); // Reasonable upper bound
//     }
//
//     SUBCASE("Pressure Gradient Validation") {
//         model.createCompleteSystem(40.0, 80.0, 60.0, 12.0, 6.0, "vertical", SubmainPosition::NORTH);
//
//         auto results = model.calculateHydraulics("PC", 15.0, 40.0);
//
//         // Pressures should generally decrease away from water source
//         // (This is a simplified check - actual hydraulic gradient may vary)
//         bool pressure_variation_exists = false;
//         double max_pressure = *std::max_element(results.nodalPressures.begin(), results.nodalPressures.end());
//         double min_pressure = *std::min_element(results.nodalPressures.begin(), results.nodalPressures.end());
//
//         if (max_pressure - min_pressure > 1.0) {
//             pressure_variation_exists = true;
//         }
//
//         DOCTEST_CHECK(pressure_variation_exists == true);
//     }
//
//     SUBCASE("Mass Conservation Check") {
//         model.createCompleteSystem(30.0, 100.0, 50.0, 10.0, 5.0, "vertical", SubmainPosition::MIDDLE);
//
//         auto results = model.calculateHydraulics("PC", 10.0, 25.0);
//
//         // Basic mass conservation: sum of emitter flows should be reasonable
//         double total_emitter_flow = 0.0;
//         for (const auto& flow : results.flowRates) {
//             total_emitter_flow += flow;
//         }
//
//         DOCTEST_CHECK(total_emitter_flow > 0.0);
//         DOCTEST_CHECK(total_emitter_flow < 100.0); // Reasonable upper bound
//     }
//
// }



int IrrigationModel::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
