#include "/home/yuanzzzy/CLionProjects/Helios/plugins/irrigation/include/IrrigationModel.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <filesystem>



int main(){

   // Context context;
    //IrrigationModel model(&context);
    IrrigationModel system;


    // Load the converted IRRICAD data
    // if (!system.loadFromTextFile("/home/yuanzzzy/CLionProjects/Helios/plugins/irrigation/utilities/convertedIrricad_system.txt")) {
    //     std::cerr << "Failed to load system data" << std::endl;
    //     return 1;
    // }

    // Create a sprinkler system with parameters:
    // - Working pressure (Pw): 30 psi
    // - Field dimensions: 100m x 50m
    // - Sprinkler spacing: 10m
    // - Lateral spacing: 12m
    // - Connection type: "vertical" (or be "horizontal")


    std::vector<Position> boundary = {
        {0, 0}, {2*44.0 * 0.305, 0}, {0, 2*32.0 * 0.305}, {2*44.0 * 0.305, 2*32.0 * 0.305}
    };

        // double lineSpacing = 22.0* FEET_TO_METER;
        // double sprinklerSpacing = 16.0* FEET_TO_METER;
        // double fieldWidth = 4 *sprinklerSpacing;
        // double fieldLength =3* lineSpacing;
        //
        // std::vector<Position> Boundary = { //rectangular field
        //      {0, 0}, {fieldLength,0}, {fieldLength, fieldWidth}, {0,fieldWidth} //coordinates need to be in anti-clockwise
        // };

   // Example of how to use the irregular system
   // std::vector<Position> irregularBoundary = {
   //     {0, 0}, {50, 0}, {75, 25}, {50, 50}, {25, 75}, {0, 50} };

    // std::vector<Position> irregularBoundary = {
    //     {0, 0}, {100, 0}, {140, 50}, {100, 100}, {50, 150}, {0, 100}
    // };
    // system.createIrregularSystem(Pw, Boundary, sprinklerSpacing, lineSpacing, "vertical", "NPC_Nelson_flat_barb",
    //                            SubmainPosition::NORTH);

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///using zone assignment
    //std::vector<Position> zone1 = {{0,0}, {13.5,0}, {13.5,14.7}, {0,14.7}};
    // std::vector<Position> zone2 = {{50, 20},{90,20}, {90, 70}, {50, 70}};
    // std::vector<Position> zone3 = {{120, 20},{160,20}, {160, 70}, {120, 70}};
   // std::vector<Position> zone1 = { {0, 0}, {50, 0},  {50, 50}, {25, 75}, {0, 50} };                 // Bottom left
   // std::vector<Position> zone1 = { {0, 0}, {50, 0},  {50, 50}, {0, 50} };                 // Bottom left
    std::vector<Position> zone1 = { {0, 0}, {339, 0},  {339, 198}, {0, 198} };                 // Bottom left
    std::vector<Position> zone2 = {{-70,0}, {-10,0}, {-10,70}, {-70,100}};                 // Top left
    std::vector<Position> zone3 = {{-70,-90}, {-10, -90}, {-10,-20}, {-70, -20}};       // Bottom right (trapezoid)
    std::vector<Position> zone4 = {{0,-90}, {50, -90}, {50, -20}, {20, -20}, {0, -40}};

    std::string sprinklerType = "NPC_Nelson_flat";
    double Pw = 0;
    system.assignZones(1,{ zone1}, Pw,
                                  16.0 * FEET_TO_METER,
                                  22.0 * FEET_TO_METER,
                                  "vertical", sprinklerType,
                                  SubmainPosition::MIDDLE);
    system.validateHydraulicSystem();

    // Print system summary
    std::cout << system.getSystemSummary() << std::endl;

  //system.createAssembliesFromIrricad(sprinklerType);
  //  system.printNetwork();
    // Get water source info
    int wsID = system.getWaterSourceId();
    if (wsID != -1) {
        const auto& wsNode = system.nodes.at(wsID);
        std::cout << "Water Source Node ID: " << wsID << "\n"
                  << "Position: (" << wsNode.position.x << ", "
                  << wsNode.position.y << ")\n"
                  << "Pressure: " << wsNode.pressure << " psi\n";
    } else {
        std::cerr << "Error: Water source not found!\n";
    }
    // priting system summary
    std::cout << "\nSystem Summary:\n";
    std::cout << "Total nodes: " << system.nodes.size() << "\n";
    std::cout << "Total links: " << system.links.size() << "\n";

    const double Q_specified = system.calculateEmitterFlow(sprinklerType, Pw, false);
    std::cout << "Q_specified = " << Q_specified << std::endl;
    std::cout << "NodalPressures: ";

    system.initialize(); // Builds valveToNodes mapping
    system.checkUnassignedNodes();
    system.activateZones({1});

    namespace fs = std::filesystem;
    fs::path projectDir = fs::path(__FILE__).parent_path();              // .../projects/IrrigationModel
    fs::path heliosRoot = projectDir.parent_path().parent_path();        // .../Helios

    ///////////////////////////////////////codes for generating system curve///////////////////////////////////
    /////simulate system at 100% pump capacity
    // std::vector<double> emitterRequiredPressure;
    // for (double p = 5.0; p <= 35.0; p += 5.0) { //feed a range of working pressures
    //     emitterRequiredPressure.push_back(p);
    // }
    // auto curve = system.generateSystemCurve(emitterRequiredPressure, "NPC_Nelson_flat", 1.5, 2.0);
    //fs::path outPath_systecurve = heliosRoot / "plugins/irrigation/utilities/system_curve.csv";
    // //system.writeSystemCurveToCsvWithFit(outPath_systecurve, curve);
    // //system curve function Head  = a + b*GPM + c*GPM^2
    // auto [a, b, c] = system.fitCurveQuadratic(curve);
    // std::cout << "Head = " << a << " + " << b << "*GPM + " << c << "*GPM^2\n";
    /////////////////////////////////////above codes for generating system curve//////////////////////////////

    // presize function is not called in the solver
    HydraulicResults results = system.calculateHydraulicsMultiZoneOptimized(false, sprinklerType,Q_specified, Pw, 1.5, 2.0);

    double totalFlow = std::accumulate(
        results.emitterFlows.begin(),
        results.emitterFlows.end(),
        0.0
    );
    std::cout << "Raw total flow: " << totalFlow << "\n";
    std::cout << "Total Flow (L/hr): " << totalFlow * 3600000 << "\n";

     for (double p : results.nodalPressures) {
         std::cout << p << " ";
     }
     std::cout << "\n";

     for (const auto& [id, node] : system.nodes) {
         std::cout << "Node " << id << " (" << node.type << ") pressure: "
                   << node.pressure << " psi" << std::endl;
     }

    // find and print the first emitter node
     for (const auto& [id, node] : system.nodes) {
         if (node.type == "emitter") {
             std::cout << "First emitter at: ("
                       << node.position.x << ", "
                       << node.position.y << ")\n";
             break;
         }
     }

    fs::path outPath = heliosRoot / "plugins/irrigation/utilities/hydraulics_output.txt";

    fs::create_directories(outPath.parent_path());
    std::ofstream out(outPath);


    out << "NODES_START\n";
    for (const auto& [id, node] : system.nodes) {
        out << id << " "
            << node.position.x << " "
            << node.position.y << " "
            << node.position.z << " "
            << node.type << " "
            //<<node.zoneID <<" "
            << node.pressure << " "
            << (node.is_fixed ? 1 : 0) << " "
            << node.flow << " " << "\n";
    }
    out << "NODES_END\n\n";

    out << "LINKS_START\n";
    for (const auto& link : system.links) {
        out << link.from << " "
            << link.to << " "
            << link.diameter << " "
            << link.length << " "
            << link.type << " "
            << link.flow << " " << "\n";
    }
    out << "LINKS_END\n";

    out.close();

    std::cout << "Hydraulic results written to hydraulics_output.txt\n";

    std::vector<PressureLossResult> barbToEmitterLosses = system.getBarbToEmitterPressureLosses();
    //system.printPressureLossAnalysis(system);
    fs::path outPath_pressureLoss = heliosRoot / "plugins/irrigation/utilities/pressure_losses.txt";

    system.writePressureLossesToFile(system, outPath_pressureLoss);



    return 0;


    // model.readDXF("../files/example.dxf");
    //
    // model.solve();
    //
    // model.writeDXF("../files/example_solved.dxf");
    //
    // return 0;

}
