#include "/home/yuanzzzy/CLionProjects/Helios/plugins/irrigation/include/IrrigationModel.h"

#include <iostream>
#include <fstream>

int main(){

   // Context context;
    //IrrigationModel model(&context);
    IrrigationModel system;

    // Create a sprinkler system with parameters:
    // - Working pressure (Pw): 30 psi
    // - Field dimensions: 100m x 50m
    // - Sprinkler spacing: 10m
    // - Lateral spacing: 12m
    // - Connection type: "vertical" (could also be "horizontal")

    double Pw = 30.0;
        // system.createCompleteSystem(
        //     Pw,    // Pw (psi)
        //     44.0 * IrrigationModel::FEET_TO_METER,   // fieldLength (m)
        //     32.0 * IrrigationModel::FEET_TO_METER,    // fieldWidth (m)
        //     22.0 * IrrigationModel::FEET_TO_METER,    // lineSpacing (m)
        //     16.0 * IrrigationModel::FEET_TO_METER,    // sprinklerSpacing (m)
        //     "vertical", // lateralDirection
        //     SubmainPosition:: MIDDLE
        // );
    //
    // std::vector<Position> boundary = {
    //     {0, 0}, {44.0 * IrrigationModel::FEET_TO_METER, 0}, {0, 32.0 * IrrigationModel::FEET_TO_METER}, {44.0 * IrrigationModel::FEET_TO_METER, 32.0 * IrrigationModel::FEET_TO_METER}
    // };


    double lineSpacing = 22.0* FEET_TO_METER;
    double sprinklerSpacing = 16.0* FEET_TO_METER;
    double fieldWidth = 3*sprinklerSpacing;
    double fieldLength = 2*lineSpacing;
    std::vector<Position> Boundary = { //rectangular field
         {0, 0}, {0,fieldWidth}, {fieldLength, fieldWidth}, {fieldLength,0} //coordinates need to be in anti-clockwise
    };

   // Example of how to use the irregular system
   // std::vector<Position> irregularBoundary = {
   //     {0, 0}, {50, 0}, {75, 25}, {50, 50}, {25, 75}, {0, 50} };

    std::vector<Position> irregularBoundary = {
        {0, 0}, {100, 0}, {140, 50}, {100, 100}, {50, 150}, {0, 100}
    };

  system.createIrregularSystem(Pw, Boundary, lineSpacing, sprinklerSpacing, "vertical", "NPC_Nelson_flat_barb",
                             SubmainPosition::NORTH);
   // system.createIrregularSystem(Pw, Boundary, sprinklerSpacing, lineSpacing, "vertical",
                          //     SubmainPosition::MIDDLE);
    // system.createIrregularSystem(Pw, irregularBoundary, 16.0, 16.0, "vertical", SubmainPosition::MIDDLE);

    // Calculate hydraulics
    //auto results = system.calculateHydraulics("PC", 1.0, 50.0);

    // Print system summary
    std::cout << system.getSystemSummary() << std::endl;



    // Create irregular irrigation system
    //system.createIrregularSystem(Pw, Boundary,60.0 * IrrigationModel::FEET_TO_METER, 40.0 * IrrigationModel::FEET_TO_METER, "vertical", SubmainPosition::MIDDLE);

    system.printNetwork();
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

    const double Q_specified = system.calculateEmitterFlow("NPC_Nelson_flat_barb", Pw);
    HydraulicResults results = system.calculateHydraulics(true, "NPC_Nelson_flat_barb",Q_specified, Pw, 1.5, 2.0);     std::cout << "NodalPressures: ";

   // HydraulicResults results = system.calculateHydraulics(false, "NPC_Nelson_flat_barb", Q_specified, Pw, 1.5, 2.0);
   // std::cout << "NodalPressures: ";
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

    std::ofstream out("/home/yuanzzzy/CLionProjects/Helios/plugins/irrigation/utilities/hydraulics_output.txt");

    out << "NODES_START\n";
    for (const auto& [id, node] : system.nodes) {
        out << id << " "
            << node.position.x << " "
            << node.position.y << " "
            << node.type << " "
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

    return 0;


    // model.readDXF("../files/example.dxf");
    //
    // model.solve();
    //
    // model.writeDXF("../files/example_solved.dxf");
    //
    // return 0;

}
