



#include "IrrigationModel.h"

#include <iostream>

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
        system.createCompleteSystem(
            Pw,    // Pw (psi)
            20.0,   // fieldLength (m)
            50.0,    // fieldWidth (m)
            10.0,    // sprinklerSpacing (m)
            25.0,    // lineSpacing (m)
            "vertical" // lateralDirection
        );
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


    auto results = system.calculateHydraulics("NPC", 0.0001, Pw);

    // find and print the first emitter node
    // for (const auto& [id, node] : system.nodes) {
    //     if (node.type == "emitter") {
    //         std::cout << "First emitter at: ("
    //                   << node.position.x << ", "
    //                   << node.position.y << ")\n";
    //         break;
    //     }
    // }

    return 0;
    // model.readDXF("../files/example.dxf");
    //
    // model.solve();
    //
    // model.writeDXF("../files/example_solved.dxf");
    //
    // return 0;

}
