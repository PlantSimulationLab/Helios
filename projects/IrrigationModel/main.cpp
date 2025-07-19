



#include "IrrigationModel.h"

using namespace helios;

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
    system.createSprinklerSystem(30.0, 20, 20, 10.0, 10, "vertical");

    // Print the network data for visualization
    system.printNetwork();

    // priting system summary
    std::cout << "\nSystem Summary:\n";
    std::cout << "Total nodes: " << system.nodes.size() << "\n";
    std::cout << "Total links: " << system.links.size() << "\n";

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
