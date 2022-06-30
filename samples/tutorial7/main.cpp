#include "Visualizer.h"

using namespace helios;  // note that we are using the helios namespace so we can omit 'helios::' before names

int main() {
    Context context;  // Declare the "Context" class

    vec3 position(0, 0, 0);  //(x,y,z) position of patch center
    vec2 size(1, 1);         // length and width of patch

    context.addPatch(position, size);  // add the patch to Context

    Visualizer vis(800);  // creates a display window 800 pixels wide

    vis.buildContextGeometry(&context);  // add all geometry in the context to the visualizer

    vis.plotUpdate();               // update the graphics window and move on
    vis.printWindow("patch.jpeg");  // print window to JPEG file

    return 0;
}
