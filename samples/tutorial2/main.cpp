#include "Visualizer.h"

using namespace helios; // note that we are using the helios namespace so we can omit 'helios::' before names

int main() {

    Context context; // Declare the "Context" class

    vec3 position(0, 0, 0); //(x,y,z) position of patch center
    vec2 size(1, 1); // length and width of patch

    uint UUID; // declare the UUID variable

    UUID = context.addPatch(position, size); // this will assign the UUID for this patch to the UUID variable

    // Visualizer code -- uncomment out this code below, it is commented to allow it to run as part of automated tests
    //  Visualizer vis(800);
    //  vis.buildContextGeometry(&context);
    //  vis.plotInteractive();

    return 0;
}
