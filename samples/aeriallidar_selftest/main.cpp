#include "AerialLiDAR.h"
#include "Context.h"

using namespace helios;

int main(int argc, char** argv) {
    AerialLiDARcloud aeriallidar;
    return aeriallidar.selfTest(argc, argv);
}
