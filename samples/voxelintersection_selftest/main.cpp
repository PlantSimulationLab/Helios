#include "VoxelIntersection.h"

using namespace helios;

int main(int argc, char** argv) {
    Context context;
    VoxelIntersection voxelintersection(&context);
    return voxelintersection.selfTest(argc, argv);
}
