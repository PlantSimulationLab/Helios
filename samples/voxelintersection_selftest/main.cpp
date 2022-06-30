#include "VoxelIntersection.h"

using namespace helios;

int main() {
    Context context;
    VoxelIntersection voxelintersection(&context);

    return voxelintersection.selfTest();
}
