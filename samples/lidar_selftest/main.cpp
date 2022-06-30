#include "Context.h"
#include "LiDAR.h"

using namespace helios;

int main(int argc, char* argv[]) {
    LiDARcloud pointcloud;

    return pointcloud.selfTest();
}
