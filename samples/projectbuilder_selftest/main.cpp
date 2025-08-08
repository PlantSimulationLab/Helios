#include "ProjectBuilder.h"

using namespace helios;

int main(int argc, char** argv) {
    // Run the self-test with command line arguments
    return ProjectBuilder::selfTest(argc, argv);
}
