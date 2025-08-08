#include "CanopyGenerator.h"

using namespace helios;

int main(int argc, char** argv) {
    // Run the self-test with command line arguments
    Context context;
    CanopyGenerator canopygenerator(&context);
    return canopygenerator.selfTest(argc, argv);
}
