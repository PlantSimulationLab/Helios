#include "CanopyGenerator.h"

using namespace helios;

int main() {
    // Run the self-test
    Context context;

    CanopyGenerator canopygenerator(&context);
    return canopygenerator.selfTest();
}
