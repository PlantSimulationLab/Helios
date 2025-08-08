#include "Context.h"

int main(int argc, char** argv) {

    // Declare and initialize the Helios context
    helios::Context context;

    // Run the self-test
    return context.selfTest(argc, argv);
}
