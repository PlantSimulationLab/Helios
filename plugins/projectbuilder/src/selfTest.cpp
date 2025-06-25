#include "ProjectBuilder.h"

using namespace helios;

int ProjectBuilder::selfTest() {
    std::cout << "Running project builder self-test..." << std::flush;
    int error_count = 0;

    {
        try {
            ProjectBuilder projectbuilder;
            projectbuilder.buildFromXML();
        } catch (const std::exception &e) {
            std::cout << "failed: " << e.what() << std::endl;
            error_count++;
        }
    }

    if ( error_count > 0 ) {
        std::cout << "completed with " << error_count << " errors." << std::endl;
        return 1;
    } else {
        std::cout << "passed all tests." << std::endl;
        return 0;
    }

}