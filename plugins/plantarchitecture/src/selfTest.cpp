/** \file "selfTest.cpp" Self-test routines for Plant Architecture plug-in.

Copyright (C) 2016-2025 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "PlantArchitecture.h"

using namespace std;
using namespace helios;

int PlantArchitecture::selfTest() {

    int error_count = 0;

    std::vector<std::string> plant_labels{"almond", "apple", "asparagus", "bindweed", "bean", "cheeseweed", "cowpea", "grapevine_VSP", "maize", "olive", "pistachio", "puncturevine", "easternredbud", "rice", "butterlettuce", "sorghum", "soybean", "strawberry", "sugarbeet", "tomato", "walnut", "wheat"};

    auto test_buildPlantFromLibrary = [&error_count]( std::string plant_label) {
        try {
            std::cout << "Building " << plant_label << " plant model..." << std::flush;
            Context context;
            PlantArchitecture plantarchitecture(&context);
            plantarchitecture.loadPlantModelFromLibrary( plant_label );
            plantarchitecture.buildPlantInstanceFromLibrary(make_vec3(0,0,0), 5000);
            std::cout << "done." << std::endl;
        }catch (std::exception &e) {
            std::cerr << plant_label << " model failed." << std::endl;
            std::cerr << e.what() << std::endl;
            error_count++;
        }
    };

    std::cout << "Running self-test to build all plants in the library..." << std::endl;

    for ( auto &plant_label : plant_labels ) {
        test_buildPlantFromLibrary( plant_label );
    }

    if ( error_count==0 ) {
        std::cout << "passed." << std::endl;
        return 0;
    } else {
        std::cout << "failed " << error_count << " tests." << std::endl;
        return 1;
    }

}
