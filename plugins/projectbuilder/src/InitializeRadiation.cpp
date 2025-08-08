#include "InitializeRadiation/InitializeRadiation.h"
#include "SolarPosition.h"

using namespace helios;

void InitializeRadiation(const std::string &xml_input_file, SolarPosition *solarposition_ptr, RadiationModel *radiation_ptr, helios::Context *context_ptr) {

    pugi::xml_document xmldoc;

    std::string xml_error_string;
    if (!open_xml_file(xml_input_file, xmldoc, xml_error_string)) {
        helios_runtime_error(xml_error_string);
    }

    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;

    // *** Parsing of general inputs *** //

    int radiation_block_count = 0;
    for (pugi::xml_node radiation_block = helios.child("radiation"); radiation_block; radiation_block = radiation_block.next_sibling("radiation")) {
        radiation_block_count++;

        if (radiation_block_count > 1) {
            std::cout << "WARNING: Only one 'radiation' block is allowed in the input file. Skipping any others..." << std::endl;
            break;
        }

        int direct_ray_count = 100;
        node = radiation_block.child("direct_ray_count");
        if (node.empty()) {
            direct_ray_count = 0;
        } else {

            const char *direct_ray_count_str = node.child_value();
            if (!parse_int(direct_ray_count_str, direct_ray_count)) {
                helios_runtime_error("ERROR: Value given for 'direct_ray_count' could not be parsed.");
            } else if (direct_ray_count < 0) {
                helios_runtime_error("ERROR: Value given for 'direct_ray_count' must be greater than or equal to 0.");
            }
        }

        int diffuse_ray_count = 1000;
        node = radiation_block.child("diffuse_ray_count");
        if (node.empty()) {
            diffuse_ray_count = 0;
        } else {

            const char *diffuse_ray_count_str = node.child_value();
            if (!parse_int(diffuse_ray_count_str, diffuse_ray_count)) {
                helios_runtime_error("ERROR: Value given for 'diffuse_ray_count' could not be parsed.");
            } else if (diffuse_ray_count < 0) {
                helios_runtime_error("ERROR: Value given for 'diffuse_ray_count' must be greater than or equal to 0.");
            }
        }

        float diffuse_extinction_coeff = 0;
        node = radiation_block.child("diffuse_extinction_coeff");
        if (node.empty() && diffuse_ray_count > 0) {
            std::cout << "WARNING: No value given for 'diffuse_extinction_coeff'. Assuming a uniform overcast sky." << std::endl;
        } else {

            const char *diffuse_extinction_coeff_str = node.child_value();
            if (!parse_float(diffuse_extinction_coeff_str, diffuse_extinction_coeff)) {
                helios_runtime_error("ERROR: Value given for 'diffuse_extinction_coeff' could not be parsed.");
            } else if (diffuse_extinction_coeff < 0) {
                helios_runtime_error("ERROR: Value given for 'diffuse_extinction_coeff' must be greater than or equal to 0.");
            }
        }
        context_ptr->setGlobalData("diffuse_extinction_coeff", diffuse_extinction_coeff);

        int scattering_depth = 0;
        node = radiation_block.child("scattering_depth");
        if (!node.empty()) {

            const char *scattering_depth_str = node.child_value();
            if (!parse_int(scattering_depth_str, scattering_depth)) {
                helios_runtime_error("ERROR: Value given for 'scattering_depth' could not be parsed.");
            } else if (scattering_depth < 0) {
                helios_runtime_error("ERROR: Value given for 'scattering_depth' must be greater than or equal to 0.");
            }
        }

        float air_turbidity = 0;
        node = radiation_block.child("air_turbidity");
        if (!node.empty()) {

            const char *air_turbidity_str = node.child_value();

            // first try parsing as a float to see if an actual turbidity value was specified
            if (!parse_float(air_turbidity_str, air_turbidity)) {

                // if parsing fails, try parsing as a string to see if 'calibrate' was specified
                if (trim_whitespace(std::string(air_turbidity_str)) == "calibrate") {
                    air_turbidity = -1; // set to -1 to indicate calibration mode
                } else {
                    helios_runtime_error("ERROR: Value given for 'air_turbidity' could not be parsed.");
                }
            } else if (air_turbidity < 0) {
                helios_runtime_error("ERROR: Value given for 'air_turbidity' must be greater than or equal to 0.");
            }
        }

        // *** Loading any XML files needed for spectra *** //

        for (pugi::xml_node p = radiation_block.child("load_xml_library_file"); p; p = p.next_sibling("load_xml_library_file")) {

            const char *xml_library_file_str = p.child_value();
            std::string xml_library_file = trim_whitespace(std::string(xml_library_file_str));

            if (xml_library_file.empty() || !std::filesystem::exists(xml_library_file)) {
                std::cout << "WARNING: Could not find XML library file: " + xml_library_file << ". Skipping..." << std::endl;
                continue;
            }

            context_ptr->loadXML(xml_library_file.c_str());
        }

        // *** Spectral data *** //

        std::string solar_direct_spectrum;
        node = radiation_block.child("solar_direct_spectrum");
        if (!node.empty()) {

            const char *solar_direct_spectrum_str = node.child_value();
            solar_direct_spectrum = trim_whitespace(std::string(solar_direct_spectrum_str));
        }

        std::string leaf_reflectivity_spectrum;
        node = radiation_block.child("leaf_reflectivity_spectrum");
        if (!node.empty()) {

            const char *leaf_reflectivity_spectrum_str = node.child_value();
            leaf_reflectivity_spectrum = trim_whitespace(std::string(leaf_reflectivity_spectrum_str));
        }

        std::string leaf_transmissivity_spectrum;
        node = radiation_block.child("leaf_transmissivity_spectrum");
        if (!node.empty()) {

            const char *leaf_transmissivity_spectrum_str = node.child_value();
            leaf_transmissivity_spectrum = trim_whitespace(std::string(leaf_transmissivity_spectrum_str));
        }

        float leaf_emissivity = -1.f;
        node = radiation_block.child("leaf_emissivity");
        if (!node.empty()) {

            const char *leaf_emissivity_str = node.child_value();
            if (!parse_float(leaf_emissivity_str, leaf_emissivity)) {
                helios_runtime_error("ERROR: Value given for 'leaf_emissivity' could not be parsed.");
            } else if (leaf_emissivity < 0 || leaf_emissivity > 1.f) {
                helios_runtime_error("ERROR: Value given for 'leaf_emissivity' must be between 0 and 1.");
            }
        }

        std::string ground_reflectivity_spectrum;
        node = radiation_block.child("ground_reflectivity_spectrum");
        if (!node.empty()) {

            const char *ground_reflectivity_spectrum_str = node.child_value();
            ground_reflectivity_spectrum = trim_whitespace(std::string(ground_reflectivity_spectrum_str));
        }

        float ground_emissivity = -1.f;
        node = radiation_block.child("ground_emissivity");
        if (!node.empty()) {

            const char *ground_emissivity_str = node.child_value();
            if (!parse_float(ground_emissivity_str, ground_emissivity)) {
                helios_runtime_error("ERROR: Value given for 'ground_emissivity' could not be parsed.");
            } else if (ground_emissivity < 0 || ground_emissivity > 1.f) {
                helios_runtime_error("ERROR: Value given for 'ground_emissivity' must be between 0 and 1.");
            }
        }

        // *** Set up simulation *** //

        radiation_ptr->addRadiationBand("PAR", 400, 700);
        radiation_ptr->disableEmission("PAR");
        radiation_ptr->addRadiationBand("NIR", 701, 2500);
        radiation_ptr->disableEmission("NIR");
        radiation_ptr->addRadiationBand("LW");

        uint sun_ID = radiation_ptr->addSunSphereRadiationSource();
        context_ptr->setGlobalData("sun_ID", sun_ID);

        if (direct_ray_count > 0) {
            radiation_ptr->setDirectRayCount("PAR", direct_ray_count);
            radiation_ptr->setDirectRayCount("NIR", direct_ray_count);
        }

        if (diffuse_ray_count > 0) {
            radiation_ptr->setDiffuseRayCount("PAR", diffuse_ray_count);
            radiation_ptr->setDiffuseRayCount("NIR", diffuse_ray_count);
            radiation_ptr->setDiffuseRayCount("LW", diffuse_ray_count);
        }

        if (scattering_depth > 0) {
            radiation_ptr->setScatteringDepth("PAR", scattering_depth);
            radiation_ptr->setScatteringDepth("NIR", scattering_depth);
        } else {
            std::cout << "WARNING: No value given for 'scattering_depth'. All objects will be assumed to be black." << std::endl;
        }

        if (air_turbidity > 0) {
            context_ptr->setGlobalData("air_turbidity", air_turbidity);
        } else if (air_turbidity < 0) { // try calibration
            if (context_ptr->doesTimeseriesVariableExist("net_radiation")) {
                air_turbidity = solarposition_ptr->calibrateTurbidityFromTimeseries("net_radiation");
                if (air_turbidity > 0 && air_turbidity < 1) {
                    context_ptr->setGlobalData("air_turbidity", air_turbidity);
                }
            }
        }
        if (!context_ptr->doesGlobalDataExist("air_turbidity")) {
            std::cout << "WARNING: Air turbidity could not be determined. Setting to a default value of 0.05." << std::endl;
            context_ptr->setGlobalData("air_turbidity", 0.05f);
        }

        if (!solar_direct_spectrum.empty()) {

            if (solar_direct_spectrum == "ASTMG173" || solar_direct_spectrum == "solar_spectrum_direct_ASTMG173") {
                solar_direct_spectrum = "solar_spectrum_direct_ASTMG173";
                context_ptr->loadXML("plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml", true);
            } else if (!context_ptr->doesGlobalDataExist(solar_direct_spectrum.c_str())) {
                helios_runtime_error("ERROR: The specified solar direct spectrum '" + solar_direct_spectrum + "' could not be found in existing global data. Make sure to load the XML file containing this spectral data.");
            }

            radiation_ptr->setSourceSpectrum(sun_ID, solar_direct_spectrum);
        } else {
            std::cout << "WARNING: No value given for 'solar_direct_spectrum'. Using a uniform spectral distribution." << std::endl;
        }

        std::vector<uint> leaf_UUIDs;
        try {
            //            assert( context_ptr->doesGlobalDataExist( "leaf_UUIDs" ) );
            context_ptr->getGlobalData("leaf_UUIDs", leaf_UUIDs);
        } catch (...) {
            std::cout << "WARNING: No leaf UUIDs found." << std::endl;
        }

        std::vector<uint> ground_UUIDs;
        try {
            //            assert( context_ptr->doesGlobalDataExist( "ground_UUIDs" ) );
            context_ptr->getGlobalData("ground_UUIDs", ground_UUIDs);
        } catch (...) {
            std::cout << "WARNING: No ground UUIDs found." << std::endl;
        }

        if (!leaf_UUIDs.empty()) {
            if (!leaf_reflectivity_spectrum.empty()) {
                context_ptr->setPrimitiveData(leaf_UUIDs, "reflectivity_spectrum", leaf_reflectivity_spectrum);
            } else {
                std::cout << "WARNING: No value given for 'leaf_reflectivity_spectrum'. Assuming leaves are black across all shortwave bands." << std::endl;
            }

            if (!leaf_transmissivity_spectrum.empty()) {
                context_ptr->setPrimitiveData(leaf_UUIDs, "transmissivity_spectrum", leaf_transmissivity_spectrum);
            } else {
                std::cout << "WARNING: No value given for 'leaf_transmissivity_spectrum'. Assuming leaves are black across all shortwave bands." << std::endl;
            }

            if (leaf_emissivity >= 0.f && leaf_emissivity <= 1.f) {
                context_ptr->setPrimitiveData(leaf_UUIDs, "emissivity", leaf_emissivity);
            } else {
                std::cout << "WARNING: No value given for 'leaf_emissivity'. Assuming leaves are perfect emitters." << std::endl;
            }
        }

        if (!ground_UUIDs.empty()) {
            if (!ground_reflectivity_spectrum.empty()) {
                if (context_ptr->doesGlobalDataExist(ground_reflectivity_spectrum.c_str())) {
                    context_ptr->setPrimitiveData(ground_UUIDs, "reflectivity_spectrum", ground_reflectivity_spectrum);
                } else {
                    std::cout << "WARNING: The specified ground reflectivity spectrum '" + ground_reflectivity_spectrum + "' could not be found in existing global data. Assuming the ground is black across all shortwave bands." << std::endl;
                }
            } else {
                std::cout << "WARNING: No value given for 'ground_reflectivity_spectrum'. Assuming the ground is black across all shortwave bands." << std::endl;
            }

            if (ground_emissivity >= 0.f && ground_emissivity <= 1.f) {
                context_ptr->setPrimitiveData(ground_UUIDs, "emissivity", ground_emissivity);
            } else {
                std::cout << "WARNING: No value given for 'ground_emissivity'. Assuming ground is a perfect emitter." << std::endl;
            }
        }
        radiation_ptr->updateGeometry();
    }

    if (radiation_block_count == 0) {
        context_ptr->setGlobalData("radiation_enabled", false);
    } else {
        context_ptr->setGlobalData("radiation_enabled", true);
    }
}
