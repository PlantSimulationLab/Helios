#include "InitializeEnergyBalance/InitializeEnergyBalance.h"

using namespace helios;

void InitializeEnergyBalance(const std::string &xml_input_file, BLConductanceModel *boundarylayerconductancemodel, EnergyBalanceModel *energybalancemodel, helios::Context *context_ptr) {

    pugi::xml_document xmldoc;

    std::string xml_error_string;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }

    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;

    // *** Parsing of general inputs *** //

    int energybalance_block_count = 0;
    for (pugi::xml_node energybalance_block = helios.child("energybalance"); energybalance_block; energybalance_block = energybalance_block.next_sibling("energybalance")) {
        energybalance_block_count++;

        if (energybalance_block_count > 1) {
            std::cout << "WARNING: Only one 'energybalance' block is allowed in the input file. Skipping any others..." << std::endl;
            break;
        }

//        int direct_ray_count = 100;
//        node = energybalance_block.child("direct_ray_count");
//        if (node.empty()) {
//            direct_ray_count = 0;
//        } else {
//
//            const char *direct_ray_count_str = node.child_value();
//            if (!parse_int(direct_ray_count_str, direct_ray_count)) {
//                helios_runtime_error("ERROR: Value given for 'direct_ray_count' could not be parsed.");
//            } else if (direct_ray_count < 0) {
//                helios_runtime_error("ERROR: Value given for 'direct_ray_count' must be greater than or equal to 0.");
//            }
//
//        }

    }

    energybalancemodel->addRadiationBand( {"PAR", "NIR", "LW"} );

    std::vector<uint> ground_UUIDs;
    assert( context_ptr->doesGlobalDataExist( "ground_UUIDs" ) );
    context_ptr->getGlobalData( "ground_UUIDs", ground_UUIDs );

    boundarylayerconductancemodel->setBoundaryLayerModel( ground_UUIDs, "Ground" );

    if( energybalance_block_count==0 ){
        context_ptr->setGlobalData( "energybalance_enabled", false );
    }else{
        context_ptr->setGlobalData( "energybalance_enabled", true );
    }


}