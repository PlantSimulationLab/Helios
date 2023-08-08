#include "Context.h"
#include "Visualizer.h"
#include "RadiationModel.h"
#include "CanopyGenerator.h"
#include "PhotosynthesisModel.h"
using namespace helios;

void buildBeanPlot( const BeanParameters &params, float germination_prob, CanopyGenerator& cgen, Context *context ){
    vec2 canopy_extent( params.plant_spacing*float(params.plant_count.x), params.row_spacing*float(params.plant_count.y) );
    uint prim_count = 0;
    for (int j = 0; j < params.plant_count.y; j++) {
        for (int i = 0; i < params.plant_count.x; i++) {

            if (context->randu() > germination_prob) {
                continue;
            }
            vec3 center = params.canopy_origin + make_vec3(-0.5f * canopy_extent.x + (float(i) + 0.5f) * params.plant_spacing, -0.5f * canopy_extent.y + (float(j) + 0.5f) * params.row_spacing, 0);
            if (params.canopy_rotation != 0) {
                center = rotatePointAboutLine(center, params.canopy_origin, make_vec3(0, 0, 1), params.canopy_rotation);
            }

            uint plant_ID = cgen.bean(params, center);
            std::cout << "plant_ID: " << plant_ID << std::endl;
            std::vector<uint> UUIDs_all = cgen.getAllUUIDs(plant_ID);
            prim_count += UUIDs_all.size();
            context->setPrimitiveData(UUIDs_all, "plant_ID", plant_ID + 1);
        }
    }
}


int main(void){

    // load spectra data;

    Context context;
    context.loadXML("plugins/radiation/spectral_data/light_spectral_library.xml" );
//    context.loadXML("plugins/radiation/spectral_data/camera_spectral_library.xml" ); //Uncalibrated camera spectral response
    context.loadXML("plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml" );
    context.loadXML("plugins/radiation/spectral_data/surface_spectral_library.xml" );
    context.loadXML("plugins/radiation/spectral_data/GEMINI_LV1/Weeds_stem_disease_spectroradiometer.xml" );

    // Calibrated camera spectral response
    context.loadXML("plugins/radiation/spectral_data/external_data/calibrated_sun_NikonD700_spectral_response_red.xml" );
    context.loadXML("plugins/radiation/spectral_data/external_data/calibrated_sun_NikonD700_spectral_response_green.xml" );
    context.loadXML("plugins/radiation/spectral_data/external_data/calibrated_sun_NikonD700_spectral_response_blue.xml" );

    // Add ground soil
//    std::vector<uint> UUIDg = context.loadOBJ("../obj/dirt_rocks.obj", make_vec3(0, 0.75, 0), 0, nullrotation, RGB::brown);
//    context.setPrimitiveData( UUIDg, "twosided_flag", uint(0) );
//    context.setPrimitiveData( UUIDg, "reflectivity_spectrum", "soil_reflectivity_0003" );
//    context.setPrimitiveData( UUIDg, "transmissivity_spectrum", "");

    // Set spectra label for objects
    std::string leaf_reflectivity_spectrum = "bean_leaf_reflectivity_0001";
    std::string leaf_transmissivity_spectrum = "bean_leaf_transmissivity_0001";
    std::string stem_reflectivity_spectrum = "Stem_Cowpea_Reflectance";
    std::string ground_reflectivity_spectrum = "soil_reflectivity_0003";
    std::string solar_intensity_spectrum = "solar_spectrum_ASTMG173";
    std::string LED_spectrum = "CREE_XLamp_XHP70p2_6500K";

    // Divide stem reflectivity spectrum by 100 to get reflectivity in [0,1]
    std::vector<vec2> stemreflectivities;
    context.getGlobalData(stem_reflectivity_spectrum.c_str(),stemreflectivities);
    for (uint i=0; i<stemreflectivities.size(); ++i){
        stemreflectivities.at(i).y=stemreflectivities.at(i).y/100;
    }
    context.setGlobalData(stem_reflectivity_spectrum.c_str(),HELIOS_TYPE_VEC2,stemreflectivities.size(),&stemreflectivities[0]);

    // set PAR
    std::vector<helios::vec2> PARcamresponse(301);
    for (int i=0; i<PARcamresponse.size(); i++){
        PARcamresponse.at(i).x=400+i;
        PARcamresponse.at(i).y=1;
    }
    context.setGlobalData("PAR",HELIOS_TYPE_VEC2,PARcamresponse.size(),&PARcamresponse[0]);  // camera response is 1

    // Sey spectra label for light sources and objects

    std::vector<std::string> sourcelabels = {solar_intensity_spectrum};

    // Se wavelength range for radiation model
    vec2 wavelengthrange = make_vec2(300,2500);

    // Set bean geometry parameters
    CanopyGenerator cgen(&context);

    BeanParameters params;
    params.plant_count = make_int2(20,2);
    params.stem_length = 0.08;
    params.leaf_subdivisions = make_int2(18,18);
    params.row_spacing = 0.6;
    params.plant_spacing = 0.15;
    params.leaf_length = 0.08;
    params.pod_length = 0;
    params.canopy_rotation = 0.5*M_PI;

    // Build bean plots
    float germination_prob = 0.9;
    bool build_plants = true;
    if( build_plants ) {
        buildBeanPlot( params, germination_prob, cgen, &context );
    }

    // Get UUIDs for bean objects
    std::vector<uint> leaf_UUIDs = cgen.getLeafUUIDs();
    std::vector<uint> branch_UUIDs = cgen.getBranchUUIDs();
    std::vector<uint> trunk_UUIDs = cgen.getTrunkUUIDs();
    std::vector<uint> fruit_UUIDs = cgen.getFruitUUIDs();

    //Set object label for primitives
    std::vector<uint> UUID_all = context.getAllUUIDs();
    context.setPrimitiveData(UUID_all,"object_label",float(0));
    context.setPrimitiveData(leaf_UUIDs,"object_label",float(1));
    context.setPrimitiveData(branch_UUIDs,"object_label",float(2));
    context.setPrimitiveData(trunk_UUIDs,"object_label",float(2));

    // Set spectra label for objects
    context.setPrimitiveData( leaf_UUIDs, "reflectivity_spectrum", leaf_reflectivity_spectrum);
    context.setPrimitiveData( leaf_UUIDs, "transmissivity_spectrum", leaf_transmissivity_spectrum);
    context.setPrimitiveData( trunk_UUIDs, "reflectivity_spectrum", stem_reflectivity_spectrum);
    context.setPrimitiveData( trunk_UUIDs, "transmissivity_spectrum", "");
    context.setPrimitiveData( branch_UUIDs, "reflectivity_spectrum", stem_reflectivity_spectrum);
    context.setPrimitiveData( branch_UUIDs, "transmissivity_spectrum", "");
    context.setPrimitiveData( leaf_UUIDs, "leaf", 1);
    // Set camera main label for one case
    std::string cameralabel = "Bean";

    // Set radiation camera parameters
    vec3 camera_position = make_vec3(0, 0, 1.6);
    vec3 camera_lookat = make_vec3(0, 0, 1);
    CameraProperties cameraproperties;
    cameraproperties.camera_resolution=make_int2(500, 1000);
    cameraproperties.focal_plane_distance = 1.35f;
    cameraproperties.lens_diameter = 0.02f;
    cameraproperties.sensor_size= make_vec2(0.05, 0.1);// set single value
    cameraproperties.HFOV=23.f;

    uint plantnumber = cgen.getPlantCount();

    // Set radiation model
    RadiationModel radiation(&context);

    // Set radiation band labels
    std::vector<std::string> bandlabels = {"red","green","blue","PAR"};
//            {"red","green","blue","PAR"};
    std::vector<std::string> bandRGBlabels = {"red","green","blue"};
    // Add radiation camera

    // Add radiation sources
    radiation.addSunSphereRadiationSource(make_SphericalCoord(deg2rad(90-40), deg2rad(40)));

    // Set radiation camera response labels
    std::vector<std::string> cameralabelscal = {"calibrated_sun_NikonD700_spectral_response_red",
                                                "calibrated_sun_NikonD700_spectral_response_green",
                                                "calibrated_sun_NikonD700_spectral_response_blue",
                                                "PAR"};

    // Load PROSPECT leaf test data on total reflectance and transmittance
    LeafOptics Leaf(&context);

    LeafOpticsProperties leafproperties;
    leafproperties.chlorophyllcontent= 30.0;
    leafproperties.carotenoidcontent = 7.0;
    leafproperties.anthocyancontent = 1;
    leafproperties.watermass= 0.015;
    leafproperties.drymass = 0.009;

    FarquharModelCoefficients fCoefficients;
    PhotosynthesisModel photosys(&context);
    float avc = 0.54;
    float bvc = 55.28;
    float e = 2.71828182845904523536;
    float ajv = 0.89;
    float bjv = 1.01;

    // Set various reflectivity and transmissivity spectra on leaf based on random chlorophyll contents
    std::vector<std::vector<uint>> leaf_UUIDs_sub(plantnumber);
    for (uint iplant = 0; iplant < plantnumber; iplant++) {
        std::vector<std::vector<uint>> ileaf_UUIDs = cgen.getLeafUUIDs(iplant);
        for (uint ileaf = 0; ileaf < ileaf_UUIDs.size(); ileaf++) {

            float chl_content = 25 + 20*context.randu(); // Random chlorophyll content between 25 and 45
            leafproperties.chlorophyllcontent = chl_content;
            leafproperties.carotenoidcontent = chl_content/4.5f;
            std::vector<uint> UUIDs_ileaf = ileaf_UUIDs.at(ileaf);

            // Set net photosynthesis coefficients based on chlorophyll content
            float Vcmax25 = avc*chl_content+bvc;
            fCoefficients.Vcmax = Vcmax25;
            fCoefficients.Jmax = std::pow(e, ajv * std::log(Vcmax25)+bjv);
            fCoefficients.Rd = 0.01f*Vcmax25;

            photosys.setModelCoefficients( fCoefficients, UUIDs_ileaf);
            //run PROSPECT leaf model for each leaf
            Leaf.run(UUIDs_ileaf ,leafproperties, std::to_string(chl_content));
        }

    }

    // Get image with variate leaf chlorophyll content
    radiation.addRadiationCamera(cameralabel +"_PROSPECT", bandlabels, camera_position, camera_lookat, cameraproperties,10);
    radiation.runRadiationImaging(cameralabel +"_PROSPECT", sourcelabels, bandlabels, cameralabelscal,wavelengthrange,1.4);
    radiation.writeCameraImage(cameralabel +"_PROSPECT", bandRGBlabels, "chlorophyll");


    // Write plant segmentation image to file
    radiation.writeBasicLabel(cameralabel + "_PROSPECT", cameralabel + "_Plants.txt", "plant_ID", HELIOS_TYPE_UINT);


    // Write chlorophyll content image to file
    radiation.writeBasicLabel(cameralabel + "_PROSPECT", cameralabel + "_Chlorophyll.txt", "chlorophyll", HELIOS_TYPE_UINT);

    photosys.run(leaf_UUIDs);
    radiation.writeBasicLabel(cameralabel + "_PROSPECT", cameralabel + "_Photosynthesis.txt", "net_photosynthesis", HELIOS_TYPE_UINT);

    // Write distorted image
//    std::vector<double> distortCoeffs = {-0.354,0.173, 0, 0};
//    vec2 focallengthxy = make_vec2(700,700);

//    CameraCalibration cameracalibration(&context);
//    cameracalibration.distortImage(cameralabel + "_PROSPECT",bandlabels,focallengthxy,distortCoeffs,cameraproperties.camera_resolution);
//    radiation.writeCameraImage(cameralabel + "_PROSPECT", bandRGBlabels,"_distorted");

    // Visualize the geometry
    Visualizer visualizer(400);
    visualizer.buildContextGeometry(&context);
    visualizer.colorContextPrimitivesByData("radiation_flux_PAR");
    visualizer.plotInteractive();

    return 0;
}
