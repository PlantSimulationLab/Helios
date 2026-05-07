#include "RadiationModel.h"
#include "PlantArchitecture.h"
#include "PhotosynthesisModel.h"
#include "LeafOptics.h"
#include "Visualizer.h"

using namespace helios;

int main() {

    // *** 1. Build the plant with a nitrogen gradient *** //

    Context context;

    std::vector<uint> UUIDs_ground = context.addTile( nullorigin, make_vec2(20, 20), nullrotation, make_int2(200, 200) );

    PlantArchitecture plantarch(&context);
    plantarch.enableNitrogenModel();

    plantarch.loadPlantModelFromLibrary("soybean");
    std::vector<uint> plantIDs = plantarch.buildPlantCanopyFromLibrary( nullorigin, make_vec2(0.3,0.7), make_int2(6,2), 0);

    plantarch.addPlantNitrogen(plantIDs, 1.0f);  // 1.0 g N applied at planting
    plantarch.advanceTime(30.f);                // grow 30 days

    std::vector<uint> UUIDs_leaves = plantarch.getAllLeafUUIDs();

    // *** 2. Author leaf biochemistry from per-leaf nitrogen *** //

    LeafOptics leafoptics(&context);

    LeafOpticsProperties_Nauto N_props;
    N_props.f_photosynthetic     = 0.50f;
    N_props.N_to_Cab_coefficient = 0.40f;
    N_props.Car_to_Cab_ratio     = 0.25f;
    N_props.numberlayers         = 1.5f;
    N_props.watermass            = 0.009f;
    N_props.drymass              = 0.012f;
    leafoptics.run(UUIDs_leaves, N_props);

    // *** 3. Configure the photosynthesis model *** //

    PhotosynthesisModel photosynthesismodel(&context);
    photosynthesismodel.optionalOutputPrimitiveData("electron_transport_ratio");

    // *** 4. Set up the radiation model *** //

    RadiationModel radiationmodel(&context);

    uint sun_ID = radiationmodel.addCollimatedRadiationSource();   // vertical sun by default
    radiationmodel.setSourceSpectrum(sun_ID, "solar_spectrum_direct_ASTMG173");

    radiationmodel.addRadiationBand("PAR",        400.f, 700.f);
    radiationmodel.addRadiationBand("SIF_red",    680.f, 700.f);
    radiationmodel.addRadiationBand("SIF_farred", 730.f, 760.f);

    radiationmodel.disableEmission("PAR");
    radiationmodel.setScatteringDepth("PAR",        3);
    radiationmodel.setScatteringDepth("SIF_red",    3);
    radiationmodel.setScatteringDepth("SIF_farred", 3);

    // *** 5. Add the SIF camera *** //

    SIFCameraProperties cam_props;
    cam_props.camera_resolution         = make_int2(1024, 1024);
    cam_props.HFOV                      = 30.f;
    cam_props.lens_diameter             = 0.f;       // pinhole
    cam_props.exposure                  = "auto";    // per-band 95th-percentile auto-exposure
    cam_props.excitation_bin_width_nm   = 10.f;      // 35 internal excitation bands

    radiationmodel.addSIFCamera(
        "SIF",
        {"SIF_red", "SIF_farred"},
        make_vec3(0.5f, 0.f, 5.f),     // camera position
        make_vec3(0.f, 0.f, 0.f),       // lookat point
        cam_props,
        100                             // antialiasing samples per pixel
    );

    // *** 6. Run the simulation in the right order *** //

    radiationmodel.updateGeometry();

    radiationmodel.runBand("PAR");

    photosynthesismodel.run(UUIDs_leaves);

    radiationmodel.runBand( std::vector<std::string>{"SIF_red", "SIF_farred"} );

    // *** 7. Write the SIF image and visualize the nitrogen gradient *** //

    radiationmodel.writeCameraImage("SIF", {"SIF_red"}, "red", "../output/");
    radiationmodel.writeCameraImage("SIF", {"SIF_farred"}, "farred", "../output/");

    Visualizer visualizer(1000);
    visualizer.buildContextGeometry(&context);
    visualizer.colorContextPrimitivesByObjectData("leaf_nitrogen_gN_m2");
    visualizer.plotInteractive();

    return 0;
}
