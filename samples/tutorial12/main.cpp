// *** Step 1: Loading Plugins *** //

#include "CanopyGenerator.h"
#include "RadiationModel.h"
#include "EnergyBalanceModel.h"
#include "PhotosynthesisModel.h"
#include "StomatalConductanceModel.h"
#include "SolarPosition.h"
#include "BoundaryLayerConductanceModel.h"
#include "Visualizer.h"

using namespace helios;

int main() {

    // *** Step 2: Defining the Context *** //

    Context context;

    //Set Date
    Date date(20, 6, 2024);
    context.setDate(date);

    //Define location and timezone
    float latitude = 38.535694;
    float longitude = 121.776360;
    int UTC = 7;

    //Atmospheric Constants
    float pressure = 101300; //Atmospheric Pressure, Pa
    float turbidity = 0.05;

    // *** Step 3: Creating Our Model Geometry *** //

    // Build the Canopy //
    CanopyGenerator canopygenerator(&context);

    std::vector<uint> leaf_UUIDs;

    float crown_radius = 1.5; //meters
    float crown_height_radius = 3.; //meters

    float spacing_row = 2 * crown_radius;
    float spacing_plant = 2 * crown_radius;

    int row_number = 3; //number of rows
    int plant_number = 3; //number of plants per row

    float l_leaf = 0.075; //leaf length
    float w_leaf = 0.05; //leaf width

    //Leaf and canopy parameters
    SphericalCrownsCanopyParameters params;
    params.crown_radius = make_vec3(crown_radius, crown_radius, crown_height_radius);
    params.plant_spacing = make_vec2(spacing_row, spacing_plant);
    params.plant_count = make_int2(row_number, plant_number);
    params.leaf_size = make_vec2(l_leaf, w_leaf);
    params.leaf_subdivisions = make_int2(3, 3);
    params.leaf_area_density = 1.f;

    //Create the crowns
    canopygenerator.buildCanopy(params);

    //Get the UUID numbers for all of the leaves
    leaf_UUIDs = canopygenerator.getLeafUUIDs();

    //ground parameters
    float x_ground = spacing_row*row_number;
    float y_ground = spacing_plant*plant_number;
    vec2 size_ground = make_vec2(x_ground, y_ground);

    //Make the Ground
    canopygenerator.buildGround(make_vec3(0,0,0), size_ground, make_int2(row_number,plant_number), make_int2(3,3), "plugins/canopygenerator/textures/dirt.jpg");

    //Ground UUIDs
    std::vector<uint> ground_UUIDs = canopygenerator.getGroundUUIDs();
    //UUIDs of all primitives in the context
    std::vector<uint> all_UUIDs = context.getAllUUIDs();

    // *** Step 4: Setting Up the Solar Position Model *** //

    SolarPosition solar_position(UTC, latitude, longitude, &context);

    // *** Step 5: Setting Up the Radiation Model *** //
    RadiationModel radiation(&context);

    uint SunSource = radiation.addSunSphereRadiationSource();

    radiation.addRadiationBand("PAR");
    radiation.disableEmission("PAR");
    radiation.setScatteringDepth("PAR", 3);

    radiation.addRadiationBand("NIR");
    radiation.disableEmission("NIR");
    radiation.setScatteringDepth("NIR", 3);

    radiation.addRadiationBand("LW");

    radiation.enforcePeriodicBoundary("xy");

    //Set leaf radiative properties
    context.setPrimitiveData(leaf_UUIDs, "reflectivity_PAR", 0.10f);
    context.setPrimitiveData(leaf_UUIDs, "transmissivity_PAR", 0.05f);
    context.setPrimitiveData(leaf_UUIDs, "reflectivity_NIR", 0.45f);
    context.setPrimitiveData(leaf_UUIDs, "transmissivity_NIR", 0.4f);

    context.setPrimitiveData(ground_UUIDs, "reflectivity_PAR", 0.15f);
    context.setPrimitiveData(ground_UUIDs, "reflectivity_NIR", 0.4f);

    //Make sure that the ground is only able to intercept radiation from the top
    context.setPrimitiveData(ground_UUIDs, "twosided_flag", uint(0));

    radiation.updateGeometry();

    //*** Step 6: Setting Up the Energy Balance Model ***//

    EnergyBalanceModel energybalance(&context);

    energybalance.addRadiationBand("PAR");
    energybalance.addRadiationBand("NIR");
    energybalance.addRadiationBand("LW");

    BLConductanceModel boundarylayerconductance(&context);

    boundarylayerconductance.setBoundaryLayerModel( ground_UUIDs, "Ground" );
    boundarylayerconductance.setBoundaryLayerModel( leaf_UUIDs, "Pohlhausen" );

    //*** Step 7: Setting Up the Stomatal Conductance Model ***//

    StomatalConductanceModel stomatalconductance(&context);

    BMFcoefficients bmfc;
    stomatalconductance.setModelCoefficients(bmfc);

    //*** Step 8: Setting Up Photosynthesis Model ***//

    PhotosynthesisModel photosynthesis(&context);

    FarquharModelCoefficients photoparams;
    photosynthesis.setModelCoefficients(photoparams);
    photosynthesis.setModelType_Farquhar();

    // *** Step 9: Reading in Our Timeseries Data *** //

    context.loadXML("../xml/6_20_2024_CIMIS.xml");

    for( int hour = 7; hour<18; hour++){

        Time time(hour, 0, 0);
        context.setTime(time);

        // this will query these timseries variables based on the date and time set in the Context
        float air_temperature = context.queryTimeseriesData("air_temperature"); // degrees C
        float air_humidity = context.queryTimeseriesData("humidity"); // Percent
        float wind_speed = context.queryTimeseriesData("wind_speed"); // m/s

        // update our primitive data values on each timestep based on timeseries data
        context.setPrimitiveData(all_UUIDs, "air_temperature", air_temperature);
        context.setPrimitiveData(all_UUIDs, "air_humidity", air_humidity);
        context.setPrimitiveData(all_UUIDs, "wind_speed", wind_speed);

        float LW = solar_position.getAmbientLongwaveFlux(air_temperature, air_humidity);
        float PAR = solar_position.getSolarFluxPAR(pressure, air_temperature, air_humidity, turbidity);
        float NIR = solar_position.getSolarFluxNIR(pressure, air_temperature, air_humidity, turbidity);
        float f_diff = solar_position.getDiffuseFraction(pressure, air_temperature, air_humidity, turbidity);

        radiation.setSourceFlux(SunSource, "NIR", NIR * (1.f - f_diff));
        radiation.setDiffuseRadiationFlux("NIR", NIR * f_diff);
        radiation.setSourceFlux(SunSource, "PAR", PAR * (1.f - f_diff));
        radiation.setDiffuseRadiationFlux("PAR", PAR * f_diff);
        radiation.setDiffuseRadiationFlux("LW", LW);

        radiation.setSourcePosition( SunSource, solar_position.getSunDirectionVector() );

        boundarylayerconductance.run();

        // *** Step 10: Running the Model *** //

        radiation.runBand({"PAR","NIR","LW"});

        stomatalconductance.run(leaf_UUIDs);
        energybalance.run();

        //Run the longwave band, stomatal conductance plugin, and energy balance plugin again to update primitive temperature values
        radiation.runBand("LW");
        stomatalconductance.run(leaf_UUIDs);
        energybalance.run();

        photosynthesis.run(leaf_UUIDs); // always run this last, since nothing depends on it

        // *** Step 11: Calculating WUE ***//

        float A_canopy = 0;
        float E_canopy = 0;
        for (uint UUID : leaf_UUIDs) {
            float E, A, WUE;
            context.getPrimitiveData(UUID, "latent_flux", E);
            context.getPrimitiveData(UUID, "net_photosynthesis", A);
            E_canopy += E / 44000 * 1000; // mmol H2O / m^2 / sec
            A_canopy += A;  //umol CO2 / m^2 / sec

            WUE = A / (E / 44000 * 1000); //umol CO2/mmol H2O
            context.setPrimitiveData(UUID, "WUE", WUE);

        }
        float WUE_canopy = A_canopy / E_canopy; //umol CO2/mmol H2O

        std::cout << "WUE of the canopy = " << WUE_canopy << " umol CO2/mmol H2O" << std::endl;

        // *** Step 12: Visualization *** //
        // This will open the visualizer window for each time step. Close it to proceed to the next time step.
        Visualizer visualizer(1000);
        visualizer.buildContextGeometry(&context);
        visualizer.colorContextPrimitivesByData( "WUE" );
        visualizer.setColorbarTitle("WUE (umol CO2/mmol H2O)");;
        visualizer.setColorbarRange(0.f,15.f);
        visualizer.setColorbarPosition(make_vec3(0.75, 0.9, 0) );
        char time_string[6];
        sprintf(time_string, "%02d:%02d", time.hour, time.minute);
        visualizer.addTextboxByCenter( time_string, make_vec3(0.5,0.9,0), nullrotation, RGB::black, 16, "Arial", Visualizer::COORDINATES_WINDOW_NORMALIZED );
        visualizer.plotInteractive(); // !!! Close the window to advance to the next time step

    }

    return 0;
}