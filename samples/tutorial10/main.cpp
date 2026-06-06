#include "RadiationModel.h"
#include "Visualizer.h"

using namespace helios; // note that we are using the helios namespace so we can omit 'helios::' before names

int main() {

    // *** 1. Model geometry creation *** //

    float row_spacing = 5; // spacing between tree rows
    float plant_spacing = 4; // spacing between trees within a row

    Context context; // Declare the "Context" class

    // Load 3D model "Tree.ply" with a base position of (0,0,0) and scaled to a height of 5
    std::vector<uint> UUIDs_tree = context.loadPLY("../../../PLY/Tree.ply", make_vec3(0, 0, 0), 5, nullrotation, RGB::black);

    // Add a ground surface with a center position of (0,0,0) and size of row_spacing x plant_spacing
    std::vector<uint> UUIDs_ground = context.addTile(make_vec3(0, 0, 0), make_vec2(row_spacing, plant_spacing), nullrotation, make_int2(500, 500));

    // *** 2. Radiation model set-up *** //

    // Declare and initialize the radiation model class
    RadiationModel radiation(&context);

    // Add a sun radiation source with elevation angle of 60 degrees and an azimuth of 45 degrees (note that we need to convert to radians)
    uint sourceID = radiation.addSunSphereRadiationSource(make_SphericalCoord(deg2rad(60), deg2rad(45)));

    // Add a shortwave radiation band called "PAR" (you can call it anything you want, just be consistent)
    radiation.addRadiationBand("PAR");
    radiation.disableEmission("PAR"); // turn off emission, no emission of primitives in solar bands
    radiation.setSourceFlux(sourceID, "PAR", 500.f); // set solar flux perpendicular to sun direction of 500 W/m^2
    radiation.setDiffuseRadiationFlux("PAR", 50.f); // set diffuse (ambient) solar radiation flux of 50 W/m^2

    // We only want the ground to absorb radiation from the top. If we left the default "twosided_flag=1", our ground would absorb diffuse radiation from below
    context.setPrimitiveData(UUIDs_ground, "twosided_flag", uint(0));

    radiation.enforcePeriodicBoundary("xy"); // Use periodic lateral boundaries so we have repeating trees

    radiation.updateGeometry(); // tell the radiation model to load all the geometry in the Context

    // *** 3. Run the model and calculate PAR interception *** //

    radiation.runBand("PAR"); // run the radiation calculations for this band

    // Calculate PAR interception
    float PAR_tree;
    context.calculatePrimitiveDataAreaWeightedSum(UUIDs_tree, "radiation_flux_PAR", PAR_tree); // sum up absorbed PAR flux for each tree primitive and weight by primitive surface area.
    float PAR_ground;
    context.calculatePrimitiveDataAreaWeightedSum(UUIDs_ground, "radiation_flux_PAR", PAR_ground); // sum up absorbed PAR flux for each ground primitive and weight by primitive surface area.

    float fPAR = PAR_tree / (PAR_tree + PAR_ground);

    std::cout << "Fraction of intercepted PAR is " << fPAR << std::endl;

    return 0;
}
