#include "RadiationModel.h"
#include "CanopyGenerator.h"
#include "VoxelIntersection.h"

using namespace helios; //note that we are using the helios namespace so we can omit 'helios::' before names
     
int main(){
  
  // !!!!!!!!! Inputs for this example study !!!!!!!!! //

  vec3 sun_direction(1,0,1);                  //Cartesian unit vector pointing in the direction of (toward) the sun
  sun_direction.normalize();

  std::string leaf_angle_dist = "spherical";              //name of leaf angle distribution for the canopy - (spherical, uniform, planophile, erectophile, plagiophile, extremophile)

  float LAI = 1.5f;                                       //one-sided leaf area index of the canopy

  vec2 canopy_extent(3,3);                        //dimension of the canopy in the x- and y-directions (horizontal)
  float canopy_height = 1.f;                              //vertical dimension of the canopy

  // *** 1. Model geometry creation *** //

  Context context;                                        //declare the Context class

  CanopyGenerator cgen(&context);                         //declare the Canopy Generator class and pass it the Context so it can add geometry

  HomogeneousCanopyParameters params;                     //structure containing parameters for homogeneous canopy
  params.buffer = "none";                                 //no buffer on the canopy edges - we will slice along border
  params.leaf_angle_distribution = leaf_angle_dist;       //set the leaf angle distribution based on the variable we set above
  params.canopy_extent = canopy_extent;                   //set the lateral canopy extent based on the variable we set above
  params.canopy_height = canopy_height;                   //set the canopy height based on the variable we set above
  params.leaf_area_index = LAI;                           //set the canopy LAI based on the variable we set above

  params.leaf_subdivisions = make_int2(5,5);              //set the number of subdivisions per leaf to be 5x5=25 primitives

  cgen.buildCanopy(params);                               //build the homogeneous canopy

  std::vector<uint> UUIDs_leaves = cgen.getLeafUUIDs();   //get UUIDs for all leaves in the canopy

  // *** 2. Slicing and cropping primitives on the boundaries *** //

  // slice any primitives that lie on the canopy boundaries (imagine taking a knife and perfectly cutting along the edges of the canopy)
  VoxelIntersection vslice(&context);

  // define variables that give the center (x,y,z) coordinate of the canopy and the overall canopy dimensions
  vec3 slice_box_center(0,0,0.5f*canopy_height);
  vec3 slice_box_size(canopy_extent.x,canopy_extent.y,canopy_height);

  // do the slicing
  vslice.slicePrimitivesUsingGrid( UUIDs_leaves, slice_box_center, slice_box_size, make_int3(1,1,1) );

  // delete any leaf slices that fall outside of the slicing volume
  context.cropDomain( make_vec2(-0.5f*canopy_extent.x,0.5f*canopy_extent.x), make_vec2(-0.5f*canopy_extent.y,0.5f*canopy_extent.y), make_vec2(0,canopy_height) );

  // our UUID vector now contains some primitives that have been deleted. We can just get all the primitives currently in the Context and store them in a vector
  UUIDs_leaves = context.getAllUUIDs();

  //make a ground
  std::vector<uint> UUIDs_ground = context.addTile(make_vec3(0, 0, 0), canopy_extent, nullrotation,make_int2(10, 10));

  // *** 3. Radiation model set-up *** //

  RadiationModel radiation(&context);                      //declare and initialize the radiation model class

  // add a sun source. We'll assume collimated radiation since that is what is assumed in Beer's law
  uint sourceID = radiation.addCollimatedRadiationSource( sun_direction );

  // set up the PAR band. We'll use separate direct and diffuse bands to keep them separate for post-processing (normally you would combine them)
  radiation.addRadiationBand("PAR");
  radiation.disableEmission("PAR");
  radiation.setSourceFlux(sourceID, "PAR", 1.f);  //set a flux of 1.0 W/m^2 to simplify calculations
  radiation.setDiffuseRadiationFlux("PAR", 0.f);      //no diffuse radiation this band

  radiation.enforcePeriodicBoundary("xy");      //use periodic boundary conditions in the horizontal to simulate an infinite canopy

  context.setPrimitiveData(UUIDs_ground, "twosided_flag",uint(0)); //only want ground to intercept/emit radiation from the top

  radiation.updateGeometry();                            //update the geometry in the radiation model

  // 4. Run model and process results //

  //Run the radiation model calculations
  radiation.runBand("PAR");

  // 4a. Calculate G(theta)
  float Gtheta = 0;
  float area_total = 0;
  for( auto UUID : UUIDs_leaves ){
    vec3 normal = context.getPrimitiveNormal(UUID);
    float area = context.getPrimitiveArea(UUID);
    Gtheta += std::abs( sun_direction*normal )*area;
    area_total += area;
  }
  Gtheta = Gtheta/area_total;  //normalize

  std::cout << "G(theta) = " << Gtheta << std::endl;

  // 4b. Calculate radiation flux absorbed by the canopy on a ground area basis - this will end up just being the area-weighted average PAR flux multiplied by LAI.

  float PAR_abs_dir;
  context.calculatePrimitiveDataAreaWeightedMean( UUIDs_leaves, "radiation_flux_PAR", PAR_abs_dir ); //recall that the output primitive data from the radiation model has the form "radiation_flux_[*band_name*]"
  PAR_abs_dir = PAR_abs_dir*LAI; //converts between leaf area basis to ground area basis

  // 4c. Calculate the theoretical absorbed PAR flux using Beer's law

  float theta_s = cart2sphere(sun_direction).zenith;  //calculate the solar zenith angle

  float R0 = cos(theta_s); //PAR flux on horizontal surface
  float intercepted_theoretical_direct = R0*(1.f-exp(-Gtheta*LAI/cos(theta_s)));  //Beer's law

  std::cout << "Calculated interception: " << PAR_abs_dir << std::endl;
  std::cout << "Theoretical interception: " << intercepted_theoretical_direct << std::endl;
  std::cout << "Error of interception: " << std::abs(PAR_abs_dir-intercepted_theoretical_direct)/intercepted_theoretical_direct*100.f << " %" << std::endl;

  // 4d. Calculate the sunlit leaf area fraction from the simulation

  float sunlit_area = 0;
  float total_area = 0;
  for( auto UUID : UUIDs_leaves ){ //looping over all leaf elements

    vec3 normal = context.getPrimitiveNormal(UUID);

    float PARmax = std::abs( normal*sun_direction );  //this is the PAR flux of a leaf with the same normal that is fully sunlit

    float PAR;
    context.getPrimitiveData( UUID, "radiation_flux_PAR", PAR ); //get this leaf's PAR flux

    float fsun_leaf = PAR/PARmax;  //PAR flux as a fraction of the fully sunlit flux

    float area = context.getPrimitiveArea(UUID);

    if( fsun_leaf>0.5 ){ //if fsun is greater than 0.5, we'll call this leaf "sunlit"
      sunlit_area += area;
    }
    total_area += area;

  }

  float fsun = sunlit_area/total_area;

  // 4e. Calculate the theoretical sunlit area fraction

  float fsun_theoretical = cos(theta_s)/(Gtheta*LAI)*(1-exp(-Gtheta*LAI/cos(theta_s)));

  std::cout << "Calculated sunlit fraction: " << fsun << std::endl;
  std::cout << "Theoretical sunlit fraction: " << fsun_theoretical << std::endl;
  std::cout << "Error of sunlit fraction: " << std::abs(fsun-fsun_theoretical)/fsun_theoretical*100.f << " %" << std::endl;

  return 0;

}
