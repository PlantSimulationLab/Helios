/**
 * The goal of this sample project is to demonstrate how to build vine stocks based on parameters defined in a configuration file.
 * First we build a fully configured canopy (multiple straight rows of plants), then we build an individual vine stock, and finally we use the flexibility of
 * individual plant generation to build a circle arc of vine stocks.
 */

#include "CanopyGenerator.h"
#include "Visualizer.h"

using namespace helios;

#define CONFIG_DIR "../config/"
#define VINE_CONFIG_FILE_NAME "vine.xml"

int main()
{
	Context context;
	CanopyGenerator canopy_generator(&context);

	/**
	 * Build the plants
	 */

	/* Seed the random number generators */
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	context.seedRandomGenerator(seed);
	canopy_generator.seedRandomGenerator(seed);

	/* Read the vine config file and build the canopy
	Each set of parameters written in the file will be stored in our CanopyGenerator. Whenever we call CanopyGenerator::buildIndividualPlants() or
	CanopyGenerator::buildCanopies(), an instance of a single plant or a whole canopy will be built for each set of parameters that is stored. */
	const char *vine_config_file_path = CONFIG_DIR VINE_CONFIG_FILE_NAME;
	canopy_generator.loadXML(vine_config_file_path, true);

	/* Build an individual plant based on the same parameters */
	vec3 vine_stock_position = make_vec3(-7.5, -10, 0);
	// This builds only one plant if only one set of parameters is configured
	canopy_generator.buildIndividualPlants(vine_stock_position);

	/* Build an arc of plants based on the same parameters */
	// Number of vine stocks
	int num_plants = 8;
	vec3 circle_center = make_vec3(7.5, 6, 0);
	float circle_radius = 6;
	// Total angle coverage of the plants arc
	float total_arc_angle = 160;
	// Angle difference between each plant
	float angle_diff_between_plants = total_arc_angle / num_plants;
	// The vine parameters. Here we assume there is at least one set of parameters, and we get the last one.
	BaseCanopyParameters *params = canopy_generator.getCanopyParametersList().back().get();
	for (int plant_index = 0; plant_index < num_plants; plant_index++) {
		float angle = deg2rad(plant_index * angle_diff_between_plants);
		// We modify the parameters here, but beware because they will be saved, overwriting the ones read from the config file
		params->canopy_rotation = angle + deg2rad(90);
		vec3 plant_position = make_vec3(circle_center.x + circle_radius * cos(angle), circle_center.y + circle_radius * sin(angle), circle_center.z);
		canopy_generator.buildIndividualPlants(plant_position);
	}

	/**
	 * Build the ground
	 */

	vec3 origin(0, 0, 0);
	vec2 extent(30, 30);
	int2 num_tiles(6, 3);
	int2 subpatches(4, 4);
	canopy_generator.buildGround(origin, extent, num_tiles, subpatches, "plugins/canopygenerator/textures/dirt.jpg");

	/**
	 * Set the Visualizer
	 */

	SphericalCoord sun_dir;
	sun_dir = make_SphericalCoord(20 * M_PI / 180.f, 205 * M_PI / 180.f);

	Visualizer vis(1000);
	vis.setLightDirection(sphere2cart(sun_dir));
	vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
	vis.buildContextGeometry(&context);
	vis.plotInteractive();

	return EXIT_SUCCESS;
}
