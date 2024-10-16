#include "WeberPennTree.h"
#include "Visualizer.h"

using namespace helios;

const char *config_file_path = "../config/tree.xml";
const char *custom_tree_label = "CustomOlive";

/* Orchard parameters */

int num_rows = 4;
float row_spacing = 2.5;
float row_spacing_spread = 0.05;
int num_trees_per_row = 10;
float tree_spacing = 1.5;
float tree_spacing_spread = 0.3;

int main(int argc, char *argv[])
{
	Context context;
	WeberPennTree weberpenntree(&context);

	// Seed the random number generators
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
	context.seedRandomGenerator(seed);
	weberpenntree.seedRandomGenerator(seed);

	/* Load the config file to read the tree geometry parameters */
	weberpenntree.loadXML(config_file_path);

	/* Build the orchard */
	for (int row_index = 0; row_index < num_rows; row_index++) {
		for (int tree_index = 0; tree_index < num_trees_per_row; tree_index++) {
			vec3 tree_position = make_vec3(tree_index * tree_spacing + context.randu(-tree_spacing_spread, tree_spacing_spread), row_index * row_spacing + context.randu(-row_spacing_spread, row_spacing_spread), 0);
			/* uint id = */weberpenntree.buildTree(custom_tree_label, tree_position);
		}
	}


	/* Set the Visualizer */

	Visualizer vis(1000);
	// Direction of the sun
	SphericalCoord sun_dir;
	sun_dir = make_SphericalCoord(20 * M_PI / 180.f, 205 * M_PI / 180.f);
	vis.setLightDirection(sphere2cart(sun_dir));
	vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

	// Get the context domain bounding coordinates
	helios::vec2 x_bounds;
	helios::vec2 y_bounds;
	helios::vec2 z_bounds;
	context.getDomainBoundingBox(x_bounds, y_bounds, z_bounds);
	float x_range = x_bounds.y - x_bounds.x;
	float y_range = y_bounds.y - y_bounds.x;

	vec3 origin(x_bounds.x + x_range / 2, y_bounds.x + y_range / 2, 0);

	vis.addSkyDomeByCenter(max(std::vector<int>(x_range * 4, y_range * 4)), origin, 30, "plugins/visualizer/textures/SkyDome_clouds.jpg");

	vis.buildContextGeometry(&context);
	vis.plotInteractive();

	return EXIT_SUCCESS;
}
