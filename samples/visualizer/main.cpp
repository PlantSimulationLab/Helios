#include "Visualizer.h" //include this header to use Visualizer

using namespace helios;

int main() {

    // Declare and initialize the Helios context
    // note that since we have used the `helios' namespace above, we do not need to declare the context as: helios::Context
    Context context;

    float D = 50; // width of the ground surface
    vec2 dx(1, 1); // lenght and width of chess board squares and grass patches
    int2 size(floor(D / dx.x), floor(D / dx.y));

    if (size.x % 2 == 0) {
        size.x += 1;
    }
    if (size.y % 2 == 0) {
        size.y += 1;
    }

    for (int j = 0; j < size.y; j++) {
        for (int i = 0; i < size.x; i++) {

            float rot = ((j * size.x + i) % 3) * M_PI * 0.5;

            vec3 position = make_vec3(-0.5 * D + i * dx.x, -0.5 * D + j * dx.y, 0);
            if (fabs(position.x) < dx.x * 5 && fabs(position.y) < dx.y * 5) {
                if ((j * size.x + i) % 2 == 0) {
                    context.addPatch(position, dx, make_SphericalCoord(0.f, rot), "plugins/visualizer/textures/marble_white.jpg");
                } else {
                    context.addPatch(position, dx, make_SphericalCoord(0.f, rot), "plugins/visualizer/textures/marble_black.jpg");
                }
            } else if (position.magnitude() < 0.5 * D) {
                context.addPatch(position - make_vec3(0, 0, 0.01), dx, make_SphericalCoord(0.f, rot), "plugins/visualizer/textures/grass3.jpg");
            }
        }
    }

    context.addPatch(make_vec3(0, 0, -0.005), make_vec2(9.1, 9.1), make_SphericalCoord(0.f, 0), make_RGBcolor(0.2, 0.2, 0.2));

    context.loadPLY("../../../PLY/king.ply", make_vec3(3, 1, 0), 1.2, RGB::white);

    SphericalCoord sun_dir;
    sun_dir = make_SphericalCoord(20 * M_PI / 180.f, 205 * M_PI / 180.f);

    Visualizer vis(1000);

    vis.setLightDirection(sphere2cart(sun_dir));

    vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

    vis.buildContextGeometry(&context);

    vis.addSkyDomeByCenter(20, make_vec3(0, 0, 0), 30, "plugins/visualizer/textures/SkyDome_clouds.jpg");

    vis.setCameraPosition(make_SphericalCoord(10, 0.05 * M_PI, 0.2f * M_PI), make_vec3(0, 0, 0));

    vis.plotInteractive();
}
