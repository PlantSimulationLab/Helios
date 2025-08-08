#include "PlantArchitecture.h"

using namespace helios;

int main() {
    uint Nx = 5; // number of plants in x-direction
    uint Ny = 5; // number of plants in y-direction
    vec2 spacing(0.3f, 0.3f); // plant spacing
    float age = 45.f; // age of plants in days

    std::ofstream outfile("../results/runtime.txt");
    Timer timer;
    double elapsed;

    Context context;
    PlantArchitecture pa(&context);
    pa.loadPlantModelFromLibrary("bean");

    timer.tic();
    pa.buildPlantCanopyFromLibrary(make_vec3(0.f, 0.f, 0.f), spacing, make_int2(Nx, Ny), age);
    elapsed = timer.toc("Canopy build");
    outfile << "Canopy build, " << elapsed << "\n";

    outfile.close();
    return EXIT_SUCCESS;
}
