#include "RadiationModel.h"

using namespace helios;

int main() {

    bool failure = false;

    uint Ndirect = 1024;
    uint Ndiffuse = 1024;

    float D = 20; // domain width
    float LAI = 1.0; // canopy leaf area index
    float h = 3; // canopy height
    float w_leaf = 0.05; // leaf width

    std::ofstream outfile("../results/runtime.txt");

    Timer timer;
    double elapsed;

    Context context;

    timer.tic();

    uint objID_ptype = context.addTileObject(make_vec3(0, 0, 0), make_vec2(w_leaf, w_leaf), make_SphericalCoord(0, 0), make_int2(2, 2), "plugins/radiation/disk.png");
    std::vector<uint> UUIDs_ptype = context.getObjectPrimitiveUUIDs(objID_ptype);

    float A_leaf = 0;
    for (uint p = 0; p < UUIDs_ptype.size(); p++) {
        A_leaf += context.getPrimitiveArea(UUIDs_ptype.at(p));
    }

    int Nleaves = round(LAI * D * D / A_leaf);

    std::vector<uint> UUIDs_leaf;

    for (int i = 0; i < Nleaves; i++) {

        vec3 position((-0.5 + context.randu()) * D, (-0.5 + context.randu()) * D, 0.5 * w_leaf + context.randu() * h);

        SphericalCoord rotation(1.f, acos(1.f - context.randu()), 2.f * M_PI * context.randu());

        uint objID = context.copyObject(objID_ptype);

        context.rotateObject(objID, -rotation.elevation, "y");
        context.rotateObject(objID, rotation.azimuth, "z");

        context.translateObject(objID, position);

        std::vector<uint> UUIDs = context.getObjectPrimitiveUUIDs(objID);

        UUIDs_leaf.insert(UUIDs_leaf.end(), UUIDs.begin(), UUIDs.end());
    }

    context.deleteObject(objID_ptype);

    std::vector<uint> UUIDs_ground = context.addTile(make_vec3(0, 0, 0), make_vec2(D, D), make_SphericalCoord(0, 0), make_int2(100, 100));

    context.setPrimitiveData(UUIDs_ground, "twosided_flag", uint(0));

    elapsed = timer.toc("Geometry creation");
    outfile << "Geometry Creation, " << elapsed << "\n";

    RadiationModel radiation(&context);
    radiation.disableMessages();

    radiation.addRadiationBand("direct");
    radiation.disableEmission("direct");
    radiation.setDirectRayCount("direct", Ndirect);
    float theta_s = 0.2 * M_PI;
    uint ID = radiation.addCollimatedRadiationSource(make_SphericalCoord(0.5 * M_PI - theta_s, 0.f));
    radiation.setSourceFlux(ID, "direct", 1.f / cos(theta_s));

    radiation.addRadiationBand("diffuse");
    radiation.disableEmission("diffuse");
    radiation.setDiffuseRayCount("diffuse", Ndiffuse);
    radiation.setDiffuseRadiationFlux("diffuse", 1.f);

    radiation.enforcePeriodicBoundary("xy");

    timer.tic();

    radiation.updateGeometry();

    elapsed = timer.toc("Radiation geometry update");
    outfile << "Radiation geometry update, " << elapsed << "\n";

    timer.tic();

    radiation.runBand("direct");

    elapsed = timer.toc("Direct ray trace");
    outfile << "Direct ray trace, " << elapsed << "\n";

    timer.tic();

    radiation.runBand("diffuse");

    elapsed = timer.toc("Diffuse ray trace");
    outfile << "Diffuse ray trace, " << elapsed << "\n";

    outfile.close();

    return EXIT_SUCCESS;
}
