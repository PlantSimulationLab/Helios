#include "EnergyBalanceModel.h"

using namespace helios;

int main(){
    uint Nx = 500;      // grid size in x
    uint Ny = 500;      // grid size in y
    float D = 10.f;     // domain width
    vec2 ground_size(D, D);

    std::ofstream outfile("../results/runtime.txt");
    Timer timer;
    double elapsed;

    Context context;
    std::vector<uint> ground_UUIDs, dragon_UUIDs, all_UUIDs;

    timer.tic();
    dragon_UUIDs = context.loadPLY("../../../../PLY/StanfordDragon.ply");
    elapsed = timer.toc("PLY model load");
    outfile << "PLY model load, " << elapsed << "\n";

    timer.tic();
    ground_UUIDs = context.addTile(nullorigin, ground_size, nullrotation, make_int2(Nx, Ny) );
    elapsed = timer.toc("Ground geometry creation");
    outfile << "Ground geometry creation, " << elapsed << "\n";

    all_UUIDs = ground_UUIDs;
    all_UUIDs.insert(all_UUIDs.end(), dragon_UUIDs.begin(), dragon_UUIDs.end());

    timer.tic();
    context.setPrimitiveData(all_UUIDs, "radiation_flux_SW", 300.f);
    float LW = 2.f * 5.67e-8f * pow(300.f, 4);
    context.setPrimitiveData(all_UUIDs, "radiation_flux_LW", LW);
    context.setPrimitiveData(all_UUIDs, "air_temperature", 300.f);
    context.setPrimitiveData(all_UUIDs, "wind_speed", 1.f);
    context.setPrimitiveData(all_UUIDs, "air_huidity", 0.5f);
    context.setPrimitiveData(all_UUIDs, "air_pressure", 101000.f);
    elapsed = timer.toc("Setting primitive data");
    outfile << "Setting primitive data, " << elapsed << "\n";

    EnergyBalanceModel eb(&context);
    eb.disableMessages();

    timer.tic();
    eb.addRadiationBand("SW");
    eb.addRadiationBand("LW");
    eb.run();
    elapsed = timer.toc("Energy balance run");
    outfile << "Energy balance run, " << elapsed << "\n";

    outfile.close();
    return EXIT_SUCCESS;
}