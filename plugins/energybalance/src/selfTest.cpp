#include "EnergyBalanceModel.h"
#include "Context.h"
#include <iostream>
#include <cmath>

using namespace helios;

int EnergyBalanceModel::selfTest() {
    std::cout << "Running energy balance model self-test..." << std::flush;

    float err_tol = 1e-3;
    int error_count = 0;  // Track errors

    // ---- Equilibrium test ---- //
    {
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2)));
        UUIDs.push_back(context_test.addTriangle(make_vec3(4, 5, 6), make_vec3(5, 5, 6), make_vec3(5, 6, 6)));

        float Tref = 350;
        context_test.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
        context_test.setPrimitiveData(UUIDs, "air_temperature", Tref);

        EnergyBalanceModel energymodeltest(&context_test);
        energymodeltest.disableMessages();
        energymodeltest.addRadiationBand("LW");
        energymodeltest.run();

        for (int p = 0; p < UUIDs.size(); p++) {
            float T;
            context_test.getPrimitiveData(UUIDs.at(p), "temperature", T);
            if (fabs(T - Tref) > err_tol) {
                std::cout << "failed equilibrium test." << std::endl;
                error_count++;
            }
        }
    }

    // ---- Energy Budget Closure ---- //
    {
        Context context_2;
        std::vector<uint> UUIDs_2;
        UUIDs_2.push_back(context_2.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2)));
        UUIDs_2.push_back(context_2.addTriangle(make_vec3(4, 5, 6), make_vec3(5, 5, 6), make_vec3(5, 6, 6)));

        float T = 300;
        context_2.setPrimitiveData(UUIDs_2, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(T, 4)));
        context_2.setPrimitiveData(UUIDs_2, "radiation_flux_SW", float(300.f));
        context_2.setPrimitiveData(UUIDs_2, "air_temperature", T);

        EnergyBalanceModel energymodel_2(&context_2);
        energymodel_2.disableMessages();
        energymodel_2.addRadiationBand("LW");
        energymodel_2.addRadiationBand("SW");
        energymodel_2.run();

        for (int p = 0; p < UUIDs_2.size(); p++) {
            float sensible_flux, latent_flux, R, temperature;
            float Rin = 0;

            context_2.getPrimitiveData(UUIDs_2.at(p), "sensible_flux", sensible_flux);
            context_2.getPrimitiveData(UUIDs_2.at(p), "latent_flux", latent_flux);
            context_2.getPrimitiveData(UUIDs_2.at(p), "radiation_flux_LW", R);
            Rin += R;
            context_2.getPrimitiveData(UUIDs_2.at(p), "radiation_flux_SW", R);
            Rin += R;
            context_2.getPrimitiveData(UUIDs_2.at(p), "temperature", temperature);

            float Rout = 2.f * 5.67e-8 * pow(temperature, 4);
            float resid = Rin - Rout - sensible_flux - latent_flux;
            if (fabs(resid) > err_tol) {
                std::cout << "failed energy budget closure test." << std::endl;
                error_count++;
            }
        }
    }

    // ---- Temperature Solution Check ---- //
    {
        Context context_3;
        uint UUID_3 = context_3.addPatch(make_vec3(1,2,3), make_vec2(3,2));

        float T = 312.f;
        context_3.setPrimitiveData(UUID_3, "radiation_flux_LW", float(5.67e-8 * pow(T, 4)));
        context_3.setPrimitiveData(UUID_3, "radiation_flux_SW", float(350.f));
        context_3.setPrimitiveData(UUID_3, "wind_speed", float(1.244f));
        context_3.setPrimitiveData(UUID_3, "moisture_conductance", float(0.05f));
        context_3.setPrimitiveData(UUID_3, "air_humidity", float(0.4f));
        context_3.setPrimitiveData(UUID_3, "air_pressure", float(956789));
        context_3.setPrimitiveData(UUID_3, "other_surface_flux", float(150.f));
        context_3.setPrimitiveData(UUID_3, "air_temperature", T);
        context_3.setPrimitiveData(UUID_3, "twosided_flag", uint(0));

        EnergyBalanceModel energymodel_3(&context_3);
        energymodel_3.disableMessages();
        energymodel_3.addRadiationBand("LW");
        energymodel_3.addRadiationBand("SW");
        energymodel_3.run();

        float sensible_flux, latent_flux, temperature;
        float sensible_flux_exact = 48.7098;
        float latent_flux_exact = 21.7644;
        float temperature_exact = 329.309;

        context_3.getPrimitiveData(UUID_3, "sensible_flux", sensible_flux);
        context_3.getPrimitiveData(UUID_3, "latent_flux", latent_flux);
        context_3.getPrimitiveData(UUID_3, "temperature", temperature);

        if (fabs(sensible_flux - sensible_flux_exact) / fabs(sensible_flux_exact) > err_tol ||
            fabs(latent_flux - latent_flux_exact) / fabs(latent_flux_exact) > err_tol ||
            fabs(temperature - temperature_exact) / fabs(temperature_exact) > err_tol) {
            std::cout << "failed temperature solver check #1." << std::endl;
            error_count++;
        }

        // Use object length instead of sqrt(area)
        context_3.setPrimitiveData(UUID_3, "object_length", float(0.374f));
        energymodel_3.run();

        sensible_flux_exact = 89.2217;
        latent_flux_exact = 20.2213;
        temperature_exact = 324.389;

        context_3.getPrimitiveData(UUID_3, "sensible_flux", sensible_flux);
        context_3.getPrimitiveData(UUID_3, "latent_flux", latent_flux);
        context_3.getPrimitiveData(UUID_3, "temperature", temperature);

        if (fabs(sensible_flux - sensible_flux_exact) / fabs(sensible_flux_exact) > err_tol ||
            fabs(latent_flux - latent_flux_exact) / fabs(latent_flux_exact) > err_tol ||
            fabs(temperature - temperature_exact) / fabs(temperature_exact) > err_tol) {
            std::cout << "failed temperature solver check #2." << std::endl;
            error_count++;
        }

        // Manually set boundary-layer conductance
        context_3.setPrimitiveData(UUID_3, "boundarylayer_conductance", float(0.134f));
        energymodel_3.run();

        sensible_flux_exact = 61.5525f;
        latent_flux_exact = 21.8290f;
        temperature_exact = 327.704f;

        context_3.getPrimitiveData(UUID_3, "sensible_flux", sensible_flux);
        context_3.getPrimitiveData(UUID_3, "latent_flux", latent_flux);
        context_3.getPrimitiveData(UUID_3, "temperature", temperature);

        if (fabs(sensible_flux - sensible_flux_exact) / fabs(sensible_flux_exact) > err_tol ||
            fabs(latent_flux - latent_flux_exact) / fabs(latent_flux_exact) > err_tol ||
            fabs(temperature - temperature_exact) / fabs(temperature_exact) > err_tol) {
            std::cout << "failed temperature solver check #3." << std::endl;
            error_count++;
        }
    }

    // ---- Optional Primitive Data Output Check ---- //
    {
        Context context_4;
        uint UUID_4 = context_4.addPatch(make_vec3(1,2,3), make_vec2(3,2));

        EnergyBalanceModel energymodel_4(&context_4);
        energymodel_4.disableMessages();
        energymodel_4.addRadiationBand("LW");
        energymodel_4.optionalOutputPrimitiveData("boundarylayer_conductance_out");
        energymodel_4.optionalOutputPrimitiveData("vapor_pressure_deficit");

        context_4.setPrimitiveData(UUID_4, "radiation_flux_LW", 0.f);
        energymodel_4.run();

        if (!context_4.doesPrimitiveDataExist(UUID_4, "vapor_pressure_deficit") ||
            !context_4.doesPrimitiveDataExist(UUID_4, "boundarylayer_conductance_out")) {
            std::cout << "failed optional primitive data output check." << std::endl;
            error_count++;
        }
    }

    // ---- Dynamic Model Check ---- //
    {
        Context context_5;
        float dt_5 = 1.f, T_5 = 3600, To_5 = 300.f, cp_5 = 2000;
        float Rlow = 50.f, Rhigh = 500.f;

        uint UUID_5 = context_5.addPatch(make_vec3(0,0,0), make_vec2(1,1));
        context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rlow);
        context_5.setPrimitiveData(UUID_5, "temperature", To_5);
        context_5.setPrimitiveData(UUID_5, "heat_capacity", cp_5);
        context_5.setPrimitiveData(UUID_5, "twosided_flag", uint(0));
        context_5.setPrimitiveData(UUID_5, "emissivity_SW", 0.f);

        EnergyBalanceModel energybalance_5(&context_5);
        energybalance_5.disableMessages();
        energybalance_5.addRadiationBand("SW");
        energybalance_5.optionalOutputPrimitiveData("boundarylayer_conductance_out");

        std::vector<float> temperature_dyn;
        int N = round(T_5/dt_5);
        for (int t = 0; t < N; t++) {
            if (t > 0.5f * N) {
                context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rhigh);
            }
            energybalance_5.run(dt_5);

            float temp;
            context_5.getPrimitiveData(UUID_5, "temperature", temp);
            temperature_dyn.push_back(temp);
        }

        float gH_5;
        context_5.getPrimitiveData(UUID_5, "boundarylayer_conductance_out", gH_5);
        float tau_5 = cp_5 / gH_5 / 29.25f;

        context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rlow);
        energybalance_5.run();
        float Tlow;
        context_5.getPrimitiveData(UUID_5, "temperature", Tlow);

        context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rhigh);
        energybalance_5.run();
        float Thigh;
        context_5.getPrimitiveData(UUID_5, "temperature", Thigh);

        float err = 0;
        for (int t = round(0.5f * N); t < N; t++) {
            float time = dt_5 * (t - round(0.5f * N));
            float temperature_ref = Tlow + (Thigh - Tlow) * (1.f - exp(-time / tau_5));
            err += pow(temperature_ref - temperature_dyn.at(t), 2);
        }

        err = sqrt(err / float(N));

        if (err > 0.2f) {
            std::cout << "failed dynamic energy balance check." << std::endl;
            error_count++;
        }
    }

    // ---- Test enableMessages() and disableMessages() ---- //
    {
        Context context_enable;
        EnergyBalanceModel testModel(&context_enable);

        testModel.enableMessages();
        if (!testModel.message_flag) {
            std::cout << "failed enableMessages() test." << std::endl;
            error_count++;
        }

        testModel.disableMessages();
        if (testModel.message_flag) {
            std::cout << "failed disableMessages() test." << std::endl;
            error_count++;
        }
    }

    // ---- Test addRadiationBand() ---- //
    {
        Context context_radiation;
        EnergyBalanceModel testModel(&context_radiation);

        testModel.addRadiationBand("LW");
        testModel.addRadiationBand("LW");  // Should not duplicate
        testModel.addRadiationBand("SW");

        if (testModel.radiation_bands.size() != 2) {
            std::cout << "failed addRadiationBand() test." << std::endl;
            error_count++;
        }
    }

    // ---- Test optionalOutputPrimitiveData() ---- //
    {
        Context context_output;
        EnergyBalanceModel testModel(&context_output);

        testModel.optionalOutputPrimitiveData("boundarylayer_conductance_out");
        testModel.optionalOutputPrimitiveData("invalid_label");  // Should print warning

        if (testModel.output_prim_data.size() != 1 || testModel.output_prim_data[0] != "boundarylayer_conductance_out") {
            std::cout << "failed optionalOutputPrimitiveData() test." << std::endl;
            error_count++;
        }
    }


    // ---- Test printDefaultValueReport() ---- //
    {
        Context context_print;
        EnergyBalanceModel testModel(&context_print);

        std::stringstream buffer;
        std::streambuf *old = std::cout.rdbuf(buffer.rdbuf()); // Redirect std::cout

        testModel.printDefaultValueReport();

        std::cout.rdbuf(old); // Restore std::cout

        std::string output = buffer.str();
        std::vector<std::string> required_keywords = {
                "surface temperature",
                "air pressure",
                "air temperature",
                "air humidity",
                "boundary-layer conductance",
                "moisture conductance",
                "surface humidity",
                "two-sided flag",
                "evaporating faces"
        };

        for (const auto& keyword : required_keywords) {
            if (output.find(keyword) == std::string::npos) {
                std::cout << "failed printDefaultValueReport() - missing: " << keyword << std::endl;
                error_count++;
            }
        }
    }

    // ---- Test printDefaultValueReport(const std::vector<uint>& UUIDs) ---- //
    {
        Context context_print;
        EnergyBalanceModel testModel(&context_print);
        std::vector<uint> testUUIDs;
        testUUIDs.push_back(context_print.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2)));

        std::stringstream buffer;
        std::streambuf *old = std::cout.rdbuf(buffer.rdbuf()); // Redirect std::cout

        testModel.printDefaultValueReport(testUUIDs);

        std::cout.rdbuf(old); // Restore std::cout

        std::string output = buffer.str();
        std::vector<std::string> required_keywords = {
                "surface temperature",
                "air pressure",
                "air temperature",
                "air humidity",
                "boundary-layer conductance",
                "moisture conductance",
                "surface humidity",
                "two-sided flag",
                "evaporating faces"
        };

        for (const auto& keyword : required_keywords) {
            if (output.find(keyword) == std::string::npos) {
                std::cout << "failed printDefaultValueReport(UUIDs) - missing: " << keyword << std::endl;
                error_count++;
            }
        }
    }


    // ---- Dynamic Model Check ---- //
    {
        Context context_dyn;
        float dt = 1.f, Tfinal = 3600, To = 300.f, cp = 2000;
        float Rlow = 50.f, Rhigh = 500.f;

        uint UUID_dyn = context_dyn.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        context_dyn.setPrimitiveData(UUID_dyn, "radiation_flux_SW", Rlow);
        context_dyn.setPrimitiveData(UUID_dyn, "temperature", To);
        context_dyn.setPrimitiveData(UUID_dyn, "heat_capacity", cp);
        context_dyn.setPrimitiveData(UUID_dyn, "twosided_flag", uint(0));
        context_dyn.setPrimitiveData(UUID_dyn, "emissivity_SW", 0.f);

        EnergyBalanceModel energybalance_dyn(&context_dyn);
        energybalance_dyn.disableMessages();
        energybalance_dyn.addRadiationBand("SW");

        std::vector<float> temperature_dyn;
        int N = round(Tfinal / dt);
        for (int t = 0; t < N; t++) {
            if (t > 0.5f * N) {
                context_dyn.setPrimitiveData(UUID_dyn, "radiation_flux_SW", Rhigh);
            }
            energybalance_dyn.run(dt);

            float temp;
            context_dyn.getPrimitiveData(UUID_dyn, "temperature", temp);
            temperature_dyn.push_back(temp);
        }

        if (temperature_dyn.empty()) {
            std::cout << "failed dynamic model test." << std::endl;
            error_count++;
        }
    }


    // ---- Final Test Summary ---- //
    if (error_count > 0) {
        std::cout << "completed with " << error_count << " errors." << std::endl;
    } else {
        std::cout << "passed all tests." << std::endl;
    }

    return error_count;
}