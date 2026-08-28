#include "EnergyBalanceModel.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

float err_tol = 1e-3;

DOCTEST_TEST_CASE("EnergyBalanceModel Equilibrium Test") {
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
    DOCTEST_CHECK_NOTHROW(energymodeltest.run());

    for (int p = 0; p < UUIDs.size(); p++) {
        float T;
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUIDs.at(p), "temperature", T));
        DOCTEST_CHECK(T == doctest::Approx(Tref).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("EnergyBalanceModel Energy Budget Closure") {
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
    DOCTEST_CHECK_NOTHROW(energymodel_2.run());

    for (int p = 0; p < UUIDs_2.size(); p++) {
        float sensible_flux, latent_flux, R, temperature;
        float Rin = 0;

        DOCTEST_CHECK_NOTHROW(context_2.getPrimitiveData(UUIDs_2.at(p), "sensible_flux", sensible_flux));
        DOCTEST_CHECK_NOTHROW(context_2.getPrimitiveData(UUIDs_2.at(p), "latent_flux", latent_flux));
        DOCTEST_CHECK_NOTHROW(context_2.getPrimitiveData(UUIDs_2.at(p), "radiation_flux_LW", R));
        Rin += R;
        DOCTEST_CHECK_NOTHROW(context_2.getPrimitiveData(UUIDs_2.at(p), "radiation_flux_SW", R));
        Rin += R;
        DOCTEST_CHECK_NOTHROW(context_2.getPrimitiveData(UUIDs_2.at(p), "temperature", temperature));

        float Rout = 2.f * 5.67e-8 * pow(temperature, 4);
        float resid = Rin - Rout - sensible_flux - latent_flux;
        DOCTEST_CHECK(resid == doctest::Approx(0.0f).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("EnergyBalanceModel Temperature Solution Check 1") {
    Context context_3;
    uint UUID_3 = context_3.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2));

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
    DOCTEST_CHECK_NOTHROW(energymodel_3.run());

    float sensible_flux, latent_flux, temperature;
    float sensible_flux_exact = 48.7017;
    float latent_flux_exact = 21.6094;
    float temperature_exact = 329.307;

    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "sensible_flux", sensible_flux));
    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "latent_flux", latent_flux));
    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "temperature", temperature));

    DOCTEST_CHECK(sensible_flux == doctest::Approx(sensible_flux_exact).epsilon(err_tol));
    DOCTEST_CHECK(latent_flux == doctest::Approx(latent_flux_exact).epsilon(err_tol));
    DOCTEST_CHECK(temperature == doctest::Approx(temperature_exact).epsilon(err_tol));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Temperature Solution Check 2 - Object Length") {
    Context context_3;
    uint UUID_3 = context_3.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2));

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

    // Use object length instead of sqrt(area)
    context_3.setPrimitiveData(UUID_3, "object_length", float(0.374f));
    DOCTEST_CHECK_NOTHROW(energymodel_3.run());

    float sensible_flux, latent_flux, temperature;
    float sensible_flux_exact = 89.2024;
    float latent_flux_exact = 20.0723;
    float temperature_exact = 324.386;

    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "sensible_flux", sensible_flux));
    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "latent_flux", latent_flux));
    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "temperature", temperature));

    DOCTEST_CHECK(sensible_flux == doctest::Approx(sensible_flux_exact).epsilon(err_tol));
    DOCTEST_CHECK(latent_flux == doctest::Approx(latent_flux_exact).epsilon(err_tol));
    DOCTEST_CHECK(temperature == doctest::Approx(temperature_exact).epsilon(err_tol));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Temperature Solution Check 3 - Manual Boundary Layer Conductance") {
    Context context_3;
    uint UUID_3 = context_3.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2));

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

    // Manually set boundary-layer conductance
    context_3.setPrimitiveData(UUID_3, "boundarylayer_conductance", float(0.134f));
    DOCTEST_CHECK_NOTHROW(energymodel_3.run());

    float sensible_flux, latent_flux, temperature;
    float sensible_flux_exact = 61.5411f;
    float latent_flux_exact = 21.6718f;
    float temperature_exact = 327.701f;

    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "sensible_flux", sensible_flux));
    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "latent_flux", latent_flux));
    DOCTEST_CHECK_NOTHROW(context_3.getPrimitiveData(UUID_3, "temperature", temperature));

    DOCTEST_CHECK(sensible_flux == doctest::Approx(sensible_flux_exact).epsilon(err_tol));
    DOCTEST_CHECK(latent_flux == doctest::Approx(latent_flux_exact).epsilon(err_tol));
    DOCTEST_CHECK(temperature == doctest::Approx(temperature_exact).epsilon(err_tol));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Optional Primitive Data Output Check") {
    Context context_4;
    uint UUID_4 = context_4.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2));

    EnergyBalanceModel energymodel_4(&context_4);
    energymodel_4.disableMessages();
    energymodel_4.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(energymodel_4.optionalOutputPrimitiveData("boundarylayer_conductance_out"));
    DOCTEST_CHECK_NOTHROW(energymodel_4.optionalOutputPrimitiveData("vapor_pressure_deficit"));

    context_4.setPrimitiveData(UUID_4, "radiation_flux_LW", 0.f);
    DOCTEST_CHECK_NOTHROW(energymodel_4.run());

    DOCTEST_CHECK(context_4.doesPrimitiveDataExist(UUID_4, "vapor_pressure_deficit"));
    DOCTEST_CHECK(context_4.doesPrimitiveDataExist(UUID_4, "boundarylayer_conductance_out"));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Dynamic Model Check") {
    Context context_5;
    float dt_5 = 1.f, T_5 = 3600, To_5 = 300.f, cp_5 = 2000;
    float Rlow = 50.f, Rhigh = 500.f;

    uint UUID_5 = context_5.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rlow);
    context_5.setPrimitiveData(UUID_5, "temperature", To_5);
    context_5.setPrimitiveData(UUID_5, "heat_capacity", cp_5);
    context_5.setPrimitiveData(UUID_5, "twosided_flag", uint(0));
    context_5.setPrimitiveData(UUID_5, "emissivity_SW", 0.f);

    EnergyBalanceModel energybalance_5(&context_5);
    energybalance_5.disableMessages();
    energybalance_5.addRadiationBand("SW");
    DOCTEST_CHECK_NOTHROW(energybalance_5.optionalOutputPrimitiveData("boundarylayer_conductance_out"));

    std::vector<float> temperature_dyn;
    int N = round(T_5 / dt_5);
    for (int t = 0; t < N; t++) {
        if (t > 0.5f * N) {
            context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rhigh);
        }
        DOCTEST_CHECK_NOTHROW(energybalance_5.run(dt_5));

        float temp;
        DOCTEST_CHECK_NOTHROW(context_5.getPrimitiveData(UUID_5, "temperature", temp));
        temperature_dyn.push_back(temp);
    }

    float gH_5;
    DOCTEST_CHECK_NOTHROW(context_5.getPrimitiveData(UUID_5, "boundarylayer_conductance_out", gH_5));
    float tau_5 = cp_5 / gH_5 / 29.25f;

    context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rlow);
    DOCTEST_CHECK_NOTHROW(energybalance_5.run());
    float Tlow;
    DOCTEST_CHECK_NOTHROW(context_5.getPrimitiveData(UUID_5, "temperature", Tlow));

    context_5.setPrimitiveData(UUID_5, "radiation_flux_SW", Rhigh);
    DOCTEST_CHECK_NOTHROW(energybalance_5.run());
    float Thigh;
    DOCTEST_CHECK_NOTHROW(context_5.getPrimitiveData(UUID_5, "temperature", Thigh));

    float err = 0;
    for (int t = round(0.5f * N); t < N; t++) {
        float time = dt_5 * (t - round(0.5f * N));
        float temperature_ref = Tlow + (Thigh - Tlow) * (1.f - exp(-time / tau_5));
        err += pow(temperature_ref - temperature_dyn.at(t), 2);
    }

    err = sqrt(err / float(N));
    DOCTEST_CHECK(err == doctest::Approx(0.0f).epsilon(0.2f));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Enable/Disable Messages") {
    Context context_enable;
    EnergyBalanceModel testModel(&context_enable);

    DOCTEST_CHECK_NOTHROW(testModel.enableMessages());
    DOCTEST_CHECK_NOTHROW(testModel.disableMessages());
}

DOCTEST_TEST_CASE("EnergyBalanceModel Radiation Band Management") {
    Context context_radiation;
    EnergyBalanceModel testModel(&context_radiation);

    DOCTEST_CHECK_NOTHROW(testModel.addRadiationBand("LW"));
    DOCTEST_CHECK_NOTHROW(testModel.addRadiationBand("LW")); // Should not duplicate
    DOCTEST_CHECK_NOTHROW(testModel.addRadiationBand("SW"));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Optional Output Primitive Data") {
    Context context_output;
    EnergyBalanceModel testModel(&context_output);

    DOCTEST_CHECK_NOTHROW(testModel.optionalOutputPrimitiveData("boundarylayer_conductance_out"));

    bool has_cerr_output;
    {
        capture_cerr cerr_buffer;
        testModel.optionalOutputPrimitiveData("invalid_label"); // Should print warning
        has_cerr_output = cerr_buffer.has_output();
    } // capture destroyed here before assertion
    DOCTEST_CHECK(has_cerr_output);
}

DOCTEST_TEST_CASE("EnergyBalanceModel Print Default Value Report") {
    Context context_print;
    EnergyBalanceModel testModel(&context_print);

    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf()); // Redirect std::cout

    DOCTEST_CHECK_NOTHROW(testModel.printDefaultValueReport());

    std::cout.rdbuf(old); // Restore std::cout

    std::string output = buffer.str();
    std::vector<std::string> required_keywords = {"surface temperature", "air pressure", "air temperature", "air humidity", "boundary-layer conductance", "moisture conductance", "surface humidity", "two-sided flag", "evaporating faces"};

    for (const auto &keyword: required_keywords) {
        DOCTEST_CHECK(output.find(keyword) != std::string::npos);
    }
}

DOCTEST_TEST_CASE("EnergyBalanceModel Print Default Value Report with UUIDs") {
    Context context_print;
    EnergyBalanceModel testModel(&context_print);
    std::vector<uint> testUUIDs;
    testUUIDs.push_back(context_print.addPatch(make_vec3(1, 2, 3), make_vec2(3, 2)));

    std::stringstream buffer;
    std::streambuf *old = std::cout.rdbuf(buffer.rdbuf()); // Redirect std::cout

    DOCTEST_CHECK_NOTHROW(testModel.printDefaultValueReport(testUUIDs));

    std::cout.rdbuf(old); // Restore std::cout

    std::string output = buffer.str();
    std::vector<std::string> required_keywords = {"surface temperature", "air pressure", "air temperature", "air humidity", "boundary-layer conductance", "moisture conductance", "surface humidity", "two-sided flag", "evaporating faces"};

    for (const auto &keyword: required_keywords) {
        DOCTEST_CHECK(output.find(keyword) != std::string::npos);
    }
}

DOCTEST_TEST_CASE("EnergyBalanceModel Additional Dynamic Model Check") {
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
        DOCTEST_CHECK_NOTHROW(energybalance_dyn.run(dt));

        float temp;
        DOCTEST_CHECK_NOTHROW(context_dyn.getPrimitiveData(UUID_dyn, "temperature", temp));
        temperature_dyn.push_back(temp);
    }

    DOCTEST_CHECK(!temperature_dyn.empty());
}

#ifdef HELIOS_CUDA_AVAILABLE
DOCTEST_TEST_CASE("EnergyBalanceModel GPU vs CPU Consistency") {
    Context context_gpu_cpu;
    uint UUID_gpu_cpu = context_gpu_cpu.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    context_gpu_cpu.setPrimitiveData(UUID_gpu_cpu, "radiation_flux_SW", 400.f);
    context_gpu_cpu.setPrimitiveData(UUID_gpu_cpu, "air_temperature", 300.f);
    context_gpu_cpu.setPrimitiveData(UUID_gpu_cpu, "air_humidity", 0.5f);
    context_gpu_cpu.setPrimitiveData(UUID_gpu_cpu, "moisture_conductance", 0.1f);

    EnergyBalanceModel model_gpu_cpu(&context_gpu_cpu);
    model_gpu_cpu.disableMessages();
    model_gpu_cpu.addRadiationBand("SW");

    // Check if GPU is available
    bool gpu_available = model_gpu_cpu.isGPUAccelerationEnabled();
    if (!gpu_available) {
        DOCTEST_WARN("GPU not available - skipping GPU vs CPU comparison test");
        return;
    }

    // Run with GPU
    DOCTEST_CHECK_NOTHROW(model_gpu_cpu.run());
    float T_gpu;
    DOCTEST_CHECK_NOTHROW(context_gpu_cpu.getPrimitiveData(UUID_gpu_cpu, "temperature", T_gpu));

    // Run with CPU
    model_gpu_cpu.disableGPUAcceleration();
    DOCTEST_CHECK(model_gpu_cpu.isGPUAccelerationEnabled() == false);
    DOCTEST_CHECK_NOTHROW(model_gpu_cpu.run());
    float T_cpu;
    DOCTEST_CHECK_NOTHROW(context_gpu_cpu.getPrimitiveData(UUID_gpu_cpu, "temperature", T_cpu));

    // Verify GPU and CPU give same result (within floating-point precision)
    DOCTEST_CHECK(T_cpu == doctest::Approx(T_gpu).epsilon(1e-4));
}
#endif

DOCTEST_TEST_CASE("EnergyBalanceModel - Default value warnings") {

    DOCTEST_SUBCASE("Missing air_temperature triggers warning") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        // Set only radiation, leave air_temperature unset
        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);

        // Capture cout to check warnings - must go out of scope before assertions
        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        DOCTEST_CHECK(output.find("missing_air_temperature") != std::string::npos);
        DOCTEST_CHECK(output.find("WARNING:") != std::string::npos);
        DOCTEST_CHECK(output.find("instance") != std::string::npos); // singular for 1 primitive
    }

    DOCTEST_SUBCASE("Missing wind_speed triggers warning") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
        context.setPrimitiveData(UUID, "air_temperature", 300.f);
        // Leave wind_speed and boundarylayer_conductance unset

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        DOCTEST_CHECK(output.find("missing_wind_speed") != std::string::npos);
        DOCTEST_CHECK(output.find("WARNING:") != std::string::npos);
    }

    DOCTEST_SUBCASE("Missing air_humidity triggers warning") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
        context.setPrimitiveData(UUID, "air_temperature", 300.f);
        // Leave air_humidity unset

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        DOCTEST_CHECK(output.find("missing_air_humidity") != std::string::npos);
        DOCTEST_CHECK(output.find("WARNING:") != std::string::npos);
    }

    DOCTEST_SUBCASE("Missing surface temperature triggers warning") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
        context.setPrimitiveData(UUID, "air_temperature", 300.f);
        // Leave temperature unset

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        DOCTEST_CHECK(output.find("missing_surface_temperature") != std::string::npos);
        DOCTEST_CHECK(output.find("WARNING:") != std::string::npos);
    }

    DOCTEST_SUBCASE("Missing emissivity triggers warning") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
        context.setPrimitiveData(UUID, "air_temperature", 300.f);
        // Leave emissivity_LW unset (will default to 1.0)

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        DOCTEST_CHECK(output.find("missing_emissivity") != std::string::npos);
        DOCTEST_CHECK(output.find("WARNING:") != std::string::npos);
    }

    DOCTEST_SUBCASE("Air temperature < 250K triggers warning") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
        context.setPrimitiveData(UUID, "air_temperature", 25.f); // Likely Celsius

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        DOCTEST_CHECK(output.find("air_temperature_likely_celsius") != std::string::npos);
        DOCTEST_CHECK(output.find("WARNING:") != std::string::npos);
    }

    DOCTEST_SUBCASE("Air humidity out of range triggers warning") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
        context.setPrimitiveData(UUID, "air_temperature", 300.f);
        context.setPrimitiveData(UUID, "air_humidity", 1.5f); // Out of range

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        DOCTEST_CHECK(output.find("air_humidity_out_of_range_high") != std::string::npos);
        DOCTEST_CHECK(output.find("WARNING:") != std::string::npos);
    }

    DOCTEST_SUBCASE("Messages can be disabled") {
        Context context;
        uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

        context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
        // Leave air_temperature unset

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.disableMessages();
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }
        // Should produce no warnings
        DOCTEST_CHECK(output.find("WARNING:") == std::string::npos);
    }

    DOCTEST_SUBCASE("Multiple missing parameters produce aggregated warnings") {
        Context context;
        std::vector<uint> UUIDs;
        // Create 10 primitives with missing parameters
        for (int i = 0; i < 10; i++) {
            uint UUID = context.addPatch(make_vec3(i, 0, 0), make_vec2(1, 1));
            UUIDs.push_back(UUID);
            context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
            // Leave air_temperature, wind_speed, air_humidity unset
        }

        std::string output;
        {
            capture_cout capture;
            EnergyBalanceModel model(&context);
            model.addRadiationBand("LW");
            model.run();
            output = capture.get_captured_output();
        }

        // Should have warnings for missing parameters
        DOCTEST_CHECK(output.find("missing_air_temperature") != std::string::npos);
        DOCTEST_CHECK(output.find("missing_wind_speed") != std::string::npos);
        DOCTEST_CHECK(output.find("missing_air_humidity") != std::string::npos);

        // Should show counts (10 instances for each)
        DOCTEST_CHECK(output.find("10 instances") != std::string::npos);

        // Should NOT show individual example messages (compact mode)
        DOCTEST_CHECK(output.find("(showing first") == std::string::npos);
    }
}

DOCTEST_TEST_CASE("EnergyBalanceModel initial temperature guess equal to internal second guess") {
    // Regression test: the secant solver uses a hardcoded second initial guess of 400 K. When the
    // user-supplied initial 'temperature' equals that value, the two initial guesses produce identical
    // residuals, the solver's degenerate-case branch is taken on the first iteration, and (on the buggy
    // code) the uninitialized result variable is written back as the surface temperature.
    Context context;
    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    float Teq = 350.f; // Purely radiative equilibrium temperature (gH = 0 => no sensible/latent flux)
    context.setPrimitiveData(UUID, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Teq, 4)));
    context.setPrimitiveData(UUID, "air_temperature", 300.f);
    context.setPrimitiveData(UUID, "boundarylayer_conductance", 0.f); // gH = 0
    context.setPrimitiveData(UUID, "temperature", 400.f); // Collides with the internal second guess

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.run());

    float T;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "temperature", T));
    DOCTEST_CHECK(std::isfinite(T));
    DOCTEST_CHECK(T == doctest::Approx(Teq).epsilon(err_tol));
}

DOCTEST_TEST_CASE("EnergyBalanceModel zero boundary-layer conductance with stomatal conductance") {
    // Regression test: with gH = 0 (e.g. zero wind) and a nonzero moisture conductance on a surface
    // whose stomatal sidedness is zero, the total moisture conductance expression evaluates 0/0. The
    // buggy guard only caught the gH == 0 && gS == 0 case, so latent_flux was written as NaN.
    Context context;
    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
    context.setPrimitiveData(UUID, "air_temperature", 300.f);
    context.setPrimitiveData(UUID, "boundarylayer_conductance", 0.f); // gH = 0
    context.setPrimitiveData(UUID, "moisture_conductance", 0.1f); // gS > 0
    // stomatal_sidedness left unset -> defaults to 0 -> 0/0 in the buggy conductance expression

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.run());

    float latent_flux, T;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "latent_flux", latent_flux));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "temperature", T));
    DOCTEST_CHECK(std::isfinite(latent_flux));
    DOCTEST_CHECK(latent_flux == doctest::Approx(0.0f)); // No boundary-layer conductance => no vapor transport
    DOCTEST_CHECK(std::isfinite(T));
}

DOCTEST_TEST_CASE("EnergyBalanceModel single emission-enabled band selects its emissivity") {
    // When the RadiationModel records emission-enabled state, thermal emission must use the emissivity of
    // the single emission-enabled band (here LW, emissivity 0.8) and ignore other bands' emissivities.
    Context context;
    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    float R = 500.f;
    context.setPrimitiveData(UUID, "radiation_flux_LW", R);
    context.setPrimitiveData(UUID, "radiation_flux_SW", 0.f);
    context.setPrimitiveData(UUID, "air_temperature", 300.f);
    context.setPrimitiveData(UUID, "boundarylayer_conductance", 0.f); // gH = 0 => pure radiative equilibrium
    context.setPrimitiveData(UUID, "emissivity_LW", 0.8f);
    context.setPrimitiveData(UUID, "emissivity_SW", 0.4f);
    context.setGlobalData("emission_enabled_LW", uint(1)); // Simulates a radiation run: emission only on LW
    context.setGlobalData("emission_enabled_SW", uint(0));

    std::string output;
    {
        capture_cout capture;
        EnergyBalanceModel model(&context);
        model.addRadiationBand("LW");
        model.addRadiationBand("SW");
        model.run();
        output = capture.get_captured_output();
    }

    float T;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "temperature", T));
    // Pure radiative equilibrium: R = Nsides * eps * sigma * T^4, with Nsides = 2 and eps = 0.8 (the LW band)
    float T_expected = float(pow(R / (2.f * 0.8f * 5.67e-8f), 0.25));
    DOCTEST_CHECK(T == doctest::Approx(T_expected).epsilon(err_tol));
    DOCTEST_CHECK(output.find("multiple_emission_bands") == std::string::npos);
}

DOCTEST_TEST_CASE("EnergyBalanceModel multiple emission-enabled bands warn and use the first") {
    // With emission enabled on more than one band the choice is ambiguous: warn and use the emissivity of
    // the first emission-enabled band (LW, added first, emissivity 0.8).
    Context context;
    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    float R = 500.f;
    context.setPrimitiveData(UUID, "radiation_flux_LW", R);
    context.setPrimitiveData(UUID, "radiation_flux_LW2", 0.f);
    context.setPrimitiveData(UUID, "air_temperature", 300.f);
    context.setPrimitiveData(UUID, "boundarylayer_conductance", 0.f);
    context.setPrimitiveData(UUID, "emissivity_LW", 0.8f);
    context.setPrimitiveData(UUID, "emissivity_LW2", 0.4f);
    context.setGlobalData("emission_enabled_LW", uint(1));
    context.setGlobalData("emission_enabled_LW2", uint(1));

    std::string output;
    {
        capture_cout capture;
        EnergyBalanceModel model(&context);
        model.addRadiationBand("LW");
        model.addRadiationBand("LW2");
        model.run();
        output = capture.get_captured_output();
    }

    float T;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "temperature", T));
    float T_expected = float(pow(R / (2.f * 0.8f * 5.67e-8f), 0.25));
    DOCTEST_CHECK(output.find("multiple_emission_bands") != std::string::npos);
    DOCTEST_CHECK(T == doctest::Approx(T_expected).epsilon(err_tol));
}

DOCTEST_TEST_CASE("EnergyBalanceModel emissivity defined on multiple bands without emission info warns") {
    // Backward-compatible fallback: when no emission-enabled information is available (radiation fluxes set
    // manually, no RadiationModel run), all bands are emissivity candidates. Defining emissivity on more
    // than one band is ambiguous and is warned about; the first band's value is used.
    Context context;
    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    context.setPrimitiveData(UUID, "radiation_flux_LW", 400.f);
    context.setPrimitiveData(UUID, "radiation_flux_SW", 300.f);
    context.setPrimitiveData(UUID, "air_temperature", 300.f);
    context.setPrimitiveData(UUID, "emissivity_LW", 0.95f);
    context.setPrimitiveData(UUID, "emissivity_SW", 0.5f);

    std::string output;
    {
        capture_cout capture;
        EnergyBalanceModel model(&context);
        model.addRadiationBand("LW");
        model.addRadiationBand("SW");
        model.run();
        output = capture.get_captured_output();
    }
    DOCTEST_CHECK(output.find("multiple_emissivity_bands") != std::string::npos);
}

int EnergyBalanceModel::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

DOCTEST_TEST_CASE("EnergyBalanceModel surface humidity scales surface vapor pressure") {
    // Regression test: "surface_humidity" (f_s) must multiply the SURFACE saturation vapor pressure
    // e_s(T_s), not the air vapor pressure e_s(T_a)*h. The buggy form computed
    // (e_s(T_s) - e_s(T_a)*h*f_s), so drying the surface RAISED the vapor-pressure difference and
    // increased evaporation, which is backwards.
    //
    // Because surface_humidity defaults to 1, both forms agree at the default and the error was
    // invisible until a user set f_s < 1.
    //
    // The observable used here is the solved surface temperature, which is driven by the energy
    // balance residual where the bug lives. Drying the surface suppresses evaporation, so it must
    // remove evaporative cooling and WARM the surface. The buggy form cooled it instead:
    //   fixed  288.68 K -> 295.55 K (warms, correct)
    //   buggy  288.68 K -> 283.54 K (cools, backwards)
    Context context;
    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    context.setPrimitiveData(UUID, "air_temperature", 300.f);
    context.setPrimitiveData(UUID, "air_humidity", 0.5f);
    context.setPrimitiveData(UUID, "boundarylayer_conductance", 1.f);
    context.setPrimitiveData(UUID, "moisture_conductance", 1.f);
    context.setPrimitiveData(UUID, "radiation_flux_LW", 459.3f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("LW");

    // Saturated surface (f_s = 1, the default) -- unaffected by the fix.
    context.setPrimitiveData(UUID, "surface_humidity", 1.f);
    DOCTEST_CHECK_NOTHROW(model.run());
    float T_saturated;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "temperature", T_saturated));

    // Dry surface (f_s = 0.2).
    context.setPrimitiveData(UUID, "surface_humidity", 0.2f);
    DOCTEST_CHECK_NOTHROW(model.run());
    float T_dry;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "temperature", T_dry));

    DOCTEST_CHECK(std::isfinite(T_saturated));
    DOCTEST_CHECK(std::isfinite(T_dry));
    // A surface that cannot evaporate freely loses less latent heat and must be warmer, not cooler.
    DOCTEST_CHECK(T_dry > T_saturated);
    // Guard the magnitude too, so a merely-marginal difference cannot pass.
    DOCTEST_CHECK(T_dry - T_saturated > 1.f);
}

// ---- Canopy airspace model ---- //

//! Build a simple layered canopy of horizontal patches spanning a 1 m x 1 m ground area
/**
 * \param[in] context Context to add the canopy to.
 * \param[in] patches_per_layer Number of patches at each height level.
 * \param[in] num_levels Number of discrete height levels between the ground and the canopy top.
 * \param[in] canopy_height_m Height of the canopy top in meters.
 * \param[in] patch_size_m Edge length of each square patch in meters.
 * \return UUIDs of the canopy patches.
 */
static std::vector<uint> buildTestCanopy(Context &context, uint patches_per_layer, uint num_levels, float canopy_height_m, float patch_size_m) {
    std::vector<uint> UUIDs;
    for (uint level = 0; level < num_levels; level++) {
        // Distribute levels evenly through the canopy depth, keeping them strictly inside the canopy.
        float z = canopy_height_m * (float(level) + 0.5f) / float(num_levels);
        for (uint p = 0; p < patches_per_layer; p++) {
            float x = 0.1f * float(p);
            UUIDs.push_back(context.addPatch(make_vec3(x, 0, z), make_vec2(patch_size_m, patch_size_m)));
        }
    }
    return UUIDs;
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Argument Validation") {
    Context context;
    std::vector<uint> UUIDs = buildTestCanopy(context, 2, 2, 2.f, 0.1f);

    EnergyBalanceModel model(&context);
    model.disableMessages();

    // Empty canopy set is an error, since there would be nothing exchanging heat with the air.
    DOCTEST_CHECK_THROWS(model.enableCanopyAirspaceModel({}, {}, 2.f, 3.f, 2.f, 1));
    // Canopy height must be positive.
    DOCTEST_CHECK_THROWS(model.enableCanopyAirspaceModel(UUIDs, {}, 0.f, 3.f, 2.f, 1));
    // Reference height must be above the canopy top.
    DOCTEST_CHECK_THROWS(model.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 2.f, 2.f, 1));
    DOCTEST_CHECK_THROWS(model.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 1.f, 2.f, 1));
    // Leaf area index must be positive.
    DOCTEST_CHECK_THROWS(model.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 3.f, 0.f, 1));
    // At least one layer is required.
    DOCTEST_CHECK_THROWS(model.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 3.f, 2.f, 0));
    // A UUID that does not exist in the Context is an error.
    DOCTEST_CHECK_THROWS(model.enableCanopyAirspaceModel({999999}, {}, 2.f, 3.f, 2.f, 1));

    // Valid arguments must be accepted.
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 3.f, 2.f, 1));

    // Convergence criteria must be positive.
    DOCTEST_CHECK_THROWS(model.setCanopyAirspaceConvergence(0.f, 10));
    DOCTEST_CHECK_THROWS(model.setCanopyAirspaceConvergence(-1.f, 10));
    DOCTEST_CHECK_THROWS(model.setCanopyAirspaceConvergence(0.01f, 0));
    DOCTEST_CHECK_NOTHROW(model.setCanopyAirspaceConvergence(0.01f, 10));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Equilibrium") {
    // With the canopy in radiative equilibrium at the reference air temperature, there is no sensible or latent
    // source, so the airspace must recover the reference air temperature exactly.
    Context context;
    float Tref = 300.f;
    float canopy_height = 2.f;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 3, canopy_height, 0.1f);

    // Longwave flux that exactly balances emission at Tref, and closed stomata so there is no latent flux.
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 2.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 2.f, 1));
    DOCTEST_CHECK_NOTHROW(model.run());

    float T_canopy_air;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature", T_canopy_air));
    DOCTEST_CHECK(T_canopy_air == doctest::Approx(Tref).epsilon(err_tol));

    // Every primitive must have been given the airspace temperature.
    for (uint UUID: UUIDs) {
        float Ta;
        DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "air_temperature", Ta));
        DOCTEST_CHECK(Ta == doctest::Approx(Tref).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Heating and Cooling") {
    // A canopy heated well above the reference air temperature must warm the within-canopy air, and the
    // within-canopy air must remain between the reference air and the leaf temperature.
    Context context;
    float Tref = 300.f;
    float canopy_height = 2.f;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 3, canopy_height, 0.1f);

    // Strong shortwave load with closed stomata forces the leaves well above air temperature.
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 600.f);
    // Incoming longwave that balances emission at Tref, so shortwave is a true net heat load.
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 3.f, 1));
    DOCTEST_CHECK_NOTHROW(model.run());

    float T_canopy_air;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature", T_canopy_air));

    float T_leaf;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUIDs.at(0), "temperature", T_leaf));

    // The heated canopy must warm the air it sits in.
    DOCTEST_CHECK(T_canopy_air > Tref);
    // But the air cannot be warmer than the leaves that are heating it.
    DOCTEST_CHECK(T_canopy_air < T_leaf);
    // The effect must be substantial, not marginal, under this radiation load.
    DOCTEST_CHECK(T_canopy_air - Tref > 0.5f);
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Transpiration Feedback") {
    // A transpiring canopy must humidify and cool the within-canopy air relative to a non-transpiring one.
    // This is the feedback the model exists to represent.
    Context context;
    float Tref = 305.f;
    float canopy_height = 2.f;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 3, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 400.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.4f);
    context.setGlobalData("wind_speed_reference", 1.f);

    // Case 1: closed stomata, no transpiration.
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    EnergyBalanceModel model_dry(&context);
    model_dry.disableMessages();
    model_dry.addRadiationBand("SW");
    model_dry.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model_dry.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 3.f, 1));
    DOCTEST_CHECK_NOTHROW(model_dry.run());

    float T_air_dry, humidity_dry;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature", T_air_dry));
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_humidity", humidity_dry));

    // Case 2: open stomata, active transpiration. Clear the warm-start so both cases begin identically.
    context.setGlobalData("canopy_air_temperature", Tref);
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.3f);
    EnergyBalanceModel model_wet(&context);
    model_wet.disableMessages();
    model_wet.addRadiationBand("SW");
    model_wet.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model_wet.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 3.f, 1));
    DOCTEST_CHECK_NOTHROW(model_wet.run());

    float T_air_wet, humidity_wet;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature", T_air_wet));
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_humidity", humidity_wet));

    // Transpiration must cool the within-canopy air relative to the non-transpiring case.
    DOCTEST_CHECK(T_air_wet < T_air_dry);
    // Transpiration must humidify the within-canopy air.
    DOCTEST_CHECK(humidity_wet > humidity_dry);
    // Humidity must remain physical.
    DOCTEST_CHECK(humidity_wet <= 1.f);
    DOCTEST_CHECK(humidity_wet > 0.f);
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Multilayer Gradient") {
    // A heated canopy ventilated from above must produce a vertical gradient in which the air is warmest at
    // the bottom, furthest from the fixed reference boundary condition at the canopy top.
    Context context;
    float Tref = 300.f;
    float canopy_height = 4.f;
    uint num_layers = 5;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 10, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 500.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 6.f, 3.f, num_layers));
    DOCTEST_CHECK_NOTHROW(model.run());

    std::vector<float> T_layers;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature_layers", T_layers));
    DOCTEST_CHECK(T_layers.size() == num_layers);

    // Layer 0 is the bottom layer. Air temperature must decrease monotonically toward the canopy top, where
    // it is tied to the reference air.
    for (uint j = 0; j + 1 < num_layers; j++) {
        DOCTEST_CHECK(T_layers.at(j) > T_layers.at(j + 1));
    }
    // Every layer must be warmer than the reference air, since the canopy is a heat source throughout.
    for (uint j = 0; j < num_layers; j++) {
        DOCTEST_CHECK(T_layers.at(j) > Tref);
    }
    // The gradient must be resolvable, not numerical noise.
    DOCTEST_CHECK(T_layers.front() - T_layers.back() > 0.1f);
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Wind Profile") {
    // The Cionco profile must attenuate wind speed with depth into the canopy, and must reach the
    // canopy-top wind speed at the canopy top.
    Context context;
    float canopy_height = 4.f;
    float LAI = 3.f;

    // Two patches at known heights: one near the canopy top, one near the ground.
    uint UUID_top = context.addPatch(make_vec3(0, 0, 4.f), make_vec2(0.1f, 0.1f));
    uint UUID_bottom = context.addPatch(make_vec3(0, 0, 0.f), make_vec2(0.1f, 0.1f));
    std::vector<uint> UUIDs = {UUID_top, UUID_bottom};

    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 100.f);
    context.setGlobalData("air_temperature_reference", 300.f);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 3.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 6.f, LAI, 1));
    DOCTEST_CHECK_NOTHROW(model.run());

    float U_top, U_bottom;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_top, "wind_speed", U_top));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_bottom, "wind_speed", U_bottom));

    // Wind must be attenuated with depth into the canopy.
    DOCTEST_CHECK(U_bottom < U_top);
    // The attenuation over the full canopy depth must follow exp(-a) with a = LAI/2.
    DOCTEST_CHECK(U_bottom / U_top == doctest::Approx(std::exp(-0.5f * LAI)).epsilon(0.01f));
    DOCTEST_CHECK(U_bottom > 0.f);
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Aerodynamic Resistance Scaling") {
    // Aerodynamic resistance is inversely proportional to wind speed, so doubling the reference wind speed
    // must halve the resistance, and a better-ventilated canopy must sit closer to the reference air.
    Context context;
    float Tref = 300.f;
    float canopy_height = 2.f;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 3, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 500.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 2.f, 1));

    context.setGlobalData("wind_speed_reference", 1.f);
    DOCTEST_CHECK_NOTHROW(model.run());
    float ra_slow, T_air_slow;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("aerodynamic_resistance", ra_slow));
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature", T_air_slow));

    context.setGlobalData("wind_speed_reference", 2.f);
    DOCTEST_CHECK_NOTHROW(model.run());
    float ra_fast, T_air_fast;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("aerodynamic_resistance", ra_fast));
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature", T_air_fast));

    // Doubling wind speed must halve the aerodynamic resistance.
    DOCTEST_CHECK(ra_fast == doctest::Approx(0.5f * ra_slow).epsilon(err_tol));
    DOCTEST_CHECK(ra_slow > 0.f);
    // Better ventilation must bring the within-canopy air closer to the reference air.
    DOCTEST_CHECK(T_air_fast - Tref < T_air_slow - Tref);
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Soil Node") {
    // The soil exchanges heat with the lowest canopy layer, and its own surface energy balance must be driven by
    // that layer's air rather than by the air above the canopy. A strongly heated soil must therefore warm the
    // bottom of the airspace, and the soil surface must equilibrate with the canopy air rather than the reference.
    Context context;
    float Tref = 300.f;
    float canopy_height = 2.f;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 4, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 200.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.f);

    // Run without a soil node.
    EnergyBalanceModel model_nosoil(&context);
    model_nosoil.disableMessages();
    model_nosoil.addRadiationBand("SW");
    model_nosoil.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model_nosoil.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 2.f, 3));
    DOCTEST_CHECK_NOTHROW(model_nosoil.run(UUIDs));
    std::vector<float> T_layers_nosoil;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature_layers", T_layers_nosoil));

    // Add a strongly heated ground surface and run with a soil node. The ground absorbs a large radiation load, so
    // it acts as a heat source into the bottom of the canopy airspace.
    uint UUID_ground = context.addPatch(make_vec3(0, 0, 0), make_vec2(1.f, 1.f));
    context.setPrimitiveData(UUID_ground, "radiation_flux_SW", 800.f);
    context.setPrimitiveData(UUID_ground, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUID_ground, "twosided_flag", uint(0));
    context.setGlobalData("canopy_air_temperature_layers", std::vector<float>{});

    std::vector<uint> UUIDs_with_ground = UUIDs;
    UUIDs_with_ground.push_back(UUID_ground);

    EnergyBalanceModel model_soil(&context);
    model_soil.disableMessages();
    model_soil.addRadiationBand("SW");
    model_soil.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model_soil.enableCanopyAirspaceModel(UUIDs, {UUID_ground}, canopy_height, 4.f, 2.f, 3));
    DOCTEST_CHECK_NOTHROW(model_soil.run(UUIDs_with_ground));
    std::vector<float> T_layers_soil;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature_layers", T_layers_soil));

    // The heated soil must warm the bottom layer.
    DOCTEST_CHECK(T_layers_soil.at(0) > T_layers_nosoil.at(0));
    DOCTEST_CHECK(T_layers_soil.at(0) - T_layers_nosoil.at(0) > 0.5f);

    // The soil must have been driven by the bottom layer's air, not by the reference air. Its air temperature
    // primitive data must equal the bottom layer value that the model solved.
    float T_air_ground;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_ground, "air_temperature", T_air_ground));
    DOCTEST_CHECK(T_air_ground == doctest::Approx(T_layers_soil.at(0)).epsilon(err_tol));
    DOCTEST_CHECK(T_air_ground != doctest::Approx(Tref).epsilon(err_tol));

    // The ground must also receive the within-canopy wind speed, which is attenuated well below the reference.
    float U_ground;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_ground, "wind_speed", U_ground));
    DOCTEST_CHECK(U_ground > 0.f);
    DOCTEST_CHECK(U_ground < 1.f);
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Requires Primitives In Run Set") {
    // Every primitive exchanging with the airspace must also have its surface energy balance solved, otherwise it
    // would be driven by an air state that nothing recomputes.
    Context context;
    float Tref = 300.f;
    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 3, 2.f, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 200.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.f);

    uint UUID_ground = context.addPatch(make_vec3(0, 0, 0), make_vec2(1.f, 1.f));
    context.setPrimitiveData(UUID_ground, "radiation_flux_SW", 200.f);
    context.setPrimitiveData(UUID_ground, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUID_ground, "temperature", 310.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {UUID_ground}, 2.f, 4.f, 3.f, 2));

    // Ground is part of the airspace but excluded from the primitives being solved.
    DOCTEST_CHECK_THROWS(model.run(UUIDs));

    // Including it must work.
    std::vector<uint> UUIDs_all = UUIDs;
    UUIDs_all.push_back(UUID_ground);
    DOCTEST_CHECK_NOTHROW(model.run(UUIDs_all));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Dynamic Mode Rejected") {
    // The airspace model solves a steady state and must refuse the dynamic form of run() rather than
    // silently ignoring the timestep.
    Context context;
    std::vector<uint> UUIDs = buildTestCanopy(context, 2, 2, 2.f, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 100.f);
    context.setGlobalData("air_temperature_reference", 300.f);
    context.setGlobalData("wind_speed_reference", 1.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 4.f, 2.f, 1));

    DOCTEST_CHECK_THROWS(model.run(10.f));

    // After disabling, the dynamic form must work again.
    DOCTEST_CHECK_NOTHROW(model.disableCanopyAirspaceModel());
    DOCTEST_CHECK_NOTHROW(model.run(10.f));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Zero Wind Rejected") {
    // Aerodynamic resistance is inversely proportional to wind speed, so a zero reference wind speed must
    // produce a clear error rather than a division by zero.
    Context context;
    std::vector<uint> UUIDs = buildTestCanopy(context, 2, 2, 2.f, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 100.f);
    context.setGlobalData("air_temperature_reference", 300.f);
    context.setGlobalData("wind_speed_reference", 0.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 4.f, 2.f, 1));
    DOCTEST_CHECK_THROWS(model.run());
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Empty Layer Regression") {
    // Requesting more layers than there are canopy primitives, or a strongly non-uniform distribution of leaf
    // area, must not silently produce layers containing no leaf area. Such a layer contributes no source term
    // and the solver would interpolate a temperature through it without any indication to the user.
    Context context;
    float Tref = 300.f;
    float canopy_height = 2.f;

    // Four primitives but ten requested layers: at most one layer can be started per primitive.
    std::vector<uint> UUIDs = buildTestCanopy(context, 1, 4, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 400.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");

    // Ten layers cannot be filled by four primitives. Layer assignment happens when the model is run, so the
    // error surfaces there rather than at configuration time.
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 3.f, 10));
    DOCTEST_CHECK_THROWS(model.run());

    // A layer count the geometry can support must still work.
    EnergyBalanceModel model_ok(&context);
    model_ok.disableMessages();
    model_ok.addRadiationBand("SW");
    model_ok.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model_ok.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 3.f, 2));
    DOCTEST_CHECK_NOTHROW(model_ok.run());
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Forces Conductance Output") {
    // The airspace solution weights leaves by their boundary-layer conductance, so enabling the model must add
    // 'boundarylayer_conductance_out' to the outputs. Boundary-layer conductance is set explicitly here so that
    // the separate path which enables this output when the conductance is calculated internally does not fire.
    Context context;
    float Tref = 300.f;
    float canopy_height = 2.f;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 3, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 300.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "boundarylayer_conductance", 1.5f);
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.2f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 4.f, 3.f, 1));
    DOCTEST_CHECK_NOTHROW(model.run());

    for (uint UUID: UUIDs) {
        DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out"));
        float g_bl;
        DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance_out", g_bl));
        DOCTEST_CHECK(g_bl == doctest::Approx(1.5f).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Mutually Exclusive With Air Energy Balance") {
    // Both models determine the within-canopy air state and share the canopy and reference heights, so enabling
    // both would silently let one overwrite the other's geometry.
    Context context;
    std::vector<uint> UUIDs = buildTestCanopy(context, 2, 2, 2.f, 0.1f);

    EnergyBalanceModel model_a(&context);
    model_a.disableMessages();
    DOCTEST_CHECK_NOTHROW(model_a.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 4.f, 2.f, 1));
    DOCTEST_CHECK_THROWS(model_a.enableAirEnergyBalance(2.f, 4.f));

    EnergyBalanceModel model_b(&context);
    model_b.disableMessages();
    DOCTEST_CHECK_NOTHROW(model_b.enableAirEnergyBalance(2.f, 4.f));
    DOCTEST_CHECK_THROWS(model_b.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 4.f, 2.f, 1));

    // Disabling the airspace model must release the restriction.
    EnergyBalanceModel model_c(&context);
    model_c.disableMessages();
    DOCTEST_CHECK_NOTHROW(model_c.enableCanopyAirspaceModel(UUIDs, {}, 2.f, 4.f, 2.f, 1));
    DOCTEST_CHECK_NOTHROW(model_c.disableCanopyAirspaceModel());
    DOCTEST_CHECK_NOTHROW(model_c.enableAirEnergyBalance(2.f, 4.f));
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Rejects Missing Soil Temperature") {
    // A ground primitive without 'temperature' data yields an area-weighted sum of zero, which would drag the
    // bottom canopy layer toward absolute zero. That must be an error, not a silently wrong profile.
    Context context;
    float Tref = 300.f;
    float canopy_height = 2.f;

    std::vector<uint> UUIDs = buildTestCanopy(context, 4, 3, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 300.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.f);

    // Ground primitive deliberately left without 'temperature' primitive data.
    uint UUID_ground = context.addPatch(make_vec3(0, 0, 0), make_vec2(1.f, 1.f));

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {UUID_ground}, canopy_height, 4.f, 3.f, 2));
    DOCTEST_CHECK_THROWS(model.run());
}

DOCTEST_TEST_CASE("EnergyBalanceModel Canopy Airspace Eddy Diffusivity Decay") {
    // Turbulent mixing is damped with depth into the canopy, so the conductance between layers must decrease
    // downward. The observable consequence is that the temperature gradient is concentrated near the bottom of the
    // canopy, where the air is least well ventilated, rather than being spread evenly across the layers.
    Context context;
    float Tref = 300.f;
    float canopy_height = 4.f;
    uint num_layers = 8;

    // Leaf area is spread uniformly with height so that the layers have near-equal thickness. Any asymmetry in the
    // resulting profile therefore comes from the conductance profile, not from the layer geometry.
    std::vector<uint> UUIDs = buildTestCanopy(context, 6, 24, canopy_height, 0.1f);
    context.setPrimitiveData(UUIDs, "radiation_flux_SW", 500.f);
    context.setPrimitiveData(UUIDs, "radiation_flux_LW", float(2.f * 5.67e-8 * pow(Tref, 4)));
    context.setPrimitiveData(UUIDs, "moisture_conductance", 0.f);
    context.setGlobalData("air_temperature_reference", Tref);
    context.setGlobalData("air_humidity_reference", 0.5f);
    context.setGlobalData("wind_speed_reference", 1.5f);

    EnergyBalanceModel model(&context);
    model.disableMessages();
    model.addRadiationBand("SW");
    model.addRadiationBand("LW");
    DOCTEST_CHECK_NOTHROW(model.enableCanopyAirspaceModel(UUIDs, {}, canopy_height, 6.f, 3.f, num_layers));
    DOCTEST_CHECK_NOTHROW(model.run());

    std::vector<float> T_layers;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("canopy_air_temperature_layers", T_layers));
    DOCTEST_CHECK(T_layers.size() == num_layers);

    // Temperature must still decrease upward everywhere.
    for (uint j = 0; j + 1 < num_layers; j++) {
        DOCTEST_CHECK(T_layers.at(j) > T_layers.at(j + 1));
    }

    // With conductance decaying downward, the temperature step across the lowest layer boundary must be larger than
    // the step across the highest one. A height-independent conductance would make these steps comparable.
    // Each layer boundary carries the heat released by every layer below it, so the flux through the topmost
    // boundary is the largest. Conductance decaying downward partly offsets this, but the flux dominates, and the
    // resulting temperature steps grow toward the canopy top.
    float step_bottom = T_layers.at(0) - T_layers.at(1);
    float step_top = T_layers.at(num_layers - 2) - T_layers.at(num_layers - 1);
    DOCTEST_CHECK(step_top > step_bottom);

    // The conductance profile must remain physical: every layer is warmer than the reference air, and the total
    // departure is bounded well away from a runaway value.
    DOCTEST_CHECK(T_layers.back() > Tref);
    DOCTEST_CHECK(T_layers.front() - Tref < 30.f);
}
