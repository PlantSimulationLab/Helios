#include "BoundaryLayerConductanceModel.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

float err_tol = 1e-3;

DOCTEST_TEST_CASE("BLConductanceModel check blc values") {

    Context context;
    float TL = 300;
    float Ta = 290;
    float U = 0.6;

    std::vector<uint> UUID_1;
    UUID_1.push_back(context.addPatch());
    UUID_1.push_back(context.addPatch());
    UUID_1.push_back(context.addPatch());
    UUID_1.push_back(context.addPatch());

    context.setPrimitiveData(UUID_1, "air_temperature", Ta);
    context.setPrimitiveData(UUID_1, "temperature", TL);
    context.setPrimitiveData(UUID_1, "wind_speed", U);

    // Instantiate the boundary layer conductance model
    BLConductanceModel blc(&context);
    blc.disableMessages();

    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(0), "Pohlhausen"));
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(1), "InclinedPlate"));
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(2), "Sphere"));
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(3), "Ground"));

    capture_cerr cerr_buffer;
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(3), "InvalidModel"));
    DOCTEST_CHECK(cerr_buffer.has_output());

    DOCTEST_CHECK_NOTHROW(blc.run());

    std::vector<float> gH(UUID_1.size());

    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(0), "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(1), "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(2), "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(3), "boundarylayer_conductance"));

    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(0), "boundarylayer_conductance", gH.at(0)));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(1), "boundarylayer_conductance", gH.at(1)));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(2), "boundarylayer_conductance", gH.at(2)));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(3), "boundarylayer_conductance", gH.at(3)));

    DOCTEST_CHECK(gH.at(0) == doctest::Approx(0.20914112f).epsilon(err_tol));
    DOCTEST_CHECK(gH.at(1) == doctest::Approx(0.135347f).epsilon(err_tol));
    DOCTEST_CHECK(gH.at(2) == doctest::Approx(0.087149f).epsilon(err_tol));
    DOCTEST_CHECK(gH.at(3) == doctest::Approx(0.465472f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("Default values handling") {
    Context context;
    uint UUID_default = context.addPatch();

    BLConductanceModel blc(&context);
    blc.disableMessages();
    blc.setBoundaryLayerModel(UUID_default, "Pohlhausen");
    blc.run();

    float default_gH = -1.f;
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_default, "boundarylayer_conductance"));
    context.getPrimitiveData(UUID_default, "boundarylayer_conductance", default_gH);
    DOCTEST_CHECK(default_gH > 0);
}

DOCTEST_TEST_CASE("Single-sided primitive handling") {
    Context context;
    float TL = 300;
    float Ta = 290;
    float U = 0.6;

    uint UUID = context.addPatch();
    context.setPrimitiveData(UUID, "air_temperature", Ta);
    context.setPrimitiveData(UUID, "temperature", TL);
    context.setPrimitiveData(UUID, "wind_speed", U);
    context.setPrimitiveData(UUID, "twosided_flag", 0u); // single-sided

    BLConductanceModel blc(&context);
    blc.disableMessages();
    blc.setBoundaryLayerModel(UUID, "Pohlhausen");
    blc.run();

    float gH = 0.0f;
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID, "boundarylayer_conductance"));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH == doctest::Approx(0.104571f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("Message flag tests") {
    Context context;
    BLConductanceModel blc(&context);

    DOCTEST_CHECK_NOTHROW(blc.enableMessages());

    DOCTEST_CHECK_NOTHROW(blc.disableMessages());
}

DOCTEST_TEST_CASE("setBoundaryLayerModel for all primitives") {
    Context context;
    uint u1 = context.addPatch();
    uint u2 = context.addPatch();

    BLConductanceModel blc(&context);
    blc.disableMessages();

    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel("Pohlhausen"));

    context.setPrimitiveData({u1, u2}, "air_temperature", 290.f);
    context.setPrimitiveData({u1, u2}, "temperature", 300.f);
    context.setPrimitiveData({u1, u2}, "wind_speed", 0.6f);

    blc.run();

    float gH1, gH2;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u1, "boundarylayer_conductance", gH1));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u2, "boundarylayer_conductance", gH2));
    DOCTEST_CHECK(gH1 == doctest::Approx(0.20914112f).epsilon(err_tol));
    DOCTEST_CHECK(gH2 == doctest::Approx(0.20914112f).epsilon(err_tol));

    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel("InclinedPlate"));
    blc.run();
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u1, "boundarylayer_conductance", gH1));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u2, "boundarylayer_conductance", gH2));
    DOCTEST_CHECK(gH1 == doctest::Approx(0.135347f).epsilon(err_tol));
    DOCTEST_CHECK(gH2 == doctest::Approx(0.135347f).epsilon(err_tol));

    capture_cerr cerr_buffer;
    blc.enableMessages();
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel("InvalidModel"));
    DOCTEST_CHECK(cerr_buffer.has_output());
}

DOCTEST_TEST_CASE("Subset run") {
    Context context;
    float TL = 300;
    float Ta = 290;
    float U = 0.6;

    std::vector<uint> UUIDs;
    UUIDs.push_back(context.addPatch());
    UUIDs.push_back(context.addPatch());
    UUIDs.push_back(context.addPatch());

    context.setPrimitiveData(UUIDs, "air_temperature", Ta);
    context.setPrimitiveData(UUIDs, "temperature", TL);
    context.setPrimitiveData(UUIDs, "wind_speed", U);

    BLConductanceModel blc(&context);
    blc.disableMessages();

    blc.setBoundaryLayerModel(UUIDs[0], "Pohlhausen");
    blc.setBoundaryLayerModel(UUIDs[1], "InclinedPlate");
    blc.setBoundaryLayerModel(UUIDs[2], "Sphere");

    std::vector<uint> subsetUUIDs = {UUIDs[0], UUIDs[2]};
    DOCTEST_CHECK_NOTHROW(blc.run(subsetUUIDs));

    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUIDs[0], "boundarylayer_conductance"));
    DOCTEST_CHECK(!context.doesPrimitiveDataExist(UUIDs[1], "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUIDs[2], "boundarylayer_conductance"));

    float gH0, gH2;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUIDs[0], "boundarylayer_conductance", gH0));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUIDs[2], "boundarylayer_conductance", gH2));

    DOCTEST_CHECK(gH0 == doctest::Approx(0.20914112f).epsilon(err_tol));
    DOCTEST_CHECK(gH2 == doctest::Approx(0.087149f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("Missing parameters and zero inputs") {
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    uint UUID = context.addPatch();
    context.setPrimitiveData(UUID, "atmospheric_pressure", 100000.f);
    context.setPrimitiveData(UUID, "object_length", 0.1f);
    context.setPrimitiveData(UUID, "inclination", 0.5f);

    // Run with default air_temperature, temperature, and wind_speed
    DOCTEST_CHECK_NOTHROW(blc.run());
    float gH;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0);

    // Test zero characteristic dimension
    context.setPrimitiveData(UUID, "object_length", 0.f);
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH != 0.f);

    // Test zero wind speed
    context.setPrimitiveData(UUID, "object_length", 0.1f);
    context.setPrimitiveData(UUID, "wind_speed", 0.f);

    blc.setBoundaryLayerModel(UUID, "Pohlhausen");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH == 0.f);

    blc.setBoundaryLayerModel(UUID, "InclinedPlate");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0);

    blc.setBoundaryLayerModel(UUID, "Sphere");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0.f);

    blc.setBoundaryLayerModel(UUID, "Ground");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0.f);
}

DOCTEST_TEST_CASE("InclinedPlate free convection at zero wind is in molar units") {
    // Regression test: the zero-wind (pure free convection) branch of the InclinedPlate model returned
    // Nu*alpha/L, a velocity in m/s, while the forced-convection branch and the documented formula
    // multiply by the molar density of air (41.4 mol/m^3) to give mol/m^2/s. At zero wind the
    // conductance therefore came out ~41x too small.
    //
    // Hand calculation with the code's constants for a horizontal patch (inclination 0):
    //   Ta = 300 K, TL = 305 K, L = 0.1 m, nu = 1.568e-5 m^2/s, Pr = 0.7, alpha = nu/Pr = 2.24e-5 m^2/s
    //   Gr = 9.81 * |TL - Ta| * L^3 / (Ta * nu^2) = 9.81 * 5 * 1e-3 / (300 * 2.4586e-10) = 6.650e5
    //   Nu = 0.54 * (Gr * cos(0))^(1/4) = 0.54 * 28.56 = 15.42
    //   gH = 41.4 * alpha / L * Nu = 41.4 * 2.24e-4 * 15.42 = 0.1430 mol/m^2/s   (buggy code: 0.00345)
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    const float Ta = 300.f;
    const float TL = 305.f;
    const float L = 0.1f;
    const float nu = 1.568e-5f;
    const float Pr = 0.7f;
    const float alpha = nu / Pr;
    const float Gr = 9.81f * fabsf(TL - Ta) * L * L * L / (Ta * nu * nu);
    const float Nu_free = 0.54f * powf(Gr, 0.25f);
    const float gH_expected = 41.4f * alpha / L * Nu_free;
    DOCTEST_CHECK(gH_expected == doctest::Approx(0.1430f).epsilon(0.01));

    uint UUID = context.addPatch(); // normal +z -> inclination 0 (horizontal plate branch)
    context.setPrimitiveData(UUID, "object_length", L);
    context.setPrimitiveData(UUID, "air_temperature", Ta);
    context.setPrimitiveData(UUID, "temperature", TL);
    context.setPrimitiveData(UUID, "wind_speed", 0.f);
    blc.setBoundaryLayerModel(UUID, "InclinedPlate");

    DOCTEST_CHECK_NOTHROW(blc.run());
    float gH_free;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH_free));
    DOCTEST_CHECK(gH_free == doctest::Approx(gH_expected).epsilon(1e-3));

    // Continuity across the free/forced switch: a very small wind speed (Re ~ 320) takes the mixed
    // free-forced branch, which for these conditions gives ~0.15 mol/m^2/s. The pure free-convection
    // value must be of the same order; on the buggy code the ratio was ~43.
    context.setPrimitiveData(UUID, "wind_speed", 0.05f);
    DOCTEST_CHECK_NOTHROW(blc.run());
    float gH_forced;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH_forced));
    DOCTEST_CHECK(gH_forced > 0.1f);
    DOCTEST_CHECK(gH_forced / gH_free > 0.5f);
    DOCTEST_CHECK(gH_forced / gH_free < 2.f);
}

DOCTEST_TEST_CASE("InclinedPlate zero-wind conductance and its jump from the low-wind limit") {
    // At zero wind the InclinedPlate model switches from the mixed-convection correlations to the McAdams free-convection
    // correlation, which is not their limit as the wind speed tends to zero, so the conductance jumps at zero wind
    // (described in BLConductance.dox). This pins the zero-wind value of the vertical-plate branch, which no other test
    // covers, and the size of the jump for both plate orientations.
    //
    // Hand calculation with the code's constants, L = 0.1 m, Ta = 300 K, TL = 305 K: 41.4 * alpha / L = 9.274e-3 mol/m^2/s
    // and Gr = 6.650e5 (see the test above).
    //   vertical, U = 0:     Nu = 0.59 Gr^(1/4) = 16.85                          -> gH = 0.1563
    //   vertical, U -> 0:    Nu = F3 Gr^(1/5) / (3 * 0.2) = 8.655, F3 = 0.3554    -> gH = 0.0803 (zero-wind value 1.95x larger)
    //   horizontal, U = 0:   Nu = 0.54 Gr^(1/4) = 15.42                          -> gH = 0.1430
    //   horizontal, U -> 0:  Nu = (4/3) F2 Gr^(1/4) = 13.37, F2 = 0.3513         -> gH = 0.1240 (zero-wind value 1.15x larger)
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    const uint horizontal_UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    const uint vertical_UUID = context.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));
    context.rotatePrimitive(vertical_UUID, 0.5f * PI_F, "x");
    DOCTEST_REQUIRE(cart2sphere(context.getPrimitiveNormal(horizontal_UUID)).zenith == doctest::Approx(0.f).epsilon(1e-4));
    DOCTEST_REQUIRE(cart2sphere(context.getPrimitiveNormal(vertical_UUID)).zenith == doctest::Approx(0.5f * PI_F).epsilon(1e-4));

    const std::vector<uint> UUIDs = {horizontal_UUID, vertical_UUID};
    context.setPrimitiveData(UUIDs, "object_length", 0.1f);
    context.setPrimitiveData(UUIDs, "air_temperature", 300.f);
    context.setPrimitiveData(UUIDs, "temperature", 305.f);
    blc.setBoundaryLayerModel(UUIDs, "InclinedPlate");

    auto conductanceAtWindSpeed = [&](float wind_speed, float &gH_horizontal, float &gH_vertical) {
        context.setPrimitiveData(UUIDs, "wind_speed", wind_speed);
        blc.run();
        context.getPrimitiveData(horizontal_UUID, "boundarylayer_conductance", gH_horizontal);
        context.getPrimitiveData(vertical_UUID, "boundarylayer_conductance", gH_vertical);
    };

    float gH_horizontal_zero, gH_vertical_zero;
    conductanceAtWindSpeed(0.f, gH_horizontal_zero, gH_vertical_zero);
    DOCTEST_CHECK(gH_horizontal_zero == doctest::Approx(0.1430f).epsilon(0.01));
    DOCTEST_CHECK(gH_vertical_zero == doctest::Approx(0.1563f).epsilon(0.01));

    // Re = 0.64, far below any leaf Reynolds number but large enough to take the mixed-convection branch.
    float gH_horizontal_low, gH_vertical_low;
    conductanceAtWindSpeed(1e-4f, gH_horizontal_low, gH_vertical_low);
    DOCTEST_CHECK(gH_horizontal_low == doctest::Approx(0.1240f).epsilon(0.01));
    DOCTEST_CHECK(gH_vertical_low == doctest::Approx(0.0803f).epsilon(0.01));
    DOCTEST_CHECK(gH_horizontal_zero / gH_horizontal_low == doctest::Approx(1.153f).epsilon(0.01));
    DOCTEST_CHECK(gH_vertical_zero / gH_vertical_low == doctest::Approx(1.947f).epsilon(0.01));
}

int BLConductanceModel::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
