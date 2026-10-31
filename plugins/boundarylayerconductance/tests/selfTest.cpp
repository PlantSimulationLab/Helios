#include "BoundaryLayerConductanceModel.h"
#include "../src/GraefeConvection.h"

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

    // an unknown model is an error and leaves the model of the primitive unchanged
    DOCTEST_CHECK_THROWS_WITH(blc.setBoundaryLayerModel(UUID_1.at(3), "InvalidModel"), doctest::Contains("InvalidModel"));

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
    DOCTEST_CHECK(gH.at(1) == doctest::Approx(0.210198f).epsilon(err_tol));
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
    DOCTEST_CHECK(gH1 == doctest::Approx(0.210198f).epsilon(err_tol));
    DOCTEST_CHECK(gH2 == doctest::Approx(0.210198f).epsilon(err_tol));

    DOCTEST_CHECK_THROWS_WITH(blc.setBoundaryLayerModel("InvalidModel"), doctest::Contains("is unknown"));
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
    // The zero-wind (pure free convection) branch of the InclinedPlate model gives Nu*alpha/L times the molar density of air (41.4 mol/m^3), in mol/m^2/s, for each face.
    //
    // Hand calculation with the code's constants for a horizontal patch:
    //   Ta = 300 K, TL = 305 K, L = 0.1 m, nu = 1.568e-5 m^2/s, Pr = 0.7, alpha = nu/Pr = 2.24e-5 m^2/s
    //   Gr = 9.81 * |TL - Ta| * L^3 / (Ta * nu^2) = 9.81 * 5 * 1e-3 / (300 * 2.4586e-10) = 6.650e5
    //   upper face (heated, facing up):  Nu = 0.54 * Gr^(1/4) = 0.54 * 28.56 = 15.42 -> gH = 41.4 * 2.24e-4 * 15.42 = 0.1430 mol/m^2/s
    //   lower face (heated, facing down): Nu = 0.27 * Gr^(1/4) = 7.71                -> gH = 0.0715 mol/m^2/s
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
    const float gH_upper_expected = 41.4f * alpha / L * 0.54f * powf(Gr, 0.25f);
    const float gH_lower_expected = 41.4f * alpha / L * 0.27f * powf(Gr, 0.25f);
    DOCTEST_CHECK(gH_upper_expected == doctest::Approx(0.1430f).epsilon(0.01));

    uint UUID = context.addPatch(); // normal +z: horizontal plate, front face up
    context.setPrimitiveData(UUID, "object_length", L);
    context.setPrimitiveData(UUID, "air_temperature", Ta);
    context.setPrimitiveData(UUID, "temperature", TL);
    context.setPrimitiveData(UUID, "wind_speed", 0.f);
    blc.setBoundaryLayerModel(UUID, "InclinedPlate");

    DOCTEST_CHECK_NOTHROW(blc.run());
    float gH_free, gH_front, gH_back;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH_free));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance_front", gH_front));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance_back", gH_back));
    DOCTEST_CHECK(gH_front == doctest::Approx(gH_upper_expected).epsilon(1e-3).scale(0));
    DOCTEST_CHECK(gH_back == doctest::Approx(gH_lower_expected).epsilon(1e-3).scale(0));
    DOCTEST_CHECK(gH_free == doctest::Approx(gH_upper_expected + gH_lower_expected).epsilon(1e-3).scale(0));

    // A very small wind speed (Re ~ 320) takes the mixed free-forced branch, which must give a value of the same order.
    context.setPrimitiveData(UUID, "wind_speed", 0.05f);
    DOCTEST_CHECK_NOTHROW(blc.run());
    float gH_forced;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH_forced));
    DOCTEST_CHECK(gH_forced > 0.1f);
    DOCTEST_CHECK(gH_forced / gH_free > 0.5f);
    DOCTEST_CHECK(gH_forced / gH_free < 2.f);
}

DOCTEST_TEST_CASE("InclinedPlate zero-wind conductance and its jump from the low-wind limit") {
    // At zero wind the InclinedPlate model switches from the mixed-convection correlations of Chen et al. (1986) to the McAdams free-convection correlations, which are not their limit as the wind
    // speed tends to zero, so the conductance jumps at zero wind (described in BLConductance.dox). This pins both branches for a horizontal and a vertical plate, summed over both faces.
    //
    // Hand calculation with the code's constants, L = 0.1 m, Ta = 300 K, TL = 305 K: 41.4 * alpha / L = 9.274e-3 mol/m^2/s and Gr = 6.650e5 (see the test above).
    //   vertical, U = 0:     Nu = 0.59 Gr^(1/4) = 16.85 on each face                                -> gH = 2 * 0.1562 = 0.3125
    //   vertical, U -> 0:    Nu = (4/3) F2 Gr^(1/4) = 13.37 on each face, F2 = 0.3513                -> gH = 2 * 0.1240 = 0.2481
    //   horizontal, U = 0:   Nu = 0.54 Gr^(1/4) = 15.42 up, 0.27 Gr^(1/4) = 7.71 down                -> gH = 0.1430 + 0.0715 = 0.2145
    //   horizontal, U -> 0:  Nu = F4 Gr^(1/5) / (6 * 0.2 * F1) * 2 F1 Re^(1/2) = F4 Gr^(1/5) / 0.6 = 8.653 up, F4 = 0.3554, and half of that down
    //                                                                                                 -> gH = 0.0802 + 0.0401 = 0.1204
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
    DOCTEST_CHECK(gH_horizontal_zero == doctest::Approx(0.2145f).epsilon(0.01).scale(0));
    DOCTEST_CHECK(gH_vertical_zero == doctest::Approx(0.3125f).epsilon(0.01).scale(0));

    // Re = 0.64, far below any leaf Reynolds number but large enough to take the mixed-convection branch.
    float gH_horizontal_low, gH_vertical_low;
    conductanceAtWindSpeed(1e-4f, gH_horizontal_low, gH_vertical_low);
    DOCTEST_CHECK(gH_horizontal_low == doctest::Approx(0.1204f).epsilon(0.01).scale(0));
    DOCTEST_CHECK(gH_vertical_low == doctest::Approx(0.2481f).epsilon(0.01).scale(0));
    DOCTEST_CHECK(gH_horizontal_zero / gH_horizontal_low == doctest::Approx(1.782f).epsilon(0.01).scale(0));
    DOCTEST_CHECK(gH_vertical_zero / gH_vertical_low == doctest::Approx(1.260f).epsilon(0.01).scale(0));
}

DOCTEST_TEST_CASE("InclinedPlate conductance of a leaf cooler than the air") {
    // Regression test. The model documents that the wind is always transverse to gravity, so the buoyancy-assisting (+) form of
    // the Chen et al. (1986) mixed-convection correlation is always taken. The code instead switched to the buoyancy-opposing
    // (-) form whenever the surface was cooler than the air, which drives the term under the cube root negative once free
    // convection dominates (low wind, large temperature difference) and returned NaN. With the assisting form the conductance
    // depends only on the magnitude of the temperature difference.
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    const float Ta = 300.f;
    const float dT = 10.f;

    // A nearly horizontal and a nearly vertical leaf, which take the two different branches of the correlation
    uint UUID_horizontal = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(30.f * float(M_PI) / 180.f, 0));
    uint UUID_vertical = context.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1), make_SphericalCoord(0.5f * float(M_PI), 0));
    std::vector<uint> UUIDs{UUID_horizontal, UUID_vertical};
    context.setPrimitiveData(UUIDs, "object_length", 0.2f);
    context.setPrimitiveData(UUIDs, "air_temperature", Ta);
    context.setPrimitiveData(UUIDs, "wind_speed", 0.1f); // free convection dominates: the opposing form is negative under the root here
    blc.setBoundaryLayerModel(UUIDs, "InclinedPlate");

    for (uint UUID: UUIDs) {
        context.setPrimitiveData(UUID, "temperature", Ta - dT);
        blc.run();
        float gH_cool = 0.f;
        context.getPrimitiveData(UUID, "boundarylayer_conductance", gH_cool);

        context.setPrimitiveData(UUID, "temperature", Ta + dT);
        blc.run();
        float gH_warm = 0.f;
        context.getPrimitiveData(UUID, "boundarylayer_conductance", gH_warm);

        DOCTEST_CAPTURE(UUID);
        DOCTEST_CHECK(std::isfinite(gH_cool));
        DOCTEST_CHECK(gH_cool > 0.f);
        DOCTEST_CHECK(gH_cool == doctest::Approx(gH_warm));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// "Graefe" model: port of the leaf-boundary-layer package (Graefe et al. 2022 and its mixed-convection extensions)

namespace {
    // clang-format off
// Subset of reference/reference_values.json of the leaf-boundary-layer package (v1.1.0), generated by its Python implementation
struct GraefeFreeCase { double L, W, alpha_x, alpha_y, TA, TL, Lh, Li, phi_deg, theta_deg, Nuc, Ra, NuF_U, NuF_L, gb_U, gb_L; };
static const GraefeFreeCase graefe_free_cases[] = {
    {0.08, 0.04, 0.0, 0.0, 293.15, 281.15, 0.012965846878020378, 0.06533460285927817, 90.0, 0.0, 66.03093981885672, 1352885962.5170162, 216.61671545339465, 149.57262316682537, 0.004434388028705436, 0.0030619199825128065},
    {0.08, 0.04, 0.0, 0.0, 293.15, 305.15, 0.012965846878020378, 0.06533460285927817, 90.0, 0.0, 66.03093981885672, 1170012772.3940856, 145.91840321353416, 211.56900337947752, 0.0032240719944388093, 0.004674624198627717},
    {0.08, 0.04, 0.0, 45.0, 293.15, 296.15, 0.012965846878020378, 0.032667301429639084, 0.0, 45.0, 66.03093981885672, 308636178.2646083, 172.13107732606773, 187.82953550631927, 0.00369726095905661, 0.004034453390829284},
    {0.08, 0.04, 30.0, 0.0, 293.15, 294.15, 0.012965846878020378, 0.06533460285927817, 90.0, 29.999999999999993, 66.03093981885672, 104126369.9614998, 119.05979280849533, 148.08096128884281, 0.002541165613674699, 0.003160582073852319},
    {0.08, 0.04, 30.0, 45.0, 293.15, 290.15, 0.012965846878020378, 0.03904489325845498, 39.23152048359225, 52.23875609296496, 66.03093981885672, 320045714.19034487, 186.08978130721124, 173.22039792493274, 0.003921535895898784, 0.0036503348200676916},
    {0.08, 0.04, 60.0, 0.0, 293.15, 281.15, 0.012965846878020378, 0.06533460285927817, 90.0, 59.99999999999999, 66.03093981885672, 1352885962.5170162, 227.73101967618993, 213.6244125354132, 0.004661910348438686, 0.00437313221929117},
    {0.08, 0.04, 60.0, 0.0, 293.15, 305.15, 0.012965846878020378, 0.06533460285927817, 90.0, 59.99999999999999, 66.03093981885672, 1170012772.3940856, 207.4543909130802, 221.42393601248622, 0.004583711698704952, 0.004892369264427735},
    {0.08, 0.04, 60.0, 45.0, 293.15, 296.15, 0.012965846878020378, 0.05466285056183697, 67.79234570140352, 69.29518894536457, 66.03093981885672, 308636178.2646083, 169.74786207374942, 176.56662792958969, 0.003646071081863584, 0.0037925336334224155},
    {0.08, 0.04, 90.0, 0.0, 293.15, 294.15, 0.012965846878020378, 0.06533460285927817, 90.0, 90.0, 66.03093981885672, 104126369.9614998, 138.64288881826374, 138.64288881826374, 0.002959139549421065, 0.002959139549421065},
    {0.08, 0.04, 90.0, 45.0, 293.15, 290.15, 0.012965846878020378, 0.06533460285927817, 90.0, 90.0, 66.03093981885672, 320045714.19034487, 168.48601125692764, 168.48601125692764, 0.0035505654123483264, 0.0035505654123483264},
    {0.084, 0.084, 0.0, 0.0, 293.15, 281.15, 0.020991094518078294, 0.06860133300224208, 90.0, 0.0, 42.563210567267426, 1352885962.5170162, 171.71626213538076, 121.7674200583506, 0.0035152252011272198, 0.0024927161722626615},
    {0.084, 0.084, 0.0, 0.0, 293.15, 305.15, 0.020991094518078294, 0.06860133300224208, 90.0, 0.0, 42.563210567267426, 1170012772.3940856, 118.40753623061677, 167.4823224227052, 0.0026162184692562606, 0.0037005275098606024},
    {0.084, 0.084, 0.0, 45.0, 293.15, 296.15, 0.020991094518078294, 0.06860133300224208, 0.0, 45.0, 42.563210567267426, 308636178.2646083, 135.25648969857824, 148.13218235373787, 0.00290521936299915, 0.0031817806703135516},
    {0.084, 0.084, 30.0, 0.0, 293.15, 294.15, 0.020991094518078294, 0.06860133300224208, 90.0, 29.999999999999993, 42.563210567267426, 104126369.9614998, 99.90498395680261, 118.70884676632099, 0.0021323328713842207, 0.0025336751587226804},
    {0.084, 0.084, 30.0, 45.0, 293.15, 290.15, 0.020991094518078294, 0.06860133300224208, 39.23152048359225, 52.23875609296496, 42.563210567267426, 320045714.19034487, 150.6714383652272, 141.0702309252998, 0.0031751526058300702, 0.0029728229595960957},
    {0.084, 0.084, 60.0, 0.0, 293.15, 281.15, 0.020991094518078294, 0.06860133300224208, 90.0, 59.99999999999999, 42.563210567267426, 1352885962.5170162, 205.45662738654167, 197.82284869321245, 0.00420592846214166, 0.004049656418312389},
    {0.084, 0.084, 60.0, 0.0, 293.15, 305.15, 0.020991094518078294, 0.06860133300224208, 90.0, 59.99999999999999, 42.563210567267426, 1170012772.3940856, 191.57087448839476, 199.08697771203092, 0.004232764872600379, 0.004398833424455181},
    {0.084, 0.084, 60.0, 45.0, 293.15, 296.15, 0.020991094518078294, 0.06860133300224207, 67.79234570140352, 69.29518894536457, 42.563210567267426, 308636178.2646083, 147.7348344219782, 151.48060903222114, 0.003173245901240597, 0.003253702646431602},
    {0.084, 0.084, 90.0, 0.0, 293.15, 294.15, 0.020991094518078294, 0.06860133300224208, 90.0, 90.0, 42.563210567267426, 104126369.9614998, 121.28332732582592, 121.28332732582592, 0.002588623864045987, 0.002588623864045987},
    {0.084, 0.084, 90.0, 45.0, 293.15, 290.15, 0.020991094518078294, 0.06860133300224208, 90.0, 90.0, 42.563210567267426, 320045714.19034487, 152.08810878235874, 152.08810878235874, 0.0032050066034779463, 0.0032050066034779463},
    {0.12, 0.04, 0.0, 0.0, 293.15, 281.15, 0.014098527928760499, 0.09800190428891724, 90.0, 0.0, 56.16133524176778, 1352885962.5170162, 204.90858956796865, 140.56350975698413, 0.0041947095110239155, 0.0028774932887083533},
    {0.12, 0.04, 0.0, 0.0, 293.15, 305.15, 0.014098527928760499, 0.09800190428891724, 90.0, 0.0, 56.16133524176778, 1170012772.3940856, 136.92141945549193, 199.98094809910313, 0.0030252833376970233, 0.004418585730026942},
    {0.12, 0.04, 0.0, 45.0, 293.15, 296.15, 0.014098527928760499, 0.032667301429639084, 0.0, 45.0, 56.16133524176778, 308636178.2646083, 165.64273867670528, 180.1228224882813, 0.00355789576393857, 0.0038689183253032048},
    {0.12, 0.04, 30.0, 0.0, 293.15, 294.15, 0.014098527928760499, 0.09800190428891724, 90.0, 29.999999999999993, 56.16133524176778, 104126369.9614998, 105.80104464194032, 136.70860110762314, 0.0022581760827302274, 0.0029178548696707847},
    {0.12, 0.04, 30.0, 45.0, 293.15, 290.15, 0.014098527928760499, 0.04069307584126715, 39.23152048359225, 52.23875609296496, 56.16133524176778, 320045714.19034487, 177.38825163851592, 165.43012011828748, 0.003738165478107366, 0.0034861675356355117},
    {0.12, 0.04, 60.0, 0.0, 293.15, 281.15, 0.014098527928760499, 0.09800190428891724, 90.0, 59.99999999999999, 56.16133524176778, 1352885962.5170162, 207.79554897956427, 191.24664445291523, 0.004253808820268555, 0.003915034114135821},
    {0.12, 0.04, 60.0, 0.0, 293.15, 305.15, 0.014098527928760499, 0.09800190428891724, 90.0, 59.99999999999999, 56.16133524176778, 1170012772.3940856, 185.63217951924858, 202.03976537717446, 0.004101549208833008, 0.004464075366575272},
    {0.12, 0.04, 60.0, 45.0, 293.15, 296.15, 0.014098527928760499, 0.06694804588125466, 67.79234570140352, 69.29518894536457, 56.16133524176778, 308636178.2646083, 157.02158750944713, 164.07764277910456, 0.0033727191756781666, 0.0035242785458901565},
    {0.12, 0.04, 90.0, 0.0, 293.15, 294.15, 0.014098527928760499, 0.09800190428891724, 90.0, 90.0, 56.16133524176778, 104126369.9614998, 122.73282552329478, 122.73282552329478, 0.0026195613861901418, 0.0026195613861901418},
    {0.12, 0.04, 90.0, 45.0, 293.15, 290.15, 0.014098527928760499, 0.09800190428891724, 90.0, 90.0, 56.16133524176778, 320045714.19034487, 149.90167641992886, 149.90167641992886, 0.003158931139618549, 0.003158931139618549},
};
struct GraefeMixedCase { double L, W, alpha_x, alpha_y, leaf_azimuth, wind_from, u, TA, TL, Lf, gb_U, gb_L; };
static const GraefeMixedCase graefe_mixed_cases[] = {
    {0.08, 0.04, 0.0, 0.0, 180.0, 0.0, 0.3, 293.15, 290.15, 0.0645993009941517, 0.007281473194947151, 0.007070977687810174},
    {0.08, 0.04, 0.0, 0.0, 180.0, 90.0, 2.0, 293.15, 296.15, 0.03229965049707585, 0.02569140110428754, 0.025708495625194658},
    {0.08, 0.04, 0.0, 0.0, 180.0, 225.0, 2.0, 293.15, 290.15, 0.040856185279259785, 0.022639954882457288, 0.022618784867664908},
    {0.08, 0.04, 0.0, 0.0, 90.0, 0.0, 0.3, 293.15, 296.15, 0.03229965049707585, 0.010002434866999567, 0.010114037219449996},
    {0.08, 0.04, 0.0, 0.0, 90.0, 180.0, 0.3, 293.15, 290.15, 0.03229965049707585, 0.010011551292054943, 0.009902202121965537},
    {0.08, 0.04, 0.0, 0.0, 90.0, 225.0, 2.0, 293.15, 296.15, 0.040856185279259785, 0.022846384427526116, 0.0228679954713531},
    {0.08, 0.04, 60.0, 0.0, 180.0, 0.0, 2.0, 293.15, 290.15, 0.0645993009941517, 0.0180311802080522, 0.018020750928899935},
    {0.08, 0.04, 60.0, 0.0, 180.0, 180.0, 0.3, 293.15, 296.15, 0.0645993009941517, 0.007301542270287995, 0.007367231487161586},
    {0.08, 0.04, 60.0, 0.0, 180.0, NAN, 0.3, 293.15, 290.15, 0.041879232255866025, 0.008865667770100564, 0.008822341371266914},
    {0.08, 0.04, 60.0, 0.0, 90.0, 0.0, 2.0, 293.15, 296.15, 0.03229965049707585, 0.025703896819204354, 0.02570924414115579},
    {0.08, 0.04, 60.0, 0.0, 90.0, 180.0, 2.0, 293.15, 290.15, 0.03229965049707585, 0.025453043091638757, 0.025447811170055405},
    {0.08, 0.04, 60.0, 0.0, 90.0, NAN, 0.3, 293.15, 296.15, 0.041879232255866025, 0.008912387890083988, 0.00895665499677959},
    {0.08, 0.04, 60.0, 45.0, 180.0, 90.0, 0.3, 293.15, 290.15, 0.03229965049707585, 0.010025302482817241, 0.010005937279387752},
    {0.08, 0.04, 60.0, 45.0, 180.0, 180.0, 2.0, 293.15, 296.15, 0.03860546634560692, 0.02352129269665987, 0.023524954674398798},
    {0.08, 0.04, 60.0, 45.0, 180.0, NAN, 2.0, 293.15, 290.15, 0.041879232255866025, 0.022365649234237752, 0.022361765135421353},
    {0.08, 0.04, 60.0, 45.0, 90.0, 90.0, 0.3, 293.15, 296.15, 0.03860546634560692, 0.009289563388597577, 0.0093129852038801},
    {0.08, 0.04, 60.0, 45.0, 90.0, 225.0, 0.3, 293.15, 290.15, 0.033937128405532105, 0.009793520565362236, 0.009773225030326644},
    {0.08, 0.04, 60.0, 45.0, 90.0, NAN, 2.0, 293.15, 296.15, 0.041879232255866025, 0.022586879595170098, 0.022590850750342905},
    {0.08, 0.04, 90.0, 0.0, 180.0, 90.0, 2.0, 293.15, 290.15, 0.03229965049707585, 0.025450376867361988, 0.025450376867361988},
    {0.08, 0.04, 90.0, 0.0, 180.0, 225.0, 0.3, 293.15, 296.15, 0.03229965049707585, 0.010101117870535135, 0.010101117870535135},
    {0.08, 0.04, 90.0, 0.0, 90.0, 0.0, 0.3, 293.15, 290.15, 0.03229965049707585, 0.009999459929455036, 0.009999459929455036},
    {0.08, 0.04, 90.0, 0.0, 90.0, 90.0, 2.0, 293.15, 296.15, 0.041879232255866025, 0.02258556552064488, 0.02258556552064488},
    {0.08, 0.04, 90.0, 0.0, 90.0, 225.0, 2.0, 293.15, 290.15, 0.03229965049707585, 0.025450376867361988, 0.025450376867361988},
    {0.12, 0.04, 0.0, 0.0, 180.0, 0.0, 0.3, 293.15, 296.15, 0.09689895149122754, 0.005871987636091956, 0.006135606592186345},
    {0.12, 0.04, 0.0, 0.0, 180.0, 180.0, 0.3, 293.15, 290.15, 0.09689895149122754, 0.006070780528030877, 0.00581221688919734},
    {0.12, 0.04, 0.0, 0.0, 180.0, 225.0, 2.0, 293.15, 296.15, 0.04333452849656789, 0.02218204993437932, 0.022201348155846345},
    {0.12, 0.04, 0.0, 0.0, 90.0, 0.0, 2.0, 293.15, 290.15, 0.03229965049707585, 0.025447890875321302, 0.02543378613331611},
    {0.12, 0.04, 0.0, 0.0, 90.0, 180.0, 0.3, 293.15, 296.15, 0.03229965049707585, 0.009990808878309351, 0.010085128823309605},
    {0.12, 0.04, 0.0, 0.0, 90.0, NAN, 0.3, 293.15, 290.15, 0.04553775246222268, 0.008483054975633732, 0.00835424948106612},
    {0.12, 0.04, 60.0, 0.0, 180.0, 0.0, 2.0, 293.15, 296.15, 0.09689895149122754, 0.014872143382546036, 0.014887461896487675},
    {0.12, 0.04, 60.0, 0.0, 180.0, 180.0, 2.0, 293.15, 290.15, 0.09689895149122754, 0.014738744910726631, 0.014723753992087424},
    {0.12, 0.04, 60.0, 0.0, 180.0, NAN, 0.3, 293.15, 296.15, 0.04553775246222268, 0.008513460612683703, 0.008560000656158424},
    {0.12, 0.04, 60.0, 0.0, 90.0, 90.0, 0.3, 293.15, 290.15, 0.09689895149122754, 0.006051697843670244, 0.005961532492146602},
    {0.12, 0.04, 60.0, 0.0, 90.0, 180.0, 2.0, 293.15, 296.15, 0.03229965049707585, 0.025697754653412966, 0.02570288957577636},
    {0.12, 0.04, 60.0, 0.0, 90.0, NAN, 2.0, 293.15, 290.15, 0.04553775246222268, 0.021442244969312022, 0.021435166973326734},
    {0.12, 0.04, 60.0, 45.0, 180.0, 90.0, 0.3, 293.15, 296.15, 0.03229965049707585, 0.01007460674158219, 0.01009233608328772},
    {0.12, 0.04, 60.0, 45.0, 180.0, 225.0, 0.3, 293.15, 290.15, 0.08574004491495586, 0.006407075077679343, 0.006364683982215148},
    {0.12, 0.04, 60.0, 45.0, 180.0, NAN, 2.0, 293.15, 296.15, 0.04553775246222268, 0.021657451713435322, 0.021661294277378864},
    {0.12, 0.04, 60.0, 45.0, 90.0, 90.0, 2.0, 293.15, 290.15, 0.04023509961956632, 0.022809328255305177, 0.022806005059259907},
    {0.12, 0.04, 60.0, 45.0, 90.0, 225.0, 0.3, 293.15, 296.15, 0.03426860836737423, 0.009792239936691599, 0.009811003566437562},
    {0.12, 0.04, 90.0, 0.0, 180.0, 0.0, 0.3, 293.15, 290.15, 0.04553775246222268, 0.008443930545600255, 0.008443930545600255},
    {0.12, 0.04, 90.0, 0.0, 180.0, 90.0, 2.0, 293.15, 296.15, 0.03229965049707585, 0.02569955834695541, 0.02569955834695541},
    {0.12, 0.04, 90.0, 0.0, 180.0, 225.0, 2.0, 293.15, 290.15, 0.03229965049707585, 0.025443562565571804, 0.025443562565571804},
    {0.12, 0.04, 90.0, 0.0, 90.0, 0.0, 0.3, 293.15, 296.15, 0.03229965049707585, 0.010055980317052989, 0.010055980317052989},
    {0.12, 0.04, 90.0, 0.0, 90.0, 180.0, 0.3, 293.15, 290.15, 0.03229965049707585, 0.009955133062007765, 0.009955133062007765},
    {0.12, 0.04, 90.0, 0.0, 90.0, 225.0, 2.0, 293.15, 296.15, 0.03229965049707585, 0.02569955834695541, 0.02569955834695541},
};
struct GraefeAngleCase { double alpha_x, alpha_y, leaf_azimuth, wind_from, u, TI, attack_deg, Re_f, NuF_forced_U, NuF_forced_L, gb_U, gb_L; };
static const GraefeAngleCase graefe_angle_cases[] = {
    {0.0, 0.0, 180.0, 0.0, 0.5, 1.0, 0.0, 2125.40590660689, 490.80939980790953, 490.80939980790953, 0.010591654890685016, 0.01069135876128178},
    {0.0, 0.0, 180.0, 45.0, 0.5, 1.0, 0.0, 1344.2247234505776, 617.1601469210826, 617.1601469210826, 0.013287502297011377, 0.013351146117835567},
    {0.0, 0.0, 180.0, 90.0, 0.5, 1.0, 0.0, 1062.702953303445, 694.1093097485444, 694.1093097485444, 0.014933778495011344, 0.014984234809962863},
    {0.0, 0.0, 180.0, 180.0, 0.5, 1.0, 0.0, 2125.40590660689, 490.80939980790953, 490.80939980790953, 0.010591654890685016, 0.01069135876128178},
    {0.0, 0.0, 180.0, 270.0, 0.5, 1.0, 0.0, 1062.702953303445, 694.1093097485444, 694.1093097485444, 0.014933778495011344, 0.014984234809962863},
    {60.0, 0.0, 180.0, 0.0, 0.5, 1.0, 59.99999999999999, 2125.40590660689, 416.9170780928743, 586.871470162774, 0.009123587316895722, 0.012713560304459404},
    {60.0, 0.0, 180.0, 45.0, 0.5, 1.0, 37.761243907035045, 1152.6631909267714, 666.5047277888668, 758.7103821635322, 0.014382939277128866, 0.01636147975891159},
    {60.0, 0.0, 180.0, 90.0, 0.5, 1.0, 3.038324251750845e-15, 1062.702953303445, 694.1093097485444, 694.1093097485444, 0.014970687773965562, 0.014986437909129586},
    {60.0, 0.0, 180.0, 180.0, 0.5, 1.0, 59.99999999999999, 2125.40590660689, 586.871470162774, 416.9170780928743, 0.012691660585076215, 0.009165842840403535},
    {60.0, 0.0, 180.0, 270.0, 0.5, 1.0, 9.114972755252537e-15, 1062.702953303445, 694.1093097485444, 694.1093097485445, 0.014970687773965562, 0.014986437909129588},
    {60.0, 45.0, 90.0, 0.0, 0.5, 1.0, 45.00000000000001, 1062.702953303445, 642.4328785105677, 805.0322341051319, 0.013883365116830847, 0.01735215721854369},
    {60.0, 45.0, 90.0, 45.0, 0.5, 1.0, 3.840965716258137, 1116.5782300481003, 726.6316842554608, 687.408350309597, 0.01567361100868817, 0.014848012737518297},
    {60.0, 45.0, 90.0, 90.0, 0.5, 1.0, 37.76124390703503, 1270.1729730124293, 722.7628230664073, 634.9258557265372, 0.015591214877304312, 0.01373485625343776},
    {60.0, 45.0, 90.0, 180.0, 0.5, 1.0, 45.000000000000014, 1062.702953303445, 805.0322341051319, 642.4328785105677, 0.017345424859840218, 0.013893869912968485},
    {60.0, 45.0, 90.0, 270.0, 0.5, 1.0, 37.76124390703502, 1270.1729730124293, 634.9258557265374, 722.7628230664073, 0.013724106527371492, 0.015599546195509567},
    {90.0, 0.0, 180.0, 0.0, 0.5, 1.0, 90.0, 1377.8843769352504, 507.80107279012043, 747.3394281617977, 0.011035878815965336, 0.016112229607362232},
    {90.0, 0.0, 180.0, 45.0, 0.5, 1.0, 45.00000000000001, 1062.702953303445, 642.4328785105677, 805.0322341051319, 0.01387988634024342, 0.01734319646002054},
    {90.0, 0.0, 180.0, 90.0, 0.5, 1.0, 3.5083546492674376e-15, 1062.702953303445, 694.1093097485444, 694.1093097485444, 0.014978354013247692, 0.014978354013247692},
    {90.0, 0.0, 180.0, 180.0, 0.5, 1.0, 90.0, 1377.8843769352504, 747.3394281617977, 507.80107279012043, 0.016112229607362232, 0.011035878815965336},
    {90.0, 0.0, 180.0, 270.0, 0.5, 1.0, 1.0525063947802315e-14, 1062.702953303445, 694.1093097485444, 694.1093097485445, 0.014978354013247692, 0.014978354013247694},
};
struct GraefeScalarCase { double dT, u, TI, alpha_x, wind_from, L, W, gb_U, gb_L, gv_U, gv_L, gc_U, gc_L, k_U, k_L; };
static const GraefeScalarCase graefe_scalar_cases[] = {
    {-6.0, 0.0, 1.0, 0.0, 90.0, 0.08, 0.04, 0.00402353562403156, 0.002990177618179168, 0.0044526297328150825, 0.003293822177649165, 0.0031291586222774582, 0.0023522976970068235, 0.8356323198219725, 0.7975508403105382},
    {-6.0, 0.0, 1.0, 60.0, 180.0, 0.6, 0.3, 0.0020477925615022983, 0.002623092480190881, 0.002248880186775686, 0.0028508690555196095, 0.0016231661506578938, 0.002133516334448724, 0.7724335671395287, 0.6866712276402406},
    {-6.0, 0.4, 5.0, 0.0, 90.0, 0.08, 0.04, 0.013260542810606265, 0.013187345547542672, 0.014385335763056272, 0.014300389545280174, 0.010835293881411263, 0.010785843408440878, 0.6713866240593274, 0.6681924959172896},
    {-6.0, 0.4, 5.0, 60.0, 180.0, 0.6, 0.3, 0.004222562238534611, 0.0034749926688394917, 0.00458481185823261, 0.003771526431851947, 0.0034426744254884537, 0.002836127772676969, 0.6787303375943292, 0.6752708202483069},
    {-6.0, 2.5, 20.0, 0.0, 90.0, 0.08, 0.04, 0.032859869661272204, 0.03284801079870851, 0.035628069943285934, 0.03561429880500598, 0.026885702695437046, 0.026877709662901896, 0.6669768542967766, 0.6667653970606845},
    {-6.0, 2.5, 20.0, 60.0, 180.0, 0.6, 0.3, 0.015178880343816415, 0.010825913709311722, 0.016457487683420268, 0.011737880162027117, 0.012419436406020976, 0.008857750939242098, 0.6669263765328974, 0.6669512281713217},
    {-6.0, 12.0, 40.0, 0.0, 90.0, 0.08, 0.04, 0.0883467560172558, 0.08834511601361107, 0.0937358000271132, 0.09373382330944073, 0.07627726840853054, 0.07627633025896824, 0.4882710043930027, 0.48825018262088254},
    {-6.0, 12.0, 40.0, 60.0, 180.0, 0.6, 0.3, 0.0547404640898662, 0.03889125946923667, 0.05754046917688426, 0.04088089401441575, 0.048368258701124794, 0.034363343882943265, 0.4113708799846271, 0.41143645034700105},
    {3.0, 0.0, 1.0, 0.0, 90.0, 0.08, 0.04, 0.002647327128667893, 0.003694289528625949, 0.002928134276886006, 0.004087300677964136, 0.00205086511463121, 0.00285989950711603, 0.8436231503488935, 0.845979255397316},
    {3.0, 0.0, 1.0, 60.0, 180.0, 0.6, 0.3, 0.0021367348475918356, 0.001765926425542201, 0.002321923032365702, 0.0019376995626648233, 0.0017311850328332128, 0.0013960078049623631, 0.6955247279359357, 0.7767716067178547},
    {3.0, 0.4, 5.0, 0.0, 90.0, 0.08, 0.04, 0.013369708748926713, 0.013428869643644672, 0.014480805543162917, 0.014548984658828805, 0.010922620251127274, 0.01096312245544946, 0.6680404665357162, 0.6703999005000838},
    {3.0, 0.4, 5.0, 60.0, 180.0, 0.6, 0.3, 0.004300525016501682, 0.003125258445089139, 0.004659128160607568, 0.0033924725290277982, 0.0035110881462967164, 0.0025389889380799725, 0.6702062701580729, 0.6865306713488109},
    {3.0, 2.5, 20.0, 0.0, 90.0, 0.08, 0.04, 0.03334311440657456, 0.03335266570105393, 0.03610856547547042, 0.03611957844793079, 0.02725084405598088, 0.027257370768780105, 0.6667552334399495, 0.6669103421947095},
    {3.0, 2.5, 20.0, 60.0, 180.0, 0.6, 0.3, 0.015282757711386673, 0.01086264389289621, 0.016550278156373898, 0.011764123359764978, 0.012490412985257542, 0.00887684848678476, 0.6667455370344261, 0.6671397285907789},
    {3.0, 12.0, 40.0, 0.0, 90.0, 0.08, 0.04, 0.08919488875838955, 0.08919622385447665, 0.09457318741187255, 0.09457478439103506, 0.07690414309510203, 0.07690492073369376, 0.48995073800675054, 0.4899667865555479},
    {3.0, 12.0, 40.0, 60.0, 180.0, 0.6, 0.3, 0.055136372071353655, 0.0391696521647636, 0.0579162838102648, 0.041144623743022184, 0.048679147872769266, 0.034582172256445845, 0.4116151196906288, 0.411632057833944},
    {12.0, 0.0, 1.0, 0.0, 90.0, 0.08, 0.04, 0.0036390616085796735, 0.004674624198627716, 0.00398057024816019, 0.005153424699435453, 0.0028861158585376345, 0.0036333016691644723, 0.7616775233859943, 0.8280252999779242},
    {12.0, 0.0, 1.0, 60.0, 180.0, 0.6, 0.3, 0.0033450148772121214, 0.0024930189982362567, 0.0036245633892401925, 0.0027294925990033743, 0.002718402059814913, 0.0019724910530144967, 0.6815474903878836, 0.7695072162514623},
    {12.0, 0.4, 5.0, 0.0, 90.0, 0.08, 0.04, 0.01362151514075445, 0.013717771124662966, 0.014737188561850672, 0.014849324444221063, 0.011113951397634483, 0.011176918777368055, 0.6684782787397457, 0.6730519718322153},
    {12.0, 0.4, 5.0, 60.0, 180.0, 0.6, 0.3, 0.004797004969490379, 0.003466386610744598, 0.0051918817421802025, 0.0037664327024717684, 0.003910081333658271, 0.0027970670028881267, 0.6717122412226989, 0.7049236003010602},
    {12.0, 2.5, 20.0, 0.0, 90.0, 0.08, 0.04, 0.03384998336432076, 0.033865673463126335, 0.036615170895598986, 0.03663346423381106, 0.027632831194040478, 0.027643062241760902, 0.6667847166907225, 0.667091044963872},
    {12.0, 2.5, 20.0, 60.0, 180.0, 0.6, 0.3, 0.015424283110203838, 0.010963264427318507, 0.01668435104139391, 0.011860371841898906, 0.012591209629630692, 0.008946693943933617, 0.6668184437381557, 0.6678759316735935},
    {12.0, 12.0, 40.0, 0.0, 90.0, 0.08, 0.04, 0.09004884148649092, 0.09005105956507459, 0.09541663008010548, 0.09541930887343165, 0.07753400255873348, 0.07753522253552723, 0.4916612936956836, 0.49169052690349274},
    {12.0, 12.0, 40.0, 60.0, 180.0, 0.6, 0.3, 0.05553435806196535, 0.039452391793868134, 0.058294634444409826, 0.04141348378909851, 0.04899115716517756, 0.03480367789019938, 0.41190513201950096, 0.41193644057404133},
};
    // clang-format on

    const double graefe_reference_tolerance = 1e-6;

    // per-face conductances written by BLConductanceModel::run()
    struct FaceData {
        float total, heat_front, heat_back, vapor_front, vapor_back, CO2_front, CO2_back;
    };

    FaceData getFaceData(Context &context, uint UUID) {
        FaceData data{};
        context.getPrimitiveData(UUID, "boundarylayer_conductance", data.total);
        context.getPrimitiveData(UUID, "boundarylayer_conductance_front", data.heat_front);
        context.getPrimitiveData(UUID, "boundarylayer_conductance_back", data.heat_back);
        context.getPrimitiveData(UUID, "boundarylayer_conductance_vapor_front", data.vapor_front);
        context.getPrimitiveData(UUID, "boundarylayer_conductance_vapor_back", data.vapor_back);
        context.getPrimitiveData(UUID, "boundarylayer_conductance_CO2_front", data.CO2_front);
        context.getPrimitiveData(UUID, "boundarylayer_conductance_CO2_back", data.CO2_back);
        return data;
    }
} // namespace

DOCTEST_TEST_CASE("Graefe free convection matches the reference implementation") {
    for (const auto &c: graefe_free_cases) {
        graefe::Vec3 normal_up, short_axis;
        graefe::orientationFromAngles(c.alpha_x, c.alpha_y, 180.0, normal_up, short_axis);
        graefe::LeafGeometry geometry = graefe::leafGeometry(c.L, c.W, normal_up, short_axis);
        graefe::FreeConvectionResult free = graefe::freeConvection(c.TL, c.TA, 101325.0, geometry, false);

        DOCTEST_CAPTURE(c.L);
        DOCTEST_CAPTURE(c.alpha_x);
        DOCTEST_CAPTURE(c.alpha_y);
        DOCTEST_CAPTURE(c.TL);
        DOCTEST_CHECK(geometry.length_horizontal == doctest::Approx(c.Lh).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(geometry.length_inclined == doctest::Approx(c.Li).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(geometry.phi * 180.0 / M_PI == doctest::Approx(c.phi_deg).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(geometry.theta * 180.0 / M_PI == doctest::Approx(c.theta_deg).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(free.Nuc == doctest::Approx(c.Nuc).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(free.Ra == doctest::Approx(c.Ra).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(free.NuF_U == doctest::Approx(c.NuF_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(free.NuF_L == doctest::Approx(c.NuF_L).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(free.air.thermal_diffusivity * free.NuF_U == doctest::Approx(c.gb_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(free.air.thermal_diffusivity * free.NuF_L == doctest::Approx(c.gb_L).epsilon(graefe_reference_tolerance).scale(1e-6));
    }
}

DOCTEST_TEST_CASE("Graefe laminar mixed convection matches the reference implementation") {
    graefe::ModelOptions options;
    options.forced_transition = false;
    options.free_transition = false;
    options.angle_model = false;
    for (const auto &c: graefe_mixed_cases) {
        graefe::Vec3 normal_up, short_axis;
        graefe::orientationFromAngles(c.alpha_x, c.alpha_y, c.leaf_azimuth, normal_up, short_axis);
        graefe::LeafGeometry geometry = graefe::leafGeometry(c.L, c.W, normal_up, short_axis);
        graefe::Vec3 wind = graefe::windVectorFromDirection(c.wind_from);
        const graefe::Vec3 *wind_ptr = std::isnan(c.wind_from) ? nullptr : &wind;
        graefe::ConductanceResult result = graefe::leafConductances(c.TL, c.TA, c.u, 101325.0, geometry, normal_up, short_axis, wind_ptr, options);

        DOCTEST_CAPTURE(c.alpha_x);
        DOCTEST_CAPTURE(c.alpha_y);
        DOCTEST_CAPTURE(c.leaf_azimuth);
        DOCTEST_CAPTURE(c.wind_from);
        DOCTEST_CAPTURE(c.u);
        DOCTEST_CHECK(result.streamline_length == doctest::Approx(c.Lf).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gb_U == doctest::Approx(c.gb_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gb_L == doctest::Approx(c.gb_L).epsilon(graefe_reference_tolerance).scale(1e-6));
    }
}

DOCTEST_TEST_CASE("Graefe angle of attack and forced transition match the reference implementation") {
    graefe::ModelOptions options;
    options.forced_transition = true;
    options.free_transition = false;
    options.angle_model = true;
    for (const auto &c: graefe_angle_cases) {
        options.turbulence_intensity_percent = c.TI;
        graefe::Vec3 normal_up, short_axis;
        graefe::orientationFromAngles(c.alpha_x, c.alpha_y, c.leaf_azimuth, normal_up, short_axis);
        graefe::LeafGeometry geometry = graefe::leafGeometry(0.08, 0.04, normal_up, short_axis);
        graefe::Vec3 wind = graefe::windVectorFromDirection(c.wind_from);
        graefe::ConductanceResult result = graefe::leafConductances(296.15, 293.15, c.u, 101325.0, geometry, normal_up, short_axis, &wind, options);

        DOCTEST_CAPTURE(c.alpha_x);
        DOCTEST_CAPTURE(c.alpha_y);
        DOCTEST_CAPTURE(c.leaf_azimuth);
        DOCTEST_CAPTURE(c.wind_from);
        DOCTEST_CAPTURE(c.u);
        DOCTEST_CHECK(result.attack_angle_deg == doctest::Approx(c.attack_deg).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.reynolds == doctest::Approx(c.Re_f).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.NuF_forced_U == doctest::Approx(c.NuF_forced_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.NuF_forced_L == doctest::Approx(c.NuF_forced_L).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gb_U == doctest::Approx(c.gb_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gb_L == doctest::Approx(c.gb_L).epsilon(graefe_reference_tolerance).scale(1e-6));
    }
}

DOCTEST_TEST_CASE("Graefe heat, vapor and CO2 conductances match the reference implementation") {
    graefe::ModelOptions options; // general model: all extensions on
    for (const auto &c: graefe_scalar_cases) {
        options.turbulence_intensity_percent = c.TI;
        graefe::Vec3 normal_up, short_axis;
        graefe::orientationFromAngles(c.alpha_x, 0.0, 180.0, normal_up, short_axis);
        graefe::LeafGeometry geometry = graefe::leafGeometry(c.L, c.W, normal_up, short_axis);
        graefe::Vec3 wind = graefe::windVectorFromDirection(c.wind_from);
        graefe::ConductanceResult result = graefe::leafConductances(293.15 + c.dT, 293.15, c.u, 101325.0, geometry, normal_up, short_axis, &wind, options);

        DOCTEST_CAPTURE(c.dT);
        DOCTEST_CAPTURE(c.u);
        DOCTEST_CAPTURE(c.alpha_x);
        DOCTEST_CAPTURE(c.L);
        DOCTEST_CHECK(result.gb_U == doctest::Approx(c.gb_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gb_L == doctest::Approx(c.gb_L).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gv_U == doctest::Approx(c.gv_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gv_L == doctest::Approx(c.gv_L).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gc_U == doctest::Approx(c.gc_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.gc_L == doctest::Approx(c.gc_L).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.k_U == doctest::Approx(c.k_U).epsilon(graefe_reference_tolerance).scale(1e-6));
        DOCTEST_CHECK(result.k_L == doctest::Approx(c.k_L).epsilon(graefe_reference_tolerance).scale(1e-6));
    }
}

DOCTEST_TEST_CASE("Graefe model gives each face of a leaf its own conductance") {
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    const float Ta = 293.15f;

    // 8 x 4 cm leaves: horizontal facing up, horizontal facing down (rotated half a turn about the long axis), and vertical
    uint UUID_up = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.08, 0.04));
    uint UUID_down = context.addPatch(make_vec3(1, 0, 0), make_vec2(0.08, 0.04), make_SphericalCoord(float(M_PI), 0));
    uint UUID_vertical = context.addPatch(make_vec3(2, 0, 0), make_vec2(0.08, 0.04), make_SphericalCoord(0.5f * float(M_PI), 0));
    DOCTEST_REQUIRE(context.getPrimitiveNormal(UUID_up).z > 0.99f);
    DOCTEST_REQUIRE(context.getPrimitiveNormal(UUID_down).z < -0.99f);
    DOCTEST_REQUIRE(std::fabs(context.getPrimitiveNormal(UUID_vertical).z) < 1e-3f);

    std::vector<uint> UUIDs{UUID_up, UUID_down, UUID_vertical};
    context.setPrimitiveData(UUIDs, "air_temperature", Ta);
    context.setPrimitiveData(UUIDs, "wind_speed", 0.f);
    blc.setBoundaryLayerModel(UUIDs, "Graefe");

    // leaf warmer than the air in still air. For a horizontal leaf the fitted model gives the lower face the larger conductance (p5 < p6 in Table 3 of Graefe et al. 2022), unlike the classical correlations for large
    // heated plates.
    context.setPrimitiveData(UUIDs, "temperature", Ta + 5.f);
    blc.run();
    FaceData up_heated = getFaceData(context, UUID_up);
    FaceData down_heated = getFaceData(context, UUID_down);
    FaceData vertical_heated = getFaceData(context, UUID_vertical);

    // leaf cooler than the air: the faces exchange roles
    context.setPrimitiveData(UUIDs, "temperature", Ta - 5.f);
    blc.run();
    FaceData up_cooled = getFaceData(context, UUID_up);

    DOCTEST_CHECK(up_heated.heat_back > 1.2f * up_heated.heat_front);
    DOCTEST_CHECK(up_heated.total == doctest::Approx(up_heated.heat_front + up_heated.heat_back).scale(0));
    // the front of a leaf facing down is its lower face
    DOCTEST_CHECK(down_heated.heat_front == doctest::Approx(up_heated.heat_back).epsilon(1e-4).scale(0));
    DOCTEST_CHECK(down_heated.heat_back == doctest::Approx(up_heated.heat_front).epsilon(1e-4).scale(0));
    DOCTEST_CHECK(up_cooled.heat_front > 1.2f * up_cooled.heat_back);
    // both faces of a vertical leaf are alike
    DOCTEST_CHECK(vertical_heated.heat_front == doctest::Approx(vertical_heated.heat_back).epsilon(1e-4).scale(0));

    // water vapor diffuses faster than heat, and CO2 slower
    DOCTEST_CHECK(up_heated.vapor_front > up_heated.heat_front);
    DOCTEST_CHECK(up_heated.CO2_front < up_heated.heat_front);
    DOCTEST_CHECK(up_heated.vapor_front / up_heated.heat_front == doctest::Approx(1.1f).epsilon(0.05).scale(0));
}

DOCTEST_TEST_CASE("Graefe model windward and leeward faces") {
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    const float Ta = 293.15f;
    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.08, 0.04), make_SphericalCoord(0.25f * float(M_PI), 0.3f));
    vec3 normal = context.getPrimitiveNormal(UUID);
    context.setPrimitiveData(UUID, "air_temperature", Ta);
    context.setPrimitiveData(UUID, "temperature", Ta + 1.f);
    context.setPrimitiveData(UUID, "wind_speed", 2.f);
    blc.setBoundaryLayerModel(UUID, "Graefe");

    // wind blowing into the front face
    context.setPrimitiveData(UUID, "wind_direction", make_vec3(-normal.x, -normal.y, 0));
    blc.run();
    FaceData into_front = getFaceData(context, UUID);

    // wind blowing into the back face
    context.setPrimitiveData(UUID, "wind_direction", make_vec3(normal.x, normal.y, 0));
    blc.run();
    FaceData into_back = getFaceData(context, UUID);

    DOCTEST_CHECK(into_front.heat_front > 1.1f * into_front.heat_back);
    DOCTEST_CHECK(into_back.heat_back > 1.1f * into_back.heat_front);
}

DOCTEST_TEST_CASE("Graefe model leaf shape from the parent object") {
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    const float Ta = 293.15f;

    // 12 x 4 cm tiles with the long axis along x and along y, with wind along x
    uint objID_along_x = context.addTileObject(make_vec3(0, 0, 0), make_vec2(0.12, 0.04), nullrotation, make_int2(3, 2));
    uint objID_along_y = context.addTileObject(make_vec3(1, 0, 0), make_vec2(0.04, 0.12), nullrotation, make_int2(2, 3));
    uint objID_override = context.addTileObject(make_vec3(2, 0, 0), make_vec2(0.12, 0.04), nullrotation, make_int2(3, 2));
    uint UUID_along_x = context.getObjectPrimitiveUUIDs(objID_along_x).front();
    uint UUID_along_y = context.getObjectPrimitiveUUIDs(objID_along_y).front();
    uint UUID_override = context.getObjectPrimitiveUUIDs(objID_override).front();

    std::vector<uint> UUIDs{UUID_along_x, UUID_along_y, UUID_override};
    context.setPrimitiveData(UUIDs, "air_temperature", Ta);
    context.setPrimitiveData(UUIDs, "temperature", Ta + 1.f);
    context.setPrimitiveData(UUIDs, "wind_speed", 1.f);
    context.setPrimitiveData(UUIDs, "wind_direction", make_vec3(1, 0, 0));
    // the equivalent ellipse of a 12 x 4 cm rectangle has the same aspect ratio (3) and area
    const float area = 0.12f * 0.04f;
    const float length = std::sqrt(4.f * area * 3.f / float(M_PI));
    context.setPrimitiveData(UUID_override, "leaf_length", length);
    context.setPrimitiveData(UUID_override, "leaf_width", length / 3.f);
    blc.setBoundaryLayerModel(UUIDs, "Graefe");
    blc.run();

    FaceData along_x = getFaceData(context, UUID_along_x);
    FaceData along_y = getFaceData(context, UUID_along_y);
    FaceData overridden = getFaceData(context, UUID_override);

    // a longer streamline gives a thicker boundary layer
    DOCTEST_CHECK(along_x.total < 0.8f * along_y.total);
    DOCTEST_CHECK(overridden.total == doctest::Approx(along_x.total).epsilon(1e-4).scale(0));
}

DOCTEST_TEST_CASE("Graefe model single-sided primitive and other models' face values") {
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    uint UUID_graefe = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.08, 0.04));
    uint UUID_pohlhausen = context.addPatch(make_vec3(1, 0, 0), make_vec2(0.08, 0.04));
    context.setPrimitiveData(UUID_graefe, "twosided_flag", 0u);
    std::vector<uint> UUIDs{UUID_graefe, UUID_pohlhausen};
    context.setPrimitiveData(UUIDs, "wind_speed", 0.5f);
    blc.setBoundaryLayerModel(UUID_graefe, "Graefe");
    blc.setBoundaryLayerModel(UUID_pohlhausen, "Pohlhausen");
    blc.run();

    FaceData graefe = getFaceData(context, UUID_graefe);
    DOCTEST_CHECK(graefe.heat_front > 0.f);
    DOCTEST_CHECK(graefe.heat_back == 0.f);
    DOCTEST_CHECK(graefe.vapor_back == 0.f);
    DOCTEST_CHECK(graefe.CO2_back == 0.f);
    DOCTEST_CHECK(graefe.total == graefe.heat_front);

    FaceData pohlhausen = getFaceData(context, UUID_pohlhausen);
    DOCTEST_CHECK(pohlhausen.heat_front == doctest::Approx(0.5f * pohlhausen.total).scale(0));
    DOCTEST_CHECK(pohlhausen.heat_back == doctest::Approx(0.5f * pohlhausen.total).scale(0));
    DOCTEST_CHECK(pohlhausen.vapor_front == doctest::Approx(1.08f * pohlhausen.heat_front).scale(0));
    DOCTEST_CHECK(pohlhausen.CO2_front == doctest::Approx(1.08f / 1.37f * pohlhausen.heat_front).scale(0));
}

DOCTEST_TEST_CASE("Graefe model input errors") {
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    uint UUID = context.addPatch(make_vec3(0, 0, 0), make_vec2(0.08, 0.04));
    blc.setBoundaryLayerModel(UUID, "Graefe");
    DOCTEST_CHECK_NOTHROW(blc.run());

    context.setPrimitiveData(UUID, "leaf_length", 0.1f);
    DOCTEST_CHECK_THROWS_WITH(blc.run(), doctest::Contains("leaf_width"));
    context.setPrimitiveData(UUID, "leaf_width", 0.2f);
    DOCTEST_CHECK_THROWS_WITH(blc.run(), doctest::Contains("longer"));
    context.clearPrimitiveData(UUID, "leaf_length");
    context.clearPrimitiveData(UUID, "leaf_width");

    context.setPrimitiveData(UUID, "wind_direction", make_vec3(0, 0, 1));
    DOCTEST_CHECK_THROWS_WITH(blc.run(), doctest::Contains("horizontal"));
    context.clearPrimitiveData(UUID, "wind_direction");

    context.setPrimitiveData(UUID, "turbulence_intensity", -1.f);
    DOCTEST_CHECK_THROWS_WITH(blc.run(), doctest::Contains("turbulence_intensity"));
    context.clearPrimitiveData(UUID, "turbulence_intensity");

    uint UUID_voxel = context.addVoxel(make_vec3(1, 0, 0), make_vec3(0.1, 0.1, 0.1));
    blc.setBoundaryLayerModel(UUID_voxel, "Graefe");
    DOCTEST_CHECK_THROWS_WITH(blc.run(), doctest::Contains("voxel"));
}


int BLConductanceModel::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
