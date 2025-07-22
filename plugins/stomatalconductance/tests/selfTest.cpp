#include "StomatalConductanceModel.h"
#include "doctest.h"

using namespace std;
using namespace helios;

int StomatalConductanceModel::selfTest() {
    
    std::cout << "Running stomatal conductance model self-test..." << std::endl;

    doctest::Context context;
    context.setOption("no-version", true);
    context.setOption("no-time-in-output", true);
    context.setOption("silence", true);

    int result = context.run();
    
    if(result == 0) {
        std::cout << "Stomatal conductance model self-test PASSED." << std::endl;
    } else {
        std::cout << "Stomatal conductance model self-test FAILED." << std::endl;
    }
    
    return result;
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Original selfTest") {
    Context context_selftest;

    float RMSE_max = 0.03;

    std::vector<float> An_ref{21.81, 22.71, 20.02, 22.60, 19.97, 17.32, 11.90, 6.87, 1.21, -1.49};
    std::vector<float> Tair_ref{20.69, 30.37, 39.19, 27.06, 27.12, 27.11, 27.08, 26.98, 26.87, 26.81};
    std::vector<float> TL_ref{21.00, 30.02, 38.01, 26.99, 27.00, 27.01, 27.01, 27.00, 27.00, 26.99};
    std::vector<float> Q_ref{2000, 2000, 2000, 2000, 1200, 800, 400, 200, 50, 0};
    float gbw_ref = 3.5;
    float Cs_ref = 400;
    float hs_ref = 0.55;
    float Patm_ref = 101300;
    std::vector<float> Gamma_ref = {43.7395, 70.2832, 105.5414, 60.0452, 60.0766, 60.1080, 60.1080, 60.0766, 60.0766, 60.0452};

    std::vector<float> gs_ref{0.3437, 0.3386, 0.3531, 0.3811, 0.3247, 0.2903, 0.2351, 0.1737, 0.0868, 0.0421};
    std::vector<float> gs_BWB(gs_ref.size());
    std::vector<float> gs_BBL(gs_ref.size());
    std::vector<float> gs_MOPT(gs_ref.size());
    std::vector<float> gs_BMF(gs_ref.size());

    uint UUID0 = context_selftest.addPatch();

    BWBcoefficients BWBcoeffs;
    BBLcoefficients BBLcoeffs;
    MOPTcoefficients MOPTcoeffs;
    BMFcoefficients BMFcoeffs;

    StomatalConductanceModel gsm(&context_selftest);

    float RMSE_BWB = 0.f;
    float RMSE_BBL = 0.f;
    float RMSE_MOPT = 0.f;
    float RMSE_BMF = 0.f;

    for (uint i = 0; i < gs_ref.size(); i++) {
        context_selftest.setPrimitiveData(UUID0, "radiation_flux_PAR", Q_ref.at(i) / 4.57f);
        context_selftest.setPrimitiveData(UUID0, "net_photosynthesis", An_ref.at(i));
        context_selftest.setPrimitiveData(UUID0, "temperature", TL_ref.at(i) + 273.f);
        context_selftest.setPrimitiveData(UUID0, "air_temperature", Tair_ref.at(i) + 273.f);
        context_selftest.setPrimitiveData(UUID0, "air_CO2", Cs_ref);
        context_selftest.setPrimitiveData(UUID0, "air_humidity", hs_ref);
        context_selftest.setPrimitiveData(UUID0, "air_pressure", Patm_ref);
        context_selftest.setPrimitiveData(UUID0, "boundarylayer_conductance", gbw_ref);
        context_selftest.setPrimitiveData(UUID0, "Gamma_CO2", Gamma_ref.at(i));

        gsm.setModelCoefficients(BWBcoeffs);
        gsm.run();
        context_selftest.getPrimitiveData(UUID0, "moisture_conductance", gs_BWB.at(i));
        RMSE_BWB += pow(gs_BWB.at(i) - gs_ref.at(i), 2) / float(gs_ref.size());

        gsm.setModelCoefficients(BBLcoeffs);
        gsm.run();
        context_selftest.getPrimitiveData(UUID0, "moisture_conductance", gs_BBL.at(i));
        RMSE_BBL += pow(gs_BBL.at(i) - gs_ref.at(i), 2) / float(gs_ref.size());

        gsm.setModelCoefficients(MOPTcoeffs);
        gsm.run();
        context_selftest.getPrimitiveData(UUID0, "moisture_conductance", gs_MOPT.at(i));
        RMSE_MOPT += pow(gs_MOPT.at(i) - gs_ref.at(i), 2) / float(gs_ref.size());

        gsm.setModelCoefficients(BMFcoeffs);
        gsm.run();
        context_selftest.getPrimitiveData(UUID0, "moisture_conductance", gs_BMF.at(i));
        RMSE_BMF += pow(gs_BMF.at(i) - gs_ref.at(i), 2) / float(gs_ref.size());
    }

    DOCTEST_CHECK(sqrtf(RMSE_BWB) <= RMSE_max);
    DOCTEST_CHECK(sqrtf(RMSE_BBL) <= RMSE_max);
    DOCTEST_CHECK(sqrtf(RMSE_MOPT) <= RMSE_max);
    DOCTEST_CHECK(sqrtf(RMSE_BMF) <= RMSE_max);
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Constructor") {
    Context context;
    StomatalConductanceModel gsm(&context);
    DOCTEST_CHECK_NOTHROW(gsm.disableMessages());
    DOCTEST_CHECK_NOTHROW(gsm.enableMessages());
}

DOCTEST_TEST_CASE("StomatalConductanceModel - BWB Model with UUID Subset") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID1 = context.addPatch();
    uint UUID2 = context.addPatch();
    
    BWBcoefficients coeffs;
    coeffs.gs0 = 0.05f;
    coeffs.a1 = 10.0f;

    std::vector<uint> UUIDs = {UUID1};
    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs, UUIDs));
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - BBL Model") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    BBLcoefficients coeffs;
    coeffs.gs0 = 0.08f;
    coeffs.a1 = 5.0f;
    coeffs.D0 = 15000.0f;

    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs));
    DOCTEST_CHECK_NOTHROW(gsm.run());

    std::vector<uint> UUIDs = {UUID};
    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs, UUIDs));
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - MOPT Model") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    MOPTcoefficients coeffs;
    coeffs.gs0 = 0.09f;
    coeffs.g1 = 3.0f;

    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs));
    DOCTEST_CHECK_NOTHROW(gsm.run());

    std::vector<uint> UUIDs = {UUID};
    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs, UUIDs));
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - BMF Model") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    BMFcoefficients coeffs;
    coeffs.Em = 300.0f;
    coeffs.i0 = 40.0f;
    coeffs.k = 250000.0f;
    coeffs.b = 600.0f;

    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs));
    DOCTEST_CHECK_NOTHROW(gsm.run());

    std::vector<uint> UUIDs = {UUID};
    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs, UUIDs));
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - BB Model") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    BBcoefficients coeffs;
    coeffs.pi_0 = 1.2f;
    coeffs.pi_m = 1.8f;
    coeffs.theta = 220.0f;
    coeffs.sigma = 0.5f;
    coeffs.chi = 2.5f;

    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs));
    DOCTEST_CHECK_NOTHROW(gsm.run());

    std::vector<uint> UUIDs = {UUID};
    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs, UUIDs));
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Multiple BMF Coefficients") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID1 = context.addPatch();
    uint UUID2 = context.addPatch();
    
    BMFcoefficients coeffs1;
    coeffs1.Em = 300.0f;
    coeffs1.i0 = 40.0f;
    coeffs1.k = 250000.0f;
    coeffs1.b = 600.0f;
    
    BMFcoefficients coeffs2;
    coeffs2.Em = 280.0f;
    coeffs2.i0 = 35.0f;
    coeffs2.k = 220000.0f;
    coeffs2.b = 580.0f;

    std::vector<BMFcoefficients> coeffs = {coeffs1, coeffs2};
    std::vector<uint> UUIDs = {UUID1, UUID2};
    
    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs, UUIDs));
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Dynamic Time Constants") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID1 = context.addPatch();
    uint UUID2 = context.addPatch();
    
    float tau_open = 120.0f;
    float tau_close = 300.0f;

    DOCTEST_CHECK_NOTHROW(gsm.setDynamicTimeConstants(tau_open, tau_close));

    std::vector<uint> UUIDs = {UUID1, UUID2};
    DOCTEST_CHECK_NOTHROW(gsm.setDynamicTimeConstants(tau_open, tau_close, UUIDs));

    float dt = 60.0f;
    DOCTEST_CHECK_NOTHROW(gsm.run(dt));
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs, dt));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Library Functions") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    std::vector<uint> UUIDs = {UUID};

    DOCTEST_CHECK_NOTHROW(gsm.setBMFCoefficientsFromLibrary("Almond"));
    DOCTEST_CHECK_NOTHROW(gsm.setBMFCoefficientsFromLibrary("Apple", UUIDs));

    BMFcoefficients coeffs;
    DOCTEST_CHECK_NOTHROW(coeffs = gsm.getBMFCoefficientsFromLibrary("Almond"));
    DOCTEST_CHECK(coeffs.Em > 0.0f);
    DOCTEST_CHECK(coeffs.i0 > 0.0f);
    DOCTEST_CHECK(coeffs.k > 0.0f);
    DOCTEST_CHECK(coeffs.b > 0.0f);
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Output Functions") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    std::vector<uint> UUIDs = {UUID};

    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("Ci"));
    DOCTEST_CHECK_NOTHROW(gsm.printDefaultValueReport());
    DOCTEST_CHECK_NOTHROW(gsm.printDefaultValueReport(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Input Validation") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    context.setPrimitiveData(UUID, "temperature", 200.0f);
    context.setPrimitiveData(UUID, "air_temperature", 200.0f);
    context.setPrimitiveData(UUID, "air_pressure", 30000.0f);
    context.setPrimitiveData(UUID, "air_humidity", 1.5f);
    context.setPrimitiveData(UUID, "boundarylayer_conductance", -0.5f);
    
    BMFcoefficients coeffs;
    gsm.setModelCoefficients(coeffs);
    
    DOCTEST_CHECK_NOTHROW(gsm.run());
    
    std::vector<uint> UUIDs = {UUID};
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Unknown Species") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();
    
    BMFcoefficients coeffs;
    DOCTEST_CHECK_NOTHROW(coeffs = gsm.getBMFCoefficientsFromLibrary("UnknownSpecies"));
    DOCTEST_CHECK(coeffs.Em == doctest::Approx(865.52f).epsilon(0.01));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Dynamic Model Error Conditions") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    std::vector<uint> UUIDs_test2 = {UUID};
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs_test2, 60.0f));
    
    gsm.setDynamicTimeConstants(30.0f, 30.0f);
    context.setPrimitiveData(UUID, "moisture_conductance", 0.1f);
    std::vector<uint> UUIDs_test3 = {UUID};
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs_test3, 100.0f));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Coefficient Size Mismatch") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID1 = context.addPatch();
    uint UUID2 = context.addPatch();
    
    BMFcoefficients coeff1;
    std::vector<BMFcoefficients> coeffs = {coeff1};
    std::vector<uint> UUIDs = {UUID1, UUID2};
    
    DOCTEST_CHECK_THROWS(gsm.setModelCoefficients(coeffs, UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Optional Output Data") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();
    
    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("invalid_label"));
    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("model_parameters"));
    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("vapor_pressure_deficit"));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Alternative Boundary Layer Data") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    context.setPrimitiveData(UUID, "boundarylayer_conductance_out", 2.0f);
    
    BMFcoefficients coeffs;
    gsm.setModelCoefficients(coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Non-existent Primitive") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    std::vector<uint> UUIDs = {999};
    
    BMFcoefficients coeffs;
    gsm.setModelCoefficients(coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.run(UUIDs));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Missing Photosynthesis Data") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    BWBcoefficients coeffs;
    gsm.setModelCoefficients(coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Edge Cases and Error Conditions") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    context.setPrimitiveData(UUID, "radiation_flux_PAR", -10.0f);
    context.setPrimitiveData(UUID, "net_photosynthesis", -5.0f);
    context.setPrimitiveData(UUID, "temperature", 100.0f);
    context.setPrimitiveData(UUID, "air_temperature", 500.0f);
    context.setPrimitiveData(UUID, "air_CO2", -100.0f);
    context.setPrimitiveData(UUID, "air_humidity", 2.0f);
    context.setPrimitiveData(UUID, "air_pressure", 10000.0f);
    
    BWBcoefficients bwb_coeffs;
    gsm.setModelCoefficients(bwb_coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    BBLcoefficients bbl_coeffs;
    gsm.setModelCoefficients(bbl_coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    MOPTcoefficients mopt_coeffs;
    gsm.setModelCoefficients(mopt_coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    BMFcoefficients bmf_coeffs;
    gsm.setModelCoefficients(bmf_coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    BBcoefficients bb_coeffs;
    gsm.setModelCoefficients(bb_coeffs);
    context.setPrimitiveData(UUID, "xylem_water_potential", -2.0f);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Constructor Edge Cases") {
    Context context;
    
    DOCTEST_CHECK_NOTHROW(StomatalConductanceModel gsm(&context));
    
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();
    gsm.enableMessages();
    gsm.disableMessages();
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Coefficient Overwriting") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    BWBcoefficients coeffs1;
    gsm.setModelCoefficients(coeffs1, {UUID});
    
    BWBcoefficients coeffs2;
    coeffs2.gs0 = 0.1f;
    coeffs2.a1 = 15.0f;
    DOCTEST_CHECK_NOTHROW(gsm.setModelCoefficients(coeffs2, {UUID}));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Additional Coverage Tests") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    // Test all model coefficient types with specific data conditions
    context.setPrimitiveData(UUID, "radiation_flux_PAR", 500.0f);
    context.setPrimitiveData(UUID, "net_photosynthesis", 20.0f);
    context.setPrimitiveData(UUID, "temperature", 298.15f);
    context.setPrimitiveData(UUID, "air_temperature", 298.15f);
    context.setPrimitiveData(UUID, "air_CO2", 400.0f);
    context.setPrimitiveData(UUID, "air_humidity", 0.6f);
    context.setPrimitiveData(UUID, "air_pressure", 101325.0f);
    context.setPrimitiveData(UUID, "boundarylayer_conductance", 1.5f);
    context.setPrimitiveData(UUID, "Gamma_CO2", 45.0f);
    
    // Test BWB model with full data
    BWBcoefficients bwb;
    bwb.gs0 = 0.08f;
    bwb.a1 = 9.0f;
    gsm.setModelCoefficients(bwb);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    // Test BBL model with full data
    BBLcoefficients bbl;
    bbl.gs0 = 0.075f;
    bbl.a1 = 4.5f;
    bbl.D0 = 15000.0f;
    gsm.setModelCoefficients(bbl);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    // Test MOPT model with full data
    MOPTcoefficients mopt;
    mopt.gs0 = 0.085f;
    mopt.g1 = 2.8f;
    gsm.setModelCoefficients(mopt);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    // Test BMF model with PAR data
    context.setPrimitiveData(UUID, "radiation_flux_PAR", 100.0f);
    BMFcoefficients bmf;
    bmf.Em = 250.0f;
    bmf.i0 = 35.0f;
    bmf.k = 200000.0f;
    bmf.b = 580.0f;
    gsm.setModelCoefficients(bmf);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
    
    // Test BB model with xylem potential
    context.setPrimitiveData(UUID, "xylem_water_potential", -1.5f);
    BBcoefficients bb;
    bb.pi_0 = 0.8f;
    bb.pi_m = 1.5f;
    bb.theta = 200.0f;
    bb.sigma = 0.4f;
    bb.chi = 2.2f;
    gsm.setModelCoefficients(bb);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Dynamic Model Comprehensive") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    // Set up initial conditions for dynamic model
    context.setPrimitiveData(UUID, "moisture_conductance", 0.15f);
    context.setPrimitiveData(UUID, "radiation_flux_PAR", 200.0f);
    context.setPrimitiveData(UUID, "net_photosynthesis", 15.0f);
    context.setPrimitiveData(UUID, "temperature", 295.0f);
    context.setPrimitiveData(UUID, "air_temperature", 295.0f);
    context.setPrimitiveData(UUID, "air_CO2", 380.0f);
    context.setPrimitiveData(UUID, "air_humidity", 0.65f);
    context.setPrimitiveData(UUID, "Gamma_CO2", 42.0f);
    
    BWBcoefficients coeffs;
    gsm.setModelCoefficients(coeffs);
    
    // Set dynamic time constants
    gsm.setDynamicTimeConstants(60.0f, 180.0f);
    
    // Test dynamic model with different time steps
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}, 30.0f));
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}, 90.0f));
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}, 200.0f));
    
    // Test with different moisture conductance values to trigger different branches
    context.setPrimitiveData(UUID, "moisture_conductance", 0.05f);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}, 45.0f));
    
    context.setPrimitiveData(UUID, "moisture_conductance", 0.25f);
    DOCTEST_CHECK_NOTHROW(gsm.run(std::vector<uint>{UUID}, 45.0f));
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Species Library Coverage") {
    Context context;
    StomatalConductanceModel gsm(&context);
    gsm.disableMessages();

    uint UUID = context.addPatch();
    
    // Test multiple species
    DOCTEST_CHECK_NOTHROW(gsm.setBMFCoefficientsFromLibrary("Apple"));
    DOCTEST_CHECK_NOTHROW(gsm.setBMFCoefficientsFromLibrary("Almond"));
    DOCTEST_CHECK_NOTHROW(gsm.setBMFCoefficientsFromLibrary("Apple", std::vector<uint>{UUID}));
    
    // Test getting coefficients for different species
    BMFcoefficients apple_coeffs;
    DOCTEST_CHECK_NOTHROW(apple_coeffs = gsm.getBMFCoefficientsFromLibrary("Apple"));
    DOCTEST_CHECK(apple_coeffs.Em > 0.0f);
    
    BMFcoefficients almond_coeffs;
    DOCTEST_CHECK_NOTHROW(almond_coeffs = gsm.getBMFCoefficientsFromLibrary("Almond"));
    DOCTEST_CHECK(almond_coeffs.Em > 0.0f);
    
    // They should be different
    DOCTEST_CHECK(apple_coeffs.Em != almond_coeffs.Em);
}

DOCTEST_TEST_CASE("StomatalConductanceModel - Message Control and Output") {
    Context context;
    StomatalConductanceModel gsm(&context);
    
    uint UUID = context.addPatch();
    
    // Test message control
    gsm.enableMessages();
    gsm.disableMessages();
    gsm.enableMessages();
    
    // Test optional output with various labels
    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("gs"));
    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("ci"));
    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("vpd"));
    DOCTEST_CHECK_NOTHROW(gsm.optionalOutputPrimitiveData("unknown_label"));
    
    // Test report functions
    DOCTEST_CHECK_NOTHROW(gsm.printDefaultValueReport());
    DOCTEST_CHECK_NOTHROW(gsm.printDefaultValueReport(std::vector<uint>{UUID}));
    
    // Test with models set up
    BMFcoefficients coeffs;
    gsm.setModelCoefficients(coeffs);
    DOCTEST_CHECK_NOTHROW(gsm.printDefaultValueReport());
}