#include "PhotosynthesisModel.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>

using namespace std;
using namespace helios;

float err_tol = 1e-3;

DOCTEST_TEST_CASE("PhotosynthesisModel Farquhar Model Type Setting") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    DOCTEST_CHECK_NOTHROW(photomodel.setModelType_Farquhar());
    
    FarquharModelCoefficients coeffs = photomodel.getFarquharModelCoefficients(UUID);
    DOCTEST_CHECK(coeffs.Vcmax == -1.0f); // Default uninitialized value
    DOCTEST_CHECK(coeffs.Jmax == -1.0f); // Default uninitialized value
}

DOCTEST_TEST_CASE("PhotosynthesisModel Light Response Curve - Farquhar Model") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    float Qin[9] = {0, 50, 100, 200, 400, 800, 1200, 1500, 2000};
    std::vector<float> AQ_expected{-2.37932, 8.29752, 12.5566, 16.2075, 16.7448, 16.7448, 16.7448, 16.7448, 16.7448};

    // Set Farquhar model coefficients
    FarquharModelCoefficients fcoeffs;
    fcoeffs.setVcmax(78.5f, 65.33f);
    fcoeffs.setJmax(150.f, 43.54f);
    fcoeffs.setRd(2.12f, 46.39f);
    fcoeffs.setQuantumEfficiency_alpha(0.45f);

    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(fcoeffs));

    for (int i = 0; i < 9; i++) {
        DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "radiation_flux_PAR", Qin[i]));
        DOCTEST_CHECK_NOTHROW(photomodel.run());
        
        float A;
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
        DOCTEST_CHECK(A == doctest::Approx(AQ_expected[i]).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("PhotosynthesisModel CO2 Response Curve - Farquhar Model") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    float CO2[9] = {100, 200, 300, 400, 500, 600, 700, 800, 1000};
    std::vector<float> ACi_expected{1.72714, 7.32672, 12.4749, 17.199, 21.5281, 25.4923, 28.4271, 29.656, 31.3813};

    FarquharModelCoefficients fcoeffs;
    fcoeffs.setVcmax(78.5f, 65.33f);
    fcoeffs.setJmax(150.f, 43.54f);
    fcoeffs.setRd(2.12f, 46.39f);
    fcoeffs.setQuantumEfficiency_alpha(0.45f);

    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(fcoeffs));
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 2000.0f)); // High light

    for (int i = 0; i < 9; i++) {
        DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "air_CO2", CO2[i]));
        DOCTEST_CHECK_NOTHROW(photomodel.run());
        
        float A;
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
        DOCTEST_CHECK(A == doctest::Approx(ACi_expected[i]).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("PhotosynthesisModel Temperature Response Curve - Farquhar Model") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    float TL[7] = {270, 280, 290, 300, 310, 320, 330};
    std::vector<float> AT_expected{3.87863, 8.74928, 14.4075, 17.199, 16.0928, 11.6431, 4.00821};

    FarquharModelCoefficients fcoeffs;
    fcoeffs.setVcmax(78.5f, 65.33f);
    fcoeffs.setJmax(150.f, 43.54f);
    fcoeffs.setRd(2.12f, 46.39f);
    fcoeffs.setQuantumEfficiency_alpha(0.45f);

    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(fcoeffs));
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 2000.0f)); // High light
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "air_CO2", 400.0f)); // Normal CO2

    for (int i = 0; i < 7; i++) {
        DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "temperature", TL[i]));
        DOCTEST_CHECK_NOTHROW(photomodel.run());
        
        float A;
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
        DOCTEST_CHECK(A == doctest::Approx(AT_expected[i]).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("PhotosynthesisModel Empirical Model Coefficients") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    EmpiricalModelCoefficients emp_coeffs;
    emp_coeffs.Asat = 20.0f;
    
    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(emp_coeffs));

    EmpiricalModelCoefficients retrieved_coeffs = photomodel.getEmpiricalModelCoefficients(UUID);
    DOCTEST_CHECK(retrieved_coeffs.Asat == doctest::Approx(20.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Empirical Model Coefficients with UUIDs") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    EmpiricalModelCoefficients emp_coeffs;
    emp_coeffs.Asat = 20.0f;
    std::vector<uint> UUIDs = {UUID};
    
    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(emp_coeffs, UUIDs));

    EmpiricalModelCoefficients retrieved_coeffs = photomodel.getEmpiricalModelCoefficients(UUID);
    DOCTEST_CHECK(retrieved_coeffs.Asat == doctest::Approx(20.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Farquhar Model Coefficients with UUIDs") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    FarquharModelCoefficients farq_coeffs;
    farq_coeffs.Vcmax = 90.0f;
    std::vector<uint> UUIDs = {UUID};
    
    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(farq_coeffs, UUIDs));

    FarquharModelCoefficients retrieved_coeffs = photomodel.getFarquharModelCoefficients(UUID);
    DOCTEST_CHECK(retrieved_coeffs.Vcmax == doctest::Approx(90.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Invalid Input Handling") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    // Set up model with valid coefficients
    FarquharModelCoefficients fcoeffs;
    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(fcoeffs));

    // Set invalid inputs
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "radiation_flux_PAR", -50.0f)); // Should be clipped to 0
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "temperature", 150.0f)); // Should be replaced with 300K default
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "air_CO2", -10.0f)); // Should be clipped to 0

    DOCTEST_CHECK_NOTHROW(photomodel.run());

    float A_invalid;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A_invalid));

    // Compute expected A using default values (PAR=0, TL=300K, CO2=390ppm)
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 0.0f));
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "temperature", 300.0f));
    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "air_CO2", 390.0f));

    DOCTEST_CHECK_NOTHROW(photomodel.run());

    float A_expected;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A_expected));

    DOCTEST_CHECK(A_invalid == doctest::Approx(A_expected).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Optional Output Primitive Data") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    DOCTEST_CHECK_NOTHROW(photomodel.optionalOutputPrimitiveData("Ci"));
    DOCTEST_CHECK_NOTHROW(photomodel.optionalOutputPrimitiveData("limitation_state"));
    DOCTEST_CHECK_NOTHROW(photomodel.optionalOutputPrimitiveData("Gamma_CO2"));

    DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 400.0f));
    DOCTEST_CHECK_NOTHROW(photomodel.run());

    float Ci, Gamma;
    int limitation_state;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "Ci", Ci));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "limitation_state", limitation_state));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "Gamma_CO2", Gamma));

    DOCTEST_CHECK(Ci != 0.0f);
    DOCTEST_CHECK(Gamma != 0.0f);
}

DOCTEST_TEST_CASE("PhotosynthesisModel Print Default Value Report") {
    Context context_test;
    PhotosynthesisModel photomodel(&context_test);

    // This should not throw and should execute successfully
    DOCTEST_CHECK_NOTHROW(photomodel.printDefaultValueReport());
}

DOCTEST_TEST_CASE("PhotosynthesisModel Empirical Model Type Setting") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    DOCTEST_CHECK_NOTHROW(photomodel.setModelType_Empirical());
    
    // After setting empirical model, should be able to get empirical coefficients
    EmpiricalModelCoefficients coeffs = photomodel.getEmpiricalModelCoefficients(UUID);
    DOCTEST_CHECK(coeffs.Asat > 0); // Should have positive default values
}

DOCTEST_TEST_CASE("PhotosynthesisModel Vector Coefficients with Size Mismatch") {
    Context context_test;
    uint UUID1 = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    uint UUID2 = context_test.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    std::vector<FarquharModelCoefficients> coeffs_vector(2);
    std::vector<uint> UUIDs = {UUID1}; // Only one UUID, but 2 coefficients
    
    // This should print a warning and return without setting coefficients
    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(coeffs_vector, UUIDs));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Vector Coefficients Matching Size") {
    Context context_test;
    uint UUID1 = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    uint UUID2 = context_test.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    std::vector<FarquharModelCoefficients> coeffs_vector(2);
    coeffs_vector[0].setVcmax(80.0f);
    coeffs_vector[1].setVcmax(90.0f);
    std::vector<uint> UUIDs = {UUID1, UUID2};
    
    DOCTEST_CHECK_NOTHROW(photomodel.setModelCoefficients(coeffs_vector, UUIDs));
    
    FarquharModelCoefficients retrieved1 = photomodel.getFarquharModelCoefficients(UUID1);
    FarquharModelCoefficients retrieved2 = photomodel.getFarquharModelCoefficients(UUID2);
    DOCTEST_CHECK(retrieved1.getVcmaxTempResponse().value_at_25C == doctest::Approx(80.0f).epsilon(err_tol));
    DOCTEST_CHECK(retrieved2.getVcmaxTempResponse().value_at_25C == doctest::Approx(90.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Library Species") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    // Test getting coefficients from library
    DOCTEST_CHECK_NOTHROW(photomodel.getFarquharCoefficientsFromLibrary("Almond"));
    
    // Test setting coefficients from library for all primitives
    DOCTEST_CHECK_NOTHROW(photomodel.setFarquharCoefficientsFromLibrary("Almond"));
    
    // Test setting coefficients from library for specific UUIDs
    std::vector<uint> UUIDs = {UUID};
    DOCTEST_CHECK_NOTHROW(photomodel.setFarquharCoefficientsFromLibrary("Apple", UUIDs));
    
    // Verify coefficients were set correctly
    FarquharModelCoefficients almond_coeffs = photomodel.getFarquharModelCoefficients(UUID);
    DOCTEST_CHECK(almond_coeffs.getVcmaxTempResponse().value_at_25C == doctest::Approx(101.08f).epsilon(err_tol)); // Apple Vcmax
}

DOCTEST_TEST_CASE("PhotosynthesisModel Message Control") {
    Context context_test;
    PhotosynthesisModel photomodel(&context_test);

    // Test disabling and enabling messages
    DOCTEST_CHECK_NOTHROW(photomodel.disableMessages());
    DOCTEST_CHECK_NOTHROW(photomodel.enableMessages());
}

DOCTEST_TEST_CASE("PhotosynthesisModel Default Value Reports") {
    Context context_test;
    uint UUID1 = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    uint UUID2 = context_test.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    // Test printing default value report for all primitives
    DOCTEST_CHECK_NOTHROW(photomodel.printDefaultValueReport());
    
    // Test printing default value report for specific UUIDs
    std::vector<uint> UUIDs = {UUID1, UUID2};
    DOCTEST_CHECK_NOTHROW(photomodel.printDefaultValueReport(UUIDs));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Run with Specific UUIDs") {
    Context context_test;
    uint UUID1 = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    uint UUID2 = context_test.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    // Set up some basic data
    context_test.setPrimitiveData(UUID1, "radiation_flux_PAR", 500.0f);
    context_test.setPrimitiveData(UUID2, "radiation_flux_PAR", 600.0f);
    
    // Test running model on specific UUIDs
    std::vector<uint> UUIDs = {UUID1};
    DOCTEST_CHECK_NOTHROW(photomodel.run(UUIDs));
    
    float A1, A2;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID1, "net_photosynthesis", A1));
    
    // UUID2 shouldn't have been processed by run(UUIDs), but run() without arguments processes all
    DOCTEST_CHECK_NOTHROW(photomodel.run());
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID2, "net_photosynthesis", A2));
    
    DOCTEST_CHECK(A1 != 0.0f);
    DOCTEST_CHECK(A2 != 0.0f);
}

DOCTEST_TEST_CASE("PhotosyntheticTemperatureResponseParameters Constructors") {
    // Test default constructor
    PhotosyntheticTemperatureResponseParameters params_default;
    DOCTEST_CHECK(params_default.value_at_25C == doctest::Approx(100.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_default.dHa == doctest::Approx(60.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_default.dHd == doctest::Approx(600.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_default.Topt == doctest::Approx(10000.0f).epsilon(err_tol));
    
    // Test single parameter constructor with value only
    PhotosyntheticTemperatureResponseParameters params_single(75.0f);
    DOCTEST_CHECK(params_single.value_at_25C == doctest::Approx(75.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_single.dHa == doctest::Approx(0.0f).epsilon(err_tol));
    
    // Test constructor with value and negative dHa (should trigger else branch)
    PhotosyntheticTemperatureResponseParameters params_negative(80.0f, -10.0f);
    DOCTEST_CHECK(params_negative.value_at_25C == doctest::Approx(80.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_negative.dHa == doctest::Approx(-10.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_negative.dHd == doctest::Approx(600.0f).epsilon(err_tol)); // Should use default
    
    // Test constructor with value, dHa, and optimum temperature  
    PhotosyntheticTemperatureResponseParameters params_three(90.0f, 50.0f, 35.0f);
    DOCTEST_CHECK(params_three.value_at_25C == doctest::Approx(90.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_three.dHa == doctest::Approx(50.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_three.dHd == doctest::Approx(500.0f).epsilon(err_tol)); // 10 * dHa
    DOCTEST_CHECK(params_three.Topt == doctest::Approx(273.15f + 35.0f).epsilon(err_tol));
    
    // Test constructor with value, negative dHa, and optimum temperature (triggers else branch)
    PhotosyntheticTemperatureResponseParameters params_three_neg(85.0f, -15.0f, 32.0f);
    DOCTEST_CHECK(params_three_neg.value_at_25C == doctest::Approx(85.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_three_neg.dHa == doctest::Approx(-15.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_three_neg.dHd == doctest::Approx(600.0f).epsilon(err_tol)); // Should use default 600
    DOCTEST_CHECK(params_three_neg.Topt == doctest::Approx(273.15f + 32.0f).epsilon(err_tol));
    
    // Test full constructor with all parameters
    PhotosyntheticTemperatureResponseParameters params_full(95.0f, 45.0f, 30.0f, 400.0f);
    DOCTEST_CHECK(params_full.value_at_25C == doctest::Approx(95.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_full.dHa == doctest::Approx(45.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_full.dHd == doctest::Approx(400.0f).epsilon(err_tol));
    DOCTEST_CHECK(params_full.Topt == doctest::Approx(273.15f + 30.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("FarquharModelCoefficients Temperature Response Methods") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    FarquharModelCoefficients coeffs;
    
    // Test all the temperature response setter methods and getters
    
    // Test Jmax with optimum temperature (3 parameter version)
    coeffs.setJmax(150.0f, 43.5f, 35.0f);
    PhotosyntheticTemperatureResponseParameters jmax_params = coeffs.getJmaxTempResponse();
    DOCTEST_CHECK(jmax_params.value_at_25C == doctest::Approx(150.0f).epsilon(err_tol));
    DOCTEST_CHECK(jmax_params.dHa == doctest::Approx(43.5f).epsilon(err_tol));
    
    // Test Jmax with optimum temperature and specified rate of decrease (4 parameter version)
    coeffs.setJmax(160.0f, 45.0f, 40.0f, 500.0f);
    jmax_params = coeffs.getJmaxTempResponse();
    DOCTEST_CHECK(jmax_params.dHd == doctest::Approx(500.0f).epsilon(err_tol));
    
    // Test Vcmax with optimum temperature (3 parameter version) 
    coeffs.setVcmax(100.0f, 65.0f, 32.0f);
    PhotosyntheticTemperatureResponseParameters vcmax_params = coeffs.getVcmaxTempResponse();
    DOCTEST_CHECK(vcmax_params.value_at_25C == doctest::Approx(100.0f).epsilon(err_tol));
    
    // Test Vcmax with optimum temperature and specified rate of decrease (4 parameter version)
    coeffs.setVcmax(110.0f, 70.0f, 38.0f, 600.0f);
    vcmax_params = coeffs.getVcmaxTempResponse();
    DOCTEST_CHECK(vcmax_params.dHd == doctest::Approx(600.0f).epsilon(err_tol));
    
    // Test TPU methods
    coeffs.setTPU(8.0f);
    PhotosyntheticTemperatureResponseParameters tpu_params = coeffs.getTPUTempResponse();
    DOCTEST_CHECK(tpu_params.value_at_25C == doctest::Approx(8.0f).epsilon(err_tol));
    DOCTEST_CHECK(coeffs.TPU_flag == 1);
    
    coeffs.setTPU(9.0f, 25.0f);
    tpu_params = coeffs.getTPUTempResponse();
    DOCTEST_CHECK(tpu_params.dHa == doctest::Approx(25.0f).epsilon(err_tol));
    
    coeffs.setTPU(10.0f, 30.0f, 33.0f);
    tpu_params = coeffs.getTPUTempResponse();
    DOCTEST_CHECK(tpu_params.Topt == doctest::Approx(273.15f + 33.0f).epsilon(err_tol));
    
    coeffs.setTPU(11.0f, 35.0f, 36.0f, 450.0f);
    tpu_params = coeffs.getTPUTempResponse();
    DOCTEST_CHECK(tpu_params.dHd == doctest::Approx(450.0f).epsilon(err_tol));
    
    // Test Rd methods
    coeffs.setRd(2.5f, 46.0f, 34.0f);
    PhotosyntheticTemperatureResponseParameters rd_params = coeffs.getRdTempResponse();
    DOCTEST_CHECK(rd_params.value_at_25C == doctest::Approx(2.5f).epsilon(err_tol));
    
    coeffs.setRd(2.8f, 48.0f, 37.0f, 480.0f);
    rd_params = coeffs.getRdTempResponse();
    DOCTEST_CHECK(rd_params.dHd == doctest::Approx(480.0f).epsilon(err_tol));
    
    // Test quantum efficiency alpha methods
    coeffs.setQuantumEfficiency_alpha(0.4f, 20.0f);
    PhotosyntheticTemperatureResponseParameters alpha_params = coeffs.getQuantumEfficiencyTempResponse();
    DOCTEST_CHECK(alpha_params.dHa == doctest::Approx(20.0f).epsilon(err_tol));
    
    coeffs.setQuantumEfficiency_alpha(0.5f, 25.0f, 30.0f);
    alpha_params = coeffs.getQuantumEfficiencyTempResponse();
    DOCTEST_CHECK(alpha_params.Topt == doctest::Approx(273.15f + 30.0f).epsilon(err_tol));
    
    coeffs.setQuantumEfficiency_alpha(0.6f, 30.0f, 32.0f, 350.0f);
    alpha_params = coeffs.getQuantumEfficiencyTempResponse();
    DOCTEST_CHECK(alpha_params.dHd == doctest::Approx(350.0f).epsilon(err_tol));
    
    // Test light response curvature theta methods
    coeffs.setLightResponseCurvature_theta(0.1f, 15.0f);
    PhotosyntheticTemperatureResponseParameters theta_params = coeffs.getLightResponseCurvatureTempResponse();
    DOCTEST_CHECK(theta_params.dHa == doctest::Approx(15.0f).epsilon(err_tol));
    
    coeffs.setLightResponseCurvature_theta(0.2f, 20.0f, 35.0f);
    theta_params = coeffs.getLightResponseCurvatureTempResponse();
    DOCTEST_CHECK(theta_params.Topt == doctest::Approx(273.15f + 35.0f).epsilon(err_tol));
    
    coeffs.setLightResponseCurvature_theta(0.3f, 25.0f, 38.0f, 400.0f);
    theta_params = coeffs.getLightResponseCurvatureTempResponse();
    DOCTEST_CHECK(theta_params.dHd == doctest::Approx(400.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("PhotosynthesisModel Edge Cases and Error Conditions") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);
    
    // Test with unknown optional output primitive data
    DOCTEST_CHECK_NOTHROW(photomodel.optionalOutputPrimitiveData("unknown_primitive"));
    
    // Test running model with no coefficients set (should use defaults)
    DOCTEST_CHECK_NOTHROW(photomodel.run());
    
    // Test getting coefficients for empirical model when none are set
    EmpiricalModelCoefficients emp_default = photomodel.getEmpiricalModelCoefficients(UUID);
    DOCTEST_CHECK(emp_default.Asat == doctest::Approx(18.18f).epsilon(err_tol)); // Default value
    
    // Test getting Farquhar coefficients when none are set  
    FarquharModelCoefficients farq_default = photomodel.getFarquharModelCoefficients(UUID);
    DOCTEST_CHECK(farq_default.Vcmax == doctest::Approx(-1.0f).epsilon(err_tol)); // Uninitialized value
    
    // Test with extreme input values that trigger warnings
    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", -100.0f); // Negative PAR
    context_test.setPrimitiveData(UUID, "temperature", 150.0f); // Very low temperature
    context_test.setPrimitiveData(UUID, "air_CO2", -50.0f); // Negative CO2
    context_test.setPrimitiveData(UUID, "moisture_conductance", -0.1f); // Negative moisture conductance
    context_test.setPrimitiveData(UUID, "boundarylayer_conductance", -1.0f); // Negative boundary layer conductance
    
    DOCTEST_CHECK_NOTHROW(photomodel.run());
    
    // Verify the model still produces reasonable output despite bad inputs
    float A;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
    DOCTEST_CHECK(!std::isnan(A)); // Should not be NaN
}

DOCTEST_TEST_CASE("FarquharModelCoefficients TPU Testing") {
    Context context_test;
    PhotosynthesisModel photomodel(&context_test);
    
    // Test that TPU flag is properly set and cleared
    FarquharModelCoefficients coeffs;
    DOCTEST_CHECK(coeffs.TPU_flag == 0); // Should start as 0
    
    // Test all TPU setting methods set the flag to 1
    coeffs.setTPU(5.0f);
    DOCTEST_CHECK(coeffs.TPU_flag == 1);
    
    coeffs.setTPU(6.0f, 20.0f);
    DOCTEST_CHECK(coeffs.TPU_flag == 1);
    
    coeffs.setTPU(7.0f, 25.0f, 30.0f);
    DOCTEST_CHECK(coeffs.TPU_flag == 1);
    
    coeffs.setTPU(8.0f, 30.0f, 35.0f, 400.0f);
    DOCTEST_CHECK(coeffs.TPU_flag == 1);
}

DOCTEST_TEST_CASE("PhotosynthesisModel Complex Farquhar Testing") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);
    
    // Test complex Farquhar model with TPU limitation
    FarquharModelCoefficients coeffs;
    coeffs.setVcmax(100.0f, 65.0f);
    coeffs.setJmax(200.0f, 43.0f);
    coeffs.setRd(2.0f, 46.0f);
    coeffs.setQuantumEfficiency_alpha(0.4f);
    coeffs.setTPU(8.0f); // Enable TPU limitation
    
    photomodel.setModelCoefficients(coeffs);
    
    // Set conditions that might trigger different limitation states
    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 1800.0f);
    context_test.setPrimitiveData(UUID, "temperature", 298.15f); // 25C
    context_test.setPrimitiveData(UUID, "air_CO2", 300.0f);
    
    // Enable optional outputs to trigger more code paths
    photomodel.optionalOutputPrimitiveData("Ci");
    photomodel.optionalOutputPrimitiveData("limitation_state");
    photomodel.optionalOutputPrimitiveData("Gamma_CO2");
    
    DOCTEST_CHECK_NOTHROW(photomodel.run());
    
    // Verify outputs exist
    float A, Ci, Gamma;
    int limitation_state;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "Ci", Ci));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "Gamma_CO2", Gamma));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "limitation_state", limitation_state));
    
    // Verify reasonable values
    DOCTEST_CHECK(A != 0.0f);
    DOCTEST_CHECK(Ci > 0.0f);
    DOCTEST_CHECK(Gamma > 0.0f);
    DOCTEST_CHECK(limitation_state >= 0);
}

DOCTEST_TEST_CASE("PhotosynthesisModel Empirical Model Edge Cases") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);
    
    // Test empirical model with extreme conditions
    EmpiricalModelCoefficients empirical_coeffs;
    empirical_coeffs.Asat = 25.0f;
    empirical_coeffs.theta = 70.0f;
    
    photomodel.setModelCoefficients(empirical_coeffs);
    
    // Test with very high light
    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 3000.0f);
    context_test.setPrimitiveData(UUID, "temperature", 310.0f); // High temperature
    context_test.setPrimitiveData(UUID, "air_CO2", 800.0f); // High CO2
    
    DOCTEST_CHECK_NOTHROW(photomodel.run());
    
    float A;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
    DOCTEST_CHECK(!std::isnan(A));
    
    // Test with zero light
    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 0.0f);
    DOCTEST_CHECK_NOTHROW(photomodel.run());
    
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
    DOCTEST_CHECK(A < 0.0f); // Should be negative (respiration only)
}

DOCTEST_TEST_CASE("PhotosynthesisModel Temperature Response Edge Cases") {
    Context context_test;
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);
    
    // Test with temperature response that has dHa = 0 (constant response)
    FarquharModelCoefficients coeffs;
    coeffs.setVcmax(100.0f); // Single parameter - constant (dHa = 0)
    coeffs.setJmax(200.0f); // Single parameter - constant (dHa = 0)
    coeffs.setRd(2.0f); // Single parameter - constant (dHa = 0)
    coeffs.setQuantumEfficiency_alpha(0.4f); // Single parameter - constant (dHa = 0)
    
    photomodel.setModelCoefficients(coeffs);
    
    // Test at different temperatures - should give same result due to dHa = 0
    std::vector<float> temperatures = {280.0f, 300.0f, 320.0f};
    std::vector<float> results;
    
    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 800.0f);
    context_test.setPrimitiveData(UUID, "air_CO2", 400.0f);
    
    for (float temp : temperatures) {
        context_test.setPrimitiveData(UUID, "temperature", temp);
        DOCTEST_CHECK_NOTHROW(photomodel.run());
        
        float A;
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "net_photosynthesis", A));
        results.push_back(A);
    }
    
    // Even with dHa = 0, there are other temperature effects (Kc, Ko, Gamma, etc.)
    // So just verify all computations completed without error and gave finite results
    for (float result : results) {
        DOCTEST_CHECK(!std::isnan(result));
        DOCTEST_CHECK(std::isfinite(result));
    }
}

int PhotosynthesisModel::selfTest() {
    // Run all the tests
    doctest::Context context;
    int res = context.run();

    if (context.shouldExit()) {
        return res;
    }

    return res;
}