#include "PhotosynthesisModel.h"
#include "Context.h"
#include <iostream>
#include <cmath>

using namespace std;
using namespace helios;


int PhotosynthesisModel::selfTest() {

    std::cout << "Running photosynthesis model self-test..." << std::flush;
    int error_count = 0;

    Context context_test;
    float errtol = 0.001f;

    // Create patches for testing
    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    PhotosynthesisModel photomodel(&context_test);

    std::vector<float> A;

    // --- Test setModelType_Farquhar --- //
    photomodel.setModelType_Farquhar();
    if (photomodel.getFarquharModelCoefficients(UUID).Vcmax == 0) {
        std::cout << "failed: setModelType_Farquhar not applied correctly." << std::endl;
        error_count++;
    } else {
        std::cout << "setModelType_Farquhar test passed." << std::endl;
    }

    // --- Light Response Curve: Empirical Model --- //
    float Qin[9] = {0, 50, 100, 200, 400, 800, 1200, 1500, 2000};
    A.resize(9);
    std::vector<float> AQ_expected{-2.39479, 8.30612, 12.5873, 16.2634, 16.6826, 16.6826, 16.6826, 16.6826, 16.6826};

    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "radiation_flux_PAR", Qin[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
    }

    // --- Light Response Curve: Farquhar Model --- //
    FarquharModelCoefficients fcoeffs; // Default parameters for Farquhar model
    fcoeffs.Vcmax = 78.5;
    fcoeffs.Jmax = 150;
    fcoeffs.alpha = 0.45;
    fcoeffs.Rd = 2.12;
    fcoeffs.c_Jmax = 17.57;
    fcoeffs.dH_Jmax = 43.54;

    photomodel.setModelCoefficients(fcoeffs);

    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "radiation_flux_PAR", Qin[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
        if (fabs(A.at(i) - AQ_expected.at(i)) / fabs(AQ_expected.at(i)) > errtol) {
            std::cout << "failed: Farquhar light response curve test." << std::endl;
            error_count++;
        }
    }

    // --- A vs. Ci Curve: Empirical Model --- //
    float CO2[9] = {100, 200, 300, 400, 500, 600, 700, 800, 1000};
    A.resize(9);

    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", Qin[8]);
    photomodel.setModelType_Empirical();

    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "air_CO2", CO2[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
    }

    // --- A vs. Ci Curve: Farquhar Model --- //
    std::vector<float> ACi_expected{1.70787, 7.29261, 12.426, 17.1353, 21.4501, 25.4004, 28.5788, 29.8179, 31.5575};

    photomodel.setModelCoefficients(fcoeffs);
    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "air_CO2", CO2[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
        if (fabs(A.at(i) - ACi_expected.at(i)) / fabs(ACi_expected.at(i)) > errtol) {
            std::cout << "failed: CO2 response curve test." << std::endl;
            error_count++;
        }
    }

    // --- A vs. Temperature Curve: Empirical Model --- //
    float TL[7] = {270, 280, 290, 300, 310, 320, 330};
    A.resize(7);

    context_test.setPrimitiveData(UUID, "air_CO2", CO2[3]);
    photomodel.setModelType_Empirical();

    for (int i = 0; i < 7; i++) {
        context_test.setPrimitiveData(UUID, "temperature", TL[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
    }

    // --- A vs. Temperature Curve: Farquhar Model --- //
    std::vector<float> AT_expected{3.8609, 8.71169, 14.3514, 17.1353, 16.0244, 11.5661, 3.91437};

    photomodel.setModelCoefficients(fcoeffs);
    for (int i = 0; i < 7; i++) {
        context_test.setPrimitiveData(UUID, "temperature", TL[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
        if (fabs(A.at(i) - AT_expected.at(i)) / fabs(AT_expected.at(i)) > errtol) {
            std::cout << "failed: temperature response curve test." << std::endl;
            error_count++;
        }
    }

    // --- Test setModelCoefficients (Empirical Model) --- //
    EmpiricalModelCoefficients emp_coeffs;
    emp_coeffs.Asat = 20.0f;
    photomodel.setModelCoefficients(emp_coeffs);

    EmpiricalModelCoefficients coeff1 = photomodel.getEmpiricalModelCoefficients(UUID);
    if (coeff1.Asat != 20.0f) {
        std::cout << "failed: setModelCoefficients (Empirical Model)." << std::endl;
        error_count++;
    }

    // --- Test setModelCoefficients (Empirical Model with UUIDs) --- //
    std::vector<uint> UUIDs = {UUID};
    photomodel.setModelCoefficients(emp_coeffs, UUIDs);

    coeff1 = photomodel.getEmpiricalModelCoefficients(UUID);
    if (coeff1.Asat != 20.0f) {
        std::cout << "failed: setModelCoefficients (Empirical Model with UUIDs)." << std::endl;
        error_count++;
    }

    // --- Test setModelCoefficients (Farquhar Model with UUIDs) --- //
    FarquharModelCoefficients farq_coeffs;
    farq_coeffs.Vcmax = 90.0f;
    photomodel.setModelCoefficients(farq_coeffs, UUIDs);

    FarquharModelCoefficients coeff2 = photomodel.getFarquharModelCoefficients(UUID);
    if (coeff2.Vcmax != 90.0f) {
        std::cout << "failed: setModelCoefficients (Farquhar Model with UUIDs)." << std::endl;
        error_count++;
    }

    // --- Edge Cases: Invalid Input Handling (Fixed) --- //
    {
        context_test.setPrimitiveData(UUID, "radiation_flux_PAR", -50.0f);  // Should be clipped to 0
        context_test.setPrimitiveData(UUID, "temperature", 150.0f);         // Should be replaced with 300K default
        context_test.setPrimitiveData(UUID, "air_CO2", -10.0f);             // Should be clipped to 0

        photomodel.run();

        float A_invalid;
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A_invalid);

        // Compute expected A using default values (PAR=0, TL=300K, CO2=390ppm)
        context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 0);
        context_test.setPrimitiveData(UUID, "temperature", 300.0f);
        context_test.setPrimitiveData(UUID, "air_CO2", 390.0f);

        photomodel.run();

        float A_expected;
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A_expected);

        if (fabs(A_invalid - A_expected) < errtol) {
            std::cout << "Invalid input handling in run() correctly applied default values." << std::endl;
        } else {
            std::cout << "failed: invalid input handling in run(). Expected A=" << A_expected << " but got " << A_invalid << std::endl;
            error_count++;
        }
    }

    // --- Test optionalOutputPrimitiveData --- //
    photomodel.optionalOutputPrimitiveData("Ci");
    photomodel.optionalOutputPrimitiveData("limitation_state");
    photomodel.optionalOutputPrimitiveData("Gamma_CO2");

    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", 400);
    photomodel.run();

    float Ci, Gamma;
    int limitation_state;
    context_test.getPrimitiveData(UUID, "Ci", Ci);
    context_test.getPrimitiveData(UUID, "limitation_state", limitation_state);
    context_test.getPrimitiveData(UUID, "Gamma_CO2", Gamma);

    if (Ci == 0 || Gamma == 0) {
        std::cout << "failed: optionalOutputPrimitiveData." << std::endl;
        error_count++;
    }

    // --- Test printDefaultValueReport --- //
    photomodel.printDefaultValueReport();

    std::cout << "printDefaultValueReport executed successfully." << std::endl;

    // --- Final Test Result --- //
    if (error_count == 0) {
        std::cout << "All self-tests passed successfully!" << std::endl;
        return 0;
    } else {
        std::cout << "Self-test failed with " << error_count << " errors." << std::endl;
        return 1;
    }
}