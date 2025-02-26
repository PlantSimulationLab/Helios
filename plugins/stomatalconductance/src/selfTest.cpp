#include "StomatalConductanceModel.h"

using namespace std;
using namespace helios;

int StomatalConductanceModel::selfTest() {
    Context context_selftest;
    std::cout << "Running stomatal conductance model self-test..." << std::flush;
    int error_count = 0;

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

    if (sqrtf(RMSE_BWB) > RMSE_max || sqrtf(RMSE_BBL) > RMSE_max || sqrtf(RMSE_MOPT) > RMSE_max || sqrtf(RMSE_BMF) > RMSE_max) {
        std::cout << "FAILED: Stomatal Conductance Model error exceeds RMSE_max." << std::endl;
        error_count++;
    }

    std::cout << "Existing tests passed." << std::endl;

    // Additional tests for untested functions
    std::cout << "Testing setModelCoefficients with UUIDs..." << std::endl;
    {
        uint UUID1 = context_selftest.addPatch();
        uint UUID2 = context_selftest.addPatch();

        BWBcoefficients bwb_coeffs;
        bwb_coeffs.gs0 = 0.1f;
        bwb_coeffs.a1 = 10.0f;

        std::vector<uint> UUIDs = {UUID1, UUID2};
        this->setModelCoefficients(bwb_coeffs, UUIDs);

        if (!context_selftest.doesPrimitiveExist(UUID1) || !context_selftest.doesPrimitiveExist(UUID2)) {
            std::cout << "FAILED: setModelCoefficients with UUIDs." << std::endl;
            error_count++;
        }
    }

    std::cout << "Testing setDynamicTimeConstants with UUIDs..." << std::endl;
    {
        uint UUID1 = context_selftest.addPatch();
        uint UUID2 = context_selftest.addPatch();

        std::vector<uint> UUIDs = {UUID1, UUID2};
        this->setDynamicTimeConstants(5.0f, 3.0f, UUIDs);

        if (!context_selftest.doesPrimitiveExist(UUID1) || !context_selftest.doesPrimitiveExist(UUID2)) {
            std::cout << "FAILED: setDynamicTimeConstants with UUIDs." << std::endl;
            error_count++;
        }
    }

    std::cout << "Testing run with dt..." << std::endl;
    {
        uint UUID1 = context_selftest.addPatch();
        context_selftest.setPrimitiveData(UUID1, "moisture_conductance", 0.1f);

        this->run(0.1f);

        float result = 0.0f;
        context_selftest.getPrimitiveData(UUID1, "moisture_conductance", result);
        if (result <= 0.0f) {
            std::cout << "FAILED: run with dt." << std::endl;
            error_count++;
        }
    }

    {
        uint UUID1 = context_selftest.addPatch();
        uint UUID2 = context_selftest.addPatch();

        BBLcoefficients bbl_coeffs;
        bbl_coeffs.gs0 = 0.2f;
        bbl_coeffs.a1 = 8.5f;
        bbl_coeffs.D0 = 15000.f;

        std::vector<uint> UUIDs = {UUID1, UUID2};
        gsm.setModelCoefficients(bbl_coeffs, UUIDs);

        if (!context_selftest.doesPrimitiveExist(UUID1) || !context_selftest.doesPrimitiveExist(UUID2)) {
            std::cout << "FAILED: setModelCoefficients (BBL) with UUIDs." << std::endl;
            error_count++;
        }
    }

    {
        uint UUID1 = context_selftest.addPatch();
        uint UUID2 = context_selftest.addPatch();

        MOPTcoefficients mopt_coeffs;
        mopt_coeffs.gs0 = 0.15f;
        mopt_coeffs.g1 = 3.0f;

        std::vector<uint> UUIDs = {UUID1, UUID2};
        gsm.setModelCoefficients(mopt_coeffs, UUIDs);

        if (!context_selftest.doesPrimitiveExist(UUID1) || !context_selftest.doesPrimitiveExist(UUID2)) {
            std::cout << "FAILED: setModelCoefficients (MOPT) with UUIDs." << std::endl;
            error_count++;
        }
    }

    {
        uint UUID1 = context_selftest.addPatch();
        uint UUID2 = context_selftest.addPatch();

        BMFcoefficients bmf_coeffs;
        bmf_coeffs.Em = 270.f;
        bmf_coeffs.i0 = 40.f;
        bmf_coeffs.k = 240000.f;
        bmf_coeffs.b = 650.f;

        std::vector<uint> UUIDs = {UUID1, UUID2};
        gsm.setModelCoefficients(bmf_coeffs, UUIDs);

        if (!context_selftest.doesPrimitiveExist(UUID1) || !context_selftest.doesPrimitiveExist(UUID2)) {
            std::cout << "FAILED: setModelCoefficients (BMF) with UUIDs." << std::endl;
            error_count++;
        }
    }

    {
        BBcoefficients bb_coeffs;
        bb_coeffs.pi_0 = 1.2f;
        bb_coeffs.pi_m = 1.8f;
        bb_coeffs.theta = 250.f;
        bb_coeffs.sigma = 0.5f;
        bb_coeffs.chi = 2.2f;

        gsm.setModelCoefficients(bb_coeffs);

        std::cout << "setModelCoefficients (BB) test completed." << std::endl;
    }

    {
        gsm.setDynamicTimeConstants(10.0f, 5.0f);
        std::cout << "setDynamicTimeConstants test completed." << std::endl;

    }

    {
        uint UUID1 = context_selftest.addPatch();
        context_selftest.setPrimitiveData(UUID1, "moisture_conductance", 0.2f);

        std::vector<uint> UUIDs = {UUID1};
        gsm.run(UUIDs, 0.5f);

        float result = 0.0f;
        context_selftest.getPrimitiveData(UUID1, "moisture_conductance", result);
        if (result <= 0.0f) {
            std::cout << "FAILED: run with UUIDs and dt." << std::endl;
            error_count++;
        }
    }

    {
        std::vector<float> variables = {21.81, 400, 101325, 0.55, 3.5, 1.0};
        BWBcoefficients coeffs;
        float result = gsm.evaluate_BWBmodel(1000, variables, &coeffs);

        if (result == 0.0f) {
            std::cout << "FAILED: evaluate_BWBmodel test." << std::endl;
            error_count++;
        }
    }

    {
        std::vector<float> variables = {21.81, 400, 43.7395, 101325, 0.55, 3.5, 101300, 1.0};
        BBLcoefficients coeffs;
        float result = gsm.evaluate_BBLmodel(1000, variables, &coeffs);

        if (result == 0.0f) {
            std::cout << "FAILED: evaluate_BBLmodel test." << std::endl;
            error_count++;
        }
    }

    {
        std::vector<float> variables = {21.81, 400, 101325, 0.55, 3.5, 1.0};
        MOPTcoefficients coeffs;
        float result = gsm.evaluate_MOPTmodel(1000, variables, &coeffs);

        if (result == 0.0f) {
            std::cout << "FAILED: evaluate_MOPTmodel test." << std::endl;
            error_count++;
        }
    }

    {
        std::vector<float> variables = {21.81, 400, 101325, 0.55, 3.5, 1.0};
        MOPTcoefficients coeffs;
        float result = gsm.evaluate_MOPTmodel(1000, variables, &coeffs);

        if (result == 0.0f) {
            std::cout << "FAILED: evaluate_MOPTmodel test." << std::endl;
            error_count++;
        }
    }


    std::cout << "All additional self-tests passed." << std::endl;
    return 0;
}
