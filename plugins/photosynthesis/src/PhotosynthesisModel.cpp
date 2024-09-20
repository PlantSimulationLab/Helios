/** \file "PhotosynthesisModel.cpp" Primary source file for photosynthesis plug-in.

Copyright (C) 2016-2023 Brian Bailey

    This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include <complex>
#include "PhotosynthesisModel.h"

using namespace std;
using namespace helios;

PhotosynthesisModel::PhotosynthesisModel(helios::Context *a_context) {
    context = a_context;

    //default values set here
    model = "farquhar";

    i_PAR_default = 0;
    TL_default = 300;
    CO2_default = 390;
    gM_default = 0.25;
    gH_default = 1;


}

int PhotosynthesisModel::selfTest() {

    std::cout << "Running photosynthesis model self-test..." << std::flush;

    Context context_test;

    float errtol = 0.001f;

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    PhotosynthesisModel photomodel(&context_test);

    std::vector<float> A;

    float Qin[9] = {0, 50, 100, 200, 400, 800, 1200, 1500, 2000};
    A.resize(9);
    std::vector<float> AQ_expected{-2.401840, 8.309921, 12.600988, 16.287579, 16.907225, 16.907225, 16.907225, 16.907225, 16.907225};

//Generate a light response curve using empirical model with default parmeters
    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "radiation_flux_PAR", Qin[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
    }

//Generate a light response curve using Farquhar model

    FarquharModelCoefficients fcoeffs; //these are prior default model parameters, which is what was used for this test

    fcoeffs.setVcmax(78.5);
    fcoeffs.setJmax(150, 43.54);
    fcoeffs.setRd(2.12);
    fcoeffs.setQuantumEfficiency_alpha(0.45);


    photomodel.setModelCoefficients(fcoeffs);

    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "radiation_flux_PAR", Qin[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
        if (fabs(A.at(i) - AQ_expected.at(i)) / fabs(AQ_expected.at(i)) > errtol) {
            std::cout << "failed. Incorrect light response curve." << std::endl;
            return 1;
        }
    }

//Generate an A vs Ci curve using empirical model with default parameters

    float CO2[9] = {100, 200, 300, 400, 500, 600, 700, 800, 1000};
    A.resize(9);

    context_test.setPrimitiveData(UUID, "radiation_flux_PAR", Qin[8]);

    photomodel.setModelType_Empirical();
    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "air_CO2", CO2[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
    }

//Generate an A vs Ci curve using Farquhar model with default parameters

    std::vector<float> ACi_expected{1.746968, 7.395710, 12.593264, 17.366213, 21.743221, 25.753895, 28.645109, 29.888800, 31.634666};

    photomodel.setModelCoefficients(fcoeffs);
    for (int i = 0; i < 9; i++) {
        context_test.setPrimitiveData(UUID, "air_CO2", CO2[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
        if (fabs(A.at(i) - ACi_expected.at(i)) / fabs(ACi_expected.at(i)) > errtol) {
            std::cout << "failed. Incorrect CO2 response curve." << std::endl;
            return 1;
        }
    }

//Generate an A vs temperature curve using empirical model with default parameters

    float TL[7] = {270, 280, 290, 300, 310, 320, 330};
    A.resize(7);

    context_test.setPrimitiveData(UUID, "air_CO2", CO2[3]);

    photomodel.setModelType_Empirical();
    for (int i = 0; i < 7; i++) {
        context_test.setPrimitiveData(UUID, "temperature", TL[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
    }

//Generate an A vs temperature curve using Farquhar model with default parameters

    std::vector<float> AT_expected{3.930926, 8.861267, 14.569322, 17.366213, 16.240763, 11.758138, 4.073297};

    photomodel.setModelCoefficients(fcoeffs);
    for (int i = 0; i < 7; i++) {
        context_test.setPrimitiveData(UUID, "temperature", TL[i]);
        photomodel.run();
        context_test.getPrimitiveData(UUID, "net_photosynthesis", A[i]);
        if (fabs(A.at(i) - AT_expected.at(i)) / fabs(AT_expected.at(i)) > errtol) {
            std::cout << "failed. Incorrect temperature response curve." << std::endl;
            //return 1;
        }
    }

    std::cout << "passed." << std::endl;

    return 0;
}

void PhotosynthesisModel::setModelType_Empirical() {
    model = "empirical";
}

void PhotosynthesisModel::setModelType_Farquhar() {
    model = "farquhar";
}


void PhotosynthesisModel::setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients) {
    empiricalmodelcoeffs = modelcoefficients;
    empiricalmodel_coefficients.clear();
    model = "empirical";
}

void PhotosynthesisModel::setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients,
                                               const std::vector<uint> &UUIDs) {
    for (uint UUID: UUIDs) {
        empiricalmodel_coefficients[UUID] = modelcoefficients;
    }
    model = "empirical";
}

void PhotosynthesisModel::setModelCoefficients(const FarquharModelCoefficients &modelcoefficients) {
    farquharmodelcoeffs = modelcoefficients;
    farquharmodel_coefficients.clear();
    model = "farquhar";
}

void PhotosynthesisModel::setModelCoefficients(const FarquharModelCoefficients &modelcoefficients,
                                               const std::vector<uint> &UUIDs) {
    for (uint UUID: UUIDs) {
        farquharmodel_coefficients[UUID] = modelcoefficients;
    }
    model = "farquhar";
}

void PhotosynthesisModel::setModelCoefficients(const std::vector<FarquharModelCoefficients> &modelcoefficients, const std::vector<uint> &UUIDs) {
    if (modelcoefficients.size() != UUIDs.size()) {
        printf("WARNING (PhotosynthesisModel::setModelCoefficients): number of model coefficients (%zu) does not match number of UUIDs (%zu).",
               modelcoefficients.size(), UUIDs.size());
        return;
    }
    for (uint i = 0; i < UUIDs.size(); i++) {
        farquharmodel_coefficients[UUIDs.at(i)] = modelcoefficients.at(i);
    }
    model = "farquhar";
}

void PhotosynthesisModel::setFarquharCoefficientsFromLibrary(const std::string &species) {
    FarquharModelCoefficients fmc;
    fmc = getFarquharCoefficientsFromLibrary(species);
    farquharmodelcoeffs = fmc;
    farquharmodel_coefficients.clear();
    model = "farquhar";
}

void PhotosynthesisModel::setFarquharCoefficientsFromLibrary(const std::string &species, const std::vector<uint> &UUIDs) {
    FarquharModelCoefficients fmc;
    fmc = getFarquharCoefficientsFromLibrary(species);
    for (uint UUID: UUIDs) {
        farquharmodel_coefficients[UUID] = fmc;
    }
    model = "farquhar";
}

FarquharModelCoefficients PhotosynthesisModel::getFarquharCoefficientsFromLibrary(const std::string &species) {
    std::string s = std::move(species);
    FarquharModelCoefficients fmc;
    bool defaultSpecies = false;
    if (s == "Almond" || s == "almond") {
        fmc.setVcmax(105.9);
        fmc.setJmax(166.34, 46.36);
        fmc.setRd(1.49);
        fmc.setQuantumEfficiency_alpha(0.336);
    } else if (s == "Apple" || s == "apple") {
        fmc.setVcmax(101.08);
        fmc.setJmax(167.03, 47.62);
        fmc.setRd(3.00);
        fmc.setQuantumEfficiency_alpha(0.432);
    } else if (s == "Cherry" || s == "cherry") {
        fmc.setVcmax(75.65);
        fmc.setJmax(129.06, 48.49);
        fmc.setRd(2.12);
        fmc.setQuantumEfficiency_alpha(0.404);
    } else if (s == "Prune" || s == "prune") {
        fmc.setVcmax(75.88);
        fmc.setJmax(129.41, 48.58);
        fmc.setRd(1.56);
        fmc.setQuantumEfficiency_alpha(0.402);
    } else if (s == "Pear" || s == "pear") {
        fmc.setVcmax(107.69);
        fmc.setJmax(176.71, 46.04);
        fmc.setRd(1.510);
        fmc.setQuantumEfficiency_alpha(0.274);
    } else if (s == "PistachioFemale" || s == "pistachiofemale" || s == "pistachio_female" ||
               s == "Pistachio_Female" || s == "Pistachio_female" || s == "pistachio" || s == "Pistachio") {
        fmc.setVcmax(138.99);
        fmc.setJmax(221.76, 43.80);
        fmc.setRd(2.850);
        fmc.setQuantumEfficiency_alpha(0.366);
    } else if (s == "PistachioMale" || s == "pistachiomale" || s == "pistachio_male" ||
               s == "Pistachio_Male" || s == "Pistachio_male") {
        fmc.setVcmax(154.17);
        fmc.setJmax(243.20, 50.89);
        fmc.setRd(2.050);
        fmc.setQuantumEfficiency_alpha(0.335);
    } else if (s == "Walnut" || s == "walnut") {
        fmc.setVcmax(121.85);
        fmc.setJmax(197.25,48.35);
        fmc.setRd(1.960);
        fmc.setQuantumEfficiency_alpha(0.404);
    } else if (s == "Grape" || s == "grape") {
        //cv. Cabernet Sauvignon
        fmc.setVcmax(74.5, 76.1, 318.8 - 273.15, 499.8);
        fmc.setJmax(180.2, 23.0, 313.8 - 273.15, 502.3);
        fmc.setTPU(7.7, 24.0, 314.6 - 273.15, 496.4);
        fmc.setRd(1.3);
        fmc.setQuantumEfficiency_alpha(0.304);
        fmc.setLightResponseCurvature_theta(0.06);
    } else if (s == "Elderberry" || s == "elderberry" || s == "blue_elderberry") {
        fmc.setVcmax(37.7, 66.0, 319.4 - 273.15, 496.0);
        fmc.setJmax(149.7, 24.5, 314.8 - 273.15, 492.9);
        fmc.setTPU(7.3, 33.6, 314.5 - 273.15, 497.5);
        fmc.setRd(1.3);
        fmc.setQuantumEfficiency_alpha(0.202);
        fmc.setLightResponseCurvature_theta(0.472);
    } else if (s == "Toyon" || s == "toyon") {
        fmc.setVcmax(52.8, 42.1, 315.1 - 273.15, 483.0);
        fmc.setJmax(142.4, 9.0, 313.0 - 273.15, 486.2);
        fmc.setTPU(6.6, 14.0, 314.8 - 273.15, 493.8);
        fmc.setRd(0.8);
        fmc.setQuantumEfficiency_alpha(0.290);
        fmc.setLightResponseCurvature_theta(0.532);
    } else if (s == "Big_Leaf_Maple" || s == "big_leaf_maple" || s == "Maple" || s == "maple") {
        fmc.setVcmax(96.4, 48.9, 307.1 - 273.15, 505.0);
        fmc.setJmax(168.0, 8.5, 304.7 - 273.15, 476.7);
        fmc.setTPU(2.7, 32.1, 308.3 - 273.15, 471.6);
        fmc.setRd(0.1);
        fmc.setQuantumEfficiency_alpha(0.077);
    } else if (s == "Western_Redbud" || s == "western_redbud" || s == "Redbud" || s == "redbud") {
        fmc.setVcmax(68.5, 66.6, 315.1 - 273.15, 496.0);
        fmc.setJmax(132.4, 41.2, 313.1 - 273.15, 474.0);
        fmc.setTPU(6.6, 34.3, 312.8 - 273.15, 463.2);
        fmc.setRd(0.8);
        fmc.setQuantumEfficiency_alpha(0.41);
        fmc.setLightResponseCurvature_theta(0.04);
    } else if (s == "Baylaurel" || s == "baylaurel" || s == "Bay_Laurel" || s == "bay_laurel" || s == "bay" ||
               s == "Bay") {
        fmc.setVcmax(97.5, 49.1, 308.6 - 273.15, 505.8);
        fmc.setJmax(193.0, 34.0, 308.5 - 273.15, 456.7);
        fmc.setTPU(3.3, 0.1, 309.4 - 273.15, 477.5);
        fmc.setRd(0.1);
        fmc.setQuantumEfficiency_alpha(0.037);
    } else if (s == "Olive" || s == "olive") {
        fmc.setVcmax(75.9, 55.4, 315.2 - 273.15, 497.0);
        fmc.setJmax(170.4, 32.2, 312.5 - 273.15, 493.4);
        fmc.setTPU(8.3, 37.2, 311.7 - 273.15, 498.9);
        fmc.setRd(1.9);
        fmc.setQuantumEfficiency_alpha(0.398);
    } else if (s == "EasternRedbudSunlit" || s == "easternredbudsunlit" || s == "easternredbud_sunlit" ||
               s == "EasternRedbud_Sunlit" || s == "EasternRedbud_sunlit" || s == "sunlitEasternRedbud" ||
               s == "SunlitEasternRedbud" || s == "eastern_redbud_sunlit" || s == "Eastern_Redbud_Sunlit") {
        fmc.setVcmax(104.35, 54.9927, 313.8828 - 273.15, 365.1581);
        fmc.setJmax(211.3090, 46.5415, 310.9710 - 273.15, 200.3197);
        fmc.setTPU(9.9965, 48.4469, 310.3164 - 273.15, 167.9181);
        fmc.setRd(1.4136);
        fmc.setQuantumEfficiency_alpha(0.4151);
    } else if (s == "EasternRedbudShaded" || s == "easternredbudshaded" || s == "easternredbud_shaded" ||
               s == "EasternRedbud_Shaded" || s == "EasternRedbud_shaded" || s == "shadedEasternRedbud" ||
               s == "ShadedEasternRedbud" || s == "eastern_redbud_shaded" || s == "Eastern_Redbud_Shaded") {
        fmc.setVcmax(84.7);
        fmc.setJmax(190.53, 0.249);
        fmc.setTPU(9.53, 0.0747);
        fmc.setRd(1.13);
        fmc.setQuantumEfficiency_alpha(0.713);
    } else {
        defaultSpecies = true;
        std::cout << "WARNING (PhotosynthesisModel::getModelCoefficients): unknown species " << s
                  << ". Setting default (Almond)." << std::endl;
    }
    if (!defaultSpecies) {
        std::cout << "Setting Photosynthesis Model Coefficients to " << s << std::endl;
    }
    return fmc;
}

void PhotosynthesisModel::run() {
    run(context->getAllUUIDs());
}

void PhotosynthesisModel::run(const std::vector<uint> &lUUIDs) {

    for (uint UUID: lUUIDs) {

        float i_PAR;
        if (context->doesPrimitiveDataExist(UUID, "radiation_flux_PAR") &&
            context->getPrimitiveDataType(UUID, "radiation_flux_PAR") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "radiation_flux_PAR", i_PAR);
            i_PAR = i_PAR *
                    4.57f; //umol/m^2-s (ref https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf)
            if (i_PAR < 0) {
                i_PAR = 0;
                std::cout << "WARNING (runPhotosynthesis): PAR flux value provided was negative.  Clipping to zero."
                          << std::endl;
            }
        } else {
            i_PAR = i_PAR_default;
        }

        float TL;
        if (context->doesPrimitiveDataExist(UUID, "temperature") &&
            context->getPrimitiveDataType(UUID, "temperature") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "temperature", TL);
            if (TL < 200) {
                std::cout << "WARNING (PhotosynthesisModel::run): Primitive temperature value was very low (" << TL
                          << "K). Assuming. Are you using absolute temperature units?" << std::endl;
                TL = TL_default;
            }
        } else {
            TL = TL_default;
        }

        float CO2;
        if (context->doesPrimitiveDataExist(UUID, "air_CO2") &&
            context->getPrimitiveDataType(UUID, "air_CO2") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "air_CO2", CO2);
            if (CO2 < 0) {
                CO2 = 0;
                std::cout
                        << "WARNING (PhotosynthesisModel::run): CO2 concentration value provided was negative. Clipping to zero."
                        << std::endl;
            }
        } else {
            CO2 = CO2_default;
        }

        float gM;
        if (context->doesPrimitiveDataExist(UUID, "moisture_conductance") &&
            context->getPrimitiveDataType(UUID, "moisture_conductance") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "moisture_conductance", gM);
            if (gM < 0) {
                gM = 0;
                std::cout
                        << "WARNING (PhotosynthesisModel::run): Moisture conductance value provided was negative. Clipping to zero."
                        << std::endl;
            }
        } else {
            gM = gM_default;
        }

        float gH;
        if (context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") &&
            context->getPrimitiveDataType(UUID, "boundarylayer_conductance") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "boundarylayer_conductance", gH);
        } else if (context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out") &&
                   context->getPrimitiveDataType(UUID, "boundarylayer_conductance_out") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);
        } else {
            gH = gH_default;
        }
        if (gH < 0) {
            gH = 0;
            std::cout
                    << "WARNING (PhotosynthesisModel::run): Boundary-layer conductance value provided was negative. Clipping to zero."
                    << std::endl;
        }

        //combine stomatal (gM) and boundary-layer (gH) conductances
        gM = 1.08f * gH * gM / (1.08f * gH + gM);

        float A, Ci, Gamma;
        int limitation_state, TPU_flag = 0;

        if (model == "farquhar") { //Farquhar-von Caemmerer-Berry Model

            FarquharModelCoefficients coeffs;
            if (farquharmodel_coefficients.empty() ||
                farquharmodel_coefficients.find(UUID) == farquharmodel_coefficients.end()) {
                coeffs = farquharmodelcoeffs;
            } else {
                coeffs = farquharmodel_coefficients.at(UUID);
            }
            A = evaluateFarquharModel(coeffs, i_PAR, TL, CO2, gM, Ci, Gamma, limitation_state, TPU_flag);

        } else { //Empirical Model

            EmpiricalModelCoefficients coeffs;
            if (empiricalmodel_coefficients.empty() ||
                empiricalmodel_coefficients.find(UUID) == empiricalmodel_coefficients.end()) {
                coeffs = empiricalmodelcoeffs;
            } else {
                coeffs = empiricalmodel_coefficients.at(UUID);
            }

            A = evaluateEmpiricalModel(coeffs, i_PAR, TL, CO2, gM);

        }

        if (A == 0) {
            std::cout << "WARNING (PhotosynthesisModel::run): Solution did not converge for primitive " << UUID << "."
                      << std::endl;
        }

        context->setPrimitiveData(UUID, "net_photosynthesis", HELIOS_TYPE_FLOAT, 1, &A);

        for (const auto &data: output_prim_data) {
            if (data == "Ci" && model == "farquhar") {
                context->setPrimitiveData(UUID, "Ci", Ci);
            } else if (data == "limitation_state" && model == "farquhar") {
                context->setPrimitiveData(UUID, "limitation_state", limitation_state);
            } else if (data == "Gamma_CO2" && model == "farquhar") {
                context->setPrimitiveData(UUID, "Gamma_CO2", Gamma);
            }
        }

    }


}

float PhotosynthesisModel::evaluateCi_Empirical(const EmpiricalModelCoefficients &params, float Ci, float CO2, float fL,
                                                float Rd, float gM) const {


//--- CO2 Response Function --- //

    float fC = params.kC * Ci / params.Ci_ref;


//--- Assimilation Rate --- //

    float A = params.Asat * fL * fC - Rd;

//--- Calculate error and update --- //

    float resid = 0.75f * gM * (CO2 - Ci) - A;


    return resid;

}

float
PhotosynthesisModel::evaluateEmpiricalModel(const EmpiricalModelCoefficients &params, float i_PAR, float TL, float CO2,
                                            float gM) {

//initial guess for intercellular CO2
    float Ci = CO2;

//--- Light Response Function --- //

    float fL = i_PAR / (params.theta + i_PAR);

    assert(fL >= 0 && fL <= 1);

//--- Respiration Rate --- //

    float Rd = params.R * sqrt(TL - 273.f) * exp(-params.ER / TL);

    float Ci_old = Ci;
    float Ci_old_old = 0.95f * Ci;

    float resid_old = evaluateCi_Empirical(params, Ci_old, CO2, fL, Rd, gM);
    float resid_old_old = evaluateCi_Empirical(params, Ci_old_old, CO2, fL, Rd, gM);

    float err = 10000, err_max = 0.01;
    int iter = 0, max_iter = 100;
    float resid;
    while (err > err_max && iter < max_iter) {

        if (resid_old == resid_old_old) {//this condition will cause NaN
            break;
        }

        Ci = fabs((Ci_old_old * resid_old - Ci_old * resid_old_old) / (resid_old - resid_old_old));

        resid = evaluateCi_Empirical(params, Ci, CO2, fL, Rd, gM);

        resid_old_old = resid_old;
        resid_old = resid;

        err = fabs(resid);

        Ci_old_old = Ci_old;
        Ci_old = Ci;

        iter++;

    }

    float A;
    if (err > err_max) {
        A = 0;
    } else {
        float fC = params.kC * Ci / params.Ci_ref;
        A = params.Asat * fL * fC - Rd;
    }

    return A;

}

float PhotosynthesisModel::evaluateCi_Farquhar(float Ci, std::vector<float> &variables, const void *parameters) {

    const FarquharModelCoefficients modelcoeffs = *reinterpret_cast<const FarquharModelCoefficients *>(parameters);

    float Vcmax, Jmax, TPU, Rd, alpha, theta;

    float CO2 = variables[0];
    float Q = variables[1];
    float TL = variables[2];
    float gM = variables[3];
    int TPUflag = int(variables[4]);

    float R = 0.0083144598;   //molar gas constant (kJ/K/mol)
    float O = 213.5; //ambient oxygen concentration (mmol/mol)
    float c_Gamma = 19.02;
    float dH_Gamma = 37.83;
    float c_Kc = 38.05;
    float dH_Kc = 79.43;
    float c_Ko = 20.30;
    float dH_Ko = 36.38;

    float RT = R * TL;
    float invDiffRT = (1.f / 298.15f - 1.f / TL) / RT;

    if (modelcoeffs.Vcmax > 0) {
        Vcmax = modelcoeffs.Vcmax * expf(modelcoeffs.dH_Vcmax * (invDiffRT));
    } else {
        Vcmax = respondToTemperature(&modelcoeffs.VcmaxTempResponse, TL);
    }

    if (modelcoeffs.Jmax > 0) {
        Jmax = modelcoeffs.Jmax * expf(modelcoeffs.dH_Jmax * (invDiffRT));
    } else {
        Jmax = respondToTemperature(&modelcoeffs.JmaxTempResponse, TL);
    }

    if (modelcoeffs.Rd > 0) {
        Rd = modelcoeffs.Rd * expf(modelcoeffs.dH_Rd * (invDiffRT));
    } else {
        Rd = respondToTemperature(&modelcoeffs.RdTempResponse, TL);
    }

    if (modelcoeffs.alpha > 0) {
        alpha = modelcoeffs.alpha;
    } else {
        alpha = respondToTemperature(&modelcoeffs.alphaTempResponse, TL);
    }


    TPU = respondToTemperature(&modelcoeffs.TPUTempResponse, TL);
    theta = respondToTemperature(&modelcoeffs.thetaTempResponse, TL);


    float Gamma_star = exp(c_Gamma - dH_Gamma / (R * TL));
    float Kc = exp(c_Kc - dH_Kc / (R * TL));
    float Ko = exp(c_Ko - dH_Ko / (R * TL));
    float Kco = Kc * (1.f + O / Ko);

    //thetaJ^2 + -(alphaQ+Jmax)J + alphaQJmax = 0
    double a = std::max(theta, 0.0001f);
    double ia = 1.000f / a;
    double b = -(alpha * Q + Jmax);
    double c = alpha * Q * Jmax;
    double J = (-b - sqrt(pow(b, 2.000f) - 4.f * a * c)) * 0.5f * ia;
    //J = Jmax * alpha * Q / (alpha * Q + Jmax);

    float Wc = Vcmax * Ci / (Ci + Kco);
    float Wj = J * Ci / (4.f * Ci + 8.f * Gamma_star);
    float Wp = 3.f * TPU * Ci / (Ci - Gamma_star);

    float smooth_factor = 0.99f;
    float s = helios::clamp(0.5f + 0.5f * (Wc - Wj) / smooth_factor, 0.0f, 1.0f);
    float smooth_min = Wc * (1.f - s) + Wj * s - smooth_factor * s * (1.f - s);

    if (TPUflag == 1) {
        smooth_factor = 0.99f;
        s = helios::clamp(0.5f + 0.5f * (smooth_min - Wp) / smooth_factor, 0.0f, 1.0f);
        smooth_min = smooth_min * (1.f - s) + Wp * s - smooth_factor * s * (1.f - s);
    }
    float A = (1.f - Gamma_star / Ci) * smooth_min - Rd;


    float limitation_state;
    if (Wj > Wc) { //Rubisco limited
        limitation_state = 0;
    } else { //Electron transport limited
        limitation_state = 1;
    }

//--- Calculate error and update --- //

    float resid = 0.75f * gM * (CO2 - Ci) - A;

    variables[4] = A;
    variables[5] = limitation_state;

    float Gamma = (Gamma_star + Kco * Rd / Vcmax) / (1.f - Rd / Vcmax);  //Equation 39 of Farquhar et al. (1980)
    variables[6] = Gamma;

    return resid;
}


float PhotosynthesisModel::respondToTemperature(const PhotosyntheticTemperatureResponseParameters *params,
                                                float T_in_Kelvin) {
    float T = T_in_Kelvin;
    float R = 0.0083144598f;
    float v25 = params->value_at_25C;
    float dHa = params->dHa;
    float dHd = params->dHd;
    float Topt = params->Topt;
    if (dHa == 0) {
        return v25;
    } else {
        float logterm = logf(dHd / dHa - 1.f);
        float t1 = 1.f + expf(dHd / R * (1.f / Topt - 1.f / 298.f) - logterm);
        float t2 = 1.f + expf(dHd / R * (1.f / Topt - 1.f / T) - logterm);
        return v25 * exp(dHa / R * (1.f / 298 - 1.f / T)) * t1 / t2;
    }
}


float
PhotosynthesisModel::evaluateFarquharModel(const FarquharModelCoefficients &params, float i_PAR, float TL, float CO2,
                                           float gM, float &Ci, float &Gamma, int &limitation_state, int &TPU_flag) {

    float A = 0;
    Ci = 100;
    std::vector<float> variables{CO2, i_PAR, TL, gM, A, float(limitation_state), Gamma, float(TPU_flag)};


    Ci = fzero(evaluateCi_Farquhar, variables, &farquharmodelcoeffs, Ci);

    A = variables[4];
    limitation_state = (int) variables[5];
    Gamma = variables[6];

    return A;

}

EmpiricalModelCoefficients PhotosynthesisModel::getEmpiricalModelCoefficients(uint UUID) {

    EmpiricalModelCoefficients coeffs;
    if (empiricalmodel_coefficients.empty() ||
        empiricalmodel_coefficients.find(UUID) == empiricalmodel_coefficients.end()) {
        coeffs = empiricalmodelcoeffs;
    } else {
        coeffs = empiricalmodel_coefficients.at(UUID);
    }

    return coeffs;

}

FarquharModelCoefficients PhotosynthesisModel::getFarquharModelCoefficients(uint UUID) {

    FarquharModelCoefficients coeffs;
    if (farquharmodel_coefficients.empty() ||
        farquharmodel_coefficients.find(UUID) == farquharmodel_coefficients.end()) {
        coeffs = farquharmodelcoeffs;
    } else {
        coeffs = farquharmodel_coefficients.at(UUID);
    }

    return coeffs;

}

void PhotosynthesisModel::optionalOutputPrimitiveData(const char *label) {

    if (strcmp(label, "Ci") == 0 || strcmp(label, "limitation_state") == 0 || strcmp(label, "Gamma_CO2") == 0) {
        output_prim_data.emplace_back(label);
    } else {
        std::cout << "WARNING (PhotosynthesisModel::optionalOutputPrimitiveData): unknown output primitive data "
                  << label << std::endl;
    }

}

void PhotosynthesisModel::printDefaultValueReport() const {
    printDefaultValueReport(context->getAllUUIDs());
}

void PhotosynthesisModel::printDefaultValueReport(const std::vector<uint> &UUIDs) const {

//    i_PAR_default = 0;
//    TL_default = 300;
//    CO2_default = 390;
//    gM_default = 0.25;
//    gH_default = 1;

    size_t assumed_default_i = 0;
    size_t assumed_default_TL = 0;
    size_t assumed_default_CO2 = 0;
    size_t assumed_default_gM = 0;
    size_t assumed_default_gH = 0;

    size_t Nprimitives = UUIDs.size();

    for (uint UUID: UUIDs) {

        if (!context->doesPrimitiveDataExist(UUID, "radiation_flux_PAR") ||
            context->getPrimitiveDataType(UUID, "radiation_flux_PAR") != HELIOS_TYPE_FLOAT) {
            assumed_default_i++;
        }

        //surface temperature (K)
        if (!context->doesPrimitiveDataExist(UUID, "temperature") ||
            context->getPrimitiveDataType(UUID, "temperature") != HELIOS_TYPE_FLOAT) {
            assumed_default_TL++;
        }

        //boundary-layer conductance to heat
        if ((!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") ||
             context->getPrimitiveDataType(UUID, "boundarylayer_conductance") != HELIOS_TYPE_FLOAT) &&
            (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out") ||
             context->getPrimitiveDataType(UUID, "boundarylayer_conductance_out") != HELIOS_TYPE_FLOAT)) {
            assumed_default_gH++;
        }

        //stomatal conductance
        if (!context->doesPrimitiveDataExist(UUID, "moisture_conductance") ||
            context->getPrimitiveDataType(UUID, "moisture_conductance") != HELIOS_TYPE_FLOAT) {
            assumed_default_gM++;
        }

        //ambient air CO2
        if (!context->doesPrimitiveDataExist(UUID, "air_CO2") ||
            context->getPrimitiveDataType(UUID, "air_CO2") != HELIOS_TYPE_FLOAT) {
            assumed_default_CO2++;
        }

    }

    std::cout << "--- Photosynthesis Model Default Value Report ---" << std::endl;

    std::cout << "PAR flux: " << assumed_default_i << " of " << Nprimitives << " used default value of "
              << i_PAR_default << " because ""radiation_flux_PAR"" primitive data did not exist" << std::endl;
    std::cout << "surface temperature: " << assumed_default_TL << " of " << Nprimitives << " used default value of "
              << TL_default << " because ""temperature"" primitive data did not exist" << std::endl;
    std::cout << "boundary-layer conductance: " << assumed_default_gH << " of " << Nprimitives
              << " used default value of " << gH_default
              << " because ""boundarylayer_conductance"" primitive data did not exist" << std::endl;
    std::cout << "moisture conductance: " << assumed_default_gM << " of " << Nprimitives << " used default value of "
              << gM_default << " because ""moisture_conductance"" primitive data did not exist" << std::endl;
    std::cout << "air CO2: " << assumed_default_CO2 << " of " << Nprimitives << " used default value of " << CO2_default
              << " because ""air_CO2"" primitive data did not exist" << std::endl;

    std::cout << "--------------------------------------------------" << std::endl;

}
