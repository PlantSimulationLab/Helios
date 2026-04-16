/** \file "PhotosynthesisModel.cpp" Primary source file for photosynthesis plug-in.

Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#include "PhotosynthesisModel.h"
#include <complex>

using namespace std;
using namespace helios;

namespace {
    // Material data labels for Empirical model
    constexpr const char *LABEL_EMP_Tref = "photo_emp_Tref";
    constexpr const char *LABEL_EMP_Ci_ref = "photo_emp_Ci_ref";
    constexpr const char *LABEL_EMP_Asat = "photo_emp_Asat";
    constexpr const char *LABEL_EMP_theta = "photo_emp_theta";
    constexpr const char *LABEL_EMP_Tmin = "photo_emp_Tmin";
    constexpr const char *LABEL_EMP_Topt = "photo_emp_Topt";
    constexpr const char *LABEL_EMP_q = "photo_emp_q";
    constexpr const char *LABEL_EMP_R = "photo_emp_R";
    constexpr const char *LABEL_EMP_ER = "photo_emp_ER";
    constexpr const char *LABEL_EMP_kC = "photo_emp_kC";

    // Material data labels for Farquhar model
    constexpr const char *LABEL_FQ_Vcmax = "photo_fq_Vcmax";
    constexpr const char *LABEL_FQ_Jmax = "photo_fq_Jmax";
    constexpr const char *LABEL_FQ_Rd = "photo_fq_Rd";
    constexpr const char *LABEL_FQ_alpha = "photo_fq_alpha";
    constexpr const char *LABEL_FQ_O = "photo_fq_O";
    constexpr const char *LABEL_FQ_TPU_flag = "photo_fq_TPU_flag";
    constexpr const char *LABEL_FQ_c_Rd = "photo_fq_c_Rd";
    constexpr const char *LABEL_FQ_c_Vcmax = "photo_fq_c_Vcmax";
    constexpr const char *LABEL_FQ_c_Jmax = "photo_fq_c_Jmax";
    constexpr const char *LABEL_FQ_c_Gamma = "photo_fq_c_Gamma";
    constexpr const char *LABEL_FQ_c_Kc = "photo_fq_c_Kc";
    constexpr const char *LABEL_FQ_c_Ko = "photo_fq_c_Ko";
    constexpr const char *LABEL_FQ_dH_Rd = "photo_fq_dH_Rd";
    constexpr const char *LABEL_FQ_dH_Vcmax = "photo_fq_dH_Vcmax";
    constexpr const char *LABEL_FQ_dH_Jmax = "photo_fq_dH_Jmax";
    constexpr const char *LABEL_FQ_dH_Gamma = "photo_fq_dH_Gamma";
    constexpr const char *LABEL_FQ_dH_Kc = "photo_fq_dH_Kc";
    constexpr const char *LABEL_FQ_dH_Ko = "photo_fq_dH_Ko";
    constexpr const char *LABEL_FQ_gm = "photo_fq_gm";
    constexpr const char *LABEL_FQ_gm_dHa = "photo_fq_gm_dHa";
    constexpr const char *LABEL_FQ_gm_dHd = "photo_fq_gm_dHd";
    constexpr const char *LABEL_FQ_gm_Topt = "photo_fq_gm_Topt";

    // Material data labels for von Caemmerer (2021) C4 model. Each temperature-responsive parameter
    // serializes its full PhotosyntheticTemperatureResponseParameters quartet (value_at_25C, dHa, dHd, Topt).
    constexpr const char *LABEL_C4_Vpmax = "photo_c4_Vpmax";
    constexpr const char *LABEL_C4_Vpmax_dHa = "photo_c4_Vpmax_dHa";
    constexpr const char *LABEL_C4_Vpmax_dHd = "photo_c4_Vpmax_dHd";
    constexpr const char *LABEL_C4_Vpmax_Topt = "photo_c4_Vpmax_Topt";
    constexpr const char *LABEL_C4_Vcmax = "photo_c4_Vcmax";
    constexpr const char *LABEL_C4_Vcmax_dHa = "photo_c4_Vcmax_dHa";
    constexpr const char *LABEL_C4_Vcmax_dHd = "photo_c4_Vcmax_dHd";
    constexpr const char *LABEL_C4_Vcmax_Topt = "photo_c4_Vcmax_Topt";
    constexpr const char *LABEL_C4_Jmax = "photo_c4_Jmax";
    constexpr const char *LABEL_C4_Jmax_dHa = "photo_c4_Jmax_dHa";
    constexpr const char *LABEL_C4_Jmax_dHd = "photo_c4_Jmax_dHd";
    constexpr const char *LABEL_C4_Jmax_Topt = "photo_c4_Jmax_Topt";
    constexpr const char *LABEL_C4_Rd = "photo_c4_Rd";
    constexpr const char *LABEL_C4_Rd_dHa = "photo_c4_Rd_dHa";
    constexpr const char *LABEL_C4_Rd_dHd = "photo_c4_Rd_dHd";
    constexpr const char *LABEL_C4_Rd_Topt = "photo_c4_Rd_Topt";
    constexpr const char *LABEL_C4_gm = "photo_c4_gm";
    constexpr const char *LABEL_C4_gm_dHa = "photo_c4_gm_dHa";
    constexpr const char *LABEL_C4_gm_dHd = "photo_c4_gm_dHd";
    constexpr const char *LABEL_C4_gm_Topt = "photo_c4_gm_Topt";
    // Kinetic constants
    constexpr const char *LABEL_C4_Kc_25 = "photo_c4_Kc_25";
    constexpr const char *LABEL_C4_Ko_25 = "photo_c4_Ko_25";
    constexpr const char *LABEL_C4_Kp_25 = "photo_c4_Kp_25";
    constexpr const char *LABEL_C4_gamma_star_25 = "photo_c4_gamma_star_25";
    constexpr const char *LABEL_C4_Om_25 = "photo_c4_Om_25";
    constexpr const char *LABEL_C4_dH_Kc = "photo_c4_dH_Kc";
    constexpr const char *LABEL_C4_dH_Ko = "photo_c4_dH_Ko";
    constexpr const char *LABEL_C4_dH_Kp = "photo_c4_dH_Kp";
    constexpr const char *LABEL_C4_dH_gamma_star = "photo_c4_dH_gamma_star";
    constexpr const char *LABEL_C4_dH_Om = "photo_c4_dH_Om";
    // Scalar parameters
    constexpr const char *LABEL_C4_alpha = "photo_c4_alpha";
    constexpr const char *LABEL_C4_x = "photo_c4_x";
    constexpr const char *LABEL_C4_Vpr = "photo_c4_Vpr";
    constexpr const char *LABEL_C4_Rm_frac = "photo_c4_Rm_frac";
    constexpr const char *LABEL_C4_fcyc = "photo_c4_fcyc";
    constexpr const char *LABEL_C4_gbs = "photo_c4_gbs";
    constexpr const char *LABEL_C4_ao = "photo_c4_ao";
    constexpr const char *LABEL_C4_absorptance = "photo_c4_absorptance";
    constexpr const char *LABEL_C4_f_spectral = "photo_c4_f_spectral";
    constexpr const char *LABEL_C4_theta_etr = "photo_c4_theta_etr";
    constexpr const char *LABEL_C4_h_protons = "photo_c4_h_protons";
} // namespace

PhotosynthesisModel::PhotosynthesisModel(helios::Context *a_context) {
    context = a_context;

    // default values set here
    model = "farquhar";

    i_PAR_default = 0;
    TL_default = 300;
    CO2_default = 390;
    gM_default = 0.25;
    gH_default = 1;
}


void PhotosynthesisModel::setModelType_Empirical() {
    model = "empirical";
}

void PhotosynthesisModel::setModelType_Farquhar() {
    model = "farquhar";
}

void PhotosynthesisModel::setModelType_C4() {
    model = "c4";
}


void PhotosynthesisModel::setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients) {
    empiricalmodelcoeffs = modelcoefficients;
    empiricalmodel_coefficients.clear();
    model = "empirical";
}

void PhotosynthesisModel::setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs) {
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

void PhotosynthesisModel::setModelCoefficients(const FarquharModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs) {
    for (uint UUID: UUIDs) {
        farquharmodel_coefficients[UUID] = modelcoefficients;
    }
    model = "farquhar";
}

void PhotosynthesisModel::setModelCoefficients(const std::vector<FarquharModelCoefficients> &modelcoefficients, const std::vector<uint> &UUIDs) {
    if (modelcoefficients.size() != UUIDs.size()) {
        std::cerr << "WARNING (PhotosynthesisModel::setModelCoefficients): number of model coefficients (" + std::to_string(modelcoefficients.size()) + ") does not match number of UUIDs (" + std::to_string(UUIDs.size()) + ")" << std::endl;
        return;
    }
    for (uint i = 0; i < UUIDs.size(); i++) {
        farquharmodel_coefficients[UUIDs.at(i)] = modelcoefficients.at(i);
    }
    model = "farquhar";
}

void PhotosynthesisModel::setModelCoefficients(const C4ModelCoefficients &modelcoefficients) {
    c4modelcoeffs = modelcoefficients;
    c4model_coefficients.clear();
    model = "c4";
}

void PhotosynthesisModel::setModelCoefficients(const C4ModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs) {
    for (uint UUID: UUIDs) {
        c4model_coefficients[UUID] = modelcoefficients;
    }
    model = "c4";
}

C4ModelCoefficients PhotosynthesisModel::getC4ModelCoefficients(uint UUID) {
    return getCoefficientsForPrimitive_C4(UUID);
}

void PhotosynthesisModel::setModelCoefficients(const std::string &material_label, const C4ModelCoefficients &coeffs) {
    // Serialize the four PhotosyntheticTemperatureResponseParameters quartets
    auto write_resp = [&](const char *base, const char *dHa, const char *dHd, const char *Topt, const PhotosyntheticTemperatureResponseParameters &r) {
        context->setMaterialData(material_label, base, r.value_at_25C);
        context->setMaterialData(material_label, dHa, r.dHa);
        context->setMaterialData(material_label, dHd, r.dHd);
        context->setMaterialData(material_label, Topt, r.Topt);
    };
    write_resp(LABEL_C4_Vpmax, LABEL_C4_Vpmax_dHa, LABEL_C4_Vpmax_dHd, LABEL_C4_Vpmax_Topt, coeffs.getVpmaxTempResponse());
    write_resp(LABEL_C4_Vcmax, LABEL_C4_Vcmax_dHa, LABEL_C4_Vcmax_dHd, LABEL_C4_Vcmax_Topt, coeffs.getVcmaxTempResponse());
    write_resp(LABEL_C4_Jmax, LABEL_C4_Jmax_dHa, LABEL_C4_Jmax_dHd, LABEL_C4_Jmax_Topt, coeffs.getJmaxTempResponse());
    write_resp(LABEL_C4_Rd, LABEL_C4_Rd_dHa, LABEL_C4_Rd_dHd, LABEL_C4_Rd_Topt, coeffs.getRdTempResponse());
    write_resp(LABEL_C4_gm, LABEL_C4_gm_dHa, LABEL_C4_gm_dHd, LABEL_C4_gm_Topt, coeffs.getMesophyllConductance_gmTempResponse());

    context->setMaterialData(material_label, LABEL_C4_Kc_25, coeffs.Kc_25);
    context->setMaterialData(material_label, LABEL_C4_Ko_25, coeffs.Ko_25);
    context->setMaterialData(material_label, LABEL_C4_Kp_25, coeffs.Kp_25);
    context->setMaterialData(material_label, LABEL_C4_gamma_star_25, coeffs.gamma_star_25);
    context->setMaterialData(material_label, LABEL_C4_Om_25, coeffs.Om_25);
    context->setMaterialData(material_label, LABEL_C4_dH_Kc, coeffs.dH_Kc);
    context->setMaterialData(material_label, LABEL_C4_dH_Ko, coeffs.dH_Ko);
    context->setMaterialData(material_label, LABEL_C4_dH_Kp, coeffs.dH_Kp);
    context->setMaterialData(material_label, LABEL_C4_dH_gamma_star, coeffs.dH_gamma_star);
    context->setMaterialData(material_label, LABEL_C4_dH_Om, coeffs.dH_Om);

    context->setMaterialData(material_label, LABEL_C4_alpha, coeffs.alpha_psII_fraction);
    context->setMaterialData(material_label, LABEL_C4_x, coeffs.x_etr_partition);
    context->setMaterialData(material_label, LABEL_C4_Vpr, coeffs.Vpr);
    context->setMaterialData(material_label, LABEL_C4_Rm_frac, coeffs.Rm_frac);
    context->setMaterialData(material_label, LABEL_C4_fcyc, coeffs.fcyc);
    context->setMaterialData(material_label, LABEL_C4_gbs, coeffs.gbs);
    context->setMaterialData(material_label, LABEL_C4_ao, coeffs.ao);
    context->setMaterialData(material_label, LABEL_C4_absorptance, coeffs.absorptance);
    context->setMaterialData(material_label, LABEL_C4_f_spectral, coeffs.f_spectral);
    context->setMaterialData(material_label, LABEL_C4_theta_etr, coeffs.theta_etr);
    context->setMaterialData(material_label, LABEL_C4_h_protons, coeffs.h_protons);

    // Invalidate the cache for this material so the next run() picks up the new values.
    uint matID = context->getMaterialIDFromLabel(material_label);
    material_coefficient_cache_c4.erase(matID);

    model = "c4";
}

C4ModelCoefficients PhotosynthesisModel::getCoefficientsForPrimitive_C4(uint UUID) const {
    // Resolution order: material data → cached material → per-UUID map → global default.
    uint materialID = context->getPrimitiveMaterialID(UUID);

    if (material_coefficient_cache_c4.find(materialID) != material_coefficient_cache_c4.end()) {
        return material_coefficient_cache_c4.at(materialID);
    }

    C4ModelCoefficients coeffs;
    bool found_in_material = false;
    try {
        const Material &mat = context->getMaterial(materialID);
        // Use Vpmax presence as the sentinel — if it's set, assume the full C4 quartet is set.
        if (mat.doesMaterialDataExist(LABEL_C4_Vpmax)) {
            // Helper to deserialize a temperature-response quartet via the appropriate setter.
            auto load_resp = [&](const char *base, const char *dHa_lbl, const char *dHd_lbl, const char *Topt_lbl, auto setter_constant, auto setter_arrhenius, auto setter_peaked) {
                float v25 = 0.f, dHa = 0.f, dHd = 0.f, Topt = 0.f;
                mat.getMaterialData(base, v25);
                mat.getMaterialData(dHa_lbl, dHa);
                mat.getMaterialData(dHd_lbl, dHd);
                mat.getMaterialData(Topt_lbl, Topt);
                if (Topt < 1000.f) {
                    setter_peaked(v25, dHa, Topt - 273.15f, dHd);
                } else if (dHa != 0.f) {
                    setter_arrhenius(v25, dHa);
                } else {
                    setter_constant(v25);
                }
            };

            load_resp(LABEL_C4_Vpmax, LABEL_C4_Vpmax_dHa, LABEL_C4_Vpmax_dHd, LABEL_C4_Vpmax_Topt, [&](float v) { coeffs.setVpmax(v); }, [&](float v, float a) { coeffs.setVpmax(v, a); }, [&](float v, float a, float t, float d) { coeffs.setVpmax(v, a, t, d); });
            load_resp(LABEL_C4_Vcmax, LABEL_C4_Vcmax_dHa, LABEL_C4_Vcmax_dHd, LABEL_C4_Vcmax_Topt, [&](float v) { coeffs.setVcmax(v); }, [&](float v, float a) { coeffs.setVcmax(v, a); }, [&](float v, float a, float t, float d) { coeffs.setVcmax(v, a, t, d); });
            load_resp(LABEL_C4_Jmax, LABEL_C4_Jmax_dHa, LABEL_C4_Jmax_dHd, LABEL_C4_Jmax_Topt, [&](float v) { coeffs.setJmax(v); }, [&](float v, float a) { coeffs.setJmax(v, a); }, [&](float v, float a, float t, float d) { coeffs.setJmax(v, a, t, d); });
            load_resp(LABEL_C4_Rd, LABEL_C4_Rd_dHa, LABEL_C4_Rd_dHd, LABEL_C4_Rd_Topt, [&](float v) { coeffs.setRd(v); }, [&](float v, float a) { coeffs.setRd(v, a); }, [&](float v, float a, float t, float d) { coeffs.setRd(v, a, t, d); });
            load_resp(LABEL_C4_gm, LABEL_C4_gm_dHa, LABEL_C4_gm_dHd, LABEL_C4_gm_Topt, [&](float v) { coeffs.setMesophyllConductance_gm(v); }, [&](float v, float a) { coeffs.setMesophyllConductance_gm(v, a); }, [&](float v, float a, float t, float d) { coeffs.setMesophyllConductance_gm(v, a, t, d); });

            // Kinetic constants — public fields, written directly. Use defaults if a label is missing
            // (older serializations may pre-date a field; missing labels keep struct defaults).
            if (mat.doesMaterialDataExist(LABEL_C4_Kc_25)) mat.getMaterialData(LABEL_C4_Kc_25, coeffs.Kc_25);
            if (mat.doesMaterialDataExist(LABEL_C4_Ko_25)) mat.getMaterialData(LABEL_C4_Ko_25, coeffs.Ko_25);
            if (mat.doesMaterialDataExist(LABEL_C4_Kp_25)) mat.getMaterialData(LABEL_C4_Kp_25, coeffs.Kp_25);
            if (mat.doesMaterialDataExist(LABEL_C4_gamma_star_25)) mat.getMaterialData(LABEL_C4_gamma_star_25, coeffs.gamma_star_25);
            if (mat.doesMaterialDataExist(LABEL_C4_Om_25)) mat.getMaterialData(LABEL_C4_Om_25, coeffs.Om_25);
            if (mat.doesMaterialDataExist(LABEL_C4_dH_Kc)) mat.getMaterialData(LABEL_C4_dH_Kc, coeffs.dH_Kc);
            if (mat.doesMaterialDataExist(LABEL_C4_dH_Ko)) mat.getMaterialData(LABEL_C4_dH_Ko, coeffs.dH_Ko);
            if (mat.doesMaterialDataExist(LABEL_C4_dH_Kp)) mat.getMaterialData(LABEL_C4_dH_Kp, coeffs.dH_Kp);
            if (mat.doesMaterialDataExist(LABEL_C4_dH_gamma_star)) mat.getMaterialData(LABEL_C4_dH_gamma_star, coeffs.dH_gamma_star);
            if (mat.doesMaterialDataExist(LABEL_C4_dH_Om)) mat.getMaterialData(LABEL_C4_dH_Om, coeffs.dH_Om);

            if (mat.doesMaterialDataExist(LABEL_C4_alpha)) mat.getMaterialData(LABEL_C4_alpha, coeffs.alpha_psII_fraction);
            if (mat.doesMaterialDataExist(LABEL_C4_x)) mat.getMaterialData(LABEL_C4_x, coeffs.x_etr_partition);
            if (mat.doesMaterialDataExist(LABEL_C4_Vpr)) mat.getMaterialData(LABEL_C4_Vpr, coeffs.Vpr);
            if (mat.doesMaterialDataExist(LABEL_C4_Rm_frac)) mat.getMaterialData(LABEL_C4_Rm_frac, coeffs.Rm_frac);
            if (mat.doesMaterialDataExist(LABEL_C4_fcyc)) mat.getMaterialData(LABEL_C4_fcyc, coeffs.fcyc);
            if (mat.doesMaterialDataExist(LABEL_C4_gbs)) mat.getMaterialData(LABEL_C4_gbs, coeffs.gbs);
            if (mat.doesMaterialDataExist(LABEL_C4_ao)) mat.getMaterialData(LABEL_C4_ao, coeffs.ao);
            if (mat.doesMaterialDataExist(LABEL_C4_absorptance)) mat.getMaterialData(LABEL_C4_absorptance, coeffs.absorptance);
            if (mat.doesMaterialDataExist(LABEL_C4_f_spectral)) mat.getMaterialData(LABEL_C4_f_spectral, coeffs.f_spectral);
            if (mat.doesMaterialDataExist(LABEL_C4_theta_etr)) mat.getMaterialData(LABEL_C4_theta_etr, coeffs.theta_etr);
            if (mat.doesMaterialDataExist(LABEL_C4_h_protons)) mat.getMaterialData(LABEL_C4_h_protons, coeffs.h_protons);

            material_coefficient_cache_c4[materialID] = coeffs;
            found_in_material = true;
        }
    } catch (const std::exception &) {
        // No assigned material — fall through to UUID map / default.
    }

    if (found_in_material) {
        return coeffs;
    }
    if (c4model_coefficients.find(UUID) != c4model_coefficients.end()) {
        return c4model_coefficients.at(UUID);
    }
    return c4modelcoeffs;
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

// Material-based coefficient setters

void PhotosynthesisModel::setModelCoefficients(const std::string &material_label, const EmpiricalModelCoefficients &coeffs) {
    context->setMaterialData(material_label, LABEL_EMP_Tref, coeffs.Tref);
    context->setMaterialData(material_label, LABEL_EMP_Ci_ref, coeffs.Ci_ref);
    context->setMaterialData(material_label, LABEL_EMP_Asat, coeffs.Asat);
    context->setMaterialData(material_label, LABEL_EMP_theta, coeffs.theta);
    context->setMaterialData(material_label, LABEL_EMP_Tmin, coeffs.Tmin);
    context->setMaterialData(material_label, LABEL_EMP_Topt, coeffs.Topt);
    context->setMaterialData(material_label, LABEL_EMP_q, coeffs.q);
    context->setMaterialData(material_label, LABEL_EMP_R, coeffs.R);
    context->setMaterialData(material_label, LABEL_EMP_ER, coeffs.ER);
    context->setMaterialData(material_label, LABEL_EMP_kC, coeffs.kC);

    // Clear cache for this material
    uint matID = context->getMaterialIDFromLabel(material_label);
    material_coefficient_cache_empirical.erase(matID);

    model = "empirical";
}

void PhotosynthesisModel::setModelCoefficients(const std::string &material_label, const FarquharModelCoefficients &coeffs) {
    // Get temperature response parameters
    PhotosyntheticTemperatureResponseParameters vcmax_resp = const_cast<FarquharModelCoefficients &>(coeffs).getVcmaxTempResponse();
    PhotosyntheticTemperatureResponseParameters jmax_resp = const_cast<FarquharModelCoefficients &>(coeffs).getJmaxTempResponse();
    PhotosyntheticTemperatureResponseParameters rd_resp = const_cast<FarquharModelCoefficients &>(coeffs).getRdTempResponse();
    PhotosyntheticTemperatureResponseParameters alpha_resp = const_cast<FarquharModelCoefficients &>(coeffs).getQuantumEfficiencyTempResponse();

    // Serialize temperature response parameters for Vcmax
    context->setMaterialData(material_label, LABEL_FQ_Vcmax, vcmax_resp.value_at_25C);
    context->setMaterialData(material_label, "photo_fq_Vcmax_dHa", vcmax_resp.dHa);
    context->setMaterialData(material_label, "photo_fq_Vcmax_dHd", vcmax_resp.dHd);
    context->setMaterialData(material_label, "photo_fq_Vcmax_Topt", vcmax_resp.Topt);

    // Serialize temperature response parameters for Jmax
    context->setMaterialData(material_label, LABEL_FQ_Jmax, jmax_resp.value_at_25C);
    context->setMaterialData(material_label, "photo_fq_Jmax_dHa", jmax_resp.dHa);
    context->setMaterialData(material_label, "photo_fq_Jmax_dHd", jmax_resp.dHd);
    context->setMaterialData(material_label, "photo_fq_Jmax_Topt", jmax_resp.Topt);

    // Serialize temperature response parameters for Rd
    context->setMaterialData(material_label, LABEL_FQ_Rd, rd_resp.value_at_25C);
    context->setMaterialData(material_label, "photo_fq_Rd_dHa", rd_resp.dHa);
    context->setMaterialData(material_label, "photo_fq_Rd_dHd", rd_resp.dHd);
    context->setMaterialData(material_label, "photo_fq_Rd_Topt", rd_resp.Topt);

    // Serialize temperature response parameters for alpha
    context->setMaterialData(material_label, LABEL_FQ_alpha, alpha_resp.value_at_25C);
    context->setMaterialData(material_label, "photo_fq_alpha_dHa", alpha_resp.dHa);

    // Serialize mesophyll conductance gm (default value_at_25C is +infinity, meaning Cc ≡ Ci).
    PhotosyntheticTemperatureResponseParameters gm_resp = coeffs.getMesophyllConductance_gmTempResponse();
    context->setMaterialData(material_label, LABEL_FQ_gm, gm_resp.value_at_25C);
    context->setMaterialData(material_label, LABEL_FQ_gm_dHa, gm_resp.dHa);
    context->setMaterialData(material_label, LABEL_FQ_gm_dHd, gm_resp.dHd);
    context->setMaterialData(material_label, LABEL_FQ_gm_Topt, gm_resp.Topt);

    // Serialize other parameters
    context->setMaterialData(material_label, LABEL_FQ_O, coeffs.O);
    context->setMaterialData(material_label, LABEL_FQ_TPU_flag, coeffs.TPU_flag);

    // Clear cache for this material
    uint matID = context->getMaterialIDFromLabel(material_label);
    material_coefficient_cache_farquhar.erase(matID);

    model = "farquhar";
}

void PhotosynthesisModel::setFarquharCoefficientsFromLibrary(const std::string &species, const std::string &material_label) {
    FarquharModelCoefficients coeffs = getFarquharCoefficientsFromLibrary(species);
    setModelCoefficients(material_label, coeffs);
}

FarquharModelCoefficients PhotosynthesisModel::getFarquharCoefficientsFromLibrary(const std::string &species) {
    std::string s = std::move(species);
    FarquharModelCoefficients fmc;
    bool defaultSpecies = false;
    if (s == "Almond" || s == "almond") {
        fmc.setVcmax(105.9, 65.33f);
        fmc.setJmax(166.34, 46.36);
        fmc.setRd(1.49, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.336);
    } else if (s == "Apple" || s == "apple") {
        fmc.setVcmax(101.08, 65.33f);
        fmc.setJmax(167.03, 47.62);
        fmc.setRd(3.00, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.432);
    } else if (s == "Cherry" || s == "cherry") {
        fmc.setVcmax(75.65, 65.33f);
        fmc.setJmax(129.06, 48.49);
        fmc.setRd(2.12, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.404);
    } else if (s == "Prune" || s == "prune") {
        fmc.setVcmax(75.88, 65.33f);
        fmc.setJmax(129.41, 48.58);
        fmc.setRd(1.56, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.402);
    } else if (s == "Pear" || s == "pear") {
        fmc.setVcmax(107.69, 65.33f);
        fmc.setJmax(176.71, 46.04);
        fmc.setRd(1.510, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.274);
    } else if (s == "PistachioFemale" || s == "pistachiofemale" || s == "pistachio_female" || s == "Pistachio_Female" || s == "Pistachio_female" || s == "pistachio" || s == "Pistachio") {
        fmc.setVcmax(138.99, 65.33f);
        fmc.setJmax(221.76, 43.80);
        fmc.setRd(2.850, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.366);
    } else if (s == "PistachioMale" || s == "pistachiomale" || s == "pistachio_male" || s == "Pistachio_Male" || s == "Pistachio_male") {
        fmc.setVcmax(154.17, 65.33f);
        fmc.setJmax(243.20, 50.89);
        fmc.setRd(2.050, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.335);
    } else if (s == "Walnut" || s == "walnut") {
        fmc.setVcmax(121.85, 65.33f);
        fmc.setJmax(197.25, 48.35);
        fmc.setRd(1.960, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.404);
    } else if (s == "Grape" || s == "grape") {
        // cv. Cabernet Sauvignon
        fmc.setVcmax(74.5, 76.1, 318.8 - 273.15, 499.8);
        fmc.setJmax(180.2, 23.0, 313.8 - 273.15, 502.3);
        fmc.setTPU(7.7, 24.0, 314.6 - 273.15, 496.4);
        fmc.setRd(1.3, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.304);
        fmc.setLightResponseCurvature_theta(0.06);
    } else if (s == "Elderberry" || s == "elderberry" || s == "blue_elderberry") {
        fmc.setVcmax(37.7, 66.0, 319.4 - 273.15, 496.0);
        fmc.setJmax(149.7, 24.5, 314.8 - 273.15, 492.9);
        fmc.setTPU(7.3, 33.6, 314.5 - 273.15, 497.5);
        fmc.setRd(1.3, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.202);
        fmc.setLightResponseCurvature_theta(0.472);
    } else if (s == "Toyon" || s == "toyon") {
        fmc.setVcmax(52.8, 42.1, 315.1 - 273.15, 483.0);
        fmc.setJmax(142.4, 9.0, 313.0 - 273.15, 486.2);
        fmc.setTPU(6.6, 14.0, 314.8 - 273.15, 493.8);
        fmc.setRd(0.8, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.290);
        fmc.setLightResponseCurvature_theta(0.532);
    } else if (s == "Big_Leaf_Maple" || s == "big_leaf_maple" || s == "Maple" || s == "maple") {
        fmc.setVcmax(96.4, 48.9, 307.1 - 273.15, 505.0);
        fmc.setJmax(168.0, 8.5, 304.7 - 273.15, 476.7);
        fmc.setTPU(2.7, 32.1, 308.3 - 273.15, 471.6);
        fmc.setRd(0.1, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.077);
    } else if (s == "Western_Redbud" || s == "western_redbud" || s == "Redbud" || s == "redbud") {
        fmc.setVcmax(68.5, 66.6, 315.1 - 273.15, 496.0);
        fmc.setJmax(132.4, 41.2, 313.1 - 273.15, 474.0);
        fmc.setTPU(6.6, 34.3, 312.8 - 273.15, 463.2);
        fmc.setRd(0.8, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.41);
        fmc.setLightResponseCurvature_theta(0.04);
    } else if (s == "Baylaurel" || s == "baylaurel" || s == "Bay_Laurel" || s == "bay_laurel" || s == "bay" || s == "Bay") {
        fmc.setVcmax(97.5, 49.1, 308.6 - 273.15, 505.8);
        fmc.setJmax(193.0, 34.0, 308.5 - 273.15, 456.7);
        fmc.setTPU(3.3, 0.1, 309.4 - 273.15, 477.5);
        fmc.setRd(0.1, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.037);
    } else if (s == "Olive" || s == "olive") {
        fmc.setVcmax(75.9, 55.4, 315.2 - 273.15, 497.0);
        fmc.setJmax(170.4, 32.2, 312.5 - 273.15, 493.4);
        fmc.setTPU(8.3, 37.2, 311.7 - 273.15, 498.9);
        fmc.setRd(1.9, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.398);
    } else if (s == "EasternRedbudSunlit" || s == "easternredbudsunlit" || s == "easternredbud_sunlit" || s == "EasternRedbud_Sunlit" || s == "EasternRedbud_sunlit" || s == "sunlitEasternRedbud" || s == "SunlitEasternRedbud" ||
               s == "eastern_redbud_sunlit" || s == "Eastern_Redbud_Sunlit") {
        fmc.setVcmax(104.35, 54.9927, 313.8828 - 273.15, 365.1581);
        fmc.setJmax(211.3090, 46.5415, 310.9710 - 273.15, 200.3197);
        fmc.setTPU(9.9965, 48.4469, 310.3164 - 273.15, 167.9181);
        fmc.setRd(1.4136, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.4151);
    } else if (s == "EasternRedbudShaded" || s == "easternredbudshaded" || s == "easternredbud_shaded" || s == "EasternRedbud_Shaded" || s == "EasternRedbud_shaded" || s == "shadedEasternRedbud" || s == "ShadedEasternRedbud" ||
               s == "eastern_redbud_shaded" || s == "Eastern_Redbud_Shaded") {
        fmc.setVcmax(84.7, 65.33f);
        fmc.setJmax(190.53, 0.249);
        fmc.setTPU(9.53, 0.0747);
        fmc.setRd(1.13, 46.39f);
        fmc.setQuantumEfficiency_alpha(0.713);
    } else {
        defaultSpecies = true;
        if (message_flag) {
            std::cerr << "WARNING (PhotosynthesisModel::getModelCoefficients): unknown species " << s << ". Setting default (Almond)." << std::endl;
        }
    }
    if (!defaultSpecies) {
        if (message_flag) {
            std::cerr << "Setting Photosynthesis Model Coefficients to " << s << std::endl;
        }
    }
    return fmc;
}

// ---------------------------------------------------------------------------
// von Caemmerer (2021) C4 model — species parameter library.
//
// Each entry is internally complete (temperature-responsive rates, kinetic
// constants, and scalar structural parameters) so that results are not
// distorted by silently mixing an entry's headline rate constants with a
// different entry's fixed assumptions. The C3 library above falls back to the
// default species with a warning when the key is unknown; the C4 library is
// new and follows the fail-fast philosophy instead — unknown keys raise
// helios_runtime_error with the list of supported entries.
// ---------------------------------------------------------------------------

void PhotosynthesisModel::setC4CoefficientsFromLibrary(const std::string &species) {
    C4ModelCoefficients coeffs = getC4CoefficientsFromLibrary(species);
    setModelCoefficients(coeffs);
}

void PhotosynthesisModel::setC4CoefficientsFromLibrary(const std::string &species, const std::vector<uint> &UUIDs) {
    C4ModelCoefficients coeffs = getC4CoefficientsFromLibrary(species);
    setModelCoefficients(coeffs, UUIDs);
}

void PhotosynthesisModel::setC4CoefficientsFromLibrary(const std::string &species, const std::string &material_label) {
    C4ModelCoefficients coeffs = getC4CoefficientsFromLibrary(species);
    setModelCoefficients(material_label, coeffs);
}

C4ModelCoefficients PhotosynthesisModel::getC4CoefficientsFromLibrary(const std::string &species) {
    // Case-insensitive exact-key match.
    std::string s = species;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });

    C4ModelCoefficients c4;

    if (s == "setariaviridis_vc2021" || s == "setariaviridis" || s == "setaria_viridis_vc2021") {
        // Setaria viridis, NADP-ME. von Caemmerer (2021) JXB 72:6003 Table 1.
        // Kinetics: Boyd et al. (2015) Plant Physiol 169:1850 (Vpmax, Vcmax, Rd, Kc, Ko, Kp, γ*).
        // gm: Ubierna et al. (2017) New Phytol 214:66. Jmax T-response: peaked-Arrhenius refit of the
        // paper's Gaussian (June et al. 2004) — value at 25 °C matches Gaussian exactly.
        c4.setVpmax(200.f, 50.1f);
        c4.setVcmax(40.f, 78.0f);
        c4.setJmax(247.69f, 77.9f, 43.f, 260.f);
        c4.setRd(0.4f, 66.4f); // spec: Rd = 0.01 · Vcmax
        c4.setMesophyllConductance_gm(1.0f, 49.8f);
        // Kinetic + scalar fields keep struct-constructor defaults (Kc=1210, Ko=292000, Kp=82,
        // γ*=3.8168e-4 with dH=-31.1, Om=210000, dH_Kc=64.2, dH_Ko=10.5, dH_Kp=38.3, alpha_psII=0,
        // x=0.4, Vpr=80, Rm_frac=0.5, fcyc=0.3, gbs=0.003, ao=0.047, absorptance=0.85, f_spectral=0.15).
    } else if (s == "genericc4_vc2000" || s == "genericc4" || s == "generic_c4" || s == "generic_vc2000") {
        // Generic NADP-ME fallback. von Caemmerer (2000) Biochemical Models of Leaf Photosynthesis
        // (CSIRO) as encoded by the plantecophys R package (Duursma 2015) AciC4 defaults.
        // Q10 → Arrhenius conversion: Ea ≈ R · T_ref² · ln(Q10) / 10 with T_ref = 298.15 K.
        c4.setVcmax(60.f, 61.6f);  // Q10 = 2.3 → Ea = 8.314 · 298.15² · ln(2.3) / 10 ≈ 61.6 kJ/mol
        c4.setVpmax(120.f, 61.6f); // Q10 = 2.3 → Ea ≈ 61.6 kJ/mol
        c4.setJmax(400.f, 61.6f);  // Q10 = 2.3 → Ea ≈ 61.6 kJ/mol
        c4.setRd(1.0f, 51.2f);     // Q10 = 2.0 → Ea = 8.314 · 298.15² · ln(2.0) / 10 ≈ 51.2 kJ/mol
        c4.setMesophyllConductance_gm(1.0e4f); // no T-response; vC2000 has no gm term (effectively infinite)
        c4.Kp_25 = 80.f;                       // vC2000 value (vs. Setaria 82); Kc/Ko/γ* left at Setaria defaults
        c4.fcyc = 0.f;                         // vC2000 has no cyclic term
    } else if (s == "maize_massad2007" || s == "maize" || s == "zea_mays_massad2007") {
        // Zea mays cv. Chambord. Temperature responses: Massad, Tuzet, Bethenod (2007) Plant Cell
        // Environ 30:1191 Fig. 6 (peaked Arrhenius: Ea, Hd, ΔS). 25 °C values: plantecophys AciC4
        // defaults (consistent with Massad's "conform to literature" statement since the paper
        // reports no single tabular point estimate).
        //
        // Massad (2007) Fig. 6 coefficients (for independent verification):
        //   Vcmax: Ea = 67.294 kJ/mol, Hd = 144.568 kJ/mol, ΔS = 472 J/mol/K
        //   Vpmax: Ea = 70.373 kJ/mol, Hd = 117.910 kJ/mol, ΔS = 376 J/mol/K
        //   Jmax:  Ea = 77.900 kJ/mol, Hd = 191.929 kJ/mol, ΔS = 627 J/mol/K
        //
        // Convert (Ea, Hd, ΔS) → peaked-Arrhenius Topt via
        //   Topt [K] = Hd / (ΔS − R · ln(Ea / (Hd − Ea)))    with R = 8.314 J/mol/K
        // yielding Topt(Vcmax) ≈ 305.5 K (32.3 °C), Topt(Vpmax) ≈ 316.3 K (43.1 °C),
        // Topt(Jmax) ≈ 304.6 K (31.5 °C). The Helios peaked-Arrhenius setter takes
        // (value_at_25, dHa, Topt_°C, dHd).
        c4.setVcmax(60.f, 67.3f, 32.3f, 144.6f);
        c4.setVpmax(120.f, 70.4f, 43.1f, 117.9f);
        c4.setJmax(400.f, 77.9f, 31.5f, 191.9f);
        c4.setRd(0.6f, 66.4f);                  // 0.01·Vcmax vC2000 convention; Massad's own fits used Rd=0 — 0.6 is the library's usable default, not a Massad measurement
        c4.setMesophyllConductance_gm(1.0e4f);  // Massad assumed infinite gm
        // Massad fit against Bernacchi (2001) C3-derived Kc/Ko — these are ~2× smaller than the
        // Boyd 2015 C4 values used for Setaria. Keeping Massad's fixed assumptions intact is
        // required to avoid biasing the Vcmax / Vpmax estimates.
        c4.Kc_25 = 650.f;
        c4.Ko_25 = 450000.f;
        c4.dH_Kc = 79.43f;
        c4.dH_Ko = 36.38f;
        // Kp: Massad assumed Q10=2.1. Per spec, replace with Boyd (2015) Arrhenius for internal
        // consistency (25 °C value essentially unchanged at 80 μbar).
        c4.Kp_25 = 80.f;
        c4.dH_Kp = 38.3f;
        c4.fcyc = 0.f; // implicit in Massad's vC&F (1999) parent framework
    } else {
        helios_runtime_error("ERROR (PhotosynthesisModel::getC4CoefficientsFromLibrary): unknown C4 species '" + species +
                             "'. Available keys (case-insensitive): SetariaViridis_vC2021, GenericC4_vC2000, Maize_Massad2007.");
    }

    if (message_flag) {
        std::cerr << "Setting C4 Photosynthesis Model Coefficients to " << species << std::endl;
    }
    return c4;
}

// Cached coefficient retrieval helpers

EmpiricalModelCoefficients PhotosynthesisModel::getCoefficientsForPrimitive_Empirical(uint UUID) const {
    uint materialID = context->getPrimitiveMaterialID(UUID);

    // Check cache first
    if (material_coefficient_cache_empirical.find(materialID) != material_coefficient_cache_empirical.end()) {
        return material_coefficient_cache_empirical.at(materialID);
    }

    // Try to load from material data
    EmpiricalModelCoefficients coeffs;
    bool found = true;

    try {
        const Material &mat = context->getMaterial(materialID);
        if (mat.doesMaterialDataExist(LABEL_EMP_Tref) && mat.doesMaterialDataExist(LABEL_EMP_Ci_ref) && mat.doesMaterialDataExist(LABEL_EMP_Asat) && mat.doesMaterialDataExist(LABEL_EMP_theta) && mat.doesMaterialDataExist(LABEL_EMP_Tmin) &&
            mat.doesMaterialDataExist(LABEL_EMP_Topt) && mat.doesMaterialDataExist(LABEL_EMP_q) && mat.doesMaterialDataExist(LABEL_EMP_R) && mat.doesMaterialDataExist(LABEL_EMP_ER) && mat.doesMaterialDataExist(LABEL_EMP_kC)) {

            mat.getMaterialData(LABEL_EMP_Tref, coeffs.Tref);
            mat.getMaterialData(LABEL_EMP_Ci_ref, coeffs.Ci_ref);
            mat.getMaterialData(LABEL_EMP_Asat, coeffs.Asat);
            mat.getMaterialData(LABEL_EMP_theta, coeffs.theta);
            mat.getMaterialData(LABEL_EMP_Tmin, coeffs.Tmin);
            mat.getMaterialData(LABEL_EMP_Topt, coeffs.Topt);
            mat.getMaterialData(LABEL_EMP_q, coeffs.q);
            mat.getMaterialData(LABEL_EMP_R, coeffs.R);
            mat.getMaterialData(LABEL_EMP_ER, coeffs.ER);
            mat.getMaterialData(LABEL_EMP_kC, coeffs.kC);

            material_coefficient_cache_empirical[materialID] = coeffs;
            return coeffs;
        } else {
            found = false;
        }
    } catch (const std::exception &) {
        found = false;
    }

    // Fallback to legacy UUID map
    if (!found && empiricalmodel_coefficients.find(UUID) != empiricalmodel_coefficients.end()) {
        return empiricalmodel_coefficients.at(UUID);
    }

    // Fallback to global default
    return empiricalmodelcoeffs;
}

FarquharModelCoefficients PhotosynthesisModel::getCoefficientsForPrimitive_Farquhar(uint UUID) const {
    uint materialID = context->getPrimitiveMaterialID(UUID);

    // Check cache first
    if (material_coefficient_cache_farquhar.find(materialID) != material_coefficient_cache_farquhar.end()) {
        return material_coefficient_cache_farquhar.at(materialID);
    }

    // Try to load from material data
    FarquharModelCoefficients coeffs;
    bool found = true;

    try {
        const Material &mat = context->getMaterial(materialID);
        if (mat.doesMaterialDataExist(LABEL_FQ_Vcmax) && mat.doesMaterialDataExist(LABEL_FQ_Jmax) && mat.doesMaterialDataExist(LABEL_FQ_Rd) && mat.doesMaterialDataExist(LABEL_FQ_alpha)) {

            // Reconstruct temperature response parameters for Vcmax
            float vcmax_25C, vcmax_dHa, vcmax_dHd, vcmax_Topt;
            mat.getMaterialData(LABEL_FQ_Vcmax, vcmax_25C);
            mat.getMaterialData("photo_fq_Vcmax_dHa", vcmax_dHa);
            mat.getMaterialData("photo_fq_Vcmax_dHd", vcmax_dHd);
            mat.getMaterialData("photo_fq_Vcmax_Topt", vcmax_Topt);

            // Reconstruct for Jmax
            float jmax_25C, jmax_dHa, jmax_dHd, jmax_Topt;
            mat.getMaterialData(LABEL_FQ_Jmax, jmax_25C);
            mat.getMaterialData("photo_fq_Jmax_dHa", jmax_dHa);
            mat.getMaterialData("photo_fq_Jmax_dHd", jmax_dHd);
            mat.getMaterialData("photo_fq_Jmax_Topt", jmax_Topt);

            // Reconstruct for Rd
            float rd_25C, rd_dHa, rd_dHd, rd_Topt;
            mat.getMaterialData(LABEL_FQ_Rd, rd_25C);
            mat.getMaterialData("photo_fq_Rd_dHa", rd_dHa);
            mat.getMaterialData("photo_fq_Rd_dHd", rd_dHd);
            mat.getMaterialData("photo_fq_Rd_Topt", rd_Topt);

            // Reconstruct for alpha
            float alpha_25C, alpha_dHa;
            mat.getMaterialData(LABEL_FQ_alpha, alpha_25C);
            mat.getMaterialData("photo_fq_alpha_dHa", alpha_dHa);

            // Use setter methods to properly populate the coefficient struct
            if (vcmax_Topt < 1000.f) { // Has optimum
                coeffs.setVcmax(vcmax_25C, vcmax_dHa, vcmax_Topt - 273.15f, vcmax_dHd);
            } else if (vcmax_dHa > 0.f) { // Monotonic increase
                coeffs.setVcmax(vcmax_25C, vcmax_dHa);
            } else { // Constant
                coeffs.setVcmax(vcmax_25C);
            }

            if (jmax_Topt < 1000.f) {
                coeffs.setJmax(jmax_25C, jmax_dHa, jmax_Topt - 273.15f, jmax_dHd);
            } else if (jmax_dHa > 0.f) {
                coeffs.setJmax(jmax_25C, jmax_dHa);
            } else {
                coeffs.setJmax(jmax_25C);
            }

            if (rd_Topt < 1000.f) {
                coeffs.setRd(rd_25C, rd_dHa, rd_Topt - 273.15f, rd_dHd);
            } else if (rd_dHa > 0.f) {
                coeffs.setRd(rd_25C, rd_dHa);
            } else {
                coeffs.setRd(rd_25C);
            }

            if (alpha_dHa > 0.f) {
                coeffs.setQuantumEfficiency_alpha(alpha_25C, alpha_dHa);
            } else {
                coeffs.setQuantumEfficiency_alpha(alpha_25C);
            }

            // Reconstruct gm temperature response if present. Missing labels leave the
            // default (value_at_25C = +infinity, i.e., Cc ≡ Ci — legacy behavior).
            if (mat.doesMaterialDataExist(LABEL_FQ_gm)) {
                float gm_25C, gm_dHa = 0.f, gm_dHd = 0.f, gm_Topt = 10000.f;
                mat.getMaterialData(LABEL_FQ_gm, gm_25C);
                if (mat.doesMaterialDataExist(LABEL_FQ_gm_dHa)) {
                    mat.getMaterialData(LABEL_FQ_gm_dHa, gm_dHa);
                }
                if (mat.doesMaterialDataExist(LABEL_FQ_gm_dHd)) {
                    mat.getMaterialData(LABEL_FQ_gm_dHd, gm_dHd);
                }
                if (mat.doesMaterialDataExist(LABEL_FQ_gm_Topt)) {
                    mat.getMaterialData(LABEL_FQ_gm_Topt, gm_Topt);
                }
                if (gm_Topt < 1000.f) {
                    coeffs.setMesophyllConductance_gm(gm_25C, gm_dHa, gm_Topt - 273.15f, gm_dHd);
                } else if (gm_dHa > 0.f) {
                    coeffs.setMesophyllConductance_gm(gm_25C, gm_dHa);
                } else {
                    coeffs.setMesophyllConductance_gm(gm_25C);
                }
            }

            // Set O and TPU_flag
            mat.getMaterialData(LABEL_FQ_O, coeffs.O);
            mat.getMaterialData(LABEL_FQ_TPU_flag, coeffs.TPU_flag);

            material_coefficient_cache_farquhar[materialID] = coeffs;
            return coeffs;
        } else {
            found = false;
        }
    } catch (const std::exception &) {
        found = false;
    }

    // Fallback to legacy UUID map
    if (!found && farquharmodel_coefficients.find(UUID) != farquharmodel_coefficients.end()) {
        return farquharmodel_coefficients.at(UUID);
    }

    // Fallback to global default
    return farquharmodelcoeffs;
}

void PhotosynthesisModel::setCm(float Cm, const std::vector<uint> &UUIDs) {
    if (Cm < 0.f) {
        helios_runtime_error("ERROR (PhotosynthesisModel::setCm): Cm must be non-negative. Received Cm = " + std::to_string(Cm) + " ubar.");
    }
    if (!std::isfinite(Cm)) {
        helios_runtime_error("ERROR (PhotosynthesisModel::setCm): Cm must be a finite value.");
    }
    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (PhotosynthesisModel::setCm): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
        }
        manual_Cm[UUID] = Cm;
    }
}

void PhotosynthesisModel::setCi(float Ci, const std::vector<uint> &UUIDs) {
    // Fail-fast validation
    if (Ci < 0.f) {
        helios_runtime_error("ERROR (PhotosynthesisModel::setCi): Ci must be non-negative. Received Ci = " + std::to_string(Ci) + " umol/mol.");
    }

    if (!std::isfinite(Ci)) {
        helios_runtime_error("ERROR (PhotosynthesisModel::setCi): Ci must be a finite value.");
    }

    // Reasonable range warning (typical C3 range: 50-800 umol/mol)
    if (Ci > 2000.f && message_flag) {
        std::cout << "WARNING (PhotosynthesisModel::setCi): Ci = " << Ci << " umol/mol is unusually high. Verify this is intentional." << std::endl;
    }

    // Set manual Ci for all specified UUIDs
    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            helios_runtime_error("ERROR (PhotosynthesisModel::setCi): Primitive with UUID " + std::to_string(UUID) + " does not exist.");
        }
        manual_Ci[UUID] = Ci;
    }
}

void PhotosynthesisModel::run() {
    run(context->getAllUUIDs());
}

void PhotosynthesisModel::run(const std::vector<uint> &lUUIDs) {

    WarningAggregator warnings;
    warnings.setEnabled(message_flag); // Respect existing message flag

    for (uint UUID: lUUIDs) {

        float i_PAR;
        if (context->doesPrimitiveDataExist(UUID, "radiation_flux_PAR") && context->getPrimitiveDataType("radiation_flux_PAR") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "radiation_flux_PAR", i_PAR);
            i_PAR = i_PAR * 4.57f; // umol/m^2-s (ref https://www.controlledenvironments.org/wp-content/uploads/sites/6/2017/06/Ch01.pdf)
            if (i_PAR < 0) {
                i_PAR = 0;
                warnings.addWarning("negative_par_clipped", "PAR flux value provided was negative. Clipping to zero.");
            }
        } else {
            i_PAR = i_PAR_default;
        }

        float TL;
        if (context->doesPrimitiveDataExist(UUID, "temperature") && context->getPrimitiveDataType("temperature") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "temperature", TL);
            if (TL < 200) {
                warnings.addWarning("low_temperature", "Primitive temperature value was very low (" + std::to_string(TL) + "K). Using default. Are you using absolute temperature units?");
                TL = TL_default;
            }
        } else {
            TL = TL_default;
        }

        float CO2;
        if (context->doesPrimitiveDataExist(UUID, "air_CO2") && context->getPrimitiveDataType("air_CO2") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "air_CO2", CO2);
            if (CO2 < 0) {
                CO2 = 0;
                warnings.addWarning("negative_co2_clipped", "CO2 concentration value provided was negative. Clipping to zero.");
            }
        } else {
            CO2 = CO2_default;
        }

        float gM;
        if (context->doesPrimitiveDataExist(UUID, "moisture_conductance") && context->getPrimitiveDataType("moisture_conductance") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "moisture_conductance", gM);
            if (gM < 0) {
                gM = 0;
                warnings.addWarning("negative_moisture_conductance_clipped", "Moisture conductance value provided was negative. Clipping to zero.");
            }
        } else {
            gM = gM_default;
        }

        // Number of sides - check material first, then primitive data
        uint twosided_flag = context->getPrimitiveTwosidedFlag(UUID, 1);
        uint Nsides = (twosided_flag == 0) ? 1 : 2;

        float stomatal_sidedness = 0.f; // default all stomata on one side (hypostomatous)
        if (Nsides == 2 && context->doesPrimitiveDataExist(UUID, "stomatal_sidedness") && context->getPrimitiveDataType("stomatal_sidedness") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "stomatal_sidedness", stomatal_sidedness);
        }

        float gH;
        if (context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") && context->getPrimitiveDataType("boundarylayer_conductance") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "boundarylayer_conductance", gH);
        } else if (context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out") && context->getPrimitiveDataType("boundarylayer_conductance_out") == HELIOS_TYPE_FLOAT) {
            context->getPrimitiveData(UUID, "boundarylayer_conductance_out", gH);
        } else {
            gH = gH_default;
        }
        if (gH < 0) {
            gH = 0;
            warnings.addWarning("negative_boundarylayer_conductance_clipped", "Boundary-layer conductance value provided was negative. Clipping to zero.");
        }

        // combine stomatal (gM) and boundary-layer (gH) conductances
        if (gH == 0 && gM == 0) { // if somehow both go to zero, can get NaN
            gM = 0;
        } else {
            gM = 1.08f * gH * gM * (stomatal_sidedness / (1.08f * gH + gM * stomatal_sidedness) + (1.f - stomatal_sidedness) / (1.08f * gH + gM * (1.f - stomatal_sidedness)));
        }

        float A, Ci, Gamma, J_over_Jmax = 0.f;
        int limitation_state, TPU_flag = 0;
        float Cm_c4 = 0.f, Vp_c4 = 0.f; // C4-specific outputs (unused for other models)

        if (model == "farquhar") { // Farquhar-von Caemmerer-Berry Model

            // Check for manual Ci override first
            if (manual_Ci.find(UUID) != manual_Ci.end()) {
                // Use manual Ci, bypassing iterative calculation
                Ci = manual_Ci.at(UUID);

                FarquharModelCoefficients coeffs = getCoefficientsForPrimitive_Farquhar(UUID);

                // Set up variables vector for evaluateCi_Farquhar
                std::vector<float> variables{CO2, i_PAR, TL, gM, 0.f, 0.f, 0.f, float(TPU_flag), 0.f};

                // Call evaluateCi_Farquhar directly to compute A for the manual Ci
                // This bypasses the fzero iteration in evaluateFarquharModel
                evaluateCi_Farquhar(Ci, variables, &coeffs); // Return value (residual) is unused

                // Extract results from variables vector
                A = variables[4];
                limitation_state = (int) variables[5];
                Gamma = variables[6];
                J_over_Jmax = variables[8];

                // Store in previous_Ci for consistency
                previous_Ci[UUID] = Ci;

            } else {
                // Normal iterative calculation

                // Initialize Ci with previous timestep value for temporal continuity, or reasonable default
                if (previous_Ci.find(UUID) != previous_Ci.end()) {
                    Ci = previous_Ci.at(UUID); // Use previous timestep's Ci
                } else {
                    Ci = CO2 * 0.7f; // Default initial guess (typical Ci/Ca ratio)
                }

                FarquharModelCoefficients coeffs = getCoefficientsForPrimitive_Farquhar(UUID);
                A = evaluateFarquharModel(coeffs, i_PAR, TL, CO2, gM, Ci, Gamma, limitation_state, TPU_flag, J_over_Jmax, warnings);

                // Store computed Ci for next timestep (temporal continuity)
                previous_Ci[UUID] = Ci;
            }

        } else if (model == "c4") { // von Caemmerer (2021) C4 Model

            C4ModelCoefficients c4coeffs = getCoefficientsForPrimitive_C4(UUID);

            if (manual_Cm.find(UUID) != manual_Cm.end()) {
                // Manual Cm path — skip both stomatal iteration and the Cm = Ci - A/gm fixed point.
                // A is computed directly at the supplied Cm; Ci is back-computed from Cm + A/gm.
                const float Cm_in = manual_Cm.at(UUID);
                std::vector<float> variables{CO2, i_PAR, TL, gM, 0.f, 0.f, 0.f, 0.f};
                evaluateCm_C4(Cm_in, variables, c4coeffs);
                A = variables[4];
                limitation_state = static_cast<int>(variables[5]);
                Cm_c4 = variables[6];
                Vp_c4 = variables[7];
                const float gm_eval = respondToTemperature(&c4coeffs.gmTempResponse, TL);
                Ci = (std::isfinite(gm_eval) && gm_eval > 0.f) ? (Cm_in + A / gm_eval) : Cm_in;
                previous_Ci[UUID] = Ci;
            } else if (manual_Ci.find(UUID) != manual_Ci.end()) {
                // Manual Ci path — bypass stomatal iteration, evaluate A via Cm = Ci - A/gm fixed point
                Ci = manual_Ci.at(UUID);
                std::vector<float> variables{CO2, i_PAR, TL, gM, 0.f, 0.f, 0.f, 0.f};
                evaluateCi_C4(Ci, variables, &c4coeffs);
                A = variables[4];
                limitation_state = static_cast<int>(variables[5]);
                Cm_c4 = variables[6];
                Vp_c4 = variables[7];
                previous_Ci[UUID] = Ci;
            } else {
                if (previous_Ci.find(UUID) != previous_Ci.end()) {
                    Ci = previous_Ci.at(UUID);
                } else {
                    Ci = CO2 * 0.4f; // C4 typically has lower Ci/Ca (~0.3-0.4) than C3
                }
                Cm_c4 = Ci;
                A = evaluateC4Model(c4coeffs, i_PAR, TL, CO2, gM, Ci, Cm_c4, Vp_c4, limitation_state, warnings);
                previous_Ci[UUID] = Ci;
            }
            Gamma = 0.f; // not defined the same way for C4 — leave zero

        } else { // Empirical Model

            EmpiricalModelCoefficients coeffs = getCoefficientsForPrimitive_Empirical(UUID);
            A = evaluateEmpiricalModel(coeffs, i_PAR, TL, CO2, gM);
        }

        if (A == 0) {
            warnings.addWarning("convergence_failure", "Solution did not converge for primitive " + std::to_string(UUID) + ".");
        }

        context->setPrimitiveData(UUID, "net_photosynthesis", A);

        for (const auto &data: output_prim_data) {
            if (data == "Ci" && (model == "farquhar" || model == "c4")) {
                context->setPrimitiveData(UUID, "Ci", Ci);
            } else if (data == "limitation_state" && (model == "farquhar" || model == "c4")) {
                context->setPrimitiveData(UUID, "limitation_state", limitation_state);
            } else if (data == "Gamma_CO2" && model == "farquhar") {
                context->setPrimitiveData(UUID, "Gamma_CO2", Gamma);
            } else if (data == "electron_transport_ratio" && model == "farquhar") {
                context->setPrimitiveData(UUID, "electron_transport_ratio", J_over_Jmax);
            } else if (data == "Cm" && model == "c4") {
                context->setPrimitiveData(UUID, "Cm", Cm_c4);
            } else if (data == "Vp" && model == "c4") {
                context->setPrimitiveData(UUID, "Vp", Vp_c4);
            }
        }
    }

    warnings.report(std::cerr);
}

float PhotosynthesisModel::evaluateCi_Empirical(const EmpiricalModelCoefficients &params, float Ci, float CO2, float fL, float Rd, float gM) const {


    //--- CO2 Response Function --- //

    float fC = params.kC * Ci / params.Ci_ref;


    //--- Assimilation Rate --- //

    float A = params.Asat * fL * fC - Rd;

    //--- Calculate error and update --- //

    float resid = 0.75f * gM * (CO2 - Ci) - A;


    return resid;
}

float PhotosynthesisModel::evaluateEmpiricalModel(const EmpiricalModelCoefficients &params, float i_PAR, float TL, float CO2, float gM) {

    // initial guess for intercellular CO2
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

        if (resid_old == resid_old_old) { // this condition will cause NaN
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
    int TPUflag = modelcoeffs.TPU_flag;

    float R = 0.0083144598; // molar gas constant (kJ/K/mol)
    float O = 213.5; // ambient oxygen concentration (mmol/mol)
    float c_Gamma = 19.02;
    float dH_Gamma = 37.83;
    float c_Kc = 38.05;
    float dH_Kc = 79.43;
    float c_Ko = 20.30;
    float dH_Ko = 36.38;

    float invDiffRT = (1.f / 298.15f - 1.f / TL) / R;

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

    // Mesophyll conductance gm (mol CO2 / m^2 / s / bar). If gm is infinite/huge,
    // Cc ≡ Ci and we take the original legacy path below. Otherwise we solve a
    // quadratic in net A that couples Cc = Ci − A/gm (Ethier & Livingston 2004 /
    // Sharkey et al. 2007).
    const float gm = respondToTemperature(&modelcoeffs.gmTempResponse, TL);
    const bool gm_infinite = (!std::isfinite(gm) || gm > 1.0e6f);


    float Gamma_star = exp(c_Gamma - dH_Gamma / (R * TL));
    float Kc = exp(c_Kc - dH_Kc / (R * TL));
    float Ko = exp(c_Ko - dH_Ko / (R * TL));
    float Kco = Kc * (1.f + O / Ko);

    // thetaJ^2 + -(alphaQ+Jmax)J + alphaQJmax = 0
    double a = std::max(theta, 0.0001f);
    double ia = 1.000f / a;
    double b = -(alpha * Q + Jmax);
    double c = alpha * Q * Jmax;
    double J = (-b - sqrt(pow(b, 2.000f) - 4.f * a * c)) * 0.5f * ia;
    // J = Jmax * alpha * Q / (alpha * Q + Jmax);

    // Store J/Jmax ratio for fluorescence calculations
    float J_over_Jmax = (Jmax > 0.f) ? static_cast<float>(J / Jmax) : 0.f;
    variables[8] = J_over_Jmax;

    float A;
    float limitation_state;

    if (gm_infinite) {
        // Legacy path: Cc ≡ Ci (no mesophyll limitation).
        float Wc = Vcmax * Ci / (Ci + Kco);
        float Wj = J * Ci / (4.f * Ci + 8.f * Gamma_star);

        float smooth_factor = 0.99f;
        float s = helios::clamp(0.5f + 0.5f * (Wc - Wj) / smooth_factor, 0.0f, 1.0f);
        float smooth_min = Wc * (1.f - s) + Wj * s - smooth_factor * s * (1.f - s);

        A = (1.f - Gamma_star / Ci) * smooth_min - Rd;

        // TPU comparison at assimilation level: A_TPU = 3*TPU - Rd (singularity-free)
        // The Ci-dependent terms in Wp = 3*TPU*Ci/(Ci - Gamma*) cancel with (1 - Gamma*/Ci),
        // so TPU-limited assimilation is simply a constant independent of Ci.
        if (TPUflag == 1) {
            float A_p = 3.f * TPU - Rd;
            smooth_factor = 0.99f;
            s = helios::clamp(0.5f + 0.5f * (A - A_p) / smooth_factor, 0.0f, 1.0f);
            A = A * (1.f - s) + A_p * s - smooth_factor * s * (1.f - s);
        }

        if (Wj > Wc) { // Rubisco limited
            limitation_state = 0;
        } else { // Electron transport limited
            limitation_state = 1;
        }
    } else {
        // Finite-gm path. Substituting Cc = Ci − A/gm into the standard FvCB
        // expressions A = Vcmax·(Cc−Γ*)/(Cc+Kco) − Rd (Rubisco-limited) and
        // A = (J/4)·(Cc−Γ*)/(Cc+2Γ*) − Rd (electron-transport-limited) yields two
        // quadratics in net A of the form (1/gm)·A² + b·A + c = 0, with the
        // physical root being the smaller root (stable as gm → ∞).
        auto solve_quadratic_min_root = [](float aq, float bq, float cq) -> float {
            if (aq <= 0.f) {
                return -cq / bq;
            }
            float disc = bq * bq - 4.f * aq * cq;
            if (disc < 0.f) {
                disc = 0.f;
            }
            float sqrtD = std::sqrt(disc);
            float q = -0.5f * (bq - sqrtD); // bq < 0 in physical regime → safely large magnitude
            if (q == 0.f) {
                return (-bq + sqrtD) * 0.5f / aq;
            }
            return cq / q;
        };

        const float inv_gm = 1.f / gm;
        // Rubisco-limited: A²/gm − A·[(Ci+Kco) + (Vcmax−Rd)/gm] + [Vcmax·(Ci−Γ*) − Rd·(Ci+Kco)] = 0
        const float a_c = inv_gm;
        const float b_c = -((Ci + Kco) + (Vcmax - Rd) * inv_gm);
        const float c_c = Vcmax * (Ci - Gamma_star) - Rd * (Ci + Kco);
        const float A_c_net = solve_quadratic_min_root(a_c, b_c, c_c);

        // Electron-transport-limited: substitute Vcmax → J/4 and Kco → 2·Γ*
        const float J4 = 0.25f * static_cast<float>(J);
        const float a_j = inv_gm;
        const float b_j = -((Ci + 2.f * Gamma_star) + (J4 - Rd) * inv_gm);
        const float c_j = J4 * (Ci - Gamma_star) - Rd * (Ci + 2.f * Gamma_star);
        const float A_j_net = solve_quadratic_min_root(a_j, b_j, c_j);

        // Smooth-min on net A — smaller A wins. s → 0 when A_c_net is smaller (Rubisco
        // limited); s → 1 when A_j_net is smaller (electron-transport limited).
        float smooth_factor = 0.99f;
        float s = helios::clamp(0.5f + 0.5f * (A_c_net - A_j_net) / smooth_factor, 0.0f, 1.0f);
        A = A_c_net * (1.f - s) + A_j_net * s - smooth_factor * s * (1.f - s);

        // TPU limitation: A_p = 3·TPU − Rd is independent of Cc, so it enters the
        // net-A smooth-min unchanged.
        if (TPUflag == 1) {
            float A_p = 3.f * TPU - Rd;
            smooth_factor = 0.99f;
            s = helios::clamp(0.5f + 0.5f * (A - A_p) / smooth_factor, 0.0f, 1.0f);
            A = A * (1.f - s) + A_p * s - smooth_factor * s * (1.f - s);
        }

        // Smaller A wins in the finite-gm branch (inequality flips vs. Wj > Wc).
        if (A_c_net < A_j_net) { // Rubisco limited
            limitation_state = 0;
        } else { // Electron transport limited
            limitation_state = 1;
        }
    }

    //--- Calculate error and update --- //

    float resid = 0.75f * gM * (CO2 - Ci) - A;

    variables[4] = A;
    variables[5] = limitation_state;

    float Gamma = (Gamma_star + Kco * Rd / Vcmax) / (1.f - Rd / Vcmax); // Equation 39 of Farquhar et al. (1980)
    variables[6] = Gamma;

    return resid;
}


namespace {
    //! Compute Ac and Aj at a fixed Cm for the von Caemmerer (2021) C4 model.
    //! Formulae follow the von Caemmerer (2021) spreadsheet (C4__model_setaria__11-06-2021.xlsm).
    //! Note: the quadratic coefficients include the g_bs·K terms that are dropped in the paper's
    //! Eqs. 22-24 text; we keep them because the spreadsheet does and they are numerically relevant.
    void computeC4RatesFromCm(float Cm, const C4ModelCoefficients &p, float Vpmax, float Vcmax, float Jmax, float Rd, float Kc, float Ko, float Kp, float gamma_star, float Om, float I_incident, float &Ac_out, float &Aj_out, float &Vp_out) {

        const float Rm = p.Rm_frac * Rd;

        // PEP carboxylation: V_p = min(C_m · V_pmax / (C_m + K_p), V_pr)   Eq. 19
        const float Vp_MM = Cm * Vpmax / (Cm + Kp);
        const float Vp = std::min(Vp_MM, p.Vpr);
        Vp_out = Vp;

        // Whole-chain electron transport rate J from non-rectangular hyperbola (Eq. 34)
        const float rho = (1.f - p.fcyc) / (2.f - p.fcyc);
        const float I2 = I_incident * p.absorptance * rho * (1.f - p.f_spectral);
        const float theta = std::max(p.theta_etr, 1.0e-4f);
        const float sum_IJ = I2 + Jmax;
        const float radicand_J = std::max(sum_IJ * sum_IJ - 4.f * theta * I2 * Jmax, 0.f);
        const float J = (sum_IJ - std::sqrt(radicand_J)) / (2.f * theta);

        // ATP / linear-electron-transport ratio  z = (3 − f_cyc) / (h · (1 − f_cyc))   Eq. 31
        const float denom_z = std::max(p.h_protons * (1.f - p.fcyc), 1.0e-6f);
        const float z = (3.f - p.fcyc) / denom_z;

        const float alpha = p.alpha_psII_fraction;
        const float ao = std::max(p.ao, 1.0e-6f);
        const float gbs = p.gbs;

        // --- Enzyme-limited rate A_c (quadratic in A) ---
        const float term_PEP = Vp - Rm + gbs * Cm; // (V_p − R_m + g_bs·C_m)
        const float term_Rub = Vcmax - Rd; // (V_cmax − R_d)
        const float Kco = 1.f + Om / Ko;
        const float a_c = 1.f - alpha * Kc / (ao * Ko);
        const float b_c = -(term_PEP + term_Rub + gbs * Kc * Kco + (alpha / ao) * (gamma_star * Vcmax + Rd * Kc / Ko));
        const float c_c = term_Rub * term_PEP - gbs * (Vcmax * gamma_star * Om + Rd * Kc * Kco);
        const float disc_c = std::max(b_c * b_c - 4.f * a_c * c_c, 0.f);
        // Smaller root — the physically meaningful solution (the larger root is unphysically high)
        Ac_out = (-b_c - std::sqrt(disc_c)) / (2.f * a_c);

        // --- Electron-transport-limited rate A_j (quadratic in A) ---
        const float Jm = p.x_etr_partition * J; // x · J
        const float Jb = (1.f - p.x_etr_partition) * J; // (1 − x) · J
        const float term_mesJ = Jm * z * 0.5f - Rm + gbs * Cm; // mesophyll ATP-driven rate (Eq. 36-ish)
        const float term_bunJ = Jb * z / 3.f - Rd; // bundle-sheath Rubisco-ATP rate
        const float a_j = 1.f - 7.f * gamma_star * alpha / (3.f * ao);
        const float b_j = -(term_mesJ + term_bunJ + gbs * 7.f * gamma_star * Om / 3.f + (alpha * gamma_star / ao) * (Jb * z / 3.f + 7.f * Rd / 3.f));
        const float c_j = term_mesJ * term_bunJ - gbs * gamma_star * Om * (Jb * z / 3.f + 7.f * Rd / 3.f);
        const float disc_j = std::max(b_j * b_j - 4.f * a_j * c_j, 0.f);
        Aj_out = (-b_j - std::sqrt(disc_j)) / (2.f * a_j);
    }
} // namespace


float PhotosynthesisModel::evaluateCi_C4(float Ci, std::vector<float> &variables, const void *parameters) {

    const C4ModelCoefficients &p = *reinterpret_cast<const C4ModelCoefficients *>(parameters);

    const float CO2 = variables[0];
    const float I_incident = variables[1];
    const float TL = variables[2];
    const float gM = variables[3];

    const float R = 0.0083144598f; // kJ/K/mol
    const float invDiffRT = (1.f / 298.15f - 1.f / TL) / R;

    // Temperature-responsive parameters (peaked Arrhenius framework)
    const float Vpmax = respondToTemperature(&p.VpmaxTempResponse, TL);
    const float Vcmax = respondToTemperature(&p.VcmaxTempResponse, TL);
    const float Jmax = respondToTemperature(&p.JmaxTempResponse, TL);
    const float Rd = respondToTemperature(&p.RdTempResponse, TL);
    const float gm = respondToTemperature(&p.gmTempResponse, TL);

    // Rubisco + PEPC kinetic constants (simple Arrhenius, same treatment as C3 Kc/Ko)
    const float Kc = p.Kc_25 * std::exp(p.dH_Kc * invDiffRT);
    const float Ko = p.Ko_25 * std::exp(p.dH_Ko * invDiffRT);
    const float Kp = p.Kp_25 * std::exp(p.dH_Kp * invDiffRT);
    const float gamma_star = p.gamma_star_25 * std::exp(p.dH_gamma_star * invDiffRT);
    const float Om = p.Om_25 * std::exp(p.dH_Om * invDiffRT);

    // Solve C_m = C_i − A/g_m by damped fixed-point iteration. For typical parameters this
    // converges in 5-15 iterations; the damping factor prevents oscillation when A is large.
    float Cm = std::max(Ci, 0.1f); // initial guess
    if (!std::isfinite(gm) || gm <= 0.f) {
        // Should not happen with default gmTempResponse, but guard against user misuse
        helios::helios_runtime_error("ERROR (PhotosynthesisModel::evaluateCi_C4): Mesophyll conductance g_m must be positive and finite (got " + std::to_string(gm) + "). Check C4ModelCoefficients::setMesophyllConductance_gm().");
    }

    float Ac = 0.f, Aj = 0.f, Vp = 0.f, A = 0.f;
    for (int iter = 0; iter < 50; ++iter) {
        computeC4RatesFromCm(Cm, p, Vpmax, Vcmax, Jmax, Rd, Kc, Ko, Kp, gamma_star, Om, I_incident, Ac, Aj, Vp);
        A = std::min(Ac, Aj);
        const float Cm_target = Ci - A / gm;
        if (std::fabs(Cm_target - Cm) < 1.0e-5f) {
            Cm = Cm_target;
            break;
        }
        Cm = 0.5f * Cm + 0.5f * Cm_target; // damped update
    }
    // Final re-evaluation at the converged C_m
    computeC4RatesFromCm(Cm, p, Vpmax, Vcmax, Jmax, Rd, Kc, Ko, Kp, gamma_star, Om, I_incident, Ac, Aj, Vp);
    A = std::min(Ac, Aj);

    // Limitation state: 1 = Rubisco/PEPC enzyme-limited, 2 = electron transport-limited
    const int limitation = (Ac < Aj) ? 1 : 2;

    variables[4] = A;
    variables[5] = static_cast<float>(limitation);
    variables[6] = Cm;
    variables[7] = Vp;

    // Outer residual: stomatal-conductance balance (used by fzero when Ci is not manually overridden)
    return 0.75f * gM * (CO2 - Ci) - A;
}


float PhotosynthesisModel::evaluateCm_C4(float Cm, std::vector<float> &variables, const C4ModelCoefficients &params) {

    const float I_incident = variables[1];
    const float TL = variables[2];

    const float R = 0.0083144598f;
    const float invDiffRT = (1.f / 298.15f - 1.f / TL) / R;

    const float Vpmax = respondToTemperature(&params.VpmaxTempResponse, TL);
    const float Vcmax = respondToTemperature(&params.VcmaxTempResponse, TL);
    const float Jmax = respondToTemperature(&params.JmaxTempResponse, TL);
    const float Rd = respondToTemperature(&params.RdTempResponse, TL);
    const float Kc = params.Kc_25 * std::exp(params.dH_Kc * invDiffRT);
    const float Ko = params.Ko_25 * std::exp(params.dH_Ko * invDiffRT);
    const float Kp = params.Kp_25 * std::exp(params.dH_Kp * invDiffRT);
    const float gamma_star = params.gamma_star_25 * std::exp(params.dH_gamma_star * invDiffRT);
    const float Om = params.Om_25 * std::exp(params.dH_Om * invDiffRT);

    float Ac = 0.f, Aj = 0.f, Vp = 0.f;
    computeC4RatesFromCm(Cm, params, Vpmax, Vcmax, Jmax, Rd, Kc, Ko, Kp, gamma_star, Om, I_incident, Ac, Aj, Vp);
    const float A = std::min(Ac, Aj);
    const int limitation = (Ac < Aj) ? 1 : 2;

    variables[4] = A;
    variables[5] = static_cast<float>(limitation);
    variables[6] = Cm;
    variables[7] = Vp;
    return A;
}


float PhotosynthesisModel::evaluateC4Model(const C4ModelCoefficients &params, float i_PAR, float TL, float CO2, float gM, float &Ci, float &Cm, float &Vp, int &limitation_state, helios::WarningAggregator &warnings) {

    float A = 0.f;
    // variables layout: [CO2, I_abs, TL, gM, A_out, limitation_out, Cm_out, Vp_out]
    std::vector<float> variables{CO2, i_PAR, TL, gM, A, static_cast<float>(limitation_state), Cm, Vp};

    std::vector<float> initial_guesses;
    if (Ci > 0 && std::isfinite(Ci)) {
        initial_guesses.push_back(Ci);
    }
    initial_guesses.insert(initial_guesses.end(), {CO2 * 0.6f, CO2 * 0.3f, 100.f, 50.f, CO2 * 0.9f});

    bool overall_converged = false;
    for (float guess: initial_guesses) {
        std::vector<float> vars_attempt = variables;
        bool attempt_converged = false;
        float Ci_attempt = fzero(evaluateCi_C4, vars_attempt, &params, guess, attempt_converged, 0.001f, 200);
        if (attempt_converged && Ci_attempt > 0 && std::isfinite(Ci_attempt)) {
            Ci = Ci_attempt;
            variables = vars_attempt;
            overall_converged = true;
            break;
        }
    }

    if (!overall_converged) {
        warnings.addWarning("photosynthesis_c4_ci_convergence_failure", "C4 photosynthesis model failed to converge for Ci after trying multiple initial guesses.");
        bool final_converged = false;
        Ci = fzero(evaluateCi_C4, variables, &params, 100.f, final_converged, 0.01f, 500);
    }

    A = variables[4];
    limitation_state = static_cast<int>(variables[5]);
    Cm = variables[6];
    Vp = variables[7];

    return A;
}


float PhotosynthesisModel::respondToTemperature(const PhotosyntheticTemperatureResponseParameters *params, float T_in_Kelvin) {
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
        float t1 = 1.f + expf(dHd / R * (1.f / Topt - 1.f / 298.15f) - logterm);
        float t2 = 1.f + expf(dHd / R * (1.f / Topt - 1.f / T) - logterm);
        return v25 * exp(dHa / R * (1.f / 298.15f - 1.f / T)) * t1 / t2;
    }
}


float PhotosynthesisModel::evaluateFarquharModel(const FarquharModelCoefficients &params, float i_PAR, float TL, float CO2, float gM, float &Ci, float &Gamma, int &limitation_state, int &TPU_flag, float &J_over_Jmax,
                                                 helios::WarningAggregator &warnings) {

    float A = 0;
    std::vector<float> variables{CO2, i_PAR, TL, gM, A, float(limitation_state), Gamma, float(TPU_flag), 0.f};

    // Temporal continuity approach: use previous Ci as initial guess, with fallbacks
    // This leverages the fact that Ci changes slowly between timesteps
    std::vector<float> initial_guesses;

    // First try previous Ci if available (temporal continuity)
    if (Ci > 0 && std::isfinite(Ci)) {
        initial_guesses.push_back(Ci);
    }

    // Add fallback guesses based on research (physiologically reasonable values)
    initial_guesses.insert(initial_guesses.end(), {CO2 * 0.7f, 100.0f, CO2 * 0.3f, CO2 * 0.9f, 50.0f});

    bool overall_converged = false;
    for (float guess: initial_guesses) {
        // Reset variables for each attempt
        std::vector<float> vars_attempt = variables;
        bool attempt_converged = false;
        float Ci_attempt = fzero(evaluateCi_Farquhar, vars_attempt, &params, guess, attempt_converged, 0.001f, 200);

        // Check if solution converged and is physiologically reasonable
        if (attempt_converged && Ci_attempt > 0 && std::isfinite(Ci_attempt)) {
            Ci = Ci_attempt;
            variables = vars_attempt;
            overall_converged = true;
            break;
        }
    }

    // If all initial guesses failed, add warning to aggregator
    if (!overall_converged) {
        warnings.addWarning("photosynthesis_ci_convergence_failure", "Photosynthesis model failed to converge for Ci calculation after trying multiple initial guesses.");
        // Use best available estimate from last attempt
        bool final_converged = false;
        Ci = fzero(evaluateCi_Farquhar, variables, &params, 100.0f, final_converged, 0.01f, 500);
    }

    A = variables[4];
    limitation_state = (int) variables[5];
    Gamma = variables[6];
    J_over_Jmax = variables[8];

    return A;
}

EmpiricalModelCoefficients PhotosynthesisModel::getEmpiricalModelCoefficients(uint UUID) {
    return getCoefficientsForPrimitive_Empirical(UUID);
}

FarquharModelCoefficients PhotosynthesisModel::getFarquharModelCoefficients(uint UUID) {
    return getCoefficientsForPrimitive_Farquhar(UUID);
}

void PhotosynthesisModel::disableMessages() {
    message_flag = false;
}

void PhotosynthesisModel::enableMessages() {
    message_flag = true;
}

void PhotosynthesisModel::optionalOutputPrimitiveData(const char *label) {

    if (strcmp(label, "Ci") == 0 || strcmp(label, "limitation_state") == 0 || strcmp(label, "Gamma_CO2") == 0 || strcmp(label, "electron_transport_ratio") == 0 || strcmp(label, "Cm") == 0 || strcmp(label, "Vp") == 0) {
        output_prim_data.emplace_back(label);
    } else {
        if (message_flag) {
            static bool unknown_output_warning_shown = false;
            if (!unknown_output_warning_shown) {
                std::cerr << "WARNING (PhotosynthesisModel::optionalOutputPrimitiveData): unknown output primitive data " << label << std::endl;
                unknown_output_warning_shown = true;
            }
        }
    }
}

void PhotosynthesisModel::printDefaultValueReport() const {
    printDefaultValueReport(context->getAllUUIDs());
}

void PhotosynthesisModel::printDefaultValueReport(const std::vector<uint> &UUIDs) const {

    size_t assumed_default_i = 0;
    size_t assumed_default_TL = 0;
    size_t assumed_default_CO2 = 0;
    size_t assumed_default_gM = 0;
    size_t assumed_default_gH = 0;

    size_t Nprimitives = UUIDs.size();

    for (uint UUID: UUIDs) {

        if (!context->doesPrimitiveDataExist(UUID, "radiation_flux_PAR") || context->getPrimitiveDataType("radiation_flux_PAR") != HELIOS_TYPE_FLOAT) {
            assumed_default_i++;
        }

        // surface temperature (K)
        if (!context->doesPrimitiveDataExist(UUID, "temperature") || context->getPrimitiveDataType("temperature") != HELIOS_TYPE_FLOAT) {
            assumed_default_TL++;
        }

        // boundary-layer conductance to heat
        if ((!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance") || context->getPrimitiveDataType("boundarylayer_conductance") != HELIOS_TYPE_FLOAT) &&
            (!context->doesPrimitiveDataExist(UUID, "boundarylayer_conductance_out") || context->getPrimitiveDataType("boundarylayer_conductance_out") != HELIOS_TYPE_FLOAT)) {
            assumed_default_gH++;
        }

        // stomatal conductance
        if (!context->doesPrimitiveDataExist(UUID, "moisture_conductance") || context->getPrimitiveDataType("moisture_conductance") != HELIOS_TYPE_FLOAT) {
            assumed_default_gM++;
        }

        // ambient air CO2
        if (!context->doesPrimitiveDataExist(UUID, "air_CO2") || context->getPrimitiveDataType("air_CO2") != HELIOS_TYPE_FLOAT) {
            assumed_default_CO2++;
        }
    }

    std::cout << "--- Photosynthesis Model Default Value Report ---" << std::endl;

    std::cout << "PAR flux: " << assumed_default_i << " of " << Nprimitives << " used default value of " << i_PAR_default
              << " because "
                 "radiation_flux_PAR"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "surface temperature: " << assumed_default_TL << " of " << Nprimitives << " used default value of " << TL_default
              << " because "
                 "temperature"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "boundary-layer conductance: " << assumed_default_gH << " of " << Nprimitives << " used default value of " << gH_default
              << " because "
                 "boundarylayer_conductance"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "moisture conductance: " << assumed_default_gM << " of " << Nprimitives << " used default value of " << gM_default
              << " because "
                 "moisture_conductance"
                 " primitive data did not exist"
              << std::endl;
    std::cout << "air CO2: " << assumed_default_CO2 << " of " << Nprimitives << " used default value of " << CO2_default
              << " because "
                 "air_CO2"
                 " primitive data did not exist"
              << std::endl;

    std::cout << "--------------------------------------------------" << std::endl;
}
