/** \file "PhotosynthesisModel.h" Primary header file for photosynthesis plug-in.

Copyright (C) 2016-2026 Brian Bailey

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

*/

#ifndef PHOTOSYNTHESIS_MODEL
#define PHOTOSYNTHESIS_MODEL

#include "Context.h"

#include <limits>

struct PhotosyntheticTemperatureResponseParameters {
private:
    static void validateOptimalTemperature(float optimum_temperature_in_C) {
        if (optimum_temperature_in_C < 0.f) {
            helios::helios_runtime_error("ERROR (PhotosyntheticTemperatureResponseParameters): Optimal temperature cannot be negative. Received Topt = " + std::to_string(optimum_temperature_in_C) +
                                         " C. Please check that temperature is provided in units of Celsius, not Kelvin.");
        }
        if (optimum_temperature_in_C > 100.f) {
            helios::helios_runtime_error("ERROR (PhotosyntheticTemperatureResponseParameters): Optimal temperature cannot exceed 100 C. Received Topt = " + std::to_string(optimum_temperature_in_C) +
                                         " C. This value is biologically unrealistic and likely indicates temperature was provided in Kelvin instead of Celsius. Please convert to Celsius (subtract 273.15 from Kelvin value).");
        }
    }

public:
    PhotosyntheticTemperatureResponseParameters() {
        value_at_25C = 100.0f;
        dHa = 60.0f;
        dHd = 600.0f;
        Topt = 10000.f;
    }

    PhotosyntheticTemperatureResponseParameters(float value_at_25C) {
        this->value_at_25C = value_at_25C;
        this->dHa = 0.f;
        this->dHd = 600.f;
        this->Topt = 10000.f;
    }

    PhotosyntheticTemperatureResponseParameters(float value_at_25C, float rate_of_increase_dHa) {
        this->value_at_25C = value_at_25C;
        this->dHa = rate_of_increase_dHa;
        if (rate_of_increase_dHa > 0.f) {
            this->dHd = 10.f * rate_of_increase_dHa;
        } else {
            this->dHd = 600.f;
        }
        this->Topt = 10000.f;
    }

    PhotosyntheticTemperatureResponseParameters(float value_at_25C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        validateOptimalTemperature(optimum_temperature_in_C);
        this->value_at_25C = value_at_25C;
        this->dHa = rate_of_increase_dHa;
        if (rate_of_increase_dHa > 0.f) {
            this->dHd = 10.f * rate_of_increase_dHa;
        } else {
            this->dHd = 600.f;
        }
        this->Topt = 273.15f + optimum_temperature_in_C;
    }

    PhotosyntheticTemperatureResponseParameters(float value_at_25C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        validateOptimalTemperature(optimum_temperature_in_C);
        this->value_at_25C = value_at_25C;
        this->dHa = rate_of_increase_dHa;
        this->dHd = rate_of_decrease_dHd;
        this->Topt = 273.15f + optimum_temperature_in_C;
    }

    float value_at_25C;
    float dHa;
    float dHd;
    float Topt;
};


struct EmpiricalModelCoefficients {

    EmpiricalModelCoefficients() {
        Tref = 298; // K
        Ci_ref = 290; // umol CO2/mol air
        Asat = 18.18; // umol/m^2-s
        theta = 62.03; // W/m^2
        Tmin = 290; // K
        Topt = 303; // K
        q = 0.344; // unitless
        R = 1.663e5; // umol-K^0.5/m^2-s
        ER = 3740; // 1/K
        kC = 0.791; // unitless
    }

    // reference values
    float Tref;
    float Ci_ref;

    // light response coefficients
    float Asat;
    float theta;

    // assimilation temperature response coefficients
    float Tmin;
    float Topt;
    float q;

    // respiration temperature response coefficients
    float R;
    float ER;

    // CO2 response coefficients
    float kC;
};

struct FarquharModelCoefficients {
public:
    FarquharModelCoefficients() {

        // parameters (at TL = 25C)
        Vcmax = -1.f; // umol/m^2/s
        Jmax = -1.f; // umol/m^2/s
        alpha = -1.f; // unitless
        Rd = -1.f; // umol/m^2/s

        O = 213.5; // ambient oxygen concentration (mmol/mol)

        // temperature parameters
        c_Rd = 18.72;
        c_Vcmax = 26.35;
        c_Jmax = 18.86;
        c_Gamma = 19.02;
        c_Kc = 38.05;
        c_Ko = 20.30;

        dH_Rd = 46.39f;
        dH_Vcmax = 65.33f;
        dH_Jmax = 46.36f;
        dH_Gamma = 37.83;
        dH_Kc = 79.43;
        dH_Ko = 36.38;

        TPU_flag = 0;
    }

    // options
    int TPU_flag; // run model with TPU limitation

    // parameters
    float Vcmax;
    float Jmax;
    float Rd;
    float alpha;
    float O;

    // temperature parameters
    float c_Rd;
    float c_Vcmax;
    float c_Jmax;
    float c_Gamma;
    float c_Kc;
    float c_Ko;

    float dH_Rd;
    float dH_Vcmax;
    float dH_Jmax;
    float dH_Gamma;
    float dH_Kc;
    float dH_Ko;

    //! Set Vcmax of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response.
    /**
     * \param[in] Vcmax Value of Vcmax.
     */
    void setVcmax(float Vcmax) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax);
    }

    //! Set Vcmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
     * \param[in] Vcmax_at_25_C Value of Vcmax at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Vcmax with temperature.
     */
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa);
    }

    //! Set Vcmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
     * \param[in] Vcmax_at_25_C Value of Vcmax at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Vcmax with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of Vcmax in Celsius.
     * \note The rate of decrease of Vcmax with temperature will be assumed with a biologically reasonable default value if not specified.
     */
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f * rate_of_increase_dHa;
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Vcmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
     * \param[in] Vcmax_at_25_C Value of Vcmax at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Vcmax with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of Vcmax in Celsius.
     * \param[in] rate_of_decrease_dHd Deactivation energy dHd controlling the rate of decrease of Vcmax with temperature beyond the optimum.
     */
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Jmax of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
     * \param[in] Jmax Value of Jmax.
     */
    void setJmax(float Jmax) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax);
    }

    //! Set Jmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
     * \param[in] Jmax_at_25_C Value of Jmax at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Jmax with temperature.
     */
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa);
    }

    //! Set Jmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
     * \param[in] Jmax_at_25_C Value of Jmax at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Jmax with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of Jmax in Celsius.
     */
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f * rate_of_increase_dHa;
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Jmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
     * \param[in] Jmax_at_25_C Value of Jmax at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Jmax with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of Jmax in Celsius.
     * \param[in] rate_of_decrease_dHd Deactivation energy dHd controlling the rate of decrease of Jmax with temperature beyond the optimum.
     */
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
     * \param[in] TPU Value of TPU at a reference of 25 Celsius.
     */
    void setTPU(float TPU) {
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU);
        this->TPU_flag = 1;
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
     * \param[in] TPU_at_25_C Value of TPU at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of TPU with temperature.
     * \note A temperature optimum will not be assumed if not specified.
     */
    void setTPU(float TPU_at_25_C, float rate_of_increase_dHa) {
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU_at_25_C, rate_of_increase_dHa);
        this->TPU_flag = 1;
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
     * \param[in] TPU_at_25_C Value of TPU at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of TPU with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of TPU in Celsius.
     * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
     */
    void setTPU(float TPU_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f * rate_of_increase_dHa;
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
        this->TPU_flag = 1;
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
     * \param[in] TPU_at_25_C Value of TPU at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of TPU with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of TPU in Celsius.
     * \param[in] rate_of_decrease_dHd Deactivation energy dHd controlling the rate of decrease of TPU with temperature beyond the optimum.
     */
    void setTPU(float TPU_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
        this->TPU_flag = 1;
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
     * \param[in] Rd Value of Rd at a reference of 25 Celsius.
     */
    void setRd(float Rd) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd);
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
     * \param[in] Rd_at_25_C Value of Rd at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Rd with temperature.
     */
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa);
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
     * \param[in] Rd_at_25_C Value of Rd at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Rd with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of Rd in Celsius.
     * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
     */
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f * rate_of_increase_dHa;
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model with temperature response ccording to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
     * \param[in] Rd_at_25_C Value of Rd at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Rd with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of Rd in Celsius.
     * \param[in] rate_of_decrease_dHd Deactivation energy dHd controlling the rate of decrease of Rd with temperature beyond the optimum.
     */
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
     * \param[in] alpha Value of alpha at a reference of 25 Celsius.
     */
    void setQuantumEfficiency_alpha(float alpha) {
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
     * \param[in] alpha_at_25_C Value of alpha at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of alpha with temperature.
     */
    void setQuantumEfficiency_alpha(float alpha_at_25_C, float rate_of_increase_dHa) {
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha_at_25_C, rate_of_increase_dHa);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model according to a modified Arrhenius equation with an optimum
    /**
     * \param[in] alpha_at_25_C Value of alpha at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of alpha with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of alpha in Celsius.
     * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
     */
    void setQuantumEfficiency_alpha(float alpha_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f * rate_of_increase_dHa;
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
     * \param[in] alpha_at_25_C Value of alpha at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of alpha with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of alpha in Celsius.
     * \param[in] rate_of_decrease_dHd Deactivation energy dHd controlling the rate of decrease of alpha with temperature beyond the optimum.
     */
    void setQuantumEfficiency_alpha(float alpha_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
     * \param[in] theta Value of theta at a reference of 25 Celsius.
     */
    void setLightResponseCurvature_theta(float theta) {
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
     * \param[in] theta_at_25_C Value of theta at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of theta with temperature.
     */
    void setLightResponseCurvature_theta(float theta_at_25_C, float rate_of_increase_dHa) {
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta_at_25_C, rate_of_increase_dHa);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
     * \param[in] theta_at_25_C Value of theta at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of theta with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of theta in Celsius.
     * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
     */
    void setLightResponseCurvature_theta(float theta_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f * rate_of_increase_dHa;
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
     * \param[in] theta_at_25_C Value of theta at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of theta with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of reaction in Celsius.
     * \param[in] rate_of_decrease_dHd Deactivation energy dHd controlling the rate of decrease of theta with temperature beyond the optimum.
     */
    void setLightResponseCurvature_theta(float theta_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set mesophyll conductance gm (mol CO2 / m^2 / s / bar) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response.
    /**
     * \param[in] gm Value of gm.
     * \note Default is std::numeric_limits<float>::infinity(), which reduces Cc to Ci (legacy behavior with no mesophyll diffusion limitation).
     */
    void setMesophyllConductance_gm(float gm) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm);
    }

    //! Set gm of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation.
    /**
     * \param[in] gm_at_25_C Value of gm at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of gm with temperature.
     */
    void setMesophyllConductance_gm(float gm_at_25_C, float rate_of_increase_dHa) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm_at_25_C, rate_of_increase_dHa);
    }

    //! Set gm of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum.
    /**
     * \param[in] gm_at_25_C Value of gm at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of gm with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of gm in Celsius.
     */
    void setMesophyllConductance_gm(float gm_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C);
    }

    //! Set gm of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum.
    /**
     * \param[in] gm_at_25_C Value of gm at a reference of 25 Celsius.
     * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of gm with temperature before the optimum.
     * \param[in] optimum_temperature_in_C Temperature optimum of gm in Celsius.
     * \param[in] rate_of_decrease_dHd Deactivation energy dHd controlling the rate of decrease of gm with temperature beyond the optimum.
     */
    void setMesophyllConductance_gm(float gm_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    PhotosyntheticTemperatureResponseParameters getVcmaxTempResponse() {
        return this->VcmaxTempResponse;
    }

    PhotosyntheticTemperatureResponseParameters getJmaxTempResponse() {
        return this->JmaxTempResponse;
    }

    PhotosyntheticTemperatureResponseParameters getTPUTempResponse() {
        return this->TPUTempResponse;
    }

    PhotosyntheticTemperatureResponseParameters getRdTempResponse() {
        return this->RdTempResponse;
    }

    PhotosyntheticTemperatureResponseParameters getQuantumEfficiencyTempResponse() {
        return this->alphaTempResponse;
    }

    PhotosyntheticTemperatureResponseParameters getLightResponseCurvatureTempResponse() {
        return this->thetaTempResponse;
    }

    PhotosyntheticTemperatureResponseParameters getMesophyllConductance_gmTempResponse() const {
        return this->gmTempResponse;
    }


private:
    PhotosyntheticTemperatureResponseParameters VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(100);
    PhotosyntheticTemperatureResponseParameters JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(200);
    PhotosyntheticTemperatureResponseParameters TPUTempResponse = PhotosyntheticTemperatureResponseParameters(2.0);
    PhotosyntheticTemperatureResponseParameters RdTempResponse = PhotosyntheticTemperatureResponseParameters(1.5);
    PhotosyntheticTemperatureResponseParameters alphaTempResponse = PhotosyntheticTemperatureResponseParameters(0.5);
    PhotosyntheticTemperatureResponseParameters thetaTempResponse = PhotosyntheticTemperatureResponseParameters(0.f);
    //! Mesophyll conductance gm (mol CO2 / m^2 / s / bar). Default = +infinity, meaning Cc ≡ Ci (no mesophyll diffusion limitation — legacy behavior).
    PhotosyntheticTemperatureResponseParameters gmTempResponse = PhotosyntheticTemperatureResponseParameters(std::numeric_limits<float>::infinity());


    friend class PhotosynthesisModel;
};


//! Coefficients for the von Caemmerer (2021) steady-state C4 photosynthesis model.
/** Parameters are divided into three groups:
 *  1) Temperature-responsive enzyme/electron-transport rates (Vpmax, Vcmax, Jmax, Rd, gm) — each has
 *     the same four setter overloads as the C3 Farquhar model (constant / Arrhenius / peaked / peaked+dHd).
 *  2) Rubisco + PEPC kinetic constants (Kc, Ko, Kp, γ*) — stored as public fields and evaluated with a
 *     simple Arrhenius response at the 25 °C reference, mirroring the C3 Farquhar treatment of Kc/Ko.
 *  3) User-tunable scalar parameters with paper defaults (α, x, Vpr, Rm_frac, fcyc, gbs, ao, Om, …).
 *  Defaults for Setaria viridis follow Table 1 of von Caemmerer, S. (2021) "Updating the steady-state
 *  model of C4 photosynthesis", Journal of Experimental Botany 72:6003–6017.
 */
struct C4ModelCoefficients {
public:
    C4ModelCoefficients() {
        // Flexible temperature responses — Setaria viridis defaults from Table 1 (Boyd et al. 2015 for Vpmax/Vcmax/Rd,
        // Ubierna et al. 2017 refit Arrhenius for gm). Jmax is re-fit from the paper's Gaussian form
        // Jmax(T) = 400·exp(−((T−43)/26)²) to the peaked-Arrhenius form used here — Jmax(25°C) = 247.69 μmol/m²/s
        // matches the spreadsheet at the 25°C reference; dHa / Topt / dHd chosen to approximate the Gaussian.
        VpmaxTempResponse = PhotosyntheticTemperatureResponseParameters(200.f, 50.1f);
        VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(40.f, 78.0f);
        JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(247.69f, 77.9f, 43.0f, 260.f);
        RdTempResponse = PhotosyntheticTemperatureResponseParameters(1.f, 66.4f);
        gmTempResponse = PhotosyntheticTemperatureResponseParameters(1.f, 49.8f);
    }

    // === (1) Flexible temperature-responsive parameters: setter overloads mirror C3 API ===

    //! Set Vpmax (maximum PEPC activity) with no temperature response.
    void setVpmax(float Vpmax) {
        this->VpmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vpmax);
    }
    //! Set Vpmax with a monotonically increasing Arrhenius response.
    void setVpmax(float Vpmax_at_25_C, float rate_of_increase_dHa) {
        this->VpmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vpmax_at_25_C, rate_of_increase_dHa);
    }
    //! Set Vpmax with a peaked Arrhenius response (optimum, default dHd).
    void setVpmax(float Vpmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        this->VpmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vpmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C);
    }
    //! Set Vpmax with a peaked Arrhenius response (optimum + explicit rate of decrease).
    void setVpmax(float Vpmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->VpmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vpmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Vcmax (maximum Rubisco activity) with no temperature response.
    void setVcmax(float Vcmax) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax);
    }
    //! Set Vcmax with a monotonically increasing Arrhenius response.
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa);
    }
    //! Set Vcmax with a peaked Arrhenius response (optimum, default dHd).
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C);
    }
    //! Set Vcmax with a peaked Arrhenius response (optimum + explicit rate of decrease).
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Jmax (maximum linear electron transport rate) with no temperature response.
    void setJmax(float Jmax) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax);
    }
    //! Set Jmax with a monotonically increasing Arrhenius response.
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa);
    }
    //! Set Jmax with a peaked Arrhenius response.
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C);
    }
    //! Set Jmax with a peaked Arrhenius response with explicit rate of decrease.
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Rd (leaf mitochondrial respiration) with no temperature response.
    void setRd(float Rd) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd);
    }
    //! Set Rd with a monotonically increasing Arrhenius response.
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa);
    }
    //! Set Rd with a peaked Arrhenius response.
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C);
    }
    //! Set Rd with a peaked Arrhenius response with explicit rate of decrease.
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set mesophyll conductance gm (mol CO2 / m^2 / s / bar) with no temperature response.
    void setMesophyllConductance_gm(float gm) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm);
    }
    //! Set gm with a monotonically increasing Arrhenius response.
    void setMesophyllConductance_gm(float gm_at_25_C, float rate_of_increase_dHa) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm_at_25_C, rate_of_increase_dHa);
    }
    //! Set gm with a peaked Arrhenius response.
    void setMesophyllConductance_gm(float gm_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C);
    }
    //! Set gm with a peaked Arrhenius response with explicit rate of decrease.
    void setMesophyllConductance_gm(float gm_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->gmTempResponse = PhotosyntheticTemperatureResponseParameters(gm_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    // Accessors (read-only) used for material-data serialization
    PhotosyntheticTemperatureResponseParameters getVpmaxTempResponse() const { return VpmaxTempResponse; }
    PhotosyntheticTemperatureResponseParameters getVcmaxTempResponse() const { return VcmaxTempResponse; }
    PhotosyntheticTemperatureResponseParameters getJmaxTempResponse() const { return JmaxTempResponse; }
    PhotosyntheticTemperatureResponseParameters getRdTempResponse() const { return RdTempResponse; }
    PhotosyntheticTemperatureResponseParameters getMesophyllConductance_gmTempResponse() const { return gmTempResponse; }

    // === (2) Rubisco + PEPC kinetic constants — hardcoded 25 °C values with simple Arrhenius responses.
    //     Edit directly if different species are needed (paper defaults are for Setaria viridis). Units: μbar
    //     for Cm-like partial pressures, kJ/mol for activation energies.

    float Kc_25 = 1210.f;             //!< Michaelis constant of Rubisco for CO2 at 25 °C (μbar)
    float Ko_25 = 292000.f;           //!< Michaelis constant of Rubisco for O2 at 25 °C (μbar = 292 mbar)
    float Kp_25 = 82.f;               //!< Michaelis constant of PEPC for CO2 at 25 °C (μbar)
    float gamma_star_25 = 3.81679e-4f;//!< 0.5 / S_Rubisco at 25 °C (dimensionless)
    float Om_25 = 210000.f;           //!< Mesophyll O2 partial pressure at 25 °C (μbar = 210 mbar, ambient)

    float dH_Kc = 64.2f;              //!< Activation energy for Kc (kJ/mol)
    float dH_Ko = 10.5f;              //!< Activation energy for Ko (kJ/mol)
    float dH_Kp = 38.3f;              //!< Activation energy for Kp (kJ/mol)
    float dH_gamma_star = -31.1f;     //!< Activation energy for γ* (kJ/mol; negative per von Caemmerer 2021 spreadsheet — γ* decreases with T in this parameterization)
    float dH_Om = 0.f;                //!< Activation energy for Om (kJ/mol; ambient O2, assumed T-independent)

    // === (3) User-tunable scalar parameters (paper defaults in comments) ===

    float alpha_psII_fraction = 0.f;  //!< α — fraction of PSII activity in the bundle sheath (0 for NADP-ME; range [0,1])
    float x_etr_partition = 0.4f;     //!< x — fraction of linear electron-transport rate partitioned to the mesophyll
    float Vpr = 80.f;                 //!< PEP regeneration rate cap (μmol/m²/s)
    float Rm_frac = 0.5f;             //!< R_m = Rm_frac · R_d (mesophyll mitochondrial respiration as a fraction of total)
    float fcyc = 0.3f;                //!< Fraction of linear electron flow that is cyclic (Table 1, von Caemmerer 2021)
    float gbs = 0.003f;               //!< Bundle-sheath conductance to CO2 (mol/m²/s/bar), constant with temperature
    float ao = 0.047f;                //!< O2/CO2 solubility-diffusivity ratio (a_o in Eq. 15)
    float absorptance = 0.85f;        //!< Leaf PAR absorptance (fraction of incident PAR absorbed)
    float f_spectral = 0.15f;         //!< Spectral-quality correction to absorbed PAR (Eq. 35)
    float theta_etr = 0.7f;           //!< Curvature of the J ~ I2 non-rectangular hyperbola (Eq. 34)
    float h_protons = 4.f;            //!< Protons required per ATP synthesized (Eq. 31)

private:
    PhotosyntheticTemperatureResponseParameters VpmaxTempResponse;
    PhotosyntheticTemperatureResponseParameters VcmaxTempResponse;
    PhotosyntheticTemperatureResponseParameters JmaxTempResponse;
    PhotosyntheticTemperatureResponseParameters RdTempResponse;
    PhotosyntheticTemperatureResponseParameters gmTempResponse;

    friend class PhotosynthesisModel;
};


class PhotosynthesisModel {
public:
    //! Default constructor
    /**
     * \param[in] a_context Pointer to the helios context
     */
    explicit PhotosynthesisModel(helios::Context *a_context);

    static int selfTest(int argc = 0, char **argv = nullptr);

    //! Sets photosynthesis to be calculated according to the empirical model
    void setModelType_Empirical();

    //! Sets photosynthesis to be calculated according to the Farquhar-von Caemmerer-Berry model
    void setModelType_Farquhar();

    //! Sets photosynthesis to be calculated according to the von Caemmerer (2021) steady-state C4 model
    void setModelType_C4();

    //! Set the empirical model coefficients for all primitives
    /**
     * \param[in] modelcoefficients Set of model coefficients, which will be applied to all primitives.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients);

    //! Set the empirical model coefficients for a subset of primitives based on their UUIDs
    /**
     * \param[in] modelcoefficients Set of model coefficients.
     * \param[in] UUIDs Universal unique identifiers for primitives to be set.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const EmpiricalModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs);

    //! Set the Farquhar-von Caemmerer-Berry model coefficients for all primitives
    /**
     * \param[in] modelcoefficients Set of model coefficients, which will be applied to all primitives.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const FarquharModelCoefficients &modelcoefficients);

    //! Set the Farquhar-von Caemmerer-Berry model coefficients for a subset of primitives based on their UUIDs
    /**
     * \param[in] modelcoefficients Set of model coefficients.
     * \param[in] UUIDs Universal unique identifiers for primitives to be set.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const FarquharModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs);

    //! Set the Farquhar-von Caemmerer-Berry model coefficients for a subset of primitives, where coefficients differ between primitives.
    /**
     * \param[in] modelcoefficients Farquhar-von Caemmerer-Berry model coefficients for each primitive.
     * \param[in] UUIDs Vector of universal unique identifiers for primitives to be set.
     * \note The size of modelcoefficients and UUIDs must be equal.
     */
    void setModelCoefficients(const std::vector<FarquharModelCoefficients> &modelcoefficients, const std::vector<uint> &UUIDs);

    //! Set the Farquhar-von Caemmerer-Berry model coefficients for all primitives based on a species from the library.
    /**
     * \param[in] species Name of species from the library.
     */
    void setFarquharCoefficientsFromLibrary(const std::string &species);

    //! Set the Farquhar-von Caemmerer-Berry model coefficients for a subset of primitives based on a species from the library.
    /**
     * \param[in] species Name of species from the library.
     * \param[in] UUIDs Vector of universal unique identifiers for primitives to be set.
     */
    void setFarquharCoefficientsFromLibrary(const std::string &species, const std::vector<uint> &UUIDs);

    //! Get Farquhar-von Caemmerer-Berry model coefficients for a species from the library.
    /**
     * \param[in] species Name of species from the library.
     * \return Farquhar-von Caemmerer-Berry model coefficients for the species.
     */
    FarquharModelCoefficients getFarquharCoefficientsFromLibrary(const std::string &species);

    //! Set empirical model coefficients for a material by label
    /**
     * \param[in] material_label String identifier for the material
     * \param[in] coeffs Model coefficient values
     */
    void setModelCoefficients(const std::string &material_label, const EmpiricalModelCoefficients &coeffs);

    //! Set Farquhar model coefficients for a material by label
    /**
     * \param[in] material_label String identifier for the material
     * \param[in] coeffs Model coefficient values
     */
    void setModelCoefficients(const std::string &material_label, const FarquharModelCoefficients &coeffs);

    //! Set Farquhar model coefficients from species library for a material
    /**
     * \param[in] species Name of species
     * \param[in] material_label String identifier for the material
     */
    void setFarquharCoefficientsFromLibrary(const std::string &species, const std::string &material_label);

    //! Set the von Caemmerer (2021) C4 model coefficients for all primitives
    /**
     * \param[in] modelcoefficients Set of C4 model coefficients, which will be applied to all primitives.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const C4ModelCoefficients &modelcoefficients);

    //! Set the von Caemmerer (2021) C4 model coefficients for a subset of primitives based on their UUIDs
    /**
     * \param[in] modelcoefficients Set of C4 model coefficients.
     * \param[in] UUIDs Universal unique identifiers for primitives to be set.
     * \note The model type will be set based on the most recent call to setModelCoefficients().
     */
    void setModelCoefficients(const C4ModelCoefficients &modelcoefficients, const std::vector<uint> &UUIDs);

    //! Set C4 model coefficients for a material by label
    /**
     * \param[in] material_label String identifier for the material
     * \param[in] coeffs C4 model coefficient values
     * \note All primitives with this material at run() time will use these coefficients (cached after first lookup).
     */
    void setModelCoefficients(const std::string &material_label, const C4ModelCoefficients &coeffs);

    //! Run the model for all UUIDs in the Context
    void run();

    //! Run the model for a select sub-set of UUIDs
    /**
     * \param[in] lUUIDs Universal unique identifiers for primitives to run the model on.
     */
    void run(const std::vector<uint> &lUUIDs);

    //! Get the current model coefficients for the empirical model
    /**
     * \param[in] UUID Universal unique identifier for the primitive
     * \return Empirical model coefficients for the primitive
     */
    EmpiricalModelCoefficients getEmpiricalModelCoefficients(uint UUID);

    //! Get the current model coefficients for the Farquhar-von Caemmerer-Berry model
    /**
     * \param[in] UUID Universal unique identifier for the primitive
     * \return Farquhar-von Caemmerer-Berry model coefficients for the primitive
     */
    FarquharModelCoefficients getFarquharModelCoefficients(uint UUID);

    //! Get the current model coefficients for the von Caemmerer (2021) C4 model
    /**
     * \param[in] UUID Universal unique identifier for the primitive
     * \return C4 model coefficients for the primitive
     */
    C4ModelCoefficients getC4ModelCoefficients(uint UUID);

    //! Disable output messages to the standard output
    void disableMessages();

    //! Enable output messages to the standard output
    void enableMessages();

    //! Add optional output primitive data values to the Context
    /**
     * \param[in] label Name of primitive data. Available labels for the Farquhar model: "Ci" (intercellular CO2), "limitation_state" (0=Rubisco-limited, 1=electron transport-limited), "Gamma_CO2" (CO2 compensation point), "electron_transport_ratio"
     * (J/Jmax ratio, useful for fluorescence calculations). Additional labels for the C4 model: "Ci", "limitation_state" (1=enzyme-limited, 2=electron-transport-limited), "Cm" (mesophyll cytosolic CO2), "Vp" (PEP carboxylation rate).
     */
    void optionalOutputPrimitiveData(const char *label);

    //! Manually set the intercellular CO2 concentration (Ci) for specified primitives, bypassing iterative calculation
    /**
     * \param[in] Ci Intercellular CO2 concentration in units of umol CO2/mol air
     * \param[in] UUIDs Universal unique identifiers for primitives to set manual Ci
     * \note This method is primarily intended for testing and validation purposes. For normal operation, Ci should be calculated from moisture_conductance.
     * \note Manual Ci values will persist across multiple run() calls until overwritten with another setCi() call.
     * \note Ci must be positive and should typically be between 50-800 umol/mol for C3 plants (typical range 0.3-0.9 times ambient CO2).
     */
    void setCi(float Ci, const std::vector<uint> &UUIDs);

    //! Manually set the mesophyll cytosolic CO2 partial pressure (Cm) for specified primitives (C4 model only)
    /**
     * \param[in] Cm Mesophyll cytosolic CO2 partial pressure in units of ubar (equivalent to umol CO2/mol at 1 atm total pressure)
     * \param[in] UUIDs Universal unique identifiers for primitives to set manual Cm
     * \note Only applies to the C4 model. When set, the C4 solver uses the supplied Cm directly and skips the Cm = Ci - A/gm fixed-point iteration; the stomatal balance on Ci is also bypassed (Ci = Cm + A/gm is back-computed).
     * \note This method is primarily intended for testing and validation against published reference data (e.g., the von Caemmerer 2021 spreadsheet sweeps Cm directly).
     */
    void setCm(float Cm, const std::vector<uint> &UUIDs);

    //! Print a report detailing usage of default input values for all primitives in the Context
    void printDefaultValueReport() const;

    //! Print a report detailing usage of default input values based on a subset of primitive UUIDs
    /**
     * \param[in] UUIDs Universal unique identifiers for report
     */
    void printDefaultValueReport(const std::vector<uint> &UUIDs) const;

private:
    //! Pointer to the Helios context
    helios::Context *context;

    std::string model;
    EmpiricalModelCoefficients empiricalmodelcoeffs;
    FarquharModelCoefficients farquharmodelcoeffs;
    C4ModelCoefficients c4modelcoeffs;

    std::unordered_map<uint, EmpiricalModelCoefficients> empiricalmodel_coefficients;
    std::unordered_map<uint, FarquharModelCoefficients> farquharmodel_coefficients;
    std::unordered_map<uint, C4ModelCoefficients> c4model_coefficients;

    // Cache to avoid repeated Context lookups during run()
    mutable std::unordered_map<uint, EmpiricalModelCoefficients> material_coefficient_cache_empirical;
    mutable std::unordered_map<uint, FarquharModelCoefficients> material_coefficient_cache_farquhar;
    mutable std::unordered_map<uint, C4ModelCoefficients> material_coefficient_cache_c4;

    // Helper to retrieve coefficients with caching
    EmpiricalModelCoefficients getCoefficientsForPrimitive_Empirical(uint UUID) const;
    FarquharModelCoefficients getCoefficientsForPrimitive_Farquhar(uint UUID) const;
    C4ModelCoefficients getCoefficientsForPrimitive_C4(uint UUID) const;

    //! Storage for previous timestep Ci values for temporal continuity (O(1) lookup performance)
    std::unordered_map<uint, float> previous_Ci;

    //! Storage for manual Ci overrides that bypass iterative calculation
    std::unordered_map<uint, float> manual_Ci;

    //! Storage for manual Cm overrides (C4 model only) that bypass both stomatal iteration and Cm = Ci - A/gm fixed point
    std::unordered_map<uint, float> manual_Cm;

    float evaluateEmpiricalModel(const EmpiricalModelCoefficients &params, float i_PAR, float TL, float CO2, float gM);

    float evaluateFarquharModel(const FarquharModelCoefficients &params, float i_PAR, float TL, float CO2, float gM, float &Ci, float &Gamma, int &limitation_state, int &TPU_flag, float &J_over_Jmax, helios::WarningAggregator &warnings);

    float evaluateC4Model(const C4ModelCoefficients &params, float i_PAR, float TL, float CO2, float gM, float &Ci, float &Cm, float &Vp, int &limitation_state, helios::WarningAggregator &warnings);

    float evaluateCi_Empirical(const EmpiricalModelCoefficients &params, float Ci, float CO2, float fL, float Rd, float gM) const;

    static float evaluateCi_Farquhar(float Ci, std::vector<float> &variables, const void *parameters);

    static float evaluateCi_C4(float Ci, std::vector<float> &variables, const void *parameters);

    //! Direct evaluation of the C4 model given Cm (skips the Cm = Ci - A/gm iteration). Writes A, limitation, Vp into `variables` at the same indices used by evaluateCi_C4. Returns A.
    static float evaluateCm_C4(float Cm, std::vector<float> &variables, const C4ModelCoefficients &params);

    static float respondToTemperature(const PhotosyntheticTemperatureResponseParameters *temperatureResponseParameters, float T);

    float i_PAR_default;
    float TL_default;
    float CO2_default;
    float gM_default;
    float gH_default;

    bool message_flag = true;

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;
};

#endif
