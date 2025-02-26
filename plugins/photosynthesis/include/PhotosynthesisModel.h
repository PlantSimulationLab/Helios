/** \file "PhotosynthesisModel.h" Primary header file for photosynthesis plug-in.

Copyright (C) 2016-2025 Brian Bailey

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

struct PhotosyntheticTemperatureResponseParameters {
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

    PhotosyntheticTemperatureResponseParameters(float value_at_25C, float rate_of_increase_dHa,
                                                float optimum_temperature_in_C) {
        this->value_at_25C = value_at_25C;
        this->dHa = rate_of_increase_dHa;
        if (rate_of_increase_dHa > 0.f) {
            this->dHd = 10.f * rate_of_increase_dHa;
        } else {
            this->dHd = 600.f;
        }
        this->Topt = 273.15f + optimum_temperature_in_C;
    }

    PhotosyntheticTemperatureResponseParameters(float value_at_25C, float rate_of_increase_dHa,
                                                float optimum_temperature_in_C, float rate_of_decrease_dHd) {
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
        Tref = 298; //K
        Ci_ref = 290;  //umol CO2/mol air
        Asat = 18.18;  //umol/m^2-s
        theta = 62.03; //W/m^2
        Tmin = 290; //K
        Topt = 303; //K
        q = 0.344; //unitless
        R = 1.663e5; //umol-K^0.5/m^2-s
        ER = 3740; //1/K
        kC = 0.791; //unitless
    }

    //reference values
    float Tref;
    float Ci_ref;

    //light response coefficients
    float Asat;
    float theta;

    //assimilation temperature response coefficients
    float Tmin;
    float Topt;
    float q;

    //respiration temperature response coefficients
    float R;
    float ER;

    //CO2 response coefficients
    float kC;

};

struct FarquharModelCoefficients {
public:
    FarquharModelCoefficients() {

        //parameters (at TL = 25C)
        Vcmax = -1.f; //umol/m^2/s
        Jmax = -1.f; //umol/m^2/s
        alpha = -1.f; //unitless
        Rd = -1.f; //umol/m^2/s

        O = 213.5; //ambient oxygen concentration (mmol/mol)

        //temperature parameters
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

    //options
    int TPU_flag; // run model with TPU limitation

    //parameters
    float Vcmax;
    float Jmax;
    float Rd;
    float alpha;
    float O;

    //temperature parameters
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
    * \param[in] Vcmax_at_25_C Value of Vcmax at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Vcmax with temperature.
    */
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa);
    }

    //! Set Vcmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
    * \param[in] Vcmax_at_25_C Value of Vcmax at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Vcmax with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of Vcmax in Celcius.
    * \note The rate of decrease of Vcmax with temperature will be assumed with a biologically reasonable default value if not specified.
    */
    void setVcmax(float Vcmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f*rate_of_increase_dHa;
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Vcmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
    * \param[in] Vcmax_at_25_C Value of Vcmax at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Vcmax with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of Vcmax in Celcius.
    * \param[in] rate_of_decrease_dHd Dectivation energy dHd controlling the rate of decrease of Vcmax with temperature beyond the optimum.
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
    * \param[in] Jmax_at_25_C Value of Jmax at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Jmax with temperature.
    */
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa);
    }

    //! Set Jmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
    * \param[in] Jmax_at_25_C Value of Jmax at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Jmax with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of Jmax in Celcius.
    */
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f*rate_of_increase_dHa;
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Jmax of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
    * \param[in] Jmax_at_25_C Value of Jmax at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Jmax with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of Jmax in Celcius.
    * \param[in] rate_of_decrease_dHd Dectivation energy dHd controlling the rate of decrease of Jmax with temperature beyond the optimum.
    */
    void setJmax(float Jmax_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
    * \param[in] TPU Value of TPU at a reference of 25 Celcius.
    */
    void setTPU(float TPU) {
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU);
        this->TPU_flag = 1;
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
    * \param[in] TPU_at_25_C Value of TPU at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of TPU with temperature.
    * \note A temperature optimum will not be assumed if not specified.
    */
    void setTPU(float TPU_at_25_C, float rate_of_increase_dHa) {
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU_at_25_C, rate_of_increase_dHa);
        this->TPU_flag = 1;
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
    * \param[in] TPU_at_25_C Value of TPU at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of TPU with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of TPU in Celcius.
    * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
    */
    void setTPU(float TPU_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f*rate_of_increase_dHa;
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
        this->TPU_flag = 1;
    }

    //! Set Triose-Phosphate Utilization (TPU) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
    * \param[in] TPU_at_25_C Value of TPU at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of TPU with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of TPU in Celcius.
    * \param[in] rate_of_decrease_dHd Dectivation energy dHd controlling the rate of decrease of TPU with temperature beyond the optimum.
    */
    void setTPU(float TPU_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
        this->TPU_flag = 1;
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
    * \param[in] Rd Value of Rd at a reference of 25 Celcius.
    */
    void setRd(float Rd) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd);
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
    * \param[in] Rd_at_25_C Value of Rd at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Rd with temperature.
    */
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa);
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
    * \param[in] Rd_at_25_C Value of Rd at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Rd with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of Rd in Celcius.
    * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
    */
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f*rate_of_increase_dHa;
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set mitochondrial respiration rate (Rd) of the Farquhar-von Caemmerer-Berry model with temperature response ccording to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
    * \param[in] Rd_at_25_C Value of Rd at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of Rd with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of Rd in Celcius.
    * \param[in] rate_of_decrease_dHd Dectivation energy dHd controlling the rate of decrease of Rd with temperature beyond the optimum.
    */
    void setRd(float Rd_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
    * \param[in] alpha Value of alpha at a reference of 25 Celcius.
    */
    void setQuantumEfficiency_alpha(float alpha) {
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
    * \param[in] alpha_at_25_C Value of alpha at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of alpha with temperature.
    */
    void setQuantumEfficiency_alpha(float alpha_at_25_C, float rate_of_increase_dHa) {
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha_at_25_C, rate_of_increase_dHa);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model according to a modified Arrhenius equation with an optimum
    /**
    * \param[in] alpha_at_25_C Value of alpha at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of alpha with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of alpha in Celcius.
    * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
    */
    void setQuantumEfficiency_alpha(float alpha_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f*rate_of_increase_dHa;
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response quantum efficiency (alpha) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
    * \param[in] alpha_at_25_C Value of alpha at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of alpha with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of alpha in Celcius.
    * \param[in] rate_of_decrease_dHd Dectivation energy dHd controlling the rate of decrease of alpha with temperature beyond the optimum.
    */
    void setQuantumEfficiency_alpha(float alpha_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model to a constant value with no temperature response
    /**
    * \param[in] theta Value of theta at a reference of 25 Celcius.
    */
    void setLightResponseCurvature_theta(float theta) {
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model with temperature response according to a monotonically increasing Arrhenius equation
    /**
    * \param[in] theta_at_25_C Value of theta at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of theta with temperature.
    */
    void setLightResponseCurvature_theta(float theta_at_25_C, float rate_of_increase_dHa) {
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta_at_25_C, rate_of_increase_dHa);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum
    /**
    * \param[in] theta_at_25_C Value of theta at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of theta with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of theta in Celcius.
    * \note The rate of decrease will be assumed with a biologically reasonable default value if not specified.
    */
    void setLightResponseCurvature_theta(float theta_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C) {
        float rate_of_decrease_dHd = 10.f*rate_of_increase_dHa;
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
    }

    //! Set light response curvature (theta) of the Farquhar-von Caemmerer-Berry model with temperature response according to a modified Arrhenius equation with an optimum and specified rate of decrease beyond the optimum
    /**
    * \param[in] theta_at_25_C Value of theta at a reference of 25 Celcius.
    * \param[in] rate_of_increase_dHa Activation energy dHa controlling the rate of increase of theta with temperature before the optimum.
    * \param[in] optimum_temperature_in_C Temperature optimum of reaction in Celcius.
    * \param[in] rate_of_decrease_dHd Dectivation energy dHd controlling the rate of decrease of theta with temperature beyond the optimum.
    */
    void setLightResponseCurvature_theta(float theta_at_25_C, float rate_of_increase_dHa, float optimum_temperature_in_C, float rate_of_decrease_dHd) {
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta_at_25_C, rate_of_increase_dHa, optimum_temperature_in_C, rate_of_decrease_dHd);
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


private:
    PhotosyntheticTemperatureResponseParameters VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(100);
    PhotosyntheticTemperatureResponseParameters JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(200);
    PhotosyntheticTemperatureResponseParameters TPUTempResponse = PhotosyntheticTemperatureResponseParameters(2.0);
    PhotosyntheticTemperatureResponseParameters RdTempResponse = PhotosyntheticTemperatureResponseParameters(1.5);
    PhotosyntheticTemperatureResponseParameters alphaTempResponse = PhotosyntheticTemperatureResponseParameters(0.5);
    PhotosyntheticTemperatureResponseParameters thetaTempResponse = PhotosyntheticTemperatureResponseParameters(0.f);


    friend class PhotosynthesisModel;
};


class PhotosynthesisModel {
public:

    //! Default constructor
    /**
     * \param[in] a_context Pointer to the helios context
     */
    explicit PhotosynthesisModel(helios::Context *a_context);

    int selfTest();

    //! Sets photosynthesis to be calculated according to the empirical model
    void setModelType_Empirical();

    //! Sets photosynthesis to be calculated according to the Farquhar-von Caemmerer-Berry model
    void setModelType_Farquhar();

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

    //! Disable output messages to the standard output
    void disableMessages();

    //! Enable output messages to the standard output
    void enableMessages();

    //! Add optional output primitive data values to the Context
    /**
    * \param[in] label Name of primitive data (e.g., Ci)
    */
    void optionalOutputPrimitiveData(const char *label);

    //! Print a report detailing usage of default input values for all primitives in the Context
    void printDefaultValueReport() const;

    //! Print a report detailing usage of default input values based on a subset of primitive UUIDs
    /**
     * \params[in] UUIDs Universal unique identifiers for report
     */
    void printDefaultValueReport(const std::vector<uint> &UUIDs) const;

private:

    //! Pointer to the Helios context
    helios::Context *context;

    std::string model;
    EmpiricalModelCoefficients empiricalmodelcoeffs;
    FarquharModelCoefficients farquharmodelcoeffs;

    std::map<uint, EmpiricalModelCoefficients> empiricalmodel_coefficients;
    std::map<uint, FarquharModelCoefficients> farquharmodel_coefficients;

    float evaluateEmpiricalModel(const EmpiricalModelCoefficients &params, float i_PAR, float TL, float CO2, float gM);

    float evaluateFarquharModel(const FarquharModelCoefficients &params, float i_PAR, float TL, float CO2, float gM,
                                float &Ci, float &Gamma, int &limitation_state, int &TPU_flag);

    float evaluateCi_Empirical(const EmpiricalModelCoefficients &params, float Ci, float CO2, float fL, float Rd,
                               float gM) const;

    static float evaluateCi_Farquhar(float Ci, std::vector<float> &variables, const void *parameters);

    static float respondToTemperature(const PhotosyntheticTemperatureResponseParameters *temperatureResponseParameters,
                                      float T);

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