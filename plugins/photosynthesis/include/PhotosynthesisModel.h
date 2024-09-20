/** \file "PhotosynthesisModel.h" Primary header file for photosynthesis plug-in.

Copyright (C) 2016-2023 Brian Bailey

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
        dHa = 2.0f;
        dHd = 2.0f;
        Topt = 273.15f + 38.f;
    }

    PhotosyntheticTemperatureResponseParameters(float value_at_25C) {
        this->value_at_25C = value_at_25C;
        this->dHa = 60.f;
        this->dHd = 1000.f;
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

    void setVcmax(float Vcmax_25C, float rate_of_increase_dHa = 65.33f, float optimum_temperature_in_C = 10000.f,
                  float rate_of_decrease_dHd = 1000.f) {
        this->VcmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Vcmax_25C, rate_of_increase_dHa,
                                                                              optimum_temperature_in_C,
                                                                              rate_of_decrease_dHd);
    }

    void setJmax(float Jmax_25C, float rate_of_increase_dHa = 46.36f, float optimum_temperature_in_C = 10000.f,
                 float rate_of_decrease_dHd = 1000.f) {
        this->JmaxTempResponse = PhotosyntheticTemperatureResponseParameters(Jmax_25C, rate_of_increase_dHa,
                                                                             optimum_temperature_in_C,
                                                                             rate_of_decrease_dHd);
    }

    void setTPU(float TPU_25C, float rate_of_increase_dHa = 46.0f, float optimum_temperature_in_C = 10000.f,
                float rate_of_decrease_dHd = 1000.f) {
        this->TPUTempResponse = PhotosyntheticTemperatureResponseParameters(TPU_25C, rate_of_increase_dHa,
                                                                            optimum_temperature_in_C,
                                                                            rate_of_decrease_dHd);
        this->TPU_flag = 1;
    }

    void setRd(float Rd_25C, float rate_of_increase_dHa = 46.39f, float optimum_temperature_in_C = 10000.f,
               float rate_of_decrease_dHd = 1000.f) {
        this->RdTempResponse = PhotosyntheticTemperatureResponseParameters(Rd_25C, rate_of_increase_dHa,
                                                                           optimum_temperature_in_C,
                                                                           rate_of_decrease_dHd);
    }

    void setQuantumEfficiency_alpha(float alpha_25C, float rate_of_increase_dHa = 0.f, float optimum_temperature_in_C = 10000.f, float rate_of_decrease_dHd = 1000.f) {
        this->alphaTempResponse = PhotosyntheticTemperatureResponseParameters(alpha_25C, rate_of_increase_dHa,
                                                                              optimum_temperature_in_C,
                                                                              rate_of_decrease_dHd);
    }

    void
    setLightResponseCurvature_theta(float theta_25C, float rate_of_increase_dHa = 0.f, float optimum_temperature_in_C = 10000.f, float rate_of_decrease_dHd = 1000.f) {
        this->thetaTempResponse = PhotosyntheticTemperatureResponseParameters(theta_25C, rate_of_increase_dHa,
                                                                              optimum_temperature_in_C,
                                                                              rate_of_decrease_dHd);
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

    //! Names of additional primitive data to add to the Context
    std::vector<std::string> output_prim_data;

};

#endif