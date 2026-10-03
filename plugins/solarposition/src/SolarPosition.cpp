/** \file "SolarPosition.cpp" Primary source file for solar position model plug-in.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#include "SolarPosition.h"
#include "PragueSkyModelInterface.h"

using namespace std;
using namespace helios;

// Thermal atmosphere look-up table (see utilities/generate_thermal_atmosphere_lut.py): the largest view zenith angle it tabulates
static constexpr float thermal_max_view_zenith_deg = 60.f;

SolarPosition::SolarPosition(helios::Context *context_ptr) {
    context = context_ptr;
    UTC = context->getLocation().UTC_offset;
    latitude = context->getLocation().latitude_deg;
    longitude = context->getLocation().longitude_deg;
}


SolarPosition::SolarPosition(float UTC_hrs, float latitude_deg, float longitude_deg, helios::Context *context_ptr) {
    context = context_ptr;
    UTC = UTC_hrs;
    latitude = latitude_deg;
    longitude = longitude_deg;

    if (latitude_deg < -90 || latitude_deg > 90) {
        std::cerr << "WARNING (SolarPosition): Latitude must be between -90 and +90 deg (a latitude of " << latitude << " was given). Default latitude is being used." << std::endl;
        latitude = 38.55;
    } else {
        latitude = latitude_deg;
    }

    if (longitude < -180 || longitude > 180) {
        std::cerr << "WARNING (SolarPosition): Longitude must be between -180 and +180 deg (a longitude of " << longitude << " was given). Default longitude is being used." << std::endl;
        longitude = 121.76;
    } else {
        longitude = longitude_deg;
    }
}

SphericalCoord SolarPosition::calculateSunDirection(const helios::Time &time, const helios::Date &date) const {

    int solstice_day;
    float Gamma, delta, time_dec, B, EoT, TC, LST, h, theta, phi, rad, LSTM;

    rad = M_PI / 180.f;

    solstice_day = 81;

    // day angle (Iqbal Eq. 1.1.2)
    Gamma = 2.f * M_PI * (float(date.JulianDay() - 1)) / 365.f;

    // solar declination angle (Iqbal Eq. 1.3.1 after Spencer)
    delta = 0.006918f - 0.399912f * cos(Gamma) + 0.070257f * sin(Gamma) - 0.006758f * cos(2.f * Gamma) + 0.000907f * sin(2.f * Gamma) - 0.002697f * cos(3.f * Gamma) + 0.00148f * sin(3.f * Gamma);

    // equation of time (Iqbal Eq. 1.4.1 after Spencer)
    EoT = 229.18f * (0.000075f + 0.001868f * cos(Gamma) - 0.032077f * sin(Gamma) - 0.014615f * cos(2.f * Gamma) - 0.04089f * sin(2.f * Gamma));

    time_dec = time.hour + time.minute / 60.f; //(hours)

    LSTM = 15.f * UTC; // degrees

    TC = 4.f * (LSTM - longitude) + EoT; // minutes
    LST = time_dec + TC / 60.f; // hours

    h = (LST - 12.f) * 15.f * rad; // hour angle (rad)

    // solar zenith angle
    theta = asin_safe(sin(latitude * rad) * sin(delta) + cos(latitude * rad) * cos(delta) * cos(h)); //(rad)

    assert(theta > -0.5f * M_PI && theta < 0.5f * M_PI);

    // solar elevation angle
    phi = acos_safe((sin(delta) - sin(theta) * sin(latitude * rad)) / (cos(theta) * cos(latitude * rad)));

    if (LST > 12.f) {
        phi = 2.f * M_PI - phi;
    }

    assert(phi > 0 && phi < 2.f * M_PI);

    return make_SphericalCoord(theta, phi);
}

Time SolarPosition::getSunriseTime() const {

    Time result = make_Time(0, 0); // default/fallback value
    bool found = false;

    for (uint h = 1; h <= 23 && !found; h++) {
        for (uint m = 1; m <= 59 && !found; m++) {
            SphericalCoord sun_dir = calculateSunDirection(make_Time(h, m, 0), context->getDate());
            if (sun_dir.elevation > 0) {
                result = make_Time(h, m);
                found = true;
            }
        }
    }

    return result;
}

Time SolarPosition::getSunsetTime() const {

    Time result = make_Time(0, 0); // default/fallback value
    bool found = false;

    for (int h = 23; h >= 1 && !found; h--) {
        for (int m = 59; m >= 1 && !found; m--) {
            SphericalCoord sun_dir = calculateSunDirection(make_Time(h, m, 0), context->getDate());
            if (sun_dir.elevation > 0) {
                result = make_Time(h, m);
                found = true;
            }
        }
    }

    return result;
}

float SolarPosition::getSunElevation() const {
    float elevation;
    if (issolarpositionoverridden) {
        elevation = sun_direction.elevation;
    } else {
        elevation = calculateSunDirection(context->getTime(), context->getDate()).elevation;
    }
    return elevation;
}

float SolarPosition::getSunZenith() const {
    float zenith;
    if (issolarpositionoverridden) {
        zenith = sun_direction.zenith;
    } else {
        zenith = calculateSunDirection(context->getTime(), context->getDate()).zenith;
    }
    return zenith;
}

float SolarPosition::getSunAzimuth() const {
    float azimuth;
    if (issolarpositionoverridden) {
        azimuth = sun_direction.azimuth;
    } else {
        azimuth = calculateSunDirection(context->getTime(), context->getDate()).azimuth;
    }
    return azimuth;
}

vec3 SolarPosition::getSunDirectionVector() const {
    SphericalCoord sundirection;
    if (issolarpositionoverridden) {
        sundirection = sun_direction;
    } else {
        sundirection = calculateSunDirection(context->getTime(), context->getDate());
    }
    return sphere2cart(sundirection);
}

SphericalCoord SolarPosition::getSunDirectionSpherical() const {
    SphericalCoord sundirection;
    if (issolarpositionoverridden) {
        sundirection = sun_direction;
    } else {
        sundirection = calculateSunDirection(context->getTime(), context->getDate());
    }
    return sundirection;
}

void SolarPosition::setSunDirection(const helios::SphericalCoord &sundirection) {
    issolarpositionoverridden = true;
    sun_direction = sundirection;
}

float SolarPosition::getSolarFlux(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const {
    // Deprecated method - kept for backward compatibility
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return Eb_PAR + Eb_NIR;
}

float SolarPosition::getSolarFluxPAR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const {
    // Deprecated method - kept for backward compatibility
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return Eb_PAR;
}

float SolarPosition::getSolarFluxNIR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const {
    // Deprecated method - kept for backward compatibility
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return Eb_NIR;
}

float SolarPosition::getDiffuseFraction(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const {
    // Deprecated method - kept for backward compatibility
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return fdiff;
}

void SolarPosition::GueymardSolarModel(float pressure, float temperature, float humidity, float turbidity, float &Eb_PAR, float &Eb_NIR, float &fdiff) const {

    float beta = turbidity;

    float theta = getSunZenith();

    if (theta <= 0.f || theta > 0.5 * M_PI) {
        Eb_PAR = 0.f;
        Eb_NIR = 0.f;
        fdiff = 1.f;
        return;
    }

    float m = pow(cos(theta) + 0.15 * pow(93.885 - theta * 180 / M_PI, -1.25), -1);

    float E0_PAR = 635.4;
    float E0_NIR = 709.7;

    vec2 alpha(1.3, 1.3);

    //---- Rayleigh ----//
    // NOTE: Rayleigh scattering dominates the atmospheric attenuation, and thus variations in the model predictions are almost entirely due to pressure (and theta)
    float mR = 1.f / (cos(theta) + 0.48353 * pow(theta * 180 / M_PI, 0.095846) / pow(96.741 - theta * 180 / M_PI, 1.754));

    float mR_p = mR * pressure / 101325;

    float TR_PAR = (1.f + 1.8169 * mR_p - 0.033454 * mR_p * mR_p) / (1.f + 2.063 * mR_p + 0.31978 * mR_p * mR_p);

    float TR_NIR = (1.f - 0.010394 * mR_p) / (1.f - 0.00011042 * mR_p * mR_p);

    //---- Uniform gasses ----//
    mR = 1.f / (cos(theta) + 0.48353 * pow(theta * 180 / M_PI, 0.095846) / pow(96.741 - theta * 180 / M_PI, 1.754));

    mR_p = mR * pressure / 101325;

    float Tg_PAR = (1.f + 0.95885 * mR_p + 0.012871 * mR_p * mR_p) / (1.f + 0.96321 * mR_p + 0.015455 * mR_p * mR_p);

    float Tg_NIR = (1.f + 0.27284 * mR_p - 0.00063699 * mR_p * mR_p) / (1.f + 0.30306 * mR_p);

    float BR_PAR = 0.5f * (0.89013 - 0.0049558 * mR + 0.000045721 * mR * mR);

    float BR_NIR = 0.5;

    float Ba = 1.f - exp(-0.6931 - 1.8326 * cos(theta));

    //---- Ozone -----//
    float uo = getOzoneColumn() * 0.001f; // O3 atm-cm
    float mo = m;

    float f1 = uo * (10.979 - 8.5421 * uo) / (1.f + 2.0115 * uo + 40.189 * uo * uo);
    float f2 = uo * (-0.027589 - 0.005138 * uo) / (1.f - 2.4857 * uo + 13.942 * uo * uo);
    float f3 = uo * (10.995 - 5.5001 * uo) / (1.f + 1.678 * uo + 42.406 * uo * uo);
    float To_PAR = (1.f + f1 * mo + f2 * mo * mo) / (1.f + f3 * mo);

    float To_NIR = 1.f;

    //---- Nitrogen ---- //
    float un = 0.0002; // N atm-cm
    float mw = 1.f / (cos(theta) + 1.065 * pow(theta * 180 / M_PI, 1.6132) / pow(111.55 - theta * 180 / M_PI, 3.2629));

    float g1 = (0.17499 + 41.654 * un - 2146.4 * un * un) / (1 + 22295 * un * un);
    float g2 = un * (-1.2134 + 59.324 * un) / (1.f + 8847.8 * un * un);
    float g3 = (0.17499 + 61.658 * un + 9196.4 * un * un) / (1.f + 74109 * un * un);
    float Tn_PAR = fmin(1.f, (1.f + g1 * mw + g2 * mw * mw) / (1.f + g3 * mw));
    float Tn_PAR_p = fmin(1.f, (1.f + g1 * 1.66 + g2 * 1.66 * 1.66) / (1.f + g3 * 1.66));

    float Tn_NIR = 1.f;
    float Tn_NIR_p = 1.f;

    //---- Water -----//
    float gamma = log(humidity) + 17.67 * (temperature - 273) / (243 + 25);
    float tdp = 243 * gamma / (17.67 - gamma) * 9 / 5 + 32; // dewpoint temperature in Fahrenheit
    float w = exp((0.1133 - log(4.0 + 1)) + 0.0393 * tdp); // cm of precipitable water
    // NOTE: precipitable water model from Viswanadham (1981), Eq. 5
    mw = 1.f / (cos(theta) + 1.1212 * pow(theta * 180 / M_PI, 0.6379) / pow(93.781 - theta * 180 / M_PI, 1.9203));

    float h1 = w * (0.065445 + 0.00029901 * w) / (1.f + 1.2728 * w);
    float h2 = w * (0.065687 + 0.0013218 * w) / (1.f + 1.2008 * w);
    float Tw_PAR = (1.f + h1 * mw) / (1.f + h2 * mw);
    float Tw_PAR_p = (1.f + h1 * 1.66) / (1.f + h2 * 1.66);

    float c1 = w * (19.566 - 1.6506 * w + 1.0672 * w * w) / (1.f + 5.4248 * w + 1.6005 * w * w);
    float c2 = w * (0.50158 - 0.14732 * w + 0.047584 * w * w) / (1.f + 1.1811 * w + 1.0699 * w * w);
    float c3 = w * (21.286 - 0.39232 * w + 1.2692 * w * w) / (1.f + 4.8318 * w + 1.412 * w * w);
    float c4 = w * (0.70992 - 0.23155 * w + 0.096541 * w * w) / (1.f + 0.44907 * w + 0.75425 * w * w);
    float Tw_NIR = (1.f + c1 * mw + c2 * mw * mw) / (1.f + c3 * mw + c4 * mw * mw);
    float Tw_NIR_p = (1.f + c1 * 1.66 + c2 * 1.66 * 1.66) / (1.f + c3 * 1.66 + c4 * 1.66 * 1.66);

    //---- Aerosol ----//
    float ma = 1.f / (cos(theta) + 0.16851 * pow(theta * 180 / M_PI, 0.18198) / pow(95.318 - theta * 180 / M_PI, 1.9542));
    float ua = log(1.f + ma * beta);

    float d0 = 0.57664 - 0.024743 * alpha.x;
    float d1 = (0.093942 - 0.2269 * alpha.x + 0.12848 * alpha.x * alpha.x) / (1.f + 0.6418 * alpha.x);
    float d2 = (-0.093819 + 0.36668 * alpha.x - 0.12775 * alpha.x * alpha.x) / (1.f - 0.11651 * alpha.x);
    float d3 = alpha.x * (0.15232 - 0.08721 * alpha.x + 0.012664 * alpha.x * alpha.x) / (1.f - 0.90454 * alpha.x + 0.26167 * alpha.x * alpha.x);
    float lambdae_PAR = (d0 + d1 * ua + d2 * ua * ua) / (1.f + d3 * ua * ua);
    float Ta_PAR = exp(-ma * beta * pow(lambdae_PAR, -alpha.x));

    float e0 = (1.183 - 0.022989 * alpha.y + 0.020829 * alpha.y * alpha.y) / (1.f + 0.11133 * alpha.y);
    float e1 = (-0.50003 - 0.18329 * alpha.y + 0.23835 * alpha.y * alpha.y) / (1.f + 1.6756 * alpha.y);
    float e2 = (-0.50001 + 1.1414 * alpha.y + 0.0083589 * alpha.y * alpha.y) / (1.f + 11.168 * alpha.y);
    float e3 = (-0.70003 - 0.73587 * alpha.y + 0.51509 * alpha.y * alpha.y) / (1.f + 4.7665 * alpha.y);
    float lambdae_NIR = (e0 + e1 * ua + e2 * ua * ua) / (1.f + e3 * ua);
    float Ta_NIR = exp(-ma * beta * pow(lambdae_NIR, -alpha.y));

    float omega_PAR = 1.0;
    float omega_NIR = 1.0;

    float Tas_PAR = exp(-ma * omega_PAR * beta * pow(lambdae_PAR, -alpha.x));

    float Tas_NIR = exp(-ma * omega_NIR * beta * pow(lambdae_NIR, -alpha.y));

    // direct irradiation
    Eb_PAR = TR_PAR * Tg_PAR * To_PAR * Tn_PAR * Tw_PAR * Ta_PAR * E0_PAR;
    Eb_NIR = TR_NIR * Tg_NIR * To_NIR * Tn_NIR * Tw_NIR * Ta_NIR * E0_NIR;
    float Eb = Eb_PAR + Eb_NIR;

    // diffuse irradiation
    float Edp_PAR = To_PAR * Tg_PAR * Tn_PAR_p * Tw_PAR_p * (BR_PAR * (1.f - TR_PAR) * pow(Ta_PAR, 0.25) + Ba * TR_PAR * (1.f - pow(Tas_PAR, 0.25))) * E0_PAR;
    float Edp_NIR = To_NIR * Tg_NIR * Tn_NIR_p * Tw_NIR_p * (BR_NIR * (1.f - TR_NIR) * pow(Ta_NIR, 0.25) + Ba * TR_NIR * (1.f - pow(Tas_NIR, 0.25))) * E0_NIR;
    float Edp = Edp_PAR + Edp_NIR;

    // diffuse fraction
    fdiff = Edp / (Eb + Edp);

    assert(fdiff >= 0.f && fdiff <= 1.f);
}

float SolarPosition::getAmbientLongwaveFlux(float temperature_K, float humidity_rel) const {
    // Deprecated method - kept for backward compatibility
    // Model from Prata (1996) Q. J. R. Meteorol. Soc.
    float e0 = 611.f * exp(17.502f * (temperature_K - 273.f) / ((temperature_K - 273.f) + 240.9f)) * humidity_rel; // Pascals
    float K = 0.465f; // cm-K/Pa
    float xi = e0 / temperature_K * K;
    float eps = 1.f - (1.f + xi) * exp(-sqrt(1.2f + 3.f * xi));

    return eps * 5.67e-8 * pow(temperature_K, 4);
}

float SolarPosition::turbidityResidualFunction(float turbidity, std::vector<float> &parameters, const void *a_solarpositionmodel) {

    auto *solarpositionmodel = reinterpret_cast<const SolarPosition *>(a_solarpositionmodel);

    float pressure = parameters.at(0);
    float temperature = parameters.at(1);
    float humidity = parameters.at(2);
    float flux_target = parameters.at(3);

    // Clamp turbidity to minimum positive value (optimization can try negative values)
    float turbidity_clamped = std::max(1e-5f, turbidity);

    // Use new API: set atmospheric conditions, then get flux
    // Note: const_cast is needed here because this is an optimization callback that needs to modify internal state
    auto *mutable_model = const_cast<SolarPosition *>(solarpositionmodel);
    mutable_model->setAtmosphericConditions(pressure, temperature, humidity, turbidity_clamped);
    float flux_model = mutable_model->getSolarFlux() * cosf(mutable_model->getSunZenith());
    return flux_model - flux_target;
}

float SolarPosition::calibrateTurbidityFromTimeseries(const std::string &timeseries_shortwave_flux_label_Wm2) const {

    if (!context->doesTimeseriesVariableExist(timeseries_shortwave_flux_label_Wm2.c_str())) {
        helios_runtime_error("ERROR (SolarPosition::calibrateTurbidityFromTimeseries): Timeseries variable " + timeseries_shortwave_flux_label_Wm2 + " does not exist.");
    }

    uint length = context->getTimeseriesLength(timeseries_shortwave_flux_label_Wm2.c_str());

    float min_flux = 1e6;
    float max_flux = 0;
    int max_flux_index = 0;
    for (int t = 0; t < length; t++) {
        float flux = context->queryTimeseriesData(timeseries_shortwave_flux_label_Wm2.c_str(), t);
        if (flux < min_flux) {
            min_flux = flux;
        }
        if (flux > max_flux) {
            max_flux = flux;
            max_flux_index = t;
        }
    }

    if (max_flux < 750 || max_flux > 1200) {
        helios_runtime_error("ERROR (SolarPosition::calibrateTurbidityFromTimeseries): The maximum flux for the timeseries data is not within the expected range. Either it is not solar flux data, or there are no clear sky days in the dataset");
    } else if (min_flux < 0) {
        helios_runtime_error("ERROR (SolarPosition::calibrateTurbidityFromTimeseries): The minimum flux for the timeseries data is negative. Solar fluxes cannot be negative.");
    }

    std::vector<float> parameters{101325, 300, 0.5, max_flux};

    SolarPosition solarposition_copy(UTC, latitude, longitude, context);
    Date date_max = context->queryTimeseriesDate(timeseries_shortwave_flux_label_Wm2.c_str(), max_flux_index);
    Time time_max = context->queryTimeseriesTime(timeseries_shortwave_flux_label_Wm2.c_str(), max_flux_index);

    solarposition_copy.setSunDirection(solarposition_copy.calculateSunDirection(time_max, date_max));

    helios::WarningAggregator warnings;
    float turbidity = fzero(turbidityResidualFunction, parameters, &solarposition_copy, 0.01, 0.0001f, 100, &warnings);
    warnings.report(std::cerr);

    return std::max(1e-4F, turbidity);
}

void SolarPosition::enableCloudCalibration(const std::string &timeseries_shortwave_flux_label_Wm2) {

    if (!context->doesTimeseriesVariableExist(timeseries_shortwave_flux_label_Wm2.c_str())) {
        helios_runtime_error("ERROR (SolarPosition::enableCloudCalibration): Timeseries variable " + timeseries_shortwave_flux_label_Wm2 + " does not exist.");
    }

    cloudcalibrationlabel = timeseries_shortwave_flux_label_Wm2;
}

void SolarPosition::disableCloudCalibration() {
    cloudcalibrationlabel = "";
}

void SolarPosition::setAtmosphericConditions(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) {
    // Validate input parameters
    if (pressure_Pa <= 0.f) {
        helios_runtime_error("ERROR (SolarPosition::setAtmosphericConditions): Atmospheric pressure must be positive. Got " + std::to_string(pressure_Pa) + " Pa.");
    }
    if (temperature_K <= 0.f) {
        helios_runtime_error("ERROR (SolarPosition::setAtmosphericConditions): Temperature must be positive Kelvin. Got " + std::to_string(temperature_K) + " K.");
    }
    if (humidity_rel < 0.f || humidity_rel > 1.f) {
        helios_runtime_error("ERROR (SolarPosition::setAtmosphericConditions): Relative humidity must be between 0 and 1. Got " + std::to_string(humidity_rel) + ".");
    }
    if (turbidity < 0.f) {
        helios_runtime_error("ERROR (SolarPosition::setAtmosphericConditions): Turbidity must be non-negative. Got " + std::to_string(turbidity) + ".");
    }

    // Set global data in the Context
    context->setGlobalData("atmosphere_pressure_Pa", pressure_Pa);
    context->setGlobalData("atmosphere_temperature_K", temperature_K);
    context->setGlobalData("atmosphere_humidity_rel", humidity_rel);
    context->setGlobalData("atmosphere_turbidity", turbidity);
}

void SolarPosition::getAtmosphericConditions(float &pressure_Pa, float &temperature_K, float &humidity_rel, float &turbidity) const {
    // Default values
    const float default_pressure = 101325.f; // 1 atm in Pa
    const float default_temperature = 300.f; // 27°C in K
    const float default_humidity = 0.5f; // 50%
    const float default_turbidity = 0.02f; // clear sky

    static bool warning_issued = false;

    // Check if global data exists and retrieve it, otherwise use defaults
    bool all_exist =
            context->doesGlobalDataExist("atmosphere_pressure_Pa") && context->doesGlobalDataExist("atmosphere_temperature_K") && context->doesGlobalDataExist("atmosphere_humidity_rel") && context->doesGlobalDataExist("atmosphere_turbidity");

    if (all_exist) {
        context->getGlobalData("atmosphere_pressure_Pa", pressure_Pa);
        context->getGlobalData("atmosphere_temperature_K", temperature_K);
        context->getGlobalData("atmosphere_humidity_rel", humidity_rel);
        context->getGlobalData("atmosphere_turbidity", turbidity);
    } else {
        if (!warning_issued) {
            std::cerr << "WARNING (SolarPosition::getAtmosphericConditions): Atmospheric conditions have not been set via setAtmosphericConditions(). Using default values: pressure=" << default_pressure << " Pa, temperature=" << default_temperature
                      << " K, humidity=" << default_humidity << ", turbidity=" << default_turbidity << std::endl;
            warning_issued = true;
        }
        pressure_Pa = default_pressure;
        temperature_K = default_temperature;
        humidity_rel = default_humidity;
        turbidity = default_turbidity;
    }
}

void SolarPosition::setGroundAlbedo(float albedo) {
    if (albedo < 0.f || albedo > 1.f) {
        helios_runtime_error("ERROR (SolarPosition::setGroundAlbedo): Ground albedo must be between 0 and 1. Got " + std::to_string(albedo) + ".");
    }
    context->setGlobalData("atmosphere_ground_albedo", albedo);
}

float SolarPosition::getGroundAlbedo() const {
    if (!context->doesGlobalDataExist("atmosphere_ground_albedo")) {
        return 0.2f;
    }
    float albedo;
    context->getGlobalData("atmosphere_ground_albedo", albedo);
    if (albedo < 0.f || albedo > 1.f) {
        helios_runtime_error("ERROR (SolarPosition::getGroundAlbedo): Global data 'atmosphere_ground_albedo' must be between 0 and 1, but has a value of " + std::to_string(albedo) + ". Set it with SolarPosition::setGroundAlbedo().");
    }
    return albedo;
}

void SolarPosition::setOzoneColumn(float ozone_DU) {
    if (!(ozone_DU > 0.f)) {
        helios_runtime_error("ERROR (SolarPosition::setOzoneColumn): The ozone column must be positive. Got " + std::to_string(ozone_DU) + " DU.");
    }
    context->setGlobalData("atmosphere_ozone_DU", ozone_DU);
}

float SolarPosition::getOzoneColumn() const {
    if (context->doesGlobalDataExist("atmosphere_ozone_DU")) {
        float ozone_DU;
        context->getGlobalData("atmosphere_ozone_DU", ozone_DU);
        if (!(ozone_DU > 0.f)) {
            helios_runtime_error("ERROR (SolarPosition::getOzoneColumn): Global data 'atmosphere_ozone_DU' must be positive, but has a value of " + std::to_string(ozone_DU) + ". Set it with SolarPosition::setOzoneColumn().");
        }
        return ozone_DU;
    }

    const std::vector<std::array<float, 12>> &climatology = getOzoneClimatology();
    const int Nbands = int(climatology.size());
    const float band_width_deg = 180.f / float(Nbands);

    // Latitude: interpolate linearly between the centers of the band containing the latitude and the adjacent band on the same side of its center. Next to a band without data, the value of the latitude's own
    // band (the zonal mean over the latitudes it spans) is used.
    const int band = std::clamp(int(std::floor((latitude + 90.f) / band_width_deg)), 0, Nbands - 1);
    const float band_center = -90.f + (float(band) + 0.5f) * band_width_deg;
    const int neighbor = latitude >= band_center ? band + 1 : band - 1;
    const float neighbor_weight = std::fabs(latitude - band_center) / band_width_deg;

    const uint DOY = context->getJulianDate();
    // Value for a month at the latitude, or NaN if the latitude's own band has no data that month
    auto monthly_value = [&](int month) {
        const float own_value = climatology[band][month];
        if (std::isnan(own_value)) {
            return own_value;
        }
        if (neighbor < 0 || neighbor >= Nbands || std::isnan(climatology[neighbor][month])) {
            return own_value;
        }
        return (1.f - neighbor_weight) * own_value + neighbor_weight * climatology[neighbor][month];
    };

    // Day of year: interpolate linearly between the middles of the months (of a non-leap year), wrapping from December to January. Next to a month without data, the value of the month containing the date is used.
    const int month_start[13] = {0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365};
    float month_middle[12];
    for (int m = 0; m < 12; m++) {
        month_middle[m] = 0.5f * float(month_start[m] + month_start[m + 1]) + 0.5f;
    }
    const float day = std::min(float(DOY), 365.f);
    int later_month = 0;
    while (later_month < 12 && month_middle[later_month] <= day) {
        later_month++;
    }
    const int earlier_month = (later_month + 11) % 12;
    later_month %= 12;
    float interval = month_middle[later_month] - month_middle[earlier_month];
    float elapsed = day - month_middle[earlier_month];
    if (interval <= 0.f) {
        interval += 365.f;
    }
    if (elapsed < 0.f) {
        elapsed += 365.f;
    }
    const float earlier_value = monthly_value(earlier_month);
    const float later_value = monthly_value(later_month);
    if (!std::isnan(earlier_value) && !std::isnan(later_value)) {
        const float later_weight = elapsed / interval;
        return (1.f - later_weight) * earlier_value + later_weight * later_value;
    }
    int calendar_month = 0;
    while (calendar_month < 11 && day > float(month_start[calendar_month + 1])) {
        calendar_month++;
    }
    const float calendar_month_value = monthly_value(calendar_month);
    if (std::isnan(calendar_month_value)) {
        helios_runtime_error("ERROR (SolarPosition::getOzoneColumn): The ozone climatology has no data for latitude " + std::to_string(latitude) + " degrees on day of year " + std::to_string(DOY) +
                             " (high-latitude winter, or poleward of 80 degrees). Set the ozone column with SolarPosition::setOzoneColumn().");
    }
    return calendar_month_value;
}

const std::vector<std::array<float, 12>> &SolarPosition::getOzoneClimatology() {
    static const std::vector<std::array<float, 12>> climatology = [] {
        const std::string path = "plugins/solarposition/ozone_climatology/sbuv_total_ozone_climatology.txt";
        std::ifstream file(path);
        if (!file.is_open()) {
            helios_runtime_error("ERROR (SolarPosition::getOzoneClimatology): Could not open ozone climatology file " + path + ".");
        }
        std::vector<std::array<float, 12>> bands;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::istringstream fields(line);
            float latitude_min, latitude_max;
            fields >> latitude_min >> latitude_max;
            if (!fields || latitude_min != -90.f + 5.f * float(bands.size()) || latitude_max != latitude_min + 5.f) {
                helios_runtime_error("ERROR (SolarPosition::getOzoneClimatology): Unexpected latitude band in " + path + ": " + line);
            }
            std::array<float, 12> monthly{};
            for (float &value: monthly) {
                std::string token;
                fields >> token;
                if (token.empty()) {
                    helios_runtime_error("ERROR (SolarPosition::getOzoneClimatology): Missing monthly value in " + path + ": " + line);
                }
                value = token == "nan" ? std::numeric_limits<float>::quiet_NaN() : std::stof(token);
            }
            bands.push_back(monthly);
        }
        if (bands.size() != 36) {
            helios_runtime_error("ERROR (SolarPosition::getOzoneClimatology): Expected 36 latitude bands in " + path + ", found " + std::to_string(bands.size()) + ".");
        }
        return bands;
    }();
    return climatology;
}

float SolarPosition::getSolarFlux() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return Eb_PAR + Eb_NIR;
}

float SolarPosition::getSolarFluxPAR() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return Eb_PAR;
}

float SolarPosition::getSolarFluxNIR() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return Eb_NIR;
}

float SolarPosition::getDiffuseFraction() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, Eb_NIR, fdiff);
    }
    return fdiff;
}

float SolarPosition::getAmbientLongwaveFlux() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    // Model from Prata (1996) Q. J. R. Meteorol. Soc.
    float e0 = 611.f * exp(17.502f * (temperature - 273.f) / ((temperature - 273.f) + 240.9f)) * humidity; // Pascals
    float K = 0.465f; // cm-K/Pa
    float xi = e0 / temperature * K;
    float eps = 1.f - (1.f + xi) * exp(-sqrt(1.2f + 3.f * xi));

    return eps * 5.67e-8 * pow(temperature, 4);
}

float SolarPosition::diffuseFractionFromClearnessIndex(float clearness_index) {
    // Hourly diffuse-fraction correlation of Erbs et al. (1982), Solar Energy 28(4):293-302, Eq. 1.
    // The three branches very nearly meet at the breakpoints kT = 0.22 and kT = 0.80, where they
    // differ by about 0.0003; this small step is a property of the published fit, not an error here.
    float kt = fmin(fmax(clearness_index, 0.f), 1.f);

    if (kt <= 0.22f) {
        return 1.f - 0.09f * kt;
    } else if (kt <= 0.80f) {
        return 0.9511f - 0.1604f * kt + 4.388f * kt * kt - 16.638f * kt * kt * kt + 12.336f * kt * kt * kt * kt;
    }
    // Above kT = 0.80 Erbs et al. adopt a constant rather than fitting the sparse data there.
    return 0.165f;
}

float SolarPosition::getExtraterrestrialHorizontalFlux() const {
    float cos_zenith = cosf(getSunZenith());
    if (cos_zenith <= 0.f) {
        return 0.f;
    }
    // Solar constant (W/m^2), corrected for the Earth-Sun distance on this day of year.
    const float solar_constant = 1361.f;
    return solar_constant * calculateGeometricFactor(context->getJulianDate()) * cos_zenith;
}

void SolarPosition::applyCloudCalibration(float &Eb_PAR_Wm2, float &Eb_NIR_Wm2, float &fdiff_calc) const {

    // The measured flux is all-wave, so the calibration must be based on the all-wave (PAR+NIR)
    // clear-sky flux. Deriving the attenuation factor from a single band would make the band cancel
    // out of the ratio below, collapsing every band onto R_meas/cos(theta).
    float R_meas = context->queryTimeseriesData(cloudcalibrationlabel.c_str());
    if (R_meas < 0.f) {
        helios_runtime_error("ERROR (SolarPosition::applyCloudCalibration): Measured shortwave flux from timeseries '" + cloudcalibrationlabel + "' is negative (" + std::to_string(R_meas) + " W/m^2). Cloud calibration requires a non-negative measured flux; check the timeseries data for invalid or sentinel values.");
    }
    float R_calc_Wm2 = Eb_PAR_Wm2 + Eb_NIR_Wm2;
    float R_calc_horiz = R_calc_Wm2 * cosf(getSunZenith());

    // Estimate the diffuse fraction from the clearness index. The previous expression here,
    // 1-(R_meas-R_calc_horiz)/R_calc_horiz, reduces to 2-R_meas/R_calc_horiz, which exceeds 1 for
    // any measurement below the clear-sky prediction and therefore clamped to a fully-diffuse sky
    // under all real cloud, varying only when the measurement exceeded clear-sky irradiance.
    float extraterrestrial_horiz = getExtraterrestrialHorizontalFlux();
    float fdiff = 1.f;
    if (extraterrestrial_horiz > 1.f) {
        fdiff = diffuseFractionFromClearnessIndex(R_meas / extraterrestrial_horiz);
    }

    if (R_calc_horiz > 1.f) {
        // Scale both bands by the same broadband cloud attenuation factor. This preserves the
        // clear-sky PAR/NIR ratio and keeps Eb_PAR+Eb_NIR equal to the calibrated all-wave flux.
        float cloud_attenuation_factor = R_meas / R_calc_horiz;
        Eb_PAR_Wm2 *= cloud_attenuation_factor;
        Eb_NIR_Wm2 *= cloud_attenuation_factor;
        fdiff_calc = fdiff;
    }
}

// ===== SSolar-GOA Spectral Solar Radiation Model =====

void SolarPosition::SpectralData::loadFromFile(const std::string &path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromFile): Could not open the extraterrestrial solar spectrum file " + path + ".");
    }

    // The file is the unmodified National Laboratory of the Rockies distribution of the Wehrli (1985) spectrum: after a header, each line holds a wavelength (nm), the spectral irradiance (W/m^2/nm)
    // and the cumulative irradiance (W/m^2)
    std::vector<double> centers;
    std::vector<double> irradiances;
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream fields(line);
        double center, irradiance;
        if (!(fields >> center >> irradiance)) {
            if (centers.empty()) {
                continue; // header and blank lines before the data
            }
            if (line.find_first_not_of(" \t\r") == std::string::npos) {
                continue;
            }
            helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromFile): Could not parse the line \"" + line + "\" of " + path + ".");
        }
        if (irradiance < 0.0) {
            helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromFile): Negative irradiance at " + std::to_string(center) + " nm in " + path + ".");
        }
        if (!centers.empty() && center <= centers.back()) {
            helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromFile): Bin centers in " + path + " are not strictly increasing at " + std::to_string(center) + " nm.");
        }
        centers.push_back(center);
        irradiances.push_back(irradiance);
    }
    if (centers.size() < 2) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromFile): " + path + " holds fewer than two spectral bins.");
    }

    // Each source bin is taken to extend halfway to its neighbours' centers. The resulting piecewise-constant spectrum is averaged over 1 nm bins centered on 300, 301, ..., 2600 nm, the grid of the spectral
    // model and the gas transmittance tables, which conserves the irradiance in every interval.
    const size_t Nbins = centers.size();
    std::vector<double> edges(Nbins + 1);
    for (size_t i = 1; i < Nbins; ++i) {
        edges[i] = 0.5 * (centers[i - 1] + centers[i]);
    }
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0]);
    edges[Nbins] = centers[Nbins - 1] + 0.5 * (centers[Nbins - 1] - centers[Nbins - 2]);
    // Irradiance integrated from the first edge to each edge
    std::vector<double> cumulative(Nbins + 1, 0.0);
    for (size_t i = 0; i < Nbins; ++i) {
        cumulative[i + 1] = cumulative[i] + irradiances[i] * (edges[i + 1] - edges[i]);
    }
    auto cumulative_at = [&](double wavelength) {
        if (wavelength < edges.front() || wavelength > edges.back()) {
            helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromFile): The spectrum in " + path + " does not cover " + std::to_string(wavelength) + " nm; it must cover 299.5-2600.5 nm.");
        }
        const size_t upper = std::min(size_t(std::upper_bound(edges.begin(), edges.end(), wavelength) - edges.begin()), Nbins);
        const size_t lower = upper - 1;
        return cumulative[lower] + (cumulative[upper] - cumulative[lower]) * (wavelength - edges[lower]) / (edges[upper] - edges[lower]);
    };

    wavelengths_nm.clear();
    toa_irradiance.clear();
    for (int wavelength = 300; wavelength <= 2600; ++wavelength) {
        wavelengths_nm.push_back(float(wavelength));
        toa_irradiance.push_back(float(cumulative_at(wavelength + 0.5) - cumulative_at(wavelength - 0.5)));
    }
}

const SolarPosition::SpectralData &SolarPosition::getSSolarSpectralData() {
    static const SpectralData spectral_data = [] {
        SpectralData data;
        data.loadFromFile("plugins/solarposition/ssolar_goa/wehrli85.txt");
        return data;
    }();
    return spectral_data;
}

float SolarPosition::calculatePrecipitableWater(float temperature_K, float humidity_rel) {
    // Dew point (in Fahrenheit) from temperature and relative humidity with a Magnus-type formula, then precipitable water from the dew point (a relation attributed in the original implementation to Viswanadham 1981)
    if (!(humidity_rel > 0.f)) {
        helios_runtime_error("ERROR (SolarPosition::calculatePrecipitableWater): The relative humidity must be greater than zero to derive the precipitable water from the dew point, but it is " + std::to_string(humidity_rel) +
                             ". Set a positive relative humidity with SolarPosition::setAtmosphericConditions().");
    }
    float gamma = logf(humidity_rel) + 17.67f * (temperature_K - 273.0f) / 268.0f;
    float tdp = 243.0f * gamma / (17.67f - gamma) * 9.0f / 5.0f + 32.0f;
    return expf((0.1133f - logf(5.0f)) + 0.0393f * tdp);
}

float SolarPosition::calculateGeometricFactor(int julian_day) const {
    if (julian_day < 1 || julian_day > 366) {
        helios_runtime_error("ERROR (SolarPosition::calculateGeometricFactor): Day of the year must be between 1 and 366, got " + std::to_string(julian_day) + ".");
    }
    // Square of the ratio of the mean to the actual Earth-Sun distance: the Fourier series of Spencer (1971), Search 2(5):172, with the day angle 2*pi*d/365 where d = 0 on January 1 (stated maximum error 0.0001)
    const float day_angle = 2.f * float(M_PI) * float(julian_day - 1) / 365.f;
    return 1.000110f + 0.034221f * cosf(day_angle) + 0.001280f * sinf(day_angle) + 0.000719f * cosf(2.f * day_angle) + 0.000077f * sinf(2.f * day_angle);
}

float SolarPosition::calculateRayleighOpticalDepth(float wavelength_um, float pressure_ratio) {
    // Cachorro et al. (2022) Eq. 7, which the paper attributes to Gueymard (1995): tau_R = 1/(117.2594 L^4 - 1.3215 L^2 + 0.00032 - 0.000076 L^-2), L in um, at sea-level pressure, scaled by the ratio of the
    // surface pressure to the sea-level pressure
    const float wavelength_squared = wavelength_um * wavelength_um;
    return pressure_ratio / (117.2594f * wavelength_squared * wavelength_squared - 1.3215f * wavelength_squared + 0.00032f - 0.000076f / wavelength_squared);
}

float SolarPosition::calculateAerosolOpticalDepth(float wavelength_um, float alpha, float beta) {
    // Angstrom law, Cachorro et al. (2022) Eq. 8: tau_a = beta L^-alpha, L in um, so that beta is the optical depth at 1 um
    return beta * powf(wavelength_um, -alpha);
}

float SolarPosition::calculateRelativeAirMass(float solar_zenith_rad) {
    // Kasten & Young (1989), Appl. Opt. 28:4735, doi:10.1364/AO.28.004735: m = 1/[sin(h) + 0.50572 (h + 6.07995)^-1.6364] for solar elevation h in degrees, maximum relative error below 0.5%. Used in place of
    // the plane-parallel m = 1/cos(z) in the equations of Cachorro et al. (2022, p. 1693), which diverges as the sun approaches the horizon.
    const float elevation_deg = 90.f - solar_zenith_rad * 180.f / float(M_PI);
    if (elevation_deg < 0.f || elevation_deg > 90.f) {
        helios_runtime_error("ERROR (SolarPosition::calculateRelativeAirMass): Solar elevation must be between 0 and 90 degrees, got " + std::to_string(elevation_deg) + ".");
    }
    return 1.f / (sinf(elevation_deg * float(M_PI) / 180.f) + 0.50572f * powf(elevation_deg + 6.07995f, -1.6364f));
}

double SolarPosition::calculateExponentialIntegral(int order, double x) {
    if (order < 1 || x < 0.0 || (order == 1 && x == 0.0)) {
        helios_runtime_error("ERROR (SolarPosition::calculateExponentialIntegral): E_n(x) requires n >= 1 and x >= 0 (x > 0 for n = 1), got n = " + std::to_string(order) + ", x = " + std::to_string(x) + ".");
    }
    if (x == 0.0) {
        return 1.0 / double(order - 1); // Abramowitz & Stegun (1964) Eq. 5.1.23
    }
    double E1;
    if (x <= 1.0) {
        // Series of Abramowitz & Stegun (1964) Eq. 5.1.11: E1(x) = -gamma - ln(x) - sum_{k>=1} (-x)^k/(k k!)
        const double euler_gamma = 0.57721566490153286;
        double power_over_factorial = 1.0;
        double series = 0.0;
        for (int k = 1; k <= 30; ++k) {
            power_over_factorial *= -x / double(k);
            series += power_over_factorial / double(k);
        }
        E1 = -euler_gamma - std::log(x) - series;
    } else {
        // Even contraction of the continued fraction of Abramowitz & Stegun (1964) Eq. 5.1.22, E1(x) = exp(-x)/(x + 1 - 1/(x + 3 - 4/(x + 5 - ...))), evaluated from its 60th term upward
        double fraction = x + 121.0;
        for (int k = 60; k >= 1; --k) {
            fraction = x + double(2 * k - 1) - double(k * k) / fraction;
        }
        E1 = std::exp(-x) / fraction;
    }
    // Recurrence of Abramowitz & Stegun (1964) Eq. 5.1.14: E_{n+1}(x) = [exp(-x) - x E_n(x)]/n
    double En = E1;
    for (int n = 1; n < order; ++n) {
        En = (std::exp(-x) - x * En) / double(n);
    }
    return En;
}

double SolarPosition::calculateRayleighSphericalAlbedo(double rayleigh_optical_depth) {
    // Closed form of 6S subroutine CSALBR (6S User Guide, Version 3, Part 2, p. 96, Eq. 2): s = [3 tau - 4 E3(tau) + 6 E4(tau)]/(4 + 3 tau). It equals 1 - 2 int_0^1 mu T(mu) dmu for the Rayleigh transmittance
    // T(mu) = [(2/3 + mu) + (2/3 - mu) exp(-tau/mu)]/(4/3 + tau) of 6S subroutine SCATRA (same guide, p. 135, Eq. 1). Cachorro et al. (2022) print tau/(2 + tau) (1 - exp(-2 tau)) as Eq. 19, which is not
    // the spherical albedo: it grows as tau^2 for a thin layer instead of as tau, and is about a tenth of the 6S value at 550 nm.
    const double tau = rayleigh_optical_depth;
    return (3.0 * tau - 4.0 * calculateExponentialIntegral(3, tau) + 6.0 * calculateExponentialIntegral(4, tau)) / (4.0 + 3.0 * tau);
}

double SolarPosition::calculateTwoStreamDiffuseReflectance(double optical_depth, double single_scattering_albedo, double asymmetry) {
    // Reflectance of a layer over a black surface under diffuse illumination, in the two-stream approximation with the hemispheric closure: Heng & Kitzmann (2017), ApJS 232:20, Eq. 1,
    // R = zeta_- zeta_+ (1 - T^2)/(zeta_+^2 - zeta_-^2 T^2), with the coupling coefficients zeta_+- = [1 +- s]/2, s = sqrt((1 - w)/(1 - w g)), of Heng, Mendonca & Lee (2014), ApJS 215:4, Eq. 72, and the
    // transmission function for diffuse illumination T = 2 E3(tau sqrt((1 - w)(1 - w g))) of Heng & Kitzmann (2017, Sect. 2). Written in terms of s, R = (1 - s^2) q/(4 + (1 - s)^2 q) with q = (1 - T^2)/s,
    // whose limit q -> 4 (1 - g) tau as w -> 1 gives the conservative-scattering value (1 - g) tau/(1 + (1 - g) tau) without cancellation.
    const double absorbed_fraction = 1.0 - single_scattering_albedo;
    const double forward_factor = 1.0 - single_scattering_albedo * asymmetry;
    const double s = std::sqrt(absorbed_fraction / forward_factor);
    double q;
    if (s < 1e-8) {
        q = 4.0 * forward_factor * optical_depth;
    } else {
        const double T = 2.0 * calculateExponentialIntegral(3, optical_depth * s * forward_factor);
        q = (1.0 - T * T) / s;
    }
    return (1.0 - s * s) * q / (4.0 + (1.0 - s) * (1.0 - s) * q);
}

double SolarPosition::calculateMixedLayerTransmittance(double optical_depth, double single_scattering_albedo, double asymmetry, double air_mass) {
    // Total (direct + diffuse) scattering transmittance of a homogeneous Rayleigh-aerosol layer, Cachorro et al. (2022) Eq. 14: T = (1 - r0^2) exp(-k tau m)/(1 - r0^2 exp(-2 k tau m)), with Eq. 15
    // r0 = (k - 1 + w)/(k + 1 - w) and k = sqrt((1 - w)(1 - w g)). The printed Eq. 15 omits the square root, which gives a negative r0 for forward scattering; with it, r0 = (1 - s)/(1 + s),
    // s = sqrt((1 - w)/(1 - w g)), the semi-infinite reflectance zeta_-/zeta_+ of the two-stream solution (Heng et al. 2014, Eqs. 72 and 100). Written in terms of s,
    // T = 4 exp(-k x)/(4 + (1 - s)^2 q) with x = tau m and q = (1 - exp(-2 k x))/s, whose limit q -> 2 (1 - g) x as w -> 1 avoids cancellation.
    const double absorbed_fraction = 1.0 - single_scattering_albedo;
    const double forward_factor = 1.0 - single_scattering_albedo * asymmetry;
    const double s = std::sqrt(absorbed_fraction / forward_factor);
    const double k = s * forward_factor;
    const double path_optical_depth = optical_depth * air_mass;
    const double q = s < 1e-8 ? 2.0 * forward_factor * path_optical_depth : -std::expm1(-2.0 * k * path_optical_depth) / s;
    return 4.0 * std::exp(-k * path_optical_depth) / (4.0 + (1.0 - s) * (1.0 - s) * q);
}

void SolarPosition::calculateSpectralIrradianceComponents(std::vector<helios::vec2> &global_spectrum, std::vector<helios::vec2> &direct_spectrum, std::vector<helios::vec2> &diffuse_spectrum, float resolution_nm) const {
    if (resolution_nm < 1.0f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSpectralIrradianceComponents): resolution_nm must be >= 1 nm, got " + std::to_string(resolution_nm));
    }
    if (resolution_nm > 2300.0f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSpectralIrradianceComponents): resolution_nm must be <= 2300 nm, got " + std::to_string(resolution_nm));
    }

    float pressure_Pa, temperature_K, humidity_rel, turbidity_beta;
    getAtmosphericConditions(pressure_Pa, temperature_K, humidity_rel, turbidity_beta);
    const float water_vapor_cm = calculatePrecipitableWater(temperature_K, humidity_rel);
    const float ozone_atm_cm = getOzoneColumn() * 0.001f;
    const float ground_albedo = getGroundAlbedo();

    const float solar_zenith = getSunZenith();
    const float mu0 = cosf(solar_zenith);
    if (mu0 <= 0.f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSpectralIrradianceComponents): Cannot calculate spectral irradiance when the sun is at or below the horizon (zenith angle = " + std::to_string(solar_zenith * 180.f / float(M_PI)) +
                             " degrees).");
    }
    const float air_mass = calculateRelativeAirMass(solar_zenith);
    const float pressure_ratio = pressure_Pa / 101325.f;

    const SpectralData &spectral_data = getSSolarSpectralData();
    const std::vector<float> &wavelengths_nm = spectral_data.wavelengths_nm;
    const size_t Nwavelengths = wavelengths_nm.size();
    const float earth_sun_factor = calculateGeometricFactor(context->getJulianDate());

    // Gas absorption from the 6S gas transmittance tables (see generate_atmosphere_lut.py) along the solar path, applied to both the direct and the diffuse irradiance as in Cachorro et al. (2022, Sect. 2.2, p. 1692). The
    // water vapor and ozone paths are the columns times the air mass; the well-mixed gases (O2, CO2, CH4, N2O, CO) are tabulated against the air mass times the surface pressure ratio.
    const AtmosphereLUT &lut = getAtmosphereLUT();
    const float water_path = air_mass * water_vapor_cm;
    const AtmosphereTable &water_table = lut.getTable("water_transmittance");
    if (water_path > water_table.axis_values.at(1).back()) {
        helios_runtime_error("ERROR (SolarPosition::calculateSpectralIrradianceComponents): The water vapor path along the solar beam (" + std::to_string(water_vapor_cm) +
                             " cm of precipitable water, derived from the temperature and humidity set with setAtmosphericConditions(), times an air mass of " + std::to_string(air_mass) + ") is " + std::to_string(water_path) +
                             " g/cm^2, which exceeds the largest path in the gas transmittance table (" + std::to_string(water_table.axis_values.at(1).back()) + " g/cm^2). Use a higher solar elevation or a lower humidity.");
    }
    const std::vector<float> water_transmittance = interpolateGasTransmittance(water_table, "water_transmittance", water_path, wavelengths_nm);
    const std::vector<float> ozone_transmittance = interpolateGasTransmittance(lut.getTable("ozone_transmittance"), "ozone_transmittance", air_mass * ozone_atm_cm, wavelengths_nm);
    const std::vector<float> mixed_gas_transmittance = interpolateGasTransmittance(lut.getTable("mixed_gas_transmittance"), "mixed_gas_transmittance", air_mass * pressure_ratio, wavelengths_nm);

    global_spectrum.resize(Nwavelengths);
    direct_spectrum.resize(Nwavelengths);
    diffuse_spectrum.resize(Nwavelengths);
    for (size_t i = 0; i < Nwavelengths; ++i) {
        const float wavelength_um = wavelengths_nm[i] * 0.001f;
        const double extraterrestrial = double(spectral_data.toa_irradiance[i]) * earth_sun_factor;
        const double gas_transmittance = double(water_transmittance[i]) * ozone_transmittance[i] * mixed_gas_transmittance[i];

        // Optical properties of the single homogeneous layer of molecules and aerosol, Cachorro et al. (2022) Eq. 16 for the single-scattering albedo. The asymmetry parameter is weighted by scattering
        // optical depth, g = w_a tau_a g_a/(tau_R + w_a tau_a) with g = 0 for Rayleigh scattering, since the phase function of the mixture is the average of the components' phase functions weighted by their
        // scattering coefficients, and g is its first moment. The printed Eq. 16, g_a tau_a/tau_t, weights by extinction and omits w_a.
        const double tau_R = calculateRayleighOpticalDepth(wavelength_um, pressure_ratio);
        const double tau_a = calculateAerosolOpticalDepth(wavelength_um, ssolar_angstrom_alpha, turbidity_beta);
        const double tau_total = tau_R + tau_a;
        const double tau_scattering = tau_R + ssolar_aerosol_ssa * tau_a;
        const double single_scattering_albedo = tau_scattering / tau_total;
        const double asymmetry = ssolar_aerosol_ssa * tau_a * ssolar_aerosol_g / tau_scattering;

        // Direct beam: Beer-Lambert-Bouguer law, Cachorro et al. (2022) Eqs. 3-5
        const double direct_transmittance = std::exp(-tau_total * air_mass);
        // Direct plus diffuse irradiance over a black ground: Eq. 14
        const double total_transmittance = calculateMixedLayerTransmittance(tau_total, single_scattering_albedo, asymmetry, air_mass);

        // Spherical albedo of the atmosphere. Cachorro et al. (2022) Eq. 18 adds separate Rayleigh and aerosol albedos, which 6S shows overstates the albedo of the mixture at short wavelengths and high aerosol
        // load, since the aerosol shields part of the Rayleigh layer. Here the aerosol instead changes the two-stream reflectance of the whole layer: S = S_R + [R(tau_total, w, g) - R(tau_R, 1, 0)], with S_R
        // the 6S closed form. This combination is not a published formula; it keeps the exact pure-Rayleigh value and was validated against 6S (see compare_spectral_with_6s.py and SpectralIrradianceTheory).
        const double spherical_albedo = calculateRayleighSphericalAlbedo(tau_R) + calculateTwoStreamDiffuseReflectance(tau_total, single_scattering_albedo, asymmetry) - calculateTwoStreamDiffuseReflectance(tau_R, 1.0, 0.0);

        // Multiple reflection between the ground and the atmosphere amplifies the global irradiance by 1/(1 - albedo S), Cachorro et al. (2022) Eq. 17 (after Lenoble 1998); it does not affect the direct beam
        const double direct_normal = extraterrestrial * gas_transmittance * direct_transmittance;
        const double global_horizontal = mu0 * extraterrestrial * gas_transmittance * total_transmittance / (1.0 - ground_albedo * spherical_albedo);

        direct_spectrum[i] = make_vec2(wavelengths_nm[i], float(direct_normal));
        global_spectrum[i] = make_vec2(wavelengths_nm[i], float(global_horizontal));
        diffuse_spectrum[i] = make_vec2(wavelengths_nm[i], float(global_horizontal - mu0 * direct_normal));
    }

    if (cloudcalibrationlabel != "") {
        applySpectralCloudCalibration(global_spectrum, direct_spectrum, diffuse_spectrum, mu0);
    }

    if (resolution_nm > 1.0f + 1e-5f) {
        global_spectrum = downsampleSpectrum(global_spectrum, resolution_nm);
        direct_spectrum = downsampleSpectrum(direct_spectrum, resolution_nm);
        diffuse_spectrum = downsampleSpectrum(diffuse_spectrum, resolution_nm);
    }
}

std::vector<helios::vec2> SolarPosition::downsampleSpectrum(const std::vector<helios::vec2> &spectrum, float resolution_nm) {
    if (spectrum.size() < 2) {
        helios_runtime_error("ERROR (SolarPosition::downsampleSpectrum): The spectrum must have at least two points.");
    }
    // Output wavelengths start at the first native wavelength and step by resolution_nm up to the last; each takes the native point nearest to it
    const float first_nm = spectrum.front().x;
    const float native_spacing_nm = spectrum[1].x - spectrum[0].x;
    const size_t output_count = size_t(std::floor((spectrum.back().x - first_nm) / resolution_nm + 1e-4f)) + 1;
    std::vector<helios::vec2> downsampled;
    downsampled.reserve(output_count);
    for (size_t k = 0; k < output_count; ++k) {
        const float target_nm = first_nm + float(k) * resolution_nm;
        const size_t nearest = std::min(spectrum.size() - 1, size_t(std::lround((target_nm - first_nm) / native_spacing_nm)));
        downsampled.push_back(spectrum[nearest]);
    }
    return downsampled;
}

float SolarPosition::integrateSpectrum(const std::vector<helios::vec2> &spectrum) {
    if (spectrum.size() < 2) {
        return 0.f;
    }
    float integral = 0.f;
    for (size_t i = 1; i < spectrum.size(); ++i) {
        float delta_lambda = spectrum.at(i).x - spectrum.at(i - 1).x;
        integral += 0.5f * (spectrum.at(i).y + spectrum.at(i - 1).y) * delta_lambda;
    }
    return integral;
}

void SolarPosition::applySpectralCloudCalibration(std::vector<helios::vec2> &global_spectrum, std::vector<helios::vec2> &direct_spectrum, std::vector<helios::vec2> &diffuse_spectrum, float mu0) const {

    // Fraction of the extraterrestrial solar irradiance falling inside the 300-2600 nm window
    // covered by the SSolar-GOA tabulated spectrum. A broadband pyranometer sees essentially the
    // whole shortwave spectrum, so the measured flux is scaled by this fraction before it is
    // compared against the integral of the modeled spectrum.
    const float model_window_fraction = 0.9596f;

    // Below this relative disagreement between the modeled and measured broadband irradiance, no
    // cloud modification is applied. Bird, Riordan & Myers (1987) use the same 5% criterion: within
    // it the discrepancy is dominated by measurement uncertainty and by error in the turbidity and
    // precipitable-water inputs, so "correcting" for it would inject noise rather than cloud effects.
    const float no_correction_threshold = 0.05f;

    float R_meas_broadband = context->queryTimeseriesData(cloudcalibrationlabel.c_str());
    if (R_meas_broadband < 0.f) {
        helios_runtime_error("ERROR (SolarPosition::applySpectralCloudCalibration): Measured shortwave flux from timeseries '" + cloudcalibrationlabel + "' is negative (" + std::to_string(R_meas_broadband) + " W/m^2). Cloud calibration requires a non-negative measured flux; check the timeseries data for invalid or sentinel values.");
    }
    float R_meas_in_window = R_meas_broadband * model_window_fraction;

    float R_calc_in_window = integrateSpectrum(global_spectrum);

    if (R_calc_in_window <= 1.f) {
        return; // sun too low for the ratio to be meaningful
    }

    float measured_to_modeled_ratio = R_meas_in_window / R_calc_in_window;

    if (std::fabs(measured_to_modeled_ratio - 1.f) < no_correction_threshold) {
        return;
    }

    // ---- Step 1: repartition the direct and diffuse components ----
    // Cloud converts direct beam into diffuse, so the two components cannot simply be scaled by the
    // same factor: doing so would leave the diffuse fraction at its clear-sky value no matter how
    // thick the cloud, and the spectral modifiers below (which act on the diffuse component) would
    // have almost nothing to act on. Bird, Riordan & Myers (1987) avoid this by normalizing the
    // global and direct-normal spectra against two independent measurements, but only one broadband
    // measurement is available here, so the diffuse fraction is estimated from the clearness index.
    float extraterrestrial_horiz = getExtraterrestrialHorizontalFlux();
    if (extraterrestrial_horiz <= 1.f) {
        return;
    }
    float diffuse_fraction = diffuseFractionFromClearnessIndex(R_meas_broadband / extraterrestrial_horiz);

    float calibrated_global = R_meas_in_window;
    float calibrated_diffuse = calibrated_global * diffuse_fraction;
    float calibrated_direct_horizontal = calibrated_global - calibrated_diffuse;

    float clear_direct_horizontal = integrateSpectrum(direct_spectrum) * mu0;
    float clear_diffuse = integrateSpectrum(diffuse_spectrum);

    // Scale the direct beam to its calibrated total, preserving its clear-sky spectral shape.
    float direct_scale = (clear_direct_horizontal > 1e-6f) ? calibrated_direct_horizontal / clear_direct_horizontal : 0.f;
    for (helios::vec2 &sample: direct_spectrum) {
        sample.y *= direct_scale;
    }

    // The clear-sky diffuse component is Rayleigh-scattered skylight, which is far bluer than light
    // scattered by cloud: cloud droplets are large compared with the wavelength and scatter almost
    // neutrally. Simply rescaling the clear-sky diffuse shape would therefore drive the spectrum
    // toward pure Rayleigh blue as the sky becomes overcast, overstating the photosynthetically
    // active fraction. The diffuse shape is instead blended from the clear-sky skylight shape toward
    // the (near-neutral) beam shape in proportion to how much of the diffuse light originates from
    // cloud rather than from clear sky.
    //
    // Cloud-scattered light is not perfectly neutral either: it retains part of the blue enrichment
    // of skylight. This weight is the fraction of that enrichment retained, chosen so the modeled PAR
    // fraction of global irradiance under heavy overcast matches the measured value of 0.483 (versus
    // 0.417 under clear sky) reported for multi-site pyranometer/quantum-sensor records.
    const float cloud_skylight_blueness_retention = 0.375f;

    float clear_sky_diffuse_fraction = (R_calc_in_window > 1e-6f) ? clear_diffuse / R_calc_in_window : 0.f;
    float cloud_share_of_diffuse = 0.f;
    if (diffuse_fraction > clear_sky_diffuse_fraction && diffuse_fraction > 1e-6f) {
        cloud_share_of_diffuse = (diffuse_fraction - clear_sky_diffuse_fraction) / diffuse_fraction;
    }
    cloud_share_of_diffuse *= (1.f - cloud_skylight_blueness_retention);

    // direct_spectrum has already been scaled above, so this is the calibrated (not clear-sky) total.
    // Only the normalized shape is used below, and the scaling cancels out of that ratio.
    float direct_normal_total = integrateSpectrum(direct_spectrum);
    for (size_t i = 0; i < diffuse_spectrum.size(); ++i) {
        float skylight_shape = (clear_diffuse > 1e-6f) ? diffuse_spectrum.at(i).y / clear_diffuse : 0.f;
        float beam_shape = (direct_normal_total > 1e-6f) ? direct_spectrum.at(i).y / direct_normal_total : 0.f;
        float blended_shape = (1.f - cloud_share_of_diffuse) * skylight_shape + cloud_share_of_diffuse * beam_shape;
        diffuse_spectrum.at(i).y = blended_shape * calibrated_diffuse;
    }

    // ---- Step 2: empirical spectral modifiers (Bird, Riordan & Myers 1987, Eq. 4-3) ----
    // Clouds transmit short wavelengths preferentially, so a purely grey rescaling leaves the
    // cloudy-sky spectrum too red. Both modifiers act on the diffuse component, which is what Bird
    // et al. derived them for and which dominates the global irradiance under overcast skies.
    for (helios::vec2 &sample: diffuse_spectrum) {
        float wavelength_um = sample.x * 0.001f;

        // Blue/UV enhancement below 0.55 um: Diffuse * (lambda + 0.45)^-1. The form is continuous at
        // the 0.55 um cutoff (where it evaluates to exactly 1) and reaches its maximum of +1/3 at
        // 0.3 um, matching the increase reported by Bird et al.
        if (wavelength_um <= 0.55f) {
            sample.y /= (wavelength_um + 0.45f);
        }

        // Constant 7% increase from 0.50 to 0.926 um.
        if (wavelength_um >= 0.50f && wavelength_um <= 0.926f) {
            sample.y *= 1.07f;
        }
    }

    // ---- Step 3: renormalize so the calibrated spectrum still integrates to the measurement ----
    // The spectral modifiers add energy, which would otherwise break agreement with the measured
    // broadband flux that the whole calibration is anchored to.
    float diffuse_after_modifiers = integrateSpectrum(diffuse_spectrum);
    if (diffuse_after_modifiers > 1e-6f) {
        float renormalization = calibrated_diffuse / diffuse_after_modifiers;
        for (helios::vec2 &sample: diffuse_spectrum) {
            sample.y *= renormalization;
        }
    }

    // Rebuild the global spectrum from its components (direct beam is normal-incidence).
    for (size_t i = 0; i < global_spectrum.size(); ++i) {
        global_spectrum.at(i).y = direct_spectrum.at(i).y * mu0 + diffuse_spectrum.at(i).y;
    }
}

void SolarPosition::calculateDirectSolarSpectrum(const std::string &label, float resolution_nm) {
    std::vector<helios::vec2> global_spectrum, direct_spectrum, diffuse_spectrum;
    calculateSpectralIrradianceComponents(global_spectrum, direct_spectrum, diffuse_spectrum, resolution_nm);
    context->setGlobalData(label.c_str(), direct_spectrum);
}

void SolarPosition::calculateDiffuseSolarSpectrum(const std::string &label, float resolution_nm) {
    std::vector<helios::vec2> global_spectrum, direct_spectrum, diffuse_spectrum;
    calculateSpectralIrradianceComponents(global_spectrum, direct_spectrum, diffuse_spectrum, resolution_nm);
    context->setGlobalData(label.c_str(), diffuse_spectrum);
}

void SolarPosition::calculateGlobalSolarSpectrum(const std::string &label, float resolution_nm) {
    std::vector<helios::vec2> global_spectrum, direct_spectrum, diffuse_spectrum;
    calculateSpectralIrradianceComponents(global_spectrum, direct_spectrum, diffuse_spectrum, resolution_nm);
    context->setGlobalData(label.c_str(), global_spectrum);
}

void SolarPosition::calculateSensorAtmosphereSpectra(const std::string &label, const helios::vec3 &direction_to_sensor, float resolution_nm) {
    const float max_view_zenith_deg = 60.f;

    if (resolution_nm < 1.0f || resolution_nm > 2300.0f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSensorAtmosphereSpectra): resolution_nm must be between 1 and 2300 nm, got " + std::to_string(resolution_nm));
    }
    if (!cloudcalibrationlabel.empty()) {
        helios_runtime_error("ERROR (SolarPosition::calculateSensorAtmosphereSpectra): The sensor atmosphere model assumes clear sky and cannot be used with cloud calibration enabled. Call disableCloudCalibration() first.");
    }
    if (direction_to_sensor.magnitude() == 0.f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSensorAtmosphereSpectra): direction_to_sensor must be a non-zero vector.");
    }
    const vec3 view_direction = direction_to_sensor / direction_to_sensor.magnitude();
    const float mu_view = view_direction.z;
    const float view_zenith_deg = rad2deg(acos_safe(mu_view));
    if (view_zenith_deg > max_view_zenith_deg) {
        helios_runtime_error("ERROR (SolarPosition::calculateSensorAtmosphereSpectra): The view zenith angle of direction_to_sensor is " + std::to_string(view_zenith_deg) + " degrees, but must not exceed " +
                             std::to_string(max_view_zenith_deg) + " degrees.");
    }

    const vec3 sun_direction = getSunDirectionVector();
    const float mu_sun = cosf(getSunZenith());
    if (mu_sun <= 0.f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSensorAtmosphereSpectra): Cannot calculate the sensor atmosphere when the sun is below the horizon.");
    }
    const float solar_zenith_deg = rad2deg(getSunZenith());

    // Relative azimuth between the sun and the sensor: 0 when the sensor is on the same side as the sun (backscattering), 180 when opposite
    float relative_azimuth_deg = 0.f;
    const float sun_horizontal = sqrtf(sun_direction.x * sun_direction.x + sun_direction.y * sun_direction.y);
    const float view_horizontal = sqrtf(view_direction.x * view_direction.x + view_direction.y * view_direction.y);
    if (sun_horizontal > 1e-6f && view_horizontal > 1e-6f) {
        const float cos_relative_azimuth = (sun_direction.x * view_direction.x + sun_direction.y * view_direction.y) / (sun_horizontal * view_horizontal);
        relative_azimuth_deg = rad2deg(acos_safe(cos_relative_azimuth));
    }

    float pressure_Pa, temperature, humidity, turbidity_beta;
    getAtmosphericConditions(pressure_Pa, temperature, humidity, turbidity_beta);
    const float pressure_hPa = pressure_Pa * 0.01f;
    // Ångström law with the SSolar-GOA exponent: tau(lambda) = beta * lambda^-alpha, lambda in um
    const float aod550 = calculateAerosolOpticalDepth(0.55f, ssolar_angstrom_alpha, turbidity_beta);
    const float water_vapor_cm = calculatePrecipitableWater(temperature, humidity);
    const float ozone_atm_cm = getOzoneColumn() * 0.001f;
    const float ground_albedo = getGroundAlbedo();

    const SpectralData &spectral_data = getSSolarSpectralData();
    const std::vector<float> &wavelengths_nm = spectral_data.wavelengths_nm;
    const size_t Nwavelengths = wavelengths_nm.size();
    const float geo_factor = calculateGeometricFactor(context->getJulianDate());

    // Scattering quantities
    const AtmosphereLUT &lut = getAtmosphereLUT();
    const std::vector<float> path_reflectance = interpolateScatteringSpectrum(lut.getTable("path_reflectance"), "path_reflectance", {pressure_hPa, aod550, solar_zenith_deg, view_zenith_deg, relative_azimuth_deg}, wavelengths_nm);
    const std::vector<float> transmittance_down = interpolateScatteringSpectrum(lut.getTable("scattering_transmittance"), "scattering_transmittance", {pressure_hPa, aod550, solar_zenith_deg}, wavelengths_nm);
    const std::vector<float> transmittance_up = interpolateScatteringSpectrum(lut.getTable("scattering_transmittance"), "scattering_transmittance", {pressure_hPa, aod550, view_zenith_deg}, wavelengths_nm);
    const std::vector<float> optical_depth = interpolateScatteringSpectrum(lut.getTable("optical_depth"), "optical_depth", {pressure_hPa, aod550}, wavelengths_nm);
    const std::vector<float> spherical_albedo = interpolateScatteringSpectrum(lut.getTable("spherical_albedo"), "spherical_albedo", {pressure_hPa, aod550}, wavelengths_nm);

    // Gas transmittance along the downward solar path, the upward path and the two-way path. The band models are not multiplicative in air mass, so the surface signal is attenuated by the two-way transmittance,
    // and the upward factor applied to radiance already attenuated on the way down is the two-way transmittance divided by the downward one. The well-mixed gases scale with surface pressure.
    const float air_mass_sun = 1.f / mu_sun;
    const float air_mass_view = 1.f / mu_view;
    const float pressure_ratio = pressure_hPa / 1013.25f;
    auto gas_transmittance = [&](float air_mass, float water_fraction) {
        std::vector<float> water = interpolateGasTransmittance(lut.getTable("water_transmittance"), "water_transmittance", air_mass * water_fraction * water_vapor_cm, wavelengths_nm);
        std::vector<float> ozone = interpolateGasTransmittance(lut.getTable("ozone_transmittance"), "ozone_transmittance", air_mass * ozone_atm_cm, wavelengths_nm);
        std::vector<float> mixed = interpolateGasTransmittance(lut.getTable("mixed_gas_transmittance"), "mixed_gas_transmittance", air_mass * pressure_ratio, wavelengths_nm);
        for (size_t i = 0; i < water.size(); ++i) {
            water[i] *= ozone[i] * mixed[i];
        }
        return water;
    };
    const std::vector<float> gas_down = gas_transmittance(air_mass_sun, 1.f);
    const std::vector<float> gas_two_way = gas_transmittance(air_mass_sun + air_mass_view, 1.f);
    // Following 6S, radiance scattered by the atmosphere is attenuated by only half of the water vapor column, since the water vapor is concentrated near the ground
    const std::vector<float> gas_path = gas_transmittance(air_mass_sun + air_mass_view, 0.5f);

    std::vector<helios::vec2> path_radiance(Nwavelengths), adjacency_radiance(Nwavelengths), upward_direct_transmittance(Nwavelengths), upward_diffuse_transmittance(Nwavelengths), spherical_albedo_spectrum(Nwavelengths);
    std::vector<helios::vec2> direct_irradiance(Nwavelengths), diffuse_irradiance(Nwavelengths), global_irradiance(Nwavelengths);
    for (size_t i = 0; i < Nwavelengths; ++i) {
        const float wavelength_nm = wavelengths_nm[i];
        const float toa_irradiance = spectral_data.toa_irradiance[i] * geo_factor;

        // Ground illumination, including multiple reflection between the ground and the atmosphere
        const float direct_normal = toa_irradiance * gas_down[i] * expf(-optical_depth[i] / mu_sun);
        const float global_horizontal = mu_sun * toa_irradiance * gas_down[i] * transmittance_down[i] / (1.f - ground_albedo * spherical_albedo[i]);
        direct_irradiance[i] = make_vec2(wavelength_nm, direct_normal);
        global_irradiance[i] = make_vec2(wavelength_nm, global_horizontal);
        diffuse_irradiance[i] = make_vec2(wavelength_nm, global_horizontal - mu_sun * direct_normal);

        path_radiance[i] = make_vec2(wavelength_nm, path_reflectance[i] * mu_sun * toa_irradiance / float(M_PI) * gas_path[i]);

        // In saturated absorption bands both transmittances are zero; the surface signal there is zero whatever the upward transmittance, so it is set to zero
        const float gas_up = gas_down[i] > 0.f ? gas_two_way[i] / gas_down[i] : 0.f;
        const float direct_up = expf(-optical_depth[i] / mu_view);
        upward_direct_transmittance[i] = make_vec2(wavelength_nm, direct_up * gas_up);
        upward_diffuse_transmittance[i] = make_vec2(wavelength_nm, (transmittance_up[i] - direct_up) * gas_up);
        spherical_albedo_spectrum[i] = make_vec2(wavelength_nm, spherical_albedo[i]);

        // The surroundings, lit by the same global irradiance, leave radiance albedo*E_global/pi
        adjacency_radiance[i] = make_vec2(wavelength_nm, upward_diffuse_transmittance[i].y * ground_albedo * global_horizontal / float(M_PI));
    }

    std::vector<std::pair<std::string, std::vector<helios::vec2> *>> outputs = {{"_path_radiance", &path_radiance},
                                                                                {"_adjacency_radiance", &adjacency_radiance},
                                                                                {"_upward_direct_transmittance", &upward_direct_transmittance},
                                                                                {"_upward_diffuse_transmittance", &upward_diffuse_transmittance},
                                                                                {"_spherical_albedo", &spherical_albedo_spectrum},
                                                                                {"_direct_irradiance", &direct_irradiance},
                                                                                {"_diffuse_irradiance", &diffuse_irradiance},
                                                                                {"_global_irradiance", &global_irradiance}};
    for (auto &[suffix, spectrum]: outputs) {
        if (resolution_nm > 1.0f + 1e-5f) {
            *spectrum = downsampleSpectrum(*spectrum, resolution_nm);
        }
        context->setGlobalData((label + suffix).c_str(), *spectrum);
    }
    // Recorded so that a camera applying these spectra can check that it views the scene from the same direction
    context->setGlobalData((label + "_direction_to_sensor").c_str(), view_direction);
}

void SolarPosition::calculateSensorThermalAtmosphere(const std::string &label, const helios::vec3 &direction_to_sensor) {
    if (direction_to_sensor.magnitude() == 0.f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSensorThermalAtmosphere): direction_to_sensor must be a non-zero vector.");
    }
    const vec3 view_direction = direction_to_sensor / direction_to_sensor.magnitude();
    const float view_zenith_deg = rad2deg(acos_safe(view_direction.z));
    if (view_zenith_deg > thermal_max_view_zenith_deg) {
        helios_runtime_error("ERROR (SolarPosition::calculateSensorThermalAtmosphere): The view zenith angle of direction_to_sensor is " + std::to_string(view_zenith_deg) + " degrees, but must not exceed " +
                             std::to_string(thermal_max_view_zenith_deg) + " degrees.");
    }
    const ThermalAtmosphereState state = getThermalAtmosphereState("calculateSensorThermalAtmosphere");
    const float view_secant = std::min(1.f / std::cos(deg2rad(view_zenith_deg)), 1.f / std::cos(deg2rad(thermal_max_view_zenith_deg)));

    const AtmosphereLUT &lut = getThermalAtmosphereLUT();
    const float log_water_vapor = std::log(state.water_vapor_cm);
    const std::vector<float> reference_transmittance = lut.getTable("thermal_transmittance").interpolateLastAxis({log_water_vapor, state.climate_coordinate, state.pressure_hPa, view_secant}, "thermal_transmittance");
    const std::vector<float> reference_upwelling = lut.getTable("thermal_upwelling_radiance").interpolateLastAxis({log_water_vapor, state.climate_coordinate, state.pressure_hPa, view_secant}, "thermal_upwelling_radiance");
    // Ozone lies mostly above the water vapor, so a change in its column scales the radiance from below by a transmittance ratio and adds emission: L = R (tau B + L_up) + D
    // (the logarithm of the ratio, which is close to linear in the ozone column, is tabulated)
    const std::vector<float> ozone_log_ratio =
            lut.getTable("thermal_ozone_log_transmittance_ratio").interpolateLastAxis({state.climate_coordinate, state.pressure_hPa, state.ozone_DU, view_secant}, "thermal_ozone_log_transmittance_ratio");
    const std::vector<float> ozone_upwelling = lut.getTable("thermal_ozone_upwelling_radiance").interpolateLastAxis({state.climate_coordinate, state.pressure_hPa, state.ozone_DU, view_secant}, "thermal_ozone_upwelling_radiance");
    const std::vector<float> &wavelengths_nm = lut.getTable("thermal_transmittance").axis_values.back();

    std::vector<helios::vec2> transmittance(wavelengths_nm.size());
    std::vector<helios::vec2> upwelling_radiance(wavelengths_nm.size());
    for (size_t i = 0; i < wavelengths_nm.size(); i++) {
        const float ozone_ratio = std::exp(ozone_log_ratio.at(i));
        transmittance.at(i) = make_vec2(wavelengths_nm.at(i), std::clamp(ozone_ratio * reference_transmittance.at(i), 0.f, 1.f));
        upwelling_radiance.at(i) = make_vec2(wavelengths_nm.at(i), std::max(0.f, ozone_ratio * reference_upwelling.at(i) + ozone_upwelling.at(i)));
    }

    context->setGlobalData((label + "_thermal_transmittance").c_str(), transmittance);
    context->setGlobalData((label + "_thermal_upwelling_radiance").c_str(), upwelling_radiance);
    context->setGlobalData((label + "_thermal_downwelling_radiance").c_str(), calculateThermalDownwellingRadiance(state));
    context->setGlobalData((label + "_thermal_water_vapor_cm").c_str(), state.water_vapor_cm);
    context->setGlobalData((label + "_direction_to_sensor").c_str(), view_direction);
}

float SolarPosition::getThermalSkyFlux(float wavelength_min_nm, float wavelength_max_nm) const {
    if (wavelength_min_nm <= 0.f || wavelength_max_nm <= wavelength_min_nm) {
        helios_runtime_error("ERROR (SolarPosition::getThermalSkyFlux): Wavelength bounds must satisfy 0 < wavelength_min_nm < wavelength_max_nm, but bounds of (" + std::to_string(wavelength_min_nm) + ", " +
                             std::to_string(wavelength_max_nm) + ") nm were given.");
    }
    const std::vector<helios::vec2> downwelling_radiance = calculateThermalDownwellingRadiance(getThermalAtmosphereState("getThermalSkyFlux"));
    if (wavelength_min_nm < downwelling_radiance.front().x || wavelength_max_nm > downwelling_radiance.back().x) {
        helios_runtime_error("ERROR (SolarPosition::getThermalSkyFlux): The band " + std::to_string(wavelength_min_nm) + "-" + std::to_string(wavelength_max_nm) + " nm extends outside the " + std::to_string(downwelling_radiance.front().x) +
                             "-" + std::to_string(downwelling_radiance.back().x) + " nm range of the thermal atmosphere look-up table.");
    }

    // Trapezoidal integral of the spectrum, interpolated linearly to the band bounds
    double band_radiance = 0.0;
    for (size_t i = 0; i + 1 < downwelling_radiance.size(); i++) {
        const helios::vec2 &point0 = downwelling_radiance.at(i);
        const helios::vec2 &point1 = downwelling_radiance.at(i + 1);
        const float lower = std::max(point0.x, wavelength_min_nm);
        const float upper = std::min(point1.x, wavelength_max_nm);
        if (upper <= lower) {
            continue;
        }
        const float slope = (point1.y - point0.y) / (point1.x - point0.x);
        band_radiance += 0.5 * ((point0.y + slope * (lower - point0.x)) + (point0.y + slope * (upper - point0.x))) * (upper - lower);
    }
    // A Lambertian surface of emissivity epsilon reflects (1-epsilon) times the downwelling radiance, which is the sky irradiance divided by pi
    return float(M_PI * band_radiance);
}

SolarPosition::ThermalAtmosphereState SolarPosition::getThermalAtmosphereState(const std::string &calling_function) const {
    const AtmosphereLUT &lut = getThermalAtmosphereLUT();

    float pressure_Pa, temperature, humidity, turbidity_beta;
    getAtmosphericConditions(pressure_Pa, temperature, humidity, turbidity_beta);
    ThermalAtmosphereState state{};
    state.water_vapor_cm = calculatePrecipitableWater(temperature, humidity);
    state.pressure_hPa = pressure_Pa * 0.01f;
    state.ozone_DU = getOzoneColumn();

    // Check each condition against the table here, so that errors name the condition rather than a table axis
    auto check_range = [&](float value, const std::vector<float> &axis, const std::string &description, const std::string &units, float max_value) {
        const float min_value = std::min(axis.front(), axis.back());
        if (value < min_value * (1.f - 1e-5f) || value > max_value * (1.f + 1e-5f)) {
            helios_runtime_error("ERROR (SolarPosition::" + calling_function + "): The " + description + " of " + std::to_string(value) + " " + units + " is outside the " + std::to_string(min_value) + "-" + std::to_string(max_value) + " " + units +
                                 " range of the thermal atmosphere look-up table.");
        }
    };
    const AtmosphereTable &transmittance_table = lut.getTable("thermal_transmittance");
    const std::vector<float> &log_water_axis = transmittance_table.axis_values.at(0);
    const std::vector<float> water_axis = {std::exp(log_water_axis.front()), std::exp(log_water_axis.back())};
    check_range(state.water_vapor_cm, water_axis, "column water vapor (from the temperature and humidity set with setAtmosphericConditions())", "g/cm^2", water_axis.back());
    check_range(state.pressure_hPa, transmittance_table.axis_values.at(2), "surface pressure (set with setAtmosphericConditions())", "hPa", transmittance_table.axis_extrapolation_limits.at(2));
    const std::vector<float> &ozone_axis = lut.getTable("thermal_ozone_log_transmittance_ratio").axis_values.at(2);
    check_range(state.ozone_DU, ozone_axis, "ozone column", "DU", std::max(ozone_axis.front(), ozone_axis.back()));
    state.climate_coordinate = calculateThermalClimateCoordinate(lut.getTable("climate_temperature_K"), temperature, state.pressure_hPa, calling_function);
    return state;
}

float SolarPosition::calculateThermalClimateCoordinate(const AtmosphereTable &climate_temperature, float air_temperature_K, float pressure_hPa, const std::string &calling_function) {
    // The table gives the surface air temperature at each climate coordinate node for each pressure node; between pressure nodes the profiles are interpolated linearly in log pressure, and above the largest pressure
    // (sea level) the surface temperature is that at the largest pressure
    const std::vector<float> &pressures = climate_temperature.axis_values.at(0);
    const std::vector<float> &coordinates = climate_temperature.axis_values.at(1);
    size_t lower = 0;
    while (lower + 2 < pressures.size() && (pressures.at(0) > pressures.back() ? pressure_hPa < pressures.at(lower + 1) : pressure_hPa > pressures.at(lower + 1))) {
        ++lower;
    }
    const float pressure_weight = std::clamp((std::log(pressure_hPa) - std::log(pressures.at(lower))) / (std::log(pressures.at(lower + 1)) - std::log(pressures.at(lower))), 0.f, 1.f);
    std::vector<float> node_temperatures(coordinates.size());
    for (size_t c = 0; c < coordinates.size(); c++) {
        node_temperatures.at(c) = (1.f - pressure_weight) * climate_temperature.data.at(lower * coordinates.size() + c) + pressure_weight * climate_temperature.data.at((lower + 1) * coordinates.size() + c);
    }
    if (air_temperature_K < node_temperatures.front() || air_temperature_K > node_temperatures.back()) {
        helios_runtime_error("ERROR (SolarPosition::" + calling_function + "): The air temperature of " + std::to_string(air_temperature_K) + " K (set with setAtmosphericConditions()) is outside the " + std::to_string(node_temperatures.front()) + "-" +
                             std::to_string(node_temperatures.back()) + " K range of the thermal atmosphere look-up table at a surface pressure of " + std::to_string(pressure_hPa) + " hPa.");
    }
    // The surface air temperature is linear in the climate coordinate between nodes
    size_t node = 0;
    while (node + 2 < node_temperatures.size() && air_temperature_K > node_temperatures.at(node + 1)) {
        ++node;
    }
    const float weight = (air_temperature_K - node_temperatures.at(node)) / (node_temperatures.at(node + 1) - node_temperatures.at(node));
    return coordinates.at(node) + weight * (coordinates.at(node + 1) - coordinates.at(node));
}

std::vector<helios::vec2> SolarPosition::calculateThermalDownwellingRadiance(const ThermalAtmosphereState &state) {
    const AtmosphereTable &table = getThermalAtmosphereLUT().getTable("thermal_downwelling_radiance");
    const std::vector<float> radiance = table.interpolateLastAxis({std::log(state.water_vapor_cm), state.climate_coordinate, state.pressure_hPa, state.ozone_DU}, "thermal_downwelling_radiance");
    const std::vector<float> &wavelengths_nm = table.axis_values.back();
    std::vector<helios::vec2> spectrum(wavelengths_nm.size());
    for (size_t i = 0; i < wavelengths_nm.size(); i++) {
        spectrum.at(i) = make_vec2(wavelengths_nm.at(i), std::max(0.f, radiance.at(i)));
    }
    return spectrum;
}

const SolarPosition::AtmosphereLUT &SolarPosition::getAtmosphereLUT() {
    static const AtmosphereLUT lut = [] {
        AtmosphereLUT loaded;
        loaded.loadFromFile("plugins/solarposition/atmosphere_lut/atmosphere_lut.bin");
        return loaded;
    }();
    return lut;
}

const SolarPosition::AtmosphereLUT &SolarPosition::getThermalAtmosphereLUT() {
    static const AtmosphereLUT lut = [] {
        AtmosphereLUT loaded;
        loaded.loadFromFile("plugins/solarposition/thermal_atmosphere_lut/thermal_atmosphere_lut.bin");
        return loaded;
    }();
    return lut;
}

void SolarPosition::AtmosphereLUT::loadFromFile(const std::string &path) {
    // Surface pressure may be extrapolated this far above the largest tabulated pressure (sea level); Rayleigh scattering, the main pressure effect, is proportional to pressure
    const float max_extrapolated_pressure_hPa = 1050.f;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): Could not open atmospheric look-up table file " + path + ".");
    }
    auto read_uint = [&]() {
        uint32_t value;
        file.read(reinterpret_cast<char *>(&value), sizeof(value));
        if (!file) {
            helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): Unexpected end of file in " + path + ".");
        }
        return value;
    };
    auto read_string = [&]() {
        std::string text(read_uint(), '\0');
        file.read(text.data(), static_cast<std::streamsize>(text.size()));
        if (!file) {
            helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): Unexpected end of file in " + path + ".");
        }
        return text;
    };
    auto read_floats = [&](size_t count) {
        std::vector<float> values(count);
        file.read(reinterpret_cast<char *>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
        if (!file) {
            helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): Unexpected end of file in " + path + ".");
        }
        return values;
    };

    char magic[8];
    file.read(magic, 8);
    if (!file || std::string(magic, 8) != "HLIOSLUT") {
        helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): " + path + " is not an atmospheric look-up table file.");
    }
    // The file is little-endian; on a big-endian machine the version would not read as 1
    const uint32_t version = read_uint();
    if (version != 1) {
        helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): Unsupported look-up table format version " + std::to_string(version) + " in " + path + ".");
    }
    provenance = read_string();

    const uint32_t Ntables = read_uint();
    for (uint32_t t = 0; t < Ntables; ++t) {
        const std::string name = read_string();
        AtmosphereTable table;
        const uint32_t Naxes = read_uint();
        size_t Nvalues = 1;
        for (uint32_t a = 0; a < Naxes; ++a) {
            table.axis_names.push_back(read_string());
            table.axis_values.push_back(read_floats(read_uint()));
            const std::vector<float> &values = table.axis_values.back();
            if (values.empty()) {
                helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): Axis " + table.axis_names.back() + " of table " + name + " is empty.");
            }
            for (size_t i = 2; i < values.size(); ++i) {
                if ((values[i] - values[i - 1]) * (values[1] - values[0]) <= 0.f) {
                    helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::loadFromFile): Axis " + table.axis_names.back() + " of table " + name + " is not strictly monotonic.");
                }
            }
            const float axis_max = std::max(values.front(), values.back());
            table.axis_extrapolation_limits.push_back(table.axis_names.back() == "pressure_hPa" ? std::max(axis_max, max_extrapolated_pressure_hPa) : axis_max);
            Nvalues *= values.size();
        }
        table.data = read_floats(Nvalues);

        tables.emplace(name, std::move(table));
    }
}

const SolarPosition::AtmosphereTable &SolarPosition::AtmosphereLUT::getTable(const std::string &name) const {
    auto table = tables.find(name);
    if (table == tables.end()) {
        helios_runtime_error("ERROR (SolarPosition::AtmosphereLUT::getTable): The atmospheric look-up table has no table named " + name + ".");
    }
    return table->second;
}

void SolarPosition::AtmosphereTable::bracket(const std::vector<float> &point, const std::string &table_name, std::vector<size_t> &lower_index, std::vector<float> &upper_weight) const {
    lower_index.assign(point.size(), 0);
    upper_weight.assign(point.size(), 0.f);
    for (size_t a = 0; a < point.size(); ++a) {
        const std::vector<float> &values = axis_values[a];
        const float x = point[a];
        if (values.size() == 1) {
            if (x != values.front()) {
                helios_runtime_error("ERROR (SolarPosition::AtmosphereTable::interpolate): Axis " + axis_names[a] + " of table " + table_name + " has the single value " + std::to_string(values.front()) + ", but " + std::to_string(x) +
                                     " was requested.");
            }
            continue;
        }
        const bool increasing = values.back() > values.front();
        const float axis_min = std::min(values.front(), values.back());
        const float axis_max = std::max(values.front(), values.back());
        // Tolerate round-off in values computed from angles
        const float tolerance = 1e-4f * (axis_max - axis_min);
        if (x < axis_min - tolerance || x > axis_extrapolation_limits[a] + tolerance) {
            helios_runtime_error("ERROR (SolarPosition::AtmosphereTable::interpolate): " + axis_names[a] + " = " + std::to_string(x) + " is outside the range of the atmospheric look-up table (" + std::to_string(axis_min) + " to " +
                                 std::to_string(axis_extrapolation_limits[a]) + ") for table " + table_name + ".");
        }
        const float x_clamped = std::clamp(x, axis_min, axis_extrapolation_limits[a]);
        size_t i = 0;
        while (i + 2 < values.size() && (increasing ? x_clamped > values[i + 1] : x_clamped < values[i + 1])) {
            ++i;
        }
        lower_index[a] = i;
        float weight = (x_clamped - values[i]) / (values[i + 1] - values[i]);
        // Inside the tabulated range the weight lies in [0,1] up to round-off; beyond the largest value (permitted extrapolation) it falls outside, extrapolating the end interval linearly
        if (x_clamped <= axis_max) {
            weight = std::clamp(weight, 0.f, 1.f);
        }
        upper_weight[a] = weight;
    }
}

float SolarPosition::AtmosphereTable::interpolate(const std::vector<float> &point, const std::string &table_name) const {
    const size_t Naxes = axis_values.size();
    if (point.size() != Naxes) {
        helios_runtime_error("ERROR (SolarPosition::AtmosphereTable::interpolate): Table " + table_name + " has " + std::to_string(Naxes) + " axes, but a point with " + std::to_string(point.size()) + " coordinates was given.");
    }
    // For each axis, the index of the lower corner of the bracketing interval and the weight of the upper corner
    std::vector<size_t> lower_index;
    std::vector<float> upper_weight;
    bracket(point, table_name, lower_index, upper_weight);

    std::vector<size_t> stride(Naxes, 1);
    for (size_t a = Naxes - 1; a > 0; --a) {
        stride[a - 1] = stride[a] * axis_values[a].size();
    }

    double result = 0.0;
    for (size_t corner = 0; corner < (size_t(1) << Naxes); ++corner) {
        double weight = 1.0;
        size_t offset = 0;
        for (size_t a = 0; a < Naxes; ++a) {
            const bool upper = (corner >> a) & 1;
            if (upper && axis_values[a].size() == 1) {
                weight = 0.0;
                break;
            }
            weight *= upper ? upper_weight[a] : 1.0 - upper_weight[a];
            offset += (lower_index[a] + (upper ? 1 : 0)) * stride[a];
        }
        if (weight != 0.0) {
            result += weight * data[offset];
        }
    }
    return static_cast<float>(result);
}

std::vector<float> SolarPosition::AtmosphereTable::interpolateLastAxis(const std::vector<float> &point, const std::string &table_name) const {
    const size_t Naxes = axis_values.size();
    if (point.size() + 1 != Naxes) {
        helios_runtime_error("ERROR (SolarPosition::AtmosphereTable::interpolateLastAxis): Table " + table_name + " has " + std::to_string(Naxes) + " axes, but a point with " + std::to_string(point.size()) +
                             " coordinates was given (one fewer than the number of axes is required).");
    }
    std::vector<size_t> lower_index;
    std::vector<float> upper_weight;
    bracket(point, table_name, lower_index, upper_weight);

    const size_t Nlast = axis_values.back().size();
    std::vector<size_t> stride(Naxes, 1);
    for (size_t a = Naxes - 1; a > 0; --a) {
        stride[a - 1] = stride[a] * axis_values[a].size();
    }

    std::vector<double> result(Nlast, 0.0);
    for (size_t corner = 0; corner < (size_t(1) << point.size()); ++corner) {
        double weight = 1.0;
        size_t offset = 0;
        for (size_t a = 0; a < point.size(); ++a) {
            const bool upper = (corner >> a) & 1;
            if (upper && axis_values[a].size() == 1) {
                weight = 0.0;
                break;
            }
            weight *= upper ? upper_weight[a] : 1.0 - upper_weight[a];
            offset += (lower_index[a] + (upper ? 1 : 0)) * stride[a];
        }
        if (weight != 0.0) {
            for (size_t i = 0; i < Nlast; ++i) {
                result[i] += weight * data[offset + i];
            }
        }
    }
    return std::vector<float>(result.begin(), result.end());
}

std::vector<float> SolarPosition::interpolateScatteringSpectrum(const AtmosphereTable &table, const std::string &table_name, const std::vector<float> &point, const std::vector<float> &wavelengths_nm) {
    const std::vector<float> &node_wavelengths = table.axis_values.front();
    std::vector<float> log_node_wavelengths(node_wavelengths.size());
    std::vector<float> log_node_values(node_wavelengths.size());
    for (size_t n = 0; n < node_wavelengths.size(); ++n) {
        std::vector<float> node_point = {node_wavelengths[n]};
        node_point.insert(node_point.end(), point.begin(), point.end());
        const float value = table.interpolate(node_point, table_name);
        if (value <= 0.f) {
            helios_runtime_error("ERROR (SolarPosition::interpolateScatteringSpectrum): Table " + table_name + " has a non-positive value at " + std::to_string(node_wavelengths[n]) + " nm, which cannot be interpolated in log space.");
        }
        log_node_wavelengths[n] = logf(node_wavelengths[n]);
        log_node_values[n] = logf(value);
    }

    std::vector<float> spectrum(wavelengths_nm.size());
    for (size_t i = 0; i < wavelengths_nm.size(); ++i) {
        const float log_wavelength = logf(wavelengths_nm[i]);
        if (log_wavelength < log_node_wavelengths.front() - 1e-6f || log_wavelength > log_node_wavelengths.back() + 1e-6f) {
            helios_runtime_error("ERROR (SolarPosition::interpolateScatteringSpectrum): Wavelength " + std::to_string(wavelengths_nm[i]) + " nm is outside the range of table " + table_name + ".");
        }
        size_t n = 0;
        while (n + 2 < log_node_wavelengths.size() && log_wavelength > log_node_wavelengths[n + 1]) {
            ++n;
        }
        const float weight = std::clamp((log_wavelength - log_node_wavelengths[n]) / (log_node_wavelengths[n + 1] - log_node_wavelengths[n]), 0.f, 1.f);
        spectrum[i] = expf(log_node_values[n] + weight * (log_node_values[n + 1] - log_node_values[n]));
    }
    return spectrum;
}

std::vector<float> SolarPosition::interpolateGasTransmittance(const AtmosphereTable &table, const std::string &table_name, float absorber_amount, const std::vector<float> &wavelengths_nm) {
    if (absorber_amount < 0.f) {
        helios_runtime_error("ERROR (SolarPosition::interpolateGasTransmittance): Negative absorber amount for table " + table_name + ".");
    }
    const std::vector<float> &amounts = table.axis_values.at(1);
    if (absorber_amount > amounts.back() * (1.f + 1e-4f)) {
        helios_runtime_error("ERROR (SolarPosition::interpolateGasTransmittance): " + table.axis_names.at(1) + " = " + std::to_string(absorber_amount) + " exceeds the largest value in table " + table_name + " (" + std::to_string(amounts.back()) + ").");
    }

    // Transmittance varies smoothly with the logarithm of the absorber amount, so the amount is interpolated in log space. Below the smallest tabulated amount it is interpolated linearly toward a
    // transmittance of one at zero amount, since no absorber transmits everything.
    const bool below_table = absorber_amount < amounts.front();
    size_t j = 0;
    float weight = 0.f;
    if (!below_table) {
        while (j + 2 < amounts.size() && absorber_amount > amounts[j + 1]) {
            ++j;
        }
        weight = std::clamp((logf(absorber_amount) - logf(amounts[j])) / (logf(amounts[j + 1]) - logf(amounts[j])), 0.f, 1.f);
    }

    std::vector<float> spectrum(wavelengths_nm.size());
    for (size_t i = 0; i < wavelengths_nm.size(); ++i) {
        if (below_table) {
            const float transmittance_at_smallest = table.interpolate({wavelengths_nm[i], amounts.front()}, table_name);
            spectrum[i] = 1.f + (absorber_amount / amounts.front()) * (transmittance_at_smallest - 1.f);
        } else {
            const float lower = table.interpolate({wavelengths_nm[i], amounts[j]}, table_name);
            const float upper = table.interpolate({wavelengths_nm[i], amounts[j + 1]}, table_name);
            spectrum[i] = lower + weight * (upper - lower);
        }
    }
    return spectrum;
}

// ===== Prague Sky Model Implementation =====

void SolarPosition::enablePragueSkyModel() {
    if (prague_enabled) {
        std::cerr << "WARNING (SolarPosition::enablePragueSkyModel): Prague model already enabled." << std::endl;
        return;
    }

    prague_model = std::make_unique<helios::PragueSkyModelInterface>();

    // Always use the reduced dataset
    std::string data_path = "plugins/solarposition/lib/prague_sky_model/PragueSkyModelReduced.dat";

    try {
        prague_model->initialize(data_path);
        prague_enabled = true;
    } catch (const std::exception &e) {
        helios_runtime_error("ERROR (SolarPosition::enablePragueSkyModel): Failed to initialize Prague Sky Model: " + std::string(e.what()));
    }
}

bool SolarPosition::isPragueSkyModelEnabled() const {
    return prague_enabled;
}

void SolarPosition::updatePragueSkyModel(float ground_albedo) {
    if (!prague_enabled) {
        helios_runtime_error("ERROR (SolarPosition::updatePragueSkyModel): Prague model not enabled. Call enablePragueSkyModel() first.");
    }

    // Read turbidity from Context atmospheric conditions
    float turbidity = 0.02f; // Default: clear sky
    if (context->doesGlobalDataExist("atmosphere_turbidity")) {
        context->getGlobalData("atmosphere_turbidity", turbidity);
    }

    helios::vec3 sun_dir = getSunDirectionVector();
    float visibility_km = helios::PragueSkyModelInterface::turbidityToVisibility(turbidity);

    // Mark data as invalid during computation
    context->setGlobalData("prague_sky_valid", 0);

    // Wavelength grid: 360-1480 nm at 5nm spacing
    const float lambda_min = 360.0f;
    const float lambda_max = 1480.0f;
    const float lambda_step = 5.0f;
    const int n_wavelengths = 225;
    const int params_per_wavelength = 6;

    // Pre-compute sampling directions ONCE (5-10% speedup)
    helios::vec3 zenith_dir(0.0f, 0.0f, 1.0f);

    helios::vec3 sun_horizontal = normalize(make_vec3(sun_dir.x, sun_dir.y, 0.0f));
    helios::vec3 horizon_dir;
    if (sun_horizontal.magnitude() > 0.01f) {
        horizon_dir = normalize(make_vec3(-sun_horizontal.y, sun_horizontal.x, 0.0f));
    } else {
        horizon_dir = make_vec3(1.0f, 0.0f, 0.0f);
    }

    helios::vec3 near_sun_dir = rotateDirectionTowardZenith(sun_dir, 5.0f * float(M_PI) / 180.0f);
    helios::vec3 mid_sun_dir = rotateDirectionTowardZenith(sun_dir, 15.0f * float(M_PI) / 180.0f);
    helios::vec3 away_dir = getDirectionAwayFromSun(sun_dir, 45.0f);

    // Allocate output: [λ, L_zen, circ_str, circ_width, horiz_bright, norm] × 225
    std::vector<float> spectral_params(n_wavelengths * params_per_wavelength);

// CRITICAL: OpenMP parallelization over wavelengths
// Expected speedup: 6-8× on 8-core CPU
#pragma omp parallel for schedule(dynamic, 8)
    for (int i = 0; i < n_wavelengths; ++i) {
        float wavelength = lambda_min + i * lambda_step;

        float L_zenith, circ_str, circ_width, horiz_bright, normalization;
        fitAngularParametersAtWavelength(wavelength, visibility_km, ground_albedo, sun_dir, L_zenith, circ_str, circ_width, horiz_bright, normalization);

        int base_idx = i * params_per_wavelength;
        spectral_params[base_idx + 0] = wavelength;
        spectral_params[base_idx + 1] = L_zenith;
        spectral_params[base_idx + 2] = circ_str;
        spectral_params[base_idx + 3] = circ_width;
        spectral_params[base_idx + 4] = horiz_bright;
        spectral_params[base_idx + 5] = normalization;
    }

    // Store in Context
    context->setGlobalData("prague_sky_spectral_params", spectral_params);
    context->setGlobalData("prague_sky_sun_direction", sun_dir);
    context->setGlobalData("prague_sky_visibility_km", visibility_km);
    context->setGlobalData("prague_sky_ground_albedo", ground_albedo);
    context->setGlobalData("prague_sky_valid", 1);

    // Update cache for lazy evaluation
    cached_sun_direction = sun_dir;
    cached_turbidity = turbidity;
    cached_albedo = ground_albedo;
}

bool SolarPosition::pragueSkyModelNeedsUpdate(float ground_albedo, float sun_tolerance, float turbidity_tolerance, float albedo_tolerance) const {
    if (!prague_enabled) {
        return false;
    }

    // Check if this is the first update
    if (cached_turbidity < 0.0f) {
        return true;
    }

    // Read current turbidity from Context
    float turbidity = 0.02f; // Default: clear sky
    if (context->doesGlobalDataExist("atmosphere_turbidity")) {
        context->getGlobalData("atmosphere_turbidity", turbidity);
    }

    // Check sun direction change
    helios::vec3 sun_dir = getSunDirectionVector();
    float sun_change = (sun_dir - cached_sun_direction).magnitude();
    if (sun_change > sun_tolerance) {
        return true;
    }

    // Check turbidity change (relative)
    float turb_change = std::abs(turbidity - cached_turbidity) / std::max(cached_turbidity, 0.01f);
    if (turb_change > turbidity_tolerance) {
        return true;
    }

    // Check albedo change (absolute)
    if (std::abs(ground_albedo - cached_albedo) > albedo_tolerance) {
        return true;
    }

    return false;
}

void SolarPosition::fitAngularParametersAtWavelength(float wavelength, float visibility_km, float albedo, const helios::vec3 &sun_dir, float &L_zenith, float &circ_str, float &circ_width, float &horiz_bright, float &normalization) {

    // Recompute directions (needed inside OpenMP parallel region)
    helios::vec3 zenith_dir(0.0f, 0.0f, 1.0f);

    helios::vec3 sun_horizontal = normalize(make_vec3(sun_dir.x, sun_dir.y, 0.0f));
    helios::vec3 horizon_dir;
    if (sun_horizontal.magnitude() > 0.01f) {
        horizon_dir = normalize(make_vec3(-sun_horizontal.y, sun_horizontal.x, 0.0f));
    } else {
        horizon_dir = make_vec3(1.0f, 0.0f, 0.0f);
    }

    helios::vec3 near_sun_dir = rotateDirectionTowardZenith(sun_dir, 5.0f * float(M_PI) / 180.0f);
    helios::vec3 mid_sun_dir = rotateDirectionTowardZenith(sun_dir, 15.0f * float(M_PI) / 180.0f);

    // Sample Prague model at 5 key directions
    L_zenith = prague_model->getSkyRadiance(zenith_dir, sun_dir, wavelength, visibility_km, albedo);

    float L_horizon = prague_model->getSkyRadiance(horizon_dir, sun_dir, wavelength, visibility_km, albedo);

    float L_near_sun = prague_model->getSkyRadiance(near_sun_dir, sun_dir, wavelength, visibility_km, albedo);

    float L_mid_sun = prague_model->getSkyRadiance(mid_sun_dir, sun_dir, wavelength, visibility_km, albedo);

    // Fit horizon brightness
    horiz_bright = std::max(1.0f, L_horizon / std::max(L_zenith, 1e-10f));

    // Fit circumsolar parameters using two sample points
    float cos_theta_near = std::max(0.0f, near_sun_dir.z);
    float cos_theta_mid = std::max(0.0f, mid_sun_dir.z);
    float h1 = 1.0f + (horiz_bright - 1.0f) * (1.0f - cos_theta_near);
    float h2 = 1.0f + (horiz_bright - 1.0f) * (1.0f - cos_theta_mid);

    circ_width = fitCircumsolarWidth(L_near_sun, L_mid_sun, h1, h2, 5.0f, 15.0f);
    circ_width = std::clamp(circ_width, 5.0f, 60.0f);

    float exp5 = std::exp(-5.0f / circ_width);
    float exp15 = std::exp(-15.0f / circ_width);

    float ratio_near = L_near_sun / std::max(L_zenith * h1, 1e-10f);
    float ratio_mid = L_mid_sun / std::max(L_zenith * h2, 1e-10f);

    float A_from_near = (ratio_near - 1.0f) / std::max(exp5, 1e-10f);
    float A_from_mid = (ratio_mid - 1.0f) / std::max(exp15, 1e-10f);
    circ_str = std::max(0.0f, 0.5f * (A_from_near + A_from_mid));
    circ_str = std::min(circ_str, 20.0f);

    // Compute normalization
    normalization = computeAngularNormalization(circ_str, circ_width, horiz_bright);
}

float SolarPosition::fitCircumsolarWidth(float L1, float L2, float h1, float h2, float gamma1, float gamma2) const {
    float best_B = 15.0f; // Default
    float best_error = 1e10f;

    float target_ratio = (L1 * h2) / std::max(L2 * h1, 1e-10f);

    // Grid search over plausible width range
    for (float B = 5.0f; B <= 50.0f; B += 1.0f) {
        float e1 = std::exp(-gamma1 / B);
        float e2 = std::exp(-gamma2 / B);

        // For each B, find A that minimizes error
        float denom = e1 - target_ratio * e2;
        if (std::abs(denom) < 1e-10f)
            continue;

        float A = (target_ratio - 1.0f) / denom;
        if (A < 0)
            continue; // Invalid solution

        float predicted_ratio = (1.0f + A * e1) / (1.0f + A * e2);
        float error = std::abs(predicted_ratio - target_ratio);

        if (error < best_error) {
            best_error = error;
            best_B = B;
        }
    }

    return best_B;
}

float SolarPosition::computeAngularNormalization(float circ_str, float circ_width, float horiz_bright) const {
    // Numerical integration of angular pattern over hemisphere
    const int N = 50;
    float integral = 0.0f;

    for (int j = 0; j < N; ++j) {
        for (int i = 0; i < N; ++i) {
            float theta = 0.5f * float(M_PI) * (i + 0.5f) / N; // 0 to π/2
            float phi = 2.0f * float(M_PI) * (j + 0.5f) / N; // 0 to 2π

            helios::vec3 dir = sphere2cart(make_SphericalCoord(0.5f * float(M_PI) - theta, phi));

            // Compute angular pattern (without L_zenith, which factors out)
            float cos_theta = std::max(0.0f, dir.z);
            float horizon_term = 1.0f + (horiz_bright - 1.0f) * (1.0f - cos_theta);

            // For normalization, average circumsolar contribution
            float circ_term = 1.0f; // Approximation: circumsolar averages out over azimuth

            float pattern = circ_term * horizon_term;

            // Solid angle element: sin(θ) dθ dφ
            integral += pattern * std::cos(theta) * std::sin(theta) * (float(M_PI) / (2.0f * N)) * (2.0f * float(M_PI) / N);
        }
    }

    return 1.0f / std::max(integral, 1e-10f);
}

helios::vec3 SolarPosition::rotateDirectionTowardZenith(const helios::vec3 &dir, float angle_rad) const {
    helios::vec3 zenith(0, 0, 1);
    helios::vec3 axis = cross(dir, zenith);

    if (axis.magnitude() < 0.01f) {
        // dir is nearly parallel to zenith, use arbitrary perpendicular
        axis = make_vec3(1, 0, 0);
    }

    axis = normalize(axis);

    // Rodrigues rotation formula
    float c = std::cos(angle_rad);
    float s = std::sin(angle_rad);

    helios::vec3 rotated = dir * c + cross(axis, dir) * s + axis * (axis.x * dir.x + axis.y * dir.y + axis.z * dir.z) * (1 - c);
    return normalize(rotated);
}

helios::vec3 SolarPosition::getDirectionAwayFromSun(const helios::vec3 &sun_dir, float zenith_angle_deg) const {
    float theta = zenith_angle_deg * float(M_PI) / 180.0f;

    // Find azimuth opposite to sun
    float sun_azimuth = std::atan2(sun_dir.y, sun_dir.x);
    float opposite_azimuth = sun_azimuth + float(M_PI);

    return make_vec3(std::sin(theta) * std::cos(opposite_azimuth), std::sin(theta) * std::sin(opposite_azimuth), std::cos(theta));
}
