/** \file "SolarPosition.cpp" Primary source file for solar position model plug-in.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SolarPosition.h"
#include "PragueSkyModelInterface.h"

using namespace std;
using namespace helios;

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

    int solstice_day, LSTM;
    float Gamma, delta, time_dec, B, EoT, TC, LST, h, theta, phi, rad;

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
    float Eb = Eb_PAR + Eb_NIR;
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb, fdiff);
    }
    return Eb;
}

float SolarPosition::getSolarFluxPAR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const {
    // Deprecated method - kept for backward compatibility
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, fdiff);
    }
    return Eb_PAR;
}

float SolarPosition::getSolarFluxNIR(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const {
    // Deprecated method - kept for backward compatibility
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_NIR, fdiff);
    }
    return Eb_NIR;
}

float SolarPosition::getDiffuseFraction(float pressure_Pa, float temperature_K, float humidity_rel, float turbidity) const {
    // Deprecated method - kept for backward compatibility
    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure_Pa, temperature_K, humidity_rel, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, fdiff);
    }
    return fdiff;
}

void SolarPosition::GueymardSolarModel(float pressure, float temperature, float humidity, float turbidity, float &Eb_PAR, float &Eb_NIR, float &fdiff) const {

    float beta = turbidity;

    uint DOY = context->getJulianDate();

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
    float uo = (235 + (150 + 40 * sin(0.9856 * (DOY - 30) * M_PI / 180.f) + 20 * sin(3 * (longitude * M_PI / 180.f + 20))) * pow(sin(1.28 * latitude * M_PI / 180.f), 2)) * 0.001f; // O3 atm-cm
    // NOTE: uo model from van Heuklon (1979)
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

float SolarPosition::getSolarFlux() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    float Eb = Eb_PAR + Eb_NIR;
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb, fdiff);
    }
    return Eb;
}

float SolarPosition::getSolarFluxPAR() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, fdiff);
    }
    return Eb_PAR;
}

float SolarPosition::getSolarFluxNIR() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_NIR, fdiff);
    }
    return Eb_NIR;
}

float SolarPosition::getDiffuseFraction() const {
    float pressure, temperature, humidity, turbidity;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity);

    float Eb_PAR, Eb_NIR, fdiff;
    GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    if (!cloudcalibrationlabel.empty()) {
        applyCloudCalibration(Eb_PAR, fdiff);
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

void SolarPosition::applyCloudCalibration(float &R_calc_Wm2, float &fdiff_calc) const {

    assert(context->doesTimeseriesVariableExist(cloudcalibrationlabel.c_str()));

    float R_meas = context->queryTimeseriesData(cloudcalibrationlabel.c_str());
    float R_calc_horiz = R_calc_Wm2 * cosf(getSunZenith());

    float fdiff = fmin(fmax(0, 1.f - (R_meas - R_calc_horiz) / (R_calc_horiz)), 1);
    float R = R_calc_Wm2 * R_meas / R_calc_horiz;

    if (fdiff > 0.001 && R_calc_horiz > 1.f) {
        R_calc_Wm2 = R;
        fdiff_calc = fdiff;
    }
}

// ===== SSolar-GOA Spectral Solar Radiation Model =====

void SolarPosition::SpectralData::loadFromDirectory(const std::string &data_path) {
    // Load wehrli.dat (TOA spectrum)
    std::string wehrli_file = data_path + "/wehrli.dat";
    std::ifstream wehrli_stream(wehrli_file);
    if (!wehrli_stream.is_open()) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromDirectory): Could not open file " + wehrli_file);
    }

    wavelengths_nm.clear();
    toa_irradiance.clear();

    std::string line;
    while (std::getline(wehrli_stream, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        float wavelength, irradiance;
        if (iss >> wavelength >> irradiance) {
            wavelengths_nm.push_back(wavelength);
            toa_irradiance.push_back(irradiance);
        }
    }
    wehrli_stream.close();

    if (wavelengths_nm.empty()) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromDirectory): No data loaded from " + wehrli_file);
    }

    // Load abscoef.dat (absorption coefficients)
    std::string abscoef_file = data_path + "/abscoef.dat";
    std::ifstream abscoef_stream(abscoef_file);
    if (!abscoef_stream.is_open()) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromDirectory): Could not open file " + abscoef_file);
    }

    h2o_coef.clear();
    h2o_exp.clear();
    o3_xsec.clear();
    o2_coef.clear();

    while (std::getline(abscoef_stream, line)) {
        // Skip comments and empty lines
        if (line.empty() || line[0] == '#') {
            continue;
        }

        std::istringstream iss(line);
        float wavelength, h2o_c, h2o_e, o3_x, o2_c;
        float no2_x, co2_c; // We don't use these, but need to read them
        if (iss >> wavelength >> h2o_c >> h2o_e >> o3_x >> o2_c >> no2_x >> co2_c) {
            h2o_coef.push_back(h2o_c);
            h2o_exp.push_back(h2o_e);
            o3_xsec.push_back(o3_x);
            o2_coef.push_back(o2_c);
        }
    }
    abscoef_stream.close();

    if (h2o_coef.empty()) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromDirectory): No data loaded from " + abscoef_file);
    }

    // Validate that wavelength arrays match
    if (wavelengths_nm.size() != h2o_coef.size()) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromDirectory): Wavelength arrays from wehrli.dat and abscoef.dat do not match in size");
    }

    // Validate wavelength range (should be 300-2600 nm)
    if (wavelengths_nm.front() < 299.5f || wavelengths_nm.front() > 300.5f) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromDirectory): Expected wavelength range to start near 300 nm, got " + std::to_string(wavelengths_nm.front()));
    }
    if (wavelengths_nm.back() < 2599.5f || wavelengths_nm.back() > 2600.5f) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::loadFromDirectory): Expected wavelength range to end near 2600 nm, got " + std::to_string(wavelengths_nm.back()));
    }
}

float SolarPosition::SpectralData::interpolate(const std::vector<float> &x, const std::vector<float> &y, float x_val) {
    if (x.empty() || y.empty() || x.size() != y.size()) {
        helios_runtime_error("ERROR (SolarPosition::SpectralData::interpolate): Invalid input vectors");
    }

    // If x_val is outside the range, return boundary values
    if (x_val <= x.front()) {
        return y.front();
    }
    if (x_val >= x.back()) {
        return y.back();
    }

    // Find the two points for interpolation using binary search
    auto it = std::lower_bound(x.begin(), x.end(), x_val);
    if (it == x.begin()) {
        return y.front();
    }
    if (it == x.end()) {
        return y.back();
    }

    size_t idx1 = std::distance(x.begin(), it) - 1;
    size_t idx2 = idx1 + 1;

    // Linear interpolation
    float x1 = x[idx1];
    float x2 = x[idx2];
    float y1 = y[idx1];
    float y2 = y[idx2];

    return y1 + (y2 - y1) * (x_val - x1) / (x2 - x1);
}

float SolarPosition::calculateGeometricFactor(int julian_day) const {
    // Fourier series coefficients for Earth-Sun distance correction
    // Based on Duffie and Beckman (2013), Solar Engineering of Thermal Processes
    const float c[] = {1.00011f, 0.03422f, 0.00128f, 0.000719f, 0.000077f};

    // Day angle in radians
    float day_angle = (julian_day - 1) * 2.0f * M_PI / 365.0f;
    float day_angle_2 = day_angle * 2.0f;

    // Calculate geometric factor (inverse square of Earth-Sun distance ratio)
    float geo_factor = c[0] + c[1] * cosf(day_angle) + c[2] * sinf(day_angle) + c[3] * cosf(day_angle_2) + c[4] * sinf(day_angle_2);

    return geo_factor;
}

void SolarPosition::calculateRayleighTransmittance(const std::vector<float> &wavelengths_um, float mu0, float pressure_ratio, std::vector<float> &tdir, std::vector<float> &tglb, std::vector<float> &tdif, std::vector<float> &atm_albedo) const {
    // Bates (1984) formula for Rayleigh optical depth with Sobolev approximation
    const float c[] = {117.2594f, -1.3215f, 0.000320f, -0.000076f};

    // Resize output vectors
    size_t n = wavelengths_um.size();
    tdir.resize(n);
    tglb.resize(n);
    tdif.resize(n);
    atm_albedo.resize(n);

    for (size_t i = 0; i < n; ++i) {
        float w = wavelengths_um[i];
        float w2 = w * w;
        float w4 = w2 * w2;

        // Rayleigh optical depth (Bates formula)
        float tau = pressure_ratio / (c[0] * w4 + c[1] * w2 + c[2] + c[3] / w2);

        // Direct beam transmittance (Beer-Lambert law)
        tdir[i] = expf(-tau / mu0);

        // Global transmittance (Sobolev's two-stream approximation)
        tglb[i] = ((2.0f / 3.0f + mu0) + (2.0f / 3.0f - mu0) * tdir[i]) / (4.0f / 3.0f + tau);

        // Diffuse transmittance
        tdif[i] = tglb[i] - tdir[i];

        // Atmospheric albedo (spherical albedo for Rayleigh scattering)
        atm_albedo[i] = tau * (1.0f - expf(-2.0f * tau)) / (2.0f + tau);
    }
}

void SolarPosition::calculateAerosolTransmittance(const std::vector<float> &wavelengths_um, float mu0, float alpha, float beta, float w0, float g, std::vector<float> &tdir, std::vector<float> &tglb, std::vector<float> &tdif,
                                                  std::vector<float> &atm_albedo) const {
    // Ångström law for aerosol optical depth with Ambartsumian solution for transmittance

    // Resize output vectors
    size_t n = wavelengths_um.size();
    tdir.resize(n);
    tglb.resize(n);
    tdif.resize(n);
    atm_albedo.resize(n);

    // Ambartsumian's parameter
    float K = sqrtf((1.0f - w0) * (1.0f - w0 * g));
    float r0 = (K - 1.0f + w0) / (K + 1.0f - w0);

    for (size_t i = 0; i < n; ++i) {
        float w = wavelengths_um[i];

        // Ångström formula for aerosol optical depth
        float tau = beta / powf(w, alpha);

        // Direct beam transmittance
        tdir[i] = expf(-tau / mu0);

        // Ambartsumian's solution for global transmittance
        float tdir_k = powf(tdir[i], K);
        tglb[i] = (1.0f - r0 * r0) * tdir_k / (1.0f - r0 * r0 * tdir_k * tdir_k);

        // Diffuse transmittance
        tdif[i] = tglb[i] - tdir[i];

        // Atmospheric albedo for aerosols
        float g_factor = (1.0f - g) * w0;
        atm_albedo[i] = g_factor * tau / (2.0f + g_factor * tau) * (1.0f + expf(-g_factor * tau));
    }
}

void SolarPosition::calculateMixtureTransmittance(const std::vector<float> &wavelengths_um, float mu0, float pressure_Pa, float turbidity_beta, float angstrom_alpha, float aerosol_ssa, float aerosol_g, bool coupling, std::vector<float> &tglb,
                                                  std::vector<float> &tdir, std::vector<float> &tdif, std::vector<float> &atm_albedo) const {
    // Combined Rayleigh-aerosol transmittance with coupling (Cachorro et al. 2022, Eq. 20)

    float pressure_ratio = pressure_Pa / 101325.0f;

    // Calculate separate Rayleigh and aerosol components
    std::vector<float> ray_tdir, ray_tglb, ray_tdif, ray_albedo;
    std::vector<float> aer_tdir, aer_tglb, aer_tdif, aer_albedo;

    calculateRayleighTransmittance(wavelengths_um, mu0, pressure_ratio, ray_tdir, ray_tglb, ray_tdif, ray_albedo);
    calculateAerosolTransmittance(wavelengths_um, mu0, angstrom_alpha, turbidity_beta, aerosol_ssa, aerosol_g, aer_tdir, aer_tglb, aer_tdif, aer_albedo);

    // Resize output vectors
    size_t n = wavelengths_um.size();
    tglb.resize(n);
    tdir.resize(n);
    tdif.resize(n);
    atm_albedo.resize(n);

    // Combine transmittances
    for (size_t i = 0; i < n; ++i) {
        if (coupling) {
            // With Rayleigh-aerosol coupling (Cachorro et al. 2022, Eq. 20)
            float beta_w = turbidity_beta / powf(wavelengths_um[i], angstrom_alpha);
            float tau_w = beta_w / mu0;
            float coup_term = tau_w * (1.0f - aer_tglb[i]);

            tglb[i] = ray_tglb[i] * aer_tglb[i] + coup_term;
            tdir[i] = ray_tdir[i] * aer_tdir[i];
        } else {
            // Without coupling (simple multiplication)
            tglb[i] = ray_tglb[i] * aer_tglb[i];
            tdir[i] = ray_tdir[i] * aer_tdir[i];
        }

        tdif[i] = tglb[i] - tdir[i];

        // Combined atmospheric albedo (weighted average)
        atm_albedo[i] = ray_albedo[i] + aer_albedo[i];
    }
}

void SolarPosition::calculateWaterVaporTransmittance(const SpectralData &data, float mu0, float water_vapor_cm, std::vector<float> &transmittance) const {
    // Water vapor absorption using empirical coefficients (Gueymard 1994)

    size_t n = data.wavelengths_nm.size();
    transmittance.resize(n);

    for (size_t i = 0; i < n; ++i) {
        float h2o_coef = data.h2o_coef[i];
        float h2o_exp = data.h2o_exp[i];

        if (h2o_exp < 1e-6f) {
            // No absorption in this band
            transmittance[i] = 1.0f;
        } else {
            // Power law absorption model
            float optical_depth = powf(h2o_coef * water_vapor_cm / mu0, h2o_exp);
            transmittance[i] = expf(-optical_depth);
        }
    }
}

void SolarPosition::calculateOzoneTransmittance(const SpectralData &data, float mu0, float ozone_DU, std::vector<float> &transmittance) const {
    // Ozone absorption using measured cross sections
    const float LOSCHMIDT = 2.687e19f; // molecules/cm³ at STP

    size_t n = data.wavelengths_nm.size();
    transmittance.resize(n);

    for (size_t i = 0; i < n; ++i) {
        // Convert cross section to absorption coefficient
        float o3_coef = LOSCHMIDT * data.o3_xsec[i]; // cm⁻¹

        // Convert ozone column from Dobson Units to cm
        float o3_path_cm = ozone_DU * 1e-3f;

        // Calculate optical depth and transmittance
        float optical_depth = o3_coef * o3_path_cm / mu0;
        transmittance[i] = expf(-optical_depth);
    }
}

void SolarPosition::calculateOxygenTransmittance(const SpectralData &data, float mu0, std::vector<float> &transmittance) const {
    // Oxygen absorption (primarily A-band at 760 nm and continuum)
    const float O2_PATH = 0.209f * 173200.0f; // atm-cm
    const float O2_EXP = 0.5641f;

    size_t n = data.wavelengths_nm.size();
    transmittance.resize(n);

    for (size_t i = 0; i < n; ++i) {
        float o2_coef = data.o2_coef[i];

        // Power law absorption model for O2
        float optical_depth = powf(o2_coef * O2_PATH / mu0, O2_EXP);
        transmittance[i] = expf(-optical_depth);
    }
}

void SolarPosition::calculateSpectralIrradianceComponents(std::vector<helios::vec2> &global_spectrum, std::vector<helios::vec2> &direct_spectrum, std::vector<helios::vec2> &diffuse_spectrum, float resolution_nm) const {
    // Validate resolution
    if (resolution_nm < 1.0f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSpectralIrradianceComponents): resolution_nm must be >= 1 nm, got " + std::to_string(resolution_nm));
    }
    if (resolution_nm > 2300.0f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSpectralIrradianceComponents): resolution_nm must be <= 2300 nm, got " + std::to_string(resolution_nm));
    }
    // Get atmospheric conditions
    float pressure, temperature, humidity, turbidity_beta;
    getAtmosphericConditions(pressure, temperature, humidity, turbidity_beta);

    // Derive additional parameters
    uint DOY = context->getJulianDate();

    // Water vapor from Viswanadham (1981)
    float gamma = logf(humidity) + 17.67f * (temperature - 273.0f) / 268.0f;
    float tdp = 243.0f * gamma / (17.67f - gamma) * 9.0f / 5.0f + 32.0f;
    float water_vapor_cm = expf((0.1133f - logf(5.0f)) + 0.0393f * tdp);

    // Ozone from van Heuklon (1979)
    float uo_atm_cm = (235.0f + (150.0f + 40.0f * sinf(0.9856f * (DOY - 30.0f) * M_PI / 180.0f) + 20.0f * sinf(3.0f * (longitude * M_PI / 180.0f + 20.0f))) * powf(sinf(1.28f * latitude * M_PI / 180.0f), 2)) * 0.001f;
    float ozone_DU = uo_atm_cm * 1000.0f;

    // Fixed parameters
    const float angstrom_alpha = 1.3f;
    const float surface_albedo = 0.2f;
    const float aerosol_ssa = 0.90f;
    const float aerosol_g = 0.85f;

    // Load spectral data (cached after first load)
    static SpectralData spectral_data;
    static bool data_loaded = false;
    if (!data_loaded) {
        spectral_data.loadFromDirectory("plugins/solarposition/ssolar_goa");
        data_loaded = true;
    }

    // Calculate solar geometry
    float sun_zenith = getSunZenith();
    float mu0 = cosf(sun_zenith);
    if (mu0 <= 0.0f) {
        helios_runtime_error("ERROR (SolarPosition::calculateSpectralIrradiance): Cannot calculate spectral irradiance when sun is below horizon (zenith angle = " + std::to_string(sun_zenith * 180.0f / M_PI) + " degrees)");
    }

    int julian_day = context->getJulianDate();
    float geo_factor = calculateGeometricFactor(julian_day);

    // Apply geometric factor to TOA spectrum
    std::vector<float> toa_irr(spectral_data.toa_irradiance.size());
    for (size_t i = 0; i < toa_irr.size(); ++i) {
        toa_irr[i] = spectral_data.toa_irradiance[i] * geo_factor;
    }

    // Convert wavelengths from nm to μm
    std::vector<float> wavelengths_um(spectral_data.wavelengths_nm.size());
    for (size_t i = 0; i < wavelengths_um.size(); ++i) {
        wavelengths_um[i] = spectral_data.wavelengths_nm[i] * 0.001f;
    }

    // Calculate atmospheric scattering transmittances
    std::vector<float> t_scat_glb, t_scat_dir, t_scat_dif, atm_alb;
    calculateMixtureTransmittance(wavelengths_um, mu0, pressure, turbidity_beta, angstrom_alpha, aerosol_ssa, aerosol_g, true, t_scat_glb, t_scat_dir, t_scat_dif, atm_alb);

    // Calculate gas absorption transmittances
    std::vector<float> t_h2o, t_o3, t_o2;
    calculateWaterVaporTransmittance(spectral_data, mu0, water_vapor_cm, t_h2o);
    calculateOzoneTransmittance(spectral_data, mu0, ozone_DU, t_o3);
    calculateOxygenTransmittance(spectral_data, mu0, t_o2);

    // Combine gas transmittances
    std::vector<float> t_gas(spectral_data.wavelengths_nm.size());
    for (size_t i = 0; i < t_gas.size(); ++i) {
        t_gas[i] = t_h2o[i] * t_o3[i] * t_o2[i];
    }

    // Calculate amplification factor
    std::vector<float> amp_factor(spectral_data.wavelengths_nm.size());
    for (size_t i = 0; i < amp_factor.size(); ++i) {
        amp_factor[i] = 1.0f / (1.0f - surface_albedo * atm_alb[i]);
    }

    // Calculate spectral irradiances
    global_spectrum.clear();
    direct_spectrum.clear();
    diffuse_spectrum.clear();
    global_spectrum.reserve(spectral_data.wavelengths_nm.size());
    direct_spectrum.reserve(spectral_data.wavelengths_nm.size());
    diffuse_spectrum.reserve(spectral_data.wavelengths_nm.size());

    for (size_t i = 0; i < spectral_data.wavelengths_nm.size(); ++i) {
        float wavelength_nm = spectral_data.wavelengths_nm[i];

        float direct_irr = toa_irr[i] * t_scat_dir[i] * t_gas[i];
        float global_irr = toa_irr[i] * mu0 * t_scat_glb[i] * t_gas[i] * amp_factor[i];
        float diffuse_irr = global_irr - direct_irr * mu0;

        global_spectrum.push_back(helios::make_vec2(wavelength_nm, global_irr));
        direct_spectrum.push_back(helios::make_vec2(wavelength_nm, direct_irr));
        diffuse_spectrum.push_back(helios::make_vec2(wavelength_nm, diffuse_irr));
    }

    // Downsample if requested resolution is coarser than 1 nm
    if (resolution_nm > 1.0f + 1e-5f) {
        std::vector<helios::vec2> global_downsampled, direct_downsampled, diffuse_downsampled;

        for (float wl = 300.0f; wl <= 2600.0f; wl += resolution_nm) {
            // Find closest wavelength in original spectrum
            float min_dist = std::numeric_limits<float>::max();
            size_t closest_idx = 0;
            for (size_t i = 0; i < global_spectrum.size(); ++i) {
                float dist = std::fabs(global_spectrum[i].x - wl);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_idx = i;
                }
            }

            global_downsampled.push_back(global_spectrum[closest_idx]);
            direct_downsampled.push_back(direct_spectrum[closest_idx]);
            diffuse_downsampled.push_back(diffuse_spectrum[closest_idx]);
        }

        global_spectrum = global_downsampled;
        direct_spectrum = direct_downsampled;
        diffuse_spectrum = diffuse_downsampled;
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
