#include "SolarPosition.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

TEST_CASE("SolarPosition sun position Boulder") {
    Context context_s;

    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(1, 1, 2000)));
    DOCTEST_CHECK_NOTHROW(context_s.setTime(make_Time(10, 30, 0)));

    SolarPosition sp(7, 40.1250f, 105.2369f, &context_s);
    float theta_s = sp.getSunElevation() * 180.f / M_PI;
    float phi_s = sp.getSunAzimuth() * 180.f / M_PI;

    DOCTEST_CHECK(std::fabs(theta_s - 29.49f) <= 10.0f);
    DOCTEST_CHECK(std::fabs(phi_s - 154.18f) <= 5.0f);
}

TEST_CASE("SolarPosition fractional UTC offset") {
    // A fractional UTC offset (e.g. +05:30 zones like India, Helios convention -5.5) must be
    // honored rather than truncated to a whole hour. The local standard time meridian scales
    // linearly with the offset, so the half-hour offset should fall between the two neighboring
    // integer offsets and be very close to their average.
    Context context_s;
    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(80, 2023))); // near equinox
    DOCTEST_CHECK_NOTHROW(context_s.setTime(make_Time(12, 0, 0)));

    const float latitude = 28.6139f; // New Delhi
    const float longitude = -77.2090f; // Helios convention: +West, so New Delhi is negative

    SolarPosition sp_lo(5.f, latitude, longitude, &context_s);
    SolarPosition sp_mid(5.5f, latitude, longitude, &context_s);
    SolarPosition sp_hi(6.f, latitude, longitude, &context_s);

    float az_lo = sp_lo.getSunAzimuth();
    float az_mid = sp_mid.getSunAzimuth();
    float az_hi = sp_hi.getSunAzimuth();

    // The half-hour offset must differ from both whole-hour offsets (i.e. it was not truncated to
    // a whole hour) and must lie strictly between them, since the local standard time meridian — and
    // hence the computed sun azimuth — varies monotonically with the UTC offset over this 1-hour span.
    DOCTEST_CHECK(std::fabs(az_mid - az_lo) > 1e-4f);
    DOCTEST_CHECK(std::fabs(az_mid - az_hi) > 1e-4f);
    DOCTEST_CHECK(((az_lo < az_mid && az_mid < az_hi) || (az_hi < az_mid && az_mid < az_lo)));
}

TEST_CASE("SolarPosition ambient longwave model") {
    Context context_s;
    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(5, 5, 2003)));
    DOCTEST_CHECK_NOTHROW(context_s.setTime(make_Time(9, 10, 0)));

    SolarPosition sp(6, 36.5289f, 97.4439f, &context_s);

    // Set atmospheric conditions
    sp.setAtmosphericConditions(101325.f, 290.f, 0.5f, 0.02f);

    float LW;
    DOCTEST_CHECK_NOTHROW(LW = sp.getAmbientLongwaveFlux());

    DOCTEST_CHECK(doctest::Approx(310.03192f).epsilon(1e-6f) == LW);
}

TEST_CASE("SolarPosition sunrise and sunset") {
    Context context_s;
    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(1, 1, 2023)));
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    Time sunrise;
    DOCTEST_CHECK_NOTHROW(sunrise = sp.getSunriseTime());
    Time sunset;
    DOCTEST_CHECK_NOTHROW(sunset = sp.getSunsetTime());

    DOCTEST_CHECK(!(sunrise.hour == 0 && sunrise.minute == 0));
    DOCTEST_CHECK(!(sunset.hour == 0 && sunset.minute == 0));
}

TEST_CASE("SolarPosition sun direction vector") {
    Context context_s;
    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(1, 1, 2023)));
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    vec3 dir;
    DOCTEST_CHECK_NOTHROW(dir = sp.getSunDirectionVector());
    DOCTEST_CHECK(dir.x != 0.f);
    DOCTEST_CHECK(dir.y != 0.f);
    DOCTEST_CHECK(dir.z != 0.f);
}

TEST_CASE("SolarPosition sun direction spherical") {
    Context context_s;
    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(1, 1, 2023)));
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    SphericalCoord dir;
    DOCTEST_CHECK_NOTHROW(dir = sp.getSunDirectionSpherical());
    DOCTEST_CHECK(dir.elevation > 0.f);
    DOCTEST_CHECK(dir.azimuth > 0.f);
}

TEST_CASE("SolarPosition flux and fractions") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Set atmospheric conditions
    sp.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.02f);

    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux());
    DOCTEST_CHECK(flux > 0.f);

    float diffuse_fraction;
    DOCTEST_CHECK_NOTHROW(diffuse_fraction = sp.getDiffuseFraction());
    DOCTEST_CHECK(diffuse_fraction >= 0.f);
    DOCTEST_CHECK(diffuse_fraction <= 1.f);

    float flux_par;
    DOCTEST_CHECK_NOTHROW(flux_par = sp.getSolarFluxPAR());
    DOCTEST_CHECK(flux_par > 0.f);

    float flux_nir;
    DOCTEST_CHECK_NOTHROW(flux_nir = sp.getSolarFluxNIR());
    DOCTEST_CHECK(flux_nir > 0.f);
}

TEST_CASE("SolarPosition elevation, zenith, azimuth") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    float elevation;
    DOCTEST_CHECK_NOTHROW(elevation = sp.getSunElevation());
    DOCTEST_CHECK(elevation >= 0.f);
    DOCTEST_CHECK(elevation <= M_PI / 2.f);

    float zenith;
    DOCTEST_CHECK_NOTHROW(zenith = sp.getSunZenith());
    DOCTEST_CHECK(zenith >= 0.f);
    DOCTEST_CHECK(zenith <= M_PI);

    float azimuth;
    DOCTEST_CHECK_NOTHROW(azimuth = sp.getSunAzimuth());
    DOCTEST_CHECK(azimuth >= 0.f);
    DOCTEST_CHECK(azimuth <= 2.f * M_PI);
}

TEST_CASE("SolarPosition turbidity calibration") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);
    std::string label = "test_flux_timeseries";

    if (!context_s.doesTimeseriesVariableExist(label.c_str())) {
        return; // skip test if data does not exist
    }

    float turbidity;
    std::string captured_warnings;
    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_NOTHROW(turbidity = sp.calibrateTurbidityFromTimeseries(label));
        captured_warnings = cerr_buffer.get_captured_output();
    } // Capture goes out of scope before assertions

    DOCTEST_CHECK(turbidity > 0.f);

    // Turbidity calibration may produce fzero warnings due to the nature of the optimization problem
    // This is expected behavior and should not be considered a test failure
    // Just verify the function completes and returns a valid result
}

TEST_CASE("SolarPosition invalid lat/long") {
    Context context_s;

    bool had_output_1, had_output_2;
    {
        capture_cerr cerr_buffer;
        SolarPosition sp_1(7, -100.f, 105.2369f, &context_s);
        had_output_1 = cerr_buffer.has_output();

        cerr_buffer.clear();
        SolarPosition sp_2(7, 40.125f, -200.f, &context_s);
        had_output_2 = cerr_buffer.has_output();
    } // capture goes out of scope before assertions
    DOCTEST_CHECK(had_output_1);
    DOCTEST_CHECK(had_output_2);
}

TEST_CASE("SolarPosition invalid solar angle") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Set atmospheric conditions
    sp.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.02f);

    DOCTEST_CHECK_NOTHROW(sp.setSunDirection(make_SphericalCoord(0.75 * M_PI, M_PI / 2.f)));

    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux());
    DOCTEST_CHECK(flux == 0.f);
}


TEST_CASE("SolarPosition solor position overridden") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    DOCTEST_CHECK_NOTHROW(sp.setSunDirection(make_SphericalCoord(M_PI / 4.f, M_PI / 2.f)));

    float elevation;
    DOCTEST_CHECK_NOTHROW(elevation = sp.getSunElevation());
    DOCTEST_CHECK(elevation >= 0.f);
    DOCTEST_CHECK(elevation <= M_PI / 2.f);

    float zenith;
    DOCTEST_CHECK_NOTHROW(zenith = sp.getSunZenith());
    DOCTEST_CHECK(zenith >= 0.f);
    DOCTEST_CHECK(zenith <= M_PI);

    float azimuth;
    DOCTEST_CHECK_NOTHROW(azimuth = sp.getSunAzimuth());
    DOCTEST_CHECK(azimuth >= 0.f);
    DOCTEST_CHECK(azimuth <= 2.f * M_PI);

    vec3 sun_vector;
    DOCTEST_CHECK_NOTHROW(sun_vector = sp.getSunDirectionVector());

    SphericalCoord sun_spherical;
    DOCTEST_CHECK_NOTHROW(sun_spherical = sp.getSunDirectionSpherical());
}

TEST_CASE("SolarPosition cloud calibration") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    DOCTEST_CHECK_NOTHROW(context_s.loadTabularTimeseriesData("lib/testdata/cimis.csv", {"CIMIS"}, ","));

    context_s.setDate(make_Date(13, 7, 2023));
    context_s.setTime(make_Time(12, 0, 0));

    // Set atmospheric conditions
    sp.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.02f);

    DOCTEST_CHECK_NOTHROW(sp.enableCloudCalibration("net_radiation"));

    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux());
    DOCTEST_CHECK(flux > 0.f);

    float diffuse_fraction;
    DOCTEST_CHECK_NOTHROW(diffuse_fraction = sp.getDiffuseFraction());
    DOCTEST_CHECK(diffuse_fraction >= 0.f);
    DOCTEST_CHECK(diffuse_fraction <= 1.f);

    float flux_par;
    DOCTEST_CHECK_NOTHROW(flux_par = sp.getSolarFluxPAR());
    DOCTEST_CHECK(flux_par > 0.f);

    float flux_nir;
    DOCTEST_CHECK_NOTHROW(flux_nir = sp.getSolarFluxNIR());
    DOCTEST_CHECK(flux_nir > 0.f);

    DOCTEST_CHECK_NOTHROW(sp.disableCloudCalibration());

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS_AS(sp.enableCloudCalibration("non_existent_timeseries"), std::runtime_error);
    }
}

TEST_CASE("SolarPosition cloud calibration preserves PAR/NIR partitioning") {
    // The cloud calibration scales the clear-sky flux by the broadband factor R_meas/R_clear,h.
    // That factor must be computed from the ALL-WAVE flux and then applied to each band, so that
    // (a) the PAR/NIR ratio is unchanged by calibration, and (b) the bands still sum to the
    // broadband total. Computing the factor from whichever band is passed in makes the input
    // cancel algebraically (R_calc*R_meas/(R_calc*cos(theta)) == R_meas/cos(theta)), collapsing
    // PAR, NIR and broadband onto a single value.
    Context context_s;

    // Noon at the summer solstice near Davis, CA, so the sun is high and the clear-sky flux large.
    const Date date = make_Date(21, 6, 2020);
    const Time time = make_Time(12, 30, 0);
    context_s.setDate(date);
    context_s.setTime(time);

    SolarPosition sp(-8, 38.55f, -121.76f, &context_s);
    sp.setAtmosphericConditions(101000.f, 300.f, 0.5f, 0.05f);

    const float cos_theta = std::cos(sp.getSunZenith());
    DOCTEST_REQUIRE(cos_theta > 0.5f); // sun must be well above the horizon for this test to mean anything

    // Clear-sky (uncalibrated) reference values.
    const float par_clear = sp.getSolarFluxPAR();
    const float nir_clear = sp.getSolarFluxNIR();
    const float total_clear = sp.getSolarFlux();

    DOCTEST_REQUIRE(par_clear > 0.f);
    DOCTEST_REQUIRE(nir_clear > 0.f);
    DOCTEST_REQUIRE(total_clear > 0.f);
    // Sanity: the two bands partition the broadband flux, and neither dominates entirely.
    DOCTEST_CHECK(par_clear + nir_clear == doctest::Approx(total_clear).epsilon(1e-4));
    const float par_fraction_clear = par_clear / total_clear;
    DOCTEST_REQUIRE(par_fraction_clear > 0.1f);
    DOCTEST_REQUIRE(par_fraction_clear < 0.9f);

    // Impose a cloudy measurement: 55% of the clear-sky flux on a horizontal plane. Using a
    // fraction well below 1 keeps the calibration branch active (fdiff > 0.001) and makes the
    // expected attenuation an order-of-magnitude effect rather than a marginal one.
    const float cloud_factor = 0.55f;
    const float R_meas_horizontal = cloud_factor * total_clear * cos_theta;
    context_s.addTimeseriesData("R_meas_Wm2", R_meas_horizontal, date, time);
    context_s.setCurrentTimeseriesPoint("R_meas_Wm2", 0);

    // Re-pin date/time: setCurrentTimeseriesPoint moves the Context clock to the data point.
    DOCTEST_REQUIRE(std::fabs(std::cos(sp.getSunZenith()) - cos_theta) < 1e-4f);

    sp.enableCloudCalibration("R_meas_Wm2");

    const float par_cloudy = sp.getSolarFluxPAR();
    const float nir_cloudy = sp.getSolarFluxNIR();
    const float total_cloudy = sp.getSolarFlux();

    // The calibration must actually have done something.
    DOCTEST_CHECK(total_cloudy < total_clear);
    DOCTEST_CHECK(total_cloudy == doctest::Approx(R_meas_horizontal / cos_theta).epsilon(1e-3));

    // (a) The spectral partitioning must survive calibration.
    DOCTEST_CHECK(par_cloudy / total_cloudy == doctest::Approx(par_fraction_clear).epsilon(1e-3));
    DOCTEST_CHECK(nir_cloudy / par_cloudy == doctest::Approx(nir_clear / par_clear).epsilon(1e-3));

    // (b) Energy must be conserved: the bands sum to the broadband total, not to twice it.
    DOCTEST_CHECK(par_cloudy + nir_cloudy == doctest::Approx(total_cloudy).epsilon(1e-3));

    // Each band should be attenuated by the same broadband cloud factor.
    DOCTEST_CHECK(par_cloudy == doctest::Approx(cloud_factor * par_clear).epsilon(1e-3));
    DOCTEST_CHECK(nir_cloudy == doctest::Approx(cloud_factor * nir_clear).epsilon(1e-3));
}

TEST_CASE("SolarPosition cloud calibration diffuse fraction responds to cloud") {
    // The calibrated diffuse fraction must increase monotonically as the measured flux falls further
    // below the clear-sky prediction. The previous heuristic, 1-(R_meas-R_calc)/R_calc, reduces to
    // 2-R_meas/R_calc, which exceeds 1 for any measurement below clear-sky and so clamped to a
    // fully-diffuse sky for every real cloud condition, reporting overcast for light haze.
    Context context_s;

    const Date date = make_Date(21, 6, 2020);
    const Time time = make_Time(12, 30, 0);
    context_s.setDate(date);
    context_s.setTime(time);

    SolarPosition sp(-8, 38.55f, -121.76f, &context_s);
    sp.setAtmosphericConditions(101000.f, 300.f, 0.5f, 0.05f);

    const float cos_theta = std::cos(sp.getSunZenith());
    DOCTEST_REQUIRE(cos_theta > 0.5f);

    const float total_clear = sp.getSolarFlux();
    const float clear_horizontal = total_clear * cos_theta;
    DOCTEST_REQUIRE(clear_horizontal > 100.f);

    // Seed one timeseries point, then overwrite it for each cloud level.
    context_s.addTimeseriesData("R_meas_Wm2", clear_horizontal, date, time);
    context_s.setCurrentTimeseriesPoint("R_meas_Wm2", 0);
    sp.enableCloudCalibration("R_meas_Wm2");

    // Thin cloud should stay predominantly direct; thick cloud should be predominantly diffuse.
    context_s.updateTimeseriesData("R_meas_Wm2", date, time, 0.95f * clear_horizontal);
    const float fdiff_thin = sp.getDiffuseFraction();

    context_s.updateTimeseriesData("R_meas_Wm2", date, time, 0.60f * clear_horizontal);
    const float fdiff_moderate = sp.getDiffuseFraction();

    context_s.updateTimeseriesData("R_meas_Wm2", date, time, 0.20f * clear_horizontal);
    const float fdiff_thick = sp.getDiffuseFraction();

    DOCTEST_CHECK(fdiff_thin >= 0.f);
    DOCTEST_CHECK(fdiff_thick <= 1.f);

    // Monotonic in cloud thickness.
    DOCTEST_CHECK(fdiff_thin < fdiff_moderate);
    DOCTEST_CHECK(fdiff_moderate < fdiff_thick);

    // A nearly clear sky must NOT be reported as fully diffuse.
    DOCTEST_CHECK(fdiff_thin < 0.6f);

    // A heavily overcast sky must be predominantly diffuse.
    DOCTEST_CHECK(fdiff_thick > 0.8f);
}

TEST_CASE("SolarPosition cloud calibration rejects negative measured flux") {
    // A negative measured shortwave flux (a sentinel or a faulty sensor reading) must be reported
    // rather than silently producing negative fluxes.
    Context context_s;

    const Date date = make_Date(21, 6, 2020);
    const Time time = make_Time(12, 30, 0);
    context_s.setDate(date);
    context_s.setTime(time);

    SolarPosition sp(-8, 38.55f, -121.76f, &context_s);
    sp.setAtmosphericConditions(101000.f, 300.f, 0.5f, 0.05f);

    context_s.addTimeseriesData("R_bad", -999.f, date, time);
    context_s.setCurrentTimeseriesPoint("R_bad", 0);
    sp.enableCloudCalibration("R_bad");

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS_AS([[maybe_unused]] float flux = sp.getSolarFluxPAR(), std::runtime_error);
        DOCTEST_CHECK_THROWS_AS(sp.calculateGlobalSolarSpectrum("spectrum_bad"), std::runtime_error);
    }
}

TEST_CASE("SolarPosition spectral cloud calibration") {
    // The spectral (SSolar-GOA) path must honor enableCloudCalibration(). The calibration normalizes
    // the modeled clear-sky spectrum to the measured broadband flux and then applies the Bird,
    // Riordan & Myers (1987) spectral modifiers, which transmit short wavelengths preferentially.
    Context context_s;

    const Date date = make_Date(21, 6, 2020);
    const Time time = make_Time(12, 30, 0);
    context_s.setDate(date);
    context_s.setTime(time);

    SolarPosition sp(-8, 38.55f, -121.76f, &context_s);
    sp.setAtmosphericConditions(101000.f, 300.f, 0.5f, 0.05f);

    const float cos_theta = std::cos(sp.getSunZenith());
    DOCTEST_REQUIRE(cos_theta > 0.5f);

    // Clear-sky reference spectra.
    sp.calculateGlobalSolarSpectrum("global_clear");
    sp.calculateDiffuseSolarSpectrum("diffuse_clear");
    std::vector<vec2> global_clear, diffuse_clear;
    context_s.getGlobalData("global_clear", global_clear);
    context_s.getGlobalData("diffuse_clear", diffuse_clear);
    DOCTEST_REQUIRE(global_clear.size() > 100);

    auto integrate = [](const std::vector<vec2> &spectrum) {
        float total = 0.f;
        for (size_t i = 1; i < spectrum.size(); ++i) {
            total += 0.5f * (spectrum.at(i).y + spectrum.at(i - 1).y) * (spectrum.at(i).x - spectrum.at(i - 1).x);
        }
        return total;
    };

    const float integral_clear = integrate(global_clear);
    DOCTEST_REQUIRE(integral_clear > 100.f);

    // Impose a cloudy measurement well outside the 5% no-correction deadband. The measured value is
    // broadband, whereas the model covers only 300-2600 nm, so scale by the window fraction to get
    // the target the calibration should reproduce inside the window.
    const float model_window_fraction = 0.9596f;
    const float cloud_factor = 0.55f;
    const float R_meas_broadband = cloud_factor * integral_clear / model_window_fraction;
    context_s.addTimeseriesData("R_meas_Wm2", R_meas_broadband, date, time);
    context_s.setCurrentTimeseriesPoint("R_meas_Wm2", 0);
    DOCTEST_REQUIRE(std::fabs(std::cos(sp.getSunZenith()) - cos_theta) < 1e-4f);

    sp.enableCloudCalibration("R_meas_Wm2");

    sp.calculateGlobalSolarSpectrum("global_cloudy");
    sp.calculateDirectSolarSpectrum("direct_cloudy");
    sp.calculateDiffuseSolarSpectrum("diffuse_cloudy");
    std::vector<vec2> global_cloudy, direct_cloudy, diffuse_cloudy;
    context_s.getGlobalData("global_cloudy", global_cloudy);
    context_s.getGlobalData("direct_cloudy", direct_cloudy);
    context_s.getGlobalData("diffuse_cloudy", diffuse_cloudy);

    DOCTEST_REQUIRE(global_cloudy.size() == global_clear.size());

    // The calibration must actually have been applied to the spectral path.
    const float integral_cloudy = integrate(global_cloudy);
    DOCTEST_CHECK(integral_cloudy < integral_clear);
    DOCTEST_CHECK(integral_cloudy == doctest::Approx(cloud_factor * integral_clear).epsilon(0.05));

    // The two components must still compose the global spectrum: global = direct*mu0 + diffuse.
    for (size_t i = 0; i < global_cloudy.size(); i += 200) {
        DOCTEST_CHECK(global_cloudy.at(i).y == doctest::Approx(direct_cloudy.at(i).y * cos_theta + diffuse_cloudy.at(i).y).epsilon(1e-3));
    }

    // Cloud must shift the diffuse fraction upward: the clear-sky sky is mostly direct beam, whereas
    // an overcast sky is almost entirely diffuse.
    const float diffuse_fraction_clear = integrate(diffuse_clear) / integral_clear;
    const float diffuse_fraction_cloudy = integrate(diffuse_cloudy) / integral_cloudy;
    DOCTEST_CHECK(diffuse_fraction_cloudy > diffuse_fraction_clear);
    DOCTEST_CHECK(diffuse_fraction_cloudy > 0.5f);

    // Cloud must enrich the photosynthetically active fraction of the global spectrum, which is the
    // physically observable consequence of the spectral treatment. Measurements put the PAR fraction
    // of global shortwave at about 0.417 under clear sky rising to about 0.483 under overcast.
    auto par_fraction = [&](const std::vector<vec2> &spectrum) {
        float par = 0.f;
        for (size_t i = 1; i < spectrum.size(); ++i) {
            if (spectrum.at(i).x < 400.f || spectrum.at(i).x > 700.f) {
                continue;
            }
            par += 0.5f * (spectrum.at(i).y + spectrum.at(i - 1).y) * (spectrum.at(i).x - spectrum.at(i - 1).x);
        }
        return par / integrate(spectrum);
    };

    const float par_fraction_clear = par_fraction(global_clear);
    const float par_fraction_cloudy = par_fraction(global_cloudy);
    DOCTEST_CHECK(par_fraction_cloudy > par_fraction_clear);

    // A purely grey rescaling would leave the PAR fraction exactly unchanged, so require a shift
    // that is clearly larger than floating-point noise but still physically plausible.
    DOCTEST_CHECK(par_fraction_cloudy - par_fraction_clear > 0.01f);
    DOCTEST_CHECK(par_fraction_cloudy - par_fraction_clear < 0.10f);
}

TEST_CASE("SolarPosition spectral cloud calibration deadband") {
    // Within 5% agreement between the modeled and measured broadband irradiance, no cloud
    // modification is applied at all (Bird, Riordan & Myers 1987): the discrepancy there is
    // dominated by measurement and input uncertainty rather than by cloud.
    Context context_s;

    const Date date = make_Date(21, 6, 2020);
    const Time time = make_Time(12, 30, 0);
    context_s.setDate(date);
    context_s.setTime(time);

    SolarPosition sp(-8, 38.55f, -121.76f, &context_s);
    sp.setAtmosphericConditions(101000.f, 300.f, 0.5f, 0.05f);

    sp.calculateGlobalSolarSpectrum("global_clear_db");
    std::vector<vec2> global_clear;
    context_s.getGlobalData("global_clear_db", global_clear);

    float integral_clear = 0.f;
    for (size_t i = 1; i < global_clear.size(); ++i) {
        integral_clear += 0.5f * (global_clear.at(i).y + global_clear.at(i - 1).y) * (global_clear.at(i).x - global_clear.at(i - 1).x);
    }

    // A measurement only 2% below the model sits inside the deadband.
    const float model_window_fraction = 0.9596f;
    context_s.addTimeseriesData("R_meas_db", 0.98f * integral_clear / model_window_fraction, date, time);
    context_s.setCurrentTimeseriesPoint("R_meas_db", 0);

    sp.enableCloudCalibration("R_meas_db");
    sp.calculateGlobalSolarSpectrum("global_deadband");
    std::vector<vec2> global_deadband;
    context_s.getGlobalData("global_deadband", global_deadband);

    DOCTEST_REQUIRE(global_deadband.size() == global_clear.size());
    for (size_t i = 0; i < global_clear.size(); i += 200) {
        DOCTEST_CHECK(global_deadband.at(i).y == doctest::Approx(global_clear.at(i).y).epsilon(1e-5));
    }
}

TEST_CASE("SolarPosition turbidity calculation") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    DOCTEST_CHECK_NOTHROW(context_s.loadTabularTimeseriesData("lib/testdata/cimis.csv", {"CIMIS"}, ","));

    float turbidity;
    {
        // Turbidity calibration may produce fzero warnings - capture them
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_NOTHROW(turbidity = sp.calibrateTurbidityFromTimeseries("net_radiation"));
    } // capture goes out of scope before assertion
    DOCTEST_CHECK(turbidity > 0.f);

    {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS_AS(turbidity = sp.calibrateTurbidityFromTimeseries("non_existent_timeseries"), std::runtime_error);
    }
}

TEST_CASE("SolarPosition setAtmosphericConditions valid inputs") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Test setting valid atmospheric conditions
    DOCTEST_CHECK_NOTHROW(sp.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.02f));

    // Verify global data was set correctly
    float pressure, temperature, humidity, turbidity;
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("atmosphere_pressure_Pa", pressure));
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("atmosphere_temperature_K", temperature));
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("atmosphere_humidity_rel", humidity));
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("atmosphere_turbidity", turbidity));

    DOCTEST_CHECK(doctest::Approx(101325.f).epsilon(1e-6f) == pressure);
    DOCTEST_CHECK(doctest::Approx(300.f).epsilon(1e-6f) == temperature);
    DOCTEST_CHECK(doctest::Approx(0.5f).epsilon(1e-6f) == humidity);
    DOCTEST_CHECK(doctest::Approx(0.02f).epsilon(1e-6f) == turbidity);
}

TEST_CASE("SolarPosition setAtmosphericConditions validation") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Test invalid pressure (negative)
    DOCTEST_CHECK_THROWS_AS(sp.setAtmosphericConditions(-1000.f, 300.f, 0.5f, 0.02f), std::runtime_error);

    // Test invalid pressure (zero)
    DOCTEST_CHECK_THROWS_AS(sp.setAtmosphericConditions(0.f, 300.f, 0.5f, 0.02f), std::runtime_error);

    // Test invalid temperature (negative)
    DOCTEST_CHECK_THROWS_AS(sp.setAtmosphericConditions(101325.f, -10.f, 0.5f, 0.02f), std::runtime_error);

    // Test invalid temperature (zero)
    DOCTEST_CHECK_THROWS_AS(sp.setAtmosphericConditions(101325.f, 0.f, 0.5f, 0.02f), std::runtime_error);

    // Test invalid humidity (negative)
    DOCTEST_CHECK_THROWS_AS(sp.setAtmosphericConditions(101325.f, 300.f, -0.1f, 0.02f), std::runtime_error);

    // Test invalid humidity (> 1)
    DOCTEST_CHECK_THROWS_AS(sp.setAtmosphericConditions(101325.f, 300.f, 1.5f, 0.02f), std::runtime_error);

    // Test invalid turbidity (negative)
    DOCTEST_CHECK_THROWS_AS(sp.setAtmosphericConditions(101325.f, 300.f, 0.5f, -0.01f), std::runtime_error);

    // Test boundary values (should succeed)
    DOCTEST_CHECK_NOTHROW(sp.setAtmosphericConditions(0.001f, 0.001f, 0.f, 0.f));
    DOCTEST_CHECK_NOTHROW(sp.setAtmosphericConditions(200000.f, 400.f, 1.f, 1.f));
}

TEST_CASE("SolarPosition getAtmosphericConditions retrieval") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Set atmospheric conditions
    sp.setAtmosphericConditions(98000.f, 295.f, 0.65f, 0.05f);

    // Retrieve atmospheric conditions
    float pressure, temperature, humidity, turbidity;
    DOCTEST_CHECK_NOTHROW(sp.getAtmosphericConditions(pressure, temperature, humidity, turbidity));

    // Verify retrieved values match what was set
    DOCTEST_CHECK(doctest::Approx(98000.f).epsilon(1e-6f) == pressure);
    DOCTEST_CHECK(doctest::Approx(295.f).epsilon(1e-6f) == temperature);
    DOCTEST_CHECK(doctest::Approx(0.65f).epsilon(1e-6f) == humidity);
    DOCTEST_CHECK(doctest::Approx(0.05f).epsilon(1e-6f) == turbidity);
}

TEST_CASE("SolarPosition getAtmosphericConditions defaults") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Get atmospheric conditions without setting them first
    float pressure, temperature, humidity, turbidity;
    bool had_warning;
    {
        capture_cerr cerr_buffer;
        sp.getAtmosphericConditions(pressure, temperature, humidity, turbidity);
        had_warning = cerr_buffer.has_output();
    } // capture goes out of scope before assertions

    // Verify warning was issued
    DOCTEST_CHECK(had_warning);

    // Verify default values are used
    DOCTEST_CHECK(doctest::Approx(101325.f).epsilon(1e-6f) == pressure);
    DOCTEST_CHECK(doctest::Approx(300.f).epsilon(1e-6f) == temperature);
    DOCTEST_CHECK(doctest::Approx(0.5f).epsilon(1e-6f) == humidity);
    DOCTEST_CHECK(doctest::Approx(0.02f).epsilon(1e-6f) == turbidity);
}

TEST_CASE("SolarPosition parameter-free flux methods") {
    Context context_s;
    context_s.setDate(make_Date(1, 6, 2023));
    context_s.setTime(make_Time(12, 0, 0));

    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Set atmospheric conditions
    sp.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.02f);

    // Test parameter-free getSolarFlux()
    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux());
    DOCTEST_CHECK(flux > 0.f);

    // Test parameter-free getSolarFluxPAR()
    float flux_par;
    DOCTEST_CHECK_NOTHROW(flux_par = sp.getSolarFluxPAR());
    DOCTEST_CHECK(flux_par > 0.f);

    // Test parameter-free getSolarFluxNIR()
    float flux_nir;
    DOCTEST_CHECK_NOTHROW(flux_nir = sp.getSolarFluxNIR());
    DOCTEST_CHECK(flux_nir > 0.f);

    // Test parameter-free getDiffuseFraction()
    float diffuse_fraction;
    DOCTEST_CHECK_NOTHROW(diffuse_fraction = sp.getDiffuseFraction());
    DOCTEST_CHECK(diffuse_fraction >= 0.f);
    DOCTEST_CHECK(diffuse_fraction <= 1.f);

    // Test parameter-free getAmbientLongwaveFlux()
    float lw_flux;
    DOCTEST_CHECK_NOTHROW(lw_flux = sp.getAmbientLongwaveFlux());
    DOCTEST_CHECK(lw_flux > 0.f);
}

TEST_CASE("SolarPosition parameter-free methods with defaults") {
    Context context_s;
    context_s.setDate(make_Date(1, 6, 2023));
    context_s.setTime(make_Time(12, 0, 0));

    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    // Call parameter-free methods without setting atmospheric conditions
    // Should use default values
    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux());
    DOCTEST_CHECK(flux > 0.f);

    // Verify defaults are being used
    float pressure, temperature, humidity, turbidity;
    {
        capture_cerr cerr_buffer;
        sp.getAtmosphericConditions(pressure, temperature, humidity, turbidity);
    } // capture goes out of scope before assertions
    DOCTEST_CHECK(doctest::Approx(101325.f).epsilon(1e-6f) == pressure);
    DOCTEST_CHECK(doctest::Approx(300.f).epsilon(1e-6f) == temperature);
    DOCTEST_CHECK(doctest::Approx(0.5f).epsilon(1e-6f) == humidity);
    DOCTEST_CHECK(doctest::Approx(0.02f).epsilon(1e-6f) == turbidity);
}

TEST_CASE("SSolar-GOA spectral irradiance") {
    Context context_s;

    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(16, 7, 2023)));
    DOCTEST_CHECK_NOTHROW(context_s.setTime(make_Time(12, 0, 0)));

    SolarPosition sp(0, 36.93f, 3.33f, &context_s);
    sp.setAtmosphericConditions(87700.f, 298.f, 0.5f, 0.026f);

    DOCTEST_CHECK_NOTHROW(sp.calculateGlobalSolarSpectrum("global_test"));
    DOCTEST_CHECK_NOTHROW(sp.calculateDirectSolarSpectrum("direct_test"));
    DOCTEST_CHECK_NOTHROW(sp.calculateDiffuseSolarSpectrum("diffuse_test"));

    std::vector<vec2> global_spectrum, direct_spectrum, diffuse_spectrum;
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("global_test", global_spectrum));
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("direct_test", direct_spectrum));
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("diffuse_test", diffuse_spectrum));

    DOCTEST_CHECK(global_spectrum.size() == 2301);
    DOCTEST_CHECK(direct_spectrum.size() == 2301);
    DOCTEST_CHECK(diffuse_spectrum.size() == 2301);

    DOCTEST_CHECK(doctest::Approx(300.f).epsilon(0.1f) == global_spectrum.front().x);
    DOCTEST_CHECK(doctest::Approx(2600.f).epsilon(0.1f) == global_spectrum.back().x);

    for (const auto &point: global_spectrum) {
        DOCTEST_CHECK(point.y >= 0.f);
        DOCTEST_CHECK(std::isfinite(point.y));
    }

    // Verify integrated PAR flux is reasonable for clear sky at solar noon
    float par_flux = 0.f;
    for (size_t i = 1; i < global_spectrum.size(); ++i) {
        float wl = global_spectrum[i].x;
        if (wl >= 400.f && wl <= 700.f) {
            float dw = global_spectrum[i].x - global_spectrum[i - 1].x;
            float avg_irr = 0.5f * (global_spectrum[i].y + global_spectrum[i - 1].y);
            par_flux += avg_irr * dw;
        }
    }
    DOCTEST_CHECK(par_flux > 400.f);
    DOCTEST_CHECK(par_flux < 500.f);
}

TEST_CASE("SSolar-GOA ground albedo") {
    Context context_s;
    context_s.setDate(make_Date(16, 7, 2023));
    context_s.setTime(make_Time(12, 0, 0));
    SolarPosition sp(0, 36.93f, 3.33f, &context_s);
    sp.setAtmosphericConditions(87700.f, 298.f, 0.5f, 0.1f);

    // Spectra computed with a given ground albedo; the albedo is only set when it is non-negative, so a negative value leaves the default in place
    auto spectra_for_albedo = [&](float albedo, std::vector<vec2> &global, std::vector<vec2> &direct, std::vector<vec2> &diffuse) {
        if (albedo >= 0.f) {
            sp.setGroundAlbedo(albedo);
        }
        sp.calculateGlobalSolarSpectrum("global_albedo", 10.f);
        sp.calculateDirectSolarSpectrum("direct_albedo", 10.f);
        sp.calculateDiffuseSolarSpectrum("diffuse_albedo", 10.f);
        context_s.getGlobalData("global_albedo", global);
        context_s.getGlobalData("direct_albedo", direct);
        context_s.getGlobalData("diffuse_albedo", diffuse);
    };

    SUBCASE("default albedo is 0.2") {
        DOCTEST_CHECK(sp.getGroundAlbedo() == 0.2f);
        std::vector<vec2> global_default, direct_default, diffuse_default, global_set, direct_set, diffuse_set;
        spectra_for_albedo(-1.f, global_default, direct_default, diffuse_default);
        spectra_for_albedo(0.2f, global_set, direct_set, diffuse_set);
        DOCTEST_CHECK(global_default == global_set);
        DOCTEST_CHECK(diffuse_default == diffuse_set);
    }

    SUBCASE("albedo amplifies global and diffuse irradiance but not direct") {
        std::vector<vec2> global_0, direct_0, diffuse_0, global_3, direct_3, diffuse_3, global_6, direct_6, diffuse_6;
        spectra_for_albedo(0.f, global_0, direct_0, diffuse_0);
        spectra_for_albedo(0.3f, global_3, direct_3, diffuse_3);
        spectra_for_albedo(0.6f, global_6, direct_6, diffuse_6);
        DOCTEST_CHECK(sp.getGroundAlbedo() == 0.6f);
        DOCTEST_REQUIRE(global_0.size() == global_6.size());

        DOCTEST_CHECK(direct_0 == direct_6);
        for (size_t i = 0; i < global_0.size(); i++) {
            if (global_0.at(i).y <= 0.f) {
                continue;
            }
            DOCTEST_CHECK(global_6.at(i).y > global_0.at(i).y);
            DOCTEST_CHECK(diffuse_6.at(i).y > diffuse_0.at(i).y);
            // Multiple reflection scales global irradiance by 1/(1 - albedo*S), so its reciprocal is linear in albedo
            const float reciprocal_midpoint = 0.5f * (1.f / global_0.at(i).y + 1.f / global_6.at(i).y);
            DOCTEST_CHECK(std::fabs(reciprocal_midpoint * global_3.at(i).y - 1.f) < 1e-4f);
        }

        // The atmosphere's spherical albedo is largest at short wavelengths, so the ground albedo matters most there
        const size_t index_400nm = 10;
        const size_t index_1600nm = 130;
        DOCTEST_REQUIRE(std::fabs(global_0.at(index_400nm).x - 400.f) < 1.f);
        DOCTEST_REQUIRE(std::fabs(global_0.at(index_1600nm).x - 1600.f) < 1.f);
        DOCTEST_CHECK(global_6.at(index_400nm).y / global_0.at(index_400nm).y > global_6.at(index_1600nm).y / global_0.at(index_1600nm).y);
    }

    SUBCASE("invalid albedo throws") {
        DOCTEST_CHECK_THROWS_AS(sp.setGroundAlbedo(-0.1f), std::runtime_error);
        DOCTEST_CHECK_THROWS_AS(sp.setGroundAlbedo(1.1f), std::runtime_error);
        DOCTEST_CHECK_NOTHROW(sp.setGroundAlbedo(0.f));
        DOCTEST_CHECK_NOTHROW(sp.setGroundAlbedo(1.f));

        // A value written directly to global data is validated when it is read
        context_s.setGlobalData("atmosphere_ground_albedo", 1.5f);
        float albedo;
        DOCTEST_CHECK_THROWS_AS(albedo = sp.getGroundAlbedo(), std::runtime_error);
    }
}

// Spherical albedo S of the atmosphere recovered from the public API: without cloud calibration the global irradiance scales as 1/(1 - albedo*S), so S = (1 - E(0)/E(albedo))/albedo. Spectra are at 1 nm, so the
// irradiance at wavelength w nm is element w - 300.
static float sphericalAlbedoFromGroundResponse(SolarPosition &sp, Context &context, int wavelength_nm) {
    const float albedo = 0.8f;
    std::vector<vec2> global_black, global_bright;
    sp.setGroundAlbedo(0.f);
    sp.calculateGlobalSolarSpectrum("spherical_albedo_black");
    context.getGlobalData("spherical_albedo_black", global_black);
    sp.setGroundAlbedo(albedo);
    sp.calculateGlobalSolarSpectrum("spherical_albedo_bright");
    context.getGlobalData("spherical_albedo_bright", global_bright);
    const size_t i = wavelength_nm - 300;
    DOCTEST_REQUIRE(std::fabs(global_black.at(i).x - float(wavelength_nm)) < 0.01f);
    return (1.f - global_black.at(i).y / global_bright.at(i).y) / albedo;
}

TEST_CASE("SSolar-GOA Rayleigh spherical albedo") {
    // With no aerosol, the spherical albedo must be the closed form of 6S subroutine CSALBR (6S User Guide v3, Part 2, p. 96), S = [3 tau - 4 E3(tau) + 6 E4(tau)]/(4 + 3 tau), at the Rayleigh optical depth tau of
    // Cachorro et al. (2022) Eq. 7 at sea-level pressure. Expected values: tau = 0.35839, 0.22023, 0.09680, 0.01511 at 400, 450, 550, 870 nm. 6S itself gives S = 0.2337 at tau = 0.3610 and 0.0822 at tau = 0.0975.
    // The printed Eq. 19 of Cachorro et al., tau/(2 + tau)*(1 - exp(-2 tau)), gives about a tenth of these values (0.0081 at 550 nm).
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setSunDirection(make_SphericalCoord(deg2rad(60.f), 0.f));
    sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.f);

    const std::vector<std::pair<int, float>> expected = {{400, 0.23247f}, {450, 0.16158f}, {550, 0.08167f}, {870, 0.01453f}};
    for (const auto &wavelength_and_value: expected) {
        const int wavelength_nm = wavelength_and_value.first;
        const float S_expected = wavelength_and_value.second;
        const float S = sphericalAlbedoFromGroundResponse(sp, context_s, wavelength_nm);
        DOCTEST_INFO("wavelength " << wavelength_nm << " nm: S = " << S << ", expected " << S_expected);
        DOCTEST_CHECK(std::fabs(S / S_expected - 1.f) < 0.005f);
    }

    // The same result stated as the ground-albedo sensitivity: at 450 nm a ground of albedo 0.8 raises the global irradiance by 0.8 S/(1 - 0.8 S) = 14.8%
    const float S_450 = sphericalAlbedoFromGroundResponse(sp, context_s, 450);
    const float increase = 0.8f * S_450 / (1.f - 0.8f * S_450);
    DOCTEST_CHECK(increase > 0.146f);
    DOCTEST_CHECK(increase < 0.151f);
}

TEST_CASE("SSolar-GOA spherical albedo of the aerosol-Rayleigh mixture") {
    // S = S_R + [S_2s(tau_t, w_mix, g_mix) - S_2s(tau_R, 1, 0)]: the CSALBR Rayleigh albedo plus the change in the two-stream diffuse-illumination reflectance (Heng & Kitzmann 2017 Eq. 1, coupling coefficients of
    // Heng et al. 2014 Eq. 72, transmission function 2 E3) when the aerosol is added to the Rayleigh layer. Aerosol: single-scattering albedo 0.893, asymmetry 0.634 (6S continental at 550 nm), Angstrom exponent 1.3. Expected
    // values from an independent evaluation of these formulas.
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setSunDirection(make_SphericalCoord(deg2rad(60.f), 0.f));
    const float angstrom_factor_550 = std::pow(0.55f, 1.3f); // turbidity (optical depth at 1 um) per unit optical depth at 550 nm

    SUBCASE("high aerosol load, blue: aerosol shields the Rayleigh albedo") {
        // 450 nm, optical depth 0.5 at 550 nm (tau_R = 0.22023, tau_a = 0.64903). Adding the separate Rayleigh and aerosol albedos (Cachorro et al. Eq. 18) would give 0.30423; 6S shows that this overstates the
        // albedo of the mixture at short wavelengths and high aerosol load (by 0.073 at 450 nm for 6S's continental aerosol).
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.5f * angstrom_factor_550);
        const float S = sphericalAlbedoFromGroundResponse(sp, context_s, 450);
        DOCTEST_INFO("S = " << S);
        DOCTEST_CHECK(std::fabs(S - 0.22877f) < 0.001f);
        DOCTEST_CHECK(S < 0.30423f - 0.04f);
    }

    SUBCASE("low aerosol load, near infrared: close to the sum of the separate albedos") {
        // 870 nm, optical depth 0.05 at 550 nm (tau_R = 0.01511, tau_a = 0.02755); the sum of the separate albedos is 0.02329
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.05f * angstrom_factor_550);
        const float S = sphericalAlbedoFromGroundResponse(sp, context_s, 870);
        DOCTEST_INFO("S = " << S);
        DOCTEST_CHECK(std::fabs(S - 0.02267f) < 0.0002f);
        DOCTEST_CHECK(std::fabs(S - 0.02329f) < 0.001f);
    }
}

TEST_CASE("SSolar-GOA mixed-layer scattering transmittance") {
    // The total (direct + diffuse) scattering transmittance of the Rayleigh-aerosol layer is Eq. 14 of Cachorro et al. (2022), T = (1 - r0^2) exp(-k tau_t m)/(1 - r0^2 exp(-2 k tau_t m)), with Eq. 15
    // r0 = (k - 1 + w)/(k + 1 - w) and k = sqrt((1 - w)(1 - w g)) (the square root is missing from the printed Eq. 15), the single-scattering albedo w of Eq. 16, the scattering-weighted asymmetry parameter
    // g = w_a tau_a g_a/(tau_R + w_a tau_a), and the relative air mass m of Kasten & Young (1989). With a black ground, gas absorption cancels from the ratio of global horizontal to direct horizontal
    // irradiance, which is T/exp(-tau_t m). Evaluated here independently at 550 nm, aerosol optical depth 1 at 550 nm, solar zenith 60 degrees.
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setSunDirection(make_SphericalCoord(deg2rad(30.f), 0.f));
    const double beta = std::pow(0.55, 1.3);
    sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, float(beta));
    sp.setGroundAlbedo(0.f);

    sp.calculateGlobalSolarSpectrum("mixture_global");
    sp.calculateDirectSolarSpectrum("mixture_direct");
    std::vector<vec2> global, direct;
    context_s.getGlobalData("mixture_global", global);
    context_s.getGlobalData("mixture_direct", direct);
    const size_t i = 550 - 300;
    DOCTEST_REQUIRE(std::fabs(global.at(i).x - 550.f) < 0.01f);

    const double zenith_deg = rad2deg(sp.getSunZenith());
    DOCTEST_REQUIRE(std::fabs(zenith_deg - 60.0) < 0.01);
    const double wavelength_um = 0.55;
    const double tau_R = 1.0 / (117.2594 * std::pow(wavelength_um, 4) - 1.3215 * wavelength_um * wavelength_um + 0.00032 - 0.000076 / (wavelength_um * wavelength_um));
    const double tau_a = beta * std::pow(wavelength_um, -1.3);
    const double tau_t = tau_R + tau_a;
    const double w_a = 0.893, g_a = 0.634; // 6S continental aerosol at 550 nm
    const double w = (tau_R + w_a * tau_a) / tau_t;
    const double g = w_a * tau_a * g_a / (tau_R + w_a * tau_a);
    const double air_mass = 1.0 / (std::cos(deg2rad(zenith_deg)) + 0.50572 * std::pow(96.07995 - zenith_deg, -1.6364));
    const double k = std::sqrt((1.0 - w) * (1.0 - w * g));
    const double r0 = (k - 1.0 + w) / (k + 1.0 - w);
    const double T_mixture = (1.0 - r0 * r0) * std::exp(-k * tau_t * air_mass) / (1.0 - r0 * r0 * std::exp(-2.0 * k * tau_t * air_mass));
    const double ratio_expected = T_mixture / std::exp(-tau_t * air_mass);

    const double ratio = global.at(i).y / (std::cos(deg2rad(zenith_deg)) * direct.at(i).y);
    DOCTEST_INFO("global/direct horizontal = " << ratio << ", expected " << ratio_expected << " (T_mixture = " << T_mixture << ")");
    DOCTEST_CHECK(std::fabs(ratio / ratio_expected - 1.0) < 1e-3);
}

TEST_CASE("Sensor atmosphere spectra") {
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.1f);

    // Value of a stored spectrum at a wavelength of the native 1 nm grid
    auto value_at = [&](const std::string &label, int wavelength_nm) {
        std::vector<vec2> spectrum;
        context_s.getGlobalData(label.c_str(), spectrum);
        const vec2 &point = spectrum.at(wavelength_nm - 300);
        DOCTEST_REQUIRE(point.x == doctest::Approx(float(wavelength_nm)));
        return point.y;
    };
    // Direction toward a sensor at the given zenith, and azimuth measured from the sun's azimuth
    auto sensor_direction = [&](float view_zenith_deg, float relative_azimuth_deg) { return sphere2cart(make_SphericalCoord(deg2rad(90.f - view_zenith_deg), sp.getSunAzimuth() + deg2rad(relative_azimuth_deg))); };

    SUBCASE("outputs") {
        sp.setSunDirection(make_SphericalCoord(deg2rad(50.f), 0.f));
        sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1));
        for (const char *suffix: {"_path_radiance", "_adjacency_radiance", "_upward_direct_transmittance", "_upward_diffuse_transmittance", "_spherical_albedo", "_direct_irradiance", "_diffuse_irradiance", "_global_irradiance"}) {
            std::vector<vec2> spectrum;
            context_s.getGlobalData(("atm" + std::string(suffix)).c_str(), spectrum);
            DOCTEST_CHECK(spectrum.size() == 2301);
            for (const vec2 &point: spectrum) {
                DOCTEST_CHECK(std::isfinite(point.y));
                DOCTEST_CHECK(point.y >= 0.f);
            }
        }
        // Global irradiance is the diffuse plus the direct beam projected onto the horizontal
        const float mu_sun = cosf(sp.getSunZenith());
        for (int wavelength: {450, 865, 1600}) {
            DOCTEST_CHECK(std::fabs(value_at("atm_diffuse_irradiance", wavelength) + mu_sun * value_at("atm_direct_irradiance", wavelength) - value_at("atm_global_irradiance", wavelength)) <
                          1e-4f * value_at("atm_global_irradiance", wavelength));
        }
        // The upward transmittances are at most one and the atmosphere scatters blue more than near-infrared
        DOCTEST_CHECK(value_at("atm_upward_direct_transmittance", 865) + value_at("atm_upward_diffuse_transmittance", 865) <= 1.f);
        DOCTEST_CHECK(value_at("atm_path_radiance", 450) > value_at("atm_path_radiance", 865));
        DOCTEST_CHECK(value_at("atm_spherical_albedo", 450) > value_at("atm_spherical_albedo", 865));

        vec3 stored_direction;
        context_s.getGlobalData("atm_direction_to_sensor", stored_direction);
        DOCTEST_CHECK(stored_direction == make_vec3(0, 0, 1));

        std::vector<vec2> coarse;
        sp.calculateSensorAtmosphereSpectra("atm_coarse", make_vec3(0, 0, 1), 10.f);
        context_s.getGlobalData("atm_coarse_path_radiance", coarse);
        DOCTEST_CHECK(coarse.size() == 231);
    }

    SUBCASE("agreement with 6S") {
        // Reference values from direct 6S runs (6SV1.1 via Py6S; continental aerosol, AOD550 = 0.1*0.55^-1.3, water vapor 1.1759 cm and ozone 0.3316 atm-cm as derived by SolarPosition for this date,
        // location, temperature and humidity; Lambertian ground of reflectance 0.2) at a geometry between the look-up table's nodes
        sp.setSunDirection(make_SphericalCoord(deg2rad(90.f - 38.f), 0.f));
        sp.calculateSensorAtmosphereSpectra("atm", sensor_direction(35.f, 70.f));
        const float mu_sun = cosf(sp.getSunZenith());
        struct Reference {
            int wavelength_nm;
            float path_over_global; // pi * path radiance / global irradiance
            float upward_total; // upward direct plus diffuse transmittance
            float spherical_albedo;
            float direct_fraction; // direct share of the global irradiance
        };
        for (const Reference &reference: {Reference{470, 0.12351f, 0.83261f, 0.17479f, 0.66685f}, Reference{550, 0.06705f, 0.85636f, 0.12304f, 0.74256f}, Reference{865, 0.01701f, 0.94923f, 0.05119f, 0.86931f},
                                          Reference{1650, 0.00374f, 0.95281f, 0.01672f, 0.94381f}}) {
            const int wavelength = reference.wavelength_nm;
            const float global = value_at("atm_global_irradiance", wavelength);
            const float path_over_global = float(M_PI) * value_at("atm_path_radiance", wavelength) / global;
            const float upward_total = value_at("atm_upward_direct_transmittance", wavelength) + value_at("atm_upward_diffuse_transmittance", wavelength);
            const float direct_fraction = mu_sun * value_at("atm_direct_irradiance", wavelength) / global;
            DOCTEST_INFO("wavelength " << wavelength << " nm: path/global " << path_over_global << " (6S " << reference.path_over_global << "), upward " << upward_total << " (6S " << reference.upward_total << "), S "
                                       << value_at("atm_spherical_albedo", wavelength) << " (6S " << reference.spherical_albedo << "), direct fraction " << direct_fraction << " (6S " << reference.direct_fraction << ")");
            DOCTEST_CHECK(std::fabs(path_over_global / reference.path_over_global - 1.f) < 0.03f);
            DOCTEST_CHECK(std::fabs(upward_total / reference.upward_total - 1.f) < 0.01f);
            // The spherical albedo is small in the infrared, so it is bounded in absolute terms
            DOCTEST_CHECK(std::fabs(value_at("atm_spherical_albedo", wavelength) - reference.spherical_albedo) < 0.001f);
            DOCTEST_CHECK(std::fabs(direct_fraction / reference.direct_fraction - 1.f) < 0.01f);
        }
    }

    SUBCASE("path radiance rises with aerosol load") {
        // Regression check: a single-scattering model with a strongly forward-scattering aerosol predicted the opposite
        sp.setSunDirection(make_SphericalCoord(deg2rad(60.f), 0.f));
        std::vector<float> path_550;
        for (float turbidity: {0.02f, 0.1f, 0.3f}) {
            sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, turbidity);
            sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1));
            path_550.push_back(value_at("atm_path_radiance", 550));
        }
        DOCTEST_CHECK(path_550.at(1) > path_550.at(0));
        DOCTEST_CHECK(path_550.at(2) > path_550.at(1));
    }

    SUBCASE("ground albedo") {
        sp.setSunDirection(make_SphericalCoord(deg2rad(50.f), 0.f));
        const vec3 view = sensor_direction(30.f, 90.f);
        sp.setGroundAlbedo(0.05f);
        sp.calculateSensorAtmosphereSpectra("dark", view);
        sp.setGroundAlbedo(0.4f);
        sp.calculateSensorAtmosphereSpectra("bright", view);
        // The path radiance never reaches the ground; the illumination and the light reflected by the surroundings increase with ground albedo
        DOCTEST_CHECK(value_at("dark_path_radiance", 550) == value_at("bright_path_radiance", 550));
        DOCTEST_CHECK(value_at("bright_global_irradiance", 450) > value_at("dark_global_irradiance", 450));
        DOCTEST_CHECK(value_at("bright_adjacency_radiance", 550) > value_at("dark_adjacency_radiance", 550));
        DOCTEST_CHECK(value_at("dark_direct_irradiance", 550) == value_at("bright_direct_irradiance", 550));
    }

    SUBCASE("sensor azimuth mirror symmetry") {
        sp.setSunDirection(make_SphericalCoord(deg2rad(45.f), deg2rad(30.f)));
        sp.calculateSensorAtmosphereSpectra("left", sensor_direction(40.f, 70.f));
        sp.calculateSensorAtmosphereSpectra("right", sensor_direction(40.f, -70.f));
        sp.calculateSensorAtmosphereSpectra("forward", sensor_direction(40.f, 180.f));
        sp.calculateSensorAtmosphereSpectra("backward", sensor_direction(40.f, 0.f));
        DOCTEST_CHECK(value_at("left_path_radiance", 550) == doctest::Approx(value_at("right_path_radiance", 550)).epsilon(1e-4));
        // Path radiance depends on the scattering angle, so it differs between a sensor on the sun's side and one opposite it
        DOCTEST_CHECK(value_at("forward_path_radiance", 865) != value_at("backward_path_radiance", 865));
    }

    SUBCASE("sun-sensor reciprocity") {
        // Path reflectance pi*L/(mu_sun*E0) is unchanged when the sun and sensor zenith angles are exchanged in the same plane
        sp.setSunDirection(make_SphericalCoord(deg2rad(90.f - 20.f), 0.f));
        sp.calculateSensorAtmosphereSpectra("a", sensor_direction(50.f, 120.f));
        const float mu_a = cosf(sp.getSunZenith());
        sp.setSunDirection(make_SphericalCoord(deg2rad(90.f - 50.f), 0.f));
        sp.calculateSensorAtmosphereSpectra("b", sensor_direction(20.f, 120.f));
        const float mu_b = cosf(sp.getSunZenith());
        for (int wavelength: {450, 650, 865}) {
            const float reflectance_a = value_at("a_path_radiance", wavelength) / mu_a;
            const float reflectance_b = value_at("b_path_radiance", wavelength) / mu_b;
            DOCTEST_CHECK(std::fabs(reflectance_a / reflectance_b - 1.f) < 0.02f);
        }
    }

    SUBCASE("surface pressure") {
        sp.setSunDirection(make_SphericalCoord(deg2rad(50.f), 0.f));
        std::vector<float> path_450;
        for (float pressure_Pa: {80000.f, 101325.f, 104000.f}) {
            sp.setAtmosphericConditions(pressure_Pa, 293.f, 0.4f, 0.1f);
            DOCTEST_CHECK_NOTHROW(sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1)));
            path_450.push_back(value_at("atm_path_radiance", 450));
        }
        // More air above the ground scatters more light into the sensor; above sea level the table is extrapolated
        DOCTEST_CHECK(path_450.at(1) > path_450.at(0));
        DOCTEST_CHECK(path_450.at(2) > path_450.at(1));
    }

    SUBCASE("conditions outside the look-up table throw") {
        sp.setSunDirection(make_SphericalCoord(deg2rad(50.f), 0.f));
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorAtmosphereSpectra("atm", sensor_direction(65.f, 0.f)), std::runtime_error);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 0)), std::runtime_error);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1), 0.5f), std::runtime_error);
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.7f);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1)), std::runtime_error);
        sp.setAtmosphericConditions(106000.f, 293.f, 0.4f, 0.1f);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1)), std::runtime_error);
        sp.setAtmosphericConditions(65000.f, 293.f, 0.4f, 0.1f);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1)), std::runtime_error);
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.1f);
        sp.setSunDirection(make_SphericalCoord(deg2rad(15.f), 0.f));
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorAtmosphereSpectra("atm", make_vec3(0, 0, 1)), std::runtime_error);
    }
}

TEST_CASE("Sensor thermal atmosphere") {
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    // 293 K and 40% humidity give 1.176 g/cm^2 of precipitable water
    sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.1f);
    sp.setOzoneColumn(300.f);

    auto spectrum = [&](const std::string &label) {
        std::vector<vec2> values;
        context_s.getGlobalData(label.c_str(), values);
        return values;
    };
    auto spectrum_at = [&](const std::string &label, float wavelength_nm) { return interp1(spectrum(label), wavelength_nm); };
    // Trapezoidal integral of a spectrum between two wavelengths that are grid points or lie between them
    auto band_integral = [](const std::vector<vec2> &values, float wavelength_min, float wavelength_max, const std::function<float(float, float)> &integrand) {
        double integral = 0.0;
        for (size_t i = 0; i + 1 < values.size(); i++) {
            const float lower = std::max(values.at(i).x, wavelength_min);
            const float upper = std::min(values.at(i + 1).x, wavelength_max);
            if (upper <= lower) {
                continue;
            }
            const float slope = (values.at(i + 1).y - values.at(i).y) / (values.at(i + 1).x - values.at(i).x);
            integral += 0.5 * (integrand(lower, values.at(i).y + slope * (lower - values.at(i).x)) + integrand(upper, values.at(i).y + slope * (upper - values.at(i).x))) * (upper - lower);
        }
        return integral;
    };

    SUBCASE("outputs") {
        sp.calculateSensorThermalAtmosphere("atm", make_vec3(0, 0, 1));
        for (const char *suffix: {"_thermal_transmittance", "_thermal_upwelling_radiance", "_thermal_downwelling_radiance"}) {
            const std::vector<vec2> values = spectrum("atm" + std::string(suffix));
            // 5 cm^-1 bins between 650 and 1820 cm^-1, at their center wavelengths
            DOCTEST_REQUIRE(values.size() == 234);
            DOCTEST_CHECK(std::fabs(values.front().x - 1e7f / 1817.5f) < 0.01f);
            DOCTEST_CHECK(std::fabs(values.back().x - 1e7f / 652.5f) < 0.01f);
            for (size_t i = 0; i < values.size(); i++) {
                DOCTEST_CHECK(std::isfinite(values.at(i).y));
                DOCTEST_CHECK(values.at(i).y >= 0.f);
                if (i > 0) {
                    DOCTEST_CHECK(values.at(i).x > values.at(i - 1).x);
                }
            }
        }
        for (const vec2 &point: spectrum("atm_thermal_transmittance")) {
            DOCTEST_CHECK(point.y <= 1.f);
        }
        float water_vapor;
        context_s.getGlobalData("atm_thermal_water_vapor_cm", water_vapor);
        DOCTEST_CHECK(std::fabs(water_vapor - 1.1759f) < 1e-3f);
        vec3 direction;
        context_s.getGlobalData("atm_direction_to_sensor", direction);
        DOCTEST_CHECK(direction == make_vec3(0, 0, 1));
        // The 11 um window is clear; the 6.3 um water vapor band and the 15 um carbon dioxide band are opaque
        DOCTEST_CHECK(spectrum_at("atm_thermal_transmittance", 11000.f) > 0.8f);
        DOCTEST_CHECK(spectrum_at("atm_thermal_transmittance", 6300.f) < 1e-3f);
        DOCTEST_CHECK(spectrum_at("atm_thermal_transmittance", 15000.f) < 1e-3f);
    }

    SUBCASE("agreement with the Landsat TM6 fit") {
        // Jiménez-Muñoz and Sobrino (2003, Eq. 15) fitted the atmospheric functions of Landsat TM band 6 (10.4-12.5 um) to MODTRAN 3.5 simulations of radiosoundings. The two are compared as the brightness temperature
        // seen above the atmosphere over a 300 K blackbody, the quantity the fit was built to reproduce; the authors report errors below 1.5 K in retrieved surface temperature for this band.
        sp.calculateSensorThermalAtmosphere("atm", make_vec3(0, 0, 1));
        float water_vapor;
        context_s.getGlobalData("atm_thermal_water_vapor_cm", water_vapor);
        const double w = water_vapor;
        const double tm6_psi1 = 0.14714 * w * w - 0.15583 * w + 1.1234;
        const double tm6_psi2 = -1.1836 * w * w - 0.37607 * w - 0.52894;
        const double tm6_psi3 = -0.04554 * w * w + 1.8719 * w - 0.39071;
        const double tm6_transmittance = 1.0 / tm6_psi1;
        const double tm6_upwelling_per_um = -tm6_transmittance * (tm6_psi2 + tm6_psi3);
        const double effective_wavelength_um = 11.457;
        auto planck_per_um = [&](double temperature) { return 1.19104e8 / (std::pow(effective_wavelength_um, 5) * (std::exp(1.43877e4 / (effective_wavelength_um * temperature)) - 1.0)); };
        const double tm6_temperature = 1.43877e4 / (effective_wavelength_um * std::log(1.0 + 1.19104e8 / (std::pow(effective_wavelength_um, 5) * (tm6_transmittance * planck_per_um(300.0) + tm6_upwelling_per_um))));

        const float band_min = 10400.f;
        const float band_max = 12500.f;
        const std::vector<vec2> transmittance = spectrum("atm_thermal_transmittance");
        const double band_radiance = band_integral(transmittance, band_min, band_max, [](float wavelength, float value) { return value * blackbodySpectralRadiance(wavelength, 300.f); }) +
                                     band_integral(spectrum("atm_thermal_upwelling_radiance"), band_min, band_max, [](float, float value) { return value; });
        double low = 250.0, high = 310.0;
        for (int iteration = 0; iteration < 60; iteration++) {
            const double middle = 0.5 * (low + high);
            (band_integral(transmittance, band_min, band_max, [&](float wavelength, float) { return blackbodySpectralRadiance(wavelength, float(middle)); }) < band_radiance ? low : high) = middle;
        }
        const double helios_temperature = 0.5 * (low + high);
        DOCTEST_INFO("brightness temperature above the atmosphere: " << helios_temperature << " K (TM6 fit " << tm6_temperature << " K)");
        DOCTEST_CHECK(std::fabs(helios_temperature - tm6_temperature) < 1.5);
        DOCTEST_CHECK(helios_temperature < 300.0);
    }

    SUBCASE("view angle, humidity and ozone") {
        sp.calculateSensorThermalAtmosphere("nadir", make_vec3(0, 0, 1));
        sp.calculateSensorThermalAtmosphere("oblique", sphere2cart(make_SphericalCoord(deg2rad(90.f - 55.f), 0.f)));
        // The stored water vapor is the vertical column, whatever the view
        float nadir_water, oblique_water;
        context_s.getGlobalData("nadir_thermal_water_vapor_cm", nadir_water);
        context_s.getGlobalData("oblique_thermal_water_vapor_cm", oblique_water);
        DOCTEST_CHECK(oblique_water == nadir_water);
        // A longer path transmits less and emits more; the sky radiance reaching the ground does not depend on where the sensor is
        DOCTEST_CHECK(spectrum_at("oblique_thermal_transmittance", 11000.f) < spectrum_at("nadir_thermal_transmittance", 11000.f));
        DOCTEST_CHECK(spectrum_at("oblique_thermal_upwelling_radiance", 11000.f) > spectrum_at("nadir_thermal_upwelling_radiance", 11000.f));
        const std::vector<vec2> nadir_downwelling = spectrum("nadir_thermal_downwelling_radiance");
        const std::vector<vec2> oblique_downwelling = spectrum("oblique_thermal_downwelling_radiance");
        for (size_t i = 0; i < nadir_downwelling.size(); i++) {
            DOCTEST_CHECK(oblique_downwelling.at(i).y == nadir_downwelling.at(i).y);
        }

        // More water vapor absorbs more in the window
        sp.setAtmosphericConditions(101325.f, 293.f, 0.8f, 0.1f);
        sp.calculateSensorThermalAtmosphere("humid", make_vec3(0, 0, 1));
        DOCTEST_CHECK(spectrum_at("humid_thermal_transmittance", 11000.f) < spectrum_at("nadir_thermal_transmittance", 11000.f));

        // Ozone absorbs in its 9.6 um band but not in the 11 um window
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.1f);
        sp.setOzoneColumn(450.f);
        sp.calculateSensorThermalAtmosphere("ozone", make_vec3(0, 0, 1));
        DOCTEST_CHECK(spectrum_at("ozone_thermal_transmittance", 9600.f) < 0.9f * spectrum_at("nadir_thermal_transmittance", 9600.f));
        DOCTEST_CHECK(std::fabs(spectrum_at("ozone_thermal_transmittance", 11000.f) / spectrum_at("nadir_thermal_transmittance", 11000.f) - 1.f) < 1e-3f);

        // The thermal atmosphere does not depend on the sun
        sp.setSunDirection(make_SphericalCoord(deg2rad(-20.f), 0.f));
        DOCTEST_CHECK_NOTHROW(sp.calculateSensorThermalAtmosphere("night", make_vec3(0, 0, 1)));
    }

    SUBCASE("sky flux") {
        sp.calculateSensorThermalAtmosphere("nadir", make_vec3(0, 0, 1));
        const std::vector<vec2> downwelling = spectrum("nadir_thermal_downwelling_radiance");

        // The irradiance is pi times the integral of the hemispherical radiance over the band
        const float window_flux = sp.getThermalSkyFlux(10500.f, 11500.f);
        const double window_integral = band_integral(downwelling, 10500.f, 11500.f, [](float, float value) { return value; });
        DOCTEST_CHECK(std::fabs(window_flux / float(M_PI * window_integral) - 1.f) < 1e-5f);
        // Adjacent bands add up
        DOCTEST_CHECK(std::fabs(sp.getThermalSkyFlux(8000.f, 14000.f) / (sp.getThermalSkyFlux(8000.f, 10750.f) + sp.getThermalSkyFlux(10750.f, 14000.f)) - 1.f) < 1e-5f);
        // Microbolometer (7.5-13.5 um), GOES ABI band 8 (5.77-6.6 um) and MODIS band 36 (14.085-14.385 um) bands lie within the table
        DOCTEST_CHECK(sp.getThermalSkyFlux(7500.f, 13500.f) > window_flux);
        DOCTEST_CHECK(sp.getThermalSkyFlux(5770.f, 6600.f) > 0.f);
        DOCTEST_CHECK(sp.getThermalSkyFlux(14085.f, 14385.f) > 0.f);

        // A clear sky emits less in the window than a blackbody at the near-surface air temperature, and less than the broadband sky longwave flux; in the opaque 6.3 um water vapor band it emits almost as that blackbody
        const float air_temperature = 293.f;
        const float stefan_boltzmann = 5.670374419e-8f;
        DOCTEST_CHECK(window_flux > 0.f);
        DOCTEST_CHECK(window_flux < stefan_boltzmann * powf(air_temperature, 4) * blackbodyBandFraction(10500.f, 11500.f, air_temperature));
        DOCTEST_CHECK(sp.getThermalSkyFlux(7500.f, 13500.f) < sp.getAmbientLongwaveFlux());
        const float opaque_flux = sp.getThermalSkyFlux(6200.f, 6400.f);
        const float opaque_blackbody = stefan_boltzmann * powf(air_temperature, 4) * blackbodyBandFraction(6200.f, 6400.f, air_temperature);
        DOCTEST_CHECK(opaque_flux / opaque_blackbody > 0.9f);
        DOCTEST_CHECK(opaque_flux / opaque_blackbody < 1.f);

        float flux;
        {
            capture_cerr cerr_buffer;
            // 3.9 um is outside the table (reflected sunlight is not modeled there), as is a band reaching beyond 15326 nm
            DOCTEST_CHECK_THROWS_AS(flux = sp.getThermalSkyFlux(3800.f, 4000.f), std::runtime_error);
            DOCTEST_CHECK_THROWS_AS(flux = sp.getThermalSkyFlux(14000.f, 15500.f), std::runtime_error);
            DOCTEST_CHECK_THROWS_AS(flux = sp.getThermalSkyFlux(11500.f, 10500.f), std::runtime_error);
        }
    }

    SUBCASE("conditions outside the table throw") {
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorThermalAtmosphere("atm", sphere2cart(make_SphericalCoord(deg2rad(90.f - 65.f), 0.f))), std::runtime_error);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorThermalAtmosphere("atm", make_vec3(0, 0, 0)), std::runtime_error);
        // Cold, dry air holds less than 0.1 g/cm^2 of water vapor
        sp.setAtmosphericConditions(101325.f, 250.f, 0.2f, 0.1f);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorThermalAtmosphere("atm", make_vec3(0, 0, 1)), std::runtime_error);
        // Colder than the table's coldest atmosphere at sea level
        sp.setAtmosphericConditions(101325.f, 240.f, 1.f, 0.1f);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorThermalAtmosphere("atm", make_vec3(0, 0, 1)), std::runtime_error);
        // Surface pressure of a site above 3 km
        sp.setAtmosphericConditions(60000.f, 280.f, 0.5f, 0.1f);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorThermalAtmosphere("atm", make_vec3(0, 0, 1)), std::runtime_error);
        // Ozone column below the table
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.1f);
        sp.setOzoneColumn(100.f);
        DOCTEST_CHECK_THROWS_AS(sp.calculateSensorThermalAtmosphere("atm", make_vec3(0, 0, 1)), std::runtime_error);
        float flux;
        DOCTEST_CHECK_THROWS_AS(flux = sp.getThermalSkyFlux(10500.f, 11500.f), std::runtime_error);
    }
}

TEST_CASE("SSolar-GOA spectral resolution") {
    Context context_s;

    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(16, 7, 2023)));
    DOCTEST_CHECK_NOTHROW(context_s.setTime(make_Time(12, 0, 0)));

    SolarPosition sp(0, 36.93f, 3.33f, &context_s);
    sp.setAtmosphericConditions(87700.f, 298.f, 0.5f, 0.026f);

    // Test default 1 nm resolution
    DOCTEST_CHECK_NOTHROW(sp.calculateGlobalSolarSpectrum("res_1nm", 1.0f));
    std::vector<vec2> spectrum_1nm;
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("res_1nm", spectrum_1nm));
    DOCTEST_CHECK(spectrum_1nm.size() == 2301);

    // Test 10 nm resolution
    DOCTEST_CHECK_NOTHROW(sp.calculateGlobalSolarSpectrum("res_10nm", 10.0f));
    std::vector<vec2> spectrum_10nm;
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("res_10nm", spectrum_10nm));
    DOCTEST_CHECK(spectrum_10nm.size() == 231); // (2600-300)/10 + 1 = 231

    // Test 50 nm resolution
    DOCTEST_CHECK_NOTHROW(sp.calculateGlobalSolarSpectrum("res_50nm", 50.0f));
    std::vector<vec2> spectrum_50nm;
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("res_50nm", spectrum_50nm));
    DOCTEST_CHECK(spectrum_50nm.size() == 47); // (2600-300)/50 + 1 = 47

    // Verify wavelength spacing
    DOCTEST_CHECK(doctest::Approx(300.f).epsilon(0.5f) == spectrum_10nm.front().x);
    DOCTEST_CHECK(doctest::Approx(310.f).epsilon(0.5f) == spectrum_10nm[1].x);

    // Verify irradiance values are still reasonable
    for (const auto &point: spectrum_10nm) {
        DOCTEST_CHECK(point.y >= 0.f);
        DOCTEST_CHECK(std::isfinite(point.y));
    }
}

TEST_CASE("SSolar-GOA aerosol does not raise global irradiance over a black ground") {
    // Over a black ground, adding absorbing, forward-scattering aerosol to the atmosphere can only reduce the global irradiance. A coupling term in earlier versions made the global irradiance grow with the
    // aerosol load at low sun, to above the extraterrestrial irradiance on a horizontal surface (e.g. by a factor of 2 at 550 nm with an aerosol optical depth of 1 at 550 nm and a solar zenith of 75 degrees).
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    for (float solar_zenith_deg: {30.f, 75.f}) {
        sp.setSunDirection(make_SphericalCoord(deg2rad(90.f - solar_zenith_deg), 0.f));
        sp.setGroundAlbedo(0.f);
        std::vector<vec2> global_clean, global_hazy;
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.f);
        sp.calculateGlobalSolarSpectrum("global_clean");
        context_s.getGlobalData("global_clean", global_clean);
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, std::pow(0.55f, 1.3f));
        sp.calculateGlobalSolarSpectrum("global_hazy");
        context_s.getGlobalData("global_hazy", global_hazy);
        DOCTEST_REQUIRE(global_clean.size() == global_hazy.size());
        size_t increases = 0;
        for (size_t i = 0; i < global_clean.size(); ++i) {
            if (global_hazy[i].y > global_clean[i].y * (1.f + 1e-5f)) {
                ++increases;
            }
        }
        DOCTEST_INFO("solar zenith " << solar_zenith_deg << " degrees");
        DOCTEST_CHECK(increases == 0);
    }
}

TEST_CASE("SSolar-GOA spectral irradiance with the sun near the horizon") {
    // The gas transmittance tables extend to the air mass of Kasten & Young (1989) at the horizon (37.9), so at standard sea-level pressure (up to about 1015 hPa) and ordinary water vapor and ozone columns the spectra can be computed for any sun above the horizon
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, 0.1f);
    for (float elevation_deg: {10.f, 3.f, 0.5f}) {
        sp.setSunDirection(make_SphericalCoord(deg2rad(elevation_deg), 0.f));
        DOCTEST_CHECK_NOTHROW(sp.calculateGlobalSolarSpectrum("horizon_global"));
        DOCTEST_CHECK_NOTHROW(sp.calculateDirectSolarSpectrum("horizon_direct"));
        DOCTEST_CHECK_NOTHROW(sp.calculateDiffuseSolarSpectrum("horizon_diffuse"));
        std::vector<vec2> global, direct, diffuse;
        context_s.getGlobalData("horizon_global", global);
        context_s.getGlobalData("horizon_direct", direct);
        context_s.getGlobalData("horizon_diffuse", diffuse);
        bool all_valid = true;
        for (size_t i = 0; i < global.size(); ++i) {
            all_valid = all_valid && std::isfinite(global[i].y) && direct[i].y >= 0.f && diffuse[i].y >= 0.f && global[i].y >= diffuse[i].y;
        }
        DOCTEST_INFO("solar elevation " << elevation_deg << " degrees");
        DOCTEST_CHECK(all_valid);
    }
    // At and below the horizon the spectra are undefined
    sp.setSunDirection(make_SphericalCoord(deg2rad(-1.f), 0.f));
    DOCTEST_CHECK_THROWS_AS(sp.calculateGlobalSolarSpectrum("below_horizon"), std::runtime_error);
}

TEST_CASE("Spectral solar irradiance against 6S") {
    // Reference values from the 6S radiative transfer code (6SV1.1 via Py6S), generated by plugins/solarposition/utilities/generate_ssolar_reference.py: continental aerosol, sea level, solar zenith 30 degrees,
    // ground albedo 0.2, the water vapor (1.1759 cm) and ozone (317.7 DU) columns this model derives for the conditions below, day of year 172. 6S transmittances are multiplied by this model's
    // extraterrestrial irradiance, so the comparison excludes the difference between the two codes' extraterrestrial spectra. The model's aerosol has the single-scattering albedo and asymmetry
    // parameter of 6S's continental aerosol at 550 nm at all wavelengths and an Angstrom exponent of 1.3, whereas those of the continental aerosol vary with wavelength (its extinction implies an
    // Angstrom exponent of about 1.16), so the tolerances are tight only at low aerosol optical depth:
    //  - direct normal within 1% and global within 2% for aerosol optical depths at 550 nm of 0 and 0.05; global within 5% at 0.5
    //  - spherical albedo within 5% with no aerosol (pure Rayleigh). At 0.05 within 0.02 in absolute terms: with a ground albedo of 0.2, an error dS changes the global irradiance by
    //    0.2 dS/(1 - 0.2 S), at most 0.5% for S <= 0.3
    //  - diffuse horizontal within 5% with no aerosol. With aerosol (0.05 and 0.2) the difference is dominated by the aerosol-model difference: within 10% at 400-870 nm and 15% at 1240-1640 nm, where the
    //    model's fixed Angstrom exponent gives less aerosol than the continental model
    struct Reference {
        float aod550;
        int wavelength_nm;
        float direct_normal, global_horizontal, diffuse_horizontal, spherical_albedo;
    };
    const std::vector<Reference> references = {
            {0.f, 400, 1.0534f, 1.2003f, 0.28806f, 0.23673f},       {0.f, 450, 1.5613f, 1.5993f, 0.24714f, 0.16396f},      {0.f, 550, 1.5765f, 1.4707f, 0.10549f, 0.08272f},      {0.f, 670, 1.3874f, 1.2426f, 0.041066f, 0.04013f},
            {0.f, 870, 0.92824f, 0.81324f, 0.0093651f, 0.01471f},   {0.f, 1240, 0.45849f, 0.3982f, 0.0011293f, 0.00363f},  {0.f, 1640, 0.2227f, 0.19304f, 0.00017819f, 0.00119f}, {0.05f, 400, 0.97454f, 1.1801f, 0.33608f, 0.24072f},
            {0.05f, 450, 1.4555f, 1.5765f, 0.31605f, 0.17147f},     {0.05f, 550, 1.488f, 1.4547f, 0.16603f, 0.09337f},     {0.05f, 670, 1.324f, 1.2319f, 0.085248f, 0.05147f},    {0.05f, 870, 0.89696f, 0.80771f, 0.030913f, 0.02446f},
            {0.05f, 1240, 0.44801f, 0.39608f, 0.008101f, 0.01036f}, {0.05f, 1640, 0.21917f, 0.19233f, 0.00252f, 0.00525f}, {0.2f, 400, 0.77162f, 1.1199f, 0.45165f, 0.25139f},    {0.2f, 450, 1.1791f, 1.5078f, 0.48667f, 0.19094f},
            {0.2f, 550, 1.2514f, 1.4047f, 0.32102f, 0.12028f},      {0.2f, 670, 1.1508f, 1.1979f, 0.20128f, 0.07966f},     {0.2f, 870, 0.80933f, 0.78994f, 0.089043f, 0.04829f},  {0.2f, 1240, 0.41796f, 0.3894f, 0.027434f, 0.02732f},
            {0.2f, 1640, 0.20892f, 0.19011f, 0.00918f, 0.01574f},   {0.5f, 400, 0.48374f, 1.0029f, 0.58398f, 0.26766f},    {0.5f, 450, 0.77384f, 1.3699f, 0.69971f, 0.22004f},    {0.5f, 550, 0.885f, 1.3003f, 0.53386f, 0.15996f},
            {0.5f, 670, 0.86944f, 1.1252f, 0.37228f, 0.12122f},     {0.5f, 870, 0.65891f, 0.75172f, 0.18108f, 0.08418f},   {0.5f, 1240, 0.36377f, 0.37525f, 0.06021f, 0.05343f},  {0.5f, 1640, 0.18983f, 0.18547f, 0.021073f, 0.03239f},
    };

    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setSunDirection(make_SphericalCoord(deg2rad(60.f), 0.f));
    DOCTEST_REQUIRE(context_s.getJulianDate() == 172);

    for (float aod550: {0.f, 0.05f, 0.2f, 0.5f}) {
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, aod550 * std::pow(0.55f, 1.3f));
        sp.setGroundAlbedo(0.2f);
        sp.calculateDirectSolarSpectrum("reference_direct");
        sp.calculateGlobalSolarSpectrum("reference_global");
        sp.calculateDiffuseSolarSpectrum("reference_diffuse");
        std::vector<vec2> direct, global, diffuse;
        context_s.getGlobalData("reference_direct", direct);
        context_s.getGlobalData("reference_global", global);
        context_s.getGlobalData("reference_diffuse", diffuse);

        for (const Reference &reference: references) {
            if (reference.aod550 != aod550) {
                continue;
            }
            const size_t i = reference.wavelength_nm - 300;
            DOCTEST_REQUIRE(std::fabs(global.at(i).x - float(reference.wavelength_nm)) < 0.01f);
            const float direct_error = direct.at(i).y / reference.direct_normal - 1.f;
            const float global_error = global.at(i).y / reference.global_horizontal - 1.f;
            const float diffuse_error = diffuse.at(i).y / reference.diffuse_horizontal - 1.f;
            DOCTEST_INFO("AOD " << aod550 << ", " << reference.wavelength_nm << " nm: direct normal " << direct.at(i).y << " (6S " << reference.direct_normal << "), global " << global.at(i).y << " (6S " << reference.global_horizontal << "), diffuse "
                                << diffuse.at(i).y << " (6S " << reference.diffuse_horizontal << ")");
            if (aod550 == 0.f) {
                DOCTEST_CHECK(std::fabs(direct_error) < 0.01f);
                DOCTEST_CHECK(std::fabs(global_error) < 0.02f);
                DOCTEST_CHECK(std::fabs(diffuse_error) < 0.05f);
            } else if (aod550 < 0.1f) {
                DOCTEST_CHECK(std::fabs(direct_error) < 0.01f);
                DOCTEST_CHECK(std::fabs(global_error) < 0.02f);
                DOCTEST_CHECK(std::fabs(diffuse_error) < (reference.wavelength_nm <= 870 ? 0.10f : 0.15f));
            } else if (aod550 < 0.3f) {
                DOCTEST_CHECK(std::fabs(diffuse_error) < (reference.wavelength_nm <= 870 ? 0.10f : 0.15f));
            } else {
                DOCTEST_CHECK(std::fabs(global_error) < 0.05f);
            }
            if (aod550 < 0.1f) {
                const float S = sphericalAlbedoFromGroundResponse(sp, context_s, reference.wavelength_nm);
                sp.setGroundAlbedo(0.2f);
                DOCTEST_INFO("spherical albedo " << S << " (6S " << reference.spherical_albedo << ")");
                if (aod550 == 0.f) {
                    DOCTEST_CHECK(std::fabs(S / reference.spherical_albedo - 1.f) < 0.05f);
                } else {
                    DOCTEST_CHECK(std::fabs(S - reference.spherical_albedo) < 0.02f);
                }
            }
        }
    }
}

TEST_CASE("Spectral solar irradiance with zero relative humidity") {
    // Regression test: the precipitable water is derived from the dew point, which needs the logarithm of the relative humidity. setAtmosphericConditions() accepts a relative humidity of zero, for which
    // the precipitable water was NaN and the spectra silently became NaN; the spectral and thermal atmosphere methods now raise an error instead.
    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setSunDirection(make_SphericalCoord(deg2rad(60.f), 0.f));
    sp.setAtmosphericConditions(101325.f, 293.f, 0.f, 0.1f);
    DOCTEST_CHECK_THROWS_AS(sp.calculateGlobalSolarSpectrum("dry_global"), std::runtime_error);
    DOCTEST_CHECK_THROWS_AS(sp.calculateSensorThermalAtmosphere("dry_thermal", make_vec3(0, 0, 1)), std::runtime_error);
}

TEST_CASE("Spectral solar irradiance against 6S with the sun low") {
    // Reference values from 6S generated by plugins/solarposition/utilities/generate_ssolar_reference.py --solar-zenith 75 --aod550 0.2 0.5 (continental aerosol, sea level, ground albedo 0.2, the same
    // water vapor, ozone and day of year as "Spectral solar irradiance against 6S"). At a solar zenith of 75 degrees the two-stream solution is least accurate. At an aerosol optical depth at 550 nm of 0.2
    // the bounds are the envelope documented in SolarPosition.dox (diffuse -4.9% to +14%, global -5.4% to +6.9% of 6S from 400 to 2200 nm, excluding 1040 nm), rounded outward to allow for the
    // five-significant-digit reference values. At 0.5, where the aerosol-model difference dominates and the documented differences are -17% to +15% (global) and -17% to +18% (diffuse) at these wavelengths, they guard against gross errors
    // (within 25%).
    struct Reference {
        float aod550;
        int wavelength_nm;
        float direct_normal, global_horizontal, diffuse_horizontal;
    };
    const std::vector<Reference> references = {
            {0.2f, 400, 0.1398f, 0.21379f, 0.1776f},       {0.2f, 450, 0.33364f, 0.30483f, 0.21848f}, {0.2f, 550, 0.52014f, 0.29101f, 0.15639f}, {0.2f, 670, 0.63471f, 0.27553f, 0.11126f},
            {0.2f, 870, 0.56304f, 0.20128f, 0.05556f},     {0.2f, 1240, 0.33154f, 0.10422f, 0.018416f}, {0.2f, 1640, 0.17196f, 0.050856f, 0.0063481f}, {0.5f, 400, 0.029306f, 0.17089f, 0.16331f},
            {0.5f, 450, 0.081522f, 0.23916f, 0.21806f},    {0.5f, 550, 0.1632f, 0.22453f, 0.18229f},  {0.5f, 670, 0.24838f, 0.21445f, 0.15016f}, {0.5f, 870, 0.28297f, 0.16121f, 0.08797f},
            {0.5f, 1240, 0.20834f, 0.08755f, 0.033628f},   {0.5f, 1640, 0.12479f, 0.045003f, 0.012704f},
    };

    Context context_s;
    context_s.setDate(make_Date(21, 6, 2025));
    context_s.setTime(make_Time(11, 0, 0));
    SolarPosition sp(8, 38.55f, 121.76f, &context_s);
    sp.setSunDirection(make_SphericalCoord(deg2rad(15.f), 0.f));
    DOCTEST_REQUIRE(context_s.getJulianDate() == 172);

    for (float aod550: {0.2f, 0.5f}) {
        sp.setAtmosphericConditions(101325.f, 293.f, 0.4f, aod550 * std::pow(0.55f, 1.3f));
        sp.setGroundAlbedo(0.2f);
        sp.calculateGlobalSolarSpectrum("low_sun_global");
        sp.calculateDiffuseSolarSpectrum("low_sun_diffuse");
        std::vector<vec2> global, diffuse;
        context_s.getGlobalData("low_sun_global", global);
        context_s.getGlobalData("low_sun_diffuse", diffuse);

        for (const Reference &reference: references) {
            if (reference.aod550 != aod550) {
                continue;
            }
            const size_t i = reference.wavelength_nm - 300;
            DOCTEST_REQUIRE(std::fabs(global.at(i).x - float(reference.wavelength_nm)) < 0.01f);
            const float global_error = global.at(i).y / reference.global_horizontal - 1.f;
            const float diffuse_error = diffuse.at(i).y / reference.diffuse_horizontal - 1.f;
            DOCTEST_INFO("AOD " << aod550 << ", " << reference.wavelength_nm << " nm: global " << global.at(i).y << " (6S " << reference.global_horizontal << ", " << 100.f * global_error << "%), diffuse " << diffuse.at(i).y
                                << " (6S " << reference.diffuse_horizontal << ", " << 100.f * diffuse_error << "%)");
            if (aod550 < 0.3f) {
                DOCTEST_CHECK(global_error > -0.055f);
                DOCTEST_CHECK(global_error < 0.07f);
                DOCTEST_CHECK(diffuse_error > -0.05f);
                DOCTEST_CHECK(diffuse_error < 0.145f);
            } else {
                DOCTEST_CHECK(std::fabs(global_error) < 0.25f);
                DOCTEST_CHECK(std::fabs(diffuse_error) < 0.25f);
            }
        }
    }
}

int SolarPosition::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

// ===== Prague Sky Model Tests =====

TEST_CASE("SolarPosition ozone column") {
    Context context_s;
    // Values of the climatology table: 40-45N March 363.9 and April 360.1, 45-50N March 385.4, 40-45N December 324.5 and January 348.2, 75-80N July 313.5
    auto ozone_at = [&](float latitude, int day, int month) {
        context_s.setDate(make_Date(day, month, 2025));
        SolarPosition sp(8, latitude, 100.f, &context_s);
        return sp.getOzoneColumn();
    };
    SUBCASE("climatology interpolation") {
        // 16 March is the middle of March and 42.5N the center of the 40-45N band
        DOCTEST_CHECK(std::fabs(ozone_at(42.5f, 16, 3) - 363.9f) < 0.05f);
        // Halfway between the centers of the 40-45N and 45-50N bands
        DOCTEST_CHECK(std::fabs(ozone_at(45.f, 16, 3) - 0.5f * (363.9f + 385.4f)) < 0.05f);
        // 31 March (day 90) lies 15 days after the middle of March and 15.5 days before the middle of April
        DOCTEST_CHECK(std::fabs(ozone_at(42.5f, 31, 3) - (363.9f + 15.f / 30.5f * (360.1f - 363.9f))) < 0.05f);
        // 1 January lies between the middles of December (day 350) and January (day 16)
        DOCTEST_CHECK(std::fabs(ozone_at(42.5f, 1, 1) - (324.5f + 16.f / 31.f * (348.2f - 324.5f))) < 0.05f);
        // Next to the 80-85N band, which has no data, the value of the latitude's own band is used (16 July is the middle of July)
        DOCTEST_CHECK(std::fabs(ozone_at(78.f, 16, 7) - 313.5f) < 0.05f);
    }
    SUBCASE("months next to months without data") {
        // The 65-70N band has no data in December and January (November 320.7, February 407.1); dates in November and February use their own month rather than interpolating toward the missing one
        DOCTEST_CHECK(std::fabs(ozone_at(67.5f, 20, 11) - 320.7f) < 0.05f);
        DOCTEST_CHECK(std::fabs(ozone_at(67.5f, 5, 2) - 407.1f) < 0.05f);
        float ozone;
        DOCTEST_CHECK_THROWS_AS(ozone = ozone_at(67.5f, 20, 12), std::runtime_error);
    }
    SUBCASE("no climatological data") {
        float ozone;
        DOCTEST_CHECK_THROWS_AS(ozone = ozone_at(85.f, 16, 7), std::runtime_error);
        DOCTEST_CHECK_THROWS_AS(ozone = ozone_at(77.5f, 16, 1), std::runtime_error);
    }
    SUBCASE("user-specified ozone column") {
        context_s.setDate(make_Date(16, 6, 2025));
        SolarPosition sp(8, 85.f, 100.f, &context_s);
        sp.setOzoneColumn(310.f);
        DOCTEST_CHECK(sp.getOzoneColumn() == 310.f);
        DOCTEST_CHECK_THROWS_AS(sp.setOzoneColumn(0.f), std::runtime_error);
        DOCTEST_CHECK_THROWS_AS(sp.setOzoneColumn(-5.f), std::runtime_error);

        // The spectral model uses the ozone column: more ozone absorbs more in the Chappuis band near 600 nm
        SolarPosition sp_mid(8, 38.55f, 121.76f, &context_s);
        sp_mid.setAtmosphericConditions(101325.f, 293.f, 0.5f, 0.05f);
        sp_mid.setSunDirection(make_SphericalCoord(deg2rad(50.f), 0.f));
        std::vector<vec2> low, high;
        sp_mid.setOzoneColumn(250.f);
        sp_mid.calculateDirectSolarSpectrum("direct_low_ozone");
        sp_mid.setOzoneColumn(450.f);
        sp_mid.calculateDirectSolarSpectrum("direct_high_ozone");
        context_s.getGlobalData("direct_low_ozone", low);
        context_s.getGlobalData("direct_high_ozone", high);
        DOCTEST_CHECK(high.at(600 - 300).y < low.at(600 - 300).y);
        DOCTEST_CHECK(high.at(870 - 300).y == doctest::Approx(low.at(870 - 300).y).epsilon(1e-4));
    }
}

TEST_CASE("SolarPosition ozone column follows the Southern Hemisphere seasonal cycle") {
    // Regression test: the total ozone column is highest in spring in each hemisphere (September-November in the south). The previous empirical formula applied Northern Hemisphere coefficients at all latitudes,
    // which put the Southern Hemisphere maximum in autumn. Ozone absorbs in the Chappuis band near 600 nm but not at 870 nm, so with the sun and atmosphere held fixed, the ratio of direct irradiance at 600 nm to
    // that at 870 nm (which cancels the Earth-Sun distance) must be lower in September than in March at Sydney.
    Context context_s;
    SolarPosition sp(-10, -33.87f, -151.21f, &context_s); // Sydney; longitude is positive west in Helios
    sp.setAtmosphericConditions(101325.f, 293.f, 0.5f, 0.05f);
    sp.setSunDirection(make_SphericalCoord(deg2rad(50.f), 0.f));

    auto chappuis_ratio = [&](int day, int month) {
        context_s.setDate(make_Date(day, month, 2025));
        sp.calculateDirectSolarSpectrum("direct_ozone_test");
        std::vector<vec2> spectrum;
        context_s.getGlobalData("direct_ozone_test", spectrum);
        return spectrum.at(600 - 300).y / spectrum.at(870 - 300).y;
    };
    const float march_ratio = chappuis_ratio(15, 3);
    const float september_ratio = chappuis_ratio(15, 9);
    DOCTEST_INFO("600/870 nm direct irradiance ratio: March " << march_ratio << ", September " << september_ratio);
    DOCTEST_CHECK(september_ratio < march_ratio);
}

TEST_CASE("SolarPosition - Prague turbidity to visibility conversion") {
    // Regression test: the turbidity is Ångström's beta (aerosol optical depth at 1 um), converted to the aerosol optical depth at 550 nm with the Ångström exponent of 1.3 used by the other models. The Prague sky
    // model's viewing distances of 27.6, 59.4 and 131.8 km are its continental polluted, average and clean atmospheres (Wilkie et al. 2021, Fig. 4), whose OPAC aerosol optical depths at 550 nm are 0.327, 0.151 and
    // 0.064 (Hess et al. 1998). The previous conversion, 3.9/turbidity, treated the turbidity as the aerosol optical depth at 500 nm.
    const float aod550_per_beta = std::pow(0.55f, -1.3f);
    DOCTEST_CHECK(std::fabs(helios::PragueSkyModelInterface::turbidityToVisibility(0.327f / aod550_per_beta) - 27.6f) < 0.05f);
    DOCTEST_CHECK(std::fabs(helios::PragueSkyModelInterface::turbidityToVisibility(0.151f / aod550_per_beta) - 59.4f) < 0.05f);
    DOCTEST_CHECK(std::fabs(helios::PragueSkyModelInterface::turbidityToVisibility(0.064f / aod550_per_beta) - 131.8f) < 0.05f);
    // Visibility decreases monotonically with turbidity between the anchors
    DOCTEST_CHECK(helios::PragueSkyModelInterface::turbidityToVisibility(0.1f / aod550_per_beta) < helios::PragueSkyModelInterface::turbidityToVisibility(0.09f / aod550_per_beta));
    // Beyond the dataset's range the visibility is limited to its cleanest and haziest atmospheres
    DOCTEST_CHECK(helios::PragueSkyModelInterface::turbidityToVisibility(0.f) == 131.8f);
    DOCTEST_CHECK(helios::PragueSkyModelInterface::turbidityToVisibility(1.f) == 20.f);
}

TEST_CASE("SolarPosition - Prague model initialization") {
    Context context;
    SolarPosition solar(&context);

    DOCTEST_CHECK(!solar.isPragueSkyModelEnabled());
    DOCTEST_CHECK_NOTHROW(solar.enablePragueSkyModel());
    DOCTEST_CHECK(solar.isPragueSkyModelEnabled());
}

TEST_CASE("SolarPosition - Prague angular parameter fitting - clear sky") {
    Context context;
    SolarPosition solar(&context);
    solar.enablePragueSkyModel();
    solar.setSunDirection(make_SphericalCoord(0.5236f, 0.0f)); // 60° elevation
    solar.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.05f); // Clear sky turbidity
    solar.updatePragueSkyModel();

    std::vector<float> params;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("prague_sky_spectral_params", params));

    DOCTEST_CHECK(params.size() == 225 * 6);

    // Check 550nm parameters (index 38: (550-360)/5 = 38)
    int idx = 38 * 6;
    float wavelength = params[idx + 0];
    float L_zenith = params[idx + 1];
    float circ_str = params[idx + 2];
    float circ_width = params[idx + 3];
    float horiz_bright = params[idx + 4];
    float norm = params[idx + 5];

    DOCTEST_CHECK(wavelength == doctest::Approx(550.0f).epsilon(1.0f));
    DOCTEST_CHECK(L_zenith > 0.0f);
    DOCTEST_CHECK(L_zenith < 0.5f);
    DOCTEST_CHECK(circ_str >= 0.0f);
    DOCTEST_CHECK(circ_str <= 20.0f); // Max clamp value
    DOCTEST_CHECK(circ_width >= 5.0f);
    DOCTEST_CHECK(circ_width <= 60.0f);
    DOCTEST_CHECK(horiz_bright >= 1.0f);
    DOCTEST_CHECK(horiz_bright < 5.0f);
    DOCTEST_CHECK(norm > 0.0f);
    DOCTEST_CHECK(norm < 2.0f);

    // Verify global data validity flag
    int valid = 0;
    DOCTEST_CHECK_NOTHROW(context.getGlobalData("prague_sky_valid", valid));
    DOCTEST_CHECK(valid == 1);
}

TEST_CASE("SolarPosition - Prague lazy evaluation") {
    Context context;
    SolarPosition solar(&context);
    solar.enablePragueSkyModel();
    solar.setSunDirection(make_SphericalCoord(0.5236f, 0.0f)); // 60° elevation
    solar.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.1f);
    solar.updatePragueSkyModel();

    // Small turbidity change - should not need update
    solar.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.101f);
    DOCTEST_CHECK(!solar.pragueSkyModelNeedsUpdate(0.33f));

    // Large turbidity change - should need update
    solar.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.15f);
    DOCTEST_CHECK(solar.pragueSkyModelNeedsUpdate(0.33f));

    // Sun direction change
    solar.setSunDirection(make_SphericalCoord(0.6109f, 0.0f)); // 55° elevation (change > 5°)
    DOCTEST_CHECK(solar.pragueSkyModelNeedsUpdate(0.33f));

    // Albedo change
    solar.setSunDirection(make_SphericalCoord(0.5236f, 0.0f)); // Back to 60°
    solar.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.1f); // Reset turbidity
    solar.updatePragueSkyModel(); // Update with new sun direction
    DOCTEST_CHECK(solar.pragueSkyModelNeedsUpdate(0.5f)); // Albedo 0.33 -> 0.5
}

TEST_CASE("SolarPosition - Prague performance benchmark") {
    Context context;
    SolarPosition solar(&context);
    solar.enablePragueSkyModel();
    solar.setSunDirection(make_SphericalCoord(0.5236f, 0.0f)); // 60° elevation
    solar.setAtmosphericConditions(101325.f, 300.f, 0.5f, 0.1f);

    auto start = std::chrono::high_resolution_clock::now();
    solar.updatePragueSkyModel();
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    DOCTEST_CHECK(duration_ms.count() < 10000); // <10 seconds
}

TEST_CASE("SolarPosition - Prague error handling") {
    Context context;
    SolarPosition solar(&context);

    // Try to update without enabling
    DOCTEST_CHECK_THROWS_WITH_AS(solar.updatePragueSkyModel(), "ERROR (SolarPosition::updatePragueSkyModel): Prague model not enabled. Call enablePragueSkyModel() first.", std::runtime_error);

    // Check needs update returns false when not enabled
    DOCTEST_CHECK(!solar.pragueSkyModelNeedsUpdate(0.33f));
}
