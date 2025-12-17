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

TEST_CASE("SSolar-GOA validation against Python reference") {
    Context context_s;

    DOCTEST_CHECK_NOTHROW(context_s.setDate(make_Date(16, 7, 2023)));
    DOCTEST_CHECK_NOTHROW(context_s.setTime(make_Time(12, 0, 0)));

    SolarPosition sp(0, 36.93f, 3.33f, &context_s);
    sp.setAtmosphericConditions(87700.f, 298.f, 0.5f, 0.026f);

    DOCTEST_CHECK_NOTHROW(sp.calculateGlobalSolarSpectrum("validation"));

    std::vector<vec2> global_spectrum;
    DOCTEST_CHECK_NOTHROW(context_s.getGlobalData("validation", global_spectrum));

    // Compare against Python reference if available
    std::ifstream ref_file("plugins/solarposition/tests/validate_reference_global.txt");
    if (ref_file.is_open()) {
        std::string header_line;
        std::getline(ref_file, header_line); // Skip header

        std::vector<float> ref_wavelengths, ref_irradiances;
        std::string line;
        while (std::getline(ref_file, line)) {
            if (line.empty() || line[0] == '#')
                continue;

            std::istringstream iss(line);
            float wl, irr;
            if (iss >> wl >> irr) {
                ref_wavelengths.push_back(wl);
                ref_irradiances.push_back(irr);
            }
        }
        ref_file.close();

        DOCTEST_CHECK(ref_wavelengths.size() == 2301);

        float max_rel_error = 0.f;
        float sum_sq_error = 0.f;
        size_t n_compared = 0;

        for (size_t i = 0; i < std::min(global_spectrum.size(), ref_wavelengths.size()); ++i) {
            DOCTEST_CHECK(doctest::Approx(ref_wavelengths[i]).epsilon(1e-6f) == global_spectrum[i].x);

            float cpp_irr = global_spectrum[i].y;
            float ref_irr = ref_irradiances[i];
            float abs_error = std::fabs(cpp_irr - ref_irr);
            float rel_error = abs_error / (ref_irr + 1e-10f);

            max_rel_error = std::max(max_rel_error, rel_error);
            sum_sq_error += abs_error * abs_error;
            n_compared++;
        }

        float rms_error = std::sqrt(sum_sq_error / n_compared);

        DOCTEST_CHECK(max_rel_error < 0.01f);
        DOCTEST_CHECK(rms_error < 0.01f);

    } else {
        DOCTEST_WARN("Python reference file not found - run validate_detailed.py to enable detailed validation");
    }
}

int SolarPosition::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

// ===== Prague Sky Model Tests =====

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
