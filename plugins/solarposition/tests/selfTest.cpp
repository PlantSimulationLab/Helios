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

    float temperature = 290.f;
    float humidity = 0.5f;

    float LW;
    DOCTEST_CHECK_NOTHROW(LW = sp.getAmbientLongwaveFlux(temperature, humidity));

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

    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux(101325.f, 300.f, 0.5f, 0.02f));
    DOCTEST_CHECK(flux > 0.f);

    float diffuse_fraction;
    DOCTEST_CHECK_NOTHROW(diffuse_fraction = sp.getDiffuseFraction(101325.f, 300.f, 0.5f, 0.02f));
    DOCTEST_CHECK(diffuse_fraction >= 0.f);
    DOCTEST_CHECK(diffuse_fraction <= 1.f);

    float flux_par;
    DOCTEST_CHECK_NOTHROW(flux_par = sp.getSolarFluxPAR(101325.f, 300.f, 0.5f, 0.02f));
    DOCTEST_CHECK(flux_par > 0.f);

    float flux_nir;
    DOCTEST_CHECK_NOTHROW(flux_nir = sp.getSolarFluxNIR(101325.f, 300.f, 0.5f, 0.02f));
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
    DOCTEST_CHECK_NOTHROW(turbidity = sp.calibrateTurbidityFromTimeseries(label));
    DOCTEST_CHECK(turbidity > 0.f);
}

TEST_CASE("SolarPosition invalid lat/long") {
    Context context_s;

    capture_cerr cerr_buffer;
    SolarPosition sp_1(7, -100.f, 105.2369f, &context_s);
    DOCTEST_CHECK(cerr_buffer.has_output());

    cerr_buffer.clear();
    SolarPosition sp_2(7, 40.125f, -200.f, &context_s);
    DOCTEST_CHECK(cerr_buffer.has_output());
}

TEST_CASE("SolarPosition invalid solar angle") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    DOCTEST_CHECK_NOTHROW(sp.setSunDirection(make_SphericalCoord(0.75 * M_PI, M_PI / 2.f)));

    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux(101325.f, 300.f, 0.5f, 0.02f));
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

    DOCTEST_CHECK_NOTHROW(sp.enableCloudCalibration("net_radiation"));

    float flux;
    DOCTEST_CHECK_NOTHROW(flux = sp.getSolarFlux(101325.f, 300.f, 0.5f, 0.02f));
    DOCTEST_CHECK(flux > 0.f);

    float diffuse_fraction;
    DOCTEST_CHECK_NOTHROW(diffuse_fraction = sp.getDiffuseFraction(101325.f, 300.f, 0.5f, 0.02f));
    DOCTEST_CHECK(diffuse_fraction >= 0.f);
    DOCTEST_CHECK(diffuse_fraction <= 1.f);

    float flux_par;
    DOCTEST_CHECK_NOTHROW(flux_par = sp.getSolarFluxPAR(101325.f, 300.f, 0.5f, 0.02f));
    DOCTEST_CHECK(flux_par > 0.f);

    float flux_nir;
    DOCTEST_CHECK_NOTHROW(flux_nir = sp.getSolarFluxNIR(101325.f, 300.f, 0.5f, 0.02f));
    DOCTEST_CHECK(flux_nir > 0.f);

    DOCTEST_CHECK_NOTHROW(sp.disableCloudCalibration());

    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(sp.enableCloudCalibration("non_existent_timeseries"), std::runtime_error);
}

TEST_CASE("SolarPosition turbidity calculation") {
    Context context_s;
    SolarPosition sp(7, 40.125f, 105.2369f, &context_s);

    DOCTEST_CHECK_NOTHROW(context_s.loadTabularTimeseriesData("lib/testdata/cimis.csv", {"CIMIS"}, ","));

    float turbidity;
    DOCTEST_CHECK_NOTHROW(turbidity = sp.calibrateTurbidityFromTimeseries("net_radiation"));
    DOCTEST_CHECK(turbidity > 0.f);

    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(turbidity = sp.calibrateTurbidityFromTimeseries("non_existent_timeseries"), std::runtime_error);
}

int SolarPosition::selfTest(int argc, char** argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
