#include "SolarPosition.h"

using namespace helios;

int SolarPosition::selfTest() const {

    std::cout << "Running solar position model self-test..." << std::flush;
    int error_count = 0;

    float latitude;
    float longitude;
    Date date;
    Time time;
    int UTC;

    float errtol = 1e-6;

    Context context_s;

    //---- Sun Zenith/Azimuth Test for Boulder, CO ---- //
    latitude = 40.1250;
    longitude = 105.2369;
    date = make_Date(1, 1, 2000);
    time = make_Time(10, 30, 0);
    UTC = 7;

    float theta_actual = 29.49;
    float phi_actual = 154.18;

    context_s.setDate(date);
    context_s.setTime(time);

    SolarPosition solarposition_1(UTC, latitude, longitude, &context_s);

    float theta_s = solarposition_1.getSunElevation() * 180.f / M_PI;
    float phi_s = solarposition_1.getSunAzimuth() * 180.f / M_PI;

    if (fabs(theta_s - theta_actual) > 10 || fabs(phi_s - phi_actual) > 5) {
        error_count++;
        std::cout << "failed: verification test for known solar position does not agree with calculated result." << std::endl;
    }

    //---- Test of Gueymard Solar Flux Model ---- //
    latitude = 36.5289; // Billings, OK
    longitude = 97.4439;
    date = make_Date(5, 5, 2003);
    time = make_Time(9, 10, 0);
    UTC = 6;

    context_s.setDate(date);
    context_s.setTime(time);

    SolarPosition solarposition_2(UTC, latitude, longitude, &context_s);

    float pressure = 96660;
    float temperature = 290;
    float humidity = 0.5;
    float turbidity = 0.025;

    float Eb, Eb_PAR, Eb_NIR, fdiff;

    solarposition_2.GueymardSolarModel(pressure, temperature, humidity, turbidity, Eb_PAR, Eb_NIR, fdiff);
    Eb = Eb_PAR + Eb_NIR;

    //----- Test of Ambient Longwave Model ------ //
    SolarPosition solarposition_3(UTC, latitude, longitude, &context_s);

    temperature = 290;
    humidity = 0.5;

    float LW = solarposition_3.getAmbientLongwaveFlux(temperature, humidity);

    if (fabs(LW - 310.03192f) > errtol) {
        error_count++;
        std::cout << "failed: verification test for ambient longwave model does not agree with known result." << std::endl;
    }

    //---- New Tests for Sunrise and Sunset Times ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        context_s.setDate(make_Date(1, 1, 2023));
        Time sunrise = solar.getSunriseTime();
        Time sunset = solar.getSunsetTime();

        if (sunrise.hour == 0 && sunrise.minute == 0) {
            std::cout << "failed: sunrise time test." << std::endl;
            error_count++;
        }
        if (sunset.hour == 0 && sunset.minute == 0) {
            std::cout << "failed: sunset time test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Sun Direction Vector ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        context_s.setDate(make_Date(1, 1, 2023));
        vec3 sun_dir_vector = solar.getSunDirectionVector();

        if (sun_dir_vector.x == 0 && sun_dir_vector.y == 0 && sun_dir_vector.z == 0) {
            std::cout << "failed: sun direction vector test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Sun Direction Spherical ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        context_s.setDate(make_Date(1, 1, 2023));
        SphericalCoord sun_dir_spherical = solar.getSunDirectionSpherical();

        if (sun_dir_spherical.elevation <= 0 || sun_dir_spherical.azimuth <= 0) {
            std::cout << "failed: sun direction spherical test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Solar Flux ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        float flux = solar.getSolarFlux(101325, 300, 0.5, 0.02);

        if (flux <= 0) {
            std::cout << "failed: solar flux test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Diffuse Fraction ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        float diffuse_fraction = solar.getDiffuseFraction(101325, 300, 0.5, 0.02);

        if (diffuse_fraction < 0 || diffuse_fraction > 1) {
            std::cout << "failed: diffuse fraction test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Sun Elevation ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        float elevation = solar.getSunElevation();

        if (elevation < 0 || elevation > M_PI / 2) {
            std::cout << "failed: sun elevation test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Sun Zenith ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        float zenith = solar.getSunZenith();

        if (zenith < 0 || zenith > M_PI) {
            std::cout << "failed: sun zenith test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Sun Azimuth ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        float azimuth = solar.getSunAzimuth();

        if (azimuth < 0 || azimuth > 2 * M_PI) {
            std::cout << "failed: sun azimuth test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Solar Flux PAR ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        float flux_par = solar.getSolarFluxPAR(101325, 300, 0.5, 0.02);

        if (flux_par <= 0) {
            std::cout << "failed: solar flux PAR test." << std::endl;
            error_count++;
        }
    }

    //---- New Test for Solar Flux NIR ---- //
    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        float flux_nir = solar.getSolarFluxNIR(101325, 300, 0.5, 0.02);

        if (flux_nir <= 0) {
            std::cout << "failed: solar flux NIR test." << std::endl;
            error_count++;
        }
    }

    {
        SolarPosition solar(7, 40.125, 105.2369, &context_s);
        std::string timeseries_label = "test_flux_timeseries";

        // Check if the timeseries exists
        if (!context_s.doesTimeseriesVariableExist(timeseries_label.c_str())) {
            std::cout << "Skipping test: Timeseries variable does not exist." << std::endl;
        } else {
            float turbidity = solar.calibrateTurbidityFromTimeseries(timeseries_label);

            if (turbidity <= 0) {
                std::cout << "failed: calibrate turbidity from timeseries test." << std::endl;
                error_count++;
            }
        }
    }


    if (error_count == 0) {
        std::cout << "passed." << std::endl;
        return 0;
    } else {
        std::cout << "Failed Context self-test with " << error_count << " errors." << std::endl;
        return 1;
    }
}
