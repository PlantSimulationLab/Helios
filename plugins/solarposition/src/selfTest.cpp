#include "SolarPosition.h"

using namespace helios;

int SolarPosition::selfTest() const{

    std::cout << "Running solar position model self-test..." << std::flush;
    int error_count = 0;

    std::string answer;

    float errtol = 1e-6;

    Context context_s;

    //---- Sun Zenith/Azimuth Test for Boulder, CO ---- //

    try {

        float latitude = 40.1250;
        float longitude = 105.2369;
        Date date = make_Date(1, 1, 2000);
        Time time = make_Time(10, 30, 0);
        int UTC = 7;

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

    }catch( std::exception &e ){
        error_count++;
        std::cout << "failed: error thrown during solar position calculation." << std::endl;
    }

    //---- Test of Gueymard Solar Flux Model ---- //

    try {

        float latitude = 36.5289; //Billings, OK
        float longitude = 97.4439;
        Date date = make_Date(5, 5, 2003);
        Time time = make_Time(9, 10, 0);
        int UTC = 6;

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

    }catch( std::exception &e ){
        error_count++;
        std::cout << "failed: error thrown during solar flux calculation." << std::endl;
    }

    //----- Test of Ambient Longwave Model ------ //

    try {

        SolarPosition solarposition_3(UTC, latitude, longitude, &context_s);

        float temperature = 290;
        float humidity = 0.5;

        float LW = solarposition_3.getAmbientLongwaveFlux(temperature, humidity);

        if (fabs(LW - 310.03192f) > errtol) {
            error_count++;
            std::cout << "failed: verification test for ambient longwave model does not agree with known result." << std::endl;
        }

    }catch( std::exception &e ){
        error_count++;
        std::cout << "failed: error thrown during ambient longwave calculation." << std::endl;
    }



    auto test_getSunriseTime = [&error_count]() {
        try {
            Context context;
            SolarPosition solarPos(8, 38.55, 121.76, &context);
            Time sunrise = solarPos.getSunriseTime();
        }catch (std::exception &e) {
            error_count++;
        }
    };

    auto test_calibrateTurbidityFromTimeseries = [&error_count]() {
        try {
            Context context;
            SolarPosition solarPos(8, 38.55, 121.76, &context);
            context.addTimeseriesData("timeseries_shortwave_flux_label_Wm2", 1000, context.getDate(), context.getTime() );
            float turbidity = solarPos.calibrateTurbidityFromTimeseries("timeseries_shortwave_flux_label_Wm2");
        }catch (std::exception &e) {
            error_count++;
        }
    };

    auto test_enableCloudCalibration = [&error_count]() {
        try {
            Context context;
            SolarPosition solarPos(8, 38.55, 121.76, &context);
            context.addTimeseriesData("timeseries_shortwave_flux_label_Wm2", 1000, context.getDate(), context.getTime() );
            solarPos.enableCloudCalibration("timeseries_shortwave_flux_label_Wm2");
        }catch (std::exception &e) {
            error_count++;
        }
    };

    auto test_applyCloudCalibration = [&error_count]() {
        try {
            Context context;
            SolarPosition solarPos(8, 38.55, 121.76, &context);
            float R_calc_Wm2 = 1000.0;
            float fdiff_calc = 0.5;
            context.addTimeseriesData("timeseries_shortwave_flux_label_Wm2", 1000, context.getDate(), context.getTime() );
            solarPos.enableCloudCalibration("timeseries_shortwave_flux_label_Wm2");
            solarPos.applyCloudCalibration(R_calc_Wm2, fdiff_calc);
        }catch (std::exception &e) {
            error_count++;
        }
    };

    auto test_turbidityResidualFunction = [&error_count]() {
        try {
            Context context;
            SolarPosition solarPos(8, 38.55, 121.76, &context);
            std::vector<float> parameters = {101325, 300, 0.5, 900.0};
            float residual = turbidityResidualFunction(2.0, parameters, &solarPos);
        }catch (std::exception &e) {
            error_count++;
        }
    };

    test_getSunriseTime();
    test_calibrateTurbidityFromTimeseries();
    test_enableCloudCalibration();
    test_applyCloudCalibration();
    test_turbidityResidualFunction();

    if( error_count==0 ){
        std::cout << "passed." << std::endl;
        return 0;
    }else{
        std::cout << "Failed Context self-test with " << error_count << " errors." << std::endl;
        return 1;
    }



}