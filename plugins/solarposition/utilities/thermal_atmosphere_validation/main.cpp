#include "SolarPosition.h"

using namespace helios;

// Validation driver for SolarPosition::calculateSensorThermalAtmosphere() and SolarPosition::getThermalSkyFlux(). Reads atmospheric states from stdin, one per line:
//   <air temperature K> <relative humidity> <pressure Pa> <ozone column DU> <view zenith angle deg>
// and prints, for each state, a header line "# state <index> water_vapor_cm=<value>" followed by CSV lines "<index>,<wavelength nm>,<transmittance>,<upwelling radiance>,<downwelling radiance>" (radiances in
// W/m^2/sr/nm), for comparison against direct libRadtran runs by compare_thermal_with_uvspec.py and against the functions of Jiménez-Muñoz and Sobrino (2003) by compare_thermal_with_jms.py.
int main() {
    Context context;
    context.setDate(21, 6, 2025);
    context.setTime(0, 11);
    SolarPosition solarposition(8, 38.55f, 121.76f, &context);

    float temperature_K, humidity, pressure_Pa, ozone_DU, view_zenith_deg;
    int index = 0;
    while (std::cin >> temperature_K >> humidity >> pressure_Pa >> ozone_DU >> view_zenith_deg) {
        solarposition.setAtmosphericConditions(pressure_Pa, temperature_K, humidity, 0.05f);
        solarposition.setOzoneColumn(ozone_DU);
        const vec3 direction_to_sensor = sphere2cart(make_SphericalCoord(deg2rad(90.f - view_zenith_deg), 0.f));
        solarposition.calculateSensorThermalAtmosphere("atm", direction_to_sensor);

        float water_vapor_cm;
        context.getGlobalData("atm_thermal_water_vapor_cm", water_vapor_cm);
        std::vector<vec2> transmittance, upwelling, downwelling;
        context.getGlobalData("atm_thermal_transmittance", transmittance);
        context.getGlobalData("atm_thermal_upwelling_radiance", upwelling);
        context.getGlobalData("atm_thermal_downwelling_radiance", downwelling);
        std::cout << "# state " << index << " water_vapor_cm=" << std::setprecision(8) << water_vapor_cm << "\n";
        for (size_t i = 0; i < transmittance.size(); i++) {
            std::cout << index << "," << std::setprecision(9) << transmittance.at(i).x << "," << transmittance.at(i).y << "," << upwelling.at(i).y << "," << downwelling.at(i).y << "\n";
        }
        index++;
    }
    return 0;
}
