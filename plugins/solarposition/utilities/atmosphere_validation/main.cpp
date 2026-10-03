#include "SolarPosition.h"

using namespace helios;

// Validation driver for SolarPosition::calculateSensorAtmosphereSpectra(). Prints the model's quantities as CSV over a sweep of geometries, aerosol loads, surface pressures and wavelengths that fall mostly between the nodes of
// the look-up table, for comparison against direct 6S runs by compare_with_6s.py (which builds and runs this program).
//
// Usage: atmosphere_validation <pressure_hPa> [<pressure_hPa> ...]
//   stdout: CSV of Helios quantities
//   stderr: the water vapor and ozone columns used, as "water_vapor_cm=<value> ozone_atm_cm=<value>"
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: atmosphere_validation <pressure_hPa> [<pressure_hPa> ...]" << std::endl;
        return 1;
    }
    std::vector<float> pressures_hPa;
    for (int arg = 1; arg < argc; arg++) {
        pressures_hPa.push_back(std::stof(argv[arg]));
    }

    Context context;
    context.setDate(21, 6, 2025);
    context.setTime(0, 11);
    const float latitude = 38.55f;
    const float longitude = 121.76f;
    SolarPosition solarposition(8, latitude, longitude, &context);

    const float temperature_K = 293.f;
    const float humidity = 0.4f;

    // The water vapor column SolarPosition derives (same expression as its calculatePrecipitableWater()) and the ozone column it uses, so that 6S can be given the same columns
    const float gamma = logf(humidity) + 17.67f * (temperature_K - 273.0f) / 268.0f;
    const float dewpoint_F = 243.0f * gamma / (17.67f - gamma) * 9.0f / 5.0f + 32.0f;
    const float water_vapor_cm = expf((0.1133f - logf(5.0f)) + 0.0393f * dewpoint_F);
    const uint DOY = context.getJulianDate();
    const float ozone_atm_cm = solarposition.getOzoneColumn() * 0.001f;
    std::cerr << "water_vapor_cm=" << water_vapor_cm << " ozone_atm_cm=" << ozone_atm_cm << std::endl;

    // Extraterrestrial irradiance: SolarPosition's Wehrli (1985) spectrum, averaged over 1 nm bins as SolarPosition::SpectralData::loadFromFile() does (each source bin extending halfway to its neighbours'
    // centers), times the Earth-Sun distance factor of Spencer (1971), used to convert radiances to reflectances
    std::vector<float> toa_irradiance;
    {
        std::ifstream file("plugins/solarposition/ssolar_goa/wehrli85.txt");
        if (!file.is_open()) {
            std::cerr << "Could not open plugins/solarposition/ssolar_goa/wehrli85.txt" << std::endl;
            return 1;
        }
        std::vector<double> centers, irradiances;
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream fields(line);
            double center, irradiance;
            if (fields >> center >> irradiance) {
                centers.push_back(center);
                irradiances.push_back(irradiance);
            }
        }
        std::vector<double> edges(centers.size() + 1), cumulative(centers.size() + 1, 0.0);
        for (size_t i = 1; i < centers.size(); i++) {
            edges[i] = 0.5 * (centers[i - 1] + centers[i]);
        }
        edges.front() = centers[0] - 0.5 * (centers[1] - centers[0]);
        edges.back() = centers.back() + 0.5 * (centers.back() - centers[centers.size() - 2]);
        for (size_t i = 0; i < centers.size(); i++) {
            cumulative[i + 1] = cumulative[i] + irradiances[i] * (edges[i + 1] - edges[i]);
        }
        auto cumulative_at = [&](double wavelength) {
            const size_t upper = std::min(size_t(std::upper_bound(edges.begin(), edges.end(), wavelength) - edges.begin()), centers.size());
            return cumulative[upper - 1] + (cumulative[upper] - cumulative[upper - 1]) * (wavelength - edges[upper - 1]) / (edges[upper] - edges[upper - 1]);
        };
        const float c[] = {1.00011f, 0.03422f, 0.00128f, 0.000719f, 0.000077f};
        const float day_angle = (DOY - 1) * 2.0f * M_PI / 365.0f;
        const float geo_factor = c[0] + c[1] * cosf(day_angle) + c[2] * sinf(day_angle) + c[3] * cosf(2 * day_angle) + c[4] * sinf(2 * day_angle);
        for (int wavelength = 300; wavelength <= 2600; wavelength++) {
            toa_irradiance.push_back(float(cumulative_at(wavelength + 0.5) - cumulative_at(wavelength - 0.5)) * geo_factor);
        }
    }

    std::cout << "pressure_hPa,sza,vza,raa,aod550,surface_reflectance,wavelength_nm,rho_path,T_dir_up,T_dif_up,S,T_down_total,rho_toa" << std::endl;
    for (float pressure: pressures_hPa) {
        for (float sza: {15.f, 35.f, 55.f, 68.f}) {
            solarposition.setSunDirection(make_SphericalCoord(deg2rad(90.f - sza), 0.f));
            const float mu_sun = cosf(solarposition.getSunZenith());
            for (float vza: {0.f, 25.f, 45.f, 58.f}) {
                for (float raa: {0.f, 45.f, 110.f, 165.f}) {
                    if (vza == 0.f && raa > 0.f) {
                        continue;
                    }
                    const vec3 direction_to_sensor = sphere2cart(make_SphericalCoord(deg2rad(90.f - vza), deg2rad(raa)));
                    for (float aod550: {0.03f, 0.15f, 0.4f, 0.9f}) {
                        const float turbidity = aod550 * powf(0.55f, 1.3f);
                        solarposition.setAtmosphericConditions(pressure * 100.f, temperature_K, humidity, turbidity);
                        for (float surface_reflectance: {0.05f, 0.3f}) {
                            solarposition.setGroundAlbedo(surface_reflectance);
                            solarposition.calculateSensorAtmosphereSpectra("atm", direction_to_sensor, 1.f);
                            std::vector<vec2> path_radiance, adjacency_radiance, direct_up, diffuse_up, spherical_albedo, global_irradiance;
                            context.getGlobalData("atm_path_radiance", path_radiance);
                            context.getGlobalData("atm_adjacency_radiance", adjacency_radiance);
                            context.getGlobalData("atm_upward_direct_transmittance", direct_up);
                            context.getGlobalData("atm_upward_diffuse_transmittance", diffuse_up);
                            context.getGlobalData("atm_spherical_albedo", spherical_albedo);
                            context.getGlobalData("atm_global_irradiance", global_irradiance);
                            for (int wavelength_nm: {470, 510, 580, 645, 705, 800, 940, 1100, 1375, 1550, 1900, 2250}) {
                                const size_t i = wavelength_nm - 300;
                                const float to_reflectance = float(M_PI) / (mu_sun * toa_irradiance.at(i));
                                // A uniform Lambertian surface of the same reflectance as its surroundings
                                const float surface_radiance = surface_reflectance * global_irradiance[i].y / float(M_PI);
                                const float sensor_radiance = path_radiance[i].y + direct_up[i].y * surface_radiance + adjacency_radiance[i].y;
                                std::printf("%g,%g,%g,%g,%g,%g,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", pressure, sza, vza, raa, aod550, surface_reflectance, wavelength_nm, path_radiance[i].y * to_reflectance, direct_up[i].y, diffuse_up[i].y,
                                            spherical_albedo[i].y, global_irradiance[i].y / (mu_sun * toa_irradiance.at(i)), sensor_radiance * to_reflectance);
                            }
                        }
                    }
                }
            }
        }
    }
    return 0;
}
