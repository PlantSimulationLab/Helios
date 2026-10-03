#include "SolarPosition.h"

using namespace helios;

// Validation driver for SolarPosition's spectral solar irradiance model (calculateGlobalSolarSpectrum(), calculateDirectSolarSpectrum() and calculateDiffuseSolarSpectrum()). Reads cases from stdin and prints the model's
// irradiances as CSV, for comparison against direct 6S runs by compare_spectral_with_6s.py and generate_ssolar_reference.py (which build and run this program).
//
// Usage: spectral_validation <wavelength_nm> [<wavelength_nm> ...] < cases
//   Each line of stdin is a case: <case_id> <solar_zenith_deg> <pressure_Pa> <temperature_K> <relative_humidity> <aod550> <ground_albedo>
//   stdout: for each case a line "case,<case_id>,<cos_zenith>,<water_vapor_cm>,<ozone_DU>,<day_of_year>", then for each wavelength a line
//           "spectrum,<case_id>,<wavelength_nm>,<direct_normal>,<global_horizontal>,<diffuse_horizontal>" in W/m^2/nm
int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: spectral_validation <wavelength_nm> [<wavelength_nm> ...] < cases" << std::endl;
        return 1;
    }
    std::vector<int> wavelengths_nm;
    for (int arg = 1; arg < argc; arg++) {
        wavelengths_nm.push_back(std::stoi(argv[arg]));
        if (wavelengths_nm.back() < 300 || wavelengths_nm.back() > 2600) {
            std::cerr << "Wavelengths must lie between 300 and 2600 nm" << std::endl;
            return 1;
        }
    }

    Context context;
    context.setDate(21, 6, 2025);
    context.setTime(0, 11);
    SolarPosition solarposition(8, 38.55f, 121.76f, &context);

    std::string case_id;
    float solar_zenith_deg, pressure_Pa, temperature_K, humidity, aod550, ground_albedo;
    while (std::cin >> case_id >> solar_zenith_deg >> pressure_Pa >> temperature_K >> humidity >> aod550 >> ground_albedo) {
        solarposition.setSunDirection(make_SphericalCoord(deg2rad(90.f - solar_zenith_deg), 0.f));
        // The model's aerosol optical depth follows the Angstrom law with exponent 1.3 and turbidity (optical depth at 1 um) beta
        const float turbidity = aod550 * powf(0.55f, 1.3f);
        solarposition.setAtmosphericConditions(pressure_Pa, temperature_K, humidity, turbidity);
        solarposition.setGroundAlbedo(ground_albedo);

        solarposition.calculateDirectSolarSpectrum("validation_direct");
        solarposition.calculateGlobalSolarSpectrum("validation_global");
        solarposition.calculateDiffuseSolarSpectrum("validation_diffuse");
        std::vector<vec2> direct, global, diffuse;
        context.getGlobalData("validation_direct", direct);
        context.getGlobalData("validation_global", global);
        context.getGlobalData("validation_diffuse", diffuse);

        // The water vapor column SolarPosition derives (same expression as its calculatePrecipitableWater()), so that 6S can be given the same column
        const float gamma = logf(humidity) + 17.67f * (temperature_K - 273.0f) / 268.0f;
        const float dewpoint_F = 243.0f * gamma / (17.67f - gamma) * 9.0f / 5.0f + 32.0f;
        const float water_vapor_cm = expf((0.1133f - logf(5.0f)) + 0.0393f * dewpoint_F);

        std::printf("case,%s,%.7f,%.6f,%.4f,%d\n", case_id.c_str(), cosf(solarposition.getSunZenith()), water_vapor_cm, solarposition.getOzoneColumn(), context.getJulianDate());
        for (int wavelength_nm: wavelengths_nm) {
            const size_t i = wavelength_nm - 300;
            std::printf("spectrum,%s,%d,%.6g,%.6g,%.6g\n", case_id.c_str(), wavelength_nm, direct.at(i).y, global.at(i).y, diffuse.at(i).y);
        }
    }
    return 0;
}
