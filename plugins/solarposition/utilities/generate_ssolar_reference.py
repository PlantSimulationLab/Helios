#!/usr/bin/env python3
"""Generate the 6S reference values hard-coded in the SolarPosition self-tests "Spectral solar irradiance against 6S" and
"Spectral solar irradiance against 6S with the sun low".

For each test case the validation driver (spectral_validation/) is run only to obtain the inputs that SolarPosition
derives itself (cosine of the solar zenith angle, water vapor column, ozone column, day of the year); the reference
values come entirely from 6S. 6S's transmittances are multiplied by the extraterrestrial irradiance SolarPosition uses
(assets/ssolar_goa/wehrli85.txt averaged over 1 nm bins as SolarPosition does, times the Earth-Sun distance factor of Spencer 1971), so that the
test can compare irradiances directly without the difference between the two models' extraterrestrial spectra.

The cases use 6S's continental aerosol. SolarPosition's aerosol has the single-scattering albedo and asymmetry parameter of
that aerosol at 550 nm at all wavelengths and an Angstrom exponent of 1.3, so the test tolerances are tight only at low
aerosol optical depth.

Requirements: as for compare_spectral_with_6s.py.

Usage:
    python generate_ssolar_reference.py <build_dir> [--processes N] [--solar-zenith DEG] [--aod550 AOD ...]
Prints C++ initializer rows for the test. The defaults (solar zenith 30 degrees, aerosol optical depths 0, 0.05, 0.2 and 0.5)
give the rows of "Spectral solar irradiance against 6S"; --solar-zenith 75 --aod550 0.2 0.5 gives those of
"Spectral solar irradiance against 6S with the sun low", whose rows omit the last (spherical albedo) column.
"""

import argparse
import os
import sys
from multiprocessing import Pool

from compare_spectral_with_6s import HERE, run_driver, run_sixs, spencer_distance_factor
from wehrli_spectrum import read_binned_spectrum

SOLAR_ZENITH_DEG = 30.0
PRESSURE_PA = 101325.0
TEMPERATURE_K = 293.0
RELATIVE_HUMIDITY = 0.4
GROUND_ALBEDO = 0.2
AOD550_VALUES = [0.0, 0.05, 0.2, 0.5]
WAVELENGTHS_NM = [400, 450, 550, 670, 870, 1240, 1640]


def read_spectrum_asset():
    return read_binned_spectrum(os.path.join(HERE, '..', 'assets', 'ssolar_goa', 'wehrli85.txt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('build_dir', help='directory in which to build the validation driver')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--solar-zenith', type=float, default=SOLAR_ZENITH_DEG, help='solar zenith angle in degrees')
    parser.add_argument('--aod550', type=float, nargs='+', default=AOD550_VALUES, help='aerosol optical depths at 550 nm')
    args = parser.parse_args()
    solar_zenith_deg, aod550_values = args.solar_zenith, args.aod550

    cases = [(f'aod{aod:g}', solar_zenith_deg, PRESSURE_PA, TEMPERATURE_K, RELATIVE_HUMIDITY, aod, GROUND_ALBEDO) for aod in aod550_values]
    metadata, _ = run_driver(args.build_dir, cases, WAVELENGTHS_NM, args.processes)
    first = metadata[cases[0][0]]
    mu, water_cm, ozone_atm_cm, day_of_year = first['mu'], first['water_cm'], first['ozone_DU'] / 1000.0, first['day_of_year']
    toa = read_spectrum_asset()
    distance_factor = spencer_distance_factor(day_of_year)

    sixs_cases = [(w, water_cm, ozone_atm_cm, aod, solar_zenith_deg, GROUND_ALBEDO) for aod in aod550_values for w in WAVELENGTHS_NM]
    with Pool(args.processes) as pool:
        results = dict(zip(sixs_cases, pool.map(run_sixs, sixs_cases)))

    print(f'// 6S (6SV1.1 via Py6S), continental aerosol, sea level, solar zenith {solar_zenith_deg:g} deg, ground albedo {GROUND_ALBEDO:g}, water vapor {water_cm:.4f} cm, ozone {ozone_atm_cm * 1000:.1f} DU, day of year {day_of_year}')
    print('// {aod550, wavelength_nm, direct normal, global horizontal, diffuse horizontal (W/m^2/nm), spherical albedo}')
    for case in sixs_cases:
        w, aod = case[0], case[3]
        r = results[case]
        extraterrestrial = toa[w] * distance_factor
        print(f'{{{aod:g}f, {w}, {extraterrestrial * r["T_direct"]:.5g}f, {extraterrestrial * mu * r["T_global"]:.5g}f, {extraterrestrial * mu * r["T_diffuse"]:.5g}f, {r["S"]:.5g}f}},')


if __name__ == '__main__':
    sys.exit(main())
