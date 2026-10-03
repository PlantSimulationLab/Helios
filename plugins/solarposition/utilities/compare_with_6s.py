#!/usr/bin/env python3
"""Validate SolarPosition::calculateSensorAtmosphereSpectra() against direct 6S runs.

Builds and runs the driver in atmosphere_validation/, which prints Helios's atmospheric quantities over a sweep of
geometries, aerosol optical depths, surface pressures and wavelengths chosen mostly to fall between the nodes of the look-up
table. The same cases are then run through 6S (continental aerosol, the same water vapor and ozone columns, a uniform
Lambertian surface) and the differences are tabulated. The top-of-atmosphere reflectance difference is the headline
number: it includes the look-up-table interpolation, the gas-absorption treatment and the way the quantities are
combined.

Requirements: the 6S executable and Py6S on PATH (see generate_atmosphere_lut.py), CMake and a C++ compiler.

Usage:
    python compare_with_6s.py <build_dir> [--processes N] [--csv <output.csv>]
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool

from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, GroundReflectance, Wavelength

TARGET_ALTITUDES_KM = [0.0, 1.5]
HERE = os.path.dirname(os.path.abspath(__file__))


def configure_sixs(wavelength_nm, water_cm, ozone_atm_cm, altitude_km, aod550, sza, vza, raa, surface_reflectance):
    s = SixS()
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water_cm, ozone_atm_cm)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = aod550
    s.geometry = Geometry.User()
    s.geometry.solar_z, s.geometry.solar_a, s.geometry.view_z, s.geometry.view_a = sza, 0, vza, raa
    s.geometry.month, s.geometry.day = 6, 21
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(surface_reflectance)
    s.altitudes.set_sensor_satellite_level()
    if altitude_km == 0.0:
        s.altitudes.set_target_sea_level()
    else:
        s.altitudes.set_target_custom_altitude(altitude_km)
    s.wavelength = Wavelength(wavelength_nm / 1000.0)
    return s


def ground_pressure(altitude_km):
    s = configure_sixs(550, 1.0, 0.3, altitude_km, 0.1, 30, 0, 0, 0.1)
    s.run()
    return s.outputs.values['ground_pressure']


def equivalent_water_column(altitude_km, water_cm):
    """Water column to give 6S so that the water above a target at altitude_km equals water_cm.

    6S interprets a user water column as the column above sea level and removes the part below an elevated target,
    whereas the column SolarPosition derives from surface humidity is the column above the site. The two are matched
    by bisecting on the 940 nm water transmittance, which depends only on the water path.
    """
    def water_transmittance(altitude, column):
        s = configure_sixs(940, column, 0.3, altitude, 0.0, 0, 0, 0, 0.1)
        s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
        s.run()
        return s.outputs.trans['water'].downward

    if altitude_km == 0.0:
        return water_cm
    target = water_transmittance(0.0, water_cm)
    low, high = water_cm, 10.0 * water_cm
    for _ in range(40):
        middle = 0.5 * (low + high)
        if water_transmittance(altitude_km, middle) > target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def run_case(case):
    water_cm, ozone_atm_cm, altitude_km, aod550, sza, vza, raa, surface_reflectance, wavelength_nm = case
    s = configure_sixs(wavelength_nm, water_cm, ozone_atm_cm, altitude_km, aod550, sza, vza, raa, surface_reflectance)
    s.run()
    o = s.outputs
    mu_s, mu_v = math.cos(math.radians(sza)), math.cos(math.radians(vza))
    E_sun = o.values['solar_spectrum']
    direct_up = math.exp(-o.optical_depth_total.total / mu_v)
    # Helios reports the upward transmittance of radiance already attenuated on the way down, i.e. the two-way gas transmittance divided by the downward one
    gas_up = o.transmittance_global_gas.total / o.transmittance_global_gas.downward if o.transmittance_global_gas.downward > 0 else 0.0
    return dict(rho_path=math.pi * o.values['atmospheric_intrinsic_radiance'] / (mu_s * E_sun), T_dir_up=direct_up * gas_up, T_dif_up=(o.transmittance_total_scattering.upward - direct_up) * gas_up,
                S=o.spherical_albedo.total, T_down_total=(o.values['direct_solar_irradiance'] + o.values['diffuse_solar_irradiance'] + o.values['environmental_irradiance']) / (mu_s * E_sun),
                rho_toa=o.values['apparent_reflectance'])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('build_dir', help='directory in which to build the validation driver')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--csv', help='write every case with Helios and 6S values to this file')
    args = parser.parse_args()

    pressures = {altitude: ground_pressure(altitude) for altitude in TARGET_ALTITUDES_KM}
    altitude_for_pressure = {round(p, 2): altitude for altitude, p in pressures.items()}

    build_dir = os.path.abspath(args.build_dir)
    subprocess.run(['cmake', '-S', os.path.join(HERE, 'atmosphere_validation'), '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['cmake', '--build', build_dir, '-j', str(args.processes)], check=True, stdout=subprocess.DEVNULL)
    driver = subprocess.run([os.path.join(build_dir, 'atmosphere_validation')] + [f'{p:.2f}' for p in pressures.values()], cwd=build_dir, check=True, capture_output=True, text=True)
    columns = dict(item.split('=') for item in driver.stderr.split())
    water_cm, ozone_atm_cm = float(columns['water_vapor_cm']), float(columns['ozone_atm_cm'])
    helios_rows = list(csv.DictReader(driver.stdout.splitlines()))

    water_for_altitude = {altitude: equivalent_water_column(altitude, water_cm) for altitude in TARGET_ALTITUDES_KM}
    cases = [(water_for_altitude[altitude_for_pressure[round(float(r['pressure_hPa']), 2)]], ozone_atm_cm, altitude_for_pressure[round(float(r['pressure_hPa']), 2)], float(r['aod550']), float(r['sza']), float(r['vza']), float(r['raa']), float(r['surface_reflectance']),
              int(r['wavelength_nm'])) for r in helios_rows]
    print(f'6S water columns (sea-level referenced) by target altitude: {water_for_altitude}')
    print(f'{len(cases)} cases; water vapor {water_cm:.3f} cm, ozone {ozone_atm_cm:.3f} atm-cm, pressures {", ".join(f"{p:.1f}" for p in pressures.values())} hPa', flush=True)
    with Pool(args.processes) as pool:
        sixs_rows = pool.map(run_case, cases, chunksize=32)

    fields = ['rho_path', 'T_dir_up', 'T_dif_up', 'S', 'T_down_total', 'rho_toa']
    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(helios_rows[0].keys()) + [f'6s_{field}' for field in fields])
            for h, s in zip(helios_rows, sixs_rows):
                writer.writerow(list(h.values()) + [s[field] for field in fields])

    for field in fields:
        groups = defaultdict(list)
        for h, s in zip(helios_rows, sixs_rows):
            groups[(int(h['wavelength_nm']), float(h['aod550']))].append((float(h[field]), s[field]))
        print(f'\n== {field}: mean and max |Helios - 6S| over geometries, pressures and surfaces (mean 6S value in brackets) ==')
        print('  wavelength ' + ''.join(f'   AOD {aod:<14g}' for aod in sorted({k[1] for k in groups})))
        for wavelength in sorted({k[0] for k in groups}):
            cells = []
            for aod in sorted({k[1] for k in groups}):
                values = groups[(wavelength, aod)]
                errors = [abs(a - b) for a, b in values]
                cells.append(f'{sum(errors) / len(errors):.4f}/{max(errors):.4f} [{sum(b for _, b in values) / len(values):.3f}]')
            print(f'  {wavelength:6d} nm  ' + '  '.join(cells))

    toa_errors = [abs(float(h['rho_toa']) - s['rho_toa']) for h, s in zip(helios_rows, sixs_rows)]
    print(f'\nTop-of-atmosphere reflectance: mean |error| {sum(toa_errors) / len(toa_errors):.4f}, max |error| {max(toa_errors):.4f}')


if __name__ == '__main__':
    sys.exit(main())
