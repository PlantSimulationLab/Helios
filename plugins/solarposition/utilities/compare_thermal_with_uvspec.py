#!/usr/bin/env python3
"""Validate SolarPosition's thermal atmosphere look-up table as Helios uses it, against direct libRadtran runs.

Builds and runs thermal_atmosphere_validation (SolarPosition::calculateSensorThermalAtmosphere()) for random atmospheric
states that fall between the table's nodes, runs uvspec for the same states exactly as generate_thermal_atmosphere_lut.py
does, and reports, for sensor bands, the difference in brightness temperature at the top of the atmosphere over a black
surface at the air temperature, and the relative difference in band downwelling radiance. This measures the error of the
table's interpolation (and of its implementation in Helios), not of the radiative transfer; see
compare_thermal_with_lblrtm.py and compare_thermal_with_jms.py for that.

With --held-out it instead measures the error of the table's profile assumption: for the subarctic summer atmosphere of
Anderson et al. (1986), which is not one of the base atmospheres of the table, it compares uvspec run with that atmosphere
(trace gases scaled to the table's) with uvspec run with the table's profile for the same surface air temperature, water
vapor and ozone columns.

Requirements: the libRadtran environment of generate_thermal_atmosphere_lut.py, CMake and a C++ compiler, and a thermal
look-up table in plugins/solarposition/assets/thermal_atmosphere_lut/.

Usage:
    python compare_thermal_with_uvspec.py <build_dir> [--states N] [--seed S] [--processes N] [--uvspec PATH] [--csv results.csv]
    python compare_thermal_with_uvspec.py --held-out [--uvspec PATH]
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

import generate_thermal_atmosphere_lut as lut_generator
from compare_thermal_with_lblrtm import BANDS

HERE = os.path.dirname(os.path.abspath(__file__))


def planck_per_nm(wavelength_nm, temperature):
    """Planck radiance in W/m2/sr/nm."""
    wavelength_m = wavelength_nm * 1e-9
    return 1.191042972e-16 / (wavelength_m ** 5 * np.expm1(1.438776877e-2 / (wavelength_m * temperature))) * 1e-9


def band_integral(wavelength_nm, values, wavelength_min, wavelength_max):
    """Trapezoidal integral of a spectrum between two wavelengths, interpolating it linearly to the bounds (as Helios does)."""
    grid = np.concatenate(([wavelength_min], wavelength_nm[(wavelength_nm > wavelength_min) & (wavelength_nm < wavelength_max)], [wavelength_max]))
    return np.trapezoid(np.interp(grid, wavelength_nm, values), grid)


def band_brightness_temperature(wavelength_nm, transmittance, upwelling, surface_temperature, wavelength_min, wavelength_max):
    """Brightness temperature at the top of the atmosphere over a black surface, for a boxcar band, on a 1 nm grid."""
    grid = np.arange(wavelength_min, wavelength_max + 0.5, 1.0)
    radiance = np.interp(grid, wavelength_nm, transmittance) * planck_per_nm(grid, surface_temperature) + np.interp(grid, wavelength_nm, upwelling)
    target = np.trapezoid(radiance, grid)
    low, high = 150.0, 400.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if np.trapezoid(planck_per_nm(grid, middle), grid) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def build_driver(build_dir, processes):
    subprocess.run(['cmake', '-S', os.path.join(HERE, 'thermal_atmosphere_validation'), '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['cmake', '--build', build_dir, '-j', str(processes)], check=True, stdout=subprocess.DEVNULL)


def run_driver(build_dir, states):
    """Run thermal_atmosphere_validation for (air temperature K, relative humidity, pressure Pa, ozone DU, view zenith deg)
    states; returns for each the column water vapor and the wavelength, transmittance, upwelling and downwelling arrays."""
    stdin = ''.join(f'{t} {rh} {p} {o3} {vza}\n' for t, rh, p, o3, vza in states)
    output = subprocess.run([os.path.join(build_dir, 'thermal_atmosphere_validation')], input=stdin, cwd=build_dir, check=True, capture_output=True, text=True).stdout
    results = [dict(rows=[]) for _ in states]
    for line in output.splitlines():
        if line.startswith('# state'):
            fields = line.split()
            results[int(fields[2])]['water_vapor_cm'] = float(fields[3].split('=')[1])
        elif line:
            index, wavelength, transmittance, upwelling, downwelling = line.split(',')
            results[int(index)]['rows'].append((float(wavelength), float(transmittance), float(upwelling), float(downwelling)))
    for result in results:
        rows = np.array(result.pop('rows'))
        result.update(wavelength_nm=rows[:, 0], transmittance=rows[:, 1], upwelling_radiance=rows[:, 2], downwelling_radiance=rows[:, 3])
    return results


def climate_coordinate(base, air_temperature, pressure_hPa):
    """The mapping SolarPosition applies: surface air temperature is linear in the climate coordinate between nodes, and the
    node temperatures are interpolated linearly in log pressure between the pressure nodes."""
    pressures = [lut_generator.surface_pressure_hPa(base, altitude) for altitude in lut_generator.SURFACE_ALTITUDES_KM]
    node_temperatures = np.array([lut_generator.climate_surface_temperatures(base, p) for p in pressures])
    log_p = np.log(pressures)
    x = math.log(min(pressure_hPa, pressures[0]))
    i = int(np.clip(np.searchsorted(-log_p, -x) - 1, 0, len(pressures) - 2))
    weight = (x - log_p[i]) / (log_p[i + 1] - log_p[i])
    temperatures = (1 - weight) * node_temperatures[i] + weight * node_temperatures[i + 1]
    return float(np.interp(air_temperature, temperatures, lut_generator.CLIMATE_COORDINATES))


def uvspec_state(args):
    """Direct uvspec result for one state, averaged over the table's bins, in increasing wavelength."""
    uvspec, data_path, water_vapor_cm, coordinate, pressure_hPa, ozone_du, view_zenith_deg = args
    base = lut_generator.load_base_profile(data_path)
    state = lut_generator.compute_state(uvspec, data_path, base, water_vapor_cm, coordinate, pressure_hPa, ozone_du, [view_zenith_deg])
    return lut_generator.to_wavelength(state)


def held_out_profile(data_path, name):
    """An atmosphere of Anderson et al. (1986) with the U.S. Standard CH4 and N2O profiles and the table's CO2, CH4 and N2O
    amounts, and its surface air temperature, water vapor column (g/cm2) and ozone column (DU)."""
    atmmod = os.path.join(data_path, 'atmmod')
    atmosphere = lut_generator.load_afgl(atmmod, name)
    standard = lut_generator.load_base_profile(data_path)['standard_vmr']
    standard_z = lut_generator.load_afgl(atmmod, 'afglus')['z']
    vmr = dict(atmosphere['vmr'])
    for species, amount in (('ch4', lut_generator.CH4_PPB * 1e-9), ('n2o', lut_generator.N2O_PPB * 1e-9)):
        vmr[species] = np.interp(atmosphere['z'], standard_z, standard[species]) * amount / standard[species][0]
    vmr['co2'] = vmr['co2'] * lut_generator.CO2_PPM * 1e-6 / vmr['co2'][0]
    air = atmosphere['p'] * 100.0 / (lut_generator.BOLTZMANN * atmosphere['t']) * 1e-6
    z_cm = atmosphere['z'] * 1e5
    water_vapor_cm = np.trapezoid(air * vmr['h2o'], z_cm) * 18.015 / 6.02214076e23
    ozone_du = np.trapezoid(air * vmr['o3'], z_cm) / 2.6867e16
    return dict(z=atmosphere['z'], p=atmosphere['p'], t=atmosphere['t'], vmr=vmr), float(atmosphere['t'][0]), float(water_vapor_cm), float(ozone_du)


def held_out_runs(args):
    uvspec, data_path, profile, water_vapor_cm, ozone_du = args
    with tempfile.TemporaryDirectory(prefix='helios_thermal_') as directory:
        lut_generator.write_profile_files(profile, directory)
        return [lut_generator.run_uvspec(uvspec, data_path, directory, water_vapor_cm, ozone_du, t, [0.0]) for t in lut_generator.SURFACE_TEMPERATURES_K]


def held_out_check(uvspec, data_path):
    base = lut_generator.load_base_profile(data_path)
    profile, air_temperature, water_vapor_cm, ozone_du = held_out_profile(data_path, 'afglss')
    table_profile = lut_generator.build_profile(base, climate_coordinate(base, air_temperature, profile['p'][0]), profile['p'][0])
    with Pool(2) as pool:
        results = pool.map(held_out_runs, [(uvspec, data_path, p, water_vapor_cm, ozone_du) for p in (profile, table_profile)])
    print(f'Subarctic summer atmosphere (surface air temperature {air_temperature} K, water vapor {water_vapor_cm:.2f} g/cm2, ozone {ozone_du:.0f} DU), nadir, black surface at the air temperature:')
    print('brightness temperature at the top of the atmosphere with the table\'s profile minus that with the actual profile (K)')
    brightness = []
    for runs in results:
        wavenumbers, surface_1, toa_1, _ = runs[0]
        _, surface_2, toa_2, _ = runs[1]
        tau = (toa_1[:, 0] - toa_2[:, 0]) / (surface_1 - surface_2)
        radiance_per_cm = tau * 1.191042972e-8 * wavenumbers ** 3 / np.expm1(1.438776877 * wavenumbers / air_temperature) + toa_1[:, 0] - tau * surface_1
        order = np.argsort(-wavenumbers)
        wavelength = 1e7 / wavenumbers[order]
        radiance_per_nm = radiance_per_cm[order] * wavenumbers[order] ** 2 * 1e-7
        brightness.append({band: band_brightness_temperature(wavelength, np.zeros_like(wavelength), radiance_per_nm, air_temperature, *limits) for band, limits in BANDS.items()})
    for band in BANDS:
        print(f'  {band:28s} {brightness[1][band] - brightness[0][band]:+.2f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('build_dir', nargs='?')
    parser.add_argument('--held-out', action='store_true')
    parser.add_argument('--states', type=int, default=80)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--processes', type=int, default=os.cpu_count())
    parser.add_argument('--uvspec', default=None)
    parser.add_argument('--libradtran-data', default=None)
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()

    uvspec = args.uvspec or lut_generator.default_uvspec()
    data_path = args.libradtran_data or lut_generator.default_data_path(uvspec)
    if args.held_out:
        held_out_check(uvspec, data_path)
        return
    if args.build_dir is None:
        parser.error('build_dir is required unless --held-out is given')
    base = lut_generator.load_base_profile(data_path)
    build_dir = os.path.abspath(args.build_dir)
    build_driver(build_dir, args.processes)

    # Random states within the table: air temperature 250-305 K, water vapor 0.1-7.5 g/cm2 (log-uniform, reached through the relative humidity), surface pressure 701-1013 hPa, ozone 150-450 DU, view zenith 0-60 degrees
    rng = np.random.default_rng(args.seed)
    states = []
    for _ in range(args.states):
        air_temperature = rng.uniform(250.0, 305.0)
        humidity = rng.uniform(0.05, 1.0)
        pressure_Pa = rng.uniform(70120.0, 101300.0)
        states.append((air_temperature, humidity, pressure_Pa, rng.uniform(150.0, 450.0), rng.uniform(0.0, 60.0)))
    # Keep the states that lie inside the table (Helios raises an error for the others, e.g. when the water vapor it derives is outside 0.1-7.5 g/cm2)
    helios = []
    kept = []
    for state in states:
        try:
            result = run_driver(build_dir, [state])[0]
        except subprocess.CalledProcessError:
            continue
        helios.append(result)
        kept.append(state)
    print(f'{len(kept)} of {len(states)} random states lie within the table')

    jobs = [(uvspec, data_path, result['water_vapor_cm'], climate_coordinate(base, t, p * 0.01), p * 0.01, o3, vza) for result, (t, rh, p, o3, vza) in zip(helios, kept)]
    with Pool(args.processes) as pool:
        reference = pool.map(uvspec_state, jobs, chunksize=1)

    rows = []
    for state, result, truth in zip(kept, helios, reference):
        air_temperature, humidity, pressure_Pa, ozone_du, view_zenith_deg = state
        if not np.allclose(result['wavelength_nm'], truth['wavelength_nm'], rtol=1e-6):
            raise RuntimeError('Helios and uvspec wavelength grids differ')
        wavelength = truth['wavelength_nm']
        for band, (wavelength_min, wavelength_max) in BANDS.items():
            bt_helios = band_brightness_temperature(wavelength, result['transmittance'], result['upwelling_radiance'], air_temperature, wavelength_min, wavelength_max)
            bt_uvspec = band_brightness_temperature(wavelength, truth['transmittance'][:, 0], truth['upwelling_radiance'][:, 0], air_temperature, wavelength_min, wavelength_max)
            downwelling_ratio = band_integral(wavelength, result['downwelling_radiance'], wavelength_min, wavelength_max) / band_integral(wavelength, truth['downwelling_radiance'], wavelength_min, wavelength_max)
            rows.append((air_temperature, humidity, pressure_Pa, ozone_du, view_zenith_deg, result['water_vapor_cm'], band, bt_helios - bt_uvspec, downwelling_ratio - 1.0))

    print(f'\nHelios table minus direct uvspec, {len(kept)} states: brightness temperature at the top of the atmosphere over a black surface at the air temperature (K), and band downwelling radiance (%)')
    for band in BANDS:
        errors = np.array([row[7] for row in rows if row[6] == band])
        downwelling = np.array([row[8] for row in rows if row[6] == band]) * 100.0
        print(f'  {band:28s} BT rms {np.sqrt(np.mean(errors ** 2)):.3f} max {np.max(np.abs(errors)):.3f} | L_down rms {np.sqrt(np.mean(downwelling ** 2)):.2f}% max {np.max(np.abs(downwelling)):.2f}%')
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('air_temperature_K,relative_humidity,pressure_Pa,ozone_DU,view_zenith_deg,water_vapor_cm,band,bt_difference_K,downwelling_relative_difference\n')
            for row in rows:
                f.write(','.join(str(value) for value in row) + '\n')


if __name__ == '__main__':
    sys.exit(main())
