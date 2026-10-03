#!/usr/bin/env python3
"""Validate SolarPosition's spectral solar irradiance model against direct 6S runs.

Builds and runs the driver in spectral_validation/, which prints Helios's direct normal, global horizontal and diffuse
horizontal spectral irradiance for a sweep of solar zenith angles, aerosol optical depths and ground albedos. The same
cases are run through 6S (continental aerosol, the same water vapor and ozone columns, a uniform Lambertian ground at sea
level) and the differences are tabulated.

Helios and 6S use different extraterrestrial spectra, so both are compared as transmittances: Helios irradiance divided
by its own extraterrestrial irradiance (the spectrum file it reads, times the Earth-Sun distance factor of Spencer 1971),
and 6S irradiance divided by the 6S solar spectrum. A relative transmittance difference equals the relative irradiance
difference the two models would give for the same extraterrestrial spectrum. The atmospheric spherical albedo of Helios
is recovered from the ratio of global irradiance with and without a reflecting ground, since the model scales global
irradiance by 1/(1 - albedo*S).

Differences reflect both the model's approximations and its aerosol model, which has the single-scattering albedo (0.893)
and asymmetry parameter (0.634) of 6S's continental aerosol at 550 nm at all wavelengths and an Angstrom exponent of 1.3,
whereas the continental aerosol's properties vary with wavelength. --formula-check
separates the two for the spherical albedo: it evaluates the model's spherical albedo formula with 6S's own aerosol
optical depth, single-scattering albedo and asymmetry parameter (the last from integrating 6S's aerosol phase function
over scattering angle) and compares it with 6S's spherical albedo.

Requirements: the 6S executable, Py6S, NumPy and SciPy on PATH (see generate_atmosphere_lut.py), CMake and a C++ compiler.

Usage:
    python compare_spectral_with_6s.py <build_dir> [--processes N] [--csv <output.csv>] [--formula-check]
"""

import argparse
import csv
import math
import os
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, GroundReflectance, Wavelength

from wehrli_spectrum import read_binned_spectrum

HERE = os.path.dirname(os.path.abspath(__file__))

WAVELENGTHS_NM = [350, 400, 450, 500, 550, 620, 670, 750, 870, 1040, 1240, 1640, 2130, 2200]
SOLAR_ZENITHS_DEG = [0.0, 30.0, 60.0, 75.0]
AOD550_VALUES = [0.0, 0.05, 0.2, 0.5, 1.0]
GROUND_ALBEDOS = [0.0, 0.2, 0.8]
TEMPERATURE_K = 293.0
RELATIVE_HUMIDITY = 0.4
SEA_LEVEL_PRESSURE_PA = 101325.0


def spencer_distance_factor(day_of_year):
    """Square of the ratio of the mean to the actual Earth-Sun distance (Spencer 1971), day_of_year = 1 on January 1."""
    day_angle = 2.0 * math.pi * (day_of_year - 1) / 365.0
    return (1.000110 + 0.034221 * math.cos(day_angle) + 0.001280 * math.sin(day_angle) + 0.000719 * math.cos(2 * day_angle)
            + 0.000077 * math.sin(2 * day_angle))


def read_toa_spectrum(build_dir):
    """Extraterrestrial spectrum read by the Helios build in build_dir, as {wavelength_nm: W/m^2/nm}."""
    return read_binned_spectrum(os.path.join(build_dir, 'plugins', 'solarposition', 'ssolar_goa', 'wehrli85.txt'))


def run_driver(build_dir, cases, wavelengths_nm, processes):
    """Build the driver and run it on cases [(case_id, sza, pressure_Pa, temperature_K, humidity, aod550, albedo)].

    Returns ({case_id: dict(mu, water_cm, ozone_DU, day_of_year)}, {(case_id, wavelength_nm): (direct_normal, global, diffuse)}).
    """
    build_dir = os.path.abspath(build_dir)
    subprocess.run(['cmake', '-S', os.path.join(HERE, 'spectral_validation'), '-B', build_dir, '-DCMAKE_BUILD_TYPE=Release'], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['cmake', '--build', build_dir, '-j', str(processes)], check=True, stdout=subprocess.DEVNULL)
    stdin = ''.join(f'{c[0]} {c[1]} {c[2]} {c[3]} {c[4]} {c[5]} {c[6]}\n' for c in cases)
    driver = subprocess.run([os.path.join(build_dir, 'spectral_validation')] + [str(w) for w in wavelengths_nm], cwd=build_dir, input=stdin, check=True, capture_output=True, text=True)
    metadata, spectra = {}, {}
    for line in driver.stdout.splitlines():
        fields = line.split(',')
        if fields[0] == 'case':
            metadata[fields[1]] = dict(mu=float(fields[2]), water_cm=float(fields[3]), ozone_DU=float(fields[4]), day_of_year=int(fields[5]))
        elif fields[0] == 'spectrum':
            spectra[(fields[1], int(fields[2]))] = tuple(float(x) for x in fields[3:6])
    return metadata, spectra


def configure_sixs(wavelength_nm, water_cm, ozone_atm_cm, aod550, sza, ground_albedo):
    s = SixS()
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water_cm, ozone_atm_cm)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = aod550
    s.geometry = Geometry.User()
    s.geometry.solar_z, s.geometry.solar_a, s.geometry.view_z, s.geometry.view_a = sza, 0, 0, 0
    s.geometry.month, s.geometry.day = 6, 21
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(ground_albedo)
    s.altitudes.set_sensor_satellite_level()
    s.altitudes.set_target_sea_level()
    s.wavelength = Wavelength(wavelength_nm / 1000.0)
    return s


def run_sixs(case):
    """6S transmittances for (wavelength_nm, water_cm, ozone_atm_cm, aod550, sza, ground_albedo)."""
    wavelength_nm, water_cm, ozone_atm_cm, aod550, sza, ground_albedo = case
    s = configure_sixs(wavelength_nm, water_cm, ozone_atm_cm, aod550, sza, ground_albedo)
    s.run()
    o = s.outputs
    mu = math.cos(math.radians(sza))
    E_sun = o.values['solar_spectrum']
    direct = o.values['direct_solar_irradiance']
    # Irradiance of the ground: direct, diffuse, and the environmental irradiance from multiple reflection between the ground and the atmosphere
    diffuse = o.values['diffuse_solar_irradiance'] + o.values['environmental_irradiance']
    return dict(T_direct=direct / (mu * E_sun), T_global=(direct + diffuse) / (mu * E_sun), T_diffuse=diffuse / (mu * E_sun), S=o.spherical_albedo.total)


def helios_transmittances(metadata, spectra, toa, case_id, wavelength_nm):
    m = metadata[case_id]
    extraterrestrial = toa[wavelength_nm] * spencer_distance_factor(m['day_of_year'])
    direct_normal, global_horizontal, diffuse_horizontal = spectra[(case_id, wavelength_nm)]
    return dict(T_direct=direct_normal / extraterrestrial, T_global=global_horizontal / (m['mu'] * extraterrestrial), T_diffuse=diffuse_horizontal / (m['mu'] * extraterrestrial))


def case_id_for(sza, aod550, albedo):
    return f'sza{sza:g}_aod{aod550:g}_rho{albedo:g}'


# ----- Formula check: the model's spherical albedo formula with 6S's own aerosol properties -----

def sixs_aerosol_asymmetry(wavelength_nm):
    """Asymmetry parameter of 6S's continental aerosol, by integrating its phase function over scattering angle (definition in the 6S User Guide, p. 117)."""
    samples = []
    for theta in list(np.arange(0.0, 88.0, 1.0)) + list(np.arange(88.0, 89.95, 0.1)):
        for relative_azimuth in (0.0, 180.0):
            s = configure_sixs(wavelength_nm, 1e-6, 1e-6, 0.2, theta, 0.0)
            s.geometry.view_z, s.geometry.view_a = theta, relative_azimuth
            try:
                s.run()
            except Exception:
                continue  # 6S rejects some near-grazing geometries
            phase = s.outputs.phase_function_I.aerosol
            if np.isfinite(phase):
                samples.append((round(s.outputs.values['scattering_angle'], 4), phase))
    samples = sorted(set(samples))
    angle = np.radians([a for a, _ in samples])
    phase = np.array([p for _, p in samples])
    return np.trapezoid(phase * np.cos(angle) * np.sin(angle), angle) / np.trapezoid(phase * np.sin(angle), angle)


def sixs_albedo_case(case):
    wavelength_nm, aod550 = case
    s = configure_sixs(wavelength_nm, 1e-6, 1e-6, aod550, 30.0, 0.0)
    s.run()
    o = s.outputs
    return dict(tau_a=o.optical_depth_total.aerosol, tau_r=o.optical_depth_total.rayleigh, ssa=o.single_scattering_albedo.aerosol, S_total=o.spherical_albedo.total)


def csalbr(tau):
    """Rayleigh spherical albedo of 6S subroutine CSALBR (6S User Guide v3, Part 2, p. 96, Eq. 2)."""
    from scipy.special import expn
    return (3 * tau - 4 * expn(3, tau) + 6 * expn(4, tau)) / (4 + 3 * tau)


def two_stream_diffuse_reflectance(tau, ssa, g):
    """Reflectance of a layer under diffuse illumination: Heng & Kitzmann (2017) Eq. 1 with the transmission function 2E3(tau*sqrt((1-ssa)(1-ssa g))) of their Sect. 2, coupling coefficients of Heng et al. (2014) Eq. 72."""
    from scipy.special import expn
    if ssa >= 1.0:
        return (1 - g) * tau / (1 + (1 - g) * tau)
    ratio = math.sqrt((1 - ssa) / (1 - ssa * g))
    r = (1 - ratio) / (1 + ratio)
    T = 2 * expn(3, tau * math.sqrt((1 - ssa) * (1 - ssa * g)))
    return r * (1 - T * T) / (1 - r * r * T * T)


def model_spherical_albedo(tau_r, tau_a, ssa, g):
    """The model's spherical albedo: CSALBR for the Rayleigh part plus the two-stream change from adding the aerosol to the layer."""
    tau_t = tau_r + tau_a
    scattering = tau_r + ssa * tau_a
    return csalbr(tau_r) + two_stream_diffuse_reflectance(tau_t, scattering / tau_t, ssa * tau_a * g / scattering) - two_stream_diffuse_reflectance(tau_r, 1.0, 0.0)


def formula_check(processes):
    wavelengths_um = [0.35, 0.40, 0.45, 0.55, 0.67, 0.87, 1.24, 1.64, 2.13]
    aods = [0.05, 0.2, 0.5, 1.0]
    with Pool(processes) as pool:
        asymmetry = dict(zip(wavelengths_um, pool.map(sixs_aerosol_asymmetry, [1000 * w for w in wavelengths_um])))
        cases = [(1000 * w, aod) for w in wavelengths_um for aod in aods]
        results = dict(zip(cases, pool.map(sixs_albedo_case, cases)))
    print('\n== Spherical albedo formula with 6S\'s continental aerosol properties vs 6S ==')
    print('The model\'s formula, and for comparison the additive form of Cachorro et al. (2022) Eq. 18 (CSALBR plus the two-stream reflectance of the aerosol alone)')
    print('  wavelength  AOD550  tau_a   ssa    g      S 6S   | model   diff     GHI error rho=0.2  rho=0.8 | additive  diff     GHI error rho=0.2  rho=0.8')
    for (wavelength_nm, aod), r in results.items():
        g = asymmetry[wavelength_nm / 1000]
        cells = []
        for S in (model_spherical_albedo(r['tau_r'], r['tau_a'], r['ssa'], g), csalbr(r['tau_r']) + two_stream_diffuse_reflectance(r['tau_a'], r['ssa'], g)):
            ghi_error = [100 * ((1 - rho * r['S_total']) / (1 - rho * S) - 1) for rho in (0.2, 0.8)]
            cells.append(f'{S:.4f}  {S - r["S_total"]:+.4f}  {ghi_error[0]:+6.2f}%  {ghi_error[1]:+6.2f}%')
        print(f'  {wavelength_nm:6.0f} nm  {aod:5.2f}  {r["tau_a"]:.3f}  {r["ssa"]:.3f}  {g:.3f}  {r["S_total"]:.4f} | ' + ' | '.join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('build_dir', help='directory in which to build the validation driver')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--csv', help='write every case with Helios and 6S values to this file')
    parser.add_argument('--formula-check', action='store_true', help='also compare the spherical albedo formula, fed 6S\'s aerosol properties, with 6S')
    args = parser.parse_args()

    cases = [(case_id_for(sza, aod, rho), sza, SEA_LEVEL_PRESSURE_PA, TEMPERATURE_K, RELATIVE_HUMIDITY, aod, rho) for sza in SOLAR_ZENITHS_DEG for aod in AOD550_VALUES for rho in GROUND_ALBEDOS]
    metadata, spectra = run_driver(args.build_dir, cases, WAVELENGTHS_NM, args.processes)
    toa = read_toa_spectrum(os.path.abspath(args.build_dir))
    first = metadata[cases[0][0]]
    water_cm, ozone_atm_cm = first['water_cm'], first['ozone_DU'] / 1000.0
    print(f'{len(cases) * len(WAVELENGTHS_NM)} cases; water vapor {water_cm:.3f} cm, ozone {ozone_atm_cm:.3f} atm-cm, sea level, 6S continental aerosol', flush=True)

    sixs_cases = [(w, water_cm, ozone_atm_cm, c[5], c[1], c[6]) for c in cases for w in WAVELENGTHS_NM]
    with Pool(args.processes) as pool:
        sixs_results = dict(zip(sixs_cases, pool.map(run_sixs, sixs_cases, chunksize=16)))

    rows = []
    for case_id, sza, _, _, _, aod, rho in cases:
        for w in WAVELENGTHS_NM:
            h = helios_transmittances(metadata, spectra, toa, case_id, w)
            s = sixs_results[(w, water_cm, ozone_atm_cm, aod, sza, rho)]
            row = dict(wavelength_nm=w, sza=sza, aod550=aod, albedo=rho, **{f'helios_{k}': v for k, v in h.items()}, **{f'6s_{k}': v for k, v in s.items()})
            if rho > 0:
                global_black = spectra[(case_id_for(sza, aod, 0.0), w)][1]
                row['helios_S'] = (1 - global_black / spectra[(case_id, w)][1]) / rho
            rows.append(row)

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
            writer.writeheader()
            writer.writerows(rows)

    def table(title, key, relative, selector=lambda r: True):
        groups = defaultdict(list)
        for r in rows:
            if selector(r) and f'helios_{key}' in r:
                h, s = r[f'helios_{key}'], r[f'6s_{key}']
                groups[(r['wavelength_nm'], r['aod550'])].append(100 * (h - s) / s if relative else h - s)
        print(f'\n== {title} ==')
        print('  wavelength ' + ''.join(f'   AOD {aod:<14g}' for aod in AOD550_VALUES))
        for w in WAVELENGTHS_NM:
            cells = []
            for aod in AOD550_VALUES:
                d = groups.get((w, aod), [])
                if d:
                    worst = max(d, key=abs)
                    cells.append(f'{sum(abs(x) for x in d) / len(d):7.3f} / {worst:+8.3f}')
                else:
                    cells.append(' ' * 19)
            print(f'  {w:6d} nm ' + '  '.join(cells))

    print('\nEach cell: mean |difference| / signed difference of largest magnitude, over the solar zenith angles (and ground albedos where applicable)')
    table('Direct normal transmittance, % difference from 6S (ground albedo 0)', 'T_direct', True, lambda r: r['albedo'] == 0.0)
    table('Global horizontal transmittance, % difference from 6S (ground albedo 0.2)', 'T_global', True, lambda r: r['albedo'] == 0.2)
    table('Global horizontal transmittance, % difference from 6S (ground albedo 0.8)', 'T_global', True, lambda r: r['albedo'] == 0.8)
    table('Diffuse horizontal transmittance, % difference from 6S (ground albedo 0.2)', 'T_diffuse', True, lambda r: r['albedo'] == 0.2)
    for sza in SOLAR_ZENITHS_DEG:
        table(f'Diffuse horizontal transmittance at solar zenith {sza:g} deg, % difference from 6S (ground albedo 0.2)', 'T_diffuse', True, lambda r, sza=sza: r['albedo'] == 0.2 and r['sza'] == sza)
    table('Spherical albedo, absolute difference from 6S', 'S', False, lambda r: r['albedo'] == 0.2 and r['sza'] == 30.0)

    if args.formula_check:
        formula_check(args.processes)


if __name__ == '__main__':
    sys.exit(main())
