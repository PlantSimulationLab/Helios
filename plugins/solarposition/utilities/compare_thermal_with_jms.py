#!/usr/bin/env python3
"""Compare SolarPosition's thermal atmosphere (look-up table computed with libRadtran) with the generalized atmospheric
functions of Jiménez-Muñoz and Sobrino (2003), which SolarPosition used before the table, within their 10-12 um range.

Jiménez-Muñoz, J. C., and J. A. Sobrino (2003), A generalized single-channel method for retrieving land surface temperature
from remote sensing data, J. Geophys. Res., 108(D22), 4688, doi:10.1029/2003JD003480; correction of Table 2 (the sign of
the constant of chi_2): J. Geophys. Res., 109, doi:10.1029/2004JD004804.

Their functions depend only on the column water vapor and the effective wavelength; they were fitted to MODTRAN 3.5
simulations of about 60 TIGR radiosoundings at nadir without aerosol. The comparison is made at nadir for sensor bands whose
effective wavelength lies within 10-12.065 um, as the surface temperature that the functions retrieve from the radiance
Helios gives at the top of the atmosphere over a blackbody at the air temperature, minus that temperature: the quantity for
which the authors report their accuracy (root-mean-square errors below 1.5 K for Landsat TM band 6 and below 2 K for AVHRR
channel 4 and ATSR-2 channel 2). The sweep covers water vapor columns of 0.25-6 g/cm2 and air temperatures of 270-310 K at
relative humidities of 0.2-0.9 (the functions have no temperature dependence; the table does). The authors' Landsat TM
band 6 fit (their Eq. 15) is compared as well.

Requirements: CMake and a C++ compiler, and a thermal look-up table in plugins/solarposition/assets/thermal_atmosphere_lut/.

Usage:
    python compare_thermal_with_jms.py <build_dir> [--processes N] [--csv results.csv]
"""

import argparse
import math
import os
import sys

import numpy as np

from compare_thermal_with_uvspec import build_driver, run_driver, planck_per_nm

# Bands (nm) within the functions' range of effective wavelengths (the center of a boxcar band)
BANDS = {'ASTER 13': (10250, 10950), 'ASTER 14': (10950, 11650), 'MODIS 31': (10780, 11280), 'MODIS 32': (11770, 12270), 'Landsat 8/9 TIRS 10': (10600, 11190),
         'Landsat 8/9 TIRS 11': (11500, 12510), 'Landsat TM/ETM+ 6': (10400, 12500)}

# Table 2 of Jiménez-Muñoz and Sobrino (2003) with the 2004 correction: for each function psi_k, the coefficients of w^3, w^2, w, 1, each a cubic in wavelength (um), {a3, a2, a1, a0}
COEFFICIENTS = [[[0.00090, -0.01638, 0.04745, 0.27436], [0.00032, -0.06148, 1.2021, -6.2051], [0.00986, -0.23672, 1.7133, -3.2199], [-0.15431, 5.2757, -60.1170, 229.3139]],
                [[-0.02883, 0.87181, -8.82712, 29.9092], [0.13515, -4.1171, 41.8295, -142.2782], [-0.22765, 6.8606, -69.2577, 233.0722], [0.41868, -14.3299, 163.6681, -623.5300]],
                [[0.00182, -0.04519, 0.32652, -0.60030], [-0.00744, 0.11431, 0.17560, -5.4588], [-0.00269, 0.31395, -5.5916, 27.9913], [-0.07972, 2.8396, -33.6843, 132.9798]]]


def jms_functions(wavelength_um, water_vapor_cm):
    """Transmittance and upwelling radiance (W/m2/sr/um) of the generalized functions."""
    psi = []
    for k in range(3):
        value = 0.0
        for a in COEFFICIENTS[k]:
            value = value * water_vapor_cm + ((a[0] * wavelength_um + a[1]) * wavelength_um + a[2]) * wavelength_um + a[3]
        psi.append(value)
    transmittance = 1.0 / psi[0]
    return transmittance, -transmittance * (psi[1] + psi[2])


def tm6_functions(water_vapor_cm):
    """Landsat TM band 6 fit of Jiménez-Muñoz and Sobrino (2003, Eq. 15)."""
    w = water_vapor_cm
    psi1 = 0.14714 * w * w - 0.15583 * w + 1.1234
    psi2 = -1.1836 * w * w - 0.37607 * w - 0.52894
    psi3 = -0.04554 * w * w + 1.8719 * w - 0.39071
    transmittance = 1.0 / psi1
    return transmittance, -transmittance * (psi2 + psi3)


def band_mean_planck(wavelength_min, wavelength_max, temperature):
    """Planck radiance averaged over a boxcar band, W/m2/sr/um."""
    grid = np.arange(wavelength_min, wavelength_max + 0.5, 1.0)
    return np.trapezoid(planck_per_nm(grid, temperature), grid) / (wavelength_max - wavelength_min) * 1e3


def band_temperature(wavelength_min, wavelength_max, radiance_per_um):
    """Temperature whose band-mean Planck radiance is radiance_per_um."""
    low, high = 150.0, 400.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if band_mean_planck(wavelength_min, wavelength_max, middle) < radiance_per_um:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def helios_band_radiance(result, wavelength_min, wavelength_max, surface_temperature):
    """Band-mean radiance (W/m2/sr/um) at the top of the atmosphere over a blackbody, from Helios's spectra."""
    grid = np.arange(wavelength_min, wavelength_max + 0.5, 1.0)
    wavelength = result['wavelength_nm']
    radiance = np.interp(grid, wavelength, result['transmittance']) * planck_per_nm(grid, surface_temperature) + np.interp(grid, wavelength, result['upwelling_radiance'])
    return np.trapezoid(radiance, grid) / (wavelength_max - wavelength_min) * 1e3


def humidity_for_water_vapor(water_vapor_cm, air_temperature):
    """Relative humidity at which SolarPosition's calculatePrecipitableWater() gives the water vapor column (inverse of its Magnus dew point and Viswanadham 1981 expression)."""
    dew_point_F = (math.log(water_vapor_cm) - (0.1133 - math.log(5.0))) / 0.0393
    dew_point_C = (dew_point_F - 32.0) * 5.0 / 9.0
    gamma = 17.67 * dew_point_C / (243.0 + dew_point_C)
    return math.exp(gamma - 17.67 * (air_temperature - 273.0) / 268.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('build_dir')
    parser.add_argument('--processes', type=int, default=os.cpu_count())
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()
    build_dir = os.path.abspath(args.build_dir)
    build_driver(build_dir, args.processes)

    states = []
    for water_vapor_cm in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0]:
        for air_temperature in [270.0, 280.0, 290.0, 300.0, 310.0]:
            humidity = humidity_for_water_vapor(water_vapor_cm, air_temperature)
            if 0.2 <= humidity <= 0.9:
                states.append((water_vapor_cm, air_temperature, humidity))
    results = run_driver(build_dir, [(t, rh, 101325.0, 300.0, 0.0) for _, t, rh in states])

    # Surface temperature retrieved with the functions (emissivity one) from Helios's radiance over a blackbody at the air temperature, minus the air temperature
    rows = []
    for (water_vapor_cm, air_temperature, humidity), result in zip(states, results):
        for band, (wavelength_min, wavelength_max) in BANDS.items():
            radiance = helios_band_radiance(result, wavelength_min, wavelength_max, air_temperature)
            functions = {'generalized': jms_functions(0.5 * (wavelength_min + wavelength_max) * 1e-3, result['water_vapor_cm'])}
            if band == 'Landsat TM/ETM+ 6':
                functions['TM6 fit'] = tm6_functions(result['water_vapor_cm'])
            for name, (transmittance, upwelling) in functions.items():
                retrieved = band_temperature(wavelength_min, wavelength_max, (radiance - upwelling) / transmittance)
                rows.append((result['water_vapor_cm'], air_temperature, band, name, retrieved - air_temperature))

    print('Surface temperature retrieved with the functions of Jiménez-Muñoz and Sobrino from Helios radiance at nadir, minus the true (air) temperature (K)')
    print(f'{"band":22s} {"functions":12s} {"w (g/cm2)":>9s} ' + ' '.join(f'{"T=" + format(t, ".0f"):>7s}' for t in [270, 280, 290, 300, 310]))
    for band in BANDS:
        for functions in ('generalized', 'TM6 fit'):
            selected = [row for row in rows if row[2] == band and row[3] == functions]
            if not selected:
                continue
            for water_vapor_cm in sorted({round(row[0], 3) for row in selected}):
                cells = []
                for air_temperature in [270.0, 280.0, 290.0, 300.0, 310.0]:
                    match = [row for row in selected if abs(row[0] - water_vapor_cm) < 1e-3 and row[1] == air_temperature]
                    cells.append(f'{match[0][4]:+7.2f}' if match else f'{"":>7s}')
                print(f'{band:22s} {functions:12s} {water_vapor_cm:9.2f} ' + ' '.join(cells))
    print('\nRoot-mean-square and largest |retrieval error| (K), for water vapor below 3 g/cm2 and for all states')
    for band in BANDS:
        for functions in ('generalized', 'TM6 fit'):
            errors = [(row[0], row[4]) for row in rows if row[2] == band and row[3] == functions]
            if errors:
                dry = np.array([e for w, e in errors if w < 3.0])
                wet = np.array([e for _, e in errors])
                print(f'  {band:22s} {functions:12s} w<3: rms {np.sqrt(np.mean(dry ** 2)):.2f} max {np.max(np.abs(dry)):.2f}   all: rms {np.sqrt(np.mean(wet ** 2)):.2f} max {np.max(np.abs(wet)):.2f}')
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('water_vapor_cm,air_temperature_K,band,functions,retrieval_error_K\n')
            for row in rows:
                f.write(','.join(str(value) for value in row) + '\n')


if __name__ == '__main__':
    sys.exit(main())
