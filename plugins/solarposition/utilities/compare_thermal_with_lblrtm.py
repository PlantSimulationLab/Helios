#!/usr/bin/env python3
"""Compare the thermal infrared radiative transfer behind SolarPosition's thermal atmosphere look-up table (libRadtran
uvspec with REPTRAN fine, see generate_thermal_atmosphere_lut.py) with the line-by-line model LBLRTM of Atmospheric and
Environmental Research (AER), for the same atmospheric profiles.

Both codes are given exactly the same profile (levels, temperatures and gas mixing ratios, built by
generate_thermal_atmosphere_lut.build_profile() with the water vapor and ozone columns scaled as uvspec scales them), a
black surface at the surface air temperature and a sensor at the top of the atmosphere, so the comparison isolates the
molecular absorption and the radiative transfer. LBLRTM's monochromatic radiance and transmittance are averaged over the
same 5 cm-1 bins as the table with a rectangular scanning function. The comparison is reported as the difference in
brightness temperature at the top of the atmosphere integrated over sensor bands (boxcar in wavelength).

LBLRTM is used here only as an offline reference; neither its code nor its output is distributed with Helios. Its licence
permits use for scientific and research purposes.

Requirements:
  - the libRadtran environment of generate_thermal_atmosphere_lut.py (uvspec with REPTRAN fine thermal data)
  - an LBLRTM executable (https://github.com/AER-RC/LBLRTM, v12.17), a TAPE3 line file made by LNFL
    (https://github.com/AER-RC/LNFL) from the AER line file (v3.8.1) covering at least 625-1845 cm-1, and the MT_CKD
    continuum file absco-ref_wv-mt-ckd.nc distributed with LBLRTM (LBLRTM/data/)

Usage:
    python compare_thermal_with_lblrtm.py --lblrtm <lblrtm executable> --tape3 <TAPE3> --continuum <absco-ref_wv-mt-ckd.nc>
        [--uvspec PATH] [--processes N] [--csv results.csv]
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

import generate_thermal_atmosphere_lut as lut_generator

AVOGADRO = 6.02214076e23
LOSCHMIDT_DU = 2.6867e16  # molecules/cm2 per Dobson unit
WATER_MOLAR_MASS = 18.015
BOLTZMANN = 1.380649e-23

# Sensor bands (nm), limits as published by the instrument teams; see SolarPosition.dox for the sources
BANDS = {
    'GOES ABI 8': (5770, 6600), 'GOES ABI 10': (7240, 7440), 'MODIS 27': (6535, 6895), 'MODIS 29': (8400, 8700), 'ASTER 10': (8125, 8475),
    'ASTER 12': (8925, 9275), 'GOES ABI 12': (9420, 9800), 'MODIS 30': (9580, 9880), 'ASTER 13': (10250, 10950), 'MODIS 31': (10780, 11280),
    'Landsat 8/9 TIRS 10': (10600, 11190), 'Landsat 8/9 TIRS 11': (11500, 12510), 'MODIS 32': (11770, 12270), 'MODIS 33': (13185, 13485),
    'MODIS 36': (14085, 14385), 'GOES ABI 16': (13000, 13600), 'Microbolometer 7.5-13.5 um': (7500, 13500),
}

# Validation cases: (label, column water vapor g/cm2, surface air temperature K, surface pressure hPa, ozone DU)
CASES = [
    ('subarctic winter-like', 0.4, 257.2, 1013.0, 380.0),
    ('midlatitude winter-like', 0.85, 272.2, 1013.0, 380.0),
    ('U.S. Standard-like', 1.4, 288.2, 1013.0, 345.0),
    ('midlatitude summer-like', 3.0, 294.2, 1013.0, 335.0),
    ('tropical-like', 4.2, 299.7, 1013.0, 285.0),
    ('dry elevated', 0.5, 280.0, 795.0, 300.0),
    ('humid hot', 6.0, 305.0, 1013.0, 260.0),
]
VIEW_ZENITH_DEG = [0.0, 60.0]


def planck_per_cm(wavenumber, temperature):
    """Planck radiance in W/m2/sr/cm-1."""
    return 1.191042972e-8 * wavenumber ** 3 / np.expm1(1.438776877 * wavenumber / temperature)


def scale_profile(profile, water_vapor_cm, ozone_du):
    """Scale the water vapor and ozone mixing ratios to the given columns above the surface, as uvspec's mol_modify does."""
    air = profile['p'] * 100.0 / (BOLTZMANN * profile['t']) * 1e-6  # cm-3
    z_cm = profile['z'] * 1e5
    scaled = dict(profile, vmr=dict(profile['vmr']))
    water_column = np.trapezoid(air * profile['vmr']['h2o'], z_cm) * WATER_MOLAR_MASS / AVOGADRO
    ozone_column = np.trapezoid(air * profile['vmr']['o3'], z_cm) / LOSCHMIDT_DU
    scaled['vmr']['h2o'] = profile['vmr']['h2o'] * water_vapor_cm / water_column
    scaled['vmr']['o3'] = profile['vmr']['o3'] * ozone_du / ozone_column
    return scaled


def lblrtm_tape5(profile, surface_temperature, view_zenith_deg):
    """LBLRTM input for the radiance at the top of the profile looking down at a black surface, scanned with a 5 cm-1
    rectangle sampled at the bin centers, with the radiance and transmittance written as ASCII to TAPE27 and TAPE28."""
    z = profile['z']
    top = z[-1]
    first_bin = lut_generator.WAVENUMBER_MIN + 0.5 * lut_generator.BIN_WIDTH
    last_bin = lut_generator.WAVENUMBER_MAX - 0.5 * lut_generator.BIN_WIDTH
    lines = ['$ Helios thermal atmosphere validation',
             ' HI=1 F4=1 CN=1 AE=0 EM=1 SC=1 FI=0 PL=0 TS=0 AM=1 MG=0 LA=0 OD=0 XS=0    0    0',
             f'{lut_generator.WAVENUMBER_MIN - 10:10.3f}{lut_generator.WAVENUMBER_MAX + 10:10.3f}',
             f'{surface_temperature:10.3f}{1.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}{0.0:10.3f}',
             # Record 3.1: user profile, slant path H1 to H2, IBMAX boundaries, 7 molecules
             f'{0:5d}{2:5d}{len(z):5d}{1:5d}{1:5d}{7:5d}{0:5d}',
             # Record 3.2: observer at the top, looking down to the surface
             f'{top:10.4f}{z[0]:10.4f}{180.0 - view_zenith_deg:10.4f}']
    for start in range(0, len(z), 8):
        lines.append(''.join(f'{value:10.4f}' for value in z[start:start + 8]))
    lines.append(f'{len(z):5d}     Helios profile')
    # Mixing ratios in ppmv of H2O, CO2, O3, N2O, CO, CH4, O2 (CO is not in the profile; REPTRAN applies the U.S. Standard CO, which LBLRTM also gets here)
    co_ppmv = 0.15
    for i in range(len(z)):
        lines.append(f'{z[i]:10.4f}{profile["p"][i]:10.4f}{profile["t"][i]:10.4f}     AA L AAAAAAA')
        values = [profile['vmr']['h2o'][i] * 1e6, profile['vmr']['co2'][i] * 1e6, profile['vmr']['o3'][i] * 1e6, profile['vmr']['n2o'][i] * 1e6, co_ppmv, profile['vmr']['ch4'][i] * 1e6,
                  profile['vmr']['o2'][i] * 1e6]
        lines.append(''.join(f'{value:15.7e}' for value in values))
    # Record 8.1: rectangular scans of the radiance (to TAPE11) and the transmittance (to TAPE13), half width 2.5 cm-1, output every 5 cm-1 at the bin centers (one bin beyond the
    # last, whose scan is incomplete at the end of the file, is discarded)
    for emit, unit in ((1, 11), (0, 13)):
        lines.append(f'{0.5 * lut_generator.BIN_WIDTH:10.3f}{first_bin:10.3f}{last_bin + lut_generator.BIN_WIDTH:10.3f}{emit:5d}{0:5d}{0:5d}{-float(lut_generator.BIN_WIDTH):10.4f}{12:5d}{1:5d}{1:5d}{unit:5d}{0:5d}')
    lines.append('-1.')
    lines.append('$ Transfer to ASCII')
    lines.append(' HI=0 F4=0 CN=0 AE=0 EM=0 SC=0 FI=0 PL=1 TS=0 AM=0 MG=0 LA=0 MS=0 XS=0    0    0')
    lines.append('# plot')
    for emit, scanned_unit, unit in ((1, 11, 27), (0, 13, 28)):
        lines.append(f'{first_bin:10.4f}{last_bin + lut_generator.BIN_WIDTH:10.4f}{10.2:10.4f}{100.0:10.4f}{5:5d}{0:5d}{scanned_unit:5d}{0:5d}{1.0:10.3f}{0:2d}{0:3d}{0:5d}')
        lines.append(f'{0.0:10.4f}{1.2:10.4f}{7.02:10.4f}{0.2:10.4f}{4:5d}{0:5d}{1:5d}{emit:5d}{0:5d}{0:5d}{0:2d}{3:5d}{unit:3d}')
    lines.append('-1.')
    lines.append('%%%%%')
    return '\n'.join(lines) + '\n'


def read_lblrtm_ascii(path):
    values = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    values.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    pass
    return np.array(values)


def run_lblrtm(lblrtm, tape3, continuum, profile, surface_temperature, view_zenith_deg):
    """Bin-averaged radiance at the top of the atmosphere (W/m2/sr/cm-1) and transmittance, in increasing wavenumber."""
    with tempfile.TemporaryDirectory(prefix='helios_lblrtm_') as directory:
        os.symlink(tape3, os.path.join(directory, 'TAPE3'))
        os.symlink(continuum, os.path.join(directory, 'absco-ref_wv-mt-ckd.nc'))
        with open(os.path.join(directory, 'TAPE5'), 'w') as f:
            f.write(lblrtm_tape5(profile, surface_temperature, view_zenith_deg))
        result = subprocess.run([lblrtm], cwd=directory, capture_output=True, text=True)
        radiance_path, transmittance_path = os.path.join(directory, 'TAPE27'), os.path.join(directory, 'TAPE28')
        if not (os.path.exists(radiance_path) and os.path.exists(transmittance_path)):
            raise RuntimeError(f'LBLRTM failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}')
        bin_count = round((lut_generator.WAVENUMBER_MAX - lut_generator.WAVENUMBER_MIN) / lut_generator.BIN_WIDTH)
        radiance = read_lblrtm_ascii(radiance_path)[:bin_count]
        transmittance = read_lblrtm_ascii(transmittance_path)[:bin_count]
    if len(radiance) != bin_count or len(transmittance) != bin_count:
        raise RuntimeError(f'LBLRTM returned {len(radiance)} radiance and {len(transmittance)} transmittance values; {bin_count} were expected')
    return radiance[:, 0], radiance[:, 1] * 1e4, transmittance[:, 1]


def uvspec_case(args):
    uvspec, data_path, profile, water_vapor_cm, ozone_du = args
    with tempfile.TemporaryDirectory(prefix='helios_thermal_') as directory:
        lut_generator.write_profile_files(profile, directory)
        runs = [lut_generator.run_uvspec(uvspec, data_path, directory, water_vapor_cm, ozone_du, t, VIEW_ZENITH_DEG) for t in lut_generator.SURFACE_TEMPERATURES_K]
    return runs


def lblrtm_case(args):
    lblrtm, tape3, continuum, profile, surface_temperature, view_zenith_deg = args
    return run_lblrtm(lblrtm, tape3, continuum, profile, surface_temperature, view_zenith_deg)


def band_brightness_temperature(wavenumber, radiance, wavelength_min, wavelength_max):
    """Brightness temperature of the radiance (per cm-1, on bins of BIN_WIDTH centered at wavenumber) integrated over a
    boxcar band in wavelength; partial bins are weighted by their overlap."""
    half = 0.5 * lut_generator.BIN_WIDTH
    overlap = np.clip(np.minimum(wavenumber + half, 1e7 / wavelength_min) - np.maximum(wavenumber - half, 1e7 / wavelength_max), 0.0, None)
    target = np.sum(overlap * radiance)
    low, high = 150.0, 400.0
    for _ in range(60):
        middle = 0.5 * (low + high)
        if np.sum(overlap * planck_per_cm(wavenumber, middle)) < target:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--lblrtm', required=True)
    parser.add_argument('--tape3', required=True)
    parser.add_argument('--continuum', required=True)
    parser.add_argument('--uvspec', default=None)
    parser.add_argument('--libradtran-data', default=None)
    parser.add_argument('--processes', type=int, default=os.cpu_count())
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()

    uvspec = args.uvspec or lut_generator.default_uvspec()
    data_path = args.libradtran_data or lut_generator.default_data_path(uvspec)
    base = lut_generator.load_base_profile(data_path)

    profiles = []
    for label, water_vapor_cm, air_temperature, pressure_hPa, ozone_du in CASES:
        temperatures = lut_generator.climate_surface_temperatures(base, pressure_hPa)
        climate_coordinate = float(np.interp(air_temperature, temperatures, lut_generator.CLIMATE_COORDINATES))
        profile = lut_generator.build_profile(base, climate_coordinate, pressure_hPa)
        profiles.append(scale_profile(profile, water_vapor_cm, ozone_du))

    with Pool(args.processes) as pool:
        uvspec_results = pool.map(uvspec_case, [(uvspec, data_path, profile, case[1], case[4]) for profile, case in zip(profiles, CASES)])
        lblrtm_jobs = [(os.path.abspath(args.lblrtm), os.path.abspath(args.tape3), os.path.abspath(args.continuum), profile, case[2], view) for profile, case in zip(profiles, CASES) for view in VIEW_ZENITH_DEG]
        lblrtm_results = pool.map(lblrtm_case, lblrtm_jobs, chunksize=1)

    rows = []
    for index, (case, runs) in enumerate(zip(CASES, uvspec_results)):
        label, water_vapor_cm, air_temperature, pressure_hPa, ozone_du = case
        wavenumbers, surface_1, toa_1, _ = runs[0]
        _, surface_2, toa_2, _ = runs[1]
        edges = np.arange(lut_generator.WAVENUMBER_MIN, lut_generator.WAVENUMBER_MAX + 1e-9, lut_generator.BIN_WIDTH)
        bin_index = np.searchsorted(edges, wavenumbers) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        for iv, view in enumerate(VIEW_ZENITH_DEG):
            # uvspec radiance over a black surface at the air temperature, from the two-temperature separation
            tau = (toa_1[:, iv] - toa_2[:, iv]) / (surface_1 - surface_2)
            upwelling = toa_1[:, iv] - tau * surface_1
            uvspec_radiance = np.bincount(bin_index, weights=tau * planck_per_cm(wavenumbers, air_temperature) + upwelling) / lut_generator.BIN_WIDTH
            uvspec_transmittance = np.bincount(bin_index, weights=tau) / lut_generator.BIN_WIDTH
            lbl_wavenumber, lbl_radiance, lbl_transmittance = lblrtm_results[index * len(VIEW_ZENITH_DEG) + iv]
            if len(lbl_wavenumber) != len(centers) or np.max(np.abs(lbl_wavenumber - centers)) > 0.05:
                raise RuntimeError('LBLRTM output is not on the bin centers (its scan spacing may differ from 5 cm-1 by up to 1e-5 relative)')
            for band, (wavelength_min, wavelength_max) in BANDS.items():
                bt_uvspec = band_brightness_temperature(centers, uvspec_radiance, wavelength_min, wavelength_max)
                bt_lbl = band_brightness_temperature(centers, lbl_radiance, wavelength_min, wavelength_max)
                in_band = (centers > 1e7 / wavelength_max) & (centers < 1e7 / wavelength_min)
                rows.append((label, view, band, bt_uvspec, bt_lbl, float(np.mean(uvspec_transmittance[in_band])), float(np.mean(lbl_transmittance[in_band]))))

    print(f'{"case":24s} {"VZA":>4s} {"band":28s} {"BT uvspec":>9s} {"BT LBLRTM":>9s} {"diff":>6s} {"tau uvspec":>10s} {"tau LBLRTM":>10s}')
    for label, view, band, bt_uvspec, bt_lbl, tau_uvspec, tau_lbl in rows:
        print(f'{label:24s} {view:4.0f} {band:28s} {bt_uvspec:9.2f} {bt_lbl:9.2f} {bt_uvspec - bt_lbl:+6.2f} {tau_uvspec:10.4f} {tau_lbl:10.4f}')
    print('\nPer band over all cases and view angles: mean and largest |BT uvspec - BT LBLRTM| (K)')
    for band in BANDS:
        differences = np.array([row[3] - row[4] for row in rows if row[2] == band])
        print(f'  {band:28s} mean {np.mean(np.abs(differences)):.2f}  max {np.max(np.abs(differences)):.2f}  bias {np.mean(differences):+.2f}')
    if args.csv:
        with open(args.csv, 'w') as f:
            f.write('case,view_zenith_deg,band,bt_uvspec_K,bt_lblrtm_K,transmittance_uvspec,transmittance_lblrtm\n')
            for row in rows:
                f.write(','.join(str(value) for value in row) + '\n')


if __name__ == '__main__':
    sys.exit(main())
