#!/usr/bin/env python3
"""Generate the thermal infrared atmospheric look-up table used by SolarPosition::calculateSensorThermalAtmosphere() and
SolarPosition::getThermalSkyFlux().

The table is computed with the uvspec radiative transfer tool of libRadtran, using the REPTRAN fine (1 cm-1) molecular
absorption parameterization (Gasteiger et al. 2014), for a clear sky without aerosol, a black surface and a sensor at the
top of the atmosphere. Every quantity is averaged over 5 cm-1 bins between 650 and 1820 cm-1 (5.49-15.38 um) and stored
against the bin-center wavelength in nm, with radiances per nm.

Atmospheric state axes:
    log_water_vapor_cm   natural logarithm of the column water vapor between the surface and the top of the atmosphere (g/cm2)
    climate_coordinate   position in the climatology of base atmospheres (see build_profile()); it is mapped from the
                         surface air temperature with the table climate_temperature_K
    pressure_hPa         surface pressure
    view_secant          1/cos(view zenith angle)
    ozone_DU             ozone column

Tables:
    climate_temperature_K(pressure_hPa, climate_coordinate)
        Air temperature at the surface of the profile at each climate coordinate node, for mapping the surface air
        temperature to the climate coordinate. It increases with the coordinate and is linear in it between nodes; between
        the pressure nodes it is interpolated linearly in log pressure, and above the largest pressure it is constant.
    thermal_transmittance(log_water_vapor_cm, climate_coordinate, pressure_hPa, view_secant, wavelength_nm)
        Transmittance between the surface and the top of the atmosphere, at the reference ozone column.
    thermal_upwelling_radiance(log_water_vapor_cm, climate_coordinate, pressure_hPa, view_secant, wavelength_nm)
        Radiance emitted by the atmosphere along the path and reaching the top of the atmosphere, W/m2/sr/nm, at the
        reference ozone column.
    thermal_ozone_log_transmittance_ratio(climate_coordinate, pressure_hPa, ozone_DU, view_secant, wavelength_nm)
    thermal_ozone_upwelling_radiance(climate_coordinate, pressure_hPa, ozone_DU, view_secant, wavelength_nm)
        Ozone correction. Most ozone lies above the water vapor, so for an ozone column other than the reference the
        radiance at the top of the atmosphere is R*(tau*B + L_up) + D, with tau and L_up from the tables above, R the
        transmittance ratio (tabulated as ln R, which is close to linear in the ozone column) and D the upwelling
        radiance term (W/m2/sr/nm). R and D are computed at the column water vapor OZONE_REFERENCE_WATER_VAPOR_CM, where
        the correction is exact; they are nearly independent of the water vapor.
    thermal_downwelling_radiance(log_water_vapor_cm, climate_coordinate, pressure_hPa, ozone_DU, wavelength_nm)
        Downwelling irradiance from the sky at the surface divided by pi, W/m2/sr/nm.

Atmospheric profiles (build_profile()): the atmospheres of Anderson et al. (1986) in BASE_ATMOSPHERES, in order of
increasing surface temperature, define a climatology with climate coordinate 0, 1, ..., 4. Between integer coordinates the
temperature, water vapor and ozone profiles are interpolated linearly between the two neighboring atmospheres; below 0 and
above 4 the coldest or warmest atmosphere is used with its temperature shifted by CLIMATE_SHIFT_K per unit of the
coordinate (in full up to 10 km, tapered linearly to zero at 15 km). The profile is then cut at the surface pressure,
its heights recomputed hydrostatically, its water vapor and ozone scaled to the given columns, and CO2, CH4 and N2O scaled
to the NOAA Global Monitoring Laboratory global annual means of GREENHOUSE_GAS_YEAR (the Anderson et al. values are those
of 1986).

Transmittance and upwelling radiance are separated by running each atmosphere at two surface temperatures: without
scattering, the radiance at the top of the atmosphere is tau*B(T_surface) + L_up exactly, so tau = (L1 - L2)/(B1 - B2) and
L_up = L1 - tau*B1, where B is uvspec's own upward radiance at the surface. For a 5 cm-1 bin the same expressions are
applied to the sums over its five 1 cm-1 bands.

Requirements: libRadtran 2.0.6 with the REPTRAN fine thermal data, e.g.
    conda create -n helios-libradtran -c conda-forge rubin-libradtran python=3.12 numpy
then download reptran_2024_all.tar.gz from the libRadtran download page (https://www.libradtran.org) and extract its
data/correlated_k/reptran/reptran_thermal_fine* files into <env>/share/libRadtran/data/correlated_k/reptran/.

Usage:
    python generate_thermal_atmosphere_lut.py <output.bin> [--processes N] [--uvspec PATH] [--libradtran-data DIR]
"""

import argparse
import datetime
import math
import os
import shutil
import subprocess
import sys
import tempfile
from multiprocessing import Pool

import numpy as np

from helios_lut import write_lut

# Spectral bins (cm-1). REPTRAN fine bands are 1 cm-1 wide; each bin averages BIN_WIDTH of them.
WAVENUMBER_MIN = 650
WAVENUMBER_MAX = 1820
BIN_WIDTH = 5

# Atmospheric state axes
WATER_VAPOR_CM = [0.1, 0.15, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.5, 7.5]
CLIMATE_COORDINATES = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
# Surface altitudes (km) of the pressure nodes: levels of the U.S. Standard atmosphere, the same pressures as the 6S table
SURFACE_ALTITUDES_KM = [0.0, 1.0, 2.0, 3.0]
VIEW_ZENITH_DEG = [0.0, 30.0, 40.0, 50.0, 60.0]
OZONE_DU = [150.0, 300.0, 450.0]
REFERENCE_OZONE_DU = 300.0
OZONE_REFERENCE_WATER_VAPOR_CM = 1.0

# Base atmospheres of Anderson et al. (1986), in order of increasing surface temperature: subarctic winter (257.2 K),
# midlatitude winter (272.2 K), U.S. Standard (288.2 K), midlatitude summer (294.2 K) and tropical (299.7 K). Subarctic
# summer (287.2 K) is left out because its surface temperature is that of the U.S. Standard atmosphere.
BASE_ATMOSPHERES = ['afglsw', 'afglmw', 'afglus', 'afglms', 'afglt']
# Temperature shift per unit of climate coordinate below the coldest and above the warmest base atmosphere (K)
CLIMATE_SHIFT_K = 11.0
# The shift is applied in full below the first height and tapered linearly to zero at the second (km)
TEMPERATURE_SHIFT_FULL_KM = 10.0
TEMPERATURE_SHIFT_ZERO_KM = 15.0

# Surface temperatures of the two runs used to separate transmittance from upwelling radiance
SURFACE_TEMPERATURES_K = (260.0, 320.0)

# NOAA Global Monitoring Laboratory global annual means (https://gml.noaa.gov/ccgg/trends/), mole fraction in dry air
GREENHOUSE_GAS_YEAR = 2025
CO2_PPM = 425.62
CH4_PPB = 1935.94
N2O_PPB = 338.85

BOLTZMANN = 1.380649e-23
GAS_CONSTANT_DRY_AIR = 287.05
GRAVITY = 9.80665


def default_uvspec():
    path = shutil.which('uvspec')
    if path is None:
        raise RuntimeError('uvspec not found on PATH; pass --uvspec')
    return path


def default_data_path(uvspec):
    path = os.path.join(os.path.dirname(os.path.dirname(uvspec)), 'share', 'libRadtran', 'data')
    if not os.path.isdir(path):
        raise RuntimeError(f'libRadtran data directory {path} not found; pass --libradtran-data')
    return path


def load_afgl(atmmod, name):
    """Load an atmosphere of Anderson et al. (1986) as bottom-up arrays of height (km), pressure (hPa), temperature (K) and volume mixing ratios."""
    table = np.loadtxt(os.path.join(atmmod, f'{name}.dat'))[::-1]
    z, p, t, air = table[:, 0], table[:, 1], table[:, 2], table[:, 3]
    vmr = {species: table[:, column] / air for species, column in (('o3', 4), ('o2', 5), ('h2o', 6), ('co2', 7), ('no2', 8))}
    return dict(z=z, p=p, t=t, vmr=vmr)


def load_base_profile(data_path):
    """The climatology from which profiles are built: the atmospheres of BASE_ATMOSPHERES interpolated to the pressure levels
    of the U.S. Standard atmosphere. CH4 and N2O profiles (not in the atmosphere files) and the O2, CO2 and NO2 profiles are
    those of the U.S. Standard atmosphere for all of them."""
    atmmod = os.path.join(data_path, 'atmmod')
    standard = load_afgl(atmmod, 'afglus')
    log_p = np.log(standard['p'])
    for species in ('ch4', 'n2o'):
        profile = np.loadtxt(os.path.join(atmmod, f'afglus_{species}_vmr.dat'))[::-1]
        standard['vmr'][species] = np.interp(standard['z'], profile[:, 0], profile[:, 1])

    climates = []
    for name in BASE_ATMOSPHERES:
        atmosphere = load_afgl(atmmod, name)

        # Interpolate in log pressure (decreasing with height, so reversed for np.interp); levels outside the atmosphere's pressure range take its end values
        def to_standard_levels(values):
            return np.interp(log_p[::-1], np.log(atmosphere['p'])[::-1], values[::-1])[::-1]

        vmr = dict(standard['vmr'])
        vmr['h2o'] = to_standard_levels(atmosphere['vmr']['h2o'])
        vmr['o3'] = to_standard_levels(atmosphere['vmr']['o3'])
        climates.append(dict(name=name, t=to_standard_levels(atmosphere['t']), vmr=vmr))
    return dict(z=standard['z'], p=standard['p'], climates=climates, standard_vmr=standard['vmr'])


def surface_pressure_hPa(base, altitude_km):
    return float(np.exp(np.interp(altitude_km, base['z'], np.log(base['p']))))


def value_at_pressure(base, values, pressure_hPa):
    """Interpolate a profile on the base levels to a pressure, linearly in log pressure (end values outside the levels)."""
    log_p = np.log(base['p'])
    return float(np.interp(math.log(pressure_hPa), log_p[::-1], values[::-1]))


def climate_surface_temperatures(base, pressure_hPa):
    """Surface air temperature of the profile at each node of CLIMATE_COORDINATES."""
    temperatures = [build_profile(base, coordinate, pressure_hPa)['t'][0] for coordinate in CLIMATE_COORDINATES]
    if any(upper <= lower for lower, upper in zip(temperatures, temperatures[1:])):
        raise RuntimeError(f'Surface air temperature does not increase with the climate coordinate at {pressure_hPa} hPa: {temperatures}')
    return temperatures


def build_profile(base, climate_coordinate, pressure_hPa):
    """Profile with its bottom level at pressure_hPa, as bottom-up arrays, for a climate coordinate.

    At integer coordinates k = 0...4 the profile is base atmosphere k; between them the temperature, water vapor and ozone
    profiles are interpolated linearly. Below 0 and above 4 the coldest or warmest atmosphere is used with its temperature
    shifted by CLIMATE_SHIFT_K per unit of the coordinate beyond the end."""
    climates = base['climates']
    last = len(climates) - 1
    lower = int(np.clip(math.floor(climate_coordinate), 0, last - 1))
    fraction = float(np.clip(climate_coordinate - lower, 0.0, 1.0))
    t_levels = (1.0 - fraction) * climates[lower]['t'] + fraction * climates[lower + 1]['t']
    vmr_levels = {species: (1.0 - fraction) * climates[lower]['vmr'][species] + fraction * climates[lower + 1]['vmr'][species] for species in climates[0]['vmr']}
    shift = CLIMATE_SHIFT_K * (min(climate_coordinate, 0.0) + max(climate_coordinate - last, 0.0))

    above = base['p'] < pressure_hPa * (1.0 - 1e-6)
    z_original = np.concatenate(([value_at_pressure(base, base['z'], pressure_hPa)], base['z'][above]))
    p = np.concatenate(([pressure_hPa], base['p'][above]))
    t = np.concatenate(([value_at_pressure(base, t_levels, pressure_hPa)], t_levels[above]))
    vmr = {species: np.concatenate(([value_at_pressure(base, values, pressure_hPa)], values[above])) for species, values in vmr_levels.items()}

    weight = np.clip((TEMPERATURE_SHIFT_ZERO_KM - z_original) / (TEMPERATURE_SHIFT_ZERO_KM - TEMPERATURE_SHIFT_FULL_KM), 0.0, 1.0)
    t = t + shift * weight

    # Heights from the hypsometric equation with the (possibly shifted) temperatures, starting from the original surface height
    z = np.empty_like(p)
    z[0] = z_original[0]
    for i in range(1, len(p)):
        layer_temperature = 0.5 * (t[i - 1] + t[i])
        z[i] = z[i - 1] + GAS_CONSTANT_DRY_AIR * layer_temperature / GRAVITY * math.log(p[i - 1] / p[i]) * 1e-3

    vmr['co2'] = vmr['co2'] * (CO2_PPM * 1e-6 / base['standard_vmr']['co2'][0])
    vmr['ch4'] = vmr['ch4'] * (CH4_PPB * 1e-9 / base['standard_vmr']['ch4'][0])
    vmr['n2o'] = vmr['n2o'] * (N2O_PPB * 1e-9 / base['standard_vmr']['n2o'][0])
    return dict(z=z, p=p, t=t, vmr=vmr)


def write_profile_files(profile, directory):
    """Write the uvspec atmosphere file and the CH4 and N2O mol_files into directory."""
    air = profile['p'] * 100.0 / (BOLTZMANN * profile['t']) * 1e-6  # cm-3
    with open(os.path.join(directory, 'atmosphere.dat'), 'w') as f:
        f.write('# z(km) p(hPa) T(K) air o3 o2 h2o co2 no2 (cm-3)\n')
        for i in reversed(range(len(profile['p']))):
            densities = [air[i] * profile['vmr'][species][i] for species in ('o3', 'o2', 'h2o', 'co2', 'no2')]
            f.write(f"{profile['z'][i]:.6f} {profile['p'][i]:.8e} {profile['t'][i]:.5f} {air[i]:.8e} " + ' '.join(f'{d:.8e}' for d in densities) + '\n')
    for species in ('ch4', 'n2o'):
        with open(os.path.join(directory, f'{species}.dat'), 'w') as f:
            for i in reversed(range(len(profile['p']))):
                f.write(f"{profile['z'][i]:.6f} {profile['vmr'][species][i]:.8e}\n")


def run_uvspec(uvspec, data_path, directory, water_vapor_cm, ozone_du, surface_temperature, view_zenith_deg):
    """Run uvspec for the profile written in directory. Returns wavenumbers (band centers), the upward radiance at the
    surface, the upward radiance at the top of the atmosphere for each view angle and the downward irradiance at the
    surface, per cm-1, for the 1 cm-1 bands with centers inside WAVENUMBER_MIN-WAVENUMBER_MAX, in increasing wavenumber."""
    view_cosines = sorted(math.cos(math.radians(angle)) for angle in view_zenith_deg)
    inp = f"""data_files_path {data_path}
atmosphere_file {os.path.join(directory, 'atmosphere.dat')}
mol_file CH4 {os.path.join(directory, 'ch4.dat')} vmr
mol_file N2O {os.path.join(directory, 'n2o.dat')} vmr
mol_modify H2O {water_vapor_cm * 10.0:.6f} MM
mol_modify O3 {ozone_du:.6f} DU
source thermal
mol_abs_param reptran fine
wavelength {1e7 / WAVENUMBER_MAX:.4f} {1e7 / WAVENUMBER_MIN:.4f}
rte_solver disort
albedo 0
sur_temperature {surface_temperature:.4f}
zout sur toa
umu {' '.join(f'{c:.10f}' for c in view_cosines)}
phi 0
output_process per_cm-1
output_user wavenumber zout edn eup uu
quiet
"""
    result = subprocess.run([uvspec], input=inp, capture_output=True, text=True, cwd=directory)
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f'uvspec failed:\n{result.stderr}\ninput:\n{inp}')
    rows = np.loadtxt(result.stdout.splitlines())
    surface = rows[rows[:, 1] == 0.0]
    toa = rows[rows[:, 1] > 0.0]
    if len(surface) != len(toa) or not np.array_equal(surface[:, 0], toa[:, 0]):
        raise RuntimeError('uvspec surface and top-of-atmosphere rows do not match')
    order = np.argsort(surface[:, 0])
    surface, toa = surface[order], toa[order]
    keep = (surface[:, 0] > WAVENUMBER_MIN) & (surface[:, 0] < WAVENUMBER_MAX)
    surface, toa = surface[keep], toa[keep]
    # Columns: wavenumber, zout, edn, eup, then uu for each umu in increasing cosine; reorder to the requested view angles
    cosine_index = [view_cosines.index(math.cos(math.radians(angle))) for angle in view_zenith_deg]
    toa_radiance = toa[:, 4:][:, cosine_index]
    surface_radiance = surface[:, 4:][:, cosine_index]
    if not np.allclose(surface_radiance, surface_radiance[:, :1], rtol=1e-5, atol=0.0):
        raise RuntimeError('Upward radiance at a black surface differs between view angles')
    return surface[:, 0], surface_radiance[:, 0], toa_radiance, surface[:, 2]


def compute_state(uvspec, data_path, base, water_vapor_cm, climate_coordinate, pressure_hPa, ozone_du, view_zenith_deg):
    """Transmittance, upwelling radiance (per view angle) and downwelling radiance averaged over the spectral bins, in
    increasing wavenumber, with radiances per cm-1."""
    with tempfile.TemporaryDirectory(prefix='helios_thermal_') as directory:
        write_profile_files(build_profile(base, climate_coordinate, pressure_hPa), directory)
        runs = [run_uvspec(uvspec, data_path, directory, water_vapor_cm, ozone_du, surface_temperature, view_zenith_deg) for surface_temperature in SURFACE_TEMPERATURES_K]
    wavenumbers, surface_1, toa_1, downward_irradiance_1 = runs[0]
    _, surface_2, toa_2, downward_irradiance_2 = runs[1]
    # The downward irradiance at a black surface is emitted by the atmosphere; only Rayleigh scattering of surface emission (a relative contribution below 2e-4) depends on the surface temperature
    if not np.allclose(downward_irradiance_1, downward_irradiance_2, rtol=1e-3, atol=1e-12):
        raise RuntimeError('Downward irradiance at a black surface depends on the surface temperature')

    edges = np.arange(WAVENUMBER_MIN, WAVENUMBER_MAX + 1e-9, BIN_WIDTH)
    bin_index = np.searchsorted(edges, wavenumbers) - 1
    bin_count = len(edges) - 1
    if np.any(np.bincount(bin_index, minlength=bin_count) != BIN_WIDTH):
        raise RuntimeError('Every spectral bin must contain exactly BIN_WIDTH REPTRAN fine bands')

    def bin_sum(values):
        if values.ndim == 1:
            return np.bincount(bin_index, weights=values, minlength=bin_count)
        return np.stack([np.bincount(bin_index, weights=values[:, k], minlength=bin_count) for k in range(values.shape[1])], axis=1)

    # The bin transmittance keeps the bin's radiance at the top of the atmosphere exact for both surface temperatures
    transmittance = bin_sum(toa_1 - toa_2) / bin_sum(surface_1 - surface_2)[:, None]
    upwelling = (bin_sum(toa_1) - transmittance * bin_sum(surface_1)[:, None]) / BIN_WIDTH
    downwelling = bin_sum(0.5 * (downward_irradiance_1 + downward_irradiance_2)) / BIN_WIDTH / math.pi
    if np.any(transmittance < -1e-4) or np.any(transmittance > 1.0 + 1e-4):
        raise RuntimeError(f'Transmittance outside [0, 1]: {transmittance.min()} to {transmittance.max()}')
    if np.any(upwelling < -1e-7):
        raise RuntimeError(f'Negative upwelling radiance: {upwelling.min()}')
    return dict(wavenumbers=0.5 * (edges[:-1] + edges[1:]), transmittance=np.clip(transmittance, 0.0, 1.0), upwelling_radiance=np.maximum(upwelling, 0.0), downwelling_radiance=downwelling)


def to_wavelength(state):
    """Reorder bins to increasing wavelength and convert radiances from per cm-1 to per nm (at the bin center)."""
    order = np.argsort(-state['wavenumbers'])
    wavenumbers = state['wavenumbers'][order]
    per_nm = wavenumbers ** 2 * 1e-7
    return dict(wavelength_nm=1e7 / wavenumbers, transmittance=state['transmittance'][order], upwelling_radiance=state['upwelling_radiance'][order] * per_nm[:, None],
                downwelling_radiance=state['downwelling_radiance'][order] * per_nm)


def compute_case(case):
    uvspec, data_path, water_vapor_cm, climate_coordinate, pressure_hPa, ozone_du = case
    base = load_base_profile(data_path)
    return to_wavelength(compute_state(uvspec, data_path, base, water_vapor_cm, climate_coordinate, pressure_hPa, ozone_du, VIEW_ZENITH_DEG))


def compute_tables(uvspec, data_path, processes):
    base = load_base_profile(data_path)
    pressures = [surface_pressure_hPa(base, altitude) for altitude in SURFACE_ALTITUDES_KM]
    view_secant = [1.0 / math.cos(math.radians(angle)) for angle in VIEW_ZENITH_DEG]
    Nw, Nc, Np, No, Nv = len(WATER_VAPOR_CM), len(CLIMATE_COORDINATES), len(pressures), len(OZONE_DU), len(VIEW_ZENITH_DEG)
    reference_ozone_index = OZONE_DU.index(REFERENCE_OZONE_DU)

    # Every combination of water vapor, climate, pressure and ozone (the downwelling table needs them all); the ozone correction also needs the reference water vapor
    water_values = sorted(set(WATER_VAPOR_CM) | {OZONE_REFERENCE_WATER_VAPOR_CM})
    states = [(w, c, p, o) for w in water_values for c in CLIMATE_COORDINATES for p in pressures for o in OZONE_DU]
    print(f'{len(states)} atmospheres, two uvspec runs each', flush=True)
    with Pool(processes) as pool:
        results = dict(zip(states, pool.map(compute_case, [(uvspec, data_path) + state for state in states], chunksize=1)))
    wavelength_nm = results[states[0]]['wavelength_nm']
    Nl = len(wavelength_nm)

    transmittance = np.zeros((Nw, Nc, Np, Nv, Nl))
    upwelling = np.zeros((Nw, Nc, Np, Nv, Nl))
    downwelling = np.zeros((Nw, Nc, Np, No, Nl))
    for iw, w in enumerate(WATER_VAPOR_CM):
        for ic, c in enumerate(CLIMATE_COORDINATES):
            for ip, p in enumerate(pressures):
                reference = results[(w, c, p, REFERENCE_OZONE_DU)]
                transmittance[iw, ic, ip] = reference['transmittance'].T
                upwelling[iw, ic, ip] = reference['upwelling_radiance'].T
                for io, o in enumerate(OZONE_DU):
                    downwelling[iw, ic, ip, io] = results[(w, c, p, o)]['downwelling_radiance']

    ozone_log_ratio = np.zeros((Nc, Np, No, Nv, Nl))
    ozone_upwelling = np.zeros((Nc, Np, No, Nv, Nl))
    for ic, c in enumerate(CLIMATE_COORDINATES):
        for ip, p in enumerate(pressures):
            reference = results[(OZONE_REFERENCE_WATER_VAPOR_CM, c, p, REFERENCE_OZONE_DU)]
            for io, o in enumerate(OZONE_DU):
                modified = results[(OZONE_REFERENCE_WATER_VAPOR_CM, c, p, o)]
                # Where the atmosphere is opaque the ratio is undefined (or its logarithm is) and the correction is carried by D alone
                opaque = (reference['transmittance'] < 1e-6) | (modified['transmittance'] < 1e-6)
                ratio = np.where(opaque, 1.0, modified['transmittance'] / np.where(opaque, 1.0, reference['transmittance']))
                ozone_log_ratio[ic, ip, io] = np.log(ratio).T
                ozone_upwelling[ic, ip, io] = (modified['upwelling_radiance'] - ratio * reference['upwelling_radiance']).T
    if not (np.all(ozone_log_ratio[:, :, reference_ozone_index] == 0.0) and np.all(ozone_upwelling[:, :, reference_ozone_index] == 0.0)):
        raise RuntimeError('The ozone correction is not the identity at the reference ozone column')

    climate_temperature = np.array([climate_surface_temperatures(base, p) for p in pressures])
    log_water = [math.log(w) for w in WATER_VAPOR_CM]
    wavelengths = list(wavelength_nm)
    return [('climate_temperature_K', [('pressure_hPa', pressures), ('climate_coordinate', CLIMATE_COORDINATES)], climate_temperature),
            ('thermal_transmittance', [('log_water_vapor_cm', log_water), ('climate_coordinate', CLIMATE_COORDINATES), ('pressure_hPa', pressures), ('view_secant', view_secant), ('wavelength_nm', wavelengths)], transmittance),
            ('thermal_upwelling_radiance', [('log_water_vapor_cm', log_water), ('climate_coordinate', CLIMATE_COORDINATES), ('pressure_hPa', pressures), ('view_secant', view_secant), ('wavelength_nm', wavelengths)], upwelling),
            ('thermal_ozone_log_transmittance_ratio', [('climate_coordinate', CLIMATE_COORDINATES), ('pressure_hPa', pressures), ('ozone_DU', OZONE_DU), ('view_secant', view_secant), ('wavelength_nm', wavelengths)], ozone_log_ratio),
            ('thermal_ozone_upwelling_radiance', [('climate_coordinate', CLIMATE_COORDINATES), ('pressure_hPa', pressures), ('ozone_DU', OZONE_DU), ('view_secant', view_secant), ('wavelength_nm', wavelengths)], ozone_upwelling),
            ('thermal_downwelling_radiance', [('log_water_vapor_cm', log_water), ('climate_coordinate', CLIMATE_COORDINATES), ('pressure_hPa', pressures), ('ozone_DU', OZONE_DU), ('wavelength_nm', wavelengths)], downwelling)]


def libradtran_version(uvspec):
    result = subprocess.run([uvspec, '-h'], capture_output=True, text=True)
    for line in (result.stdout + result.stderr).splitlines():
        if 'version' in line:
            return line.strip()
    raise RuntimeError('Could not determine the libRadtran version from uvspec -h')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output')
    parser.add_argument('--processes', type=int, default=os.cpu_count())
    parser.add_argument('--uvspec', default=None)
    parser.add_argument('--libradtran-data', default=None)
    args = parser.parse_args()

    uvspec = args.uvspec or default_uvspec()
    data_path = args.libradtran_data or default_data_path(uvspec)
    tables = compute_tables(uvspec, data_path, args.processes)

    provenance = (f'Generated {datetime.date.today().isoformat()} by plugins/solarposition/utilities/generate_thermal_atmosphere_lut.py with libRadtran {libradtran_version(uvspec)}, '
                  f'REPTRAN fine thermal parameterization, DISORT solver, clear sky without aerosol, black surface, sensor at the top of the atmosphere. '
                  f'Profiles interpolated between the atmospheres {", ".join(BASE_ATMOSPHERES)} of Anderson et al. (1986) by surface air temperature, '
                  f'with CO2 {CO2_PPM} ppm, CH4 {CH4_PPB} ppb and N2O {N2O_PPB} ppb (NOAA GML {GREENHOUSE_GAS_YEAR} global means). '
                  f'{BIN_WIDTH} cm-1 bins over {WAVENUMBER_MIN}-{WAVENUMBER_MAX} cm-1; reference ozone column {REFERENCE_OZONE_DU:g} DU.')
    write_lut(args.output, tables, provenance)
    print(f'wrote {args.output} ({os.path.getsize(args.output)} bytes)')


if __name__ == '__main__':
    sys.exit(main())
