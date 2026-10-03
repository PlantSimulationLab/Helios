#!/usr/bin/env python3
"""Generate the atmospheric look-up table used by SolarPosition's sensor atmosphere model and, for its gas
transmittance tables, by the spectral solar irradiance model (calculateGlobalSolarSpectrum() and related methods).

The table is computed with the 6S radiative transfer code (6SV1.1) through the Py6S wrapper. It holds:

  Scattering quantities (computed without gas absorption, continental aerosol, at 16 wavelengths between which
  they vary smoothly, so SolarPosition interpolates them log-log in wavelength):
    path_reflectance(wavelength_nm, pressure_hPa, aod550, solar_zenith_deg, view_zenith_deg, relative_azimuth_deg)
        Reflectance of the atmosphere alone (pi*L_path/(mu_s*E0)) seen from above the atmosphere. A relative
        azimuth of 0 puts the sensor on the same side as the sun (backscattering).
    scattering_transmittance(wavelength_nm, pressure_hPa, aod550, zenith_deg)
        Total (direct + diffuse) transmittance of the atmosphere along a path with the given zenith angle.
    optical_depth(wavelength_nm, pressure_hPa, aod550)
        Total (Rayleigh + aerosol) extinction optical depth; the direct transmittance is exp(-tau/mu).
    spherical_albedo(wavelength_nm, pressure_hPa, aod550)

  Gas transmittance (computed at 1 nm spacing, the resolution of SolarPosition's spectra), each a function of the absorber amount along the path:
    water_transmittance(wavelength_nm, water_path_g_cm2)        water path = air mass * column water vapor
    ozone_transmittance(wavelength_nm, ozone_path_atm_cm)       ozone path = air mass * ozone column
    mixed_gas_transmittance(wavelength_nm, pressure_weighted_air_mass)
        O2, CO2, CH4, N2O and CO along a path of air mass m to a surface at pressure p: the axis is m*p/p0 with
        p0 = 1013.25 hPa. Runs at sea level (6S surface pressure 1013 hPa) with the sun at a zenith of acos(1/m) give
        axis values m*1013/1013.25; the smaller axis values come from overhead-sun runs to an elevated target, whose
        6S surface pressure sets the axis value.
  The absorber axes extend to an air mass of 38, just above the relative optical air mass of Kasten and Young (1989)
  at a solar zenith angle of 90 degrees (37.92), so that the spectral irradiance model covers the sun up to the horizon.

Requirements: a conda environment with the 6S executable and Py6S, e.g.
    conda create -p ./sixs_env -c conda-forge python=3.11 sixs py6s numpy
and the environment's bin directory on PATH.

Usage:
    python generate_atmosphere_lut.py <output.bin> [--quick] [--processes N] [--gas-only-from <existing.bin>]

--quick uses coarse grids for testing the pipeline. --gas-only-from copies the scattering tables from an existing
look-up table and recomputes only the gas tables, which takes minutes rather than hours. The binary format is
described in write_lut().
"""

import argparse
import datetime
import importlib.metadata
import math
import sys
from multiprocessing import Pool

import numpy as np
from Py6S import SixS, AtmosProfile, AeroProfile, Geometry, GroundReflectance, Wavelength

from helios_lut import read_lut, write_lut

# Scattering is computed without gas absorption at these wavelengths; it varies smoothly enough with wavelength to be interpolated between them
SCATTERING_WAVELENGTHS_NM = [300, 330, 360, 400, 440, 490, 550, 620, 670, 750, 870, 1040, 1240, 1640, 2130, 2600]
# Target altitudes; 6S reports the corresponding surface pressure, which is the axis stored in the table
TARGET_ALTITUDES_KM = [0.0, 1.0, 2.0, 3.0]
AOD550_VALUES = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2]
SOLAR_ZENITHS_DEG = [0, 10, 20, 30, 40, 50, 55, 60, 65, 70]
VIEW_ZENITHS_DEG = [0, 10, 20, 30, 40, 50, 55, 60]
RELATIVE_AZIMUTHS_DEG = [0, 20, 40, 60, 80, 100, 120, 140, 160, 180]

GAS_WAVELENGTHS_NM = list(np.arange(300.0, 2600.0 + 1e-6, 1.0))
GAS_POINTS = 24
# Largest air mass: Kasten and Young (1989) give 37.92 at a solar zenith of 90 degrees
MAX_AIR_MASS = 38.0
# Water and ozone runs pair each path with an air mass, so that the column (path / air mass) stays physical: up to 8.4 g/cm2 of water and 0.66 atm-cm of ozone
AIR_MASSES = list(np.logspace(0.0, math.log10(MAX_AIR_MASS), GAS_POINTS))
WATER_PATHS_G_CM2 = list(np.logspace(math.log10(0.02), math.log10(320.0), GAS_POINTS))
OZONE_PATHS_ATM_CM = list(np.logspace(math.log10(0.1), math.log10(25.0), GAS_POINTS))
# Mixed-gas runs: target altitudes (km) for the pressure-weighted air masses below one, then sea-level air masses
MIXED_GAS_TARGET_ALTITUDES_KM = [5.5, 4.5, 3.5, 2.5, 1.5, 0.75]
MIXED_GAS_AIR_MASSES = list(np.logspace(0.0, math.log10(MAX_AIR_MASS), GAS_POINTS - len(MIXED_GAS_TARGET_ALTITUDES_KM)))

# Negligible water vapor and ozone for the scattering runs (6S does not accept exactly zero)
NO_ABSORBER = 1e-6


def base_sixs(wavelength_nm, water_g_cm2, ozone_atm_cm, solar_zenith_deg, view_zenith_deg, relative_azimuth_deg):
    s = SixS()
    s.atmos_profile = AtmosProfile.UserWaterAndOzone(water_g_cm2, ozone_atm_cm)
    s.geometry = Geometry.User()
    s.geometry.solar_z = solar_zenith_deg
    s.geometry.solar_a = 0
    s.geometry.view_z = view_zenith_deg
    s.geometry.view_a = relative_azimuth_deg
    s.geometry.month = 6
    s.geometry.day = 21
    s.ground_reflectance = GroundReflectance.HomogeneousLambertian(0.0)
    s.altitudes.set_sensor_satellite_level()
    s.wavelength = Wavelength(wavelength_nm / 1000.0)
    return s


def run_scattering_case(case):
    wavelength_nm, altitude_km, aod550, solar_zenith, view_zenith, relative_azimuth = case
    s = base_sixs(wavelength_nm, NO_ABSORBER, NO_ABSORBER, solar_zenith, view_zenith, relative_azimuth)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.Continental)
    s.aot550 = aod550
    if altitude_km == 0.0:
        s.altitudes.set_target_sea_level()
    else:
        s.altitudes.set_target_custom_altitude(altitude_km)
    s.run()
    o = s.outputs
    mu_s = math.cos(math.radians(solar_zenith))
    # 6S's intrinsic radiance includes the (here negligible) gas absorption along the path, so it is divided out
    path_reflectance = math.pi * o.values['atmospheric_intrinsic_radiance'] / (mu_s * o.values['solar_spectrum']) / o.transmittance_global_gas.total
    return dict(path_reflectance=path_reflectance, transmittance_down=o.transmittance_total_scattering.downward, optical_depth=o.optical_depth_total.total,
                spherical_albedo=o.spherical_albedo.total, pressure_hPa=o.values['ground_pressure'])


def run_absorber_case(case):
    wavelength_nm, index = case
    # Each run sets a solar air mass and scales the water and ozone columns so that the absorber paths are those of this index
    air_mass = AIR_MASSES[index]
    solar_zenith = math.degrees(math.acos(1.0 / air_mass))
    s = base_sixs(wavelength_nm, WATER_PATHS_G_CM2[index] / air_mass, OZONE_PATHS_ATM_CM[index] / air_mass, solar_zenith, 0, 0)
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
    s.altitudes.set_target_sea_level()
    s.run()
    o = s.outputs
    return dict(water=o.trans['water'].downward, ozone=o.trans['ozone'].downward)


def run_mixed_gas_case(case):
    wavelength_nm, index = case
    # Below one, the pressure-weighted air mass is set by the surface pressure of an elevated target under an overhead sun; from one upward, by the solar air mass at sea level
    if index < len(MIXED_GAS_TARGET_ALTITUDES_KM):
        s = base_sixs(wavelength_nm, NO_ABSORBER, NO_ABSORBER, 0, 0, 0)
        s.altitudes.set_target_custom_altitude(MIXED_GAS_TARGET_ALTITUDES_KM[index])
        air_mass = 1.0
    else:
        air_mass = MIXED_GAS_AIR_MASSES[index - len(MIXED_GAS_TARGET_ALTITUDES_KM)]
        s = base_sixs(wavelength_nm, NO_ABSORBER, NO_ABSORBER, math.degrees(math.acos(1.0 / air_mass)), 0, 0)
        s.altitudes.set_target_sea_level()
    s.aero_profile = AeroProfile.PredefinedType(AeroProfile.NoAerosols)
    s.run()
    o = s.outputs
    # 6S labels nitrous oxide "no2"
    mixed = o.trans['oxygen'].downward * o.trans['co2'].downward * o.trans['ch4'].downward * o.trans['no2'].downward * o.trans['co'].downward
    return dict(mixed=mixed, pressure_weighted_air_mass=air_mass * o.values['ground_pressure'] / 1013.25)


def compute_scattering_tables(processes):
    """Run 6S for the scattering quantities and return them as (name, axes, data) tables."""
    # A zero solar or view zenith makes the relative azimuth meaningless, so only azimuth 0 is run and copied.
    scattering_cases = []
    for wavelength in SCATTERING_WAVELENGTHS_NM:
        for altitude in TARGET_ALTITUDES_KM:
            for aod in AOD550_VALUES:
                for sza in SOLAR_ZENITHS_DEG:
                    for vza in VIEW_ZENITHS_DEG:
                        for raa in RELATIVE_AZIMUTHS_DEG:
                            if (sza == 0 or vza == 0) and raa != RELATIVE_AZIMUTHS_DEG[0]:
                                continue
                            scattering_cases.append((wavelength, altitude, aod, sza, vza, raa))
    print(f'{len(scattering_cases)} scattering runs', flush=True)
    with Pool(processes) as pool:
        scattering_results = dict(zip(scattering_cases, pool.map(run_scattering_case, scattering_cases, chunksize=64)))
    print('scattering runs done', flush=True)

    Nw, Na, Nt = len(SCATTERING_WAVELENGTHS_NM), len(TARGET_ALTITUDES_KM), len(AOD550_VALUES)
    Ns, Nv, Nr = len(SOLAR_ZENITHS_DEG), len(VIEW_ZENITHS_DEG), len(RELATIVE_AZIMUTHS_DEG)
    path_reflectance = np.zeros((Nw, Na, Nt, Ns, Nv, Nr))
    transmittance = np.zeros((Nw, Na, Nt, Ns))
    optical_depth = np.zeros((Nw, Na, Nt))
    spherical_albedo = np.zeros((Nw, Na, Nt))
    pressures = np.zeros(Na)
    for iw, wavelength in enumerate(SCATTERING_WAVELENGTHS_NM):
        for ia, altitude in enumerate(TARGET_ALTITUDES_KM):
            for it, aod in enumerate(AOD550_VALUES):
                for i_s, sza in enumerate(SOLAR_ZENITHS_DEG):
                    for iv, vza in enumerate(VIEW_ZENITHS_DEG):
                        for ir, raa in enumerate(RELATIVE_AZIMUTHS_DEG):
                            azimuth = raa if (sza != 0 and vza != 0) else RELATIVE_AZIMUTHS_DEG[0]
                            result = scattering_results[(wavelength, altitude, aod, sza, vza, azimuth)]
                            path_reflectance[iw, ia, it, i_s, iv, ir] = result['path_reflectance']
                    result = scattering_results[(wavelength, altitude, aod, sza, VIEW_ZENITHS_DEG[0], RELATIVE_AZIMUTHS_DEG[0])]
                    transmittance[iw, ia, it, i_s] = result['transmittance_down']
                    optical_depth[iw, ia, it] = result['optical_depth']
                    spherical_albedo[iw, ia, it] = result['spherical_albedo']
                    pressures[ia] = result['pressure_hPa']

    # The view zenith axis must be a subset of the solar zenith axis, since both directions use the same transmittance table
    if not set(VIEW_ZENITHS_DEG).issubset(SOLAR_ZENITHS_DEG):
        raise RuntimeError('VIEW_ZENITHS_DEG must be a subset of SOLAR_ZENITHS_DEG')

    scattering_axes = [('wavelength_nm', SCATTERING_WAVELENGTHS_NM), ('pressure_hPa', list(pressures)), ('aod550', AOD550_VALUES)]
    return [
        ('path_reflectance', scattering_axes + [('solar_zenith_deg', SOLAR_ZENITHS_DEG), ('view_zenith_deg', VIEW_ZENITHS_DEG), ('relative_azimuth_deg', RELATIVE_AZIMUTHS_DEG)], path_reflectance),
        ('scattering_transmittance', scattering_axes + [('zenith_deg', SOLAR_ZENITHS_DEG)], transmittance),
        ('optical_depth', scattering_axes, optical_depth),
        ('spherical_albedo', scattering_axes, spherical_albedo),
    ]


def compute_gas_tables(processes):
    """Run 6S for the gas transmittances and return them as (name, axes, data) tables."""
    absorber_cases = [(wavelength, index) for wavelength in GAS_WAVELENGTHS_NM for index in range(GAS_POINTS)]
    print(f'{2 * len(absorber_cases)} gas runs', flush=True)
    with Pool(processes) as pool:
        absorber_results = dict(zip(absorber_cases, pool.map(run_absorber_case, absorber_cases, chunksize=64)))
        mixed_results = dict(zip(absorber_cases, pool.map(run_mixed_gas_case, absorber_cases, chunksize=64)))
    print('gas runs done', flush=True)

    Ng = len(GAS_WAVELENGTHS_NM)
    water = np.zeros((Ng, GAS_POINTS))
    ozone = np.zeros((Ng, GAS_POINTS))
    mixed = np.zeros((Ng, GAS_POINTS))
    for ig, wavelength in enumerate(GAS_WAVELENGTHS_NM):
        for index in range(GAS_POINTS):
            water[ig, index] = absorber_results[(wavelength, index)]['water']
            ozone[ig, index] = absorber_results[(wavelength, index)]['ozone']
            mixed[ig, index] = mixed_results[(wavelength, index)]['mixed']

    # The pressure-weighted air mass of a run depends only on its geometry and target altitude, not on the wavelength
    pressure_weighted_air_masses = [mixed_results[(GAS_WAVELENGTHS_NM[0], index)]['pressure_weighted_air_mass'] for index in range(GAS_POINTS)]
    for (wavelength, index), result in mixed_results.items():
        if abs(result['pressure_weighted_air_mass'] - pressure_weighted_air_masses[index]) > 1e-6:
            raise RuntimeError(f'Pressure-weighted air mass of mixed-gas run {index} differs between wavelengths')
    if any(b <= a for a, b in zip(pressure_weighted_air_masses, pressure_weighted_air_masses[1:])):
        raise RuntimeError(f'Pressure-weighted air masses are not strictly increasing: {pressure_weighted_air_masses}')

    return [
        ('water_transmittance', [('wavelength_nm', GAS_WAVELENGTHS_NM), ('water_path_g_cm2', WATER_PATHS_G_CM2)], water),
        ('ozone_transmittance', [('wavelength_nm', GAS_WAVELENGTHS_NM), ('ozone_path_atm_cm', OZONE_PATHS_ATM_CM)], ozone),
        ('mixed_gas_transmittance', [('wavelength_nm', GAS_WAVELENGTHS_NM), ('pressure_weighted_air_mass', pressure_weighted_air_masses)], mixed),
    ]


def main():
    global SCATTERING_WAVELENGTHS_NM, TARGET_ALTITUDES_KM, AOD550_VALUES, SOLAR_ZENITHS_DEG, VIEW_ZENITHS_DEG, RELATIVE_AZIMUTHS_DEG, GAS_WAVELENGTHS_NM

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output')
    parser.add_argument('--quick', action='store_true', help='coarse grids, for testing the pipeline')
    parser.add_argument('--processes', type=int, default=8)
    parser.add_argument('--gas-only-from', metavar='EXISTING_LUT', help='copy the scattering tables from this look-up table and recompute only the gas tables')
    args = parser.parse_args()

    if args.quick:
        SCATTERING_WAVELENGTHS_NM = [400, 550, 870, 1640]
        TARGET_ALTITUDES_KM = [0.0, 2.0]
        AOD550_VALUES = [0.0, 0.2, 0.8]
        SOLAR_ZENITHS_DEG = [0, 30, 60]
        VIEW_ZENITHS_DEG = [0, 30, 60]
        RELATIVE_AZIMUTHS_DEG = [0, 90, 180]
        GAS_WAVELENGTHS_NM = list(np.arange(300.0, 2600.0 + 1e-6, 100.0))

    # Built before the runs so that a failure here does not waste them
    today = datetime.date.today().isoformat()
    run_description = (f'by plugins/solarposition/utilities/generate_atmosphere_lut.py{" (--quick)" if args.quick else ""} with 6SV1.1 via Py6S {importlib.metadata.version("Py6S")}')
    gas_table_names = ('water_transmittance', 'ozone_transmittance', 'mixed_gas_transmittance')
    if args.gas_only_from:
        existing_provenance, existing_tables = read_lut(args.gas_only_from)
        scattering_tables = [table for table in existing_tables if table[0] not in gas_table_names]
        provenance = f'{existing_provenance} Gas transmittance tables regenerated {today} {run_description}.'
    else:
        scattering_tables = None
        provenance = f'Generated {today} {run_description}. Continental aerosol model, Lambertian surface, sensor above the atmosphere.'

    if scattering_tables is None:
        scattering_tables = compute_scattering_tables(args.processes)
    tables = scattering_tables + compute_gas_tables(args.processes)
    write_lut(args.output, tables, provenance)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    sys.exit(main())
