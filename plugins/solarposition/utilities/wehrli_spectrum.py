"""Read the extraterrestrial solar spectrum used by SolarPosition's spectral solar irradiance model.

SolarPosition reads plugins/solarposition/assets/ssolar_goa/wehrli85.txt, an unmodified copy of the 1985 Wehrli
standard extraterrestrial solar irradiance spectrum as distributed by the National Laboratory of the Rockies (see
NLR_DATA_NOTICE.txt beside it), and averages it over 1 nm bins centred on 300, 301, ..., 2600 nm, each source value being
taken to apply halfway to the neighbouring wavelengths. read_binned_spectrum() applies the same averaging
(SolarPosition::SpectralData::loadFromFile() in SolarPosition.cpp), for utilities that need the spectrum the model uses.
"""

import numpy as np

OUTPUT_WAVELENGTHS_NM = np.arange(300, 2601)


def read_source(path):
    """Return the wavelengths (nm) and spectral irradiances (W/m^2/nm) of the source file."""
    centres, irradiances = [], []
    with open(path, encoding='latin-1') as f:
        for line in f:
            fields = line.split()
            try:
                centre, irradiance = float(fields[0]), float(fields[1])
            except (IndexError, ValueError):
                if centres and line.strip():
                    raise RuntimeError(f'Could not parse the line "{line.rstrip()}" of {path}')
                continue  # header and blank lines
            centres.append(centre)
            irradiances.append(irradiance)
    centres = np.array(centres)
    if len(centres) < 2 or np.any(np.diff(centres) <= 0):
        raise RuntimeError(f'{path} does not contain strictly increasing wavelengths')
    return centres, np.array(irradiances)


def read_binned_spectrum(path):
    """Return {wavelength_nm: W/m^2/nm} for 300-2600 nm, averaged over 1 nm bins as SolarPosition does."""
    centres, irradiances = read_source(path)
    edges = np.empty(len(centres) + 1)
    edges[1:-1] = 0.5 * (centres[:-1] + centres[1:])
    edges[0] = centres[0] - 0.5 * (centres[1] - centres[0])
    edges[-1] = centres[-1] + 0.5 * (centres[-1] - centres[-2])
    cumulative = np.concatenate(([0.0], np.cumsum(irradiances * np.diff(edges))))
    lower, upper = OUTPUT_WAVELENGTHS_NM - 0.5, OUTPUT_WAVELENGTHS_NM + 0.5
    if lower[0] < edges[0] or upper[-1] > edges[-1]:
        raise RuntimeError(f'The spectrum in {path} does not cover 299.5-2600.5 nm')
    values = np.interp(upper, edges, cumulative) - np.interp(lower, edges, cumulative)
    return {int(w): float(v) for w, v in zip(OUTPUT_WAVELENGTHS_NM, values)}
