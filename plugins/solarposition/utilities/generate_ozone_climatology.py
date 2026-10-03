#!/usr/bin/env python3
"""Generate the total column ozone climatology used by SolarPosition::getOzoneColumn().

The climatology is the average, over the years 2005-2024, of the monthly zonal-mean total ozone of the NASA SBUV Merged
Ozone Data Set (MOD, version 8.7; described for version 8.6 by Frith et al. 2014, doi:10.1002/2014JD021889), downloaded from
    https://acd-ext.gsfc.nasa.gov/Data_services/merged/data/sbuv.v87.mod_v14.70-25.za.txt
That file gives, for each year, the monthly mean total ozone in Dobson units in 36 latitude bands 5 degrees wide, with 0.0
marking months without data (the backscattered-ultraviolet measurements need sunlight, so high-latitude winter months are missing, and the
bands poleward of 80 degrees have no data). A band-month is kept only if at least MIN_YEARS of the averaging years have data; otherwise it is written as
"nan" and SolarPosition reports that no climatological value is available there.

Data from NASA-led missions that carry no other license are released as Creative Commons Zero (CC0); NASA asks that the
data set be cited (https://www.earthdata.nasa.gov/engage/open-data-services-software-policies/data-use-guidance).

Usage:
    python generate_ozone_climatology.py <output.txt>
"""

import datetime
import re
import sys
import urllib.request

SOURCE_URL = 'https://acd-ext.gsfc.nasa.gov/Data_services/merged/data/sbuv.v87.mod_v14.70-25.za.txt'
FIRST_YEAR = 2005
LAST_YEAR = 2024
MIN_YEARS = 10
NBANDS = 36


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1

    text = urllib.request.urlopen(SOURCE_URL, timeout=120).read().decode('ascii')

    # Parse the year blocks: a header line "<year> SBUV_V87 ...", a month header, then one row per latitude band
    years = {}
    year = None
    for line in text.split('\n'):
        header = re.match(r'^\s*(\d{4}) SBUV', line)
        if header:
            year = int(header.group(1))
            years[year] = []
            continue
        fields = line.split()
        if year is not None and len(fields) == 14:
            try:
                values = [float(field) for field in fields]
            except ValueError:
                continue
            years[year].append((values[0], values[1], values[2:]))

    for y in range(FIRST_YEAR, LAST_YEAR + 1):
        if y not in years or len(years[y]) != NBANDS:
            raise RuntimeError(f'Year {y} is missing or incomplete in {SOURCE_URL}')
        for band, (lat_min, lat_max, _) in enumerate(years[y]):
            if lat_min != -90 + 5 * band or lat_max != lat_min + 5:
                raise RuntimeError(f'Unexpected latitude band ({lat_min}, {lat_max}) in year {y}')

    rows = []
    for band in range(NBANDS):
        lat_min = -90 + 5 * band
        monthly = []
        for month in range(12):
            values = [years[y][band][2][month] for y in range(FIRST_YEAR, LAST_YEAR + 1) if years[y][band][2][month] > 0]
            monthly.append(sum(values) / len(values) if len(values) >= MIN_YEARS else None)
        rows.append((lat_min, lat_min + 5, monthly))

    with open(sys.argv[1], 'w') as f:
        f.write(f'# Monthly zonal-mean total column ozone climatology (Dobson units), average of {FIRST_YEAR}-{LAST_YEAR}\n')
        f.write(f'# Source: NASA SBUV Merged Ozone Data Set, version 8.7 (described for version 8.6 by Frith et al. 2014, doi:10.1002/2014JD021889), {SOURCE_URL}\n')
        f.write(f'# Generated {datetime.date.today().isoformat()} by plugins/solarposition/utilities/generate_ozone_climatology.py\n')
        f.write(f'# A band-month with data in fewer than {MIN_YEARS} of the years is nan (high-latitude winter months, when the sun is too low for the measurements, and all months poleward of 80 degrees)\n')
        f.write('# Columns: latitude_min latitude_max Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec\n')
        for lat_min, lat_max, monthly in rows:
            f.write(f'{lat_min:4d} {lat_max:4d} ' + ' '.join(f'{v:6.1f}' if v is not None else '   nan' for v in monthly) + '\n')
    print(f'wrote {sys.argv[1]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
