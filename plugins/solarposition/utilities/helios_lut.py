"""Reader and writer for the HLIOSLUT binary look-up table format loaded by SolarPosition (AtmosphereLUT::loadFromFile()).

Shared by generate_atmosphere_lut.py (6S shortwave table) and generate_thermal_atmosphere_lut.py (libRadtran thermal table).
"""

import struct

import numpy as np


def write_lut(path, tables, provenance):
    """Binary format (little endian):
         char[8]  magic "HLIOSLUT"
         uint32   format version (1)
         uint32   provenance length, then that many bytes of UTF-8 text
         uint32   number of tables
         for each table:
             uint32 name length, name bytes
             uint32 number of axes
             for each axis: uint32 name length, name bytes, uint32 count, float32[count] values
             float32[product of axis counts] data, row-major (last axis varies fastest)
    """
    def write_string(f, text):
        encoded = text.encode('utf-8')
        f.write(struct.pack('<I', len(encoded)))
        f.write(encoded)

    with open(path, 'wb') as f:
        f.write(b'HLIOSLUT')
        f.write(struct.pack('<I', 1))
        write_string(f, provenance)
        f.write(struct.pack('<I', len(tables)))
        for name, axes, data in tables:
            write_string(f, name)
            f.write(struct.pack('<I', len(axes)))
            for axis_name, values in axes:
                write_string(f, axis_name)
                f.write(struct.pack('<I', len(values)))
                f.write(np.asarray(values, dtype='<f4').tobytes())
            expected_shape = tuple(len(values) for _, values in axes)
            if data.shape != expected_shape:
                raise RuntimeError(f'Table {name} has shape {data.shape}, expected {expected_shape}')
            f.write(np.ascontiguousarray(data, dtype='<f4').tobytes())


def read_lut(path):
    """Read a look-up table written by write_lut(); returns the provenance and a list of (name, axes, data)."""
    def read_uint(f):
        return struct.unpack('<I', f.read(4))[0]

    def read_string(f):
        return f.read(read_uint(f)).decode('utf-8')

    with open(path, 'rb') as f:
        if f.read(8) != b'HLIOSLUT' or read_uint(f) != 1:
            raise RuntimeError(f'{path} is not a version 1 atmospheric look-up table')
        provenance = read_string(f)
        tables = []
        for _ in range(read_uint(f)):
            name = read_string(f)
            axes = []
            for _ in range(read_uint(f)):
                axis_name = read_string(f)
                count = read_uint(f)
                axes.append((axis_name, list(np.frombuffer(f.read(4 * count), dtype='<f4').astype(float))))
            shape = tuple(len(values) for _, values in axes)
            data = np.frombuffer(f.read(4 * int(np.prod(shape))), dtype='<f4').reshape(shape).astype(float)
            tables.append((name, axes, data))
    return provenance, tables
