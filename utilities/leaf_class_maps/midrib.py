"""Trace the midrib of a grass leaf photograph by its shape, and build a hybrid albedo: the decomposition's albedo for the blade, and
inside the midrib the blade's albedo times the photograph's local midrib-to-blade colour ratio.

The intrinsic decomposition takes a thin pale midrib for lighting and drops it from the albedo. The midrib is found here as what it
is: a pale, unsaturated ridge running along the leaf, one row per texture column, whose centre line is smooth.

Usage:  python midrib.py TEXTURE.png ALBEDO.png OUTPUT_PREFIX
Writes OUTPUT_PREFIX_hybrid_albedo.png (sRGB 8-bit RGBA, the texture's alpha), OUTPUT_PREFIX_midrib_mask.png (255 in the midrib),
and OUTPUT_PREFIX_midrib_overlay.png (the photograph with the traced midrib outlined), and prints what it found.
"""
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import median_filter, uniform_filter1d

OPAQUE = 250


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, np.power((c + 0.055) / 1.055, 2.4))


def linear_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    texture_file, albedo_file, prefix = sys.argv[1:]
    rgba = np.asarray(Image.open(texture_file).convert("RGBA"))
    albedo_rgba = np.asarray(Image.open(albedo_file).convert("RGBA"))
    if rgba.shape != albedo_rgba.shape:
        sys.exit(f"the texture {rgba.shape} and its albedo {albedo_rgba.shape} differ in size")
    height, width = rgba.shape[:2]
    opaque = rgba[..., 3] >= OPAQUE
    photo = srgb_to_linear(rgba[..., :3].astype(np.float64) / 255.0)
    albedo = srgb_to_linear(albedo_rgba[..., :3].astype(np.float64) / 255.0)

    # Paleness: bright and unsaturated. The midrib is the palest row of each column, away from the leaf's edge.
    value = photo.max(axis=2)
    saturation = np.where(value > 0, (value - photo.min(axis=2)) / np.maximum(value, 1e-6), 0)
    paleness = value * (1.0 - saturation)
    interior = opaque.copy()
    for shift in (-3, -2, -1, 1, 2, 3):  # at least three opaque rows on both sides
        interior &= np.roll(opaque, shift, axis=0)
    paleness_interior = np.where(interior, uniform_filter1d(paleness, 3, axis=0), -1.0)

    columns = np.where(interior.any(axis=0))[0]
    peak_row = paleness_interior[:, columns].argmax(axis=0)
    peak = paleness_interior[peak_row, columns]
    blade = np.array([np.median(paleness[opaque[:, x], x]) for x in columns])
    contrast = (peak - blade) / np.maximum(blade, 1e-6)

    # Centre line: a smooth curve through the peaks of the columns where the midrib stands out, refitted without outliers
    strong = contrast > 0.5
    if strong.sum() < 50:
        sys.exit(f"no midrib: only {strong.sum()} columns have a pale ridge")
    weights = strong.astype(float)
    for _ in range(4):
        coefficients = np.polyfit(columns, peak_row, 4, w=weights)
        residual = np.abs(np.polyval(coefficients, columns) - peak_row)
        weights = (strong & (residual < 3)).astype(float)
    centre = np.polyval(coefficients, columns)

    # Where the midrib runs: from the base to the last column where it still stands out (smoothed along the leaf)
    standing_out = median_filter(strong.astype(np.uint8), size=31) > 0
    end = columns[np.where(standing_out)[0].max()]
    start = columns[np.where(standing_out)[0].min()]

    # Width: the rows around the centre that are above half the ridge's contrast over the blade, as a smooth taper along the leaf
    half_widths = []
    for i, x in enumerate(columns):
        if not (start <= x <= end):
            half_widths.append(np.nan)
            continue
        c = int(round(centre[i]))
        threshold = blade[i] + 0.5 * (paleness_interior[c, x] - blade[i])
        up = 0
        while c - up - 1 >= 0 and paleness[c - up - 1, x] > threshold and up < 12:
            up += 1
        down = 0
        while c + down + 1 < height and paleness[c + down + 1, x] > threshold and down < 12:
            down += 1
        half_widths.append(0.5 * (up + down + 1))
    half_widths = np.array(half_widths)
    valid = ~np.isnan(half_widths)
    taper = np.polyfit(columns[valid], half_widths[valid], 1)
    half_width = np.clip(np.polyval(taper, columns), 0.75, None)

    mask = np.zeros((height, width), dtype=bool)
    rows = np.arange(height)
    for i, x in enumerate(columns):
        if start <= x <= end:
            mask[:, x] = (np.abs(rows - centre[i]) <= half_width[i]) & opaque[:, x]

    # Hybrid albedo: inside the midrib, the albedo of the neighbouring blade times the photograph's local midrib-to-blade ratio
    hybrid = albedo.copy()
    for i, x in enumerate(columns):
        if not mask[:, x].any():
            continue
        lo, hi = max(0, x - 8), min(width, x + 9)
        c = centre[i]
        band = (np.abs(rows - c)[:, None] > half_width[i] + 1) & (np.abs(rows - c)[:, None] <= half_width[i] + 8)
        neighbourhood = band & opaque[:, lo:hi] & ~mask[:, lo:hi]
        if neighbourhood.sum() < 10:
            continue
        blade_photo = np.median(photo[:, lo:hi][neighbourhood], axis=0)
        blade_albedo = np.median(albedo[:, lo:hi][neighbourhood], axis=0)
        ratio = photo[mask[:, x], x] / np.maximum(blade_photo, 1e-6)
        hybrid[mask[:, x], x] = blade_albedo * ratio
    hybrid = np.clip(hybrid, 0, 1)

    Image.fromarray(np.dstack([np.round(255 * linear_to_srgb(hybrid)).astype(np.uint8), rgba[..., 3]]), mode="RGBA").save(prefix + "_hybrid_albedo.png")
    Image.fromarray((mask * 255).astype(np.uint8)).save(prefix + "_midrib_mask.png")
    overlay = rgba[..., :3].copy()
    edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0))
    overlay[edge] = [255, 0, 200]
    Image.fromarray(overlay).save(prefix + "_midrib_overlay.png")
    print(f"midrib from column {start} to {end} of {width}, half-width {np.polyval(taper, start):.1f} to {np.polyval(taper, end):.1f} texels, "
          f"{mask.sum()} texels ({100 * mask.sum() / opaque.sum():.1f}% of the leaf); "
          f"midrib/blade ratio (mean, linear RGB) {np.mean(hybrid[mask], axis=0) / np.mean(hybrid[opaque & ~mask], axis=0)}")


if __name__ == "__main__":
    main()
