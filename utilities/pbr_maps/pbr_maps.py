"""Generate a tangent-space normal map and a bump (height) map from an RGB(A) texture photograph, for Helios materials.

Usage:
    python pbr_maps.py -o OUTPUT_DIR [options] TEXTURE [TEXTURE ...]

For each TEXTURE it writes, at the texture's resolution:
    <stem>_normal.png   tangent-space normal map, 8-bit RGB (OpenGL convention: +X = +u = image right, +Y = +v = image up)
    <stem>_height.png   height (bump) map, 16-bit grayscale by default (0 = lowest, 65535 = highest)
    <stem>_pbr.json     the parameters used and the equivalent Helios calls
    <stem>_preview.png  (with --preview) input | normal map | height | the surface relit by a low sun

The normal map is computed from the stored (quantized) height map with the same kernel Helios uses for bump maps, so
    context.setMaterialBumpMap("label", "<stem>_height.png", STRENGTH)
and
    context.setMaterialNormalMap("label", "<stem>_normal.png")
give the same normals (to 1 LSB of the 8-bit normal map) over the opaque part of the texture.

Method (classical, no learned model): a photograph's brightness mixes the surface's albedo with how it is lit. Under the
diffuse light of most texture photographs, cavities (furrows, grooves, pores) are darker because they see less sky and raised
parts are brighter, so brightness at the scale of a feature is a usable proxy for relief, while brightness varying over the
whole image is mostly lighting. The steps are:
  1. the height source: linear luminance of the texture (sRGB decoded, Rec. 709 weights), or the diffuse shading image of an
     intrinsic decomposition (utilities/leaf_class_maps/decompose.py, --height-source shading), which removes albedo marks
  2. log(Y + eps), so the result depends on relative, not absolute, brightness, and an optional light pre-smoothing that
     suppresses JPEG and sensor noise
  3. a mask-aware fill of transparent texels (normalized convolution at growing scales), so the cut-out edge of a leaf
     produces no false rim
  4. illumination flattening: the brightness averaged over a large scale (--flatten-sigma) is removed
  5. a multi-scale band-pass: differences of Gaussians at scales 1, 2, 4, ... texels, each weighted by its scale (contrast
     at feature width w becomes height ~ contrast x w) and by the fine / medium / coarse weights (--weights)
  6. optional inversion (--invert, for surfaces whose raised parts are dark), robust normalization to [0, 1] between the
     0.5th and 99.5th percentiles of the opaque texels, and quantization to the stored height
  7. normals from the quantized height with a normalized Scharr kernel: n = normalize(-s*dh/du, -s*dh/dv, 1), with s the
     strength in texels of displacement for the full height range

Limits: dark albedo marks (lichen, spots, printed lines, the dark veins of some leaves) become false grooves, and cast
shadows of a directional light become false slopes; --height-source shading removes most of the former. Learned single-image
methods (e.g. DeepBump, GPL-3.0) can do better on some textures but cannot be bundled with Helios.

Requires numpy, scipy and Pillow.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

OPAQUE_ALPHA = 250  # Helios treats alpha >= 250 (of 255) as opaque (helios::readPNGAlpha)
LOG_EPS = 1e-3      # added to linear luminance before the log, so black texels do not dominate
SCHARR_W = np.array([3.0, 10.0, 3.0]) / 32.0  # normalized Scharr smoothing weights (a ramp h = a*col gives gradient a exactly)
DIRECT_BLUR_MAX_SIGMA = 16.0  # larger Gaussian blurs of non-tileable images run on a reduced copy


# ----------------------------------------------------------------------------------------------------------------------
# Image I/O
# ----------------------------------------------------------------------------------------------------------------------

def srgb_to_linear(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_texture(path):
    """Read a texture. Returns (linear luminance HxW in [0,1], opaque mask HxW bool, RGB preview HxWx3 float in [0,1])."""
    image = Image.open(path)
    if image.mode in ("I;16", "I;16B", "I;16L", "I"):
        # A 16-bit grayscale image: taken as linear luminance
        values = np.asarray(image).astype(np.float64)
        peak = 65535.0 if values.max() > 255 else 255.0
        luminance = np.clip(values / peak, 0.0, 1.0)
        mask = np.ones(luminance.shape, dtype=bool)
        return luminance, mask, np.repeat(luminance[..., None], 3, axis=2)
    rgba = np.asarray(image.convert("RGBA")).astype(np.float64) / 255.0
    linear = srgb_to_linear(rgba[..., :3])
    luminance = 0.2126 * linear[..., 0] + 0.7152 * linear[..., 1] + 0.0722 * linear[..., 2]
    mask = np.asarray(image.convert("RGBA"))[..., 3] >= OPAQUE_ALPHA
    return luminance, mask, rgba[..., :3]


def load_shading(path, shape):
    """Read a linear shading image (e.g. decompose.py's 16-bit <stem>_shading.png), scaled so its maximum is 1."""
    values = np.asarray(Image.open(path)).astype(np.float64)
    if values.ndim == 3:
        values = values[..., :3].mean(axis=2)
    if values.shape != shape:
        raise ValueError(f"The shading image '{path}' is {values.shape[1]}x{values.shape[0]}, but the texture is {shape[1]}x{shape[0]}.")
    peak = values.max()
    if peak <= 0:
        raise ValueError(f"The shading image '{path}' is black.")
    return values / peak


def write_height_png(path, height, bits):
    if bits == 16:
        Image.fromarray(np.round(height * 65535.0).astype(np.uint16)).save(path)
    else:
        Image.fromarray(np.round(height * 255.0).astype(np.uint8), mode="L").save(path)


def read_height_png(path):
    values = np.asarray(Image.open(path)).astype(np.float64)
    peak = 65535.0 if Image.open(path).mode.startswith("I") else 255.0
    return values / peak


def encode_normals(normals, mask=None, directx=False):
    """Unit normals HxWx3 -> 8-bit RGB, RGB = round((n*0.5+0.5)*255); transparent texels (mask False) get the flat normal."""
    n = normals.copy()
    if directx:
        n[..., 1] = -n[..., 1]
    rgb = np.round((n * 0.5 + 0.5) * 255.0)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    if mask is not None:
        rgb[~mask] = (128, 128, 255)
    return rgb


def decode_normals(rgb, directx=False):
    n = rgb.astype(np.float64) / 255.0 * 2.0 - 1.0
    if directx:
        n[..., 1] = -n[..., 1]
    return n / np.linalg.norm(n, axis=2, keepdims=True)


# ----------------------------------------------------------------------------------------------------------------------
# Filters
# ----------------------------------------------------------------------------------------------------------------------

def gaussian_blur(image, sigma, tileable):
    """Gaussian blur; periodic (FFT) for tileable images, otherwise edge-replicating, on a reduced copy for large sigma."""
    if sigma <= 0:
        return image.copy()
    if tileable:
        return np.real(np.fft.ifft2(ndimage.fourier_gaussian(np.fft.fft2(image), sigma)))
    if sigma <= DIRECT_BLUR_MAX_SIGMA:
        return ndimage.gaussian_filter(image, sigma, mode="nearest", truncate=4.0)
    factor = int(sigma // (DIRECT_BLUR_MAX_SIGMA / 2))
    h, w = image.shape
    ph, pw = (-h) % factor, (-w) % factor
    padded = np.pad(image, ((0, ph), (0, pw)), mode="edge")
    reduced = padded.reshape(padded.shape[0] // factor, factor, padded.shape[1] // factor, factor).mean(axis=(1, 3))
    reduced = ndimage.gaussian_filter(reduced, sigma / factor, mode="nearest", truncate=4.0)
    enlarged = ndimage.zoom(reduced, factor, order=3, mode="nearest", grid_mode=True)
    return enlarged[:h, :w]


def fill_transparent(values, mask, tileable):
    """Fill texels outside the mask by normalized convolution at growing scales (push-pull), keeping opaque texels as they are."""
    if mask.all():
        return values.copy()
    if not mask.any():
        raise ValueError("The texture has no opaque texels (alpha >= 250).")
    filled = values.copy()
    weights = mask.astype(np.float64)
    unfilled = ~mask
    sigma = 1.0
    largest = 2.0 * max(values.shape)
    while unfilled.any():
        numerator = gaussian_blur(values * weights, sigma, tileable)
        denominator = gaussian_blur(weights, sigma, tileable)
        ready = unfilled & (denominator > 1e-4)
        filled[ready] = numerator[ready] / denominator[ready]
        unfilled &= ~ready
        sigma *= 2.0
        if sigma > largest and unfilled.any():
            filled[unfilled] = values[mask].mean()
            break
    # Smooth the filled region slightly so it carries no texel-scale steps from the scale changes above
    smooth = gaussian_blur(filled, 1.0, tileable)
    filled[~mask] = smooth[~mask]
    return filled


def scharr_gradients(height, tileable):
    """Normalized Scharr gradients of a height map with rows top-first: (dh/du, dh/dv) per texel, v pointing up the image.

    gx = sum_dr w_dr (h[r+dr][c+1] - h[r+dr][c-1]),  gv = sum_dc w_dc (h[r-1][c+dc] - h[r+1][c+dc]),  w = (3, 10, 3) / 32,
    with clamp-to-edge borders (wrap for tileable textures). This is the kernel Helios applies to bump maps.
    """
    p = np.pad(height, 1, mode="wrap" if tileable else "edge")
    H, W = height.shape
    def at(dr, dc):
        return p[1 + dr:1 + dr + H, 1 + dc:1 + dc + W]
    gx = sum(SCHARR_W[i] * (at(dr, 1) - at(dr, -1)) for i, dr in enumerate((-1, 0, 1)))
    gv = sum(SCHARR_W[i] * (at(-1, dc) - at(1, dc)) for i, dc in enumerate((-1, 0, 1)))
    return gx, gv


def normals_from_height(height, strength, tileable):
    """Unit tangent-space normals HxWx3 of a height map in [0, 1], for a displacement of `strength` texels over the full range."""
    gx, gv = scharr_gradients(height, tileable)
    n = np.stack([-strength * gx, -strength * gv, np.ones_like(gx)], axis=2)
    return n / np.linalg.norm(n, axis=2, keepdims=True)


# ----------------------------------------------------------------------------------------------------------------------
# Height estimation
# ----------------------------------------------------------------------------------------------------------------------

def band_scales(flatten_sigma):
    """Difference-of-Gaussian scales 1, 2, 4, ... texels, the largest band ending at the flattening scale."""
    scales = [0.0]
    sigma = 1.0
    while sigma <= flatten_sigma / 2.0:
        scales.append(sigma)
        sigma *= 2.0
    scales.append(min(sigma, flatten_sigma))
    return scales


def band_weight(k, weights):
    """Weight of band k (scale 2^(k-1)): fine for k = 0-1, medium for 2-3, coarse for 4 and up."""
    fine, medium, coarse = weights
    return fine if k <= 1 else (medium if k <= 3 else coarse)


def estimate_height(luminance, mask, tileable=False, presmooth=0.7, flatten_sigma=None, weights=(1.0, 0.5, 0.25), invert=False,
                    height_bits=16, edge_erode=2):
    """Height map in [0, 1] (quantized to height_bits) from a linear luminance (or shading) image and its opaque mask.

    The outermost edge_erode texels of the opaque region are treated like transparent ones (filled from the interior), because the
    cut-out edge of a photographed leaf is usually blended with the background it was cut from and would otherwise read as a rim.
    """
    H, W = luminance.shape
    if edge_erode > 0 and not mask.all():
        eroded = ndimage.binary_erosion(mask, iterations=int(edge_erode), border_value=0 if not tileable else 1)
        if eroded.sum() >= 0.25 * mask.sum():
            mask = eroded
    if flatten_sigma is None:
        flatten_sigma = 0.125 * min(H, W)
    flatten_sigma = max(2.0, float(flatten_sigma))
    L = np.log(np.clip(luminance, 0.0, None) + LOG_EPS)
    L = fill_transparent(L, mask, tileable)
    if presmooth > 0:
        L = gaussian_blur(L, presmooth, tileable)

    # Illumination flattening: remove the brightness averaged over the flattening scale, over the opaque texels only
    m = mask.astype(np.float64)
    if mask.all():
        background = gaussian_blur(L, flatten_sigma, tileable)
    else:
        numerator = gaussian_blur(L * m, flatten_sigma, tileable)
        denominator = gaussian_blur(m, flatten_sigma, tileable)
        background = np.where(denominator > 1e-6, numerator / np.maximum(denominator, 1e-6), gaussian_blur(L, flatten_sigma, tileable))
    detail = L - background

    # Multi-scale band-pass: sum_k w_k * sigma_k * (G_sigma_k - G_sigma_k+1) detail, with the finest band starting at the texel itself
    scales = band_scales(flatten_sigma)
    blurred = [gaussian_blur(detail, s, tileable) for s in scales]
    height = np.zeros_like(detail)
    for k in range(len(scales) - 1):
        size = max(scales[k], 0.5)
        height += band_weight(k, weights) * size * (blurred[k] - blurred[k + 1])
    if invert:
        height = -height

    # Robust normalization over the opaque texels
    lo, hi = np.percentile(height[mask], [0.5, 99.5])
    if hi - lo < 1e-6:
        normalized = np.full_like(height, 0.5)
    else:
        normalized = np.clip((height - lo) / (hi - lo), 0.0, 1.0)
    levels = 65535.0 if height_bits == 16 else 255.0
    return np.round(normalized * levels) / levels


# ----------------------------------------------------------------------------------------------------------------------
# Preview
# ----------------------------------------------------------------------------------------------------------------------

def relight(normals, sun_elevation_deg=25.0, sun_azimuth_deg=135.0):
    """Gray Lambertian relighting by a low sun (azimuth measured from +u toward +v; 135 = from the upper left) plus a dim sky."""
    el, az = np.radians(sun_elevation_deg), np.radians(sun_azimuth_deg)
    sun = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    direct = np.clip(normals @ sun, 0.0, None) / sun[2]
    sky = 0.5 * (1.0 + normals[..., 2])
    return np.clip(0.75 * direct * sun[2] + 0.25 * sky, 0.0, 1.0)


def write_preview(path, rgb, normal_rgb, height, normals, mask):
    lit = relight(normals)
    lit[~mask] = 0.0
    panels = [
        np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8),
        normal_rgb,
        np.repeat(np.round(height * 255).astype(np.uint8)[..., None], 3, axis=2),
        np.repeat(np.round(lit ** (1 / 2.2) * 255).astype(np.uint8)[..., None], 3, axis=2),
    ]
    gap = np.full((rgb.shape[0], 4, 3), 255, dtype=np.uint8)
    strip = np.concatenate([panels[0], gap, panels[1], gap, panels[2], gap, panels[3]], axis=1)
    longest = max(strip.shape[:2])
    image = Image.fromarray(strip)
    if longest > 2400:
        scale = 2400 / longest
        image = image.resize((int(strip.shape[1] * scale), int(strip.shape[0] * scale)), Image.LANCZOS)
    image.save(path)


# ----------------------------------------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------------------------------------

def generate(texture_path, output_dir, strength=8.0, height_source="luminance", shading_path=None, invert=False, weights=(1.0, 0.5, 0.25),
             flatten_sigma=None, presmooth=0.7, tileable=False, directx=False, height_bits=16, keep_alpha=False, preview=False, edge_erode=2):
    """Write <stem>_normal.png, <stem>_height.png and <stem>_pbr.json (and <stem>_preview.png) for one texture; returns the output paths."""
    luminance, mask, rgb = load_texture(texture_path)
    if height_source == "shading":
        if shading_path is None:
            raise ValueError("--height-source shading needs a shading image (--shading).")
        luminance = load_shading(shading_path, luminance.shape)
    elif height_source != "luminance":
        raise ValueError(f"Unknown height source '{height_source}' (use 'luminance' or 'shading').")
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError(f"The strength must be positive (got {strength}).")

    height = estimate_height(luminance, mask, tileable=tileable, presmooth=presmooth, flatten_sigma=flatten_sigma, weights=weights,
                             invert=invert, height_bits=height_bits, edge_erode=edge_erode)
    normals = normals_from_height(height, strength, tileable)
    normal_rgb = encode_normals(normals, mask, directx=directx)

    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(texture_path))[0]
    normal_path = os.path.join(output_dir, f"{stem}_normal.png")
    height_path = os.path.join(output_dir, f"{stem}_height.png")
    json_path = os.path.join(output_dir, f"{stem}_pbr.json")
    if keep_alpha and not mask.all():
        alpha = np.asarray(Image.open(texture_path).convert("RGBA"))[..., 3]
        Image.fromarray(np.dstack([normal_rgb, alpha])).save(normal_path)
    else:
        Image.fromarray(normal_rgb).save(normal_path)
    write_height_png(height_path, height, height_bits)

    flatten_used = flatten_sigma if flatten_sigma is not None else 0.125 * min(mask.shape)
    tilt = np.degrees(np.arccos(np.clip(normals[..., 2][mask], -1, 1)))
    record = {
        "texture": os.path.abspath(texture_path),
        "normal_map": os.path.basename(normal_path),
        "height_map": os.path.basename(height_path),
        "parameters": {"strength": strength, "height_source": height_source, "shading": shading_path, "invert": invert,
                       "weights": list(weights), "flatten_sigma": flatten_used, "presmooth": presmooth, "tileable": tileable,
                       "directx": directx, "height_bits": height_bits, "edge_erode": edge_erode},
        "statistics": {"opaque_fraction": float(mask.mean()), "tilt_deg_median": float(np.median(tilt)), "tilt_deg_p95": float(np.percentile(tilt, 95))},
        "helios": {
            "normal_map": f'context.setMaterialNormalMap("label", "{os.path.basename(normal_path)}"{", 1.f, true" if directx else ""});',
            "bump_map": f'context.setMaterialBumpMap("label", "{os.path.basename(height_path)}", {strength:g}f);',
        },
    }
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)
    outputs = [normal_path, height_path, json_path]
    if preview:
        preview_path = os.path.join(output_dir, f"{stem}_preview.png")
        write_preview(preview_path, rgb, normal_rgb if not directx else encode_normals(normals, mask), height, normals, mask)
        outputs.append(preview_path)
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate tangent-space normal and bump (height) maps from RGB(A) textures for Helios.")
    parser.add_argument("textures", nargs="+", help="texture images (PNG or JPEG; PNG alpha is honoured)")
    parser.add_argument("-o", "--output", required=True, help="output directory")
    parser.add_argument("--strength", type=float, default=8.0, help="texels of displacement for the full height range (= Helios bump strength); default 8")
    parser.add_argument("--height-source", choices=["luminance", "shading"], default="luminance", help="luminance of the texture, or a shading image")
    parser.add_argument("--shading", nargs="+", help="shading image(s) from utilities/leaf_class_maps/decompose.py, one per texture, in order")
    parser.add_argument("--invert", action="store_true", help="dark parts are raised (default: dark parts are low)")
    parser.add_argument("--weights", type=float, nargs=3, metavar=("FINE", "MEDIUM", "COARSE"), default=(1.0, 0.5, 0.25),
                        help="weights of fine (1-2 texel), medium (4-8) and coarse (16+) relief; default 1 0.5 0.25")
    parser.add_argument("--flatten-sigma", type=float, default=None, help="scale (texels) of the lighting removed; default 1/8 of the smaller image side")
    parser.add_argument("--presmooth", type=float, default=0.7, help="Gaussian pre-smoothing (texels) against JPEG/sensor noise; default 0.7, 0 for none")
    parser.add_argument("--tileable", action="store_true", help="treat the texture as seamlessly tiling (wrap-around borders)")
    parser.add_argument("--directx", action="store_true", help="write a DirectX (green down) normal map; Helios then needs flip_green = true")
    parser.add_argument("--height-bits", type=int, choices=[8, 16], default=16, help="bit depth of the height map; default 16")
    parser.add_argument("--edge-erode", type=int, default=2, help="texels at the edge of a cut-out treated as background (blended edge colours); default 2")
    parser.add_argument("--keep-alpha", action="store_true", help="copy the texture's alpha channel into the normal map")
    parser.add_argument("--preview", action="store_true", help="also write <stem>_preview.png")
    args = parser.parse_args(argv)

    if args.height_source == "shading":
        if not args.shading or len(args.shading) != len(args.textures):
            parser.error("--height-source shading needs one --shading image per texture")
    elif args.shading:
        parser.error("--shading is used only with --height-source shading")

    for i, texture in enumerate(args.textures):
        outputs = generate(texture, args.output, strength=args.strength, height_source=args.height_source,
                           shading_path=args.shading[i] if args.shading else None, invert=args.invert, weights=tuple(args.weights),
                           flatten_sigma=args.flatten_sigma, presmooth=args.presmooth, tileable=args.tileable, directx=args.directx,
                           height_bits=args.height_bits, keep_alpha=args.keep_alpha, preview=args.preview, edge_erode=args.edge_erode)
        print(f"{texture}: " + ", ".join(outputs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
