"""Split leaf texture photographs into albedo and shading with Careaga & Aksoy's intrinsic decomposition (compphoto/Intrinsic, models 'v2':
"Colorful Diffuse Intrinsic Image Decomposition in the Wild", TOG 2024), for fitting spectral-texture classes to the albedo.

Usage (inside the venv that has the `intrinsic` package):  python decompose.py [--scale N] OUTPUT_DIR TEXTURE.png [TEXTURE.png ...]

Runs on CUDA when PyTorch has it, and on the CPU otherwise. On first use torch.hub asks, on the terminal, whether to trust the
rwightman/gen-efficientnet-pytorch repository the models are built on (answer y); when run without a terminal, add the line
"rwightman_gen-efficientnet-pytorch" to ~/.cache/torch/hub/trusted_list beforehand, or the script waits for an answer forever.

--scale N enlarges each photo N times (Lanczos) before decomposing and reduces the results back, for textures whose thin features
(a narrow midrib) are only a few pixels wide and would otherwise be taken for highlights.

For each texture it writes, at the texture's resolution:
  <stem>_albedo.png   the albedo, sRGB-encoded 8-bit RGBA, with the texture's own alpha channel (so it matches the Helios mask exactly)
  <stem>_shading.png  the grey diffuse shading (16-bit, linear, scaled so its 99th percentile over the leaf is 65535), for a later normal map
  <stem>_residual.png the non-diffuse residual (specular glints), sRGB 8-bit, for inspection
Transparent texels are filled with the leaf's median colour before decomposing, so the background does not shade the leaf edge.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image
from intrinsic.pipeline import load_models, run_pipeline

# The pipeline defaults to CUDA; without it (an Apple-silicon Mac, say) the models run on the CPU, which takes a minute or so per texture.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

OPAQUE = 250  # Helios treats alpha >= 250 (of 255) as opaque


def to_srgb(linear):
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * np.power(linear, 1 / 2.4) - 0.055)


def resize_to(array, width, height):
    """Resize an HxWxC or HxW float array to the texture's resolution, if the pipeline returned another"""
    if array.shape[0] == height and array.shape[1] == width:
        return array
    channels = [array] if array.ndim == 2 else [array[..., c] for c in range(array.shape[2])]
    resized = [np.asarray(Image.fromarray(c.astype(np.float32), mode="F").resize((width, height), Image.BILINEAR)) for c in channels]
    return resized[0] if array.ndim == 2 else np.stack(resized, axis=-1)


def main():
    arguments = sys.argv[1:]
    scale = 1
    if arguments[:1] == ["--scale"]:
        scale = int(arguments[1])
        arguments = arguments[2:]
    if len(arguments) < 2 or scale < 1:
        sys.exit(__doc__)
    sys.argv = [sys.argv[0]] + arguments
    output = sys.argv[1]
    os.makedirs(output, exist_ok=True)
    models = load_models("v2", device=DEVICE)
    for path in sys.argv[2:]:
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        height, width = rgba.shape[:2]
        alpha = rgba[..., 3]
        opaque = alpha >= OPAQUE
        if not opaque.any():
            sys.exit(f"{path}: no opaque texels")
        rgb = rgba[..., :3].astype(np.float32) / 255.0
        rgb[~opaque] = np.median(rgb[opaque], axis=0)
        if scale > 1:
            enlarged = Image.fromarray(np.round(255 * rgb).astype(np.uint8)).resize((width * scale, height * scale), Image.LANCZOS)
            rgb = np.asarray(enlarged).astype(np.float32) / 255.0
        results = run_pipeline(models, rgb, device=DEVICE)
        albedo = resize_to(np.asarray(results["hr_alb"], dtype=np.float32), width, height)
        shading = resize_to(np.asarray(results["dif_shd"], dtype=np.float32), width, height)
        residual = resize_to(np.asarray(results["residual"], dtype=np.float32), width, height)
        if albedo.ndim != 3 or albedo.shape[2] != 3:
            sys.exit(f"{path}: unexpected albedo shape {albedo.shape}")
        stem = os.path.splitext(os.path.basename(path))[0]
        albedo_rgba = np.dstack([np.round(255 * to_srgb(albedo)).astype(np.uint8), alpha])
        Image.fromarray(albedo_rgba, mode="RGBA").save(os.path.join(output, stem + "_albedo.png"))
        grey = shading.mean(axis=2) if shading.ndim == 3 else shading
        scale = np.percentile(grey[opaque], 99)
        if not np.isfinite(scale) or scale <= 0:
            sys.exit(f"{path}: shading has no positive values over the leaf")
        grey16 = np.round(65535 * np.clip(grey / scale, 0, 1)).astype(np.uint16)
        Image.fromarray(grey16).save(os.path.join(output, stem + "_shading.png"))
        Image.fromarray(np.round(255 * to_srgb(np.abs(residual))).astype(np.uint8)).save(os.path.join(output, stem + "_residual.png"))
        print(f"{stem}: {width} x {height}, albedo mean (linear) {albedo[opaque].mean(axis=0)}")


if __name__ == "__main__":
    main()
