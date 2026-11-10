"""Self-test of pbr_maps.py: conventions, edge handling, tiling, height/normal consistency, and recovery of a known relief.

Usage:  python selftest.py [SCRATCH_DIR]
Exits with status 1 if any check fails. SCRATCH_DIR (default: a temporary directory) receives the test images.
"""
import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True  # keep __pycache__ out of the source tree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pbr_maps  # noqa: E402

failures = []


def check(name, condition, detail):
    print(f"[{'PASS' if condition else 'FAIL'}] {name}: {detail}")
    if not condition:
        failures.append(name)


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def write_srgb(path, linear_rgb, alpha=None):
    c = np.clip(linear_rgb, 0, 1)
    srgb = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    data = np.round(srgb * 255).astype(np.uint8)
    if alpha is not None:
        data = np.dstack([data, alpha.astype(np.uint8)])
    Image.fromarray(data).save(path)


# ----------------------------------------------------------------------------------------------------------------------
# (a) Recovery of a known relief from a sky-lit rendering with ambient occlusion
# ----------------------------------------------------------------------------------------------------------------------

def bumps_on_flat(size, rng):
    """Height in texels: Gaussian bumps, two ridges and a groove on a flat base."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    h = np.zeros((size, size))
    for _ in range(14):
        cx, cy = rng.uniform(20, size - 20, 2)
        r = rng.uniform(5, 14)
        h += rng.uniform(3, 8) * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * r * r))
    h += 5.0 * np.exp(-((x - 0.3 * size) ** 2) / (2 * 3.0 ** 2)) * (y > 0.15 * size) * (y < 0.85 * size)   # ridge along v
    h += 4.0 * np.exp(-((y - 0.75 * size) ** 2) / (2 * 2.5 ** 2)) * (x > 0.45 * size)                        # ridge along u
    h -= 5.0 * np.exp(-((x + y - 1.1 * size) ** 2) / (2 * (3.0 * np.sqrt(2)) ** 2)) * (x < 0.9 * size)     # diagonal groove
    return h


def furrowed_surface(size, rng):
    """Height in texels: wandering furrows (as in bark or between planks) cut into a mildly rough surface."""
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    h = np.zeros((size, size))
    for _ in range(10):
        centre, width, wander, period = rng.uniform(0, size), rng.uniform(2, 5), rng.uniform(-8, 8), rng.uniform(15, 40)
        h -= 4.0 * np.exp(-((x - centre - wander * np.sin(y / period)) ** 2) / (2 * width * width))
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    k = np.maximum(np.sqrt(fx ** 2 + fy ** 2), 1e-9)
    spectrum = (rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))) * k ** -2.0 * (k > 1 / 64) * np.exp(-(k / 0.15) ** 2)
    rough = np.real(np.fft.ifft2(spectrum))
    return h + 1.0 * rough / rough.std()


def true_normals(h):
    """Normals of a height field in texel units, rows top-first (v up), with central differences."""
    hp = np.pad(h, 1, mode="edge")
    gx = 0.5 * (hp[1:-1, 2:] - hp[1:-1, :-2])
    gv = 0.5 * (hp[:-2, 1:-1] - hp[2:, 1:-1])
    n = np.stack([-gx, -gv, np.ones_like(h)], axis=2)
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def sky_visibility(h, directions=16, radius=40):
    """Cosine-weighted sky visibility from the horizon angle in each azimuth: V = mean_d (1 - sin^2(max(0, horizon_d)))."""
    size = h.shape[0]
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    visibility = np.zeros_like(h)
    for k in range(directions):
        a = 2 * np.pi * k / directions
        dx, dy = np.cos(a), np.sin(a)
        tan_horizon = np.zeros_like(h)
        for t in range(1, radius + 1):
            xs = np.clip(np.round(x + dx * t).astype(int), 0, size - 1)
            ys = np.clip(np.round(y + dy * t).astype(int), 0, size - 1)
            tan_horizon = np.maximum(tan_horizon, (h[ys, xs] - h) / t)
        sin2 = tan_horizon ** 2 / (1 + tan_horizon ** 2)
        visibility += 1.0 - sin2
    return visibility / directions


def recover(h, scratch, name):
    """Render h under a uniform sky with ambient occlusion, I = rho (1 + n_z)/2 V, recover normals with the best global strength."""
    n_true = true_normals(h)
    image = 0.5 * 0.5 * (1.0 + n_true[..., 2]) * sky_visibility(h)
    write_srgb(os.path.join(scratch, f"recovery_{name}_input.png"), np.repeat(image[..., None], 3, axis=2))
    height = pbr_maps.estimate_height(image, np.ones(h.shape, dtype=bool))
    tilted = np.degrees(np.arccos(n_true[..., 2])) > 5.0
    # The relief's absolute scale is not recoverable from shading: fit the best global strength on the tilted texels
    strengths = np.geomspace(0.5, 300, 150)
    errors = [np.median(angle_deg(pbr_maps.normals_from_height(height, s, False), n_true)[tilted]) for s in strengths]
    best = strengths[int(np.argmin(errors))]
    n_est = pbr_maps.normals_from_height(height, best, False)
    err = angle_deg(n_est, n_true)
    flat_err = np.degrees(np.arccos(n_true[..., 2]))
    agree = np.mean(np.sum(n_est[..., :2] * n_true[..., :2], axis=2)[tilted] > 0)
    corr = np.corrcoef(height.ravel(), h.ravel())[0, 1]
    Image.fromarray(pbr_maps.encode_normals(n_est)).save(os.path.join(scratch, f"recovery_{name}_normal.png"))
    Image.fromarray(pbr_maps.encode_normals(n_true)).save(os.path.join(scratch, f"recovery_{name}_truth.png"))
    print(f"    {name}: best strength {best:.1f}; height correlation {corr:.2f}; error over all texels mean {err.mean():.1f}, median {np.median(err):.1f}, "
          f"p90 {np.percentile(err, 90):.1f} deg; tilted > 5 deg ({tilted.mean() * 100:.0f}% of texels): median {np.median(err[tilted]):.1f}, "
          f"p90 {np.percentile(err[tilted], 90):.1f} deg vs {np.median(flat_err[tilted]):.1f} for a flat map; tilt direction right on {agree * 100:.0f}%")
    return err, flat_err, tilted, agree, corr


def test_recovery(scratch):
    # Relief whose shading is dominated by its cavities, which is what the method assumes (bark, planks, pitted or wrinkled skins)
    err, flat_err, tilted, agree, corr = recover(furrowed_surface(256, np.random.default_rng(7)), scratch, "furrows")
    check("(a) furrowed relief: median error over all texels < 12 deg", np.median(err) < 12.0, f"{np.median(err):.1f} deg")
    check("(a) furrowed relief: median error on tilted texels at least 15% below that of a flat map", np.median(err[tilted]) < 0.85 * np.median(flat_err[tilted]),
          f"{np.median(err[tilted]):.1f} vs {np.median(flat_err[tilted]):.1f} deg")
    check("(a) furrowed relief: tilt direction right on > 70% of tilted texels", agree > 0.7, f"{agree * 100:.0f}%")
    check("(a) furrowed relief: recovered height correlates with the truth (r > 0.5)", corr > 0.5, f"r = {corr:.2f}")
    # Known limitation, reported but not asserted: under a uniform sky, the flanks of an isolated convex bump on a flat base are darker
    # than both its top and the flat around it, which brightness cannot tell from a groove
    recover(bumps_on_flat(256, np.random.default_rng(7)), scratch, "bumps_on_flat")
    print("    (bumps_on_flat is a known limitation of brightness-as-height and is not asserted; see README)")


# ----------------------------------------------------------------------------------------------------------------------
# (b) Sign convention
# ----------------------------------------------------------------------------------------------------------------------

def test_convention(scratch):
    # The kernel on an exact ramp: h = a * column gives gx = a exactly, n = normalize(-s a, 0, 1)
    a, s = 0.01, 8.0
    ramp = a * np.tile(np.arange(64, dtype=np.float64), (32, 1))
    n = pbr_maps.normals_from_height(ramp, s, False)
    expected = np.array([-s * a, 0.0, 1.0]) / np.linalg.norm([-s * a, 0.0, 1.0])
    check("(b) ramp along +u gives n = normalize(-s*a, 0, 1) in the interior", np.allclose(n[5:-5, 5:-5], expected, atol=1e-12), f"n = {n[16, 32]}")
    up = a * np.tile(np.arange(64, dtype=np.float64)[::-1, None], (1, 32))  # larger toward row 0 = the top
    nv = pbr_maps.normals_from_height(up, s, False)
    check("(b) height rising toward the top of the image gives n = normalize(0, -s*a, 1)", np.allclose(nv[5:-5, 5:-5], [0.0, expected[0], expected[2]], atol=1e-12),
          f"n = {nv[32, 16]}")

    # End to end through an image: a bright (raised) bump
    size = 128
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    bump = 0.2 + 0.6 * np.exp(-((x - 64) ** 2 + (y - 64) ** 2) / (2 * 10.0 ** 2))
    path = os.path.join(scratch, "convention_bump.png")
    write_srgb(path, np.repeat(bump[..., None], 3, axis=2))
    pbr_maps.generate(path, scratch, strength=8.0)
    rgb = np.asarray(Image.open(os.path.join(scratch, "convention_bump_normal.png"))).astype(int)
    left, right, below, above = rgb[64, 52], rgb[64, 76], rgb[76, 64], rgb[52, 64]
    check("(b) height rising toward +u (left flank of a bump) gives R < 128, the right flank R > 128", left[0] < 128 < right[0], f"R {left[0]} / {right[0]}")
    check("(b) height rising toward the top (lower flank) gives G < 128, the upper flank G > 128", below[1] < 128 < above[1], f"G {below[1]} / {above[1]}")
    pbr_maps.generate(path, scratch, strength=8.0, directx=True)
    rgb_dx = np.asarray(Image.open(os.path.join(scratch, "convention_bump_normal.png"))).astype(int)
    check("(b) --directx flips only green", rgb_dx[76, 64, 1] > 128 and rgb_dx[64, 52, 0] == left[0], f"G {rgb_dx[76, 64, 1]}, R {rgb_dx[64, 52, 0]}")


# ----------------------------------------------------------------------------------------------------------------------
# (c) No rim at the edge of a cut-out
# ----------------------------------------------------------------------------------------------------------------------

def test_cutout(scratch):
    size = 160
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    disk = (x - 80) ** 2 + (y - 80) ** 2 < 55 ** 2
    color = np.zeros((size, size, 3))
    color[...] = (0.9, 0.9, 0.9)      # bright background where transparent
    color[disk] = (0.05, 0.3, 0.05)    # dark green disk where opaque
    alpha = np.where(disk, 255, 0)
    path = os.path.join(scratch, "cutout_disk.png")
    write_srgb(path, color, alpha)
    pbr_maps.generate(path, scratch, strength=8.0)
    n = pbr_maps.decode_normals(np.asarray(Image.open(os.path.join(scratch, "cutout_disk_normal.png"))))
    tilt = np.degrees(np.arccos(np.clip(n[..., 2], -1, 1)))
    check("(c) constant-colour opaque disk on a contrasting transparent background: max tilt < 1 deg", tilt[disk].max() < 1.0, f"{tilt[disk].max():.2f} deg")

    # A textured disk: the texels at the rim must tilt no more than the interior does
    rng = np.random.default_rng(3)
    noise = pbr_maps.gaussian_blur(rng.normal(size=(size, size)), 2.0, False)
    color2 = color.copy()
    color2[disk] = np.clip(0.2 + 0.1 * noise[disk, None] / noise.std(), 0.01, 1)
    write_srgb(path, color2, alpha)
    pbr_maps.generate(path, scratch, strength=8.0)
    n2 = pbr_maps.decode_normals(np.asarray(Image.open(os.path.join(scratch, "cutout_disk_normal.png"))))
    tilt2 = np.degrees(np.arccos(np.clip(n2[..., 2], -1, 1)))
    r = np.sqrt((x - 80) ** 2 + (y - 80) ** 2)
    rim, inner = disk & (r > 52), disk & (r < 40)
    check("(c) textured disk: mean tilt at the rim within 1.3x the interior's", tilt2[rim].mean() < 1.3 * tilt2[inner].mean(),
          f"rim {tilt2[rim].mean():.1f} vs interior {tilt2[inner].mean():.1f} deg")


# ----------------------------------------------------------------------------------------------------------------------
# (d) Tileable textures
# ----------------------------------------------------------------------------------------------------------------------

def test_tileable(scratch):
    size = 192
    rng = np.random.default_rng(11)
    spectrum = np.fft.fft2(rng.normal(size=(size, size)))
    fy, fx = np.meshgrid(np.fft.fftfreq(size), np.fft.fftfreq(size), indexing="ij")
    periodic = np.real(np.fft.ifft2(spectrum * np.exp(-(fx ** 2 + fy ** 2) / (2 * 0.04 ** 2))))
    periodic = 0.3 + 0.15 * periodic / periodic.std()
    image = np.clip(periodic, 0.01, 1)
    mask = np.ones(image.shape, dtype=bool)
    shift = 80
    h1 = pbr_maps.estimate_height(image, mask, tileable=True)
    h2 = pbr_maps.estimate_height(np.roll(image, (shift, shift), axis=(0, 1)), mask, tileable=True)
    shift_error = np.abs(np.roll(h1, (shift, shift), axis=(0, 1)) - h2).max()
    check("(d) --tileable output is shift-invariant (wrap-around everywhere)", shift_error <= 2.0 / 65535, f"max height difference {shift_error * 65535:.2f} LSB")
    n = pbr_maps.normals_from_height(h1, 8.0, True)
    interior_jump = np.abs(np.diff(n, axis=1)).max()
    seam_jump = np.abs(n[:, 0] - n[:, -1]).max()
    seam_jump_v = np.abs(n[0, :] - n[-1, :]).max()
    check("(d) the seam is no rougher than the interior", max(seam_jump, seam_jump_v) <= interior_jump * 1.0001,
          f"seam {max(seam_jump, seam_jump_v):.3f} vs interior max step {interior_jump:.3f}")


# ----------------------------------------------------------------------------------------------------------------------
# (e) The normal map equals the normals Helios computes from the height map
# ----------------------------------------------------------------------------------------------------------------------

def test_height_consistency(scratch):
    size = 128
    rng = np.random.default_rng(5)
    image = 0.25 + 0.1 * pbr_maps.gaussian_blur(rng.normal(size=(size, size)), 1.5, False)
    y, x = np.mgrid[0:size, 0:size]
    alpha = np.where((x - 64) ** 2 + (y - 64) ** 2 < 58 ** 2, 255, 0)
    path = os.path.join(scratch, "consistency.png")
    write_srgb(path, np.repeat(np.clip(image, 0.01, 1)[..., None], 3, axis=2), alpha)
    for strength in (2.0, 8.0, 30.0):
        pbr_maps.generate(path, scratch, strength=strength)
        stored = np.asarray(Image.open(os.path.join(scratch, "consistency_normal.png"))).astype(int)
        height = pbr_maps.read_height_png(os.path.join(scratch, "consistency_height.png"))
        recomputed = pbr_maps.encode_normals(pbr_maps.normals_from_height(height, strength, False)).astype(int)
        opaque = alpha >= 250
        diff = np.abs(stored - recomputed)[opaque].max()
        check(f"(e) normal map == normals recomputed from the 16-bit height map (strength {strength:g}) within 1 LSB", diff <= 1, f"max {diff} LSB")


def main():
    scratch = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="pbr_maps_selftest_")
    os.makedirs(scratch, exist_ok=True)
    print(f"scratch directory: {scratch}")
    test_convention(scratch)
    test_cutout(scratch)
    test_tileable(scratch)
    test_height_consistency(scratch)
    test_recovery(scratch)
    if failures:
        print(f"\n{len(failures)} check(s) FAILED: " + "; ".join(failures))
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
