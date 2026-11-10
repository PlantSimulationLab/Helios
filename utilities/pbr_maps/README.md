# Normal and bump maps from a texture photograph

`pbr_maps.py` makes a tangent-space normal map and a bump (height) map from an RGB(A) texture photograph, for the
normal and bump maps of Helios materials (see the User Guide, "Normal and Bump Maps"). `selftest.py` checks it.

    python pbr_maps.py -o OUTPUT_DIR [options] TEXTURE [TEXTURE ...]

For each texture it writes, at the texture's resolution:

| File | Contents |
|---|---|
| `<stem>_normal.png` | tangent-space normal map, 8-bit RGB, OpenGL convention: +X (red) = +u = image right, +Y (green) = +v = image up |
| `<stem>_height.png` | height map, 16-bit grayscale (0 lowest, 65535 highest) |
| `<stem>_pbr.json` | the parameters used, statistics, and the equivalent Helios calls |
| `<stem>_preview.png` | with `--preview`: input, normal map, height, and the surface relit by a low sun from the upper left |

Use either map in Helios:

```cpp
context.setMaterialNormalMap("bark", "AlmondBark_normal.png");
// or, equivalently (same normals to 1 LSB over the opaque texels):
context.setMaterialBumpMap("bark", "AlmondBark_height.png", 12.f);   // 12 = the --strength the maps were made with
```

The normal map is computed from the stored 16-bit height with the kernel Helios uses for bump maps, so the two calls
agree. The bump map gives you a strength you can change later without regenerating anything. With `--tileable` the
generator wraps around at the borders but Helios clamps there, so the two differ in the outermost texel row and column.

## Requirements

numpy, scipy and Pillow (`requirements.txt`). On Farm, for example:

    module load conda
    conda create -p /group/<pigrp>/<user>/envs/pbrmaps numpy scipy pillow   # keep environments out of the 20 GB home
    conda activate /group/<pigrp>/<user>/envs/pbrmaps
    python selftest.py

Run large batches on a compute node (`srun`), not a login node.

## Options

| Option | Default | Meaning |
|---|---|---|
| `--strength S` | 8 | displacement, in texels, of the full height range: larger gives steeper normals. Same meaning as the Helios bump-map strength |
| `--invert` | off | dark parts are raised (by default dark parts are low) |
| `--weights F M C` | 1 0.5 0.25 | weights of fine (about 1-2 texels), medium (4-8) and coarse (16 and up) relief |
| `--flatten-sigma PX` | 1/8 of the smaller side | brightness varying over larger scales is treated as lighting and removed |
| `--presmooth PX` | 0.7 | Gaussian pre-smoothing against JPEG and sensor noise; 0 for none |
| `--edge-erode N` | 2 | texels at the edge of a cut-out (alpha) treated as background, because their colour is blended with the background the leaf was cut from |
| `--height-source shading --shading FILE` | luminance | use a diffuse shading image instead of the texture's luminance (below) |
| `--tileable` | off | the texture tiles seamlessly: wrap around at the borders |
| `--directx` | off | write a DirectX (green down) normal map; Helios then needs `flip_green = true` |
| `--height-bits 8` | 16 | 8-bit height map |
| `--keep-alpha` | off | copy the texture's alpha into the normal map |

Texels with alpha below 250 (Helios' opacity threshold) get the flat normal (128, 128, 255).

## Method

A photograph's brightness mixes the surface's albedo with how it is lit. Under the diffuse light most texture photographs
are taken in, cavities (furrows, grooves, pores, the gaps between bark plates) see less sky and are darker, while raised parts
are brighter. So brightness at the scale of a feature is a usable proxy for relief, and brightness that varies over the whole
image is mostly lighting. This is the classical approach that tools such as Materialize and AwesomeBump take. The steps are:

1. **Height source.** Linear luminance (sRGB decoded, Rec. 709 weights), or the diffuse shading of an intrinsic
   decomposition (`utilities/leaf_class_maps/decompose.py` writes `<stem>_shading.png`), which separates albedo marks from
   shading.
2. **Log brightness.** L = log(Y + 0.001), so the result depends on relative brightness. A light pre-smoothing follows.
3. **Mask-aware fill.** Transparent texels, and the blended outermost opaque texels, are filled by normalized convolution
   at growing scales. The cut-out edge then produces no false rim.
4. **Illumination flattening.** L minus its mean over the opaque texels at the scale `--flatten-sigma`.
5. **Multi-scale band-pass.** H = Σ_k w_k σ_k (G_σk − G_σk+1) L, with Gaussian scales σ_k = 1, 2, 4, ... texels. The σ_k
   factor turns contrast at feature width σ into height of order contrast × σ. w_k is the fine, medium or coarse weight.
6. **Normalization.** Optional inversion, then robust normalization to [0, 1] between the 0.5th and 99.5th percentiles of the
   opaque texels, then quantization.
7. **Normals.** Normals from the quantized height with the normalized Scharr kernel Helios uses:
   - gx = Σ w (h[r+dr][c+1] − h[r+dr][c−1]) and gv = Σ w (h[r−1][c+dc] − h[r+1][c+dc]), with w = (3, 10, 3)/32;
   - n = normalize(−S gx, −S gv, 1).

   A height ramp h = a·column therefore gives n = normalize(−S a, 0, 1).

## Limits

- **Dark marks.** Dark albedo marks (lichen, spots, pigmented veins, printed lines) become false grooves. Cast shadows of a
  directional light become false slopes. `--height-source shading` removes most albedo marks.
- **Isolated bumps.** The method recovers cavity-dominated relief (bark, planks, wrinkled or pitted skins) well. It does
  poorly on isolated convex bumps on a flat base: under a uniform sky their flanks are darker than both their top and the
  surroundings, which brightness cannot tell from a groove. `selftest.py` measures both cases:

  | Case | Height correlation | Median angular error, tilted texels | Flat map, same texels | Tilt direction right |
  |---|---|---|---|---|
  | Furrowed surface | 0.70 | 11.3° | 14.5° | 72% |
  | Bumps on a flat base | −0.19 | 14.1° | 14.2° | 39% |

- **Sign ambiguity.** Whether bright is high is a property of the surface. Check the preview, and use `--invert` when raised
  parts are dark. On the upper side of most broad leaves, for example, the veins are pale and slightly sunken.
- **Absolute height.** The absolute height cannot be recovered from one photograph, so `--strength` is a choice. Normals
  are also tied to the texel scale: the same map on a larger primitive looks steeper than it physically should.
- **Learned models.** Learned single-image models (e.g. DeepBump, GPL-3.0) can do better on some textures, but their licence
  keeps them out of Helios; use them as external tools if you like. Their output loads with `setMaterialNormalMap()`.

## Examples (the maps of the Helios normal-map demo)

    python pbr_maps.py -o assets --preview --strength 12 --tileable plugins/plantarchitecture/assets/textures/AlmondBark.jpg
    python pbr_maps.py -o assets --preview --strength 8 plugins/visualizer/textures/wood_board.jpg
    python pbr_maps.py -o assets --preview --invert --weights 1 0.5 0.1 plugins/plantarchitecture/assets/textures/GrapeLeaf.png
    python pbr_maps.py -o assets --preview --strength 3 --weights 1 0.4 0 plugins/plantarchitecture/assets/textures/AppleFruit.jpg

- **Almond bark.** The lenticels and plate edges come out as relief. The bark tile is close to seamless, hence `--tileable`.
- **Wood board.** Only the grain is relief.
- **Grape leaf.** The veins are set sunken (`--invert`), and coarse relief is weakened because the photographed leaf's own
  folds shade it.
- **Apple fruit.** Only the fine skin texture is kept (coarse weight 0), at low strength. The dark bands at the top and bottom
  of that texture (the poles of the sphere) still give a ridge there.
