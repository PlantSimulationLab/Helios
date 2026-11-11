# Leaf class maps

Scripts that make a class map of a leaf texture photograph and its class definitions, which LeafOptics turns into a spectral
texture (see the LeafOptics documentation, "Leaves with a Class Map"). This is how
`plugins/plantarchitecture/assets/textures/SorghumLeaf_classes.png` and `SorghumLeaf_classes.xml` were made.

A photograph's colours mix the leaf's reflectance with its lighting: the relief of veins and the leaf's curvature shade it, and
those shadows would otherwise be fitted as darker, greener tissue. So the photograph is first split into albedo and shading, and the
classes are fitted to the albedo.

## 1. Remove the shading: `decompose.py`

Uses the intrinsic decomposition of Careaga & Aksoy, "Colorful Diffuse Intrinsic Image Decomposition in the Wild" (ACM TOG 2024,
<https://github.com/compphoto/Intrinsic>, models `v2`), which needs PyTorch and a GPU is advisable. Install it in its own environment,
for example:

    python -m venv intrinsic-env && . intrinsic-env/bin/activate
    pip install torch torchvision                      # for your CUDA version
    pip install https://github.com/compphoto/Intrinsic/archive/main.zip

then

    python decompose.py [--scale N] OUTPUT_DIR TEXTURE.png [TEXTURE.png ...]

writes `<stem>_albedo.png` (with the texture's transparency), `<stem>_shading.png` and `<stem>_residual.png`. `--scale N` decomposes an
enlarged copy, which helps thin features survive.

The decomposition takes thin pale features, such as the midrib of a grass leaf, for highlights, and removes them with the shading.

## 2. For grass leaves, restore the midrib: `midrib.py`

    python midrib.py TEXTURE.png ALBEDO.png OUTPUT_PREFIX

traces the midrib as a pale, unsaturated ridge along the leaf with a smooth centre line and a tapering width. It writes
`OUTPUT_PREFIX_hybrid_albedo.png`, which is the albedo for the blade and, inside the midrib, the neighbouring blade's albedo times the
photograph's local midrib-to-blade colour ratio. It also writes a mask and an overlay of the traced midrib for checking. It expects the
leaf to run along the image's columns. It needs numpy, scipy and Pillow.

## 3. Fit the classes: `fit_classes`

A Helios program (it uses LeafOptics). Build it like any Helios project:

    cd fit_classes && mkdir build && cd build && cmake .. && make
    ./fit_classes TEXTURE.png SOURCE.png SPECIES CLASS_COUNT OUTPUT.png

It clusters the colours of SOURCE (the albedo, or the hybrid albedo) into CLASS_COUNT classes, from darkest to lightest. It then fits each
class relative to the LeafOptics library leaf SPECIES, as an RGB camera sees it under the sun. The class's hue gives its chlorophyll
(`chlorophyll_scale`, which multiplies chlorophyll and carotenoids), and the brightness that remains gives a factor on its visible
reflectance (`visible_reflectance_scale`). The program writes the class map `OUTPUT.png` and its definitions `OUTPUT.xml`. Only the colour
differences across the leaf are used, because a gain per channel puts the source's mean colour on the library leaf's colour. So a leaf
given other properties keeps its overall colour, and its tissues differ from it as in the photograph.

## The sorghum leaf

    python decompose.py albedo plugins/plantarchitecture/assets/textures/SorghumLeaf.png
    python midrib.py plugins/plantarchitecture/assets/textures/SorghumLeaf.png albedo/SorghumLeaf_albedo.png albedo/SorghumLeaf
    ./fit_classes plugins/plantarchitecture/assets/textures/SorghumLeaf.png albedo/SorghumLeaf_hybrid_albedo.png corn 24 \
        plugins/plantarchitecture/assets/textures/SorghumLeaf_classes.png

The midrib runs from the base to about 96% of the leaf's length, 5.5 to 0.8 texels wide on each side, and covers 7.8% of the leaf. The
24 classes are used because the albedo's colour varies smoothly, so a few classes would show as patches. LeafOptics has no sorghum
entry, so the fit is relative to its corn leaf.

To use a class map with a PlantArchitecture leaf, set `LeafPrototype::leaf_class_map_file` alongside `leaf_texture_file`.

## Surfaces that are not leaves: `fit_classes --surface`

    ./fit_classes --surface [--scale-exponent G] [--max-reflectance R] [--reflectance-factor F] TEXTURE SOURCE REFLECTIVITY_LABEL[,TRANSMISSIVITY_LABEL] CLASS_COUNT OUTPUT.png

fits a class map for bark, a fruit skin, or any other textured surface, relative to the spectra of the radiation plugin's spectral
library named by the labels (a texture without transparency gets a class on every texel). SOURCE is the image the classes are fitted
to: the albedo from `decompose.py`, as for a leaf, so that the relief of bark is not fitted as darker tissue, or TEXTURE itself for a
texture without shading. The source's colours are clustered as for a leaf, and each class is defined by a `reflectance_scale`, the
ratio of its luminance to the source's mean luminance in linear RGB: only the brightness differences across the surface are used, and
the surface keeps its spectrum's colour. The definitions are a `<surface_class_map>`, which `RadiationModel::applySpectralClassMaps()`
reads (see the radiation plugin documentation, "Spectral Textures"). `--scale-exponent G` raises each luminance ratio to the power
G before it is written (default 1), compressing the range of the classes: a photograph's brightness mixes reflectance with curvature
and gloss, and scaling a coloured spectrum up by the full ratio desaturates it, so a value such as 0.5 keeps the pattern while holding
the classes nearer the spectrum. Whatever the exponent, no class may reflect more than `--max-reflectance R` (default 0.6) at any
wavelength from 400 to 700 nm: the brightest parts of a photograph of a matte natural surface are gloss and lit curvature, not a
reflectance a diffuse surface can have, and a class scaled past this renders washed out. A class that would exceed it is held at the
cap, and the program says so. Expect this on any coloured fruit: the apple's two lightest classes were held before the exponent was
added. `--reflectance-factor F` multiplies every class by F, for a surface darker overall than the spectrum it is fitted relative to.

## The apple textures

    ./fit_classes T/AppleLeaf.png T/AppleLeaf.png default 16 T/AppleLeaf_classes.png
    python decompose.py albedo T/AppleBark.jpg
    ./fit_classes --surface T/AppleBark.jpg albedo/AppleBark_albedo.png bark_reflectivity_0000 8 T/AppleBark_classes.png
    ./fit_classes --surface --scale-exponent 0.5 --reflectance-factor 0.6 T/AppleFruit.jpg T/AppleFruit.jpg strawberry_reflectivity_0000 8 T/AppleFruit_classes.png

with `T` = `plugins/plantarchitecture/assets/textures`. The apple leaf is a flat scan under even light, and the fruit skin is
photographed evenly lit, so both are fitted from the photograph itself; the leaf relative to the LeafOptics default leaf (the library
has no apple entry). The bark photograph is shaded by its own relief, so it is decomposed first and fitted to its albedo. The library
has no apple fruit spectrum, so the fruit is fitted relative to its strawberry, with the class range compressed (exponent 0.5) and
every class scaled by 0.6: at the full range the lit side of an apple reflected more than twice the strawberry spectrum and rendered
pink, and at the strawberry's own level a red apple's skin, darker than a strawberry's, still rendered pale. The "apple" and "apple_fruitingwall" models set
`LeafPrototype::leaf_class_map_file`, `InternodeParameters::image_class_map_file` and `InflorescenceParameters::fruit_class_map_file`
to these maps.
