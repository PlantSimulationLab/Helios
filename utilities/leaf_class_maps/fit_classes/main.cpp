/** \file "main.cpp" Fit the classes of a class map to a texture photograph: a leaf's, relative to a LeafOptics leaf, or any other surface's, relative to a spectrum.

Usage (from a build directory below this one):

    ./fit_classes TEXTURE.png SOURCE.png SPECIES CLASS_COUNT OUTPUT.png
    ./fit_classes --surface [--scale-exponent G] [--max-reflectance R] [--reflectance-factor F] TEXTURE SOURCE REFLECTIVITY_LABEL[,TRANSMISSIVITY_LABEL] CLASS_COUNT OUTPUT.png

Leaf mode: TEXTURE is the leaf texture (its transparency decides which texels are leaf), SOURCE an image of the same size and transparency
whose colours the classes are fitted to (the albedo from decompose.py, or the hybrid albedo from midrib.py), and SPECIES the LeafOptics
library entry the classes are fitted relative to. Writes the class map OUTPUT.png and its class definitions OUTPUT.xml, which LeafOptics
reads.

The source's colours are clustered (k-means in linear RGB) into CLASS_COUNT classes, numbered from darkest to lightest. The fit then
uses only the colour differences across the leaf: a gain per channel puts the source's mean colour on the colour of the library leaf, as
an iPhone 12 Pro Max camera sees it under the ASTM G173 sun after a colour correction fitted to the ColorChecker. For each class, the
chlorophyll content (carotenoids in proportion) of the LeafOptics leaf whose hue best matches the class colour gives its chlorophyll_scale
(relative to the library leaf), and the remaining brightness difference its visible_reflectance_scale.

Surface mode (bark, fruit, or any surface that is not a leaf): TEXTURE is the texture itself (PNG or JPEG; the texels of a PNG's
transparent pixels are left out), SOURCE the image of the same size whose colours the classes are fitted to (the albedo from
decompose.py, or TEXTURE again for a texture without shading), and REFLECTIVITY_LABEL (with an optional TRANSMISSIVITY_LABEL after a
comma) names the spectra of the radiation plugin's spectral library that the surface is given. The source's colours are clustered as
above, and each class is defined by a reflectance_scale, the ratio of its luminance to the source's mean luminance in linear RGB: only
the brightness differences across the surface are used, and the surface keeps the mean colour of its spectrum. --scale-exponent G
raises each luminance ratio to the power G (default 1) before it is written, compressing the range of the classes: the brightness of a
photograph mixes reflectance with the surface's curvature and gloss, and scaling a coloured spectrum up by the full ratio desaturates
it (a red apple skin scaled by 2 reflects twice as much green and blue too), so a value such as 0.5 keeps the pattern while holding
the classes nearer the spectrum. Whatever the exponent, no class may reflect more than --max-reflectance R (default 0.6) at any
wavelength from 400 to 700 nm: the brightest parts of a photograph of a matte natural surface are its gloss and its lit curvature, not
a reflectance a diffuse surface can have, and a class scaled past this renders washed out. A class that would exceed it is held at it,
and the program says so. --reflectance-factor F multiplies every class by F (default 1), for a surface that is darker overall than the
spectrum it is fitted relative to (an apple skin fitted relative to a strawberry, say). Writes OUTPUT.png and OUTPUT.xml, which
RadiationModel::applySpectralClassMaps() reads.
*/

#include "LeafOptics.h"

using namespace helios;

namespace {

    //! A file of the radiation plugin's spectral library, read from the repository: this program does not build the radiation plugin, so its data is not copied to the build directory
    std::string spectralData(const std::string &file) {
        const std::filesystem::path path = std::filesystem::path(__FILE__).parent_path() / "../../../plugins/radiation/spectral_data" / file;
        if (!std::filesystem::exists(path)) {
            helios_runtime_error("ERROR (fit_classes): the spectral library file " + path.string() + " does not exist.");
        }
        return std::filesystem::canonical(path).string();
    }

    float srgbToLinear(float c) {
        return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
    }

    float linearToSrgb(float linear) {
        linear = std::clamp(linear, 0.f, 1.f);
        return linear <= 0.0031308f ? 12.92f * linear : 1.055f * std::pow(linear, 1.f / 2.4f) - 0.055f;
    }

    //! The sun's spectrum and the RGB camera's spectral responses
    struct CameraSpectra {
        std::vector<vec2> sun;
        std::vector<vec2> response[3];
    };

    CameraSpectra loadCameraSpectra(Context &context) {
        context.loadXML(spectralData("solar_spectrum_ASTMG173.xml").c_str(), true);
        context.loadXML(spectralData("camera_spectral_library.xml").c_str(), true);
        CameraSpectra spectra;
        context.getGlobalData("solar_spectrum_ASTMG173", spectra.sun);
        const char *channels[3] = {"iPhone12ProMAX_red", "iPhone12ProMAX_green", "iPhone12ProMAX_blue"};
        for (int c = 0; c < 3; c++) {
            context.getGlobalData(channels[c], spectra.response[c]);
        }
        return spectra;
    }

    //! Reflectance factor of a reflectance spectrum in each raw camera channel under the sun, over 400-700 nm
    std::array<float, 3> cameraReflectance(const CameraSpectra &spectra, const std::vector<vec2> &rho) {
        std::array<float, 3> rgb{};
        for (int c = 0; c < 3; c++) {
            double weighted = 0, weight = 0;
            for (int wavelength = 400; wavelength <= 700; wavelength += 2) {
                const double w = interp1(spectra.sun, float(wavelength)) * interp1(spectra.response[c], float(wavelength));
                weighted += w * interp1(rho, float(wavelength));
                weight += w;
            }
            rgb[c] = float(weighted / weight);
        }
        return rgb;
    }

    //! A 3 x 3 colour correction from raw camera reflectance factors to linear sRGB
    struct ColourCorrection {
        float M[3][3];
        std::array<float, 3> apply(const std::array<float, 3> &raw) const {
            return {M[0][0] * raw[0] + M[0][1] * raw[1] + M[0][2] * raw[2], M[1][0] * raw[0] + M[1][1] * raw[1] + M[1][2] * raw[2], M[2][0] * raw[0] + M[2][1] * raw[1] + M[2][2] * raw[2]};
        }
    };

    //! Least-squares fit to the 24 patches of the ColorChecker Classic: their spectra seen by the camera under the sun, against their published sRGB values
    ColourCorrection fitColourCorrection(Context &context, const CameraSpectra &spectra) {
        context.loadXML(spectralData("color_board/Calibrite_ColorChecker_Classic_colorboard.xml").c_str(), true);
        const int reference[24][3] = {{115, 82, 68}, {194, 150, 130}, {98, 122, 157}, {87, 108, 67},  {133, 128, 177}, {103, 189, 170}, {214, 126, 44},  {80, 91, 166},   {193, 90, 99},   {94, 60, 108},   {157, 188, 64}, {224, 163, 46},
                                      {56, 61, 150}, {70, 148, 73},   {175, 54, 60},  {231, 199, 31}, {187, 86, 149},  {8, 133, 161},   {243, 243, 242}, {200, 200, 200}, {160, 160, 160}, {122, 122, 121}, {85, 85, 85},   {52, 52, 52}};
        double AtA[3][3] = {}, Aty[3][3] = {};
        for (int k = 0; k < 24; k++) {
            char label[64];
            std::snprintf(label, sizeof(label), "ColorReference_Calibrite_%02d", k + 1);
            std::vector<vec2> rho;
            context.getGlobalData(label, rho);
            const std::array<float, 3> raw = cameraReflectance(spectra, rho);
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    AtA[i][j] += raw[i] * raw[j];
                    Aty[i][j] += raw[i] * srgbToLinear(reference[k][j] / 255.f);
                }
            }
        }
        auto det3 = [](double m[3][3]) { return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]); };
        const double D = det3(AtA);
        if (std::fabs(D) < 1e-12) {
            helios_runtime_error("ERROR (fit_classes): the colour checker's camera values are degenerate.");
        }
        ColourCorrection ccm;
        for (int out = 0; out < 3; out++) {
            for (int col = 0; col < 3; col++) {
                double m[3][3];
                for (int i = 0; i < 3; i++) {
                    for (int j = 0; j < 3; j++) {
                        m[i][j] = j == col ? Aty[i][out] : AtA[i][j];
                    }
                }
                ccm.M[out][col] = float(det3(m) / D);
            }
        }
        return ccm;
    }

} // namespace

//! Texels of an image clustered into classes from darkest to lightest
struct ClusteredTexture {
    uint width = 0, height = 0;
    std::vector<uint16_t> classes; // per texel, 0 for a transparent texel
    std::vector<std::array<float, 3>> centres; // linear RGB of each class, darkest first
    std::vector<size_t> class_texels;
    std::array<double, 3> mean{0, 0, 0}; // linear RGB over the opaque texels
    size_t opaque_texels = 0;
};

//! Linear RGB of the opaque texels of `source`, with the opacity of `texture` as the Context reads it (every texel of a texture without transparency)
static void loadTexels(Context &context, const std::string &texture, const std::string &source, uint &width, uint &height, std::vector<std::array<float, 3>> &texels, std::vector<size_t> &texel_index) {
    const uint patch = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, texture.c_str());
    const std::vector<std::vector<bool>> *opaque = context.primitiveTextureHasTransparencyChannel(patch) ? context.getPrimitiveTextureTransparencyData(patch) : nullptr;

    std::vector<RGBAcolor> image;
    const std::string extension = std::filesystem::path(source).extension().string();
    if (extension == ".png" || extension == ".PNG") {
        readPNG(source, width, height, image);
    } else {
        std::vector<RGBcolor> rgb;
        readJPEG(source, width, height, rgb);
        image.reserve(rgb.size());
        for (const RGBcolor &c: rgb) {
            image.push_back(make_RGBAcolor(c.r, c.g, c.b, 1.f));
        }
    }
    if (opaque != nullptr) {
        if (height != opaque->size() || width != opaque->front().size()) {
            helios_runtime_error("ERROR (fit_classes): " + source + " and " + texture + " differ in size.");
        }
        size_t mismatches = 0;
        for (uint y = 0; y < height; y++) {
            for (uint x = 0; x < width; x++) {
                mismatches += (image[y * width + x].a * 255.f >= 249.5f) != bool(opaque->at(y).at(x));
            }
        }
        if (mismatches > 0) {
            helios_runtime_error("ERROR (fit_classes): " + std::to_string(mismatches) + " texels differ in transparency between " + source + " and " + texture + ".");
        }
    } else {
        const int2 texture_size = context.getPrimitiveTextureSize(patch);
        if (int(width) != texture_size.x || int(height) != texture_size.y) {
            helios_runtime_error("ERROR (fit_classes): " + source + " and " + texture + " differ in size.");
        }
    }
    for (size_t i = 0; i < image.size(); i++) {
        if (opaque == nullptr || opaque->at(i / width).at(i % width)) {
            texels.push_back({srgbToLinear(image[i].r), srgbToLinear(image[i].g), srgbToLinear(image[i].b)});
            texel_index.push_back(i);
        }
    }
}

//! k-means (seeded with k-means++) of the texels into `class_count` classes, numbered from darkest to lightest
static ClusteredTexture clusterTexels(const std::vector<std::array<float, 3>> &texels, const std::vector<size_t> &texel_index, uint width, uint height, int class_count) {
    if (texels.size() < size_t(class_count)) {
        helios_runtime_error("ERROR (fit_classes): the texture has fewer opaque texels than classes.");
    }
    auto distance = [](const std::array<float, 3> &a, const std::array<float, 3> &b) { return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]); };

    std::mt19937 generator(3);
    std::vector<std::array<float, 3>> centres = {texels[generator() % texels.size()]};
    std::vector<float> nearest(texels.size(), 1e9f);
    while (int(centres.size()) < class_count) {
        double total = 0;
        for (size_t i = 0; i < texels.size(); i++) {
            nearest[i] = std::min(nearest[i], distance(texels[i], centres.back()));
            total += nearest[i];
        }
        double pick = std::uniform_real_distribution<double>(0, total)(generator);
        size_t chosen = 0;
        for (; chosen + 1 < texels.size() && pick > nearest[chosen]; chosen++) {
            pick -= nearest[chosen];
        }
        centres.push_back(texels[chosen]);
    }
    std::vector<int> assignment(texels.size(), 0);
    for (int iteration = 0; iteration < 25; iteration++) {
        std::vector<std::array<double, 3>> sums(class_count, {0, 0, 0});
        std::vector<size_t> counts(class_count, 0);
        for (size_t i = 0; i < texels.size(); i++) {
            int best = 0;
            for (int k = 1; k < class_count; k++) {
                if (distance(texels[i], centres[k]) < distance(texels[i], centres[best])) {
                    best = k;
                }
            }
            assignment[i] = best;
            counts[best]++;
            for (int c = 0; c < 3; c++) {
                sums[best][c] += texels[i][c];
            }
        }
        for (int k = 0; k < class_count; k++) {
            if (counts[k] == 0) {
                helios_runtime_error("ERROR (fit_classes): a colour cluster is empty; use fewer classes.");
            }
            for (int c = 0; c < 3; c++) {
                centres[k][c] = float(sums[k][c] / double(counts[k]));
            }
        }
    }

    // Classes from darkest to lightest
    std::vector<int> order(class_count);
    std::iota(order.begin(), order.end(), 0);
    auto luminance = [](const std::array<float, 3> &c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; };
    std::sort(order.begin(), order.end(), [&](int a, int b) { return luminance(centres[a]) < luminance(centres[b]); });
    std::vector<int> rank(class_count);
    for (int k = 0; k < class_count; k++) {
        rank[order[k]] = k;
    }
    ClusteredTexture clustered;
    clustered.width = width;
    clustered.height = height;
    clustered.classes.assign(size_t(width) * height, 0);
    clustered.class_texels.assign(class_count, 0);
    clustered.opaque_texels = texels.size();
    for (size_t i = 0; i < texels.size(); i++) {
        const int k = rank[assignment[i]];
        clustered.classes[texel_index[i]] = uint16_t(k + 1);
        clustered.class_texels[k]++;
        for (int c = 0; c < 3; c++) {
            clustered.mean[c] += texels[i][c] / double(texels.size());
        }
    }
    for (int k = 0; k < class_count; k++) {
        clustered.centres.push_back(centres[order[k]]);
    }
    return clustered;
}

static std::string srgbAttribute(const std::array<float, 3> &centre) {
    return std::to_string(int(std::round(255 * linearToSrgb(centre[0])))) + " " + std::to_string(int(std::round(255 * linearToSrgb(centre[1])))) + " " + std::to_string(int(std::round(255 * linearToSrgb(centre[2]))));
}

static void writeOutputs(const std::string &map_file, const ClusteredTexture &clustered, const std::string &xml_body) {
    writePNGSingleChannel(map_file, clustered.width, clustered.height, clustered.classes, clustered.centres.size() > 255 ? 16 : 8);
    const std::string definitions_file = std::filesystem::path(map_file).replace_extension(".xml").string();
    std::ofstream xml(definitions_file);
    xml << "<helios>\n" << xml_body << "</helios>\n";
    std::cout << "Wrote " << map_file << " and " << definitions_file << " (" << clustered.centres.size() << " classes over " << clustered.opaque_texels << " texels)." << std::endl;
}

//! Surface mode: each class is the surface's spectrum scaled by the ratio of the class's luminance to the texture's mean luminance
static int fitSurface(int argc, char **argv) {
    float scale_exponent = 1.f;
    float max_reflectance = 0.6f;
    float reflectance_factor = 1.f;
    int first = 2;
    while (argc >= first + 2 && std::string(argv[first]).rfind("--", 0) == 0) {
        const std::string option = argv[first];
        if (option == "--scale-exponent") {
            scale_exponent = std::stof(argv[first + 1]);
            if (!(scale_exponent > 0.f)) {
                helios_runtime_error("ERROR (fit_classes): --scale-exponent must be positive.");
            }
        } else if (option == "--max-reflectance") {
            max_reflectance = std::stof(argv[first + 1]);
            if (!(max_reflectance > 0.f) || max_reflectance > 1.f) {
                helios_runtime_error("ERROR (fit_classes): --max-reflectance must be between 0 and 1.");
            }
        } else if (option == "--reflectance-factor") {
            reflectance_factor = std::stof(argv[first + 1]);
            if (!(reflectance_factor > 0.f)) {
                helios_runtime_error("ERROR (fit_classes): --reflectance-factor must be positive.");
            }
        } else {
            helios_runtime_error("ERROR (fit_classes): unknown option " + option + ".");
        }
        first += 2;
    }
    if (argc != first + 5) {
        std::cerr << "Usage: fit_classes --surface [--scale-exponent G] [--max-reflectance R] [--reflectance-factor F] TEXTURE SOURCE REFLECTIVITY_LABEL[,TRANSMISSIVITY_LABEL] CLASS_COUNT OUTPUT.png" << std::endl;
        return 1;
    }
    const std::string texture = argv[first];
    const std::string source = argv[first + 1];
    const std::string labels = argv[first + 2];
    const int class_count = std::stoi(argv[first + 3]);
    const std::string map_file = argv[first + 4];
    if (class_count < 1 || class_count > 255) {
        helios_runtime_error("ERROR (fit_classes): CLASS_COUNT must be from 1 to 255.");
    }
    if (std::filesystem::path(map_file).extension() != ".png") {
        helios_runtime_error("ERROR (fit_classes): OUTPUT must be a .png file.");
    }
    const size_t comma = labels.find(',');
    const std::string reflectivity_label = labels.substr(0, comma);
    const std::string transmissivity_label = comma == std::string::npos ? "" : labels.substr(comma + 1);
    if (reflectivity_label.empty()) {
        helios_runtime_error("ERROR (fit_classes): a reflectivity spectrum label is required.");
    }

    Context context;
    uint width, height;
    std::vector<std::array<float, 3>> texels;
    std::vector<size_t> texel_index;
    loadTexels(context, texture, source, width, height, texels, texel_index);
    const ClusteredTexture clustered = clusterTexels(texels, texel_index, width, height, class_count);

    // The surface's spectrum, from the radiation plugin's spectral library, for the reflectance cap
    for (const char *library: {"bark_surface_spectral_library.xml", "fruit_surface_spectral_library.xml", "soil_surface_spectral_library.xml", "leaf_surface_spectral_library.xml"}) {
        if (!context.doesGlobalDataExist(reflectivity_label.c_str()) && Context::scanXMLForTag(spectralData(library), "globaldata_vec2", reflectivity_label)) {
            context.loadXML(spectralData(library).c_str(), true);
        }
    }
    if (!context.doesGlobalDataExist(reflectivity_label.c_str())) {
        helios_runtime_error("ERROR (fit_classes): the reflectivity spectrum '" + reflectivity_label + "' is not in the radiation plugin's bark, fruit, soil or leaf spectral library.");
    }
    std::vector<vec2> base_reflectivity;
    context.getGlobalData(reflectivity_label.c_str(), base_reflectivity);
    float peak_visible_reflectance = 0.f;
    for (int wavelength = 400; wavelength <= 700; wavelength += 2) {
        peak_visible_reflectance = std::fmax(peak_visible_reflectance, interp1(base_reflectivity, float(wavelength)));
    }
    if (!(peak_visible_reflectance > 0.f)) {
        helios_runtime_error("ERROR (fit_classes): the spectrum '" + reflectivity_label + "' has no reflectance between 400 and 700 nm.");
    }
    const float scale_cap = max_reflectance / peak_visible_reflectance;

    auto luminance = [](const std::array<float, 3> &c) { return 0.2126f * c[0] + 0.7152f * c[1] + 0.0722f * c[2]; };
    const float mean_luminance = luminance({float(clustered.mean[0]), float(clustered.mean[1]), float(clustered.mean[2])});
    std::ostringstream body;
    body << "  <!-- Class definitions of " << std::filesystem::path(map_file).filename().string() << ", relative to the spectra named below from the radiation plugin's spectral library (see the radiation plugin documentation, \"Spectral Textures\").\n"
         << "       reflectance_scale multiplies the surface's reflectivity (and transmissivity) spectrum: the ratio of the class's luminance to the source's mean luminance"
         << (scale_exponent == 1.f ? std::string(".\n") : ", raised to the power " + std::to_string(scale_exponent).substr(0, 4) + " (--scale-exponent).\n")
         << (reflectance_factor == 1.f ? std::string("") : "       Every class is multiplied by " + std::to_string(reflectance_factor).substr(0, 4) + " (--reflectance-factor), the surface being darker than the spectrum it is fitted relative to.\n")
         << "       No class reflects more than " << max_reflectance << " at any wavelength from 400 to 700 nm (--max-reflectance).\n"
         << "       Fitted by utilities/leaf_class_maps/fit_classes --surface to " << std::filesystem::path(source).filename().string() << ". -->\n"
         << "  <surface_class_map reflectivity_spectrum=\"" << reflectivity_label << "\" transmissivity_spectrum=\"" << transmissivity_label << "\" provenance=\"" << std::filesystem::path(texture).filename().string()
         << " via utilities/leaf_class_maps/fit_classes --surface (fitted to " << std::filesystem::path(source).filename().string() << ", relative to " << reflectivity_label << ")\">\n";
    for (int k = 0; k < class_count; k++) {
        char line[256];
        float scale = reflectance_factor * std::pow(luminance(clustered.centres[k]) / mean_luminance, scale_exponent);
        if (scale > scale_cap) {
            std::cout << "Class " << k + 1 << " would reflect " << scale * peak_visible_reflectance << " at the peak of " << reflectivity_label << " in the visible; its scale is held at " << scale_cap << " (from " << scale
                      << ") so that it reflects at most " << max_reflectance << "." << std::endl;
            scale = scale_cap;
        }
        std::snprintf(line, sizeof(line), "    <class id=\"%d\" reflectance_scale=\"%.4g\" texels=\"%zu\" source_srgb=\"%s\"/>\n", k + 1, scale, clustered.class_texels[k], srgbAttribute(clustered.centres[k]).c_str());
        body << line;
    }
    body << "  </surface_class_map>\n";
    writeOutputs(map_file, clustered, body.str());
    return 0;
}

int main(int argc, char **argv) {
    if (argc >= 2 && std::string(argv[1]) == "--surface") {
        return fitSurface(argc, argv);
    }
    if (argc != 6) {
        std::cerr << "Usage: fit_classes TEXTURE.png SOURCE.png SPECIES CLASS_COUNT OUTPUT.png\n       fit_classes --surface [--scale-exponent G] TEXTURE SOURCE REFLECTIVITY_LABEL[,TRANSMISSIVITY_LABEL] CLASS_COUNT OUTPUT.png" << std::endl;
        return 1;
    }
    const std::string texture = argv[1];
    const std::string source = argv[2];
    const std::string species = argv[3];
    const int class_count = std::stoi(argv[4]);
    const std::string map_file = argv[5];
    if (class_count < 1 || class_count > 255) {
        helios_runtime_error("ERROR (fit_classes): CLASS_COUNT must be from 1 to 255.");
    }
    if (std::filesystem::path(map_file).extension() != ".png") {
        helios_runtime_error("ERROR (fit_classes): OUTPUT must be a .png file.");
    }

    Context context;
    LeafOptics leafoptics(&context);
    leafoptics.disableMessages();
    LeafOpticsProperties base;
    leafoptics.getPropertiesFromLibrary(species, base);
    const CameraSpectra spectra = loadCameraSpectra(context);
    const ColourCorrection ccm = fitColourCorrection(context, spectra);

    uint width, height;
    std::vector<std::array<float, 3>> texels;
    std::vector<size_t> texel_index;
    loadTexels(context, texture, source, width, height, texels, texel_index);
    const ClusteredTexture clustered = clusterTexels(texels, texel_index, width, height, class_count);

    // LeafOptics leaves of the library leaf's structure with chlorophyll from 4 to 100 ug/cm^2 (carotenoids in proportion), as the camera sees them
    std::vector<float> grid_chlorophyll;
    std::vector<std::array<float, 3>> grid_rgb;
    for (int k = 0; k < 49; k++) {
        LeafOpticsProperties properties = base;
        properties.chlorophyllcontent = 4.f + 2.f * float(k);
        properties.carotenoidcontent = base.carotenoidcontent * properties.chlorophyllcontent / base.chlorophyllcontent;
        std::vector<vec2> rho, tau;
        leafoptics.getLeafSpectra(properties, rho, tau);
        grid_chlorophyll.push_back(properties.chlorophyllcontent);
        grid_rgb.push_back(ccm.apply(cameraReflectance(spectra, rho)));
    }
    std::vector<vec2> base_rho, base_tau;
    leafoptics.getLeafSpectra(base, base_rho, base_tau);
    const std::array<float, 3> anchor = ccm.apply(cameraReflectance(spectra, base_rho));
    std::array<float, 3> gains;
    for (int c = 0; c < 3; c++) {
        gains[c] = anchor[c] / float(clustered.mean[c]);
    }

    // For each class, the grid leaf whose hue (log colour less its mean over the channels) best matches, and the brightness factor that remains
    std::ostringstream body;
    body << "  <!-- Class definitions of " << std::filesystem::path(map_file).filename().string() << ", relative to the leaf LeafOptics is given (see the LeafOptics documentation, \"Leaves with a class map\").\n"
         << "       chlorophyll_scale multiplies its chlorophyll and carotenoid contents, and visible_reflectance_scale its reflectance below 700 nm (blending to 1 at 750 nm).\n"
         << "       Fitted by utilities/leaf_class_maps/fit_classes to " << std::filesystem::path(source).filename().string() << " relative to the LeafOptics library leaf '" << species << "'. -->\n"
         << "  <leaf_class_map provenance=\"" << std::filesystem::path(texture).filename().string() << " via utilities/leaf_class_maps (fitted to " << std::filesystem::path(source).filename().string() << ", relative to LeafOptics '" << species
         << "')\">\n";
    for (int k = 0; k < class_count; k++) {
        const std::array<float, 3> &centre = clustered.centres[k];
        size_t best = 0;
        double best_error = 1e30;
        float brightness = 1.f;
        for (size_t g = 0; g < grid_rgb.size(); g++) {
            double d[3], offset = 0;
            for (int c = 0; c < 3; c++) {
                d[c] = std::log(std::max(gains[c] * centre[c], 1e-5f)) - std::log(std::max(grid_rgb[g][c], 1e-5f));
                offset += d[c] / 3.0;
            }
            const double error = (d[0] - offset) * (d[0] - offset) + (d[1] - offset) * (d[1] - offset) + (d[2] - offset) * (d[2] - offset);
            if (error < best_error) {
                best_error = error;
                best = g;
                brightness = float(std::exp(offset));
            }
        }
        char line[256];
        std::snprintf(line, sizeof(line), "    <class id=\"%d\" chlorophyll_scale=\"%.4g\" visible_reflectance_scale=\"%.4g\" texels=\"%zu\" source_srgb=\"%s\"/>\n", k + 1, grid_chlorophyll[best] / base.chlorophyllcontent, brightness,
                      clustered.class_texels[k], srgbAttribute(centre).c_str());
        body << line;
    }
    body << "  </leaf_class_map>\n";
    writeOutputs(map_file, clustered, body.str());
    return 0;
}
