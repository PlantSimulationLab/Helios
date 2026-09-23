/** \file "LeafOptics.cpp" Implementation of PROSPECT-PRO leaf optical model.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#include "LeafOptics.h"
using namespace std;
using namespace helios;

// ----------------------------------------------------------------------------
// Numerical building blocks of the PROSPECT plate model
// ----------------------------------------------------------------------------
//
// References:
//   Jacquemoud, S. & Baret, F. (1990) PROSPECT: A model of leaf optical properties spectra.
//     Remote Sensing of Environment 34:75-91. ("J&B 1990" below)
//   Feret, J.-B. et al. (2021) PROSPECT-PRO for estimating content of nitrogen-containing leaf
//     proteins and other carbon-based constituents (preprint arXiv:2003.11961).
//   NIST Digital Library of Mathematical Functions (DLMF), chapter 6 (exponential integral).

namespace {

    //! Exponential integral E1(x) for x > 0.
    /**
     * For x <= 1 the convergent power series of DLMF 6.6.2 is used,
     *   E1(x) = -gamma - ln(x) - sum_{k>=1} (-x)^k / (k k!).
     * For x > 1 the continued fraction of DLMF 6.9.1 (in its even-contracted form) is evaluated with the modified
     * Lentz algorithm, which yields the scaled value exp(x) E1(x) directly:
     *   exp(x) E1(x) = 1 / (x + 1 - 1^2 / (x + 3 - 2^2 / (x + 5 - 3^2 / (x + 7 - ...)))).
     * Returning the scaled value for large x lets callers combine it with exp(-x) without overflow or underflow.
     *
     * \param[in] x Argument (must be > 0).
     * \param[out] scaled_by_exp True if the returned value is exp(x) E1(x) (x > 1), false if it is E1(x) (x <= 1).
     * \return E1(x) or exp(x) E1(x), as indicated by scaled_by_exp. Relative accuracy is ~1e-15.
     */
    double exponentialIntegralE1(double x, bool &scaled_by_exp) {
        constexpr double euler_gamma = 0.57721566490153286060651209;
        if (x <= 1.0) {
            scaled_by_exp = false;
            double series_sum = 0.0;
            double power_over_factorial = 1.0; // (-x)^k / k!
            for (int k = 1; k < 100; k++) {
                power_over_factorial *= -x / static_cast<double>(k);
                const double term = power_over_factorial / static_cast<double>(k);
                series_sum += term;
                if (std::abs(term) < 1e-17 * std::abs(series_sum)) {
                    break;
                }
            }
            return -euler_gamma - std::log(x) - series_sum;
        }

        scaled_by_exp = true;
        constexpr double tiny = 1e-300;
        double denominator_term = x + 1.0;
        double lentz_c = 1.0 / tiny;
        double lentz_d = 1.0 / denominator_term;
        double fraction = lentz_d;
        for (int i = 1; i < 10000; i++) {
            const double numerator_term = -static_cast<double>(i) * static_cast<double>(i);
            denominator_term += 2.0;
            lentz_d = 1.0 / (numerator_term * lentz_d + denominator_term);
            lentz_c = denominator_term + numerator_term / lentz_c;
            const double delta = lentz_c * lentz_d;
            fraction *= delta;
            if (std::abs(delta - 1.0) < 1e-16) {
                return fraction;
            }
        }
        helios_runtime_error("ERROR (LeafOptics): continued fraction for the exponential integral E1(" + std::to_string(x) + ") failed to converge.");
        return 0.0;
    }

    //! Transmission coefficient theta(k) of a compact elementary layer for isotropic (diffuse) light.
    /**
     * J&B 1990 Eq. (11): theta = (1 - k) exp(-k) + k^2 E1(k), where k is the absorption coefficient of the layer.
     * theta(0) = 1 (no absorption). For large k, both terms are formed from exp(-k) times the scaled continued
     * fraction so the result underflows smoothly to zero instead of producing NaN.
     */
    double elementaryLayerTransmission(double k) {
        if (k <= 0.0) {
            return 1.0;
        }
        bool scaled_by_exp = false;
        const double e1_value = exponentialIntegralE1(k, scaled_by_exp);
        if (!scaled_by_exp) {
            return (1.0 - k) * std::exp(-k) + k * k * e1_value;
        }
        return std::exp(-k) * ((1.0 - k) + k * k * e1_value);
    }

    //! Gauss-Legendre quadrature rule on [-1, 1].
    struct GaussLegendreRule {
        std::vector<double> nodes;
        std::vector<double> weights;
    };

    //! Compute Gauss-Legendre nodes and weights on [-1, 1] by Newton iteration on the Legendre three-term recurrence.
    GaussLegendreRule makeGaussLegendreRule(int order) {
        GaussLegendreRule rule;
        rule.nodes.resize(order);
        rule.weights.resize(order);
        for (int i = 0; i < order; i++) {
            // Initial guess for the i-th root, refined by Newton's method.
            double x = std::cos(M_PI * (static_cast<double>(i) + 0.75) / (static_cast<double>(order) + 0.5));
            double derivative = 1.0;
            for (int iteration = 0; iteration < 100; iteration++) {
                double p_current = 1.0;
                double p_previous = 0.0;
                for (int degree = 1; degree <= order; degree++) {
                    const double p_older = p_previous;
                    p_previous = p_current;
                    p_current = ((2.0 * degree - 1.0) * x * p_previous - (degree - 1.0) * p_older) / static_cast<double>(degree);
                }
                derivative = static_cast<double>(order) * (x * p_current - p_previous) / (x * x - 1.0);
                const double step = p_current / derivative;
                x -= step;
                if (std::abs(step) < 1e-15) {
                    break;
                }
            }
            rule.nodes.at(i) = x;
            rule.weights.at(i) = 2.0 / ((1.0 - x * x) * derivative * derivative);
        }
        return rule;
    }

    //! Fresnel transmissivity of a planar dielectric interface for unpolarized light incident at angle theta from air.
    double fresnelTransmissivity(double incidence_angle, double refractive_index) {
        const double cos_incidence = std::cos(incidence_angle);
        const double sin_incidence = std::sin(incidence_angle);
        const double n_squared = refractive_index * refractive_index;
        // n cos(theta_t), from Snell's law sin(theta_t) = sin(theta_i) / n
        const double n_cos_transmitted = std::sqrt(n_squared - sin_incidence * sin_incidence);
        const double r_perpendicular = (cos_incidence - n_cos_transmitted) / (cos_incidence + n_cos_transmitted);
        const double r_parallel = (n_squared * cos_incidence - n_cos_transmitted) / (n_squared * cos_incidence + n_cos_transmitted);
        return 1.0 - 0.5 * (r_perpendicular * r_perpendicular + r_parallel * r_parallel);
    }

    //! Average interface transmissivity t_av(alpha, n) (J&B 1990, Fig. 1 and Eqs. 1-2).
    /**
     * Fresnel transmissivity for unpolarized, isotropic light averaged over all incidence directions within a cone of
     * half-angle alpha:
     *   t_av(alpha, n) = [ integral_0^alpha t(theta) sin(theta) cos(theta) dtheta ] / [ sin^2(alpha) / 2 ].
     * The integrand is smooth in theta for n >= 1, so a 32-point Gauss-Legendre rule gives machine precision.
     */
    double averageInterfaceTransmissivity(double alpha_degrees, double refractive_index) {
        static const GaussLegendreRule rule = makeGaussLegendreRule(32);

        if (refractive_index < 1.0) {
            helios_runtime_error("ERROR (LeafOptics): refractive index " + std::to_string(refractive_index) + " is below 1; the interface transmissivity is only defined for light entering an optically denser medium.");
        }
        const double alpha = alpha_degrees * M_PI / 180.0;
        const double half_alpha = 0.5 * alpha;
        double integral = 0.0;
        for (size_t i = 0; i < rule.nodes.size(); i++) {
            const double theta = half_alpha * (rule.nodes[i] + 1.0);
            integral += rule.weights[i] * half_alpha * fresnelTransmissivity(theta, refractive_index) * std::sin(theta) * std::cos(theta);
        }
        const double sin_alpha = std::sin(alpha);
        return 2.0 * integral / (sin_alpha * sin_alpha);
    }

    //! Reflectance and transmittance of a stack of identical diffusing layers (Stokes 1862; J&B 1990 Eq. 9).
    /**
     * \param[in] layer_reflectance Reflectance rho of one layer for diffuse light.
     * \param[in] layer_transmittance Transmittance tau of one layer for diffuse light.
     * \param[in] number_of_layers Number of layers m (may be non-integer, m >= 0).
     * \param[out] stack_reflectance Reflectance of the stack.
     * \param[out] stack_transmittance Transmittance of the stack.
     */
    void stokesLayerStack(double layer_reflectance, double layer_transmittance, double number_of_layers, double &stack_reflectance, double &stack_transmittance) {
        if (number_of_layers <= 0.0) {
            // An empty stack neither reflects nor attenuates.
            stack_reflectance = 0.0;
            stack_transmittance = 1.0;
            return;
        }
        const double layer_absorptance = 1.0 - layer_reflectance - layer_transmittance;
        if (layer_absorptance <= 1e-10) {
            // Non-absorbing limit of Eq. 9 (a, b -> 1): resistances add, so T_m = tau / (tau + m rho) and R_m = 1 - T_m.
            stack_transmittance = layer_transmittance / (layer_transmittance + number_of_layers * layer_reflectance);
            stack_reflectance = 1.0 - stack_transmittance;
            return;
        }
        const double rho2 = layer_reflectance * layer_reflectance;
        const double tau2 = layer_transmittance * layer_transmittance;
        // delta^2 = (tau^2 - rho^2 - 1)^2 - 4 rho^2, written in factored form for accuracy.
        const double delta = std::sqrt((1.0 + layer_reflectance + layer_transmittance) * (1.0 + layer_reflectance - layer_transmittance) * (1.0 - layer_reflectance + layer_transmittance) * layer_absorptance);
        const double a = (1.0 + rho2 - tau2 + delta) / (2.0 * layer_reflectance);
        const double b = (1.0 - rho2 + tau2 + delta) / (2.0 * layer_transmittance);
        // Eq. 9 with numerator and denominator divided by b^m, so that strongly absorbing layers (b >> 1, possibly
        // infinite when tau underflows to zero) give b^-m -> 0 instead of an overflow.
        const double b_inverse_power = std::exp(-number_of_layers * std::log(b));
        const double b_inverse_power2 = b_inverse_power * b_inverse_power;
        const double denominator = a - b_inverse_power2 / a;
        stack_reflectance = (1.0 - b_inverse_power2) / denominator;
        stack_transmittance = (a - 1.0 / a) * b_inverse_power / denominator;
    }

} // namespace

LeafOptics::LeafOptics(helios::Context *a_context) {

    context = a_context; // just copying the pointer to the context

    // Load leaf refraction index and specific absorption coefficients  (400,401,...2500 nm)
    context->loadXML("plugins/leafoptics/spectral_data/prospect_spectral_library.xml", true);

    // Load leaf refraction index -  refractiveindex (n)
    std::vector<helios::vec2> data;
    if (!context->doesGlobalDataExist("refraction_index")) {
        helios_runtime_error("Refraction index data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("refraction_index", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of refraction index data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    refractiveindex.resize(nw);
    wave_length.resize(nw);
    for (int i = 0; i < nw; i++) {
        refractiveindex.at(i) = data.at(i).y;
        wave_length.at(i) = data.at(i).x;
    }
    // Load specific absorption coefficient (per elementary layer depth)  for total chlorophyll  -  absorption_chlorophyll (cm^2/micro_g)
    if (!context->doesGlobalDataExist("absorption_chlorophyll")) {
        helios_runtime_error("Chlorophyll absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_chlorophyll", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of chlorophyll absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_chlorophyll.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_chlorophyll.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for total carotenoids  -  absorption_carotenoid (cm^2/micro_g)
    if (!context->doesGlobalDataExist("absorption_carotenoid")) {
        helios_runtime_error("Carotenoid absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_carotenoid", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of carotenoid absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_carotenoid.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_carotenoid.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for anthocyanins  -  sac_an (cm^2/micro_g)
    if (!context->doesGlobalDataExist("absorption_anthocyanin")) {
        helios_runtime_error("Anothocyanin absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_anthocyanin", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of anthocyanin absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_anthocyanin.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_anthocyanin.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for specific brown pigments (phenols during leaf death)   -  absorption_brown (arbitrary unit)
    if (!context->doesGlobalDataExist("absorption_brown")) {
        helios_runtime_error("Brown pigment absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_brown", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of brown pigment absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_brown.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_brown.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for mass of water per leaf area (EWT)-  absorption_water (1/cm or cm^2/g)
    if (!context->doesGlobalDataExist("absorption_water")) {
        helios_runtime_error("Water absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_water", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of water absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_water.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_water.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for dry mass per leaf area (LMA)-  absorption_drymass (cm^2/g)
    if (!context->doesGlobalDataExist("absorption_drymass")) {
        helios_runtime_error("Dry mass absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_drymass", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of dry mass absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_drymass.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_drymass.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for proteins- absorption_proteins (cm2.g-1)
    if (!context->doesGlobalDataExist("absorption_proteins")) {
        helios_runtime_error("Protein absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_proteins", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of protein absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_protein.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_protein.at(i) = data.at(i).y;
    }

    // Load specific absorption coefficient for carbon based constituents-  absorption_carbonconstituents (cm^2/g)
    if (!context->doesGlobalDataExist("absorption_carbonconstituents")) {
        helios_runtime_error("Carbon constituent absorption spectral data was not loaded properly from the prospect_spectral_library.xml file.");
    }
    context->getGlobalData("absorption_carbonconstituents", data);
    if (data.size() != nw) {
        helios_runtime_error("Size of carbon constituent absorption spectral data loaded from the prospect_spectral_library.xml file was not correct.");
    }
    absorption_carbonconstituents.resize(nw);
    for (int i = 0; i < nw; i++) {
        absorption_carbonconstituents.at(i) = data.at(i).y;
    }

    LeafOptics::surface(40.0, R_spec_normal);
    // get surface (i.e. fresnel)  reflectances for usage within Prospect function, relation from Stern
    // 0..40° degrees range from the vertical = normal incidence on an average rough leaf

    LeafOptics::surface(90.0, R_spec_diffuse);
    // 0..90° degrees range from the vertical = diffuse incidence on a perfectly smooth leaf

    // Initialize the species library with PROSPECT-D parameters from LOPEX93 dataset
    initializeSpeciesLibrary();
}

void LeafOptics::initializeSpeciesLibrary() {
    // PROSPECT-D parameters fitted to LOPEX93 spectral library samples
    // All species use PROSPECT-D mode (drymass > 0, protein = 0, carbonconstituents = 0)
    // Fitted using fit_prospect_visrobust.py with --no-calib flag

    LeafOpticsProperties props;

    // Default species (original Helios default values)
    props.numberlayers = 1.5f;
    props.chlorophyllcontent = 30.0f;
    props.carotenoidcontent = 7.0f;
    props.anthocyancontent = 1.0f;
    props.brownpigments = 0.0f;
    props.watermass = 0.015f;
    props.drymass = 0.09f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["default"] = props;

    // Garden lettuce (Lactuca sativa L.) - LOPEX93 sample 0021
    // RMSE: 0.008355
    props.numberlayers = 2.00517f;
    props.chlorophyllcontent = 30.2697f;
    props.carotenoidcontent = 6.9869f;
    props.anthocyancontent = 1.35975f;
    props.brownpigments = 0.107067f;
    props.watermass = 0.0281985f;
    props.drymass = 0.0052668f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["garden_lettuce"] = props;

    // Alfalfa (Medicago sativa L.) - LOPEX93 sample 0036
    // RMSE: 0.007743
    props.numberlayers = 2.00758f;
    props.chlorophyllcontent = 43.6375f;
    props.carotenoidcontent = 10.3145f;
    props.anthocyancontent = 1.33894f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0189936f;
    props.drymass = 0.00473702f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["alfalfa"] = props;

    // Corn (Zea mays L.) - LOPEX93 sample 0041
    // RMSE: 0.015966
    props.numberlayers = 1.59203f;
    props.chlorophyllcontent = 22.8664f;
    props.carotenoidcontent = 3.9745f;
    props.anthocyancontent = 0.0f;
    props.brownpigments = 0.72677f;
    props.watermass = 0.0149645f;
    props.drymass = 0.00441283f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["corn"] = props;

    // Sunflower (Helianthus annuus L.) - LOPEX93 sample 0081
    // RMSE: 0.007353
    props.numberlayers = 1.76358f;
    props.chlorophyllcontent = 54.0514f;
    props.carotenoidcontent = 12.9027f;
    props.anthocyancontent = 1.75194f;
    props.brownpigments = 0.0112026f;
    props.watermass = 0.0185557f;
    props.drymass = 0.00644855f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["sunflower"] = props;

    // English walnut (Juglans regia L.) - LOPEX93 sample 0091
    // RMSE: 0.007828
    props.numberlayers = 1.56274f;
    props.chlorophyllcontent = 55.9211f;
    props.carotenoidcontent = 12.4596f;
    props.anthocyancontent = 1.73981f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0127743f;
    props.drymass = 0.00583351f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["english_walnut"] = props;

    // Rice (Oryza sativa L.) - LOPEX93 sample 0106
    // RMSE: 0.004041
    props.numberlayers = 1.67081f;
    props.chlorophyllcontent = 37.233f;
    props.carotenoidcontent = 9.98756f;
    props.anthocyancontent = 0.0f;
    props.brownpigments = 0.0275106f;
    props.watermass = 0.0100962f;
    props.drymass = 0.00484587f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["rice"] = props;

    // Soybean (Glycine max L.) - LOPEX93 sample 0116
    // RMSE: 0.005875
    props.numberlayers = 1.5375f;
    props.chlorophyllcontent = 46.4121f;
    props.carotenoidcontent = 12.1394f;
    props.anthocyancontent = 0.648353f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0101049f;
    props.drymass = 0.00292814f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["soybean"] = props;

    // Wine grape (Vitis vinifera L.) - LOPEX93 sample 0276
    // RMSE: 0.005585
    props.numberlayers = 1.42673f;
    props.chlorophyllcontent = 50.918f;
    props.carotenoidcontent = 12.5466f;
    props.anthocyancontent = 1.43905f;
    props.brownpigments = 0.0798702f;
    props.watermass = 0.010922f;
    props.drymass = 0.00599315f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["wine_grape"] = props;

    // Tomato (Lycopersicum esculentum) - LOPEX93 sample 0316
    // RMSE: 0.005524
    props.numberlayers = 1.40304f;
    props.chlorophyllcontent = 48.3467f;
    props.carotenoidcontent = 11.604f;
    props.anthocyancontent = 1.45113f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0155627f;
    props.drymass = 0.00261571f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["tomato"] = props;

    // Common bean (Phaseolus vulgaris L.) - GEMINI field experiments, day 35
    // RMSE: 0.009479
    props.numberlayers = 1.44041f;
    props.chlorophyllcontent = 42.3619f;
    props.carotenoidcontent = 15.6263f;
    props.anthocyancontent = 0.844536f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0150048f;
    props.drymass = 0.00196285f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["common_bean"] = props;

    // Cowpea (Vigna unguiculata L.) - GEMINI field experiments, day 48
    // RMSE: 0.010680
    props.numberlayers = 1.22669f;
    props.chlorophyllcontent = 61.5204f;
    props.carotenoidcontent = 25.6171f;
    props.anthocyancontent = 2.51899f;
    props.brownpigments = 0.0f;
    props.watermass = 0.0221158f;
    props.drymass = 0.00108066f;
    props.protein = 0.0f;
    props.carbonconstituents = 0.0f;
    species_library["cowpea"] = props;
}

void LeafOptics::run(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties, const std::string &label) {
    std::vector<vec2> reflectivities_fit;
    std::vector<vec2> transmissivities_fit;
    getLeafSpectra(leafproperties, reflectivities_fit, transmissivities_fit);

    std::string leaf_reflectivity_label = "leaf_reflectivity_" + label;
    std::string leaf_transmissivity_label = "leaf_transmissivity_" + label;
    context->setGlobalData(leaf_reflectivity_label.c_str(), reflectivities_fit);
    context->setGlobalData(leaf_transmissivity_label.c_str(), transmissivities_fit);

    context->setPrimitiveData(UUIDs, "reflectivity_spectrum", leaf_reflectivity_label);
    context->setPrimitiveData(UUIDs, "transmissivity_spectrum", leaf_transmissivity_label);
    setProperties(UUIDs, leafproperties);

    // Write Fluspect-B biochemistry parameters to global data under the same label
    // so the radiation plugin's SIF pipeline can look them up per primitive via the
    // "fluspect_spectrum" primitive-data key below. Field order is fixed by convention
    // and documented in plugins/radiation/include/FluspectB.h. This write is harmless
    // when SIF is not in use — it just sits as unused global data.
    //
    // Field order: Cab, Cca, Cw, Cdm, Cs, Cant, Cp, Cbc, N, V2Z, fqe (11 fields).
    std::string fluspect_biochem_label = "fluspect_biochem_" + label;
    std::vector<float> fluspect_biochem = {
        leafproperties.chlorophyllcontent,  // Cab
        leafproperties.carotenoidcontent,   // Cca
        leafproperties.watermass,           // Cw
        leafproperties.drymass,             // Cdm
        leafproperties.brownpigments,       // Cs
        leafproperties.anthocyancontent,    // Cant
        leafproperties.protein,             // Cp
        leafproperties.carbonconstituents,  // Cbc
        leafproperties.numberlayers,        // N
        leafproperties.V2Z,
        leafproperties.fqe,
    };
    context->setGlobalData(fluspect_biochem_label.c_str(), fluspect_biochem);
    context->setPrimitiveData(UUIDs, "fluspect_spectrum", fluspect_biochem_label);

    // Store parameters in map for later retrieval
    spectrum_parameters_map[label] = leafproperties;
}

void LeafOptics::run(const LeafOpticsProperties &leafproperties, const std::string &label) {
    std::vector<vec2> reflectivities_fit;
    std::vector<vec2> transmissivities_fit;
    getLeafSpectra(leafproperties, reflectivities_fit, transmissivities_fit);

    std::string leaf_reflectivity_label = "leaf_reflectivity_" + label;
    std::string leaf_transmissivity_label = "leaf_transmissivity_" + label;
    context->setGlobalData(leaf_reflectivity_label.c_str(), reflectivities_fit);
    context->setGlobalData(leaf_transmissivity_label.c_str(), transmissivities_fit);

    // Fluspect-B biochemistry parameters, written to global data under a parallel label
    // so the radiation plugin's SIF pipeline can look them up. See the UUID-specific
    // overload of run() for the field-order convention.
    std::string fluspect_biochem_label = "fluspect_biochem_" + label;
    std::vector<float> fluspect_biochem = {
        leafproperties.chlorophyllcontent,  leafproperties.carotenoidcontent, leafproperties.watermass,
        leafproperties.drymass,             leafproperties.brownpigments,     leafproperties.anthocyancontent,
        leafproperties.protein,             leafproperties.carbonconstituents, leafproperties.numberlayers,
        leafproperties.V2Z,                 leafproperties.fqe,
    };
    context->setGlobalData(fluspect_biochem_label.c_str(), fluspect_biochem);

    // Store parameters in map for later retrieval
    spectrum_parameters_map[label] = leafproperties;
}


void LeafOptics::PROSPECT(float numberlayers, float Chlorophyllcontent, float carotenoidcontent, float anthocyancontent, float brownpigments, float watermass, float drymass, float protein, float carbonconstituents,
                          std::vector<float> &reflectivities_fit, std::vector<float> &transmissivities_fit) {

    // PROSPECT() is a spectrum-computing entry point in its own right, so it must apply the
    // same input validation as getLeafSpectra(). Validating through a LeafOpticsProperties
    // struct keeps the rules in one place rather than duplicating them here.
    LeafOpticsProperties validation_properties;
    validation_properties.numberlayers = numberlayers;
    validation_properties.chlorophyllcontent = Chlorophyllcontent;
    validation_properties.carotenoidcontent = carotenoidcontent;
    validation_properties.anthocyancontent = anthocyancontent;
    validation_properties.brownpigments = brownpigments;
    validation_properties.watermass = watermass;
    validation_properties.drymass = drymass;
    validation_properties.protein = protein;
    validation_properties.carbonconstituents = carbonconstituents;
    validateProperties(validation_properties);

    // These are output arguments - overwrite rather than append so that repeated
    // calls with the same vectors do not accumulate results.
    reflectivities_fit.clear();
    transmissivities_fit.clear();
    reflectivities_fit.reserve(nw);
    transmissivities_fit.reserve(nw);

    for (uint iwave = 0; iwave < nw; iwave++) {
        // Absorption coefficient of one elementary layer: total absorption of all constituents shared among the N
        // layers (PROSPECT-PRO Eq. 3, k = sum_i K_i C_i / N). In PROSPECT-PRO mode the dry mass has already been
        // replaced by its protein and carbon-based-constituent fractions (PROSPECT-PRO Eq. 5).
        const double total_absorption = static_cast<double>(Chlorophyllcontent) * absorption_chlorophyll.at(iwave) + static_cast<double>(carotenoidcontent) * absorption_carotenoid.at(iwave) +
                                        static_cast<double>(anthocyancontent) * absorption_anthocyanin.at(iwave) + static_cast<double>(brownpigments) * absorption_brown.at(iwave) + static_cast<double>(watermass) * absorption_water.at(iwave) +
                                        static_cast<double>(drymass) * absorption_drymass.at(iwave) + static_cast<double>(protein) * absorption_protein.at(iwave) + static_cast<double>(carbonconstituents) * absorption_carbonconstituents.at(iwave);
        const double layer_absorption = total_absorption / static_cast<double>(numberlayers);
        const double theta = elementaryLayerTransmission(layer_absorption); // J&B 1990 Eq. 11

        const double n = refractiveindex.at(iwave);
        const double n2 = n * n;
        const double t_alpha = 1.0 - static_cast<double>(R_spec_normal.at(iwave)); // t_av(40, n): incidence on a rough leaf
        const double t_90 = 1.0 - static_cast<double>(R_spec_diffuse.at(iwave)); // t_av(90, n): isotropic incidence

        // Plate model for a single compact layer (J&B 1990 Eqs. 1-2). The inner face of each interface has
        // transmissivity t21 = t_av(90, n) / n^2 and reflectivity r21 = 1 - t21 (diffuse light leaving the plate).
        const double theta2 = theta * theta;
        const double plate_denominator = n2 * n2 - theta2 * (n2 - t_90) * (n2 - t_90);
        const double rho_alpha = (1.0 - t_alpha) + t_90 * t_alpha * theta2 * (n2 - t_90) / plate_denominator;
        const double tau_alpha = t_90 * t_alpha * theta * n2 / plate_denominator;
        const double rho_90 = (1.0 - t_90) + t_90 * t_90 * theta2 * (n2 - t_90) / plate_denominator;
        const double tau_90 = t_90 * t_90 * theta * n2 / plate_denominator;

        // The remaining N-1 layers are stacked with the Stokes solution (J&B 1990 Eq. 9) ...
        double stack_reflectance = 0.0;
        double stack_transmittance = 1.0;
        stokesLayerStack(rho_90, tau_90, static_cast<double>(numberlayers) - 1.0, stack_reflectance, stack_transmittance);

        // ... and combined with the directly illuminated top layer (J&B 1990 Eqs. 7-8).
        const double interreflection = 1.0 - rho_90 * stack_reflectance;
        const double leaf_reflectance = rho_alpha + tau_alpha * tau_90 * stack_reflectance / interreflection;
        const double leaf_transmittance = tau_alpha * stack_transmittance / interreflection;

        reflectivities_fit.push_back(static_cast<float>(leaf_reflectance));
        transmissivities_fit.push_back(static_cast<float>(leaf_transmittance));
    }
}

void LeafOptics::validateProperties(const LeafOpticsProperties &leafproperties) {

    // The structure parameter N counts the elementary layers making up the leaf.
    // Values below 1 are unphysical: they make the Stokes exponent (N-1) negative,
    // which inverts the layer stack, and N == 0 divides by zero when computing the
    // mean absorption coefficient of each layer.
    if (leafproperties.numberlayers < 1.f) {
        helios_runtime_error(
                "ERROR (LeafOptics): the leaf structure parameter 'numberlayers' (N) was " + std::to_string(leafproperties.numberlayers) +
                ", but PROSPECT requires numberlayers >= 1 because it counts the number of elementary layers in the leaf. Set numberlayers to 1 for a single compact layer, or a larger value (typically 1.0-3.0) for a more scattering leaf.");
    }

    // All constituent contents are masses (or concentrations) per unit leaf area and
    // cannot be negative. A negative value would contribute negative absorption.
    auto check_non_negative = [](float value, const char *name, const char *units) {
        if (value < 0.f) {
            helios_runtime_error("ERROR (LeafOptics): the leaf constituent '" + std::string(name) + "' was " + std::to_string(value) + " " + units + ", but constituent contents cannot be negative. Set it to zero to disable this constituent.");
        }
    };

    check_non_negative(leafproperties.chlorophyllcontent, "chlorophyllcontent", "ug/cm^2");
    check_non_negative(leafproperties.carotenoidcontent, "carotenoidcontent", "ug/cm^2");
    check_non_negative(leafproperties.anthocyancontent, "anthocyancontent", "ug/cm^2");
    check_non_negative(leafproperties.brownpigments, "brownpigments", "(unitless)");
    check_non_negative(leafproperties.watermass, "watermass", "g/cm^2");
    check_non_negative(leafproperties.drymass, "drymass", "g/cm^2");
    check_non_negative(leafproperties.protein, "protein", "g/cm^2");
    check_non_negative(leafproperties.carbonconstituents, "carbonconstituents", "g/cm^2");
}

void LeafOptics::getLeafSpectra(const LeafOpticsProperties &leafproperties, std::vector<helios::vec2> &reflectivities_fit, std::vector<helios::vec2> &transmissivities_fit) {

    std::vector<float> reflectivities_fit_y, transmissivities_fit_y;

    validateProperties(leafproperties);

    // These are output arguments - overwrite rather than append so that repeated
    // calls with the same vectors do not accumulate results.
    reflectivities_fit.clear();
    transmissivities_fit.clear();
    reflectivities_fit.reserve(nw);
    transmissivities_fit.reserve(nw);

    float numberlayers = leafproperties.numberlayers;
    float chlorophyllcontent = leafproperties.chlorophyllcontent;
    float carotenoidcontent = leafproperties.carotenoidcontent;
    float anthocyancontent = leafproperties.anthocyancontent;
    float brownpigments = leafproperties.brownpigments;
    float watermass = leafproperties.watermass;
    float drymass = leafproperties.drymass;
    float protein = leafproperties.protein;
    float carbonconstituents = leafproperties.carbonconstituents;

    if (protein == 0 && carbonconstituents == 0) {
        if (drymass == 0 && message_flag) {
            std::cerr << "Warning: No leaf mass given" << std::endl;
        }
    } else {
        drymass = 0;
    }
    PROSPECT(numberlayers, chlorophyllcontent, carotenoidcontent, anthocyancontent, brownpigments, watermass, drymass, protein, carbonconstituents, reflectivities_fit_y, transmissivities_fit_y);

    // Convert float to vec2
    for (int iwave = 0; iwave < nw; iwave++) {
        reflectivities_fit.push_back(make_vec2(iwave + 400, reflectivities_fit_y.at(iwave)));
        transmissivities_fit.push_back(make_vec2(iwave + 400, transmissivities_fit_y.at(iwave)));
    }
}

void LeafOptics::surface(float degree, std::vector<float> &reflectivities) {
    // Surface reflectivity 1 - t_av(alpha, n) of the leaf-air interface for incidence within a cone of half-angle
    // 'degree' (J&B 1990, Eq. 1 and Fig. 1).
    if (degree <= 0.f || degree > 90.f) {
        helios_runtime_error("ERROR (LeafOptics::surface): the incidence cone half-angle must be in (0, 90] degrees, but " + std::to_string(degree) + " was given.");
    }
    reflectivities.resize(nw);
    for (uint iwave = 0; iwave < nw; iwave++) {
        reflectivities.at(iwave) = static_cast<float>(1.0 - averageInterfaceTransmissivity(degree, refractiveindex.at(iwave)));
    }
}

float LeafOptics::transmittance(double k) {
    return static_cast<float>(elementaryLayerTransmission(k));
}

void LeafOptics::setProperties(const std::vector<uint> &UUIDs, const LeafOpticsProperties &leafproperties) {
    for (const auto &data: output_prim_data) {
        if (data == "chlorophyll") {
            context->setPrimitiveData(UUIDs, "chlorophyll", leafproperties.chlorophyllcontent);
        } else if (data == "carotenoid") {
            context->setPrimitiveData(UUIDs, "carotenoid", leafproperties.carotenoidcontent);
        } else if (data == "anthocyanin") {
            context->setPrimitiveData(UUIDs, "anthocyanin", leafproperties.anthocyancontent);
        } else if (data == "brown" && leafproperties.brownpigments > 0.0) {
            context->setPrimitiveData(UUIDs, "brown", leafproperties.brownpigments);
        } else if (data == "water") {
            context->setPrimitiveData(UUIDs, "water", leafproperties.watermass);
        } else if (data == "drymass" && leafproperties.drymass > 0.0) {
            context->setPrimitiveData(UUIDs, "drymass", leafproperties.drymass);
        } else if (data == "protein" && leafproperties.drymass == 0.0) {
            context->setPrimitiveData(UUIDs, "protein", leafproperties.protein);
        } else if (data == "cellulose" && leafproperties.drymass == 0.0) {
            context->setPrimitiveData(UUIDs, "cellulose", leafproperties.carbonconstituents);
        }
    }
}

void LeafOptics::getPropertiesFromSpectrum(const std::vector<uint> &UUIDs) {
    const std::string prefix = "leaf_reflectivity_";

    for (uint UUID: UUIDs) {
        // Check if primitive has reflectivity_spectrum data
        if (!context->doesPrimitiveDataExist(UUID, "reflectivity_spectrum")) {
            continue; // Skip silently if no spectrum data
        }

        // Get the spectrum label
        std::string spectrum_label;
        context->getPrimitiveData(UUID, "reflectivity_spectrum", spectrum_label);

        // Check if this is a LeafOptics-generated spectrum (starts with prefix)
        if (spectrum_label.find(prefix) != 0) {
            continue; // Not a LeafOptics spectrum, skip silently
        }

        // Extract the user label by removing the prefix
        std::string user_label = spectrum_label.substr(prefix.length());

        // Check if we have parameters stored for this label
        auto it = spectrum_parameters_map.find(user_label);
        if (it == spectrum_parameters_map.end()) {
            continue; // No parameters found, skip silently
        }

        // Retrieve the stored parameters
        const LeafOpticsProperties &props = it->second;

        // Assign primitive data using the same logic as setProperties()
        for (const auto &data: output_prim_data) {
            if (data == "chlorophyll") {
                context->setPrimitiveData(UUID, "chlorophyll", props.chlorophyllcontent);
            } else if (data == "carotenoid") {
                context->setPrimitiveData(UUID, "carotenoid", props.carotenoidcontent);
            } else if (data == "anthocyanin") {
                context->setPrimitiveData(UUID, "anthocyanin", props.anthocyancontent);
            } else if (data == "brown" && props.brownpigments > 0.0) {
                context->setPrimitiveData(UUID, "brown", props.brownpigments);
            } else if (data == "water") {
                context->setPrimitiveData(UUID, "water", props.watermass);
            } else if (data == "drymass" && props.drymass > 0.0) {
                context->setPrimitiveData(UUID, "drymass", props.drymass);
            } else if (data == "protein" && props.drymass == 0.0) {
                context->setPrimitiveData(UUID, "protein", props.protein);
            } else if (data == "cellulose" && props.drymass == 0.0) {
                context->setPrimitiveData(UUID, "cellulose", props.carbonconstituents);
            }
        }
    }
}

void LeafOptics::getPropertiesFromSpectrum(uint UUID) {
    getPropertiesFromSpectrum(std::vector<uint>{UUID});
}

void LeafOptics::getPropertiesFromLibrary(const std::string &species, LeafOpticsProperties &leafproperties) {
    // Convert species name to lowercase for case-insensitive lookup
    std::string species_lower = species;
    std::transform(species_lower.begin(), species_lower.end(), species_lower.begin(), ::tolower);

    // Look up species in library
    auto it = species_library.find(species_lower);
    if (it != species_library.end()) {
        // Species found in library
        leafproperties = it->second;
        if (message_flag) {
            std::cout << "Setting Leaf Optics Properties to species: " << species << std::endl;
        }
    } else {
        // Species not found - use default and issue warning
        if (message_flag) {
            std::cerr << "WARNING (LeafOptics): unknown species \"" << species << "\". Using default properties." << std::endl;
        }
        leafproperties = species_library["default"];
    }
}

void LeafOptics::disableMessages() {
    message_flag = false;
}

void LeafOptics::enableMessages() {
    message_flag = true;
}

void LeafOptics::optionalOutputPrimitiveData(const char *label) {
    if (strcmp(label, "chlorophyll") == 0 || strcmp(label, "carotenoid") == 0 || strcmp(label, "anthocyanin") == 0 || strcmp(label, "brown") == 0 || strcmp(label, "water") == 0 || strcmp(label, "drymass") == 0 || strcmp(label, "protein") == 0 ||
        strcmp(label, "cellulose") == 0) {
        output_prim_data.emplace_back(label);
    } else {
        if (message_flag) {
            std::cout << "WARNING (LeafOptics::optionalOutputPrimitiveData): unknown output primitive data " << label << std::endl;
        }
    }
}

// === Nitrogen mode helper methods ===

LeafOpticsProperties LeafOptics::computePropertiesFromNitrogen(float N_area_gN_m2, const LeafOpticsProperties_Nauto &params) {
    LeafOpticsProperties props;

    // Convert nitrogen concentration to chlorophyll content
    // N (g/m2) -> N (ug/cm2): multiply by 100
    // Then apply photosynthetic fraction and empirical coefficient
    float N_area_ug_cm2 = N_area_gN_m2 * 100.0f;
    props.chlorophyllcontent = N_area_ug_cm2 * params.f_photosynthetic * params.N_to_Cab_coefficient;

    // Clamp chlorophyll to physically reasonable range [5, 80] ug/cm2
    props.chlorophyllcontent = std::max(5.0f, std::min(80.0f, props.chlorophyllcontent));

    // Calculate carotenoids from chlorophyll ratio
    props.carotenoidcontent = props.chlorophyllcontent * params.Car_to_Cab_ratio;

    // Copy fixed parameters from Nauto struct
    props.numberlayers = params.numberlayers;
    props.anthocyancontent = params.anthocyancontent;
    props.brownpigments = params.brownpigments;
    props.watermass = params.watermass;
    props.drymass = params.drymass;
    props.protein = params.protein;
    props.carbonconstituents = params.carbonconstituents;
    props.V2Z = params.V2Z;
    props.fqe = params.fqe;

    return props;
}

std::map<uint, std::vector<uint>> LeafOptics::groupPrimitivesByObject(const std::vector<uint> &UUIDs) {
    std::map<uint, std::vector<uint>> object_groups;

    helios::WarningAggregator warnings;
    warnings.setEnabled(message_flag);

    for (uint UUID: UUIDs) {
        if (!context->doesPrimitiveExist(UUID)) {
            warnings.addWarning("primitive_does_not_exist", "Primitive UUID " + std::to_string(UUID) + " does not exist, skipping.");
            continue;
        }

        uint objID = context->getPrimitiveParentObjectID(UUID);

        // Object ID 0 means no parent object
        if (objID == 0) {
            warnings.addWarning("primitive_no_parent_object", "Primitive UUID " + std::to_string(UUID) + " has no parent object, skipping.");
            continue;
        }

        object_groups[objID].push_back(UUID);
    }

    warnings.report(std::cerr);

    return object_groups;
}

void LeafOptics::createAdaptiveBins(const std::vector<float> &nitrogen_values) {
    nitrogen_bins.clear();

    if (nitrogen_values.empty()) {
        return;
    }

    // Bin centres are spaced evenly across the range the leaves currently span. Placing them at quantiles of
    // the population concentrated them wherever many leaves shared a value -- in a growing canopy, every
    // leaf that has just emerged sits at zero -- and left the rest of the range to one or two spectra, so
    // leaves of quite different nitrogen rendered identically and the canopy showed hard colour steps.
    const auto [lowest, highest] = std::minmax_element(nitrogen_values.begin(), nitrogen_values.end());
    const float range_low = *lowest;
    const float range_high = *highest;

    uint target_bins = std::max(1u, nitrogen_params.num_bins);
    if (range_high - range_low < 0.001f) {
        target_bins = 1; // every leaf holds the same nitrogen
    }

    std::vector<float> bin_centers;
    for (uint i = 0; i < target_bins; i++) {
        if (target_bins == 1) {
            bin_centers.push_back(0.5f * (range_low + range_high));
        } else {
            bin_centers.push_back(range_low + (range_high - range_low) * float(i) / float(target_bins - 1));
        }
    }

    // Generate PROSPECT spectrum for each bin
    for (uint i = 0; i < bin_centers.size(); i++) {
        SpectrumBin bin;
        bin.N_center = bin_centers[i];
        bin.spectrum_label = "Nauto_" + std::to_string(i);

        // Compute PROSPECT properties and generate spectrum
        LeafOpticsProperties props = computePropertiesFromNitrogen(bin.N_center, nitrogen_params);
        run(props, bin.spectrum_label);

        nitrogen_bins.push_back(bin);
    }

    if (message_flag) {
        std::cout << "LeafOptics: Created " << nitrogen_bins.size() << " nitrogen-based spectrum bins." << std::endl;
    }
}

uint LeafOptics::findNearestBin(float N_value) {
    if (nitrogen_bins.empty()) {
        helios_runtime_error("ERROR (LeafOptics::findNearestBin): No nitrogen bins have been created.");
    }

    uint best_bin = 0;
    float best_distance = std::abs(N_value - nitrogen_bins[0].N_center);

    for (uint i = 1; i < nitrogen_bins.size(); i++) {
        float distance = std::abs(N_value - nitrogen_bins[i].N_center);
        if (distance < best_distance) {
            best_distance = distance;
            best_bin = i;
        }
    }

    return best_bin;
}

bool LeafOptics::shouldReassign(float current_N, uint current_bin) {
    if (current_bin >= nitrogen_bins.size()) {
        return false;
    }

    float bin_center = nitrogen_bins[current_bin].N_center;
    float absolute_change = std::abs(current_N - bin_center);

    // Check absolute threshold first
    if (absolute_change < nitrogen_params.min_reassignment_change) {
        return false;
    }

    // Check relative threshold. A bin centered at zero is the one a leaf is placed in when it
    // emerges holding no nitrogen, and it has no scale to measure a relative change against: any
    // movement away from it is an unbounded relative change. Treating that as no change at all
    // would leave every leaf that ever emerged permanently in the zero bin, rendering at the
    // minimum chlorophyll however much nitrogen it went on to accumulate.
    if (bin_center <= 0.0f) {
        return true;
    }
    float relative_change = absolute_change / bin_center;
    return relative_change > nitrogen_params.reassignment_threshold;
}

bool LeafOptics::isSignificantImprovement(float current_N, uint old_bin, uint new_bin) {
    if (old_bin >= nitrogen_bins.size() || new_bin >= nitrogen_bins.size()) {
        return false;
    }

    float old_distance = std::abs(current_N - nitrogen_bins[old_bin].N_center);
    float new_distance = std::abs(current_N - nitrogen_bins[new_bin].N_center);

    // Require improvement of at least half the minimum change threshold (hysteresis)
    return (old_distance - new_distance) > nitrogen_params.min_reassignment_change * 0.5f;
}

bool LeafOptics::nitrogenOutsideBinRange(const std::vector<float> &nitrogen_values) {
    if (nitrogen_bins.empty()) {
        return true;
    }

    // Bin centers are evenly spaced in ascending order, so the first and last bound the range the
    // bins can represent. A single bin has no spacing of its own, so the reassignment threshold
    // supplies the scale instead.
    const float range_low = nitrogen_bins.front().N_center;
    const float range_high = nitrogen_bins.back().N_center;
    float margin = nitrogen_params.min_reassignment_change;
    if (nitrogen_bins.size() > 1) {
        margin = 0.5f * (range_high - range_low) / float(nitrogen_bins.size() - 1);
    }

    for (float N_value: nitrogen_values) {
        if (N_value < range_low - margin || N_value > range_high + margin) {
            return true;
        }
    }

    return false;
}

void LeafOptics::assignObjectsToNearestBins(const std::map<uint, std::vector<uint>> &object_groups, const std::map<uint, float> &object_nitrogen) {
    for (const auto &pair: object_groups) {
        uint objID = pair.first;
        const std::vector<uint> &obj_UUIDs = pair.second;
        float N_area = object_nitrogen.at(objID);

        uint best_bin = findNearestBin(N_area);
        assignSpectrumToPrimitives(obj_UUIDs, best_bin);

        ObjectAssignment assignment;
        assignment.bin_index = best_bin;
        assignment.N_at_assignment = N_area;
        assignment.primitive_UUIDs = obj_UUIDs;
        object_assignments[objID] = assignment;

        for (uint UUID: obj_UUIDs) {
            primitive_to_object[UUID] = objID;
        }
    }
}

void LeafOptics::assignSpectrumToPrimitives(const std::vector<uint> &UUIDs, uint bin_index) {
    if (bin_index >= nitrogen_bins.size()) {
        helios_runtime_error("ERROR (LeafOptics::assignSpectrumToPrimitives): Invalid bin index " + std::to_string(bin_index));
    }

    const std::string &label = nitrogen_bins[bin_index].spectrum_label;
    std::string refl_label = "leaf_reflectivity_" + label;
    std::string trans_label = "leaf_transmissivity_" + label;

    context->setPrimitiveData(UUIDs, "reflectivity_spectrum", refl_label);
    context->setPrimitiveData(UUIDs, "transmissivity_spectrum", trans_label);

    // Stamp the Fluspect-B biochemistry label for this bin so the radiation plugin's
    // SIF pipeline can find it. The bin's global data is written in ensureNitrogenBinsInitialized
    // alongside the spectrum; here we just point the primitive at the right label.
    std::string fluspect_label = "fluspect_biochem_" + label;
    context->setPrimitiveData(UUIDs, "fluspect_spectrum", fluspect_label);

    // Write optional primitive data if any are enabled
    if (!output_prim_data.empty()) {
        // Compute properties from the bin's nitrogen center value
        LeafOpticsProperties props = computePropertiesFromNitrogen(nitrogen_bins[bin_index].N_center, nitrogen_params);

        for (const auto &data: output_prim_data) {
            if (data == "chlorophyll") {
                context->setPrimitiveData(UUIDs, "chlorophyll", props.chlorophyllcontent);
            } else if (data == "carotenoid") {
                context->setPrimitiveData(UUIDs, "carotenoid", props.carotenoidcontent);
            } else if (data == "anthocyanin") {
                context->setPrimitiveData(UUIDs, "anthocyanin", props.anthocyancontent);
            } else if (data == "brown" && props.brownpigments > 0.0) {
                context->setPrimitiveData(UUIDs, "brown", props.brownpigments);
            } else if (data == "water") {
                context->setPrimitiveData(UUIDs, "water", props.watermass);
            } else if (data == "drymass" && props.drymass > 0.0) {
                context->setPrimitiveData(UUIDs, "drymass", props.drymass);
            } else if (data == "protein" && props.drymass == 0.0) {
                context->setPrimitiveData(UUIDs, "protein", props.protein);
            } else if (data == "cellulose" && props.drymass == 0.0) {
                context->setPrimitiveData(UUIDs, "cellulose", props.carbonconstituents);
            }
        }
    }
}

void LeafOptics::run(const std::vector<uint> &UUIDs, const LeafOpticsProperties_Nauto &params) {
    if (UUIDs.empty()) {
        if (message_flag) {
            std::cout << "LeafOptics: Empty UUID list provided to nitrogen mode, nothing to do." << std::endl;
        }
        return;
    }

    // Group primitives by parent object
    std::map<uint, std::vector<uint>> object_groups = groupPrimitivesByObject(UUIDs);

    if (object_groups.empty()) {
        if (message_flag) {
            std::cerr << "WARNING (LeafOptics::run): No valid objects found for nitrogen-based leaf optics." << std::endl;
        }
        return;
    }

    // Collect nitrogen values from all objects
    std::vector<float> nitrogen_values;
    std::map<uint, float> object_nitrogen;

    for (const auto &pair: object_groups) {
        uint objID = pair.first;

        // Check if nitrogen data exists on this object
        if (!context->doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            helios_runtime_error("ERROR (LeafOptics::run): Object " + std::to_string(objID) + " does not have 'leaf_nitrogen_gN_m2' data. Enable PlantArchitecture nitrogen model first.");
        }

        float N_area;
        context->getObjectData(objID, "leaf_nitrogen_gN_m2", N_area);
        nitrogen_values.push_back(N_area);
        object_nitrogen[objID] = N_area;
    }

    if (!nitrogen_mode_active) {
        // First call: Initialize nitrogen mode
        nitrogen_params = params;

        // Create adaptive bins based on nitrogen distribution
        createAdaptiveBins(nitrogen_values);

        // Assign each object to nearest bin
        assignObjectsToNearestBins(object_groups, object_nitrogen);

        nitrogen_mode_active = true;

        if (message_flag) {
            std::cout << "LeafOptics: Nitrogen mode initialized with " << object_assignments.size() << " leaf objects assigned to " << nitrogen_bins.size() << " spectrum bins." << std::endl;
        }

    } else {
        // Subsequent call: Update assignments

        // Build set of current object IDs
        std::set<uint> current_objects;
        for (const auto &pair: object_groups) {
            current_objects.insert(pair.first);
        }

        // Build set of previously tracked objects
        std::set<uint> tracked_objects;
        for (const auto &pair: object_assignments) {
            tracked_objects.insert(pair.first);
        }

        // Find removed objects (tracked but not in current)
        std::vector<uint> removed_objects;
        for (uint objID: tracked_objects) {
            if (current_objects.find(objID) == current_objects.end()) {
                removed_objects.push_back(objID);
            }
        }

        // Remove tracking for removed objects
        for (uint objID: removed_objects) {
            const ObjectAssignment &assignment = object_assignments[objID];
            for (uint UUID: assignment.primitive_UUIDs) {
                primitive_to_object.erase(UUID);
            }
            object_assignments.erase(objID);
        }

        if (nitrogenOutsideBinRange(nitrogen_values)) {
            // Nitrogen has moved beyond what the existing bins can represent, which happens
            // routinely over a simulated season. Snapping these leaves to the nearest bin would
            // clamp them to an extreme bin and hold their spectrum fixed for the rest of the run,
            // so the bins are rebuilt over the current distribution. Every object is reassigned
            // because the bin labels are reused: an assignment left over from the previous bins
            // would otherwise point a leaf at a different bin's spectrum.
            createAdaptiveBins(nitrogen_values);
            assignObjectsToNearestBins(object_groups, object_nitrogen);

            if (message_flag) {
                std::cout << "LeafOptics: Leaf nitrogen moved outside the established bin range. Rebuilt " << nitrogen_bins.size() << " spectrum bins and reassigned " << object_assignments.size() << " leaf objects." << std::endl;
            }

            return;
        }

        // Process current objects
        for (const auto &pair: object_groups) {
            uint objID = pair.first;
            const std::vector<uint> &obj_UUIDs = pair.second;
            float N_area = object_nitrogen[objID];

            if (tracked_objects.find(objID) == tracked_objects.end()) {
                // New object: assign to nearest existing bin
                uint best_bin = findNearestBin(N_area);
                assignSpectrumToPrimitives(obj_UUIDs, best_bin);

                ObjectAssignment assignment;
                assignment.bin_index = best_bin;
                assignment.N_at_assignment = N_area;
                assignment.primitive_UUIDs = obj_UUIDs;
                object_assignments[objID] = assignment;

                for (uint UUID: obj_UUIDs) {
                    primitive_to_object[UUID] = objID;
                }

            } else {
                // Existing object: check for reassignment
                ObjectAssignment &assignment = object_assignments[objID];

                // Detect new primitives not in the previous assignment
                std::set<uint> prev_UUIDs(assignment.primitive_UUIDs.begin(), assignment.primitive_UUIDs.end());
                std::vector<uint> new_UUIDs;
                for (uint UUID: obj_UUIDs) {
                    if (prev_UUIDs.find(UUID) == prev_UUIDs.end()) {
                        new_UUIDs.push_back(UUID);
                    }
                }

                // Update primitive list (in case it changed)
                assignment.primitive_UUIDs = obj_UUIDs;
                for (uint UUID: obj_UUIDs) {
                    primitive_to_object[UUID] = objID;
                }

                // Always assign spectra to new primitives using the current bin
                if (!new_UUIDs.empty()) {
                    uint best_bin = findNearestBin(N_area);
                    assignSpectrumToPrimitives(new_UUIDs, best_bin);
                    assignment.bin_index = best_bin;
                    assignment.N_at_assignment = N_area;
                } else if (shouldReassign(N_area, assignment.bin_index)) {
                    uint best_bin = findNearestBin(N_area);

                    if (best_bin != assignment.bin_index && isSignificantImprovement(N_area, assignment.bin_index, best_bin)) {
                        assignSpectrumToPrimitives(obj_UUIDs, best_bin);
                        assignment.bin_index = best_bin;
                        assignment.N_at_assignment = N_area;
                    }
                }
            }
        }
    }
}

void LeafOptics::updateNitrogenBasedSpectra() {
    if (!nitrogen_mode_active) {
        helios_runtime_error("ERROR (LeafOptics::updateNitrogenBasedSpectra): Nitrogen mode is not active. "
                             "Call run() with LeafOpticsProperties_Nauto first to initialize nitrogen mode.");
    }

    helios::WarningAggregator warnings;
    warnings.setEnabled(message_flag);

    for (auto &pair: object_assignments) {
        uint objID = pair.first;
        ObjectAssignment &assignment = pair.second;

        // Read current nitrogen value
        if (!context->doesObjectDataExist(objID, "leaf_nitrogen_gN_m2")) {
            warnings.addWarning("missing_nitrogen_data", "Object " + std::to_string(objID) + " no longer has 'leaf_nitrogen_gN_m2' data, skipping.");
            continue;
        }

        float current_N;
        context->getObjectData(objID, "leaf_nitrogen_gN_m2", current_N);

        // Check for reassignment
        if (shouldReassign(current_N, assignment.bin_index)) {
            uint best_bin = findNearestBin(current_N);

            if (best_bin != assignment.bin_index && isSignificantImprovement(current_N, assignment.bin_index, best_bin)) {
                assignSpectrumToPrimitives(assignment.primitive_UUIDs, best_bin);
                assignment.bin_index = best_bin;
                assignment.N_at_assignment = current_N;
            }
        }
    }

    warnings.report(std::cerr);
}
