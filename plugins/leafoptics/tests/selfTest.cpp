#include "LeafOptics.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

float err_tol = 1e-3;

DOCTEST_TEST_CASE("LeafOpticsProperties Default Constructor") {
    LeafOpticsProperties props;

    DOCTEST_CHECK(props.numberlayers == doctest::Approx(1.5f).epsilon(err_tol));
    DOCTEST_CHECK(props.brownpigments == doctest::Approx(0.f).epsilon(err_tol));
    DOCTEST_CHECK(props.chlorophyllcontent == doctest::Approx(30.f).epsilon(err_tol));
    DOCTEST_CHECK(props.carotenoidcontent == doctest::Approx(7.f).epsilon(err_tol));
    DOCTEST_CHECK(props.anthocyancontent == doctest::Approx(1.f).epsilon(err_tol));
    DOCTEST_CHECK(props.watermass == doctest::Approx(0.015f).epsilon(err_tol));
    DOCTEST_CHECK(props.drymass == doctest::Approx(0.09f).epsilon(err_tol));
    DOCTEST_CHECK(props.protein == doctest::Approx(0.f).epsilon(err_tol));
    DOCTEST_CHECK(props.carbonconstituents == doctest::Approx(0.f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOpticsProperties Parameterized Constructor") {
    float chl = 50.0f, car = 10.0f, ant = 2.0f, water = 0.020f, dry = 0.08f, prot = 0.001f, carb = 0.005f;
    LeafOpticsProperties props(chl, car, ant, water, dry, prot, carb);

    DOCTEST_CHECK(props.chlorophyllcontent == doctest::Approx(chl).epsilon(err_tol));
    DOCTEST_CHECK(props.carotenoidcontent == doctest::Approx(car).epsilon(err_tol));
    DOCTEST_CHECK(props.anthocyancontent == doctest::Approx(ant).epsilon(err_tol));
    DOCTEST_CHECK(props.watermass == doctest::Approx(water).epsilon(err_tol));
    DOCTEST_CHECK(props.drymass == doctest::Approx(dry).epsilon(err_tol));
    DOCTEST_CHECK(props.protein == doctest::Approx(prot).epsilon(err_tol));
    DOCTEST_CHECK(props.carbonconstituents == doctest::Approx(carb).epsilon(err_tol));

    // Default values should remain unchanged
    DOCTEST_CHECK(props.numberlayers == doctest::Approx(1.5f).epsilon(err_tol));
    DOCTEST_CHECK(props.brownpigments == doctest::Approx(0.f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOpticsProperties Edge Cases") {
    // Test with zero values
    LeafOpticsProperties zero_props(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
    DOCTEST_CHECK(zero_props.chlorophyllcontent == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(zero_props.carotenoidcontent == doctest::Approx(0.0f).epsilon(err_tol));

    // Test with extreme values
    LeafOpticsProperties extreme_props(1000.0f, 500.0f, 100.0f, 1.0f, 2.0f, 0.5f, 1.5f);
    DOCTEST_CHECK(extreme_props.chlorophyllcontent == doctest::Approx(1000.0f).epsilon(err_tol));
    DOCTEST_CHECK(extreme_props.protein == doctest::Approx(0.5f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics Constructor and Initialization") {
    Context context_test;

    DOCTEST_CHECK_NOTHROW(LeafOptics leafoptics(&context_test));

    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Test that spectral data was loaded correctly
    DOCTEST_CHECK(context_test.doesGlobalDataExist("refraction_index"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_chlorophyll"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_carotenoid"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_anthocyanin"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_brown"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_water"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_drymass"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_proteins"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("absorption_carbonconstituents"));
}

DOCTEST_TEST_CASE("LeafOptics Enable/Disable Messages") {
    Context context_test;
    LeafOptics leafoptics(&context_test);

    DOCTEST_CHECK_NOTHROW(leafoptics.enableMessages());
    DOCTEST_CHECK_NOTHROW(leafoptics.disableMessages());
}

DOCTEST_TEST_CASE("LeafOptics Basic Run with Label Only") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    std::string label = "test_basic";

    DOCTEST_CHECK_NOTHROW(leafoptics.run(props, label));

    // Verify global data was created
    std::string refl_label = "leaf_reflectivity_" + label;
    std::string trans_label = "leaf_transmissivity_" + label;

    DOCTEST_CHECK(context_test.doesGlobalDataExist(refl_label.c_str()));
    DOCTEST_CHECK(context_test.doesGlobalDataExist(trans_label.c_str()));

    // Verify data sizes
    std::vector<vec2> refl_data, trans_data;
    DOCTEST_CHECK_NOTHROW(context_test.getGlobalData(refl_label.c_str(), refl_data));
    DOCTEST_CHECK_NOTHROW(context_test.getGlobalData(trans_label.c_str(), trans_data));

    DOCTEST_CHECK(refl_data.size() == 2101); // nw wavelengths
    DOCTEST_CHECK(trans_data.size() == 2101);
}

DOCTEST_TEST_CASE("LeafOptics Run with UUIDs") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create test primitives
    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));
    UUIDs.push_back(context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)));

    LeafOpticsProperties props;
    std::string label = "test_uuids";

    DOCTEST_CHECK_NOTHROW(leafoptics.run(UUIDs, props, label));

    // Verify global data was created
    std::string refl_label = "leaf_reflectivity_" + label;
    std::string trans_label = "leaf_transmissivity_" + label;

    DOCTEST_CHECK(context_test.doesGlobalDataExist(refl_label.c_str()));
    DOCTEST_CHECK(context_test.doesGlobalDataExist(trans_label.c_str()));

    // Verify primitive data was set
    for (uint UUID: UUIDs) {
        DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUID, "reflectivity_spectrum"));
        DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUID, "transmissivity_spectrum"));

        std::string refl_spectrum_label, trans_spectrum_label;
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "reflectivity_spectrum", refl_spectrum_label));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "transmissivity_spectrum", trans_spectrum_label));

        DOCTEST_CHECK(refl_spectrum_label == refl_label);
        DOCTEST_CHECK(trans_spectrum_label == trans_label);
    }
}

DOCTEST_TEST_CASE("LeafOptics Run with Empty UUID Vector") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> empty_UUIDs;
    LeafOpticsProperties props;
    std::string label = "test_empty";

    DOCTEST_CHECK_NOTHROW(leafoptics.run(empty_UUIDs, props, label));

    // Global data should still be created
    std::string refl_label = "leaf_reflectivity_" + label;
    std::string trans_label = "leaf_transmissivity_" + label;

    DOCTEST_CHECK(context_test.doesGlobalDataExist(refl_label.c_str()));
    DOCTEST_CHECK(context_test.doesGlobalDataExist(trans_label.c_str()));
}

DOCTEST_TEST_CASE("LeafOptics GetLeafSpectra - Default Properties") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    std::vector<vec2> reflectivities, transmissivities;

    DOCTEST_CHECK_NOTHROW(leafoptics.getLeafSpectra(props, reflectivities, transmissivities));

    DOCTEST_CHECK(reflectivities.size() == 2101);
    DOCTEST_CHECK(transmissivities.size() == 2101);

    // Check wavelength range (400-2500 nm)
    DOCTEST_CHECK(reflectivities[0].x == doctest::Approx(400.0f).epsilon(err_tol));
    DOCTEST_CHECK(reflectivities.back().x == doctest::Approx(2500.0f).epsilon(err_tol));
    DOCTEST_CHECK(transmissivities[0].x == doctest::Approx(400.0f).epsilon(err_tol));
    DOCTEST_CHECK(transmissivities.back().x == doctest::Approx(2500.0f).epsilon(err_tol));

    // Check that values are physically reasonable (0-1 range)
    for (const auto &refl: reflectivities) {
        DOCTEST_CHECK(refl.y >= 0.0f);
        DOCTEST_CHECK(refl.y <= 1.0f);
    }
    for (const auto &trans: transmissivities) {
        DOCTEST_CHECK(trans.y >= 0.0f);
        DOCTEST_CHECK(trans.y <= 1.0f);
    }

    // Energy conservation: R + T <= 1 (allowing for absorption)
    for (size_t i = 0; i < reflectivities.size(); ++i) {
        DOCTEST_CHECK(reflectivities[i].y + transmissivities[i].y <= 1.01f); // Small tolerance for numerical precision
    }
}

DOCTEST_TEST_CASE("LeafOptics GetLeafSpectra - High Chlorophyll") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    props.chlorophyllcontent = 80.0f; // High chlorophyll
    props.carotenoidcontent = 15.0f;

    std::vector<vec2> reflectivities, transmissivities;
    DOCTEST_CHECK_NOTHROW(leafoptics.getLeafSpectra(props, reflectivities, transmissivities));

    DOCTEST_CHECK(reflectivities.size() == 2101);
    DOCTEST_CHECK(transmissivities.size() == 2101);

    // High chlorophyll should result in strong absorption in blue/red regions
    // and higher reflectance in green/NIR regions
}

DOCTEST_TEST_CASE("LeafOptics GetLeafSpectra - Autumn Leaves") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    props.chlorophyllcontent = 5.0f; // Low chlorophyll
    props.carotenoidcontent = 20.0f; // High carotenoids
    props.anthocyancontent = 15.0f; // High anthocyanins
    props.brownpigments = 0.5f; // Some brown pigments

    std::vector<vec2> reflectivities, transmissivities;
    DOCTEST_CHECK_NOTHROW(leafoptics.getLeafSpectra(props, reflectivities, transmissivities));

    DOCTEST_CHECK(reflectivities.size() == 2101);
    DOCTEST_CHECK(transmissivities.size() == 2101);
}

DOCTEST_TEST_CASE("LeafOptics GetLeafSpectra - Prospect-PRO Mode") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    props.protein = 0.002f; // Enable PRO mode
    props.carbonconstituents = 0.008f;
    props.drymass = 0.0f; // Should use protein/carbon instead

    std::vector<vec2> reflectivities, transmissivities;
    DOCTEST_CHECK_NOTHROW(leafoptics.getLeafSpectra(props, reflectivities, transmissivities));

    DOCTEST_CHECK(reflectivities.size() == 2101);
    DOCTEST_CHECK(transmissivities.size() == 2101);
}

DOCTEST_TEST_CASE("LeafOptics GetLeafSpectra - Extreme Values") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Test with extreme values
    LeafOpticsProperties extreme_props;
    extreme_props.numberlayers = 5.0f; // Very thick leaf
    extreme_props.chlorophyllcontent = 200.0f; // Very high
    extreme_props.watermass = 0.1f; // Very high water content

    std::vector<vec2> reflectivities, transmissivities;
    DOCTEST_CHECK_NOTHROW(leafoptics.getLeafSpectra(extreme_props, reflectivities, transmissivities));

    DOCTEST_CHECK(reflectivities.size() == 2101);
    DOCTEST_CHECK(transmissivities.size() == 2101);

    // Very thick leaf should have very low transmittance
    float avg_transmittance = 0.0f;
    for (const auto &trans: transmissivities) {
        avg_transmittance += trans.y;
    }
    avg_transmittance /= transmissivities.size();
    DOCTEST_CHECK(avg_transmittance < 0.1f); // Should be quite low
}

DOCTEST_TEST_CASE("LeafOptics GetLeafSpectra - Zero Values") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Test with zero pigment values
    LeafOpticsProperties zero_props;
    zero_props.chlorophyllcontent = 0.0f;
    zero_props.carotenoidcontent = 0.0f;
    zero_props.anthocyancontent = 0.0f;
    zero_props.brownpigments = 0.0f;

    std::vector<vec2> reflectivities, transmissivities;
    DOCTEST_CHECK_NOTHROW(leafoptics.getLeafSpectra(zero_props, reflectivities, transmissivities));

    DOCTEST_CHECK(reflectivities.size() == 2101);
    DOCTEST_CHECK(transmissivities.size() == 2101);

    // Without pigments, should have higher overall transmittance
    float avg_transmittance = 0.0f;
    for (const auto &trans: transmissivities) {
        avg_transmittance += trans.y;
    }
    avg_transmittance /= transmissivities.size();
    DOCTEST_CHECK(avg_transmittance > 0.1f); // Should be relatively high
}

DOCTEST_TEST_CASE("LeafOptics SetProperties") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create test primitives
    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));
    UUIDs.push_back(context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)));

    LeafOpticsProperties props;
    props.chlorophyllcontent = 45.0f;
    props.carotenoidcontent = 12.0f;
    props.anthocyancontent = 3.0f;
    props.watermass = 0.018f;
    props.drymass = 0.085f;

    DOCTEST_CHECK_NOTHROW(leafoptics.setProperties(UUIDs, props));

    // Verify properties were set on primitives
    for (uint UUID: UUIDs) {
        float chl, car, ant, water, dry;

        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "chlorophyll", chl));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "carotenoid", car));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "anthocyanin", ant));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "water", water));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "drymass", dry));

        DOCTEST_CHECK(chl == doctest::Approx(props.chlorophyllcontent).epsilon(err_tol));
        DOCTEST_CHECK(car == doctest::Approx(props.carotenoidcontent).epsilon(err_tol));
        DOCTEST_CHECK(ant == doctest::Approx(props.anthocyancontent).epsilon(err_tol));
        DOCTEST_CHECK(water == doctest::Approx(props.watermass).epsilon(err_tol));
        DOCTEST_CHECK(dry == doctest::Approx(props.drymass).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("LeafOptics SetProperties with Brown Pigments") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

    LeafOpticsProperties props;
    props.brownpigments = 0.3f; // Non-zero brown pigments

    DOCTEST_CHECK_NOTHROW(leafoptics.setProperties(UUIDs, props));

    // Verify brown pigments were set
    float brown;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUIDs[0], "brown", brown));
    DOCTEST_CHECK(brown == doctest::Approx(props.brownpigments).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics SetProperties with Protein Mode") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

    LeafOpticsProperties props;
    props.drymass = 0.0f; // Zero dry mass
    props.protein = 0.001f;
    props.carbonconstituents = 0.005f;

    DOCTEST_CHECK_NOTHROW(leafoptics.setProperties(UUIDs, props));

    // Should set protein and carbon instead of dry mass
    float protein, carbon;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUIDs[0], "protein", protein));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUIDs[0], "cellulose", carbon));

    DOCTEST_CHECK(protein == doctest::Approx(props.protein).epsilon(err_tol));
    DOCTEST_CHECK(carbon == doctest::Approx(props.carbonconstituents).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics SetProperties with Empty UUID Vector") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> empty_UUIDs;
    LeafOpticsProperties props;

    // Should not crash with empty vector
    DOCTEST_CHECK_NOTHROW(leafoptics.setProperties(empty_UUIDs, props));
}

DOCTEST_TEST_CASE("LeafOptics Different Label Formats") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;

    // Test empty label
    DOCTEST_CHECK_NOTHROW(leafoptics.run(props, ""));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_reflectivity_"));

    // Test label with special characters
    std::string special_label = "test_123-abc.xyz";
    DOCTEST_CHECK_NOTHROW(leafoptics.run(props, special_label));
    std::string expected_refl = "leaf_reflectivity_" + special_label;
    DOCTEST_CHECK(context_test.doesGlobalDataExist(expected_refl.c_str()));

    // Test very long label
    std::string long_label(100, 'x');
    DOCTEST_CHECK_NOTHROW(leafoptics.run(props, long_label));
    std::string expected_long_refl = "leaf_reflectivity_" + long_label;
    DOCTEST_CHECK(context_test.doesGlobalDataExist(expected_long_refl.c_str()));
}

DOCTEST_TEST_CASE("LeafOptics Wavelength Range and Data Consistency") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    std::vector<vec2> reflectivities, transmissivities;

    leafoptics.getLeafSpectra(props, reflectivities, transmissivities);

    // Verify wavelength consistency
    for (size_t i = 0; i < reflectivities.size(); ++i) {
        DOCTEST_CHECK(reflectivities[i].x == doctest::Approx(transmissivities[i].x).epsilon(err_tol));

        // Check wavelength increment (should be 1 nm)
        if (i > 0) {
            float wavelength_diff = reflectivities[i].x - reflectivities[i - 1].x;
            DOCTEST_CHECK(wavelength_diff == doctest::Approx(1.0f).epsilon(err_tol));
        }
    }
}

DOCTEST_TEST_CASE("LeafOptics Physical Realism Checks") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    props.numberlayers = 2.0f;
    props.chlorophyllcontent = 60.0f;

    std::vector<vec2> reflectivities, transmissivities;
    leafoptics.getLeafSpectra(props, reflectivities, transmissivities);

    // Check for physically realistic behavior
    // 1. Red edge should show increased reflectance around 700-750 nm
    bool found_red_edge = false;
    for (size_t i = 1; i < reflectivities.size() - 1; ++i) {
        float wavelength = reflectivities[i].x;
        if (wavelength >= 700.0f && wavelength <= 750.0f) {
            float slope = reflectivities[i + 1].y - reflectivities[i - 1].y;
            if (slope > 0.001f) { // Positive slope indicating red edge
                found_red_edge = true;
                break;
            }
        }
    }
    DOCTEST_CHECK(found_red_edge);

    // 2. NIR reflectance should generally be higher than visible
    float visible_avg = 0.0f, nir_avg = 0.0f;
    int visible_count = 0, nir_count = 0;

    for (const auto &refl: reflectivities) {
        if (refl.x >= 400.0f && refl.x <= 700.0f) {
            visible_avg += refl.y;
            visible_count++;
        } else if (refl.x >= 800.0f && refl.x <= 1200.0f) {
            nir_avg += refl.y;
            nir_count++;
        }
    }

    visible_avg /= visible_count;
    nir_avg /= nir_count;

    DOCTEST_CHECK(nir_avg > visible_avg); // NIR should be higher than visible
}

DOCTEST_TEST_CASE("LeafOptics Multiple Runs Data Consistency") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;
    props.chlorophyllcontent = 40.0f;

    std::vector<vec2> refl1, trans1, refl2, trans2;

    // Run twice with same parameters
    leafoptics.getLeafSpectra(props, refl1, trans1);
    leafoptics.getLeafSpectra(props, refl2, trans2);

    // Results should be identical
    DOCTEST_CHECK(refl1.size() == refl2.size());
    DOCTEST_CHECK(trans1.size() == trans2.size());

    for (size_t i = 0; i < refl1.size(); ++i) {
        DOCTEST_CHECK(refl1[i].x == doctest::Approx(refl2[i].x).epsilon(err_tol));
        DOCTEST_CHECK(refl1[i].y == doctest::Approx(refl2[i].y).epsilon(err_tol));
        DOCTEST_CHECK(trans1[i].x == doctest::Approx(trans2[i].x).epsilon(err_tol));
        DOCTEST_CHECK(trans1[i].y == doctest::Approx(trans2[i].y).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Basic Functionality") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create test primitives
    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));
    UUIDs.push_back(context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)));

    // Create custom properties
    LeafOpticsProperties props;
    props.chlorophyllcontent = 45.0f;
    props.carotenoidcontent = 12.0f;
    props.anthocyancontent = 3.0f;
    props.watermass = 0.018f;
    props.drymass = 0.085f;

    std::string label = "test_get_props";

    // Generate spectra and assign to primitives
    leafoptics.run(UUIDs, props, label);

    // Clear existing parameter data from primitives (run() calls setProperties())
    for (uint UUID : UUIDs) {
        context_test.clearPrimitiveData(UUID, "chlorophyll");
        context_test.clearPrimitiveData(UUID, "carotenoid");
        context_test.clearPrimitiveData(UUID, "anthocyanin");
        context_test.clearPrimitiveData(UUID, "water");
        context_test.clearPrimitiveData(UUID, "drymass");
    }

    // Verify parameters were cleared
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "chlorophyll"));

    // Now retrieve parameters from spectrum
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromSpectrum(UUIDs));

    // Verify all parameters were correctly assigned
    for (uint UUID : UUIDs) {
        float chl, car, ant, water, dry;

        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "chlorophyll", chl));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "carotenoid", car));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "anthocyanin", ant));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "water", water));
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "drymass", dry));

        DOCTEST_CHECK(chl == doctest::Approx(props.chlorophyllcontent).epsilon(err_tol));
        DOCTEST_CHECK(car == doctest::Approx(props.carotenoidcontent).epsilon(err_tol));
        DOCTEST_CHECK(ant == doctest::Approx(props.anthocyancontent).epsilon(err_tol));
        DOCTEST_CHECK(water == doctest::Approx(props.watermass).epsilon(err_tol));
        DOCTEST_CHECK(dry == doctest::Approx(props.drymass).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Single UUID Overload") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    LeafOpticsProperties props;
    props.chlorophyllcontent = 55.0f;
    props.carotenoidcontent = 14.0f;

    std::string label = "test_single_uuid";

    // Generate spectrum
    leafoptics.run(std::vector<uint>{UUID}, props, label);

    // Clear parameters
    context_test.clearPrimitiveData(UUID, "chlorophyll");
    context_test.clearPrimitiveData(UUID, "carotenoid");

    // Retrieve using single UUID overload
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromSpectrum(UUID));

    // Verify parameters were assigned
    float chl, car;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "chlorophyll", chl));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "carotenoid", car));

    DOCTEST_CHECK(chl == doctest::Approx(props.chlorophyllcontent).epsilon(err_tol));
    DOCTEST_CHECK(car == doctest::Approx(props.carotenoidcontent).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Non-LeafOptics Spectrum") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    // Assign a spectrum label that doesn't match LeafOptics pattern
    context_test.setPrimitiveData(UUID, "reflectivity_spectrum", "custom_spectrum");

    // Should not crash, just skip silently
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromSpectrum(UUID));

    // No parameters should be set
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUID, "chlorophyll"));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Missing Spectrum") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    // Primitive has no reflectivity_spectrum data
    // Should not crash, just skip silently
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromSpectrum(UUID));

    // No parameters should be set
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUID, "chlorophyll"));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Unknown Label") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    // Assign a LeafOptics-style label that wasn't generated by this instance
    context_test.setPrimitiveData(UUID, "reflectivity_spectrum", "leaf_reflectivity_unknown");

    // Should not crash, just skip silently
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromSpectrum(UUID));

    // No parameters should be set (or they might remain from previous operations)
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - With Brown Pigments") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    LeafOpticsProperties props;
    props.chlorophyllcontent = 20.0f;
    props.brownpigments = 0.4f; // Non-zero brown pigments

    std::string label = "test_brown";

    leafoptics.run(std::vector<uint>{UUID}, props, label);

    // Clear parameters
    context_test.clearPrimitiveData(UUID, "chlorophyll");
    context_test.clearPrimitiveData(UUID, "brown");

    // Retrieve parameters
    leafoptics.getPropertiesFromSpectrum(UUID);

    // Verify brown pigments were assigned
    float brown;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "brown", brown));
    DOCTEST_CHECK(brown == doctest::Approx(props.brownpigments).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - PROSPECT-PRO Mode") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    LeafOpticsProperties props;
    props.chlorophyllcontent = 40.0f;
    props.drymass = 0.0f; // Zero dry mass
    props.protein = 0.002f;
    props.carbonconstituents = 0.008f;

    std::string label = "test_pro_mode";

    leafoptics.run(std::vector<uint>{UUID}, props, label);

    // Clear parameters
    context_test.clearPrimitiveData(UUID, "chlorophyll");
    context_test.clearPrimitiveData(UUID, "protein");
    context_test.clearPrimitiveData(UUID, "cellulose");

    // Retrieve parameters
    leafoptics.getPropertiesFromSpectrum(UUID);

    // Verify protein and carbon were assigned (not dry mass)
    float protein, carbon;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "protein", protein));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUID, "cellulose", carbon));

    DOCTEST_CHECK(protein == doctest::Approx(props.protein).epsilon(err_tol));
    DOCTEST_CHECK(carbon == doctest::Approx(props.carbonconstituents).epsilon(err_tol));

    // Should not have dry mass data
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUID, "drymass"));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Multiple Spectra") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create two primitives with different spectra
    uint UUID1 = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
    uint UUID2 = context_test.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));

    LeafOpticsProperties props1;
    props1.chlorophyllcontent = 30.0f;
    props1.carotenoidcontent = 8.0f;

    LeafOpticsProperties props2;
    props2.chlorophyllcontent = 60.0f;
    props2.carotenoidcontent = 16.0f;

    // Generate different spectra
    leafoptics.run(std::vector<uint>{UUID1}, props1, "spectrum1");
    leafoptics.run(std::vector<uint>{UUID2}, props2, "spectrum2");

    // Clear parameters
    context_test.clearPrimitiveData(UUID1, "chlorophyll");
    context_test.clearPrimitiveData(UUID2, "chlorophyll");

    // Retrieve parameters for both
    leafoptics.getPropertiesFromSpectrum(std::vector<uint>{UUID1, UUID2});

    // Verify each primitive got its correct parameters
    float chl1, chl2;
    context_test.getPrimitiveData(UUID1, "chlorophyll", chl1);
    context_test.getPrimitiveData(UUID2, "chlorophyll", chl2);

    DOCTEST_CHECK(chl1 == doctest::Approx(props1.chlorophyllcontent).epsilon(err_tol));
    DOCTEST_CHECK(chl2 == doctest::Approx(props2.chlorophyllcontent).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Empty UUID Vector") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> empty_UUIDs;

    // Should not crash with empty vector
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromSpectrum(empty_UUIDs));
}

int LeafOptics::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
