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

    // Enable optional outputs before calling setProperties
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("carotenoid");
    leafoptics.optionalOutputPrimitiveData("anthocyanin");
    leafoptics.optionalOutputPrimitiveData("water");
    leafoptics.optionalOutputPrimitiveData("drymass");

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

    // Enable optional output for brown pigments
    leafoptics.optionalOutputPrimitiveData("brown");

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

    // Enable optional outputs for protein mode
    leafoptics.optionalOutputPrimitiveData("protein");
    leafoptics.optionalOutputPrimitiveData("cellulose");

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

    // Enable optional outputs for the parameters we want to retrieve
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("carotenoid");
    leafoptics.optionalOutputPrimitiveData("anthocyanin");
    leafoptics.optionalOutputPrimitiveData("water");
    leafoptics.optionalOutputPrimitiveData("drymass");

    // Generate spectra and assign to primitives
    leafoptics.run(UUIDs, props, label);

    // Clear existing parameter data from primitives (run() calls setProperties())
    for (uint UUID: UUIDs) {
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

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromSpectrum - Single UUID Overload") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    LeafOpticsProperties props;
    props.chlorophyllcontent = 55.0f;
    props.carotenoidcontent = 14.0f;

    std::string label = "test_single_uuid";

    // Enable optional outputs
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("carotenoid");

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

    // Enable optional outputs
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("brown");

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

    // Enable optional outputs for PRO mode
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("protein");
    leafoptics.optionalOutputPrimitiveData("cellulose");
    leafoptics.optionalOutputPrimitiveData("drymass"); // Request drymass but it won't be written since drymass=0

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

    // Should not have dry mass data (drymass=0 so it's not written even if requested)
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

    // Enable optional outputs
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("carotenoid");

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

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - Default Species") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;

    // Test with "default" species
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary("default", props));

    // Verify all parameters match expected default values
    DOCTEST_CHECK(props.numberlayers == doctest::Approx(1.5f).epsilon(err_tol));
    DOCTEST_CHECK(props.chlorophyllcontent == doctest::Approx(30.0f).epsilon(err_tol));
    DOCTEST_CHECK(props.carotenoidcontent == doctest::Approx(7.0f).epsilon(err_tol));
    DOCTEST_CHECK(props.anthocyancontent == doctest::Approx(1.0f).epsilon(err_tol));
    DOCTEST_CHECK(props.brownpigments == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(props.watermass == doctest::Approx(0.015f).epsilon(err_tol));
    DOCTEST_CHECK(props.drymass == doctest::Approx(0.09f).epsilon(err_tol));
    DOCTEST_CHECK(props.protein == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(props.carbonconstituents == doctest::Approx(0.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - Case Insensitivity") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props_lower, props_upper;

    // Test case insensitivity
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary("default", props_lower));
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary("Default", props_upper));

    // Both should have identical values
    DOCTEST_CHECK(props_lower.chlorophyllcontent == doctest::Approx(props_upper.chlorophyllcontent).epsilon(err_tol));
    DOCTEST_CHECK(props_lower.watermass == doctest::Approx(props_upper.watermass).epsilon(err_tol));
    DOCTEST_CHECK(props_lower.drymass == doctest::Approx(props_upper.drymass).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - Unknown Species") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props;

    // Test with unknown species - should use default without crashing
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary("UnknownSpecies", props));

    // Should have default values
    DOCTEST_CHECK(props.chlorophyllcontent == doctest::Approx(30.0f).epsilon(err_tol));
    DOCTEST_CHECK(props.watermass == doctest::Approx(0.015f).epsilon(err_tol));
    DOCTEST_CHECK(props.drymass == doctest::Approx(0.09f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - Integration with Run") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create test primitives
    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

    // Get properties from library
    LeafOpticsProperties props;
    leafoptics.getPropertiesFromLibrary("default", props);

    // Enable optional outputs
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("water");
    leafoptics.optionalOutputPrimitiveData("drymass");

    // Use properties to run the model
    std::string label = "test_library_integration";
    DOCTEST_CHECK_NOTHROW(leafoptics.run(UUIDs, props, label));

    // Verify spectra were created
    std::string refl_label = "leaf_reflectivity_" + label;
    std::string trans_label = "leaf_transmissivity_" + label;

    DOCTEST_CHECK(context_test.doesGlobalDataExist(refl_label.c_str()));
    DOCTEST_CHECK(context_test.doesGlobalDataExist(trans_label.c_str()));

    // Verify primitive data was set with correct values
    float chl, water, dry;
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUIDs[0], "chlorophyll", chl));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUIDs[0], "water", water));
    DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveData(UUIDs[0], "drymass", dry));

    DOCTEST_CHECK(chl == doctest::Approx(30.0f).epsilon(err_tol));
    DOCTEST_CHECK(water == doctest::Approx(0.015f).epsilon(err_tol));
    DOCTEST_CHECK(dry == doctest::Approx(0.09f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - LOPEX93 Species Library") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Test all 9 LOPEX93 species in the library
    struct SpeciesTestData {
        std::string name;
        float expected_N;
        float expected_Cab;
        float expected_drymass;
    };

    std::vector<SpeciesTestData> species_data = {{"garden_lettuce", 2.00517f, 30.2697f, 0.0052668f}, {"alfalfa", 2.00758f, 43.6375f, 0.00473702f},        {"corn", 1.59203f, 22.8664f, 0.00441283f},
                                                 {"sunflower", 1.76358f, 54.0514f, 0.00644855f},     {"english_walnut", 1.56274f, 55.9211f, 0.00583351f}, {"rice", 1.67081f, 37.233f, 0.00484587f},
                                                 {"soybean", 1.5375f, 46.4121f, 0.00292814f},        {"wine_grape", 1.42673f, 50.918f, 0.00599315f},      {"tomato", 1.40304f, 48.3467f, 0.00261571f}};

    for (const auto &species: species_data) {
        LeafOpticsProperties props;
        DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary(species.name, props));

        // Check key parameters match fitted values
        DOCTEST_CHECK(props.numberlayers == doctest::Approx(species.expected_N).epsilon(err_tol));
        DOCTEST_CHECK(props.chlorophyllcontent == doctest::Approx(species.expected_Cab).epsilon(err_tol));
        DOCTEST_CHECK(props.drymass == doctest::Approx(species.expected_drymass).epsilon(err_tol));

        // All LOPEX93 species should use PROSPECT-D mode
        DOCTEST_CHECK(props.drymass > 0.0f);
        DOCTEST_CHECK(props.protein == doctest::Approx(0.0f).epsilon(err_tol));
        DOCTEST_CHECK(props.carbonconstituents == doctest::Approx(0.0f).epsilon(err_tol));
    }
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - LOPEX93 Species Case Insensitivity") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties props_lower, props_upper, props_mixed;

    // Test case insensitivity for specific species
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary("corn", props_lower));
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary("CORN", props_upper));
    DOCTEST_CHECK_NOTHROW(leafoptics.getPropertiesFromLibrary("Corn", props_mixed));

    // All three should have identical values
    DOCTEST_CHECK(props_lower.chlorophyllcontent == doctest::Approx(props_upper.chlorophyllcontent).epsilon(err_tol));
    DOCTEST_CHECK(props_lower.chlorophyllcontent == doctest::Approx(props_mixed.chlorophyllcontent).epsilon(err_tol));
    DOCTEST_CHECK(props_lower.drymass == doctest::Approx(props_upper.drymass).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - LOPEX93 Complete Parameter Check") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Test one species with all parameters explicitly checked (sunflower)
    LeafOpticsProperties props;
    leafoptics.getPropertiesFromLibrary("sunflower", props);

    DOCTEST_CHECK(props.numberlayers == doctest::Approx(1.76358f).epsilon(err_tol));
    DOCTEST_CHECK(props.chlorophyllcontent == doctest::Approx(54.0514f).epsilon(err_tol));
    DOCTEST_CHECK(props.carotenoidcontent == doctest::Approx(12.9027f).epsilon(err_tol));
    DOCTEST_CHECK(props.anthocyancontent == doctest::Approx(1.75194f).epsilon(err_tol));
    DOCTEST_CHECK(props.brownpigments == doctest::Approx(0.0112026f).epsilon(err_tol));
    DOCTEST_CHECK(props.watermass == doctest::Approx(0.0185557f).epsilon(err_tol));
    DOCTEST_CHECK(props.drymass == doctest::Approx(0.00644855f).epsilon(err_tol));
    DOCTEST_CHECK(props.protein == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(props.carbonconstituents == doctest::Approx(0.0f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - LOPEX93 Integration Test") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create test primitives
    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));
    UUIDs.push_back(context_test.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1)));

    // Get properties for soybean from library
    LeafOpticsProperties soybean_props;
    leafoptics.getPropertiesFromLibrary("soybean", soybean_props);

    // Enable optional outputs
    leafoptics.optionalOutputPrimitiveData("chlorophyll");

    // Run model with library properties
    std::string label = "test_soybean";
    DOCTEST_CHECK_NOTHROW(leafoptics.run(UUIDs, soybean_props, label));

    // Verify spectra were created
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_reflectivity_test_soybean"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_transmissivity_test_soybean"));

    // Verify primitive data matches library values
    for (uint UUID: UUIDs) {
        float chl;
        context_test.getPrimitiveData(UUID, "chlorophyll", chl);
        DOCTEST_CHECK(chl == doctest::Approx(46.4121f).epsilon(err_tol)); // Soybean chlorophyll
    }
}

DOCTEST_TEST_CASE("LeafOptics GetPropertiesFromLibrary - LOPEX93 Species Comparison") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    LeafOpticsProperties lettuce_props, walnut_props;
    leafoptics.getPropertiesFromLibrary("garden_lettuce", lettuce_props);
    leafoptics.getPropertiesFromLibrary("english_walnut", walnut_props);

    // Verify species have different properties
    DOCTEST_CHECK(lettuce_props.chlorophyllcontent != doctest::Approx(walnut_props.chlorophyllcontent).epsilon(err_tol));
    DOCTEST_CHECK(lettuce_props.numberlayers != doctest::Approx(walnut_props.numberlayers).epsilon(err_tol));

    // English walnut should have higher chlorophyll than lettuce
    DOCTEST_CHECK(walnut_props.chlorophyllcontent > lettuce_props.chlorophyllcontent);
}

DOCTEST_TEST_CASE("LeafOptics optionalOutputPrimitiveData - No Output By Default") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create test primitives
    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

    LeafOpticsProperties props;
    props.chlorophyllcontent = 45.0f;
    props.carotenoidcontent = 12.0f;
    props.watermass = 0.018f;
    props.drymass = 0.085f;

    // Run without enabling any optional outputs
    leafoptics.run(UUIDs, props, "test_no_output");

    // Verify no primitive data was written (except for spectrum labels)
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "chlorophyll"));
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "carotenoid"));
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "anthocyanin"));
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "water"));
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "drymass"));

    // Spectrum labels should still be written
    DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUIDs[0], "reflectivity_spectrum"));
    DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUIDs[0], "transmissivity_spectrum"));
}

DOCTEST_TEST_CASE("LeafOptics optionalOutputPrimitiveData - Selective Output") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> UUIDs;
    UUIDs.push_back(context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)));

    LeafOpticsProperties props;
    props.chlorophyllcontent = 45.0f;
    props.carotenoidcontent = 12.0f;
    props.anthocyancontent = 3.0f;
    props.watermass = 0.018f;
    props.drymass = 0.085f;

    // Enable only chlorophyll and water
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("water");

    leafoptics.run(UUIDs, props, "test_selective");

    // Only chlorophyll and water should be written
    DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUIDs[0], "chlorophyll"));
    DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUIDs[0], "water"));

    // Other properties should not be written
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "carotenoid"));
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "anthocyanin"));
    DOCTEST_CHECK(!context_test.doesPrimitiveDataExist(UUIDs[0], "drymass"));

    // Verify correct values
    float chl, water;
    context_test.getPrimitiveData(UUIDs[0], "chlorophyll", chl);
    context_test.getPrimitiveData(UUIDs[0], "water", water);
    DOCTEST_CHECK(chl == doctest::Approx(props.chlorophyllcontent).epsilon(err_tol));
    DOCTEST_CHECK(water == doctest::Approx(props.watermass).epsilon(err_tol));
}

DOCTEST_TEST_CASE("LeafOptics optionalOutputPrimitiveData - Invalid Label Warning") {
    Context context_test;
    LeafOptics leafoptics(&context_test);

    // Capture stdout for warning message
    helios::capture_cout capture;

    // Try to add invalid label (messages enabled)
    leafoptics.optionalOutputPrimitiveData("invalid_label");

    std::string output = capture.get_captured_output();
    DOCTEST_CHECK(output.find("WARNING") != std::string::npos);
    DOCTEST_CHECK(output.find("invalid_label") != std::string::npos);
}

DOCTEST_TEST_CASE("LeafOptics optionalOutputPrimitiveData - All Valid Labels") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // All these should be valid labels (no warnings)
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("chlorophyll"));
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("carotenoid"));
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("anthocyanin"));
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("brown"));
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("water"));
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("drymass"));
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("protein"));
    DOCTEST_CHECK_NOTHROW(leafoptics.optionalOutputPrimitiveData("cellulose"));
}

// ============== Nitrogen Mode Tests ==============

DOCTEST_TEST_CASE("LeafOpticsProperties_Nauto Default Constructor") {
    LeafOpticsProperties_Nauto params;

    // Nitrogen-to-pigment conversion coefficients
    DOCTEST_CHECK(params.f_photosynthetic == doctest::Approx(0.50f).epsilon(err_tol));
    DOCTEST_CHECK(params.N_to_Cab_coefficient == doctest::Approx(0.40f).epsilon(err_tol));
    DOCTEST_CHECK(params.Car_to_Cab_ratio == doctest::Approx(0.25f).epsilon(err_tol));

    // Fixed PROSPECT parameters
    DOCTEST_CHECK(params.numberlayers == doctest::Approx(1.5f).epsilon(err_tol));
    DOCTEST_CHECK(params.anthocyancontent == doctest::Approx(1.0f).epsilon(err_tol));
    DOCTEST_CHECK(params.brownpigments == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(params.watermass == doctest::Approx(0.015f).epsilon(err_tol));
    DOCTEST_CHECK(params.drymass == doctest::Approx(0.006f).epsilon(err_tol));
    DOCTEST_CHECK(params.protein == doctest::Approx(0.0f).epsilon(err_tol));
    DOCTEST_CHECK(params.carbonconstituents == doctest::Approx(0.0f).epsilon(err_tol));

    // Binning parameters
    DOCTEST_CHECK(params.num_bins == 20);
    DOCTEST_CHECK(params.reassignment_threshold == doctest::Approx(0.30f).epsilon(err_tol));
    DOCTEST_CHECK(params.min_reassignment_change == doctest::Approx(0.3f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("Nitrogen Mode - First Call Creates Bins") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create some compound objects with leaf nitrogen data
    std::vector<uint> all_UUIDs;
    std::vector<uint> objIDs;

    for (int i = 0; i < 5; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i), 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
        objIDs.push_back(objID);

        // Set nitrogen data on each object (varying nitrogen levels)
        float N_area = 1.0f + float(i) * 0.5f;  // 1.0, 1.5, 2.0, 2.5, 3.0
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", N_area);

        // Get primitives for this object
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 3;  // Request 3 bins

    // Run nitrogen mode
    DOCTEST_CHECK_NOTHROW(leafoptics.run(all_UUIDs, params));

    // Verify spectra were created
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_reflectivity_Nauto_0"));
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_transmissivity_Nauto_0"));

    // Verify primitives were assigned spectra
    std::string refl_label;
    context_test.getPrimitiveData(all_UUIDs[0], "reflectivity_spectrum", refl_label);
    DOCTEST_CHECK(refl_label.find("leaf_reflectivity_Nauto_") == 0);
}

DOCTEST_TEST_CASE("Nitrogen Mode - Subsequent Call Reuses Bins") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create objects with nitrogen data
    std::vector<uint> all_UUIDs;
    std::vector<uint> objIDs;

    for (int i = 0; i < 3; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i), 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
        objIDs.push_back(objID);
        float N_area = 1.5f + float(i) * 0.3f;
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", N_area);
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 2;

    // First call
    leafoptics.run(all_UUIDs, params);

    // Record initial spectrum assignments
    std::string initial_label;
    context_test.getPrimitiveData(all_UUIDs[0], "reflectivity_spectrum", initial_label);

    // Subsequent call with same primitives (no nitrogen change)
    DOCTEST_CHECK_NOTHROW(leafoptics.run(all_UUIDs, params));

    // Spectrum assignment should remain the same
    std::string after_label;
    context_test.getPrimitiveData(all_UUIDs[0], "reflectivity_spectrum", after_label);
    DOCTEST_CHECK(initial_label == after_label);
}

DOCTEST_TEST_CASE("Nitrogen Mode - New Objects Assigned to Existing Bins") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create initial objects
    std::vector<uint> all_UUIDs;
    for (int i = 0; i < 3; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i), 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
        float N_area = 1.5f + float(i) * 0.5f;
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", N_area);
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 2;

    // First call initializes bins
    leafoptics.run(all_UUIDs, params);

    // Add a new object
    uint new_objID = context_test.addTileObject(make_vec3(5, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(new_objID, "leaf_nitrogen_gN_m2", 2.0f);
    std::vector<uint> new_UUIDs = context_test.getObjectPrimitiveUUIDs(new_objID);
    all_UUIDs.insert(all_UUIDs.end(), new_UUIDs.begin(), new_UUIDs.end());

    // Second call should assign new object to existing bin
    DOCTEST_CHECK_NOTHROW(leafoptics.run(all_UUIDs, params));

    // New object should have a spectrum assigned
    std::string refl_label;
    context_test.getPrimitiveData(new_UUIDs[0], "reflectivity_spectrum", refl_label);
    DOCTEST_CHECK(refl_label.find("leaf_reflectivity_Nauto_") == 0);
}

DOCTEST_TEST_CASE("Nitrogen Mode - Removed Objects Untracked") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create objects
    std::vector<uint> all_UUIDs;
    std::vector<uint> objIDs;
    for (int i = 0; i < 3; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i), 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
        objIDs.push_back(objID);
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 1.5f + float(i) * 0.3f);
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 2;

    // First call
    leafoptics.run(all_UUIDs, params);

    // Remove last object from the list
    std::vector<uint> reduced_UUIDs;
    for (int i = 0; i < 2; i++) {
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objIDs[i]);
        reduced_UUIDs.insert(reduced_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    // Second call with reduced list (should not throw)
    DOCTEST_CHECK_NOTHROW(leafoptics.run(reduced_UUIDs, params));
}

DOCTEST_TEST_CASE("Nitrogen Mode - Reassignment When N Changes Significantly") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create objects with distinct nitrogen levels
    std::vector<uint> all_UUIDs;
    std::vector<uint> objIDs;
    for (int i = 0; i < 4; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i), 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
        objIDs.push_back(objID);
        float N_area = 1.0f + float(i) * 1.0f;  // 1.0, 2.0, 3.0, 4.0
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", N_area);
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 2;
    params.reassignment_threshold = 0.3f;
    params.min_reassignment_change = 0.3f;

    // First call
    leafoptics.run(all_UUIDs, params);

    // Record initial assignment for first object
    std::vector<uint> first_obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objIDs[0]);
    std::string initial_label;
    context_test.getPrimitiveData(first_obj_UUIDs[0], "reflectivity_spectrum", initial_label);

    // Change nitrogen significantly for first object (from 1.0 to 3.5)
    context_test.setObjectData(objIDs[0], "leaf_nitrogen_gN_m2", 3.5f);

    // Second call should reassign
    leafoptics.run(all_UUIDs, params);

    std::string after_label;
    context_test.getPrimitiveData(first_obj_UUIDs[0], "reflectivity_spectrum", after_label);

    // Label may or may not change depending on bin structure, but it should not throw
    DOCTEST_CHECK(!after_label.empty());
}

DOCTEST_TEST_CASE("Nitrogen Mode - Missing Nitrogen Data Error") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create object WITHOUT nitrogen data
    uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

    LeafOpticsProperties_Nauto params;

    // Should throw error because nitrogen data is missing
    DOCTEST_CHECK_THROWS(leafoptics.run(UUIDs, params));
}

DOCTEST_TEST_CASE("Nitrogen Mode - Primitives Without Parent Skipped") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create a standalone primitive (no parent object)
    uint standalone_UUID = context_test.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));

    // Create an object with nitrogen data
    uint objID = context_test.addTileObject(make_vec3(1, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 2.0f);
    std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

    // Mix standalone and object primitives
    std::vector<uint> mixed_UUIDs = {standalone_UUID};
    mixed_UUIDs.insert(mixed_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());

    LeafOpticsProperties_Nauto params;
    params.num_bins = 1;

    // Should handle gracefully (warning but no throw)
    DOCTEST_CHECK_NOTHROW(leafoptics.run(mixed_UUIDs, params));

    // Object primitives should still be assigned
    std::string refl_label;
    context_test.getPrimitiveData(obj_UUIDs[0], "reflectivity_spectrum", refl_label);
    DOCTEST_CHECK(!refl_label.empty());
}

DOCTEST_TEST_CASE("Nitrogen Mode - updateNitrogenBasedSpectra Without Init Error") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Should throw because nitrogen mode was never initialized
    DOCTEST_CHECK_THROWS(leafoptics.updateNitrogenBasedSpectra());
}

DOCTEST_TEST_CASE("Nitrogen Mode - updateNitrogenBasedSpectra After Init") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create objects with nitrogen data
    std::vector<uint> all_UUIDs;
    std::vector<uint> objIDs;
    for (int i = 0; i < 3; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i), 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
        objIDs.push_back(objID);
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 1.5f + float(i) * 0.5f);
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 2;

    // Initialize nitrogen mode
    leafoptics.run(all_UUIDs, params);

    // updateNitrogenBasedSpectra should work now
    DOCTEST_CHECK_NOTHROW(leafoptics.updateNitrogenBasedSpectra());
}

DOCTEST_TEST_CASE("Nitrogen Mode - Spectrum Count Bounded by num_bins") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create many objects with varying nitrogen
    std::vector<uint> all_UUIDs;
    for (int i = 0; i < 50; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i) * 0.2f, 0, 0), make_vec2(0.05f, 0.05f), make_SphericalCoord(0, 0), make_int2(1, 1));
        float N_area = 0.5f + float(i) * 0.1f;  // Wide range of nitrogen values
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", N_area);
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 5;  // Limit to 5 bins

    leafoptics.run(all_UUIDs, params);

    // Count unique spectra created (should be <= num_bins)
    uint spectrum_count = 0;
    for (uint i = 0; i < 100; i++) {
        std::string label = "leaf_reflectivity_Nauto_" + std::to_string(i);
        if (context_test.doesGlobalDataExist(label.c_str())) {
            spectrum_count++;
        }
    }

    DOCTEST_CHECK(spectrum_count <= params.num_bins);
    DOCTEST_CHECK(spectrum_count > 0);
}

DOCTEST_TEST_CASE("Nitrogen Mode - Empty UUID List") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> empty_UUIDs;
    LeafOpticsProperties_Nauto params;

    // Should handle gracefully (no throw, no crash)
    DOCTEST_CHECK_NOTHROW(leafoptics.run(empty_UUIDs, params));
}

DOCTEST_TEST_CASE("Nitrogen Mode - Single Object") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 2.0f);
    std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

    LeafOpticsProperties_Nauto params;
    params.num_bins = 5;

    DOCTEST_CHECK_NOTHROW(leafoptics.run(UUIDs, params));

    // Should create exactly 1 bin (since only 1 object)
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_reflectivity_Nauto_0"));

    // Primitives should be assigned
    std::string refl_label;
    context_test.getPrimitiveData(UUIDs[0], "reflectivity_spectrum", refl_label);
    DOCTEST_CHECK(refl_label == "leaf_reflectivity_Nauto_0");
}

DOCTEST_TEST_CASE("Nitrogen Mode - Chlorophyll Clamping Low") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create object with very low nitrogen (should clamp Cab to 5)
    uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 0.1f);  // Very low N
    std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

    LeafOpticsProperties_Nauto params;
    params.num_bins = 1;

    // Should not throw
    DOCTEST_CHECK_NOTHROW(leafoptics.run(UUIDs, params));

    // Verify spectrum was created
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_reflectivity_Nauto_0"));
}

DOCTEST_TEST_CASE("Nitrogen Mode - Chlorophyll Clamping High") {
    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create object with very high nitrogen (should clamp Cab to 80)
    uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 10.0f);  // Very high N
    std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

    LeafOpticsProperties_Nauto params;
    params.num_bins = 1;

    // Should not throw
    DOCTEST_CHECK_NOTHROW(leafoptics.run(UUIDs, params));

    // Verify spectrum was created
    DOCTEST_CHECK(context_test.doesGlobalDataExist("leaf_reflectivity_Nauto_0"));
}

DOCTEST_TEST_CASE("Nitrogen Mode - Chlorophyll Calculation Verification") {
    // Verify the nitrogen-to-chlorophyll conversion formula:
    // Cab (ug/cm2) = N_area (g/m2) * 100 * f_photosynthetic * N_to_Cab_coefficient
    // With defaults: f_photosynthetic=0.5, N_to_Cab_coefficient=0.4
    // N=2.0 g/m2 -> Cab = 2.0 * 100 * 0.5 * 0.4 = 40 ug/cm2

    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Enable chlorophyll output so we can verify
    leafoptics.optionalOutputPrimitiveData("chlorophyll");
    leafoptics.optionalOutputPrimitiveData("carotenoid");

    uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 2.0f);
    std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

    LeafOpticsProperties_Nauto params;
    params.f_photosynthetic = 0.5f;
    params.N_to_Cab_coefficient = 0.4f;
    params.Car_to_Cab_ratio = 0.25f;
    params.num_bins = 1;

    leafoptics.run(UUIDs, params);

    // Use getPropertiesFromSpectrum to retrieve the PROSPECT parameters
    leafoptics.getPropertiesFromSpectrum(UUIDs);

    // Check chlorophyll: 2.0 * 100 * 0.5 * 0.4 = 40.0 ug/cm2
    float chl;
    context_test.getPrimitiveData(UUIDs[0], "chlorophyll", chl);
    DOCTEST_CHECK(chl == doctest::Approx(40.0f).epsilon(0.01f));

    // Check carotenoid: 40.0 * 0.25 = 10.0 ug/cm2
    float car;
    context_test.getPrimitiveData(UUIDs[0], "carotenoid", car);
    DOCTEST_CHECK(car == doctest::Approx(10.0f).epsilon(0.01f));
}

DOCTEST_TEST_CASE("Nitrogen Mode - Hysteresis Prevents Oscillation") {
    // Test that a leaf at a bin boundary doesn't flip-flop between bins
    // due to the hysteresis mechanism

    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    // Create two objects with distinctly different nitrogen to create 2 bins
    std::vector<uint> all_UUIDs;
    std::vector<uint> objIDs;

    // Object 0: low nitrogen
    uint obj0 = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(obj0, "leaf_nitrogen_gN_m2", 1.0f);
    objIDs.push_back(obj0);
    std::vector<uint> UUIDs0 = context_test.getObjectPrimitiveUUIDs(obj0);
    all_UUIDs.insert(all_UUIDs.end(), UUIDs0.begin(), UUIDs0.end());

    // Object 1: high nitrogen
    uint obj1 = context_test.addTileObject(make_vec3(1, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(obj1, "leaf_nitrogen_gN_m2", 3.0f);
    objIDs.push_back(obj1);
    std::vector<uint> UUIDs1 = context_test.getObjectPrimitiveUUIDs(obj1);
    all_UUIDs.insert(all_UUIDs.end(), UUIDs1.begin(), UUIDs1.end());

    LeafOpticsProperties_Nauto params;
    params.num_bins = 2;
    params.reassignment_threshold = 0.3f;
    params.min_reassignment_change = 0.3f;

    // First call creates bins at ~1.0 and ~3.0
    leafoptics.run(all_UUIDs, params);

    // Record initial assignment for object 0
    std::string initial_label;
    context_test.getPrimitiveData(UUIDs0[0], "reflectivity_spectrum", initial_label);

    // Move object 0 to boundary region (2.0 - midpoint between bins)
    // This is within reassignment threshold of bin center 1.0
    // but hysteresis should prevent oscillation
    context_test.setObjectData(obj0, "leaf_nitrogen_gN_m2", 1.5f);
    leafoptics.run(all_UUIDs, params);

    std::string after_small_change;
    context_test.getPrimitiveData(UUIDs0[0], "reflectivity_spectrum", after_small_change);

    // Small change should NOT trigger reassignment due to hysteresis
    // (relative change is 50% but improvement may not be significant)
    // The key is that repeated calls with same value don't oscillate

    // Call again with same nitrogen - should be stable
    leafoptics.run(all_UUIDs, params);
    std::string after_repeat;
    context_test.getPrimitiveData(UUIDs0[0], "reflectivity_spectrum", after_repeat);

    // Assignment should be stable (no oscillation)
    DOCTEST_CHECK(after_small_change == after_repeat);
}

DOCTEST_TEST_CASE("Nitrogen Mode - All Same Nitrogen Values") {
    // When all objects have identical nitrogen, only 1 bin should be created
    // regardless of num_bins setting

    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    std::vector<uint> all_UUIDs;
    const float same_nitrogen = 2.0f;

    // Create 10 objects all with identical nitrogen
    for (int i = 0; i < 10; i++) {
        uint objID = context_test.addTileObject(make_vec3(float(i) * 0.2f, 0, 0), make_vec2(0.05f, 0.05f), make_SphericalCoord(0, 0), make_int2(1, 1));
        context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", same_nitrogen);
        std::vector<uint> obj_UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        all_UUIDs.insert(all_UUIDs.end(), obj_UUIDs.begin(), obj_UUIDs.end());
    }

    LeafOpticsProperties_Nauto params;
    params.num_bins = 5;  // Request 5 bins, but should only create 1

    leafoptics.run(all_UUIDs, params);

    // Count how many bins were actually created
    uint spectrum_count = 0;
    for (uint i = 0; i < 10; i++) {
        std::string label = "leaf_reflectivity_Nauto_" + std::to_string(i);
        if (context_test.doesGlobalDataExist(label.c_str())) {
            spectrum_count++;
        }
    }

    // Should only have 1 bin since all nitrogen values are identical
    DOCTEST_CHECK(spectrum_count == 1);

    // All primitives should be assigned to the same spectrum
    std::string first_label;
    context_test.getPrimitiveData(all_UUIDs[0], "reflectivity_spectrum", first_label);

    for (uint UUID : all_UUIDs) {
        std::string label;
        context_test.getPrimitiveData(UUID, "reflectivity_spectrum", label);
        DOCTEST_CHECK(label == first_label);
    }
}

DOCTEST_TEST_CASE("Nitrogen Mode - Verify Spectrum Physical Properties") {
    // Verify that nitrogen-derived spectra have physically reasonable properties

    Context context_test;
    LeafOptics leafoptics(&context_test);
    leafoptics.disableMessages();

    uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(0.1f, 0.1f), make_SphericalCoord(0, 0), make_int2(2, 2));
    context_test.setObjectData(objID, "leaf_nitrogen_gN_m2", 2.0f);
    std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

    LeafOpticsProperties_Nauto params;
    params.num_bins = 1;

    leafoptics.run(UUIDs, params);

    // Retrieve the generated spectrum
    std::vector<vec2> refl_data, trans_data;
    context_test.getGlobalData("leaf_reflectivity_Nauto_0", refl_data);
    context_test.getGlobalData("leaf_transmissivity_Nauto_0", trans_data);

    // Check spectrum size (400-2500nm at 1nm resolution = 2101 points)
    DOCTEST_CHECK(refl_data.size() == 2101);
    DOCTEST_CHECK(trans_data.size() == 2101);

    // Check physical bounds: 0 <= R, T <= 1
    for (size_t i = 0; i < refl_data.size(); i++) {
        DOCTEST_CHECK(refl_data[i].y >= 0.0f);
        DOCTEST_CHECK(refl_data[i].y <= 1.0f);
        DOCTEST_CHECK(trans_data[i].y >= 0.0f);
        DOCTEST_CHECK(trans_data[i].y <= 1.0f);

        // Energy conservation: R + T <= 1 (with small tolerance)
        DOCTEST_CHECK(refl_data[i].y + trans_data[i].y <= 1.01f);
    }

    // Check wavelength range
    DOCTEST_CHECK(refl_data.front().x == doctest::Approx(400.0f).epsilon(0.01f));
    DOCTEST_CHECK(refl_data.back().x == doctest::Approx(2500.0f).epsilon(0.01f));
}

int LeafOptics::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
