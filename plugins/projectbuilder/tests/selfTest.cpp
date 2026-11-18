#include "ProjectBuilder.h"

#ifdef ENABLE_RADIATION_MODEL
#include "RadiationModel.h"
#endif

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

/// Skip the current test case if radiation requires a GPU backend that is not available.
#ifdef ENABLE_RADIATION_MODEL
#define SKIP_IF_NO_GPU() \
    do { \
        if (!RadiationModel::isGPUBackendAvailable()) { \
            helios::requireGPUOrFail("A projectbuilder test needs the radiation plugin's GPU backend and would have skipped."); \
            DOCTEST_MESSAGE("SKIPPED: No GPU backend available (radiation plugin requires GPU)"); \
            return; \
        } \
    } while (0)
#else
#define SKIP_IF_NO_GPU() ((void)0)
#endif

//! Access to the ProjectBuilder state that the GUI drives, for the tests below
struct ProjectBuilderTestHelper {
    static std::string addCanopy(ProjectBuilder &projectbuilder) {
        projectbuilder.addCanopy();
        return projectbuilder.current_canopy;
    }
    static canopy &getCanopy(ProjectBuilder &projectbuilder, const std::string &label) {
        return projectbuilder.canopy_dict.at(label);
    }
    //! Evaluate the Calculation tab's expression over every primitive
    static float calculate(ProjectBuilder &projectbuilder, const std::vector<std::string> &variables, const std::vector<std::string> &operators, const std::vector<std::string> &aggregations) {
        projectbuilder.calculation_selection_datagroup["All"] = true;
        projectbuilder.calculation_selection_primitive["All"] = true;
        projectbuilder.calculation_variables_global = variables;
        projectbuilder.calculation_operators_global = operators;
        projectbuilder.calculation_aggregations = aggregations;
        projectbuilder.globalCalculation();
        return projectbuilder.calculation_result_global;
    }
    static std::vector<uint> allGroupedPrimitives(ProjectBuilder &projectbuilder) {
        std::vector<uint> UUIDs;
        for (const auto &group: projectbuilder.primitive_UUIDs) {
            UUIDs.insert(UUIDs.end(), group.second.begin(), group.second.end());
        }
        return UUIDs;
    }
};

DOCTEST_TEST_CASE("ProjectBuilder - Constructor") {
    DOCTEST_CHECK_NOTHROW(ProjectBuilder projectbuilder);
}

DOCTEST_TEST_CASE("ProjectBuilder - buildFromXML") {
    SKIP_IF_NO_GPU();
    // Building the shipped project must not warn, e.g. about a deprecated camera property that the project builder itself sets.
    // The project builder is scoped so it is destroyed before the check: while it exists it redirects std::cout, which would swallow a failure message.
    std::string warnings;
    {
        ProjectBuilder projectbuilder;
        capture_cerr capture;
        projectbuilder.buildFromXML();
        warnings = capture.get_captured_output();
    }
    DOCTEST_CHECK_MESSAGE(warnings.empty(), warnings);
}

DOCTEST_TEST_CASE("ProjectBuilder - updateGround replaces the ground built from XML") {
    SKIP_IF_NO_GPU();
    ProjectBuilder projectbuilder;
    DOCTEST_CHECK_NOTHROW(projectbuilder.buildFromXML());
    DOCTEST_REQUIRE(projectbuilder.context != nullptr);

    // Coverage for updateGround(), which had none, plus the object ID that BuildGeometry() now hands back.
    //
    // This is deliberately not claimed as a regression test for the uninitialized ProjectBuilder::ground_objID that
    // updateGround() used to read: that read has no observable effect in the normal path, because updateGround()
    // deletes the ground primitives first and emptying an object deletes it, so the ground object goes away whether
    // or not the ID is correct. The only behavior the stale read changed was the case where the indeterminate value
    // happened to name an unrelated live object, which was then deleted -- not something a test can provoke
    // deterministically. The fix is the in-class initializer plus handing the real ID back.
    // BuildGeometry() is only compiled in when the plant architecture plug-in is enabled, and it is what builds the
    // initial ground and hands its object ID back. Note the ID is not expected to still name a live object here:
    // buildFromXML() finishes by calling updateGround(), which replaces that ground with a freshly built one.
#ifdef ENABLE_PLANT_ARCHITECTURE
    uint published_ground_objID = 0;
    DOCTEST_REQUIRE(projectbuilder.context->doesGlobalDataExist("ground_objID"));
    projectbuilder.context->getGlobalData("ground_objID", published_ground_objID);
    DOCTEST_CHECK(published_ground_objID != 0);
#endif

    size_t objects_after_build = projectbuilder.context->getAllObjectIDs().size();
    DOCTEST_CHECK_NOTHROW(projectbuilder.updateGround());
    size_t objects_after_update = projectbuilder.context->getAllObjectIDs().size();
    DOCTEST_CHECK(objects_after_update == objects_after_build);
}

#ifdef ENABLE_PLANT_ARCHITECTURE
DOCTEST_TEST_CASE("ProjectBuilder - BuildGeometry loads a canopy model file named relative to the build directory") {
    // Regression test. The existence check for 'canopy_model_obj_file' tested the ground model file instead, so a canopy model given
    // without a ground model was always reported as missing; and neither was looked for in the HELIOS_BUILD directory.
    ScopedForeignWorkingDirectory foreign_working_directory;

    const std::string xml_file = "buildgeometry_canopy_model.xml";
    {
        std::ofstream xml(xml_file);
        xml << "<helios>\n  <canopy_block>\n    <canopy_model_obj_file>lib/models/obj_object_test.obj</canopy_model_obj_file>\n  </canopy_block>\n</helios>\n";
    }

    Context context;
    PlantArchitecture plant_architecture(&context);
    std::vector<std::vector<uint>> canopy_UUID_vector;
    std::vector<std::vector<vec3>> individual_plant_locations;
    std::string error_message;
    {
        capture_cout suppress_warnings;
        try {
            BuildGeometry(xml_file, &plant_architecture, &context, canopy_UUID_vector, individual_plant_locations);
        } catch (const std::runtime_error &e) {
            error_message = e.what();
        }
    }
    DOCTEST_CHECK_MESSAGE(error_message.empty(), error_message);
    DOCTEST_CHECK(context.getPrimitiveCount() > 0);
}
#endif

#ifdef ENABLE_RADIATION_MODEL
DOCTEST_TEST_CASE("ProjectBuilder - InitializeRadiation assigns a ground spectrum that is not loaded yet") {
    // The radiation model loads a library spectrum by label when it needs it, so a project that names one without listing its library file must not end up with a black ground.
    SKIP_IF_NO_GPU();
    const std::filesystem::path xml_path = std::filesystem::temp_directory_path() / "helios_projectbuilder_ground_spectrum_test.xml";
    {
        std::ofstream xml_file(xml_path);
        xml_file << "<helios><radiation><diffuse_extinction_coeff>0.1</diffuse_extinction_coeff><air_turbidity>0.05</air_turbidity><ground_reflectivity_spectrum>soil_reflectivity_0000</ground_reflectivity_spectrum></radiation></helios>";
    }

    Context context;
    const uint ground_UUID = context.addPatch();
    context.setGlobalData("ground_UUIDs", std::vector<uint>{ground_UUID});
    DOCTEST_REQUIRE(!context.doesGlobalDataExist("soil_reflectivity_0000"));

    std::unique_ptr<RadiationModel> radiation_ptr;
    {
        capture_cout capture; // the constructor reports the ray-tracing backend before messages can be disabled
        radiation_ptr = std::make_unique<RadiationModel>(&context);
    }
    RadiationModel &radiation = *radiation_ptr;
    radiation.disableMessages();
    {
        capture_cout capture;
        InitializeRadiation(xml_path.string(), nullptr, &radiation, &context); // the solar position model is only used to calibrate a turbidity that is not given
    }
    std::filesystem::remove(xml_path);

    std::string ground_spectrum;
    if (context.doesPrimitiveDataExist(ground_UUID, "reflectivity_spectrum")) {
        context.getPrimitiveData(ground_UUID, "reflectivity_spectrum", ground_spectrum);
    }
    DOCTEST_CHECK(ground_spectrum == "soil_reflectivity_0000");
}
#endif

int ProjectBuilder::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}

DOCTEST_TEST_CASE("ProjectBuilder - the Calculation tab evaluates each term with its own variable and aggregation") {
    SKIP_IF_NO_GPU();
    ProjectBuilder projectbuilder;
    projectbuilder.buildFromXML();
    const std::vector<uint> UUIDs = ProjectBuilderTestHelper::allGroupedPrimitives(projectbuilder);
    DOCTEST_REQUIRE(UUIDs.size() > 1);
    projectbuilder.context->setPrimitiveData(UUIDs, "calc_a", 2.f);
    projectbuilder.context->setPrimitiveData(UUIDs, "calc_b", 5.f);
    const float count = float(UUIDs.size());

    DOCTEST_CHECK(ProjectBuilderTestHelper::calculate(projectbuilder, {"calc_a"}, {}, {"Sum"}) == doctest::Approx(2.f * count));
    DOCTEST_CHECK(ProjectBuilderTestHelper::calculate(projectbuilder, {"calc_a", "calc_b"}, {"+"}, {"Mean", "Mean"}) == doctest::Approx(7.f));
    DOCTEST_CHECK(ProjectBuilderTestHelper::calculate(projectbuilder, {"calc_a", "calc_b"}, {"x"}, {"Mean", "Mean"}) == doctest::Approx(10.f));
    DOCTEST_CHECK(ProjectBuilderTestHelper::calculate(projectbuilder, {"calc_b", "calc_a"}, {"/"}, {"Sum", "Mean"}) == doctest::Approx(5.f * count / 2.f));
    DOCTEST_CHECK(ProjectBuilderTestHelper::calculate(projectbuilder, {"calc_a", "calc_b", "calc_a"}, {"+", "-"}, {"Mean", "Sum", "Mean"}) == doctest::Approx(2.f + 5.f * count - 2.f));
}

#ifdef ENABLE_PLANT_ARCHITECTURE // canopy blocks are only read when the PlantArchitecture plug-in is part of the build
DOCTEST_TEST_CASE("ProjectBuilder - saving the project writes each canopy block with its own values after a canopy is added") {
    SKIP_IF_NO_GPU();
    // Work on a copy of the shipped project file, since saving writes to the file the project was built from
    const std::string project_file = (std::filesystem::temp_directory_path() / "helios_test_projectbuilder_save.xml").string();
    std::filesystem::copy_file(helios::resolveFilePath("plugins/projectbuilder/inputs/inputs.xml"), project_file, std::filesystem::copy_options::overwrite_existing);

    ProjectBuilder projectbuilder;
    projectbuilder.buildFromXML(project_file);
    const vec3 origin_0 = ProjectBuilderTestHelper::getCanopy(projectbuilder, "canopy_0").origin;
    const vec3 origin_1 = ProjectBuilderTestHelper::getCanopy(projectbuilder, "canopy_1").origin;
    DOCTEST_REQUIRE(origin_0 != origin_1);

    const std::string new_label = ProjectBuilderTestHelper::addCanopy(projectbuilder);
    canopy &new_canopy = ProjectBuilderTestHelper::getCanopy(projectbuilder, new_label);
    new_canopy.origin = make_vec3(7, 8, 9);
    new_canopy.age = 77.f;
    projectbuilder.xmlSetValues(project_file);

    pugi::xml_document saved;
    DOCTEST_REQUIRE(saved.load_file(project_file.c_str()));
    std::map<std::string, std::string> saved_origin, saved_age;
    for (pugi::xml_node block = saved.child("helios").child("canopy_block"); block; block = block.next_sibling("canopy_block")) {
        saved_origin[block.attribute("label").value()] = trim_whitespace(block.child("canopy_origin").child_value());
        saved_age[block.attribute("label").value()] = trim_whitespace(block.child("plant_age").child_value());
    }
    DOCTEST_REQUIRE(saved_origin.size() == 3);
    DOCTEST_CHECK(string2vec3(saved_origin.at("canopy_0").c_str()) == origin_0);
    DOCTEST_CHECK(string2vec3(saved_origin.at("canopy_1").c_str()) == origin_1);
    DOCTEST_CHECK(string2vec3(saved_origin.at(new_label).c_str()) == make_vec3(7, 8, 9));
    DOCTEST_CHECK(std::stof(saved_age.at(new_label)) == doctest::Approx(77.f));
    DOCTEST_CHECK(std::stof(saved_age.at("canopy_0")) == doctest::Approx(13.f));

    std::filesystem::remove(project_file);
}
#endif
