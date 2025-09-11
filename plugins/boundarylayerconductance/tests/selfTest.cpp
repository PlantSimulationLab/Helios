#include "BoundaryLayerConductanceModel.h"

#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest.h>
#include "doctest_utils.h"

using namespace helios;

float err_tol = 1e-3;

DOCTEST_TEST_CASE("BLConductanceModel check blc values") {

    Context context;
    float TL = 300;
    float Ta = 290;
    float U = 0.6;

    std::vector<uint> UUID_1;
    UUID_1.push_back(context.addPatch());
    UUID_1.push_back(context.addPatch());
    UUID_1.push_back(context.addPatch());
    UUID_1.push_back(context.addPatch());

    context.setPrimitiveData(UUID_1, "air_temperature", Ta);
    context.setPrimitiveData(UUID_1, "temperature", TL);
    context.setPrimitiveData(UUID_1, "wind_speed", U);

    // Instantiate the boundary layer conductance model
    BLConductanceModel blc(&context);
    blc.disableMessages();

    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(0), "Pohlhausen"));
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(1), "InclinedPlate"));
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(2), "Sphere"));
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(3), "Ground"));

    capture_cerr cerr_buffer;
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel(UUID_1.at(3), "InvalidModel"));
    DOCTEST_CHECK(cerr_buffer.has_output());

    DOCTEST_CHECK_NOTHROW(blc.run());

    std::vector<float> gH(UUID_1.size());

    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(0), "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(1), "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(2), "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_1.at(3), "boundarylayer_conductance"));

    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(0), "boundarylayer_conductance", gH.at(0)));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(1), "boundarylayer_conductance", gH.at(1)));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(2), "boundarylayer_conductance", gH.at(2)));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID_1.at(3), "boundarylayer_conductance", gH.at(3)));

    DOCTEST_CHECK(gH.at(0) == doctest::Approx(0.20914112f).epsilon(err_tol));
    DOCTEST_CHECK(gH.at(1) == doctest::Approx(0.135347f).epsilon(err_tol));
    DOCTEST_CHECK(gH.at(2) == doctest::Approx(0.087149f).epsilon(err_tol));
    DOCTEST_CHECK(gH.at(3) == doctest::Approx(0.465472f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("Default values handling") {
    Context context;
    uint UUID_default = context.addPatch();

    BLConductanceModel blc(&context);
    blc.disableMessages();
    blc.setBoundaryLayerModel(UUID_default, "Pohlhausen");
    blc.run();

    float default_gH = -1.f;
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID_default, "boundarylayer_conductance"));
    context.getPrimitiveData(UUID_default, "boundarylayer_conductance", default_gH);
    DOCTEST_CHECK(default_gH > 0);
}

DOCTEST_TEST_CASE("Single-sided primitive handling") {
    Context context;
    float TL = 300;
    float Ta = 290;
    float U = 0.6;

    uint UUID = context.addPatch();
    context.setPrimitiveData(UUID, "air_temperature", Ta);
    context.setPrimitiveData(UUID, "temperature", TL);
    context.setPrimitiveData(UUID, "wind_speed", U);
    context.setPrimitiveData(UUID, "twosided_flag", 0u); // single-sided

    BLConductanceModel blc(&context);
    blc.disableMessages();
    blc.setBoundaryLayerModel(UUID, "Pohlhausen");
    blc.run();

    float gH = 0.0f;
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUID, "boundarylayer_conductance"));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH == doctest::Approx(0.104571f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("Message flag tests") {
    Context context;
    BLConductanceModel blc(&context);

    DOCTEST_CHECK_NOTHROW(blc.enableMessages());

    DOCTEST_CHECK_NOTHROW(blc.disableMessages());
}

DOCTEST_TEST_CASE("setBoundaryLayerModel for all primitives") {
    Context context;
    uint u1 = context.addPatch();
    uint u2 = context.addPatch();

    BLConductanceModel blc(&context);
    blc.disableMessages();

    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel("Pohlhausen"));

    context.setPrimitiveData({u1, u2}, "air_temperature", 290.f);
    context.setPrimitiveData({u1, u2}, "temperature", 300.f);
    context.setPrimitiveData({u1, u2}, "wind_speed", 0.6f);

    blc.run();

    float gH1, gH2;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u1, "boundarylayer_conductance", gH1));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u2, "boundarylayer_conductance", gH2));
    DOCTEST_CHECK(gH1 == doctest::Approx(0.20914112f).epsilon(err_tol));
    DOCTEST_CHECK(gH2 == doctest::Approx(0.20914112f).epsilon(err_tol));

    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel("InclinedPlate"));
    blc.run();
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u1, "boundarylayer_conductance", gH1));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(u2, "boundarylayer_conductance", gH2));
    DOCTEST_CHECK(gH1 == doctest::Approx(0.135347f).epsilon(err_tol));
    DOCTEST_CHECK(gH2 == doctest::Approx(0.135347f).epsilon(err_tol));

    capture_cerr cerr_buffer;
    blc.enableMessages();
    DOCTEST_CHECK_NOTHROW(blc.setBoundaryLayerModel("InvalidModel"));
    DOCTEST_CHECK(cerr_buffer.has_output());
}

DOCTEST_TEST_CASE("Subset run") {
    Context context;
    float TL = 300;
    float Ta = 290;
    float U = 0.6;

    std::vector<uint> UUIDs;
    UUIDs.push_back(context.addPatch());
    UUIDs.push_back(context.addPatch());
    UUIDs.push_back(context.addPatch());

    context.setPrimitiveData(UUIDs, "air_temperature", Ta);
    context.setPrimitiveData(UUIDs, "temperature", TL);
    context.setPrimitiveData(UUIDs, "wind_speed", U);

    BLConductanceModel blc(&context);
    blc.disableMessages();

    blc.setBoundaryLayerModel(UUIDs[0], "Pohlhausen");
    blc.setBoundaryLayerModel(UUIDs[1], "InclinedPlate");
    blc.setBoundaryLayerModel(UUIDs[2], "Sphere");

    std::vector<uint> subsetUUIDs = {UUIDs[0], UUIDs[2]};
    DOCTEST_CHECK_NOTHROW(blc.run(subsetUUIDs));

    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUIDs[0], "boundarylayer_conductance"));
    DOCTEST_CHECK(!context.doesPrimitiveDataExist(UUIDs[1], "boundarylayer_conductance"));
    DOCTEST_CHECK(context.doesPrimitiveDataExist(UUIDs[2], "boundarylayer_conductance"));

    float gH0, gH2;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUIDs[0], "boundarylayer_conductance", gH0));
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUIDs[2], "boundarylayer_conductance", gH2));

    DOCTEST_CHECK(gH0 == doctest::Approx(0.20914112f).epsilon(err_tol));
    DOCTEST_CHECK(gH2 == doctest::Approx(0.087149f).epsilon(err_tol));
}

DOCTEST_TEST_CASE("Missing parameters and zero inputs") {
    Context context;
    BLConductanceModel blc(&context);
    blc.disableMessages();

    uint UUID = context.addPatch();
    context.setPrimitiveData(UUID, "atmospheric_pressure", 100000.f);
    context.setPrimitiveData(UUID, "object_length", 0.1f);
    context.setPrimitiveData(UUID, "inclination", 0.5f);

    // Run with default air_temperature, temperature, and wind_speed
    DOCTEST_CHECK_NOTHROW(blc.run());
    float gH;
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0);

    // Test zero characteristic dimension
    context.setPrimitiveData(UUID, "object_length", 0.f);
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH != 0.f);

    // Test zero wind speed
    context.setPrimitiveData(UUID, "object_length", 0.1f);
    context.setPrimitiveData(UUID, "wind_speed", 0.f);

    blc.setBoundaryLayerModel(UUID, "Pohlhausen");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH == 0.f);

    blc.setBoundaryLayerModel(UUID, "InclinedPlate");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0);

    blc.setBoundaryLayerModel(UUID, "Sphere");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0.f);

    blc.setBoundaryLayerModel(UUID, "Ground");
    DOCTEST_CHECK_NOTHROW(blc.run());
    DOCTEST_CHECK_NOTHROW(context.getPrimitiveData(UUID, "boundarylayer_conductance", gH));
    DOCTEST_CHECK(gH > 0.f);
}

int BLConductanceModel::selfTest(int argc, char **argv) {
    return helios::runDoctestWithValidation(argc, argv);
}
