#pragma once
// =================================================================================
// Suite 5: Context Data
//
// Tests for managing primitive, object, and global data within the Context class.
// =================================================================================
TEST_CASE("Global Data") {
    Context ctx;

    SUBCASE("Set, Get, Exists, List, Clear") {
        ctx.setGlobalData("test_int", 123);
        ctx.setGlobalData("test_float", 456.789f);
        ctx.setGlobalData("test_string", "hello");
        std::vector<double> double_vec = {1.1, 2.2, 3.3};
        ctx.setGlobalData("test_double_vec", double_vec);

        DOCTEST_CHECK(ctx.doesGlobalDataExist("test_int"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("test_float"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("test_string"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("test_double_vec"));
        DOCTEST_CHECK(!ctx.doesGlobalDataExist("non_existent"));

        int i;
        ctx.getGlobalData("test_int", i);
        DOCTEST_CHECK(i == 123);

        float f;
        ctx.getGlobalData("test_float", f);
        DOCTEST_CHECK(f == doctest::Approx(456.789f));

        std::string s;
        ctx.getGlobalData("test_string", s);
        DOCTEST_CHECK(s == "hello");

        std::vector<double> dv;
        ctx.getGlobalData("test_double_vec", dv);
        DOCTEST_CHECK(dv.size() == 3);
        DOCTEST_CHECK(dv[1] == doctest::Approx(2.2));

        std::vector<std::string> labels = ctx.listGlobalData();
        DOCTEST_CHECK(labels.size() == 4);

        ctx.clearGlobalData("test_int");
        DOCTEST_CHECK(!ctx.doesGlobalDataExist("test_int"));
    }

    SUBCASE("Management") {
        ctx.setGlobalData("g_int", 5);
        DOCTEST_CHECK(ctx.getGlobalDataType("g_int") == HELIOS_TYPE_INT);
        DOCTEST_CHECK(ctx.getGlobalDataSize("g_int") == 1);

        ctx.duplicateGlobalData("g_int", "g_int_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int_copy"));

        ctx.renameGlobalData("g_int", "g_int_new");
        DOCTEST_CHECK(!ctx.doesGlobalDataExist("g_int"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int_new"));
    }

    SUBCASE("Increment") {
        ctx.setGlobalData("inc_me_int", 10);
        ctx.incrementGlobalData("inc_me_int", 5);
        int val_i;
        ctx.getGlobalData("inc_me_int", val_i);
        DOCTEST_CHECK(val_i == 15);

        ctx.setGlobalData("inc_me_uint", std::vector<uint>{1});
        ctx.incrementGlobalData("inc_me_uint", (uint) 1);
        std::vector<uint> val_u;
        ctx.getGlobalData("inc_me_uint", val_u);
        DOCTEST_CHECK(val_u[0] == 2);

        ctx.setGlobalData("inc_me_float", std::vector<float>{1.f});
        ctx.incrementGlobalData("inc_me_float", 1.f);
        std::vector<float> val_f;
        ctx.getGlobalData("inc_me_float", val_f);
        DOCTEST_CHECK(val_f[0] == doctest::Approx(2.f));

        ctx.setGlobalData("inc_me_double", std::vector<double>{1.0});
        ctx.incrementGlobalData("inc_me_double", 1.0);
        std::vector<double> val_d;
        ctx.getGlobalData("inc_me_double", val_d);
        DOCTEST_CHECK(val_d[0] == doctest::Approx(2.0));
    }

    SUBCASE("All Vector Types") {
        ctx.setGlobalData("g_uint_vec", std::vector<uint>{1});
        ctx.setGlobalData("g_vec2_vec", std::vector<vec2>{make_vec2(1, 1)});
        ctx.setGlobalData("g_vec3_vec", std::vector<vec3>{make_vec3(1, 1, 1)});
        ctx.setGlobalData("g_vec4_vec", std::vector<vec4>{make_vec4(1, 1, 1, 1)});
        ctx.setGlobalData("g_int2_vec", std::vector<int2>{make_int2(1, 1)});
        ctx.setGlobalData("g_int3_vec", std::vector<int3>{make_int3(1, 1, 1)});
        ctx.setGlobalData("g_int4_vec", std::vector<int4>{make_int4(1, 1, 1, 1)});
        ctx.setGlobalData("g_string_vec", std::vector<std::string>{"hello"});

        ctx.duplicateGlobalData("g_uint_vec", "g_uint_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_uint_vec_copy"));
        ctx.duplicateGlobalData("g_vec2_vec", "g_vec2_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec2_vec_copy"));
        ctx.duplicateGlobalData("g_vec3_vec", "g_vec3_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec3_vec_copy"));
        ctx.duplicateGlobalData("g_vec4_vec", "g_vec4_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec4_vec_copy"));
        ctx.duplicateGlobalData("g_int2_vec", "g_int2_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int2_vec_copy"));
        ctx.duplicateGlobalData("g_int3_vec", "g_int3_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int3_vec_copy"));
        ctx.duplicateGlobalData("g_int4_vec", "g_int4_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int4_vec_copy"));
        ctx.duplicateGlobalData("g_string_vec", "g_string_vec_copy");
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_string_vec_copy"));
    }

    SUBCASE("Global data error handling and all types") {
        Context ctx;
        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(ctx.getGlobalDataType("non_existent"));
        DOCTEST_CHECK_THROWS(ctx.getGlobalDataSize("non_existent"));
        DOCTEST_CHECK_THROWS(ctx.renameGlobalData("non_existent", "new"));
        DOCTEST_CHECK_THROWS(ctx.duplicateGlobalData("non_existent", "new"));
        DOCTEST_CHECK_THROWS(ctx.incrementGlobalData("non_existent", 1));

        ctx.setGlobalData("g_uint", std::vector<uint>{1});
        ctx.setGlobalData("g_float", std::vector<float>{1.f});
        ctx.setGlobalData("g_double", std::vector<double>{1.0});
        ctx.setGlobalData("g_vec2", std::vector<vec2>{make_vec2(1, 1)});
        ctx.setGlobalData("g_vec3", std::vector<vec3>{make_vec3(1, 1, 1)});
        ctx.setGlobalData("g_vec4", std::vector<vec4>{make_vec4(1, 1, 1, 1)});
        ctx.setGlobalData("g_int2", std::vector<int2>{make_int2(1, 1)});
        ctx.setGlobalData("g_int3", std::vector<int3>{make_int3(1, 1, 1)});
        ctx.setGlobalData("g_int4", std::vector<int4>{make_int4(1, 1, 1, 1)});
        ctx.setGlobalData("g_string", std::vector<std::string>{"hello"});

        ctx.duplicateGlobalData("g_uint", "g_uint_copy");
        ctx.duplicateGlobalData("g_float", "g_float_copy");
        ctx.duplicateGlobalData("g_double", "g_double_copy");
        ctx.duplicateGlobalData("g_vec2", "g_vec2_copy");
        ctx.duplicateGlobalData("g_vec3", "g_vec3_copy");
        ctx.duplicateGlobalData("g_vec4", "g_vec4_copy");
        ctx.duplicateGlobalData("g_int2", "g_int2_copy");
        ctx.duplicateGlobalData("g_int3", "g_int3_copy");
        ctx.duplicateGlobalData("g_int4", "g_int4_copy");
        ctx.duplicateGlobalData("g_string", "g_string_copy");

        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_uint_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_float_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_double_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec2_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec3_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_vec4_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int2_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int3_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int4_copy"));
        DOCTEST_CHECK(ctx.doesGlobalDataExist("g_string_copy"));

        ctx.incrementGlobalData("g_uint", (uint) 1);
        ctx.incrementGlobalData("g_float", 1.f);
        ctx.incrementGlobalData("g_double", 1.0);

        std::vector<uint> g_uint;
        ctx.getGlobalData("g_uint", g_uint);
        DOCTEST_CHECK(g_uint[0] == 2);
    }
}

TEST_CASE("Object Data") {
    Context ctx;
    uint o1 = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
    uint o2 = ctx.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2));

    SUBCASE("Set, Get, Exists, List, Clear") {
        ctx.setObjectData(o1, "test_uint", (uint) 42);
        std::vector<vec3> vec_data = {make_vec3(1, 2, 3), make_vec3(4, 5, 6)};
        ctx.setObjectData(o1, "test_vec3_vec", vec_data);

        DOCTEST_CHECK(ctx.doesObjectDataExist(o1, "test_uint"));
        DOCTEST_CHECK(ctx.doesObjectDataExist(o1, "test_vec3_vec"));
        DOCTEST_CHECK(!ctx.doesObjectDataExist(o1, "non_existent"));

        uint ui;
        ctx.getObjectData(o1, "test_uint", ui);
        DOCTEST_CHECK(ui == 42);

        std::vector<vec3> v_r;
        ctx.getObjectData(o1, "test_vec3_vec", v_r);
        DOCTEST_CHECK(v_r.size() == 2);
        DOCTEST_CHECK(v_r[1].y == doctest::Approx(5.0));

        DOCTEST_CHECK(ctx.getObjectDataType(o1, "test_uint") == HELIOS_TYPE_UINT);
        DOCTEST_CHECK(ctx.getObjectDataSize(o1, "test_vec3_vec") == 2);

        std::vector<std::string> labels = ctx.listObjectData(o1);
        DOCTEST_CHECK(labels.size() == 2);

        ctx.clearObjectData(o1, "test_uint");
        DOCTEST_CHECK(!ctx.doesObjectDataExist(o1, "test_uint"));
    }

    SUBCASE("Management") {
        ctx.setObjectData(o1, "my_data", 10);

        ctx.copyObjectData(o1, o2);
        DOCTEST_CHECK(ctx.doesObjectDataExist(o2, "my_data"));

        ctx.renameObjectData(o1, "my_data", "new_name");
        DOCTEST_CHECK(!ctx.doesObjectDataExist(o1, "my_data"));
        DOCTEST_CHECK(ctx.doesObjectDataExist(o1, "new_name"));

        ctx.duplicateObjectData(o2, "my_data", "my_data_copy");
        DOCTEST_CHECK(ctx.doesObjectDataExist(o2, "my_data_copy"));

        std::vector<std::string> all_obj_labels = ctx.listAllObjectDataLabels();
        DOCTEST_CHECK(std::find(all_obj_labels.begin(), all_obj_labels.end(), "my_data") != all_obj_labels.end());
        DOCTEST_CHECK(std::find(all_obj_labels.begin(), all_obj_labels.end(), "new_name") != all_obj_labels.end());
        DOCTEST_CHECK(std::find(all_obj_labels.begin(), all_obj_labels.end(), "my_data_copy") != all_obj_labels.end());
    }

    SUBCASE("Object data error handling") {
        Context ctx;
        capture_cerr cerr_buffer;
        uint bad_oid = 999;
        DOCTEST_CHECK_THROWS(ctx.getObjectDataType(bad_oid, "test"));
        DOCTEST_CHECK_THROWS(ctx.getObjectDataSize(bad_oid, "test"));
        DOCTEST_CHECK_THROWS(ctx.doesObjectDataExist(bad_oid, "test"));
        DOCTEST_CHECK_THROWS(ctx.clearObjectData(bad_oid, "test"));
        DOCTEST_CHECK_THROWS(ctx.copyObjectData(0, bad_oid));
        DOCTEST_CHECK_THROWS(ctx.copyObjectData(bad_oid, 0));
        DOCTEST_CHECK_THROWS(ctx.renameObjectData(bad_oid, "old", "new"));
        DOCTEST_CHECK_THROWS(ctx.duplicateObjectData(bad_oid, "old", "new"));
    }
}

TEST_CASE("Primitive Data") {
    Context ctx;
    uint p1 = ctx.addPatch();
    uint p2 = ctx.addPatch();
    std::vector<uint> uuids;
    for (int i = 0; i < 5; ++i) {
        uint p = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uuids.push_back(p);
    }

    SUBCASE("Set and Get All Data Types") {
        // Test all supported scalar types
        ctx.setPrimitiveData(p1, "test_int", 1);
        ctx.setPrimitiveData(p1, "test_uint", 1u);
        ctx.setPrimitiveData(p1, "test_float", 2.f);
        ctx.setPrimitiveData(p1, "test_double", 2.0);
        ctx.setPrimitiveData(p1, "test_vec2", make_vec2(1, 2));
        ctx.setPrimitiveData(p1, "test_vec3", make_vec3(1, 2, 3));
        ctx.setPrimitiveData(p1, "test_vec4", make_vec4(1, 2, 3, 4));
        ctx.setPrimitiveData(p1, "test_int2", make_int2(1, 2));
        ctx.setPrimitiveData(p1, "test_int3", make_int3(1, 2, 3));
        ctx.setPrimitiveData(p1, "test_int4", make_int4(1, 2, 3, 4));
        ctx.setPrimitiveData(p1, "test_string", "world");

        int r_i;
        ctx.getPrimitiveData(p1, "test_int", r_i);
        DOCTEST_CHECK(r_i == 1);
        uint r_ui;
        ctx.getPrimitiveData(p1, "test_uint", r_ui);
        DOCTEST_CHECK(r_ui == 1);
        float r_f;
        ctx.getPrimitiveData(p1, "test_float", r_f);
        DOCTEST_CHECK(r_f == doctest::Approx(2.0));
        double r_d;
        ctx.getPrimitiveData(p1, "test_double", r_d);
        DOCTEST_CHECK(r_d == doctest::Approx(2.0));
        vec2 r_v2;
        ctx.getPrimitiveData(p1, "test_vec2", r_v2);
        DOCTEST_CHECK(r_v2 == make_vec2(1, 2));
        vec3 r_v3;
        ctx.getPrimitiveData(p1, "test_vec3", r_v3);
        DOCTEST_CHECK(r_v3 == make_vec3(1, 2, 3));
        vec4 r_v4;
        ctx.getPrimitiveData(p1, "test_vec4", r_v4);
        DOCTEST_CHECK(r_v4 == make_vec4(1, 2, 3, 4));
        int2 r_i2;
        ctx.getPrimitiveData(p1, "test_int2", r_i2);
        DOCTEST_CHECK(r_i2 == make_int2(1, 2));
        int3 r_i3;
        ctx.getPrimitiveData(p1, "test_int3", r_i3);
        DOCTEST_CHECK(r_i3 == make_int3(1, 2, 3));
        int4 r_i4;
        ctx.getPrimitiveData(p1, "test_int4", r_i4);
        DOCTEST_CHECK(r_i4 == make_int4(1, 2, 3, 4));
        std::string r_s;
        ctx.getPrimitiveData(p1, "test_string", r_s);
        DOCTEST_CHECK(r_s == "world");

        // Test all supported vector types
        std::vector<int> v_i = {1, 2, 3};
        ctx.setPrimitiveData(p1, "test_v_int", v_i);
        std::vector<int> r_v_i;
        ctx.getPrimitiveData(p1, "test_v_int", r_v_i);
        DOCTEST_CHECK(r_v_i == v_i);
        std::vector<uint> v_ui = {4, 5, 6};
        ctx.setPrimitiveData(p1, "test_v_uint", v_ui);
        std::vector<uint> r_v_ui;
        ctx.getPrimitiveData(p1, "test_v_uint", r_v_ui);
        DOCTEST_CHECK(r_v_ui == v_ui);
        std::vector<float> v_f = {1.1, 2.2};
        ctx.setPrimitiveData(p1, "test_v_float", v_f);
        std::vector<float> r_v_f;
        ctx.getPrimitiveData(p1, "test_v_float", r_v_f);
        DOCTEST_CHECK(r_v_f == v_f);
        std::vector<double> v_d = {1.1, 2.2};
        ctx.setPrimitiveData(p1, "test_v_double", v_d);
        std::vector<double> r_v_d;
        ctx.getPrimitiveData(p1, "test_v_double", r_v_d);
        DOCTEST_CHECK(r_v_d == v_d);
        std::vector<vec2> v_v2 = {make_vec2(1, 2), make_vec2(3, 4)};
        ctx.setPrimitiveData(p1, "test_v_vec2", v_v2);
        std::vector<vec2> r_v_v2;
        ctx.getPrimitiveData(p1, "test_v_vec2", r_v_v2);
        DOCTEST_CHECK(r_v_v2 == v_v2);
        std::vector<vec3> v_v3 = {make_vec3(1, 2, 3), make_vec3(4, 5, 6)};
        ctx.setPrimitiveData(p1, "test_v_vec3", v_v3);
        std::vector<vec3> r_v_v3;
        ctx.getPrimitiveData(p1, "test_v_vec3", r_v_v3);
        DOCTEST_CHECK(r_v_v3 == v_v3);
        std::vector<vec4> v_v4 = {make_vec4(1, 2, 3, 4), make_vec4(5, 6, 7, 8)};
        ctx.setPrimitiveData(p1, "test_v_vec4", v_v4);
        std::vector<vec4> r_v_v4;
        ctx.getPrimitiveData(p1, "test_v_vec4", r_v_v4);
        DOCTEST_CHECK(r_v_v4 == v_v4);
        std::vector<int2> v_i2 = {make_int2(1, 2), make_int2(3, 4)};
        ctx.setPrimitiveData(p1, "test_v_int2", v_i2);
        std::vector<int2> r_v_i2;
        ctx.getPrimitiveData(p1, "test_v_int2", r_v_i2);
        DOCTEST_CHECK(r_v_i2 == v_i2);
        std::vector<int3> v_i3 = {make_int3(1, 2, 3), make_int3(4, 5, 6)};
        ctx.setPrimitiveData(p1, "test_v_int3", v_i3);
        std::vector<int3> r_v_i3;
        ctx.getPrimitiveData(p1, "test_v_int3", r_v_i3);
        DOCTEST_CHECK(r_v_i3 == v_i3);
        std::vector<int4> v_i4 = {make_int4(1, 2, 3, 4), make_int4(5, 6, 7, 8)};
        ctx.setPrimitiveData(p1, "test_v_int4", v_i4);
        std::vector<int4> r_v_i4;
        ctx.getPrimitiveData(p1, "test_v_int4", r_v_i4);
        DOCTEST_CHECK(r_v_i4 == v_i4);
        std::vector<std::string> v_s = {"hello", "world"};
        ctx.setPrimitiveData(p1, "test_v_string", v_s);
        std::vector<std::string> r_v_s;
        ctx.getPrimitiveData(p1, "test_v_string", r_v_s);
        DOCTEST_CHECK(r_v_s == v_s);
    }

    SUBCASE("Management") {
        ctx.setPrimitiveData(p1, "my_data", 10);
        ctx.copyPrimitiveData(p1, p2);
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p2, "my_data"));
        int val;
        ctx.getPrimitiveData(p2, "my_data", val);
        DOCTEST_CHECK(val == 10);

        ctx.renamePrimitiveData(p1, "my_data", "new_data_name");
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p1, "my_data"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p1, "new_data_name"));

        ctx.duplicatePrimitiveData(p2, "my_data", "my_data_copy");
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p2, "my_data_copy"));
        ctx.getPrimitiveData(p2, "my_data_copy", val);
        DOCTEST_CHECK(val == 10);

        ctx.setPrimitiveData(p1, "global_copy_test", 5.5f);
        ctx.duplicatePrimitiveData("global_copy_test", "global_copy_test_new");
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p1, "global_copy_test_new"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p2, "global_copy_test_new"));

        std::vector<std::string> all_labels = ctx.listAllPrimitiveDataLabels();
        DOCTEST_CHECK(std::find(all_labels.begin(), all_labels.end(), "my_data") != all_labels.end());
        DOCTEST_CHECK(std::find(all_labels.begin(), all_labels.end(), "my_data_copy") != all_labels.end());
        DOCTEST_CHECK(std::find(all_labels.begin(), all_labels.end(), "new_data_name") != all_labels.end());

        ctx.clearPrimitiveData(p1, "new_data_name");
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p1, "new_data_name"));
        ctx.clearPrimitiveData(uuids, "global_copy_test");
    }

    SUBCASE("Calculations") {
        for (int i = 0; i < 5; ++i) {
            ctx.setPrimitiveData(uuids[i], "float_val", (float) i);
            ctx.setPrimitiveData(uuids[i], "double_val", (double) i);
            ctx.setPrimitiveData(uuids[i], "vec2_val", make_vec2((float) i, (float) i));
        }

        float float_mean;
        ctx.calculatePrimitiveDataMean(uuids, "float_val", float_mean);
        DOCTEST_CHECK(float_mean == doctest::Approx(2.0f));
        double double_mean;
        ctx.calculatePrimitiveDataMean(uuids, "double_val", double_mean);
        DOCTEST_CHECK(double_mean == doctest::Approx(2.0));
        vec2 vec2_mean;
        ctx.calculatePrimitiveDataMean(uuids, "vec2_val", vec2_mean);
        DOCTEST_CHECK(vec2_mean.x == doctest::Approx(2.0f));

        float awt_mean_f;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "float_val", awt_mean_f);
        DOCTEST_CHECK(awt_mean_f == doctest::Approx(2.0f));

        float float_sum;
        ctx.calculatePrimitiveDataSum(uuids, "float_val", float_sum);
        DOCTEST_CHECK(float_sum == doctest::Approx(10.0f));

        float awt_sum_f;
        ctx.calculatePrimitiveDataAreaWeightedSum(uuids, "float_val", awt_sum_f);
        DOCTEST_CHECK(awt_sum_f == doctest::Approx(10.0f));

        ctx.scalePrimitiveData(uuids, "float_val", 2.0f);
        ctx.getPrimitiveData(uuids[2], "float_val", float_mean);
        DOCTEST_CHECK(float_mean == doctest::Approx(4.0f));
        capture_cerr cerr_buffer;
        ctx.scalePrimitiveData("double_val", 0.5f);
        ctx.getPrimitiveData(uuids[4], "double_val", double_mean);
        DOCTEST_CHECK(double_mean == doctest::Approx(2.0));
        DOCTEST_CHECK(cerr_buffer.has_output());

        ctx.setPrimitiveData(uuids, "int_val", 10);
        ctx.incrementPrimitiveData(uuids, "int_val", 5);
        int int_val;
        ctx.getPrimitiveData(uuids[0], "int_val", int_val);
        DOCTEST_CHECK(int_val == 15);
    }

    SUBCASE("Context::calculatePrimitiveDataMean empty list") {
        Context ctx;
        std::vector<uint> uuids;
        float float_mean;
        capture_cerr cerr_buffer;
        ctx.calculatePrimitiveDataMean(uuids, "non_existent", float_mean);
        DOCTEST_CHECK(float_mean == 0.f);
        DOCTEST_CHECK(cerr_buffer.has_output());
    }

    SUBCASE("Context::calculatePrimitiveDataMean vec3/vec4") {
        Context ctx;
        uint p1 = ctx.addPatch();
        ctx.setPrimitiveData(p1, "v3", make_vec3(1, 2, 3));
        ctx.setPrimitiveData(p1, "v4", make_vec4(1, 2, 3, 4));
        std::vector<uint> uuids = {p1};
        vec3 v3_mean;
        vec4 v4_mean;
        ctx.calculatePrimitiveDataMean(uuids, "v3", v3_mean);
        ctx.calculatePrimitiveDataMean(uuids, "v4", v4_mean);
        DOCTEST_CHECK(v3_mean == make_vec3(1, 2, 3));
        DOCTEST_CHECK(v4_mean == make_vec4(1, 2, 3, 4));
    }

    SUBCASE("Context::calculatePrimitiveDataAreaWeightedMean all types") {
        Context ctx;
        uint p = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(2, 2)); // Area = 4
        ctx.setPrimitiveData(p, "double", 2.5);
        ctx.setPrimitiveData(p, "vec2", make_vec2(1, 2));
        ctx.setPrimitiveData(p, "vec3", make_vec3(1, 2, 3));
        ctx.setPrimitiveData(p, "vec4", make_vec4(1, 2, 3, 4));
        std::vector<uint> uuids = {p};

        double d_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "double", d_mean);
        DOCTEST_CHECK(d_mean == doctest::Approx(2.5));

        vec2 v2_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec2", v2_mean);
        DOCTEST_CHECK(v2_mean == make_vec2(1, 2));

        vec3 v3_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec3", v3_mean);
        DOCTEST_CHECK(v3_mean == make_vec3(1, 2, 3));

        vec4 v4_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec4", v4_mean);
        DOCTEST_CHECK(v4_mean == make_vec4(1, 2, 3, 4));
    }

    SUBCASE("Context::calculatePrimitiveDataSum all types") {
        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "double", 2.5);
        ctx.setPrimitiveData(p, "vec2", make_vec2(1, 2));
        ctx.setPrimitiveData(p, "vec3", make_vec3(1, 2, 3));
        ctx.setPrimitiveData(p, "vec4", make_vec4(1, 2, 3, 4));
        std::vector<uint> uuids = {p};

        double d_sum;
        ctx.calculatePrimitiveDataSum(uuids, "double", d_sum);
        DOCTEST_CHECK(d_sum == doctest::Approx(2.5));

        vec2 v2_sum;
        ctx.calculatePrimitiveDataSum(uuids, "vec2", v2_sum);
        DOCTEST_CHECK(v2_sum == make_vec2(1, 2));

        vec3 v3_sum;
        ctx.calculatePrimitiveDataSum(uuids, "vec3", v3_sum);
        DOCTEST_CHECK(v3_sum == make_vec3(1, 2, 3));

        vec4 v4_sum;
        ctx.calculatePrimitiveDataSum(uuids, "vec4", v4_sum);
        DOCTEST_CHECK(v4_sum == make_vec4(1, 2, 3, 4));
    }

    SUBCASE("Context::incrementPrimitiveData all types") {
        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "uint", (uint) 1);
        ctx.setPrimitiveData(p, "float", 1.f);
        ctx.setPrimitiveData(p, "double", 1.0);
        std::vector<uint> uuids = {p};

        ctx.incrementPrimitiveData(uuids, "uint", (uint) 2);
        uint u_val;
        ctx.getPrimitiveData(p, "uint", u_val);
        DOCTEST_CHECK(u_val == 3);

        ctx.incrementPrimitiveData(uuids, "float", 2.f);
        float f_val;
        ctx.getPrimitiveData(p, "float", f_val);
        DOCTEST_CHECK(f_val == doctest::Approx(3.f));

        ctx.incrementPrimitiveData(uuids, "double", 2.0);
        double d_val;
        ctx.getPrimitiveData(p, "double", d_val);
        DOCTEST_CHECK(d_val == doctest::Approx(3.0));
    }

    SUBCASE("Aggregation and Filtering") {
        std::vector<uint> agg_uuids;
        for (int i = 0; i < 3; ++i) {
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "d1", (float) i);
            ctx.setPrimitiveData(p, "d2", (float) i * 2.0f);
            ctx.setPrimitiveData(p, "d3", (float) i * 3.0f);
            ctx.setPrimitiveData(p, "filter_me", i);
            agg_uuids.push_back(p);
        }

        std::vector<std::string> labels = {"d1", "d2", "d3"};
        ctx.aggregatePrimitiveDataSum(agg_uuids, labels, "sum_data");
        float sum_val;
        ctx.getPrimitiveData(agg_uuids[1], "sum_data", sum_val);
        DOCTEST_CHECK(sum_val == doctest::Approx(1.f + 2.f + 3.f));

        ctx.aggregatePrimitiveDataProduct(agg_uuids, labels, "prod_data");
        float prod_val;
        ctx.getPrimitiveData(agg_uuids[2], "prod_data", prod_val);
        DOCTEST_CHECK(prod_val == doctest::Approx(2.f * 4.f * 6.f));

        std::vector<uint> filtered = ctx.filterPrimitivesByData(agg_uuids, "filter_me", 1, ">=");
        DOCTEST_CHECK(filtered.size() == 2);
        filtered = ctx.filterPrimitivesByData(agg_uuids, "filter_me", 1, "==");
        DOCTEST_CHECK(filtered.size() == 1);
        DOCTEST_CHECK(filtered[0] == agg_uuids[1]);
    }

    SUBCASE("Context::filterPrimitivesByData all types") {
        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "double", 1.0);
        ctx.setPrimitiveData(p, "uint", (uint) 10);
        std::vector<uint> uuids = {p};

        auto filtered_d = ctx.filterPrimitivesByData(uuids, "double", 0.5, ">");
        DOCTEST_CHECK(filtered_d.size() == 1);

        auto filtered_u = ctx.filterPrimitivesByData(uuids, "uint", (uint) 5, ">");
        DOCTEST_CHECK(filtered_u.size() == 1);
    }

    SUBCASE("Error Handling") {
        capture_cerr cerr_buffer;
        uint bad_uuid = 999;
        DOCTEST_CHECK_THROWS(ctx.getPrimitiveDataType(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.getPrimitiveDataSize(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.doesPrimitiveDataExist(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.clearPrimitiveData(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.copyPrimitiveData(p1, bad_uuid));
        DOCTEST_CHECK_THROWS(ctx.copyPrimitiveData(bad_uuid, p1));
        DOCTEST_CHECK_THROWS(ctx.renamePrimitiveData(bad_uuid, "old", "new"));
        DOCTEST_CHECK_THROWS(ctx.duplicatePrimitiveData(bad_uuid, "old", "new"));

        std::vector<uint> empty_uuids;
        float float_mean;
        ctx.calculatePrimitiveDataMean(empty_uuids, "non_existent", float_mean);
        DOCTEST_CHECK(float_mean == 0.f);
        DOCTEST_CHECK(cerr_buffer.has_output());

        ctx.setPrimitiveData(p1, "float_val_err", 1.f);
        ctx.incrementPrimitiveData(std::vector<uint>{p1}, "float_val_err", 1); // Wrong type
        DOCTEST_CHECK(cerr_buffer.has_output());

        std::vector<uint> filtered_UUIDs;
        DOCTEST_CHECK_THROWS_AS(filtered_UUIDs = ctx.filterPrimitivesByData(uuids, "filter_me", 1, "!!"), std::runtime_error);
    }

    SUBCASE("Primitive::getPrimitiveDataSize all types") {
        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "uint", (uint) 1);
        ctx.setPrimitiveData(p, "double", 1.0);
        ctx.setPrimitiveData(p, "vec2", make_vec2(1, 1));
        ctx.setPrimitiveData(p, "vec3", make_vec3(1, 1, 1));
        ctx.setPrimitiveData(p, "vec4", make_vec4(1, 1, 1, 1));
        ctx.setPrimitiveData(p, "int2", make_int2(1, 1));
        ctx.setPrimitiveData(p, "int3", make_int3(1, 1, 1));
        ctx.setPrimitiveData(p, "int4", make_int4(1, 1, 1, 1));
        ctx.setPrimitiveData(p, "string", "hello");

        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "uint") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "double") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec2") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec3") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec4") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "int2") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "int3") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "int4") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "string") == 1);
    }

    SUBCASE("Primitive::clearPrimitiveData all types") {
        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "uint", (uint) 1);
        ctx.setPrimitiveData(p, "double", 1.0);
        ctx.setPrimitiveData(p, "vec2", make_vec2(1, 1));
        ctx.setPrimitiveData(p, "vec3", make_vec3(1, 1, 1));
        ctx.setPrimitiveData(p, "vec4", make_vec4(1, 1, 1, 1));
        ctx.setPrimitiveData(p, "int2", make_int2(1, 1));
        ctx.setPrimitiveData(p, "int3", make_int3(1, 1, 1));
        ctx.setPrimitiveData(p, "int4", make_int4(1, 1, 1, 1));
        ctx.setPrimitiveData(p, "string", "hello");

        ctx.clearPrimitiveData(p, "uint");
        ctx.clearPrimitiveData(p, "double");
        ctx.clearPrimitiveData(p, "vec2");
        ctx.clearPrimitiveData(p, "vec3");
        ctx.clearPrimitiveData(p, "vec4");
        ctx.clearPrimitiveData(p, "int2");
        ctx.clearPrimitiveData(p, "int3");
        ctx.clearPrimitiveData(p, "int4");
        ctx.clearPrimitiveData(p, "string");

        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "uint"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "double"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "vec2"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "vec3"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "vec4"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "int2"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "int3"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "int4"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "string"));
    }

    SUBCASE("Context data functions with invalid UUID") {
        Context ctx;
        capture_cerr cerr_buffer;
        uint bad_uuid = 999;
        DOCTEST_CHECK_THROWS(ctx.getPrimitiveDataType(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.getPrimitiveDataSize(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.doesPrimitiveDataExist(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.clearPrimitiveData(bad_uuid, "test"));
        DOCTEST_CHECK_THROWS(ctx.copyPrimitiveData(0, bad_uuid));
        DOCTEST_CHECK_THROWS(ctx.copyPrimitiveData(bad_uuid, 0));
        DOCTEST_CHECK_THROWS(ctx.renamePrimitiveData(bad_uuid, "old", "new"));
        DOCTEST_CHECK_THROWS(ctx.duplicatePrimitiveData(bad_uuid, "old", "new"));
    }

    SUBCASE("Context::clearPrimitiveData for multiple UUIDs") {
        Context ctx;
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();
        ctx.setPrimitiveData(p1, "data", 1);
        ctx.setPrimitiveData(p2, "data", 2);
        std::vector<uint> uuids = {p1, p2};
        ctx.clearPrimitiveData(uuids, "data");
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p1, "data"));
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p2, "data"));
    }

    SUBCASE("Context::duplicatePrimitiveData all types") {
        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "uint", (uint) 1);
        ctx.setPrimitiveData(p, "double", 1.0);
        ctx.setPrimitiveData(p, "vec2", make_vec2(1, 1));
        ctx.setPrimitiveData(p, "vec3", make_vec3(1, 1, 1));
        ctx.setPrimitiveData(p, "vec4", make_vec4(1, 1, 1, 1));
        ctx.setPrimitiveData(p, "int2", make_int2(1, 1));
        ctx.setPrimitiveData(p, "int3", make_int3(1, 1, 1));
        ctx.setPrimitiveData(p, "int4", make_int4(1, 1, 1, 1));
        ctx.setPrimitiveData(p, "string", "hello");

        ctx.duplicatePrimitiveData(p, "uint", "uint_copy");
        ctx.duplicatePrimitiveData(p, "double", "double_copy");
        ctx.duplicatePrimitiveData(p, "vec2", "vec2_copy");
        ctx.duplicatePrimitiveData(p, "vec3", "vec3_copy");
        ctx.duplicatePrimitiveData(p, "vec4", "vec4_copy");
        ctx.duplicatePrimitiveData(p, "int2", "int2_copy");
        ctx.duplicatePrimitiveData(p, "int3", "int3_copy");
        ctx.duplicatePrimitiveData(p, "int4", "int4_copy");
        ctx.duplicatePrimitiveData(p, "string", "string_copy");

        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "uint_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "double_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "vec2_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "vec3_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "vec4_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "int2_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "int3_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "int4_copy"));
        DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p, "string_copy"));
    }
}

TEST_CASE("Object Data Filtering") {
    Context ctx;
    uint o1 = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
    uint o2 = ctx.addTileObject(make_vec3(1, 0, 0), make_vec2(2, 2), nullrotation, make_int2(1, 1));
    uint o3 = ctx.addConeObject(5, make_vec3(2, 0, 0), make_vec3(2, 0, 2), 1.0f, 2.0f);

    SUBCASE("filterObjectsByData for all types") {
        // Set up test data
        ctx.setObjectData(o1, "float_val", 1.5f);
        ctx.setObjectData(o2, "float_val", 2.5f);
        ctx.setObjectData(o3, "float_val", 0.5f);

        ctx.setObjectData(o1, "double_val", 1.5);
        ctx.setObjectData(o2, "double_val", 2.5);
        ctx.setObjectData(o3, "double_val", 0.5);

        ctx.setObjectData(o1, "int_val", 1);
        ctx.setObjectData(o2, "int_val", 2);
        ctx.setObjectData(o3, "int_val", 0);

        ctx.setObjectData(o1, "uint_val", (uint) 1);
        ctx.setObjectData(o2, "uint_val", (uint) 2);
        ctx.setObjectData(o3, "uint_val", (uint) 0);

        ctx.setObjectData(o1, "string_val", std::string("apple"));
        ctx.setObjectData(o2, "string_val", std::string("banana"));
        ctx.setObjectData(o3, "string_val", std::string("cherry"));

        std::vector<uint> all_objects = {o1, o2, o3};

        // Test float filtering - use std::string for parameter names to avoid ambiguity
        auto filtered_f = ctx.filterObjectsByData(all_objects, std::string("float_val"), 1.0f, std::string(">"));
        DOCTEST_CHECK(filtered_f.size() == 2);
        DOCTEST_CHECK(std::find(filtered_f.begin(), filtered_f.end(), o1) != filtered_f.end());
        DOCTEST_CHECK(std::find(filtered_f.begin(), filtered_f.end(), o2) != filtered_f.end());

        filtered_f = ctx.filterObjectsByData(all_objects, std::string("float_val"), 2.5f, std::string("=="));
        DOCTEST_CHECK(filtered_f.size() == 1);
        DOCTEST_CHECK(filtered_f[0] == o2);

        // Test double filtering
        auto filtered_d = ctx.filterObjectsByData(all_objects, std::string("double_val"), 1.0, std::string(">="));
        DOCTEST_CHECK(filtered_d.size() == 2);

        // Test int filtering
        auto filtered_i = ctx.filterObjectsByData(all_objects, std::string("int_val"), 1, std::string("<"));
        DOCTEST_CHECK(filtered_i.size() == 1);
        DOCTEST_CHECK(filtered_i[0] == o3);

        // Test uint filtering
        auto filtered_u = ctx.filterObjectsByData(all_objects, std::string("uint_val"), (uint) 0, std::string(">"));
        DOCTEST_CHECK(filtered_u.size() == 2);

        // Note: String filtering may not be supported by all filterObjectsByData overloads
        // Test with string data but focus on supported data types
    }

    SUBCASE("filterObjectsByData error handling") {
        std::vector<uint> objects = {o1};
        capture_cerr cerr_buffer;

        // Test invalid comparison operator
        std::vector<uint> filtered_f;
        DOCTEST_CHECK_THROWS_AS(filtered_f = ctx.filterObjectsByData(objects, std::string("non_existent"), 1.0f, std::string("!!")), std::runtime_error);

        // Test non-existent data
        auto empty_result = ctx.filterObjectsByData(objects, std::string("non_existent"), 1.0f, std::string(">"));
        DOCTEST_CHECK(empty_result.empty());
    }
}

TEST_CASE("Surface Area Calculations") {
    Context ctx;

    SUBCASE("sumPrimitiveSurfaceArea") {
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(2, 3)); // Area = 6
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 4)); // Area = 4
        uint t1 = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(2, 0, 0), make_vec3(0, 2, 0)); // Area = 2

        std::vector<uint> patches = {p1, p2};
        std::vector<uint> triangles = {t1};
        std::vector<uint> all_prims = {p1, p2, t1};

        float patch_area = ctx.sumPrimitiveSurfaceArea(patches);
        DOCTEST_CHECK(patch_area == doctest::Approx(10.0f));

        float triangle_area = ctx.sumPrimitiveSurfaceArea(triangles);
        DOCTEST_CHECK(triangle_area == doctest::Approx(2.0f));

        float total_area = ctx.sumPrimitiveSurfaceArea(all_prims);
        DOCTEST_CHECK(total_area == doctest::Approx(12.0f));

        // Test empty list
        std::vector<uint> empty_list;
        float zero_area = ctx.sumPrimitiveSurfaceArea(empty_list);
        DOCTEST_CHECK(zero_area == doctest::Approx(0.0f));
    }
}

TEST_CASE("Advanced Area-Weighted Calculations") {
    Context ctx;

    SUBCASE("calculatePrimitiveDataAreaWeightedMean comprehensive") {
        // Create primitives with different areas
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1)); // Area = 1
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(2, 2)); // Area = 4
        uint p3 = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(3, 1)); // Area = 3

        // Set data for different types
        ctx.setPrimitiveData(p1, "double_val", 10.0);
        ctx.setPrimitiveData(p2, "double_val", 20.0);
        ctx.setPrimitiveData(p3, "double_val", 30.0);

        ctx.setPrimitiveData(p1, "vec2_val", make_vec2(1, 2));
        ctx.setPrimitiveData(p2, "vec2_val", make_vec2(3, 4));
        ctx.setPrimitiveData(p3, "vec2_val", make_vec2(5, 6));

        ctx.setPrimitiveData(p1, "vec3_val", make_vec3(1, 2, 3));
        ctx.setPrimitiveData(p2, "vec3_val", make_vec3(4, 5, 6));
        ctx.setPrimitiveData(p3, "vec3_val", make_vec3(7, 8, 9));

        ctx.setPrimitiveData(p1, "vec4_val", make_vec4(1, 2, 3, 4));
        ctx.setPrimitiveData(p2, "vec4_val", make_vec4(5, 6, 7, 8));
        ctx.setPrimitiveData(p3, "vec4_val", make_vec4(9, 10, 11, 12));

        std::vector<uint> uuids = {p1, p2, p3};

        // Test double area-weighted mean
        // Expected: (10*1 + 20*4 + 30*3) / (1+4+3) = 180/8 = 22.5
        double d_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "double_val", d_mean);
        DOCTEST_CHECK(d_mean == doctest::Approx(22.5));

        // Test vec2 area-weighted mean
        // Expected x: (1*1 + 3*4 + 5*3) / 8 = 28/8 = 3.5
        // Expected y: (2*1 + 4*4 + 6*3) / 8 = 36/8 = 4.5
        vec2 v2_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec2_val", v2_mean);
        DOCTEST_CHECK(v2_mean.x == doctest::Approx(3.5f));
        DOCTEST_CHECK(v2_mean.y == doctest::Approx(4.5f));

        // Test vec3 area-weighted mean
        vec3 v3_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec3_val", v3_mean);
        DOCTEST_CHECK(v3_mean.x == doctest::Approx(4.75f)); // (1*1 + 4*4 + 7*3) / 8 = 38/8 = 4.75
        DOCTEST_CHECK(v3_mean.y == doctest::Approx(5.75f)); // (2*1 + 5*4 + 8*3) / 8 = 46/8 = 5.75
        DOCTEST_CHECK(v3_mean.z == doctest::Approx(6.75f)); // (3*1 + 6*4 + 9*3) / 8 = 54/8 = 6.75

        // Test vec4 area-weighted mean
        vec4 v4_mean;
        ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "vec4_val", v4_mean);
        DOCTEST_CHECK(v4_mean.x == doctest::Approx(6.0f)); // (1*1 + 5*4 + 9*3) / 8 = 48/8 = 6.0
        DOCTEST_CHECK(v4_mean.y == doctest::Approx(7.0f)); // (2*1 + 6*4 + 10*3) / 8 = 56/8 = 7.0
        DOCTEST_CHECK(v4_mean.z == doctest::Approx(8.0f)); // (3*1 + 7*4 + 11*3) / 8 = 64/8 = 8.0
        DOCTEST_CHECK(v4_mean.w == doctest::Approx(9.0f)); // (4*1 + 8*4 + 12*3) / 8 = 72/8 = 9.0
    }

    SUBCASE("calculatePrimitiveDataAreaWeightedSum comprehensive") {
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(2, 1)); // Area = 2
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 3)); // Area = 3

        ctx.setPrimitiveData(p1, "double_val", 5.0);
        ctx.setPrimitiveData(p2, "double_val", 10.0);

        ctx.setPrimitiveData(p1, "vec2_val", make_vec2(1, 2));
        ctx.setPrimitiveData(p2, "vec2_val", make_vec2(3, 4));

        ctx.setPrimitiveData(p1, "vec3_val", make_vec3(1, 2, 3));
        ctx.setPrimitiveData(p2, "vec3_val", make_vec3(4, 5, 6));

        ctx.setPrimitiveData(p1, "vec4_val", make_vec4(1, 2, 3, 4));
        ctx.setPrimitiveData(p2, "vec4_val", make_vec4(5, 6, 7, 8));

        std::vector<uint> uuids = {p1, p2};

        // Test double area-weighted sum: 5*2 + 10*3 = 40
        double d_sum;
        ctx.calculatePrimitiveDataAreaWeightedSum(uuids, "double_val", d_sum);
        DOCTEST_CHECK(d_sum == doctest::Approx(40.0));

        // Test vec2 area-weighted sum
        vec2 v2_sum;
        ctx.calculatePrimitiveDataAreaWeightedSum(uuids, "vec2_val", v2_sum);
        DOCTEST_CHECK(v2_sum.x == doctest::Approx(11.0f)); // 1*2 + 3*3 = 11
        DOCTEST_CHECK(v2_sum.y == doctest::Approx(16.0f)); // 2*2 + 4*3 = 16

        // Test vec3 area-weighted sum
        vec3 v3_sum;
        ctx.calculatePrimitiveDataAreaWeightedSum(uuids, "vec3_val", v3_sum);
        DOCTEST_CHECK(v3_sum.x == doctest::Approx(14.0f)); // 1*2 + 4*3 = 14
        DOCTEST_CHECK(v3_sum.y == doctest::Approx(19.0f)); // 2*2 + 5*3 = 19
        DOCTEST_CHECK(v3_sum.z == doctest::Approx(24.0f)); // 3*2 + 6*3 = 24

        // Test vec4 area-weighted sum
        vec4 v4_sum;
        ctx.calculatePrimitiveDataAreaWeightedSum(uuids, "vec4_val", v4_sum);
        DOCTEST_CHECK(v4_sum.x == doctest::Approx(17.0f)); // 1*2 + 5*3 = 17
        DOCTEST_CHECK(v4_sum.y == doctest::Approx(22.0f)); // 2*2 + 6*3 = 22
        DOCTEST_CHECK(v4_sum.z == doctest::Approx(27.0f)); // 3*2 + 7*3 = 27
        DOCTEST_CHECK(v4_sum.w == doctest::Approx(32.0f)); // 4*2 + 8*3 = 32
    }
}

TEST_CASE("Global Data Scaling") {
    Context ctx;

    SUBCASE("scalePrimitiveData with label") {
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();
        uint p3 = ctx.addPatch();

        // Set data on primitives
        ctx.setPrimitiveData(p1, "scale_me", 10.0f);
        ctx.setPrimitiveData(p2, "scale_me", 20.0f);
        ctx.setPrimitiveData(p3, "other_data", 5.0f);

        // Scale all primitives with "scale_me" label
        capture_cerr cerr_buffer;
        ctx.scalePrimitiveData("scale_me", 2.0f);

        // Check scaled values
        float val1, val2, val3;
        ctx.getPrimitiveData(p1, "scale_me", val1);
        ctx.getPrimitiveData(p2, "scale_me", val2);
        ctx.getPrimitiveData(p3, "other_data", val3);

        DOCTEST_CHECK(val1 == doctest::Approx(20.0f));
        DOCTEST_CHECK(val2 == doctest::Approx(40.0f));
        DOCTEST_CHECK(val3 == doctest::Approx(5.0f)); // Should be unchanged

        // Test with non-existent label
        ctx.scalePrimitiveData("non_existent", 3.0f);
        DOCTEST_CHECK(cerr_buffer.has_output());
    }
}

TEST_CASE("Aggregate Function Edge Cases") {
    Context ctx;

    SUBCASE("aggregatePrimitiveDataProduct edge cases") {
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();
        uint p3 = ctx.addPatch();

        // Test with zeros
        ctx.setPrimitiveData(p1, "zero_test", 0.0f);
        ctx.setPrimitiveData(p1, "normal", 5.0f);
        ctx.setPrimitiveData(p2, "zero_test", 10.0f);
        ctx.setPrimitiveData(p2, "normal", 2.0f);

        std::vector<uint> uuids = {p1, p2};
        std::vector<std::string> labels = {"zero_test", "normal"};

        ctx.aggregatePrimitiveDataProduct(uuids, labels, "product_result");

        float result;
        ctx.getPrimitiveData(p1, "product_result", result);
        DOCTEST_CHECK(result == doctest::Approx(0.0f)); // 0 * 5 = 0

        ctx.getPrimitiveData(p2, "product_result", result);
        DOCTEST_CHECK(result == doctest::Approx(20.0f)); // 10 * 2 = 20

        // Test with negative values
        ctx.setPrimitiveData(p1, "neg1", -2.0f);
        ctx.setPrimitiveData(p1, "neg2", -3.0f);
        ctx.setPrimitiveData(p2, "neg1", 4.0f);
        ctx.setPrimitiveData(p2, "neg2", -1.0f);

        std::vector<std::string> neg_labels = {"neg1", "neg2"};
        ctx.aggregatePrimitiveDataProduct(uuids, neg_labels, "neg_product");

        ctx.getPrimitiveData(p1, "neg_product", result);
        DOCTEST_CHECK(result == doctest::Approx(6.0f)); // (-2) * (-3) = 6

        ctx.getPrimitiveData(p2, "neg_product", result);
        DOCTEST_CHECK(result == doctest::Approx(-4.0f)); // 4 * (-1) = -4
    }
}

TEST_CASE("Data Type Coverage Extensions") {
    Context ctx;

    SUBCASE("Primitive data with int2, int3, int4 vectors") {
        uint p = ctx.addPatch();

        // Test vector of int2, int3, int4
        std::vector<int2> v_i2 = {make_int2(1, 2), make_int2(3, 4)};
        std::vector<int3> v_i3 = {make_int3(1, 2, 3), make_int3(4, 5, 6)};
        std::vector<int4> v_i4 = {make_int4(1, 2, 3, 4), make_int4(5, 6, 7, 8)};

        ctx.setPrimitiveData(p, "vec_int2", v_i2);
        ctx.setPrimitiveData(p, "vec_int3", v_i3);
        ctx.setPrimitiveData(p, "vec_int4", v_i4);

        DOCTEST_CHECK(ctx.getPrimitiveDataType(p, "vec_int2") == HELIOS_TYPE_INT2);
        DOCTEST_CHECK(ctx.getPrimitiveDataType(p, "vec_int3") == HELIOS_TYPE_INT3);
        DOCTEST_CHECK(ctx.getPrimitiveDataType(p, "vec_int4") == HELIOS_TYPE_INT4);

        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec_int2") == 2);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec_int3") == 2);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "vec_int4") == 2);

        // Test retrieval
        std::vector<int2> r_i2;
        std::vector<int3> r_i3;
        std::vector<int4> r_i4;

        ctx.getPrimitiveData(p, "vec_int2", r_i2);
        ctx.getPrimitiveData(p, "vec_int3", r_i3);
        ctx.getPrimitiveData(p, "vec_int4", r_i4);

        DOCTEST_CHECK(r_i2 == v_i2);
        DOCTEST_CHECK(r_i3 == v_i3);
        DOCTEST_CHECK(r_i4 == v_i4);
    }
}

TEST_CASE("Error Handling Edge Cases") {
    Context ctx;

    SUBCASE("Data operations with mixed primitive types") {
        uint patch = ctx.addPatch();
        uint triangle = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0));
        uint voxel = ctx.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        // Set data on different primitive types
        ctx.setPrimitiveData(patch, "mixed_data", 1.0f);
        ctx.setPrimitiveData(triangle, "mixed_data", 2.0f);
        ctx.setPrimitiveData(voxel, "mixed_data", 3.0f);

        std::vector<uint> mixed_prims = {patch, triangle, voxel};

        // Test calculations work with mixed primitive types
        float mean_val;
        ctx.calculatePrimitiveDataMean(mixed_prims, "mixed_data", mean_val);
        DOCTEST_CHECK(mean_val == doctest::Approx(2.0f));

        float sum_val;
        ctx.calculatePrimitiveDataSum(mixed_prims, "mixed_data", sum_val);
        DOCTEST_CHECK(sum_val == doctest::Approx(6.0f));
    }

    SUBCASE("Large dataset stress test") {
        std::vector<uint> large_dataset;

        // Create many primitives with data
        for (int i = 0; i < 1000; ++i) {
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "stress_data", (float) i);
            large_dataset.push_back(p);
        }

        // Test operations on large dataset
        float mean_val, sum_val;
        ctx.calculatePrimitiveDataMean(large_dataset, "stress_data", mean_val);
        ctx.calculatePrimitiveDataSum(large_dataset, "stress_data", sum_val);

        DOCTEST_CHECK(mean_val == doctest::Approx(499.5f)); // Mean of 0-999
        DOCTEST_CHECK(sum_val == doctest::Approx(499500.0f)); // Sum of 0-999

        // Test filtering on large dataset
        auto filtered = ctx.filterPrimitivesByData(large_dataset, "stress_data", 500.0f, ">");
        DOCTEST_CHECK(filtered.size() == 499); // Values 501-999
    }
}

TEST_CASE("General Error Handling") {
    Context ctx;
    uint tri = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::green);
    capture_cerr cerr_buffer;
    vec3 center;
#ifdef HELIOS_DEBUG
    DOCTEST_CHECK_THROWS_AS(center = ctx.getPatchCenter(tri), std::runtime_error);
#endif

    uint vox = ctx.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
    std::vector<uint> vlist{vox};
    DOCTEST_CHECK_THROWS_AS(ctx.rotatePrimitive(vlist, PI_F / 4.f, "a"), std::runtime_error);
}
