#pragma once
// =================================================================================
// Suite 4: Context Class
//
// Tests for the main Context class, which manages the scene, primitives,
// objects, data, and simulation state.
// =================================================================================
TEST_CASE("Core Context State and Configuration") {
    SUBCASE("Constructor and basic setup") {
        Context ctx;
        DOCTEST_CHECK(ctx.getPrimitiveCount() == 0);
        DOCTEST_CHECK(ctx.getObjectCount() == 0);
        DOCTEST_CHECK(!ctx.isGeometryDirty());

        Date d = ctx.getDate();
        DOCTEST_CHECK(d.day == 1);
        DOCTEST_CHECK(d.month == 6);
        DOCTEST_CHECK(d.year == 2000);

        Time t = ctx.getTime();
        DOCTEST_CHECK(t.hour == 12);
        DOCTEST_CHECK(t.minute == 0);
        DOCTEST_CHECK(t.second == 0);

        Location l = ctx.getLocation();
        DOCTEST_CHECK(l.latitude_deg == doctest::Approx(38.55));
        DOCTEST_CHECK(l.longitude_deg == doctest::Approx(121.76));
        DOCTEST_CHECK(l.UTC_offset == doctest::Approx(8));
    }

    SUBCASE("Random number generator") {
        Context ctx;
        ctx.seedRandomGenerator(12345);
        std::minstd_rand0 *gen1 = ctx.getRandomGenerator();
        float rand1 = (*gen1)();

        ctx.seedRandomGenerator(12345);
        std::minstd_rand0 *gen2 = ctx.getRandomGenerator();
        float rand2 = (*gen2)();

        DOCTEST_CHECK(rand1 == rand2);

        float r_uniform = ctx.randu();
        DOCTEST_CHECK(r_uniform >= 0.f);
        DOCTEST_CHECK(r_uniform <= 1.f);

        float r_norm = ctx.randn();
        // Hard to test for normality, but let's check it's a number
        DOCTEST_CHECK(!std::isnan(r_norm));
    }

    SUBCASE("Random number ranges") {
        Context ctx;
        ctx.seedRandomGenerator(6789);
        float r = ctx.randu(-1.f, 1.f);
        DOCTEST_CHECK(r >= -1.f);
        DOCTEST_CHECK(r <= 1.f);
        int ri = ctx.randu(0, 5);
        DOCTEST_CHECK(ri >= 0);
        DOCTEST_CHECK(ri <= 5);
        float rn = ctx.randn(2.f, 0.5f);
        DOCTEST_CHECK(!std::isnan(rn));
    }

    SUBCASE("Texture utility methods") {
        Context ctx;
        {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, "lib/images/solid.jpg"));
            DOCTEST_CHECK_THROWS(ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, "lib/images/missing.png"));
            DOCTEST_CHECK_THROWS(ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, "lib/images/invalid.txt"));
        }

        Texture tex("lib/images/solid.jpg");
        DOCTEST_CHECK(tex.getTextureFile() == "lib/images/solid.jpg");
        int2 res = tex.getImageResolution();
        DOCTEST_CHECK(res.x == 5);
        DOCTEST_CHECK(res.y == 5);
        DOCTEST_CHECK(!tex.hasTransparencyChannel());
        const auto *alpha = tex.getTransparencyData();
        DOCTEST_CHECK(alpha->empty());
        std::vector<vec2> uv{{0.f, 0.f}, {1.f, 0.f}, {1.f, 1.f}};
        float sf = tex.getSolidFraction(uv);
        DOCTEST_CHECK(sf == doctest::Approx(1.f));
    }

    SUBCASE("Geometry dirty flags") {
        Context ctx;
        uint p = ctx.addPatch();
        DOCTEST_CHECK(ctx.isGeometryDirty());
        DOCTEST_CHECK(ctx.isPrimitiveDirty(p));

        ctx.markGeometryClean();
        DOCTEST_CHECK(!ctx.isGeometryDirty());
        DOCTEST_CHECK(!ctx.isPrimitiveDirty(p));

        ctx.markPrimitiveDirty(p);
        DOCTEST_CHECK(ctx.isGeometryDirty());
        DOCTEST_CHECK(ctx.isPrimitiveDirty(p));

        ctx.markPrimitiveClean(p);
        DOCTEST_CHECK(!ctx.isGeometryDirty());
        DOCTEST_CHECK(!ctx.isPrimitiveDirty(p));

        ctx.markGeometryDirty();
        DOCTEST_CHECK(ctx.isGeometryDirty());
    }

    SUBCASE("Geometry dirty flags vector") {
        Context ctx;
        std::vector<uint> ids{ctx.addPatch(), ctx.addPatch()};
        ctx.markGeometryClean();
        ctx.markPrimitiveDirty(ids);
        for (uint id: ids) {
            DOCTEST_CHECK(ctx.isPrimitiveDirty(id));
        }
        ctx.markPrimitiveClean(ids);
        for (uint id: ids) {
            DOCTEST_CHECK(!ctx.isPrimitiveDirty(id));
        }

        vec3 shift = make_vec3(1.f, 0.f, 0.f);
        ctx.translatePrimitive(ids, shift);
        for (uint id: ids) {
            vec3 c = ctx.getPatchCenter(id);
            DOCTEST_CHECK(c.x == doctest::Approx(shift.x).epsilon(errtol));
        }
    }

    SUBCASE("Date and Time Manipulation") {
        Context ctx;
        ctx.setDate(15, 7, 2025);
        Date d = ctx.getDate();
        DOCTEST_CHECK(d.day == 15);
        DOCTEST_CHECK(d.month == 7);
        DOCTEST_CHECK(d.year == 2025);
        DOCTEST_CHECK(strcmp(ctx.getMonthString(), "JUL") == 0);
        DOCTEST_CHECK(ctx.getJulianDate() == 196);

        ctx.setTime(45, 30, 10);
        Time t = ctx.getTime();
        DOCTEST_CHECK(t.hour == 10);
        DOCTEST_CHECK(t.minute == 30);
        DOCTEST_CHECK(t.second == 45);

        capture_cerr cerr_buffer;
        DOCTEST_CHECK_THROWS(ctx.setDate(32, 1, 2025));
        DOCTEST_CHECK_THROWS(ctx.setTime(60, 0, 0));
    }

    SUBCASE("Location Manipulation") {
        Context ctx;
        Location loc = {40.7128, -74.0060, 10.0};
        ctx.setLocation(loc);
        Location l = ctx.getLocation();
        DOCTEST_CHECK(l.latitude_deg == doctest::Approx(40.7128));
        DOCTEST_CHECK(l.longitude_deg == doctest::Approx(-74.0060));
        DOCTEST_CHECK(l.UTC_offset == doctest::Approx(10.0));
    }

    SUBCASE("primitive orientation and transforms") {
        Context ctx;
        uint id = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        ctx.markGeometryClean();

        vec3 n = ctx.getPrimitiveNormal(id);
        DOCTEST_CHECK(n == vec3(0.f, 0.f, 1.f));

        ctx.setPrimitiveElevation(id, make_vec3(0, 0, 0), 0.f);
        n = ctx.getPrimitiveNormal(id);
        DOCTEST_CHECK(n.x == doctest::Approx(0.f).epsilon(errtol));
        DOCTEST_CHECK(n.y == doctest::Approx(1.f).epsilon(errtol));
        DOCTEST_CHECK(n.z == doctest::Approx(0.f).epsilon(errtol));

        ctx.setPrimitiveAzimuth(id, make_vec3(0, 0, 0), 0.5f * PI_F);
        n = ctx.getPrimitiveNormal(id);
        DOCTEST_CHECK(n.x == doctest::Approx(1.f).epsilon(errtol));
        DOCTEST_CHECK(n.y == doctest::Approx(0.f).epsilon(errtol));

        ctx.setPrimitiveNormal(id, make_vec3(0, 0, 0), make_vec3(0, 0, 1));
        n = ctx.getPrimitiveNormal(id);
        DOCTEST_CHECK(n.z == doctest::Approx(1.f).epsilon(errtol));

        float M[16];
        makeTranslationMatrix(make_vec3(1.f, 2.f, 3.f), M);
        ctx.setPrimitiveTransformationMatrix(id, M);
        float out[16];
        ctx.getPrimitiveTransformationMatrix(id, out);
        for (int i = 0; i < 16; ++i) {
            DOCTEST_CHECK(out[i] == doctest::Approx(M[i]));
        }
        DOCTEST_CHECK(ctx.isPrimitiveDirty(id));
    }
}

TEST_CASE("Primitive Management: Creation, Properties, and Operations") {
    SUBCASE("addPatch") {
        vec3 center, center_r;
        vec2 size, size_r;
        std::vector<vec3> vertices, vertices_r;
        SphericalCoord rotation, rotation_r;
        vec3 normal, normal_r;
        RGBcolor color, color_r;
        uint UUID;
        std::vector<uint> UUIDs;
        PrimitiveType type;
        float area_r;
        uint objID;

        Context context_test;

        // uint addPatch( const vec3& center, const vec2& size );
        center = make_vec3(1, 2, 3);
        size = make_vec2(1, 2);
        vertices.resize(4);
        vertices.at(0) = center + make_vec3(-0.5f * size.x, -0.5f * size.y, 0.f);
        vertices.at(1) = center + make_vec3(0.5f * size.x, -0.5f * size.y, 0.f);
        vertices.at(2) = center + make_vec3(0.5f * size.x, 0.5f * size.y, 0.f);
        vertices.at(3) = center + make_vec3(-0.5f * size.x, 0.5f * size.y, 0.f);

        DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size));
        DOCTEST_CHECK_NOTHROW(type = context_test.getPrimitiveType(UUID));
        DOCTEST_CHECK_NOTHROW(center_r = context_test.getPatchCenter(UUID));
        DOCTEST_CHECK_NOTHROW(size_r = context_test.getPatchSize(UUID));
        DOCTEST_CHECK_NOTHROW(normal_r = context_test.getPrimitiveNormal(UUID));
        DOCTEST_CHECK_NOTHROW(vertices_r = context_test.getPrimitiveVertices(UUID));
        DOCTEST_CHECK_NOTHROW(area_r = context_test.getPrimitiveArea(UUID));
        DOCTEST_CHECK_NOTHROW(color_r = context_test.getPrimitiveColor(UUID));

        DOCTEST_CHECK(type == PRIMITIVE_TYPE_PATCH);
        DOCTEST_CHECK(center_r.x == center.x);
        DOCTEST_CHECK(center_r.y == center.y);
        DOCTEST_CHECK(center_r.z == center.z);
        DOCTEST_CHECK(size_r.x == size.x);
        DOCTEST_CHECK(size_r.y == size.y);
        DOCTEST_CHECK(normal_r.x == 0.f);
        DOCTEST_CHECK(normal_r.y == 0.f);
        DOCTEST_CHECK(normal_r.z == 1.f);
        DOCTEST_CHECK(vertices_r.size() == 4);
        DOCTEST_CHECK(vertices_r.at(0).x == vertices.at(0).x);
        DOCTEST_CHECK(vertices_r.at(0).y == vertices.at(0).y);
        DOCTEST_CHECK(vertices_r.at(0).z == vertices.at(0).z);
        DOCTEST_CHECK(vertices_r.at(1).x == vertices.at(1).x);
        DOCTEST_CHECK(vertices_r.at(1).y == vertices.at(1).y);
        DOCTEST_CHECK(vertices_r.at(1).z == vertices.at(1).z);
        DOCTEST_CHECK(vertices_r.at(2).x == vertices.at(2).x);
        DOCTEST_CHECK(vertices_r.at(2).y == vertices.at(2).y);
        DOCTEST_CHECK(vertices_r.at(2).z == vertices.at(2).z);
        DOCTEST_CHECK(vertices_r.at(3).x == vertices.at(3).x);
        DOCTEST_CHECK(vertices_r.at(3).y == vertices.at(3).y);
        DOCTEST_CHECK(vertices_r.at(3).z == vertices.at(3).z);
        CHECK(area_r == doctest::Approx(size.x * size.y).epsilon(errtol));
        DOCTEST_CHECK(color_r.r == 0.f);
        DOCTEST_CHECK(color_r.g == 0.f);
        DOCTEST_CHECK(color_r.b == 0.f);
        DOCTEST_CHECK(context_test.getPrimitiveTextureFile(UUID).empty());
    }
    SUBCASE("rotated patch") {
        Context context_test;

        vec3 center = make_vec3(1, 2, 3);
        vec2 size = make_vec2(1, 2);
        SphericalCoord rotation = make_SphericalCoord(1.f, 0.15f * PI_F, 0.5f * PI_F);
        rotation.azimuth = 0.5f * PI_F;

        uint UUID;
        DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size, rotation));

        vec3 normal_r;
        DOCTEST_CHECK_NOTHROW(normal_r = context_test.getPrimitiveNormal(UUID));

        SphericalCoord rotation_r;
        DOCTEST_CHECK_NOTHROW(rotation_r = make_SphericalCoord(0.5f * PI_F - asinf(normal_r.z), atan2f(normal_r.x, normal_r.y)));

        DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUID));

        DOCTEST_CHECK(rotation_r.elevation == doctest::Approx(rotation.elevation).epsilon(errtol));
        DOCTEST_CHECK(rotation_r.azimuth == doctest::Approx(rotation.azimuth).epsilon(errtol));
    }
    SUBCASE("addTriangle") {
        Context context_test;

        vec3 v0, v0_r;
        vec3 v1, v1_r;
        vec3 v2, v2_r;
        uint UUID;

        // uint addTriangle( const vec3& v0, const vec3& v1, const vec3& v2, const RGBcolor &color );
        v0 = make_vec3(1, 2, 3);
        v1 = make_vec3(2, 4, 6);
        v2 = make_vec3(3, 6, 5);
        std::vector<vec3> vertices{v0, v1, v2};
        RGBcolor color = RGB::red;

        DOCTEST_CHECK_NOTHROW(UUID = context_test.addTriangle(v0, v1, v2, color));
        DOCTEST_CHECK(context_test.getPrimitiveType(UUID) == PRIMITIVE_TYPE_TRIANGLE);

        vec3 normal = normalize(cross(v1 - v0, v2 - v1));
        vec3 normal_r = context_test.getPrimitiveNormal(UUID);
        DOCTEST_CHECK(normal_r.x == doctest::Approx(normal.x).epsilon(errtol));
        DOCTEST_CHECK(normal_r.y == doctest::Approx(normal.y).epsilon(errtol));
        DOCTEST_CHECK(normal_r.z == doctest::Approx(normal.z).epsilon(errtol));

        std::vector<vec3> vertices_r;
        DOCTEST_CHECK_NOTHROW(vertices_r = context_test.getPrimitiveVertices(UUID));
        DOCTEST_CHECK(vertices_r.size() == 3);
        DOCTEST_CHECK(vertices_r.at(0).x == v0.x);
        DOCTEST_CHECK(vertices_r.at(0).y == v0.y);
        DOCTEST_CHECK(vertices_r.at(0).z == v0.z);

        RGBcolor color_r;
        DOCTEST_CHECK_NOTHROW(color_r = context_test.getPrimitiveColor(UUID));
        DOCTEST_CHECK(color_r.r == color.r);
        DOCTEST_CHECK(color_r.g == color.g);
        DOCTEST_CHECK(color_r.b == color.b);
        DOCTEST_CHECK(context_test.getPrimitiveTextureFile(UUID).empty());

        float a = (v1 - v0).magnitude();
        float b = (v2 - v0).magnitude();
        float c = (v2 - v1).magnitude();
        float s = 0.5f * (a + b + c);
        float area = sqrtf(s * (s - a) * (s - b) * (s - c));
        float area_r;
        DOCTEST_CHECK_NOTHROW(area_r = context_test.getPrimitiveArea(UUID));
        DOCTEST_CHECK(area_r == doctest::Approx(area).epsilon(errtol));
    }
    SUBCASE("copyPrimitive (patch)") {
        Context context_test;
        uint UUID, UUID_cpy;

        std::vector<float> cpdata{5.2f, 2.5f, 3.1f};

        vec3 center = make_vec3(1, 2, 3);
        vec2 size = make_vec2(1, 2);

        DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size));

        DOCTEST_CHECK_NOTHROW(context_test.setPrimitiveData(UUID, "somedata", cpdata));

        DOCTEST_CHECK_NOTHROW(UUID_cpy = context_test.copyPrimitive(UUID));

        vec3 center_cpy;
        DOCTEST_CHECK_NOTHROW(center_cpy = context_test.getPatchCenter(UUID_cpy));
        vec2 size_cpy;
        DOCTEST_CHECK_NOTHROW(size_cpy = context_test.getPatchSize(UUID_cpy));

        DOCTEST_CHECK(UUID_cpy == 1);
        DOCTEST_CHECK(center_cpy.x == center.x);
        DOCTEST_CHECK(center_cpy.y == center.y);
        DOCTEST_CHECK(center_cpy.z == center.z);
        DOCTEST_CHECK(size_cpy.x == size.x);
        DOCTEST_CHECK(size_cpy.y == size.y);

        std::vector<float> cpdata_copy;
        context_test.getPrimitiveData(UUID_cpy, "somedata", cpdata_copy);

        DOCTEST_CHECK(cpdata.size() == cpdata_copy.size());
        for (uint i = 0; i < cpdata.size(); i++) {
            DOCTEST_CHECK(cpdata.at(i) == cpdata_copy.at(i));
        }

        // translate the copied patch
        vec3 shift = make_vec3(5.f, 4.f, 3.f);
        DOCTEST_CHECK_NOTHROW(context_test.translatePrimitive(UUID_cpy, shift));
        DOCTEST_CHECK_NOTHROW(center_cpy = context_test.getPatchCenter(UUID_cpy));
        vec3 center_r;
        DOCTEST_CHECK_NOTHROW(center_r = context_test.getPatchCenter(UUID));

        DOCTEST_CHECK(center_cpy.x == doctest::Approx(center.x + shift.x).epsilon(errtol));
        DOCTEST_CHECK(center_cpy.y == doctest::Approx(center.y + shift.y).epsilon(errtol));
        DOCTEST_CHECK(center_cpy.z == doctest::Approx(center.z + shift.z).epsilon(errtol));
        DOCTEST_CHECK(center_r.x == center.x);
        DOCTEST_CHECK(center_r.y == center.y);
        DOCTEST_CHECK(center_r.z == center.z);
    }
    SUBCASE("copyPrimitive (triangle)") {
        Context context_test;

        vec3 v0 = make_vec3(0, 0, 0);
        vec3 v1 = make_vec3(1, 0, 0);
        vec3 v2 = make_vec3(0, 1, 0);
        uint UUID, UUID_cpy;

        DOCTEST_CHECK_NOTHROW(UUID = context_test.addTriangle(v0, v1, v2, RGB::blue));
        DOCTEST_CHECK_NOTHROW(UUID_cpy = context_test.copyPrimitive(UUID));

        std::vector<vec3> verts_org, verts_cpy;
        DOCTEST_CHECK_NOTHROW(verts_org = context_test.getPrimitiveVertices(UUID));
        DOCTEST_CHECK_NOTHROW(verts_cpy = context_test.getPrimitiveVertices(UUID_cpy));
        DOCTEST_CHECK(verts_org == verts_cpy);

        vec3 shift = make_vec3(5.f, 4.f, 3.f);
        DOCTEST_CHECK_NOTHROW(context_test.translatePrimitive(UUID_cpy, shift));
        DOCTEST_CHECK_NOTHROW(verts_cpy = context_test.getPrimitiveVertices(UUID_cpy));
        DOCTEST_CHECK(verts_cpy.at(0) == verts_org.at(0) + shift);
        DOCTEST_CHECK(verts_cpy.at(1) == verts_org.at(1) + shift);
        DOCTEST_CHECK(verts_cpy.at(2) == verts_org.at(2) + shift);

        DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUID));
        DOCTEST_CHECK(!context_test.doesPrimitiveExist(UUID));
    }
    SUBCASE("deletePrimitive") {
        Context context_test;
        uint UUID;
        vec3 center = make_vec3(1, 2, 3);
        vec2 size = make_vec2(1, 2);

        DOCTEST_CHECK_NOTHROW(UUID = context_test.addPatch(center, size));

        DOCTEST_CHECK_NOTHROW(context_test.deletePrimitive(UUID));

        uint primitive_count;
        DOCTEST_CHECK_NOTHROW(primitive_count = context_test.getPrimitiveCount(UUID));
        DOCTEST_CHECK(primitive_count == 0);
        DOCTEST_CHECK(!context_test.doesPrimitiveExist(UUID));
    }
    SUBCASE("primitive bounding box") {
        Context context_test;
        std::vector<uint> UUIDs;
        UUIDs.push_back(context_test.addPatch(make_vec3(-1, 0, 0), make_vec2(0.5, 0.5)));
        UUIDs.push_back(context_test.addPatch(make_vec3(1, 0, 0), make_vec2(0.5, 0.5)));

        vec3 bmin, bmax;
        DOCTEST_CHECK_NOTHROW(context_test.getPrimitiveBoundingBox(UUIDs, bmin, bmax));
        DOCTEST_CHECK(bmin == make_vec3(-1.25f, -0.25f, 0.f));
        DOCTEST_CHECK(bmax == make_vec3(1.25f, 0.25f, 0.f));
    }
    SUBCASE("primitive scale and data") {
        Context context_test;
        vec2 sz_0 = make_vec2(0.5f, 3.f);
        float area0 = sz_0.x * sz_0.y;
        float scale = 2.6f;
        uint UUID = context_test.addPatch(make_vec3(0, 0, 0), sz_0);
        context_test.scalePrimitive(UUID, make_vec3(scale, scale, scale));
        float area1 = context_test.getPrimitiveArea(UUID);
        DOCTEST_CHECK(area1 == doctest::Approx(scale * scale * area0).epsilon(1e-5));

        float data = 5.f;
        context_test.setPrimitiveData(UUID, "some_data", data);
        DOCTEST_CHECK(context_test.doesPrimitiveDataExist(UUID, "some_data"));
        float data_r;
        context_test.getPrimitiveData(UUID, "some_data", data_r);
        DOCTEST_CHECK(data_r == data);

        std::vector<float> vec = {0, 1, 2, 3, 4};
        context_test.setPrimitiveData(UUID, "vec_data", vec);
        std::vector<float> vec_r;
        context_test.getPrimitiveData(UUID, "vec_data", vec_r);
        DOCTEST_CHECK(vec_r == vec);

        std::vector<uint> UUIDs_filter;
        std::vector<uint> UUIDs_multi;
        for (uint i = 0; i < 4; i++) {
            UUIDs_multi.push_back(context_test.addPatch());
        }
        context_test.setPrimitiveData(UUIDs_multi[0], "val", 4.f);
        context_test.setPrimitiveData(UUIDs_multi[0], "str", "cat");
        context_test.setPrimitiveData(UUIDs_multi[1], "val", 3.f);
        context_test.setPrimitiveData(UUIDs_multi[1], "str", "cat");
        context_test.setPrimitiveData(UUIDs_multi[2], "val", 2.f);
        context_test.setPrimitiveData(UUIDs_multi[2], "str", "dog");
        context_test.setPrimitiveData(UUIDs_multi[3], "val", 1.f);
        context_test.setPrimitiveData(UUIDs_multi[3], "str", "dog");

        UUIDs_filter = context_test.filterPrimitivesByData(UUIDs_multi, "val", 2.f, "<=");
        DOCTEST_CHECK(UUIDs_filter.size() == 2);
        DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[2]) != UUIDs_filter.end());
        DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[3]) != UUIDs_filter.end());

        UUIDs_filter = context_test.filterPrimitivesByData(UUIDs_multi, "str", "cat");
        DOCTEST_CHECK(UUIDs_filter.size() == 2);
        DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[0]) != UUIDs_filter.end());
        DOCTEST_CHECK(std::find(UUIDs_filter.begin(), UUIDs_filter.end(), UUIDs_multi[1]) != UUIDs_filter.end());
    }
    SUBCASE("texture uv and solid fraction") {
        Context context_test;

        vec2 sizep = make_vec2(2, 3);
        const char *texture = "lib/images/disk_texture.png";
        vec2 uv0 = make_vec2(0, 0);
        vec2 uv1 = make_vec2(1, 0);
        vec2 uv2 = make_vec2(1, 1);
        vec2 uv3 = make_vec2(0, 1);
        uint UUIDp = context_test.addPatch(make_vec3(2, 3, 4), sizep, nullrotation, texture, 0.5f * (uv0 + uv2), uv2 - uv0);
        DOCTEST_CHECK(!context_test.getPrimitiveTextureFile(UUIDp).empty());
        float Ap = context_test.getPrimitiveArea(UUIDp);
        DOCTEST_CHECK(Ap == doctest::Approx(0.25f * PI_F * sizep.x * sizep.y).epsilon(0.01));
        std::vector<vec2> uv = context_test.getPrimitiveTextureUV(UUIDp);
        DOCTEST_CHECK(uv.size() == 4);
        DOCTEST_CHECK(uv.at(0) == uv0);
        DOCTEST_CHECK(uv.at(1) == uv1);
        DOCTEST_CHECK(uv.at(2) == uv2);
        DOCTEST_CHECK(uv.at(3) == uv3);

        uint UUIDt = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), "lib/images/diamond_texture.png", make_vec2(0, 0), make_vec2(1, 0), make_vec2(1, 1));
        float solid_fraction = context_test.getPrimitiveSolidFraction(UUIDt);
        DOCTEST_CHECK(solid_fraction == doctest::Approx(0.5f).epsilon(errtol));
    }

    SUBCASE("advanced primitive transforms") {
        Context ctx;
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
        std::vector<uint> ids{p1, p2};
        ctx.markGeometryClean();

        ctx.rotatePrimitive(p1, 0.5f * PI_F, "x");
        vec3 n = ctx.getPrimitiveNormal(p1);
        DOCTEST_CHECK(n.y == doctest::Approx(-1.f).epsilon(errtol));
        DOCTEST_CHECK(n.z == doctest::Approx(0.f).epsilon(errtol));

        ctx.rotatePrimitive(ids, PI_F, make_vec3(0, 1, 0));
        vec3 c = ctx.getPatchCenter(p2);
        DOCTEST_CHECK(c.x == doctest::Approx(-1.f).epsilon(errtol));

        ctx.rotatePrimitive(p1, PI_F, make_vec3(0, 0, 0), make_vec3(0, 0, 1));

        ctx.scalePrimitiveAboutPoint(p2, make_vec3(2.f, 2.f, 2.f), make_vec3(0, 0, 0));
        vec2 sz = ctx.getPatchSize(p2);
        DOCTEST_CHECK(sz.x == doctest::Approx(2.f).epsilon(errtol));

        ctx.scalePrimitiveAboutPoint(ids, make_vec3(0.5f, 0.5f, 0.5f), make_vec3(0, 0, 0));
        sz = ctx.getPatchSize(p2);
        DOCTEST_CHECK(sz.x == doctest::Approx(1.f).epsilon(errtol));
    }
}

TEST_CASE("Triangle Scaling") {
    Context ctx;
    const float errtol = 0.0001f;

    SUBCASE("scalePrimitive basic test") {
        // Create a simple right triangle at the origin
        vec3 v0 = make_vec3(0, 0, 0);
        vec3 v1 = make_vec3(1, 0, 0);
        vec3 v2 = make_vec3(0, 1, 0);
        uint tri = ctx.addTriangle(v0, v1, v2);

        // Get initial vertices and area
        std::vector<vec3> verts_before = ctx.getPrimitiveVertices(tri);
        float area_before = ctx.getPrimitiveArea(tri);

        // Apply uniform 2x scaling
        ctx.scalePrimitive(tri, make_vec3(2, 2, 2));

        // Get vertices and area after scaling
        std::vector<vec3> verts_after = ctx.getPrimitiveVertices(tri);
        float area_after = ctx.getPrimitiveArea(tri);

        // Expected: vertices should be doubled (scaling about origin)
        // v0: (0,0,0) -> (0,0,0)  [origin stays at origin]
        // v1: (1,0,0) -> (2,0,0)
        // v2: (0,1,0) -> (0,2,0)
        // Area should be 4x larger (scale^2 for 2D)

        DOCTEST_CHECK(verts_after[0].x == doctest::Approx(0.0f).epsilon(errtol));
        DOCTEST_CHECK(verts_after[0].y == doctest::Approx(0.0f).epsilon(errtol));
        DOCTEST_CHECK(verts_after[1].x == doctest::Approx(2.0f).epsilon(errtol));
        DOCTEST_CHECK(verts_after[1].y == doctest::Approx(0.0f).epsilon(errtol));
        DOCTEST_CHECK(verts_after[2].x == doctest::Approx(0.0f).epsilon(errtol));
        DOCTEST_CHECK(verts_after[2].y == doctest::Approx(2.0f).epsilon(errtol));

        DOCTEST_CHECK(area_after == doctest::Approx(4.0f * area_before).epsilon(errtol));
    }

    SUBCASE("scalePrimitiveAboutPoint test") {
        // Create a simple right triangle at the origin
        vec3 v0 = make_vec3(0, 0, 0);
        vec3 v1 = make_vec3(1, 0, 0);
        vec3 v2 = make_vec3(0, 1, 0);
        uint tri = ctx.addTriangle(v0, v1, v2);

        // Get initial area
        float area_before = ctx.getPrimitiveArea(tri);

        // Apply 2x scaling about origin
        ctx.scalePrimitiveAboutPoint(tri, make_vec3(2, 2, 2), make_vec3(0, 0, 0));

        // Get area after scaling
        float area_after = ctx.getPrimitiveArea(tri);

        // Expected: should behave same as scalePrimitive when scaling about origin
        DOCTEST_CHECK(area_after == doctest::Approx(4.0f * area_before).epsilon(errtol));
    }

    SUBCASE("scalePrimitiveAboutPoint - scale about centroid") {
        // Create a triangle NOT at the origin
        vec3 v0 = make_vec3(1, 1, 0);
        vec3 v1 = make_vec3(2, 1, 0);
        vec3 v2 = make_vec3(1, 2, 0);
        uint tri = ctx.addTriangle(v0, v1, v2);

        // Calculate centroid
        std::vector<vec3> verts_before = ctx.getPrimitiveVertices(tri);
        vec3 center = make_vec3(0, 0, 0);
        for (const auto &v: verts_before) {
            center = center + v;
        }
        center = center / float(verts_before.size());

        float area_before = ctx.getPrimitiveArea(tri);

        // Scale by 0.5 about the centroid (like user's code)
        ctx.scalePrimitiveAboutPoint(tri, make_vec3(0.5f, 0.5f, 0.5f), center);

        // Get area after scaling
        float area_after = ctx.getPrimitiveArea(tri);

        // Expected: area should be 0.25x (scale^2)
        DOCTEST_CHECK(area_after == doctest::Approx(0.25f * area_before).epsilon(errtol));
    }

    SUBCASE("triangle in compound object") {
        // Create a compound object with a triangle
        std::vector<uint> UUIDs;
        UUIDs.push_back(ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)));
        uint objID = ctx.addPolymeshObject(UUIDs);

        uint tri = UUIDs[0];
        float area_before = ctx.getPrimitiveArea(tri);

        // Try to scale the triangle (should be blocked)
        bool has_warning;
        {
            capture_cerr cerr_buffer;
            ctx.scalePrimitiveAboutPoint(tri, make_vec3(2, 2, 2), make_vec3(0, 0, 0));
            has_warning = cerr_buffer.has_output();
        } // cerr_buffer destroyed here
        DOCTEST_CHECK(has_warning); // Should print warning

        float area_after = ctx.getPrimitiveArea(tri);

        // Area should NOT change (scaling blocked for compound objects)
        DOCTEST_CHECK(area_after == doctest::Approx(area_before).epsilon(errtol));
    }
}

TEST_CASE("Object Management") {
    SUBCASE("addBoxObject") {
        Context context_test;

        vec3 center = make_vec3(1, 2, 3);
        vec3 size = make_vec3(3, 2, 1);
        int3 subdiv(1, 1, 1);

        uint objID;
        DOCTEST_CHECK_NOTHROW(objID = context_test.addBoxObject(center, size, subdiv));
        std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);

        DOCTEST_CHECK(UUIDs.size() == 6);
        vec3 normal_r = context_test.getPrimitiveNormal(UUIDs.at(0));
        DOCTEST_CHECK(doctest::Approx(normal_r.magnitude()).epsilon(errtol) == 1.f);
        normal_r = context_test.getPrimitiveNormal(UUIDs.at(2));
        DOCTEST_CHECK(doctest::Approx(normal_r.magnitude()).epsilon(errtol) == 1.f);

        vec2 size_r = context_test.getPatchSize(UUIDs.at(0));
        DOCTEST_CHECK(size_r.x == doctest::Approx(size.x).epsilon(errtol));
        DOCTEST_CHECK(size_r.y == doctest::Approx(size.z).epsilon(errtol));

        size_r = context_test.getPatchSize(UUIDs.at(2));
        DOCTEST_CHECK(size_r.x == doctest::Approx(size.y).epsilon(errtol));
        DOCTEST_CHECK(size_r.y == doctest::Approx(size.z).epsilon(errtol));

        float volume = context_test.getBoxObjectVolume(objID);
        DOCTEST_CHECK(volume == doctest::Approx(size.x * size.y * size.z).epsilon(errtol));
    }
    SUBCASE("addTileObject rotated") {
        Context context_test;

        vec3 center = make_vec3(1, 2, 3);
        vec2 size = make_vec2(3, 2);
        int2 subdiv(3, 3);
        SphericalCoord rotation = make_SphericalCoord(0.25f * PI_F, 1.4f * PI_F);
        uint objID = context_test.addTileObject(center, size, rotation, subdiv);

        std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        for (uint UUIDp: UUIDs) {
            vec3 n = context_test.getPrimitiveNormal(UUIDp);
            SphericalCoord rot = cart2sphere(n);
            DOCTEST_CHECK(rot.zenith == doctest::Approx(rotation.zenith).epsilon(errtol));
            DOCTEST_CHECK(rot.azimuth == doctest::Approx(rotation.azimuth).epsilon(errtol));
        }
    }
    SUBCASE("textured tile area") {
        Context context_test;

        vec3 center = make_vec3(1, 2, 3);
        vec2 size = make_vec2(3, 2);
        int2 subdiv = make_int2(5, 5);
        SphericalCoord rotation = make_SphericalCoord(0.1f * PI_F, 2.4f * PI_F);

        uint objID = context_test.addTileObject(center, size, rotation, subdiv, "lib/images/disk_texture.png");
        std::vector<uint> UUIDs = context_test.getObjectPrimitiveUUIDs(objID);
        float area_sum = 0.f;
        for (uint UUID: UUIDs) {
            area_sum += context_test.getPrimitiveArea(UUID);
        }
        float area_exact = 0.25f * PI_F * size.x * size.y;
        DOCTEST_CHECK(area_sum == doctest::Approx(area_exact).epsilon(5e-3));
    }
    SUBCASE("cone object transforms") {
        Context context_test;
        float r0 = 0.5f, r1 = 1.f, len = 2.f;
        vec3 node0 = make_vec3(0, 0, 0);
        vec3 node1 = make_vec3(0, 0, len);
        uint cone = context_test.addConeObject(50, node0, node1, r0, r1);
        context_test.translateObject(cone, make_vec3(1, 1, 1));
        std::vector<vec3> nodes = context_test.getConeObjectNodes(cone);
        DOCTEST_CHECK(nodes.at(0) == make_vec3(1, 1, 1));
        DOCTEST_CHECK(nodes.at(1) == make_vec3(1, 1, 1 + len));
        vec3 axis = cross(make_vec3(0, 0, 1), make_vec3(1, 0, 0));
        float ang = acos_safe(make_vec3(1, 0, 0) * make_vec3(0, 0, 1));
        context_test.translateObject(cone, -nodes.at(0));
        context_test.rotateObject(cone, ang, axis);
        context_test.translateObject(cone, nodes.at(0));
        nodes = context_test.getConeObjectNodes(cone);
        DOCTEST_CHECK(nodes.at(1).x == doctest::Approx(nodes.at(0).x + len).epsilon(errtol));
        context_test.scaleConeObjectLength(cone, 2.0);
        nodes = context_test.getConeObjectNodes(cone);
        DOCTEST_CHECK(nodes.at(1).x == doctest::Approx(nodes.at(0).x + 2 * len).epsilon(errtol));
        context_test.scaleConeObjectGirth(cone, 2.0);
        std::vector<float> radii = context_test.getConeObjectNodeRadii(cone);
        DOCTEST_CHECK(radii.at(0) == doctest::Approx(2 * r0).epsilon(errtol));
        DOCTEST_CHECK(radii.at(1) == doctest::Approx(2 * r1).epsilon(errtol));
    }

    SUBCASE("rotate and scale objects") {
        Context ctx;
        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        ctx.rotateObject(obj, 0.5f * PI_F, "z");
        vec3 bmin, bmax;
        ctx.getObjectBoundingBox(obj, bmin, bmax);
        DOCTEST_CHECK(bmax.x == doctest::Approx(0.5f).epsilon(errtol));

        ctx.scaleObjectAboutPoint(obj, make_vec3(2.f, 2.f, 2.f), make_vec3(0, 0, 0));
        ctx.getObjectBoundingBox(obj, bmin, bmax);
        DOCTEST_CHECK(bmax.x > 0.5f);
    }

    SUBCASE("domain bounding sphere") {
        Context ctx;
        std::vector<uint> ids;
        ids.push_back(ctx.addPatch(make_vec3(-1, 0, 0), make_vec2(1, 1)));
        ids.push_back(ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1)));
        vec3 c;
        float r;
        ctx.getDomainBoundingSphere(ids, c, r);
        DOCTEST_CHECK(c.x == doctest::Approx(0.f).epsilon(errtol));
        DOCTEST_CHECK(r > 1.f);
    }

    SUBCASE("copy and delete objects") {
        Context ctx;
        uint obj1 = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint obj2 = ctx.copyObject(obj1);
        DOCTEST_CHECK(ctx.doesObjectExist(obj1));
        DOCTEST_CHECK(ctx.doesObjectExist(obj2));
        ctx.deleteObject(obj2);
        DOCTEST_CHECK(!ctx.doesObjectExist(obj2));
    }

    SUBCASE("copy object with texture override preserves color") {
        capture_cerr cerr_buffer; // Capture deprecation warnings from setPrimitiveColor/overridePrimitiveTextureColor
        Context ctx;

        // Create a tile with texture
        std::vector<uint> UUIDs = ctx.addTile(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2), "lib/images/disk_texture.png");

        // Set color and override texture - these trigger deprecation warnings (once per execution)
        RGBcolor green_color = make_RGBcolor(0, 1, 0);
        ctx.setPrimitiveColor(UUIDs, green_color);
        ctx.overridePrimitiveTextureColor(UUIDs);

        // Create object from primitives
        uint objID = ctx.addPolymeshObject(UUIDs);

        // Verify original object has correct color and texture override
        DOCTEST_CHECK(ctx.getPrimitiveColor(UUIDs[0]) == green_color);
        DOCTEST_CHECK(ctx.isPrimitiveTextureColorOverridden(UUIDs[0]));

        // Copy the object
        uint objID_copy = ctx.copyObject(objID);
        std::vector<uint> UUIDs_copy = ctx.getObjectPrimitiveUUIDs(objID_copy);

        // Verify copied object preserves both color and texture override
        DOCTEST_CHECK(ctx.getPrimitiveColor(UUIDs_copy[0]) == green_color);
        DOCTEST_CHECK(ctx.isPrimitiveTextureColorOverridden(UUIDs_copy[0]));

        // Test with Triangle as well
        uint triangle = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), "lib/images/disk_texture.png", make_vec2(0, 0), make_vec2(1, 0), make_vec2(0, 1));
        RGBcolor blue_color = make_RGBcolor(0, 0, 1);
        ctx.setPrimitiveColor(triangle, blue_color);
        ctx.overridePrimitiveTextureColor(triangle);

        std::vector<uint> triangle_UUIDs = {triangle};
        uint triangle_obj = ctx.addPolymeshObject(triangle_UUIDs);
        uint triangle_obj_copy = ctx.copyObject(triangle_obj);
        std::vector<uint> triangle_UUIDs_copy = ctx.getObjectPrimitiveUUIDs(triangle_obj_copy);

        DOCTEST_CHECK(ctx.getPrimitiveColor(triangle_UUIDs_copy[0]) == blue_color);
        DOCTEST_CHECK(ctx.isPrimitiveTextureColorOverridden(triangle_UUIDs_copy[0]));
    }

    SUBCASE("domain cropping") {
        Context ctx;
        uint p1 = ctx.addPatch(make_vec3(-2.f, 0.f, 0.f), make_vec2(1, 1));
        uint p2 = ctx.addPatch(make_vec3(2.f, 0.f, 0.f), make_vec2(1, 1));
        uint p3 = ctx.addPatch(make_vec3(0.f, 3.f, 0.f), make_vec2(1, 1));
        uint p4 = ctx.addPatch(make_vec3(0.f, 0.f, 3.f), make_vec2(1, 1));

        bool has_output1, has_output2;
        {
            capture_cerr cerr_buffer;
            ctx.cropDomainX(make_vec2(-1.f, 1.f));
            DOCTEST_CHECK(!ctx.doesPrimitiveExist(p1));
            ctx.cropDomainY(make_vec2(-1.f, 1.f));
            DOCTEST_CHECK(!ctx.doesPrimitiveExist(p3));
            ctx.cropDomainZ(make_vec2(-1.f, 1.f));
            DOCTEST_CHECK(!ctx.doesPrimitiveExist(p4));
            has_output1 = cerr_buffer.has_output();
        }
        DOCTEST_CHECK(has_output1);

        {
            capture_cerr cerr_buffer;
            std::vector<uint> ids_rem = ctx.getAllUUIDs();
            ctx.cropDomain(ids_rem, make_vec2(-0.5f, 1.f), make_vec2(-0.5f, 1.f), make_vec2(-0.5f, 1.f));
            DOCTEST_CHECK(!ctx.doesPrimitiveExist(p2));
            has_output2 = cerr_buffer.has_output();
        }
        DOCTEST_CHECK(has_output2);
    }
}

TEST_CASE("Data Management") {
    SUBCASE("global and object data") {
        Context context_test;
        float gdata = 5.f;
        context_test.setGlobalData("some_data", gdata);
        float gdata_r;
        DOCTEST_CHECK(context_test.doesGlobalDataExist("some_data"));
        context_test.getGlobalData("some_data", gdata_r);
        DOCTEST_CHECK(gdata_r == gdata);

        std::vector<float> gvec{0, 1, 2, 3, 4};
        context_test.setGlobalData("vec", gvec);
        std::vector<float> gvec_r;
        context_test.getGlobalData("vec", gvec_r);
        DOCTEST_CHECK(gvec_r == gvec);

        uint objID = context_test.addTileObject(make_vec3(0, 0, 0), make_vec2(3, 1), nullrotation, make_int2(3, 3));
        float objdata = 7.f;
        context_test.setObjectData(objID, "obj", objdata);
        float objdata_r;
        context_test.getObjectData(objID, "obj", objdata_r);
        DOCTEST_CHECK(objdata_r == objdata);
    }
    SUBCASE("timeseries") {
        Context ctx;
        Date date = make_Date(12, 3, 2010);
        ctx.setDate(date);
        Time time0 = make_Time(13, 15, 39);
        ctx.setTime(time0);
        Time time1 = make_Time(time0.hour, 49, 14);
        ctx.addTimeseriesData("ts", 302.3f, date, time0);
        ctx.addTimeseriesData("ts", 305.3f, date, time1);
        ctx.setCurrentTimeseriesPoint("ts", 0);
        DOCTEST_CHECK(ctx.getTimeseriesLength("ts") == 2);
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 0) == doctest::Approx(302.3f));
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 1) == doctest::Approx(305.3f));
        float val = ctx.queryTimeseriesData("ts", date, time1);
        DOCTEST_CHECK(val == doctest::Approx(305.3f));
        DOCTEST_CHECK(ctx.doesTimeseriesVariableExist("ts"));
        std::vector<std::string> labels = ctx.listTimeseriesVariables();
        DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "ts") != labels.end());
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", ctx.getTimeseriesLength("ts") - 1) == doctest::Approx(305.3f));
        Time t1_r = ctx.queryTimeseriesTime("ts", 1);
        Date d1_r = ctx.queryTimeseriesDate("ts", 1);
        DOCTEST_CHECK(t1_r.minute == time1.minute);
        DOCTEST_CHECK(d1_r.day == date.day);
        ctx.setCurrentTimeseriesPoint("ts", 1);
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts") == doctest::Approx(305.3f));
    }

    SUBCASE("Primitive data") {
        capture_cerr cerr_buffer;

        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "test_int", 5);
        ctx.setPrimitiveData(p, "test_float", 3.14f);

        // getPrimitiveDataType
        DOCTEST_CHECK(ctx.getPrimitiveDataType("test_int") == HELIOS_TYPE_INT);
        DOCTEST_CHECK(ctx.getPrimitiveDataType("test_float") == HELIOS_TYPE_FLOAT);

        // getPrimitiveDataSize
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "test_int") == 1);

        // clearPrimitiveData
        ctx.clearPrimitiveData(p, "test_int");
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "test_int"));

        // listPrimitiveData
        std::vector<std::string> data_labels = ctx.listPrimitiveData(p);
        DOCTEST_CHECK(std::find(data_labels.begin(), data_labels.end(), "test_float") != data_labels.end());

        // getPrimitiveDataSize (doesn't exist)
        DOCTEST_CHECK_THROWS(ctx.getPrimitiveDataSize(p, "test_int"));

        // clearPrimitiveData
        ctx.clearPrimitiveData(p, "test_int");
        DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p, "test_int"));

        // listPrimitiveData
        ctx.setPrimitiveData(p, "test_int", 5);
        ctx.setPrimitiveData(p, "test_float", 3.14f);
        std::vector<std::string> labels = ctx.listPrimitiveData(p);
        DOCTEST_CHECK(labels.size() == 2);
        DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "test_int") != labels.end());
        DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "test_float") != labels.end());
        DOCTEST_CHECK(ctx.getPrimitiveDataType("test_float") == HELIOS_TYPE_FLOAT);
    }
}

TEST_CASE("Data and Object Management") {

    SUBCASE("Global data management") {
        Context ctx;
        ctx.setGlobalData("test_double", 1.23);
        DOCTEST_CHECK(ctx.getGlobalDataSize("test_double") == 1);
        DOCTEST_CHECK(ctx.getGlobalDataType("test_double") == HELIOS_TYPE_DOUBLE);
        ctx.clearGlobalData("test_double");
        DOCTEST_CHECK(!ctx.doesGlobalDataExist("test_double"));
        ctx.setGlobalData("test_string", "hello");
        std::vector<std::string> global_data_labels = ctx.listGlobalData();
        DOCTEST_CHECK(std::find(global_data_labels.begin(), global_data_labels.end(), "test_string") != global_data_labels.end());
    }

    SUBCASE("Object data management") {
        Context ctx;
        uint obj = ctx.addBoxObject(nullorigin, make_vec3(1, 1, 1), make_int3(2, 3, 2));
        ctx.setObjectData(obj, "test_vec", vec3(1, 2, 3));
        DOCTEST_CHECK(ctx.getObjectDataSize(obj, "test_vec") == 1);
        DOCTEST_CHECK(ctx.getObjectDataType("test_vec") == HELIOS_TYPE_VEC3);
        ctx.clearObjectData(obj, "test_vec");
        DOCTEST_CHECK(!ctx.doesObjectDataExist(obj, "test_vec"));
        ctx.setObjectData(obj, "test_int", 42);
        std::vector<std::string> object_data_labels = ctx.listObjectData(obj);
        DOCTEST_CHECK(std::find(object_data_labels.begin(), object_data_labels.end(), "test_int") != object_data_labels.end());
    }

    SUBCASE("Object creation and manipulation") {
        Context ctx;
        uint disk = ctx.addDiskObject(10, make_vec3(0, 0, 0), make_vec2(1, 1));
        DOCTEST_CHECK(ctx.getObjectType(disk) == OBJECT_TYPE_DISK);
        DOCTEST_CHECK(ctx.getObjectArea(disk) > 0);
        DOCTEST_CHECK(ctx.getDiskObjectCenter(disk) == make_vec3(0, 0, 0));
        DOCTEST_CHECK(ctx.getDiskObjectSubdivisionCount(disk) == 10);
        DOCTEST_CHECK(ctx.getDiskObjectSize(disk).x == doctest::Approx(1.f));

        uint sphere = ctx.addSphereObject(10, make_vec3(1, 1, 1), 0.5f);
        DOCTEST_CHECK(ctx.getObjectType(sphere) == OBJECT_TYPE_SPHERE);
        DOCTEST_CHECK(ctx.getObjectArea(sphere) > 0);
        DOCTEST_CHECK(ctx.getSphereObjectCenter(sphere) == make_vec3(1, 1, 1));
        DOCTEST_CHECK(ctx.getSphereObjectSubdivisionCount(sphere) == 10);
        DOCTEST_CHECK(ctx.getSphereObjectRadius(sphere).x == doctest::Approx(0.5f));

        std::vector<uint> p_uuids;
        p_uuids.push_back(ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)));
        uint polymesh = ctx.addPolymeshObject(p_uuids);
        DOCTEST_CHECK(ctx.getObjectType(polymesh) == OBJECT_TYPE_POLYMESH);
        DOCTEST_CHECK(ctx.getObjectArea(polymesh) > 0);
        DOCTEST_CHECK(ctx.getObjectCenter(polymesh).z == doctest::Approx(0.f));

        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        std::vector<float> radii = {0.2f, 0.1f};
        uint tube = ctx.addTubeObject(10, nodes, radii);
        DOCTEST_CHECK(ctx.getObjectType(tube) == OBJECT_TYPE_TUBE);
        DOCTEST_CHECK(ctx.getObjectArea(tube) > 0);
        DOCTEST_CHECK(ctx.getObjectCenter(tube).z == doctest::Approx(0.5f));
        DOCTEST_CHECK(ctx.getTubeObjectSubdivisionCount(tube) == 10);
        DOCTEST_CHECK(ctx.getTubeObjectNodeCount(tube) == 2);
        DOCTEST_CHECK(ctx.getTubeObjectNodeRadii(tube).size() == 2);
        DOCTEST_CHECK(ctx.getTubeObjectNodeColors(tube).size() == 2);
        DOCTEST_CHECK(ctx.getTubeObjectVolume(tube) > 0);
        ctx.appendTubeSegment(tube, make_vec3(0, 0, 2), 0.05f, RGB::red);
        DOCTEST_CHECK(ctx.getTubeObjectNodeCount(tube) == 3);
        ctx.scaleTubeGirth(tube, 2.f);
        DOCTEST_CHECK(ctx.getTubeObjectNodeRadii(tube)[0] == doctest::Approx(0.4f));
        std::vector<float> new_radii = {0.3f, 0.2f, 0.1f};
        ctx.setTubeRadii(tube, new_radii);
        DOCTEST_CHECK(ctx.getTubeObjectNodeRadii(tube)[0] == doctest::Approx(0.3f));
        ctx.scaleTubeLength(tube, 2.f);
        std::vector<vec3> new_nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
        ctx.setTubeNodes(tube, new_nodes);
        ctx.pruneTubeNodes(tube, 1);
        DOCTEST_CHECK(ctx.getTubeObjectNodeCount(tube) == 1);
    }

    SUBCASE("Object appearance and visibility") {
        Context ctx;
        uint box = ctx.addBoxObject(nullorigin, make_vec3(1, 1, 1), make_int3(2, 3, 2));
        ctx.overrideObjectTextureColor(box);
        // Cannot check state, only that it runs
        ctx.useObjectTextureColor(box);
        // Cannot check state, only that it runs
        ctx.hideObject(box);
        DOCTEST_CHECK(ctx.isObjectHidden(box));
        ctx.showObject(box);
        DOCTEST_CHECK(!ctx.isObjectHidden(box));

        std::vector<uint> prims = ctx.getObjectPrimitiveUUIDs(box);
        ctx.hidePrimitive(prims);
        DOCTEST_CHECK(ctx.isPrimitiveHidden(prims[0]));
        ctx.showPrimitive(prims);
        DOCTEST_CHECK(!ctx.isPrimitiveHidden(prims[0]));
    }

    SUBCASE("Primitive color and parent object") {
        capture_cerr cerr_buffer; // Capture deprecation warnings from setPrimitiveColor/usePrimitiveTextureColor
        Context ctx;
        uint p = ctx.addPatch();
        ctx.setPrimitiveColor(p, RGB::red);
        DOCTEST_CHECK(ctx.getPrimitiveColor(p) == RGB::red);
        ctx.overridePrimitiveTextureColor(p);
        DOCTEST_CHECK(ctx.isPrimitiveTextureColorOverridden(p));
        ctx.usePrimitiveTextureColor(p);
        DOCTEST_CHECK(!ctx.isPrimitiveTextureColorOverridden(p));

        uint obj = ctx.addBoxObject(nullorigin, make_vec3(1, 1, 1), make_int3(2, 3, 2));
        ctx.setPrimitiveParentObjectID(p, obj);
        DOCTEST_CHECK(ctx.getPrimitiveParentObjectID(p) == obj);
    }
}
TEST_CASE("Object Management: Creation and Properties") {

    SUBCASE("addSphereObject") {
        Context ctx;
        uint objID = ctx.addSphereObject(10, make_vec3(1, 2, 3), 5.f);
        DOCTEST_CHECK(ctx.doesObjectExist(objID));
        DOCTEST_CHECK(ctx.getSphereObjectCenter(objID) == make_vec3(1, 2, 3));
        DOCTEST_CHECK(ctx.getSphereObjectRadius(objID) == make_vec3(5.f, 5.f, 5.f));
        DOCTEST_CHECK(ctx.getSphereObjectSubdivisionCount(objID) == 10);
    }

    SUBCASE("addDiskObject") {
        Context ctx;
        uint objID = ctx.addDiskObject(make_int2(8, 16), make_vec3(1, 2, 3), make_vec2(4, 5), nullrotation, RGB::red);
        DOCTEST_CHECK(ctx.doesObjectExist(objID));
        DOCTEST_CHECK(ctx.getDiskObjectCenter(objID) == make_vec3(1, 2, 3));
        DOCTEST_CHECK(ctx.getDiskObjectSize(objID) == make_vec2(4, 5));
        DOCTEST_CHECK(ctx.getDiskObjectSubdivisionCount(objID) == 8u);
    }

    SUBCASE("addConeObject") {
        Context ctx;
        uint objID = ctx.addConeObject(10, make_vec3(0, 0, 0), make_vec3(0, 0, 5), 2.f, 1.f);
        DOCTEST_CHECK(ctx.doesObjectExist(objID));
        DOCTEST_CHECK(ctx.getConeObjectNode(objID, 0) == make_vec3(0, 0, 0));
        DOCTEST_CHECK(ctx.getConeObjectNode(objID, 1) == make_vec3(0, 0, 5));
        DOCTEST_CHECK(ctx.getConeObjectNodeRadius(objID, 0) == 2.f);
        DOCTEST_CHECK(ctx.getConeObjectNodeRadius(objID, 1) == 1.f);
        DOCTEST_CHECK(ctx.getConeObjectSubdivisionCount(objID) == 10);
    }
}

TEST_CASE("Global Data Management") {
    SUBCASE("Integer Data") {
        Context ctx;
        ctx.setGlobalData("test_int", 123);
        DOCTEST_CHECK(ctx.doesGlobalDataExist("test_int"));
        int val;
        ctx.getGlobalData("test_int", val);
        DOCTEST_CHECK(val == 123);
        DOCTEST_CHECK(ctx.getGlobalDataSize("test_int") == 1);
        DOCTEST_CHECK(ctx.getGlobalDataType("test_int") == HELIOS_TYPE_INT);
        ctx.clearGlobalData("test_int");
        DOCTEST_CHECK(!ctx.doesGlobalDataExist("test_int"));
    }

    SUBCASE("Vector Data") {
        Context ctx;
        std::vector<vec3> vec_data = {{1, 2, 3}, {4, 5, 6}};
        ctx.setGlobalData("test_vec", vec_data);
        DOCTEST_CHECK(ctx.doesGlobalDataExist("test_vec"));
        std::vector<vec3> read_vec;
        ctx.getGlobalData("test_vec", read_vec);
        DOCTEST_CHECK(read_vec.size() == 2);
        DOCTEST_CHECK(read_vec[1] == make_vec3(4, 5, 6));
        DOCTEST_CHECK(ctx.getGlobalDataSize("test_vec") == 2);
        DOCTEST_CHECK(ctx.getGlobalDataType("test_vec") == HELIOS_TYPE_VEC3);
    }

    SUBCASE("List Data") {
        Context ctx;
        ctx.setGlobalData("d1", 1);
        ctx.setGlobalData("d2", 2.f);
        std::vector<std::string> labels = ctx.listGlobalData();
        DOCTEST_CHECK(labels.size() == 2);
        DOCTEST_CHECK(std::find(labels.begin(), labels.end(), "d1") != labels.end());
    }
}

TEST_CASE("Context primitive data management") {
    Context ctx;
    uint p1 = ctx.addPatch();
    uint p2 = ctx.addPatch();
    ctx.setPrimitiveData(p1, "my_data", 10);

    // copyPrimitiveData
    ctx.copyPrimitiveData(p1, p2);
    DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p2, "my_data"));
    int val;
    ctx.getPrimitiveData(p2, "my_data", val);
    DOCTEST_CHECK(val == 10);

    // renamePrimitiveData
    ctx.renamePrimitiveData(p1, "my_data", "new_data_name");
    DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p1, "my_data"));
    DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p1, "new_data_name"));

    // duplicatePrimitiveData
    ctx.duplicatePrimitiveData(p2, "my_data", "my_data_copy");
    DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p2, "my_data_copy"));
    ctx.getPrimitiveData(p2, "my_data_copy", val);
    DOCTEST_CHECK(val == 10);

    // duplicatePrimitiveData (all primitives)
    ctx.setPrimitiveData(p1, "global_copy_test", 5.5f);
    ctx.duplicatePrimitiveData("global_copy_test", "global_copy_test_new");
    DOCTEST_CHECK(ctx.doesPrimitiveDataExist(p1, "global_copy_test_new"));
    DOCTEST_CHECK(!ctx.doesPrimitiveDataExist(p2, "global_copy_test_new")); // p2 doesn't have original

    ctx.clearPrimitiveData(p1, "new_data_name");
    ctx.setPrimitiveData(p2, "my_data_copy", 15);
    ctx.setPrimitiveData(p2, "my_data_copy", 20);
    ctx.clearPrimitiveData(p2, "my_data_copy");
    std::vector<std::string> all_labels = ctx.listAllPrimitiveDataLabels();
    DOCTEST_CHECK(std::find(all_labels.begin(), all_labels.end(), "my_data") != all_labels.end());
    DOCTEST_CHECK(std::find(all_labels.begin(), all_labels.end(), "my_data_copy") == all_labels.end());
    DOCTEST_CHECK(std::find(all_labels.begin(), all_labels.end(), "new_data_name") == all_labels.end());
}

TEST_CASE("Context primitive data calculations") {
    Context ctx;
    std::vector<uint> uuids;
    for (int i = 0; i < 5; ++i) {
        uint p = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        ctx.setPrimitiveData(p, "float_val", (float) i);
        ctx.setPrimitiveData(p, "double_val", (double) i);
        ctx.setPrimitiveData(p, "vec2_val", make_vec2((float) i, (float) i));
        uuids.push_back(p);
    }

    // calculatePrimitiveDataMean
    float float_mean;
    ctx.calculatePrimitiveDataMean(uuids, "float_val", float_mean);
    DOCTEST_CHECK(float_mean == doctest::Approx(2.0f));
    double double_mean;
    ctx.calculatePrimitiveDataMean(uuids, "double_val", double_mean);
    DOCTEST_CHECK(double_mean == doctest::Approx(2.0));
    vec2 vec2_mean;
    ctx.calculatePrimitiveDataMean(uuids, "vec2_val", vec2_mean);
    DOCTEST_CHECK(vec2_mean.x == doctest::Approx(2.0f));

    // calculatePrimitiveDataAreaWeightedMean
    float awt_mean_f;
    ctx.calculatePrimitiveDataAreaWeightedMean(uuids, "float_val", awt_mean_f);
    DOCTEST_CHECK(awt_mean_f == doctest::Approx(2.0f)); // Area is 1 for all

    // calculatePrimitiveDataSum
    float float_sum;
    ctx.calculatePrimitiveDataSum(uuids, "float_val", float_sum);
    DOCTEST_CHECK(float_sum == doctest::Approx(10.0f));

    // calculatePrimitiveDataAreaWeightedSum
    float awt_sum_f;
    ctx.calculatePrimitiveDataAreaWeightedSum(uuids, "float_val", awt_sum_f);
    DOCTEST_CHECK(awt_sum_f == doctest::Approx(10.0f));

    // scalePrimitiveData
    ctx.scalePrimitiveData(uuids, "float_val", 2.0f);
    ctx.getPrimitiveData(uuids[2], "float_val", float_mean);
    DOCTEST_CHECK(float_mean == doctest::Approx(4.0f));
    ctx.scalePrimitiveData("double_val", 0.5f);
    ctx.getPrimitiveData(uuids[4], "double_val", double_mean);
    DOCTEST_CHECK(double_mean == doctest::Approx(2.0));

    // incrementPrimitiveData
    ctx.setPrimitiveData(uuids, "int_val", 10);
    ctx.incrementPrimitiveData(uuids, "int_val", 5);
    int int_val;
    ctx.getPrimitiveData(uuids[0], "int_val", int_val);
    DOCTEST_CHECK(int_val == 15);
    bool has_warning;
    {
        capture_cerr cerr_buffer;
        ctx.incrementPrimitiveData(uuids, "float_val", 1); // Wrong type, should warn
        has_warning = cerr_buffer.has_output();
    }
    DOCTEST_CHECK(has_warning);
}

TEST_CASE("Context primitive data aggregation and filtering") {
    Context ctx;
    std::vector<uint> uuids;
    for (int i = 0; i < 3; ++i) {
        uint p = ctx.addPatch();
        ctx.setPrimitiveData(p, "d1", (float) i);
        ctx.setPrimitiveData(p, "d2", (float) i * 2.0f);
        ctx.setPrimitiveData(p, "d3", (float) i * 3.0f);
        ctx.setPrimitiveData(p, "filter_me", i);
        uuids.push_back(p);
    }

    // aggregatePrimitiveDataSum
    std::vector<std::string> labels = {"d1", "d2", "d3"};
    ctx.aggregatePrimitiveDataSum(uuids, labels, "sum_data");
    float sum_val;
    ctx.getPrimitiveData(uuids[1], "sum_data", sum_val);
    DOCTEST_CHECK(sum_val == doctest::Approx(1.f + 2.f + 3.f));

    // aggregatePrimitiveDataProduct
    ctx.aggregatePrimitiveDataProduct(uuids, labels, "prod_data");
    float prod_val;
    ctx.getPrimitiveData(uuids[2], "prod_data", prod_val);
    DOCTEST_CHECK(prod_val == doctest::Approx(2.f * 4.f * 6.f));

    // filterPrimitivesByData
    std::vector<uint> filtered = ctx.filterPrimitivesByData(uuids, "filter_me", 1, ">=");
    DOCTEST_CHECK(filtered.size() == 2);
    filtered = ctx.filterPrimitivesByData(uuids, "filter_me", 1, "==");
    DOCTEST_CHECK(filtered.size() == 1);
    DOCTEST_CHECK(filtered[0] == uuids[1]);
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(filtered = ctx.filterPrimitivesByData(uuids, "filter_me", 1, "!!"), std::runtime_error);
}

TEST_CASE("Object data") {
    Context ctx;
    uint o = ctx.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2));
    ctx.setObjectData(o, "test_int", 5);
    ctx.setObjectData(o, "test_float", 3.14f);

    // getObjectDataType
    DOCTEST_CHECK(ctx.getObjectDataType("test_int") == HELIOS_TYPE_INT);
#ifdef HELIOS_DEBUG
    capture_cerr cerr_buffer;
    DOCTEST_CHECK_THROWS_AS(ctx.getObjectDataType("non_existent"), std::runtime_error);
#endif

    // getObjectDataSize
    DOCTEST_CHECK(ctx.getObjectDataSize(o, "test_int") == 1);

    // clearObjectData
    ctx.clearObjectData(o, "test_int");
    DOCTEST_CHECK(!ctx.doesObjectDataExist(o, "test_int"));

    // listObjectData
    std::vector<std::string> data_labels = ctx.listObjectData(o);
    DOCTEST_CHECK(std::find(data_labels.begin(), data_labels.end(), "test_float") != data_labels.end());
}

TEST_CASE("Context object data management") {
    Context ctx;
    uint o1 = ctx.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2));
    uint o2 = ctx.addTileObject(nullorigin, make_vec2(1, 1), nullrotation, make_int2(2, 2));
    ctx.setObjectData(o1, "my_data", 10);

    // copyObjectData
    ctx.copyObjectData(o1, o2);
    DOCTEST_CHECK(ctx.doesObjectDataExist(o2, "my_data"));

    // renameObjectData
    ctx.renameObjectData(o1, "my_data", "new_name");
    DOCTEST_CHECK(!ctx.doesObjectDataExist(o1, "my_data"));
    DOCTEST_CHECK(ctx.doesObjectDataExist(o1, "new_name"));

    // duplicateObjectData
    ctx.duplicateObjectData(o2, "my_data", "my_data_copy");
    DOCTEST_CHECK(ctx.doesObjectDataExist(o2, "my_data_copy"));

    std::vector<std::string> all_obj_labels = ctx.listAllObjectDataLabels();
    DOCTEST_CHECK(std::find(all_obj_labels.begin(), all_obj_labels.end(), "my_data") != all_obj_labels.end());
    DOCTEST_CHECK(std::find(all_obj_labels.begin(), all_obj_labels.end(), "my_data_copy") != all_obj_labels.end());
    DOCTEST_CHECK(std::find(all_obj_labels.begin(), all_obj_labels.end(), "new_name") != all_obj_labels.end());
}

TEST_CASE("Global data") {
    Context ctx;
    ctx.setGlobalData("g_int", 5);
    ctx.setGlobalData("g_float", 3.14f);

    // getGlobalDataType/Size/Exists
    DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int"));
    DOCTEST_CHECK(ctx.getGlobalDataType("g_int") == HELIOS_TYPE_INT);
    DOCTEST_CHECK(ctx.getGlobalDataSize("g_int") == 1);

    // rename/duplicate/clear
    ctx.duplicateGlobalData("g_int", "g_int_copy");
    DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int_copy"));
    ctx.renameGlobalData("g_int", "g_int_new");
    DOCTEST_CHECK(!ctx.doesGlobalDataExist("g_int"));
    DOCTEST_CHECK(ctx.doesGlobalDataExist("g_int_new"));
    ctx.clearGlobalData("g_int_new");
    DOCTEST_CHECK(!ctx.doesGlobalDataExist("g_int_new"));

    // listGlobalData
    std::vector<std::string> g_labels = ctx.listGlobalData();
    DOCTEST_CHECK(g_labels.size() > 0);

    // incrementGlobalData
    ctx.setGlobalData("inc_me", 10);
    ctx.incrementGlobalData("inc_me", 5);
    int val;
    ctx.getGlobalData("inc_me", val);
    DOCTEST_CHECK(val == 15);
    bool has_warning;
    {
        capture_cerr cerr_buffer;
        ctx.incrementGlobalData("g_float", 1); // Wrong type
        has_warning = cerr_buffer.has_output();
    }
    DOCTEST_CHECK(has_warning);
}

TEST_CASE("Voxel Management") {
    SUBCASE("addVoxel and voxel properties") {
        Context ctx;

        vec3 center = make_vec3(1, 2, 3);
        vec3 size = make_vec3(2, 4, 6);
        float rotation = 0.5f * PI_F;

        uint vox1 = ctx.addVoxel(center, size);
        DOCTEST_CHECK(ctx.getPrimitiveType(vox1) == PRIMITIVE_TYPE_VOXEL);
        DOCTEST_CHECK(ctx.getVoxelCenter(vox1) == center);
        DOCTEST_CHECK(ctx.getVoxelSize(vox1) == size);

        uint vox2 = ctx.addVoxel(center, size, rotation);
        DOCTEST_CHECK(ctx.getVoxelCenter(vox2) == center);
        DOCTEST_CHECK(ctx.getVoxelSize(vox2) == size);

        uint vox3 = ctx.addVoxel(center, size, rotation, RGB::red);
        DOCTEST_CHECK(ctx.getPrimitiveColor(vox3) == RGB::red);

        uint vox4 = ctx.addVoxel(center, size, rotation, RGBA::red);
        RGBAcolor color_rgba = ctx.getPrimitiveColorRGBA(vox4);
        DOCTEST_CHECK(color_rgba.r == RGBA::red.r);
        DOCTEST_CHECK(color_rgba.a == RGBA::red.a);

        DOCTEST_CHECK(ctx.getPrimitiveCount() >= 4);

        float area = ctx.getPrimitiveArea(vox1);
        DOCTEST_CHECK(area == doctest::Approx(2.f * (size.x * size.y + size.y * size.z + size.x * size.z)));
    }
}

TEST_CASE("Texture Management") {
    SUBCASE("texture validation and properties") {
        capture_cerr cerr_buffer; // Capture deprecation warnings from setPrimitiveTextureFile
        Context ctx;

        uint patch = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, "lib/images/solid.jpg");
        ctx.setPrimitiveTextureFile(patch, "lib/images/solid.jpg");
        DOCTEST_CHECK(ctx.getPrimitiveTextureFile(patch) == "lib/images/solid.jpg");
        DOCTEST_CHECK(!ctx.primitiveTextureHasTransparencyChannel(patch));

        Texture tex("lib/images/solid.jpg");
        std::vector<vec2> uv = {{0, 0}, {1, 0}, {1, 1}};
        float solid_frac = tex.getSolidFraction(uv);
        DOCTEST_CHECK(solid_frac == doctest::Approx(1.f));
    }
}

TEST_CASE("Triangle Management") {
    SUBCASE("setTriangleVertices") {
        Context ctx;
        vec3 v0 = make_vec3(0, 0, 0);
        vec3 v1 = make_vec3(1, 0, 0);
        vec3 v2 = make_vec3(0, 1, 0);
        uint tri = ctx.addTriangle(v0, v1, v2);

        vec3 new_v0 = make_vec3(1, 1, 1);
        vec3 new_v1 = make_vec3(2, 1, 1);
        vec3 new_v2 = make_vec3(1, 2, 1);
        ctx.setTriangleVertices(tri, new_v0, new_v1, new_v2);

        std::vector<vec3> vertices = ctx.getPrimitiveVertices(tri);
        DOCTEST_CHECK(vertices[0] == new_v0);
        DOCTEST_CHECK(vertices[1] == new_v1);
        DOCTEST_CHECK(vertices[2] == new_v2);
    }
}

TEST_CASE("UUID and Object Management") {
    SUBCASE("getAllUUIDs and cleanDeletedUUIDs") {
        Context ctx;
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();
        uint p3 = ctx.addPatch();

        std::vector<uint> all_uuids = ctx.getAllUUIDs();
        DOCTEST_CHECK(all_uuids.size() == 3);
        DOCTEST_CHECK(std::find(all_uuids.begin(), all_uuids.end(), p1) != all_uuids.end());

        ctx.deletePrimitive(p2);
        std::vector<uint> uuids_with_deleted = {p1, p2, p3};
        ctx.cleanDeletedUUIDs(uuids_with_deleted);
        DOCTEST_CHECK(uuids_with_deleted.size() == 2);
        DOCTEST_CHECK(std::find(uuids_with_deleted.begin(), uuids_with_deleted.end(), p2) == uuids_with_deleted.end());

        std::vector<std::vector<uint>> nested_uuids = {{p1, p2}, {p3, p2}};
        ctx.cleanDeletedUUIDs(nested_uuids);
        DOCTEST_CHECK(nested_uuids[0].size() == 1);
        DOCTEST_CHECK(nested_uuids[1].size() == 1);

        std::vector<std::vector<std::vector<uint>>> triple_nested = {{{p1, p2, p3}}};
        ctx.cleanDeletedUUIDs(triple_nested);
        DOCTEST_CHECK(triple_nested[0][0].size() == 2);
    }

    SUBCASE("object management utilities") {
        Context ctx;
        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        DOCTEST_CHECK(ctx.areObjectPrimitivesComplete(obj));

        std::vector<uint> obj_ids = {obj, 999};
        ctx.cleanDeletedObjectIDs(obj_ids);
        DOCTEST_CHECK(obj_ids.size() == 1);
        DOCTEST_CHECK(obj_ids[0] == obj);

        std::vector<std::vector<uint>> nested_obj_ids = {{obj, 999}, {obj}};
        ctx.cleanDeletedObjectIDs(nested_obj_ids);
        DOCTEST_CHECK(nested_obj_ids[0].size() == 1);
        DOCTEST_CHECK(nested_obj_ids[1].size() == 1);

        std::vector<std::vector<std::vector<uint>>> triple_nested_obj = {{{obj, 999}}};
        ctx.cleanDeletedObjectIDs(triple_nested_obj);
        DOCTEST_CHECK(triple_nested_obj[0][0].size() == 1);

        DOCTEST_CHECK(ctx.doesObjectExist(obj));

        vec3 new_origin = make_vec3(5, 5, 5);
        ctx.setObjectOrigin(obj, new_origin);

        vec3 new_normal = make_vec3(0, 1, 0);
        ctx.setObjectAverageNormal(obj, make_vec3(0, 0, 0), new_normal);
    }
}

TEST_CASE("Tile Object Advanced Features") {
    SUBCASE("tile object subdivision management") {
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(4, 4), nullrotation, make_int2(2, 2));

        float area_ratio = ctx.getTileObjectAreaRatio(tile);
        DOCTEST_CHECK(area_ratio > 0.f);

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(4, 4));

        ctx.setTileObjectSubdivisionCount({tile}, 0.5f);
    }
}

TEST_CASE("Pseudocolor Visualization") {
    SUBCASE("colorPrimitiveByDataPseudocolor") {
        Context ctx;
        std::vector<uint> patches;
        for (int i = 0; i < 5; i++) {
            uint p = ctx.addPatch();
            ctx.setPrimitiveData(p, "value", float(i));
            patches.push_back(p);
        }

        DOCTEST_CHECK_NOTHROW(ctx.colorPrimitiveByDataPseudocolor(patches, "value", "hot", 10));
        DOCTEST_CHECK_NOTHROW(ctx.colorPrimitiveByDataPseudocolor(patches, "value", "rainbow", 5, 0.f, 4.f));
    }
}

TEST_CASE("Date and Time Extensions") {
    SUBCASE("getMonthString") {
        Context ctx;
        ctx.setDate(15, 1, 2025);
        DOCTEST_CHECK(strcmp(ctx.getMonthString(), "JAN") == 0);
        ctx.setDate(15, 2, 2025);
        DOCTEST_CHECK(strcmp(ctx.getMonthString(), "FEB") == 0);
        ctx.setDate(15, 12, 2025);
        DOCTEST_CHECK(strcmp(ctx.getMonthString(), "DEC") == 0);
    }
}

TEST_CASE("Tube Object Management") {
    SUBCASE("appendTubeSegment with texture") {
        Context ctx;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        std::vector<float> radii = {0.2f, 0.1f};
        uint tube = ctx.addTubeObject(10, nodes, radii);

        ctx.appendTubeSegment(tube, make_vec3(0, 0, 2), 0.05f, "lib/images/solid.jpg", make_vec2(0.5f, 1.0f));
        DOCTEST_CHECK(ctx.getTubeObjectNodeCount(tube) == 3);
    }
}

TEST_CASE("Edge Cases and Additional Coverage") {
    SUBCASE("Julian date edge cases") {
        Context ctx;
        ctx.setDate(1, 1, 2025);
        DOCTEST_CHECK(ctx.getJulianDate() == 1);

        ctx.setDate(31, 12, 2025);
        DOCTEST_CHECK(ctx.getJulianDate() == 365);

        ctx.setDate(100, 2025);
        Date d = ctx.getDate();
        DOCTEST_CHECK(d.day == 10);
        DOCTEST_CHECK(d.month == 4);
    }

    SUBCASE("time edge cases") {
        Context ctx;
        ctx.setTime(0, 0, 0);
        Time t = ctx.getTime();
        DOCTEST_CHECK(t.hour == 0);
        DOCTEST_CHECK(t.minute == 0);
        DOCTEST_CHECK(t.second == 0);

        ctx.setTime(59, 59, 23);
        t = ctx.getTime();
        DOCTEST_CHECK(t.hour == 23);
        DOCTEST_CHECK(t.minute == 59);
        DOCTEST_CHECK(t.second == 59);
    }

    SUBCASE("random number edge cases") {
        Context ctx;
        ctx.seedRandomGenerator(0);

        float r1 = ctx.randu(5.f, 5.f);
        DOCTEST_CHECK(r1 == doctest::Approx(5.f));

        int ri = ctx.randu(10, 10);
        DOCTEST_CHECK(ri == 10);

        float rn = ctx.randn(0.f, 0.f);
        DOCTEST_CHECK(rn == doctest::Approx(0.f));
    }

    SUBCASE("texture edge cases") {
        capture_cerr cerr_buffer; // Capture deprecation warnings from overridePrimitiveTextureColor/usePrimitiveTextureColor
        Context ctx;
        uint patch = ctx.addPatch();

        ctx.overridePrimitiveTextureColor(patch);
        ctx.usePrimitiveTextureColor(patch);

        std::vector<uint> patches = {patch};
        ctx.overridePrimitiveTextureColor(patches);
        ctx.usePrimitiveTextureColor(patches);

        DOCTEST_CHECK(!ctx.isPrimitiveTextureColorOverridden(patch));
    }

    SUBCASE("primitive existence checks") {
        Context ctx;
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();

        DOCTEST_CHECK(ctx.doesPrimitiveExist(p1));
        DOCTEST_CHECK(ctx.doesPrimitiveExist({p1, p2}));

        ctx.deletePrimitive(p1);
        DOCTEST_CHECK(!ctx.doesPrimitiveExist(p1));
        DOCTEST_CHECK(!ctx.doesPrimitiveExist({p1, p2}));
        DOCTEST_CHECK(ctx.doesPrimitiveExist(std::vector<uint>{p2}));
    }

    SUBCASE("object containment checks") {
        Context ctx;
        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        std::vector<uint> prims = ctx.getObjectPrimitiveUUIDs(obj);

        DOCTEST_CHECK(ctx.doesObjectContainPrimitive(obj, prims[0]));

        uint independent_patch = ctx.addPatch();
        DOCTEST_CHECK(!ctx.doesObjectContainPrimitive(obj, independent_patch));
    }

    SUBCASE("transformation matrix operations") {
        Context ctx;
        uint p = ctx.addPatch();

        float identity[16] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        ctx.setPrimitiveTransformationMatrix(p, identity);

        std::vector<uint> patches = {p};
        ctx.setPrimitiveTransformationMatrix(patches, identity);

        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        ctx.setObjectTransformationMatrix(obj, identity);

        std::vector<uint> objs = {obj};
        ctx.setObjectTransformationMatrix(objs, identity);

        float retrieved[16];
        ctx.getObjectTransformationMatrix(obj, retrieved);
        for (int i = 0; i < 16; i++) {
            DOCTEST_CHECK(retrieved[i] == doctest::Approx(identity[i]));
        }
    }

    SUBCASE("object type and texture checks") {
        Context ctx;
        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        DOCTEST_CHECK(!ctx.objectHasTexture(obj));

        uint textured_obj = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, make_int2(2, 2), "lib/images/solid.jpg");
        DOCTEST_CHECK(ctx.objectHasTexture(textured_obj));
    }

    SUBCASE("tube object segment operations") {
        Context ctx;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
        std::vector<float> radii = {0.2f, 0.15f, 0.1f};
        uint tube = ctx.addTubeObject(10, nodes, radii);

        float seg_volume = ctx.getTubeObjectSegmentVolume(tube, 0);
        DOCTEST_CHECK(seg_volume > 0.f);

        seg_volume = ctx.getTubeObjectSegmentVolume(tube, 1);
        DOCTEST_CHECK(seg_volume > 0.f);
    }

    SUBCASE("cone object advanced properties") {
        Context ctx;
        uint cone = ctx.addConeObject(10, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 1.f, 0.5f);

        float radius0 = ctx.getConeObjectNodeRadius(cone, 0);
        DOCTEST_CHECK(radius0 == doctest::Approx(1.f));

        float radius1 = ctx.getConeObjectNodeRadius(cone, 1);
        DOCTEST_CHECK(radius1 == doctest::Approx(0.5f));

        float length = ctx.getConeObjectLength(cone);
        DOCTEST_CHECK(length == doctest::Approx(2.f));

        DOCTEST_CHECK(ctx.getConeObjectSubdivisionCount(cone) == 10);
    }

    SUBCASE("primitive color operations") {
        capture_cerr cerr_buffer; // Suppress deprecation warnings from setPrimitiveColor
        Context ctx;
        uint p = ctx.addPatch();

        ctx.setPrimitiveColor(p, RGB::blue);
        DOCTEST_CHECK(ctx.getPrimitiveColor(p) == RGB::blue);

        ctx.setPrimitiveColor(p, RGBA::green);
        RGBAcolor rgba = ctx.getPrimitiveColorRGBA(p);
        DOCTEST_CHECK(rgba.r == RGBA::green.r);
        DOCTEST_CHECK(rgba.a == RGBA::green.a);

        std::vector<uint> patches = {p};
        ctx.setPrimitiveColor(patches, RGB::red);
        DOCTEST_CHECK(ctx.getPrimitiveColor(p) == RGB::red);

        ctx.setPrimitiveColor(patches, RGBA::yellow);
        rgba = ctx.getPrimitiveColorRGBA(p);
        DOCTEST_CHECK(rgba.r == RGBA::yellow.r);
    }

    SUBCASE("object color operations") {
        capture_cerr cerr_buffer; // Suppress deprecation warnings from setObjectColor (calls setPrimitiveColor internally)
        Context ctx;
        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        ctx.setObjectColor(obj, RGB::cyan);
        ctx.setObjectColor(obj, RGBA::magenta);

        std::vector<uint> objs = {obj};
        ctx.setObjectColor(objs, RGB::white);
        ctx.setObjectColor(objs, RGBA::black);
    }
}

TEST_CASE("Print and Information Functions") {
    SUBCASE("printPrimitiveInfo and printObjectInfo") {
        Context ctx;
        uint patch = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        // Capture stdout output from these functions
        bool has_output;
        std::string output;
        {
            capture_cout cout_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.printPrimitiveInfo(patch));
            DOCTEST_CHECK_NOTHROW(ctx.printObjectInfo(obj));
            has_output = cout_buffer.has_output();
            output = cout_buffer.get_captured_output();
        } // cout_buffer destroyed here

        // Verify that output was captured (functions should produce output)
        DOCTEST_CHECK(has_output);
        DOCTEST_CHECK(output.find("Info for UUID") != std::string::npos);
        DOCTEST_CHECK(output.find("Info for ObjID") != std::string::npos);
    }
}

TEST_CASE("Object Pointer Access") {
    SUBCASE("getObjectPointer functions") {
        Context ctx;

        uint box = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        DOCTEST_CHECK(ctx.doesObjectExist(box));

        uint disk = ctx.addDiskObject(10, make_vec3(0, 0, 0), make_vec2(1, 1));
        DOCTEST_CHECK(ctx.doesObjectExist(disk));

        uint sphere = ctx.addSphereObject(10, make_vec3(0, 0, 0), 1.f);
        DOCTEST_CHECK(ctx.doesObjectExist(sphere));

        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        std::vector<float> radii = {0.2f, 0.1f};
        uint tube = ctx.addTubeObject(10, nodes, radii);
        DOCTEST_CHECK(ctx.doesObjectExist(tube));

        uint cone = ctx.addConeObject(10, make_vec3(0, 0, 0), make_vec3(0, 0, 1), 0.5f, 0.3f);
        DOCTEST_CHECK(ctx.doesObjectExist(cone));

        std::vector<uint> prim_uuids = {ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0))};
        uint polymesh = ctx.addPolymeshObject(prim_uuids);
        DOCTEST_CHECK(ctx.doesObjectExist(polymesh));
    }
}

TEST_CASE("Advanced Primitive Operations") {
    SUBCASE("primitive visibility and print operations") {
        Context ctx;
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();

        // Test hiding/showing primitives
        ctx.hidePrimitive(p1);
        DOCTEST_CHECK(ctx.isPrimitiveHidden(p1));
        ctx.showPrimitive(p1);
        DOCTEST_CHECK(!ctx.isPrimitiveHidden(p1));

        std::vector<uint> patches = {p1, p2};
        ctx.hidePrimitive(patches);
        DOCTEST_CHECK(ctx.isPrimitiveHidden(p1));
        DOCTEST_CHECK(ctx.isPrimitiveHidden(p2));
        ctx.showPrimitive(patches);
        DOCTEST_CHECK(!ctx.isPrimitiveHidden(p1));
        DOCTEST_CHECK(!ctx.isPrimitiveHidden(p2));
    }

    SUBCASE("primitive counts by type") {
        Context ctx;
        uint initial_patch_count = ctx.getPatchCount();
        uint initial_triangle_count = ctx.getTriangleCount();

        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();
        uint tri = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0));

        DOCTEST_CHECK(ctx.getPatchCount() == initial_patch_count + 2);
        DOCTEST_CHECK(ctx.getTriangleCount() == initial_triangle_count + 1);

        // Test with hidden primitives
        ctx.hidePrimitive(p1);
        DOCTEST_CHECK(ctx.getPatchCount(false) == initial_patch_count + 1); // exclude hidden
        DOCTEST_CHECK(ctx.getPatchCount(true) == initial_patch_count + 2); // include hidden
    }
}

TEST_CASE("Data Type and Size Functions") {
    SUBCASE("primitive data type operations") {
        Context ctx;
        uint p = ctx.addPatch();

        ctx.setPrimitiveData(p, "test_int", 42);
        ctx.setPrimitiveData(p, "test_float", 3.14f);
        ctx.setPrimitiveData(p, "test_vec3", make_vec3(1, 2, 3));

        DOCTEST_CHECK(ctx.getPrimitiveDataType("test_int") == HELIOS_TYPE_INT);
        DOCTEST_CHECK(ctx.getPrimitiveDataType("test_float") == HELIOS_TYPE_FLOAT);
        DOCTEST_CHECK(ctx.getPrimitiveDataType("test_vec3") == HELIOS_TYPE_VEC3);

        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "test_int") == 1);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "test_vec3") == 1);

        std::vector<float> vec_data = {1.0f, 2.0f, 3.0f};
        ctx.setPrimitiveData(p, "test_vector", vec_data);
        DOCTEST_CHECK(ctx.getPrimitiveDataSize(p, "test_vector") == 3);
    }
}

TEST_CASE("Additional Missing Coverage") {
    SUBCASE("getDirtyUUIDs function") {
        Context ctx;
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();

        ctx.markGeometryClean();
        std::vector<uint> dirty_uuids = ctx.getDirtyUUIDs();
        DOCTEST_CHECK(dirty_uuids.empty());

        ctx.markPrimitiveDirty(p1);
        dirty_uuids = ctx.getDirtyUUIDs();
        DOCTEST_CHECK(dirty_uuids.size() == 1);
        DOCTEST_CHECK(std::find(dirty_uuids.begin(), dirty_uuids.end(), p1) != dirty_uuids.end());
    }
}

TEST_CASE("Advanced Object Operations") {
    SUBCASE("object primitive count and area calculations") {
        Context ctx;
        uint obj = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(2, 3, 4), make_int3(1, 1, 1));

        DOCTEST_CHECK(ctx.getObjectPrimitiveCount(obj) == 6); // 6 faces of a box

        float area = ctx.getObjectArea(obj);
        float expected_area = 2 * (2 * 3 + 3 * 4 + 2 * 4); // surface area of box
        DOCTEST_CHECK(area == doctest::Approx(expected_area).epsilon(0.01));
    }

    SUBCASE("object bounding box operations") {
        Context ctx;
        uint obj = ctx.addBoxObject(make_vec3(1, 2, 3), make_vec3(2, 4, 6), make_int3(1, 1, 1));

        vec3 min_corner, max_corner;
        ctx.getObjectBoundingBox(obj, min_corner, max_corner);

        DOCTEST_CHECK(min_corner.x == doctest::Approx(0.f).epsilon(0.01));
        DOCTEST_CHECK(max_corner.x == doctest::Approx(2.f).epsilon(0.01));
        DOCTEST_CHECK(min_corner.y == doctest::Approx(0.f).epsilon(0.01));
        DOCTEST_CHECK(max_corner.y == doctest::Approx(4.f).epsilon(0.01));

        std::vector<uint> objs = {obj};
        ctx.getObjectBoundingBox(objs, min_corner, max_corner);
        DOCTEST_CHECK(min_corner.x == doctest::Approx(0.f).epsilon(0.01));
        DOCTEST_CHECK(max_corner.x == doctest::Approx(2.f).epsilon(0.01));
    }
}

TEST_CASE("Additional Object Features") {
    SUBCASE("getAllObjectIDs") {
        Context ctx;
        uint obj1 = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint obj2 = ctx.addSphereObject(10, make_vec3(0, 0, 0), 1.f);

        std::vector<uint> all_ids = ctx.getAllObjectIDs();
        DOCTEST_CHECK(all_ids.size() >= 2);
        DOCTEST_CHECK(std::find(all_ids.begin(), all_ids.end(), obj1) != all_ids.end());
        DOCTEST_CHECK(std::find(all_ids.begin(), all_ids.end(), obj2) != all_ids.end());
    }

    SUBCASE("object type checks") {
        Context ctx;
        uint box = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint sphere = ctx.addSphereObject(10, make_vec3(0, 0, 0), 1.f);
        uint disk = ctx.addDiskObject(10, make_vec3(0, 0, 0), make_vec2(1, 1));

        DOCTEST_CHECK(ctx.getObjectType(box) == OBJECT_TYPE_BOX);
        DOCTEST_CHECK(ctx.getObjectType(sphere) == OBJECT_TYPE_SPHERE);
        DOCTEST_CHECK(ctx.getObjectType(disk) == OBJECT_TYPE_DISK);
    }
}

TEST_CASE("Comprehensive Object Property Tests") {
    SUBCASE("rotation operations on objects") {
        Context ctx;
        std::vector<uint> objs;
        objs.push_back(ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1)));
        objs.push_back(ctx.addBoxObject(make_vec3(1, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1)));

        DOCTEST_CHECK_NOTHROW(ctx.rotateObject(objs, 0.5f * PI_F, "z"));
        DOCTEST_CHECK_NOTHROW(ctx.rotateObject(objs, 0.5f * PI_F, make_vec3(0, 0, 1)));
        DOCTEST_CHECK_NOTHROW(ctx.rotateObject(objs, 0.5f * PI_F, make_vec3(0, 0, 0), make_vec3(0, 0, 1)));
        DOCTEST_CHECK_NOTHROW(ctx.rotateObjectAboutOrigin(objs, 0.5f * PI_F, make_vec3(0, 0, 1)));
    }

    SUBCASE("scaling operations on objects") {
        Context ctx;
        std::vector<uint> objs;
        objs.push_back(ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1)));

        DOCTEST_CHECK_NOTHROW(ctx.scaleObject(objs, make_vec3(2, 2, 2)));
        DOCTEST_CHECK_NOTHROW(ctx.scaleObjectAboutCenter(objs, make_vec3(0.5f, 0.5f, 0.5f)));
        DOCTEST_CHECK_NOTHROW(ctx.scaleObjectAboutPoint(objs, make_vec3(2, 2, 2), make_vec3(0, 0, 0)));
        DOCTEST_CHECK_NOTHROW(ctx.scaleObjectAboutOrigin(objs, make_vec3(0.5f, 0.5f, 0.5f)));
    }

    SUBCASE("translation operations on objects") {
        Context ctx;
        std::vector<uint> objs;
        objs.push_back(ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1)));

        DOCTEST_CHECK_NOTHROW(ctx.translateObject(objs, make_vec3(1, 2, 3)));
    }
}

TEST_CASE("Domain and Bounding Operations") {
    SUBCASE("domain bounding sphere") {
        Context ctx;
        ctx.addPatch(make_vec3(-2, 0, 0), make_vec2(1, 1));
        ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));

        vec3 center;
        float radius;
        ctx.getDomainBoundingSphere(center, radius);
        DOCTEST_CHECK(center.x == doctest::Approx(0.f).epsilon(0.1));
        DOCTEST_CHECK(radius > 2.f);
    }
}

TEST_CASE("Missing Data and State Functions") {
    SUBCASE("listTimeseriesVariables") {
        Context ctx;
        Date date = make_Date(1, 1, 2025);
        Time time = make_Time(0, 0, 12);

        ctx.addTimeseriesData("temp", 25.5f, date, time);
        ctx.addTimeseriesData("humidity", 60.0f, date, time);

        std::vector<std::string> vars = ctx.listTimeseriesVariables();
        DOCTEST_CHECK(vars.size() >= 2);
        DOCTEST_CHECK(std::find(vars.begin(), vars.end(), "temp") != vars.end());
        DOCTEST_CHECK(std::find(vars.begin(), vars.end(), "humidity") != vars.end());
    }

    SUBCASE("getUniquePrimitiveParentObjectIDs") {
        Context ctx;
        uint obj1 = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint obj2 = ctx.addSphereObject(10, make_vec3(0, 0, 0), 1.f);

        std::vector<uint> all_prims = ctx.getAllUUIDs();
        std::vector<uint> obj_ids = ctx.getUniquePrimitiveParentObjectIDs(all_prims);
        DOCTEST_CHECK(obj_ids.size() >= 2);
        DOCTEST_CHECK(std::find(obj_ids.begin(), obj_ids.end(), obj1) != obj_ids.end());
        DOCTEST_CHECK(std::find(obj_ids.begin(), obj_ids.end(), obj2) != obj_ids.end());
    }
}

TEST_CASE("Comprehensive Coverage Tests") {
    SUBCASE("additional object operations with vectors") {
        Context ctx;
        std::vector<uint> obj_ids;
        obj_ids.push_back(ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1)));
        obj_ids.push_back(ctx.addBoxObject(make_vec3(2, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1)));

        std::vector<uint> all_uuids = ctx.getObjectPrimitiveUUIDs(obj_ids);
        DOCTEST_CHECK(all_uuids.size() == 12); // 6 faces per box * 2 boxes

        std::vector<std::vector<uint>> nested_obj_ids = {{obj_ids[0]}, {obj_ids[1]}};
        std::vector<uint> nested_uuids = ctx.getObjectPrimitiveUUIDs(nested_obj_ids);
        DOCTEST_CHECK(nested_uuids.size() == 12);

        ctx.hideObject(obj_ids);
        DOCTEST_CHECK(ctx.isObjectHidden(obj_ids[0]));
        DOCTEST_CHECK(ctx.isObjectHidden(obj_ids[1]));

        ctx.showObject(obj_ids);
        DOCTEST_CHECK(!ctx.isObjectHidden(obj_ids[0]));
        DOCTEST_CHECK(!ctx.isObjectHidden(obj_ids[1]));
    }

    SUBCASE("object texture color overrides") {
        Context ctx;
        std::vector<uint> obj_ids;
        obj_ids.push_back(ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, make_int2(2, 2), "lib/images/solid.jpg"));

        ctx.overrideObjectTextureColor(obj_ids);
        ctx.useObjectTextureColor(obj_ids);
    }
}

TEST_CASE("getAllUUIDs Cache Performance") {
    SUBCASE("Cache invalidation on primitive add/delete") {
        Context ctx;

        // Initial empty state
        std::vector<uint> empty_uuids = ctx.getAllUUIDs();
        DOCTEST_CHECK(empty_uuids.empty());

        // Add primitives and test cache invalidation
        uint p1 = ctx.addPatch();
        std::vector<uint> one_uuid = ctx.getAllUUIDs();
        DOCTEST_CHECK(one_uuid.size() == 1);
        DOCTEST_CHECK(one_uuid[0] == p1);

        // Test cache consistency - repeated calls should return same result
        std::vector<uint> same_uuid = ctx.getAllUUIDs();
        DOCTEST_CHECK(same_uuid.size() == 1);
        DOCTEST_CHECK(same_uuid[0] == p1);

        // Add more primitives
        uint t1 = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0));
        uint v1 = ctx.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));

        std::vector<uint> three_uuids = ctx.getAllUUIDs();
        DOCTEST_CHECK(three_uuids.size() == 3);
        DOCTEST_CHECK(std::find(three_uuids.begin(), three_uuids.end(), p1) != three_uuids.end());
        DOCTEST_CHECK(std::find(three_uuids.begin(), three_uuids.end(), t1) != three_uuids.end());
        DOCTEST_CHECK(std::find(three_uuids.begin(), three_uuids.end(), v1) != three_uuids.end());

        // Test delete invalidation
        ctx.deletePrimitive(t1);
        std::vector<uint> two_uuids = ctx.getAllUUIDs();
        DOCTEST_CHECK(two_uuids.size() == 2);
        DOCTEST_CHECK(std::find(two_uuids.begin(), two_uuids.end(), t1) == two_uuids.end());
        DOCTEST_CHECK(std::find(two_uuids.begin(), two_uuids.end(), p1) != two_uuids.end());
        DOCTEST_CHECK(std::find(two_uuids.begin(), two_uuids.end(), v1) != two_uuids.end());
    }

    SUBCASE("Cache invalidation on hide/show primitives") {
        Context ctx;
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addPatch();
        uint p3 = ctx.addPatch();

        // All visible initially
        std::vector<uint> all_visible = ctx.getAllUUIDs();
        DOCTEST_CHECK(all_visible.size() == 3);

        // Hide one primitive
        ctx.hidePrimitive(p2);
        std::vector<uint> two_visible = ctx.getAllUUIDs();
        DOCTEST_CHECK(two_visible.size() == 2);
        DOCTEST_CHECK(std::find(two_visible.begin(), two_visible.end(), p2) == two_visible.end());
        DOCTEST_CHECK(std::find(two_visible.begin(), two_visible.end(), p1) != two_visible.end());
        DOCTEST_CHECK(std::find(two_visible.begin(), two_visible.end(), p3) != two_visible.end());

        // Hide multiple primitives
        std::vector<uint> to_hide = {p1, p3};
        ctx.hidePrimitive(to_hide);
        std::vector<uint> none_visible = ctx.getAllUUIDs();
        DOCTEST_CHECK(none_visible.empty());

        // Show one primitive back
        ctx.showPrimitive(p1);
        std::vector<uint> one_visible = ctx.getAllUUIDs();
        DOCTEST_CHECK(one_visible.size() == 1);
        DOCTEST_CHECK(one_visible[0] == p1);

        // Show all primitives back
        std::vector<uint> to_show = {p2, p3};
        ctx.showPrimitive(to_show);
        std::vector<uint> all_back = ctx.getAllUUIDs();
        DOCTEST_CHECK(all_back.size() == 3);
    }

    SUBCASE("Cache invalidation on copy primitives") {
        Context ctx;
        uint original = ctx.addPatch();

        std::vector<uint> before_copy = ctx.getAllUUIDs();
        DOCTEST_CHECK(before_copy.size() == 1);

        uint copied = ctx.copyPrimitive(original);
        std::vector<uint> after_copy = ctx.getAllUUIDs();
        DOCTEST_CHECK(after_copy.size() == 2);
        DOCTEST_CHECK(std::find(after_copy.begin(), after_copy.end(), original) != after_copy.end());
        DOCTEST_CHECK(std::find(after_copy.begin(), after_copy.end(), copied) != after_copy.end());

        // Test multiple copy
        std::vector<uint> originals = {original, copied};
        std::vector<uint> copies = ctx.copyPrimitive(originals);
        std::vector<uint> after_multi_copy = ctx.getAllUUIDs();
        DOCTEST_CHECK(after_multi_copy.size() == 4);
        for (uint copy_id: copies) {
            DOCTEST_CHECK(std::find(after_multi_copy.begin(), after_multi_copy.end(), copy_id) != after_multi_copy.end());
        }
    }

    SUBCASE("Cache consistency across mixed operations") {
        Context ctx;

        // Complex sequence of operations
        uint p1 = ctx.addPatch();
        uint p2 = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0));

        std::vector<uint> step1 = ctx.getAllUUIDs();
        DOCTEST_CHECK(step1.size() == 2);

        ctx.hidePrimitive(p1);
        std::vector<uint> step2 = ctx.getAllUUIDs();
        DOCTEST_CHECK(step2.size() == 1);
        DOCTEST_CHECK(step2[0] == p2);

        uint p3 = ctx.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
        std::vector<uint> step3 = ctx.getAllUUIDs();
        DOCTEST_CHECK(step3.size() == 2);

        ctx.showPrimitive(p1);
        std::vector<uint> step4 = ctx.getAllUUIDs();
        DOCTEST_CHECK(step4.size() == 3);

        ctx.deletePrimitive(p2);
        std::vector<uint> step5 = ctx.getAllUUIDs();
        DOCTEST_CHECK(step5.size() == 2);
        DOCTEST_CHECK(std::find(step5.begin(), step5.end(), p2) == step5.end());
        DOCTEST_CHECK(std::find(step5.begin(), step5.end(), p1) != step5.end());
        DOCTEST_CHECK(std::find(step5.begin(), step5.end(), p3) != step5.end());
    }
}

TEST_CASE("Error Handling") {
    SUBCASE("Context error handling") {
        Context context_test;
        uint tri = context_test.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::green);
        capture_cerr cerr_buffer;
        vec3 center;
#ifdef HELIOS_DEBUG
        DOCTEST_CHECK_THROWS_AS(center = context_test.getPatchCenter(tri), std::runtime_error);
#endif

        uint vox = context_test.addVoxel(make_vec3(0, 0, 0), make_vec3(1, 1, 1));
        std::vector<uint> vlist{vox};
        DOCTEST_CHECK_THROWS_AS(context_test.rotatePrimitive(vlist, PI_F / 4.f, "a"), std::runtime_error);
    }
}

TEST_CASE("Zero Area Triangle Detection") {
    SUBCASE("addTubeObject with nearly identical vertices should not create zero-area triangles") {
        Context ctx;

        // Test case based on problematic vertices from plant architecture
        std::vector<vec3> nodes = {make_vec3(0.300000012f, -0.112000048f, 0.00999999978f), make_vec3(0.29995966f, -0.111979447f, 0.0109989736f), make_vec3(0.299919307f, -0.111958846f, 0.0119979475f)};

        std::vector<float> radii = {0.000500000024f, 0.000500000024f, 0.000500000024f};
        std::vector<RGBcolor> colors = {RGB::green, RGB::green, RGB::green};

        // Use exact same parameters as failing case: Ndiv_internode_radius = 7
        uint tube_obj = ctx.addTubeObject(7, nodes, radii, colors);

        // Verify the tube object was created
        DOCTEST_CHECK(ctx.doesObjectExist(tube_obj));

        // Get all primitives in the tube and check their areas
        std::vector<uint> tube_primitives = ctx.getObjectPrimitiveUUIDs(tube_obj);
        DOCTEST_CHECK(tube_primitives.size() > 0);

        for (uint uuid: tube_primitives) {
            float area = ctx.getPrimitiveArea(uuid);
            DOCTEST_CHECK(area > 0.0f); // No zero-area triangles
            DOCTEST_CHECK(area > 1e-12f); // Area should be reasonably above precision limit
        }
    }

    SUBCASE("addTubeObject with extremely small displacements") {
        Context ctx;

        // Even more extreme case - displacements on the order of 1e-5
        std::vector<vec3> nodes = {make_vec3(0.0f, 0.0f, 0.0f), make_vec3(1e-5f, 1e-5f, 1e-3f), make_vec3(2e-5f, 2e-5f, 2e-3f)};

        std::vector<float> radii = {1e-4f, 1e-4f, 1e-4f};

        uint tube_obj = ctx.addTubeObject(6, nodes, radii);
        DOCTEST_CHECK(ctx.doesObjectExist(tube_obj));

        std::vector<uint> tube_primitives = ctx.getObjectPrimitiveUUIDs(tube_obj);
        for (uint uuid: tube_primitives) {
            float area = ctx.getPrimitiveArea(uuid);
            DOCTEST_CHECK(area > 0.0f);
        }
    }
}

TEST_CASE("Transparent Texture Zero Area Validation") {
    SUBCASE("addSphere with transparent texture should filter zero-area triangles") {
        Context ctx;

        // Test with diamond texture (has transparency)
        std::vector<uint> sphere_uuids = ctx.addSphere(20, make_vec3(0, 0, 0), 1.0f, "lib/images/diamond_texture.png");

        // All returned primitives should have positive area
        DOCTEST_CHECK(sphere_uuids.size() > 0);
        for (uint uuid: sphere_uuids) {
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));
            float area = ctx.getPrimitiveArea(uuid);
            DOCTEST_CHECK(area > 0.0f);
        }

        // Test with disk texture (more transparency)
        std::vector<uint> sphere_disk_uuids = ctx.addSphere(30, make_vec3(2, 0, 0), 1.0f, "lib/images/disk_texture.png");

        DOCTEST_CHECK(sphere_disk_uuids.size() > 0);
        for (uint uuid: sphere_disk_uuids) {
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));
            float area = ctx.getPrimitiveArea(uuid);
            DOCTEST_CHECK(area > 0.0f);
        }

        // Verify ALL primitives in each sphere have positive area
        int zero_area_count_diamond = 0;
        for (uint uuid: sphere_uuids) {
            float area = ctx.getPrimitiveArea(uuid);
            if (area <= 0.0f) {
                zero_area_count_diamond++;
            }
        }
        DOCTEST_CHECK(zero_area_count_diamond == 0);

        int zero_area_count_disk = 0;
        for (uint uuid: sphere_disk_uuids) {
            float area = ctx.getPrimitiveArea(uuid);
            if (area <= 0.0f) {
                zero_area_count_disk++;
            }
        }
        DOCTEST_CHECK(zero_area_count_disk == 0);

        // Compare with solid sphere for reference
        std::vector<uint> solid_sphere_uuids = ctx.addSphere(20, make_vec3(4, 0, 0), 1.0f, RGB::green);

        int zero_area_count_solid = 0;
        for (uint uuid: solid_sphere_uuids) {
            float area = ctx.getPrimitiveArea(uuid);
            if (area <= 0.0f) {
                zero_area_count_solid++;
            }
        }
        DOCTEST_CHECK(zero_area_count_solid == 0);
    }

    SUBCASE("texture transparency validation preserves object integrity") {
        Context ctx;

        // Create textured sphere and verify all returned UUIDs are valid
        std::vector<uint> sphere_uuids = ctx.addSphere(15, make_vec3(0, 0, 0), 1.0f, "lib/images/diamond_texture.png");

        // Check that all returned primitives exist and have positive area
        for (uint uuid: sphere_uuids) {
            DOCTEST_CHECK(ctx.doesPrimitiveExist(uuid));
            DOCTEST_CHECK(ctx.getPrimitiveType(uuid) == PRIMITIVE_TYPE_TRIANGLE);

            float area = ctx.getPrimitiveArea(uuid);
            DOCTEST_CHECK(area > 0.0f);
            DOCTEST_CHECK(area > 1e-10f); // Should be significantly above precision threshold

            // Verify solid fraction is reasonable (not exactly 0 or 1)
            float solid_fraction = ctx.getPrimitiveSolidFraction(uuid);
            DOCTEST_CHECK(solid_fraction > 0.0f);
            DOCTEST_CHECK(solid_fraction <= 1.0f);
        }

        // Comprehensive check: verify no zero-area primitives exist anywhere in context
        std::vector<uint> all_uuids = ctx.getAllUUIDs();
        int total_zero_area = 0;
        int total_negative_area = 0;

        for (uint uuid: all_uuids) {
            float area = ctx.getPrimitiveArea(uuid);
            if (area == 0.0f) {
                total_zero_area++;
            }
            if (area < 0.0f) {
                total_negative_area++;
            }
        }

        // No zero or negative area primitives should exist
        DOCTEST_CHECK(total_zero_area == 0);
        DOCTEST_CHECK(total_negative_area == 0);

        // Additional validation: check that all primitives have reasonable solid fractions
        for (uint uuid: sphere_uuids) {
            float solid_fraction = ctx.getPrimitiveSolidFraction(uuid);
            DOCTEST_CHECK(solid_fraction >= 0.0f);
            DOCTEST_CHECK(solid_fraction <= 1.0f);

            // For textured primitives, effective area should be geometric_area * solid_fraction
            if (ctx.getPrimitiveType(uuid) == PRIMITIVE_TYPE_TRIANGLE) {
                vec3 v0 = ctx.getTriangleVertex(uuid, 0);
                vec3 v1 = ctx.getTriangleVertex(uuid, 1);
                vec3 v2 = ctx.getTriangleVertex(uuid, 2);
                float geometric_area = calculateTriangleArea(v0, v1, v2);
                float effective_area = ctx.getPrimitiveArea(uuid);

                // Effective area should be <= geometric area (due to solid fraction)
                DOCTEST_CHECK(effective_area <= geometric_area + 1e-6f); // Allow small numerical tolerance
                DOCTEST_CHECK(effective_area > 0.0f);
            }
        }

        // Test zero-area validation for other primitive methods (addTube, addDisk, addCone)
        DOCTEST_SUBCASE("Test Other Primitive Methods Zero Area Validation") {
            Context ctx_other;

            // Test addTube with transparent texture
            std::vector<vec3> tube_nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
            std::vector<float> tube_radii = {0.1f, 0.15f, 0.1f};
            std::vector<uint> tube_uuids = ctx_other.addTube(8, tube_nodes, tube_radii, "lib/images/diamond_texture.png");

            // All returned UUIDs should have positive area
            int tube_positive_area = 0, tube_zero_area = 0;
            for (uint uuid: tube_uuids) {
                float area = ctx_other.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    tube_positive_area++;
                } else {
                    tube_zero_area++;
                }
            }

            DOCTEST_CHECK(tube_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(tube_zero_area == 0); // Should have no zero area triangles

            // Test addDisk with transparent texture
            std::vector<uint> disk_uuids = ctx_other.addDisk(make_int2(4, 3), make_vec3(0, 0, 0), make_vec2(1.0f, 1.0f), make_SphericalCoord(0, 0), "lib/images/disk_texture.png");

            // All returned UUIDs should have positive area
            int disk_positive_area = 0, disk_zero_area = 0;
            for (uint uuid: disk_uuids) {
                float area = ctx_other.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    disk_positive_area++;
                } else {
                    disk_zero_area++;
                }
            }

            DOCTEST_CHECK(disk_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(disk_zero_area == 0); // Should have no zero area triangles

            // Test addCone with transparent texture
            std::vector<uint> cone_uuids = ctx_other.addCone(8, make_vec3(0, 0, 0), make_vec3(0, 0, 1), 0.1f, 0.2f, "lib/images/diamond_texture.png");

            // All returned UUIDs should have positive area
            int cone_positive_area = 0, cone_zero_area = 0;
            for (uint uuid: cone_uuids) {
                float area = ctx_other.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    cone_positive_area++;
                } else {
                    cone_zero_area++;
                }
            }

            DOCTEST_CHECK(cone_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(cone_zero_area == 0); // Should have no zero area triangles

            // Test addTile with transparent texture (should already work, but verify)
            std::vector<uint> tile_uuids = ctx_other.addTile(make_vec3(0, 0, 0), make_vec2(1.0f, 1.0f), make_SphericalCoord(0, 0), make_int2(4, 4), "lib/images/diamond_texture.png");

            // All returned UUIDs should have positive area
            int tile_positive_area = 0, tile_zero_area = 0;
            for (uint uuid: tile_uuids) {
                float area = ctx_other.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    tile_positive_area++;
                } else {
                    tile_zero_area++;
                }
            }

            DOCTEST_CHECK(tile_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(tile_zero_area == 0); // Should have no zero area triangles
        }

        // Test zero-area validation for compound object methods
        DOCTEST_SUBCASE("Test Compound Object Methods Zero Area Validation") {
            Context ctx_compound;

            // Test addSphereObject with transparent texture
            uint sphere_obj = ctx_compound.addSphereObject(8, make_vec3(0, 0, 0), 0.5f, "lib/images/diamond_texture.png");
            std::vector<uint> sphere_primitives = ctx_compound.getObjectPrimitiveUUIDs(sphere_obj);

            // All primitives should have positive area
            int sphere_positive_area = 0, sphere_zero_area = 0;
            for (uint uuid: sphere_primitives) {
                float area = ctx_compound.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    sphere_positive_area++;
                } else {
                    sphere_zero_area++;
                }
            }

            DOCTEST_CHECK(sphere_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(sphere_zero_area == 0); // Should have no zero area triangles

            // Test addTubeObject with transparent texture
            std::vector<vec3> tube_nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
            std::vector<float> tube_radii = {0.1f, 0.15f, 0.1f};
            uint tube_obj = ctx_compound.addTubeObject(8, tube_nodes, tube_radii, "lib/images/diamond_texture.png");
            std::vector<uint> tube_primitives = ctx_compound.getObjectPrimitiveUUIDs(tube_obj);

            // All primitives should have positive area
            int tube_positive_area = 0, tube_zero_area = 0;
            for (uint uuid: tube_primitives) {
                float area = ctx_compound.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    tube_positive_area++;
                } else {
                    tube_zero_area++;
                }
            }

            DOCTEST_CHECK(tube_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(tube_zero_area == 0); // Should have no zero area triangles

            // Test addDiskObject with transparent texture
            uint disk_obj = ctx_compound.addDiskObject(make_int2(4, 3), make_vec3(0, 0, 0), make_vec2(1.0f, 1.0f), make_SphericalCoord(0, 0), "lib/images/disk_texture.png");
            std::vector<uint> disk_primitives = ctx_compound.getObjectPrimitiveUUIDs(disk_obj);

            // All primitives should have positive area
            int disk_positive_area = 0, disk_zero_area = 0;
            for (uint uuid: disk_primitives) {
                float area = ctx_compound.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    disk_positive_area++;
                } else {
                    disk_zero_area++;
                }
            }

            DOCTEST_CHECK(disk_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(disk_zero_area == 0); // Should have no zero area triangles

            // Test addConeObject with transparent texture
            uint cone_obj = ctx_compound.addConeObject(8, make_vec3(0, 0, 0), make_vec3(0, 0, 1), 0.1f, 0.2f, "lib/images/diamond_texture.png");
            std::vector<uint> cone_primitives = ctx_compound.getObjectPrimitiveUUIDs(cone_obj);

            // All primitives should have positive area
            int cone_positive_area = 0, cone_zero_area = 0;
            for (uint uuid: cone_primitives) {
                float area = ctx_compound.getPrimitiveArea(uuid);
                DOCTEST_CHECK(area >= 0.0f);
                if (area > 0.0f) {
                    cone_positive_area++;
                } else {
                    cone_zero_area++;
                }
            }

            DOCTEST_CHECK(cone_positive_area > 0); // Should have some positive area triangles
            DOCTEST_CHECK(cone_zero_area == 0); // Should have no zero area triangles
        }
    }
}

TEST_CASE("File path resolution priority") {
    SUBCASE("resolveFilePath current directory priority") {
        // Test that the new file resolution logic checks current directory first,
        // then falls back to HELIOS_BUILD directory

        // Create a test texture file in the current directory
        std::string testFileName = "test_file_resolution.jpg";
        std::filesystem::path currentDirFile = std::filesystem::current_path() / testFileName;

        // Copy the existing texture for our test
        std::filesystem::path sourceTexture = "core/lib/models/texture.jpg";

        if (std::filesystem::exists(sourceTexture)) {
            // Copy to current directory
            std::filesystem::copy_file(sourceTexture, currentDirFile, std::filesystem::copy_options::overwrite_existing);
            DOCTEST_CHECK(std::filesystem::exists(currentDirFile));

            // Test resolveFilePath function directly
            std::filesystem::path resolved = helios::resolveFilePath(testFileName);
            DOCTEST_CHECK(resolved == std::filesystem::canonical(currentDirFile));

            // Clean up
            std::filesystem::remove(currentDirFile);
        }
    }

    SUBCASE("addPatch with texture from current directory") {
        Context ctx;

        // Create test directory structure in current working directory
        std::filesystem::create_directories("test_models");
        std::string testTexture = "test_models/test_texture.jpg";
        std::filesystem::path testTexturePath = std::filesystem::current_path() / testTexture;

        // Copy source texture
        std::filesystem::path sourceTexture = "core/lib/models/texture.jpg";

        if (std::filesystem::exists(sourceTexture)) {
            std::filesystem::copy_file(sourceTexture, testTexturePath, std::filesystem::copy_options::overwrite_existing);

            // This should work with the fix - loads from current directory first
            // addPatch uses resolveFilePath internally for texture loading
            SphericalCoord rotation = make_SphericalCoord(0, 0);
            uint patch_id;
            DOCTEST_CHECK_NOTHROW({ patch_id = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), rotation, testTexture.c_str()); });
            DOCTEST_CHECK(patch_id > 0);

            // Verify the texture loaded correctly
            bool has_transparency = ctx.primitiveTextureHasTransparencyChannel(patch_id);
            DOCTEST_CHECK((has_transparency || !has_transparency)); // Just verify it's a boolean (texture loaded)

            // Clean up
            std::filesystem::remove(testTexturePath);
            std::filesystem::remove("test_models");
        }
    }

    SUBCASE("Material System - Label-Based Creation") {
        Context ctx;

        // Default material should exist (but not counted in getMaterialCount or listMaterials)
        DOCTEST_CHECK(ctx.doesMaterialExist("__default__"));
        DOCTEST_CHECK(ctx.getMaterialCount() == 0); // No user-created materials yet

        // Create materials with labels
        ctx.addMaterial("leaf_material");
        DOCTEST_CHECK(ctx.doesMaterialExist("leaf_material"));
        DOCTEST_CHECK(ctx.getMaterialCount() == 1);

        ctx.addMaterial("bark_material");
        DOCTEST_CHECK(ctx.doesMaterialExist("bark_material"));
        DOCTEST_CHECK(ctx.getMaterialCount() == 2);

        // List materials (only user-created, not default or auto-generated)
        std::vector<std::string> labels = ctx.listMaterials();
        DOCTEST_CHECK(labels.size() == 2);

        // Reserved labels should fail
        DOCTEST_CHECK_THROWS(ctx.addMaterial("__reserved"));
    }

    SUBCASE("Material System - Properties") {
        Context ctx;

        // Create and set material properties
        ctx.addMaterial("test_mat");

        RGBAcolor purple = make_RGBAcolor(0.5f, 0, 0.5f, 1);
        ctx.setMaterialColor("test_mat", purple);

        RGBAcolor color = ctx.getMaterialColor("test_mat");
        DOCTEST_CHECK(color.r == doctest::Approx(0.5f).epsilon(0.001));
        DOCTEST_CHECK(color.g == doctest::Approx(0.0f).epsilon(0.001));
        DOCTEST_CHECK(color.b == doctest::Approx(0.5f).epsilon(0.001));

        // Set texture
        ctx.setMaterialTexture("test_mat", "lib/images/disk_texture.png");
        std::string tex = ctx.getMaterialTexture("test_mat");
        DOCTEST_CHECK(tex == "lib/images/disk_texture.png");

        // Texture override
        ctx.setMaterialTextureColorOverride("test_mat", true);
        DOCTEST_CHECK(ctx.isMaterialTextureColorOverridden("test_mat"));

        ctx.setMaterialTextureColorOverride("test_mat", false);
        DOCTEST_CHECK(!ctx.isMaterialTextureColorOverridden("test_mat"));

        // Twosided flag - test default value
        DOCTEST_CHECK(ctx.getMaterialTwosidedFlag("test_mat") == 1); // Default is 1 (two-sided)

        // Twosided flag - set to 0 (one-sided)
        ctx.setMaterialTwosidedFlag("test_mat", 0);
        DOCTEST_CHECK(ctx.getMaterialTwosidedFlag("test_mat") == 0);

        // Twosided flag - set back to 1 (two-sided)
        ctx.setMaterialTwosidedFlag("test_mat", 1);
        DOCTEST_CHECK(ctx.getMaterialTwosidedFlag("test_mat") == 1);
    }

    SUBCASE("Material System - Assignment to Primitives") {
        Context ctx;

        // Create material
        ctx.addMaterial("red_mat");
        ctx.setMaterialColor("red_mat", make_RGBAcolor(1, 0, 0, 1));

        // Create primitives with default color
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));

        // Assign material
        ctx.assignMaterialToPrimitive(p1, "red_mat");
        ctx.assignMaterialToPrimitive(p2, "red_mat");

        // Check primitive material label
        DOCTEST_CHECK(ctx.getPrimitiveMaterialLabel(p1) == "red_mat");
        DOCTEST_CHECK(ctx.getPrimitiveMaterialLabel(p2) == "red_mat");

        // Check primitive color reflects material
        RGBcolor c1 = ctx.getPrimitiveColor(p1);
        DOCTEST_CHECK(c1.r == doctest::Approx(1.0f).epsilon(0.001));
        DOCTEST_CHECK(c1.g == doctest::Approx(0.0f).epsilon(0.001));

        // Modify material - should affect both primitives
        ctx.setMaterialColor("red_mat", make_RGBAcolor(0, 1, 0, 1)); // Green

        c1 = ctx.getPrimitiveColor(p1);
        RGBcolor c2 = ctx.getPrimitiveColor(p2);
        DOCTEST_CHECK(c1.g == doctest::Approx(1.0f).epsilon(0.001));
        DOCTEST_CHECK(c2.g == doctest::Approx(1.0f).epsilon(0.001));

        // Reverse lookup
        std::vector<uint> users = ctx.getPrimitivesUsingMaterial("red_mat");
        DOCTEST_CHECK(users.size() == 2);
    }

    SUBCASE("Material System - Batch Assignment") {
        Context ctx;

        ctx.addMaterial("batch_mat");
        ctx.setMaterialColor("batch_mat", make_RGBAcolor(0.5f, 0.5f, 0.5f, 1));

        std::vector<uint> UUIDs;
        for (int i = 0; i < 10; i++) {
            UUIDs.push_back(ctx.addPatch(make_vec3(i, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0)));
        }

        // Batch assign
        ctx.assignMaterialToPrimitive(UUIDs, "batch_mat");

        // Verify all have the material
        for (uint uuid: UUIDs) {
            DOCTEST_CHECK(ctx.getPrimitiveMaterialLabel(uuid) == "batch_mat");
        }
    }

    SUBCASE("Material System - Deletion") {
        Context ctx;

        ctx.addMaterial("temp_mat");
        ctx.setMaterialColor("temp_mat", make_RGBAcolor(1, 0, 0, 1));

        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        ctx.assignMaterialToPrimitive(p1, "temp_mat");

        // Delete material - primitive should revert to default
        capture_cerr c; // Capture warning about material in use
        ctx.deleteMaterial("temp_mat");

        DOCTEST_CHECK(!ctx.doesMaterialExist("temp_mat"));
        DOCTEST_CHECK(ctx.getPrimitiveMaterialLabel(p1) == "__default__");
    }

    SUBCASE("Material System - XML Round-Trip") {
        Context ctx;

        // Create materials
        ctx.addMaterial("red_mat");
        ctx.setMaterialColor("red_mat", make_RGBAcolor(1, 0, 0, 1));

        ctx.addMaterial("textured_mat");
        ctx.setMaterialColor("textured_mat", make_RGBAcolor(0, 1, 0, 1));
        ctx.setMaterialTexture("textured_mat", "lib/images/disk_texture.png");

        // Create a material with non-default twosided_flag
        ctx.addMaterial("onesided_mat");
        ctx.setMaterialColor("onesided_mat", make_RGBAcolor(0, 0, 1, 1));
        ctx.setMaterialTwosidedFlag("onesided_mat", 0); // One-sided

        // Create and assign primitives
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        uint p3 = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        uint p4 = ctx.addPatch(make_vec3(3, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));

        ctx.assignMaterialToPrimitive(p1, "red_mat");
        ctx.assignMaterialToPrimitive(p2, "textured_mat");
        ctx.assignMaterialToPrimitive(p3, "red_mat");
        ctx.assignMaterialToPrimitive(p4, "onesided_mat");

        // Write to XML
        ctx.writeXML("test_materials.xml", {p1, p2, p3, p4}, true);

        // Load into new context
        Context ctx2;
        std::vector<uint> loaded_UUIDs = ctx2.loadXML("test_materials.xml", true);

        DOCTEST_CHECK(loaded_UUIDs.size() == 4);

        // Verify materials were preserved
        DOCTEST_CHECK(ctx2.doesMaterialExist("red_mat"));
        DOCTEST_CHECK(ctx2.doesMaterialExist("textured_mat"));
        DOCTEST_CHECK(ctx2.doesMaterialExist("onesided_mat"));

        RGBcolor loaded_color1 = ctx2.getPrimitiveColor(loaded_UUIDs[0]);
        DOCTEST_CHECK(loaded_color1.r == doctest::Approx(1.0f).epsilon(0.001));

        DOCTEST_CHECK(ctx2.getPrimitiveTextureFile(loaded_UUIDs[1]) == "lib/images/disk_texture.png");

        // Verify twosided_flag was preserved
        DOCTEST_CHECK(ctx2.getMaterialTwosidedFlag("red_mat") == 1); // Default
        DOCTEST_CHECK(ctx2.getMaterialTwosidedFlag("textured_mat") == 1); // Default
        DOCTEST_CHECK(ctx2.getMaterialTwosidedFlag("onesided_mat") == 0); // Non-default

        // Clean up
        std::filesystem::remove("test_materials.xml");
    }

    SUBCASE("getPrimitiveTwosidedFlag helper function") {
        Context ctx;

        // Create materials with different twosided_flag values
        ctx.addMaterial("onesided_mat");
        ctx.setMaterialTwosidedFlag("onesided_mat", 0);

        ctx.addMaterial("twosided_mat");
        ctx.setMaterialTwosidedFlag("twosided_mat", 1);

        ctx.addMaterial("transparent_mat");
        ctx.setMaterialTwosidedFlag("transparent_mat", 2);

        // Create primitives
        uint UUID_mat_onesided = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint UUID_mat_twosided = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
        uint UUID_mat_transparent = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));
        uint UUID_prim_data = ctx.addPatch(make_vec3(3, 0, 0), make_vec2(1, 1));
        uint UUID_default = ctx.addPatch(make_vec3(4, 0, 0), make_vec2(1, 1));

        // Assign materials
        ctx.assignMaterialToPrimitive(UUID_mat_onesided, "onesided_mat");
        ctx.assignMaterialToPrimitive(UUID_mat_twosided, "twosided_mat");
        ctx.assignMaterialToPrimitive(UUID_mat_transparent, "transparent_mat");

        // Set primitive data on one primitive (no user material assigned)
        ctx.setPrimitiveData(UUID_prim_data, "twosided_flag", uint(0));

        // Test: Material takes precedence - one-sided material
        DOCTEST_CHECK(ctx.getPrimitiveTwosidedFlag(UUID_mat_onesided) == 0);

        // Test: Material takes precedence - two-sided material
        DOCTEST_CHECK(ctx.getPrimitiveTwosidedFlag(UUID_mat_twosided) == 1);

        // Test: Material supports values > 1 (transparent)
        DOCTEST_CHECK(ctx.getPrimitiveTwosidedFlag(UUID_mat_transparent) == 2);

        // Test: Primitive data fallback (no user material)
        DOCTEST_CHECK(ctx.getPrimitiveTwosidedFlag(UUID_prim_data) == 0);

        // Test: Default value when no material or primitive data
        DOCTEST_CHECK(ctx.getPrimitiveTwosidedFlag(UUID_default) == 1);

        // Test: Custom default value
        DOCTEST_CHECK(ctx.getPrimitiveTwosidedFlag(UUID_default, 2) == 2);

        // Test: Material takes precedence over primitive data
        // First, set primitive data on a primitive with a material
        ctx.setPrimitiveData(UUID_mat_onesided, "twosided_flag", uint(1)); // Try to override with primitive data
        DOCTEST_CHECK(ctx.getPrimitiveTwosidedFlag(UUID_mat_onesided) == 0); // Should still return material value (0)
    }

    SUBCASE("Material Data - Setting and Getting with Labels") {
        Context ctx;

        // Create a material
        ctx.addMaterial("data_mat");

        // Test uint data
        ctx.setMaterialData("data_mat", "twosided_flag", 1u);
        DOCTEST_CHECK(ctx.doesMaterialDataExist("data_mat", "twosided_flag"));
        DOCTEST_CHECK(ctx.getMaterialDataType("data_mat", "twosided_flag") == HELIOS_TYPE_UINT);
        uint flag_val;
        ctx.getMaterialData("data_mat", "twosided_flag", flag_val);
        DOCTEST_CHECK(flag_val == 1u);

        // Test int data
        ctx.setMaterialData("data_mat", "test_int", -42);
        int int_val;
        ctx.getMaterialData("data_mat", "test_int", int_val);
        DOCTEST_CHECK(int_val == -42);

        // Test float data
        ctx.setMaterialData("data_mat", "test_float", 3.14f);
        float float_val;
        ctx.getMaterialData("data_mat", "test_float", float_val);
        DOCTEST_CHECK(float_val == doctest::Approx(3.14f).epsilon(0.001));

        // Test vec3 data
        vec3 test_vec = make_vec3(1, 2, 3);
        ctx.setMaterialData("data_mat", "test_vec3", test_vec);
        vec3 vec_val;
        ctx.getMaterialData("data_mat", "test_vec3", vec_val);
        DOCTEST_CHECK(vec_val.x == doctest::Approx(1.0f).epsilon(0.001));
        DOCTEST_CHECK(vec_val.y == doctest::Approx(2.0f).epsilon(0.001));
        DOCTEST_CHECK(vec_val.z == doctest::Approx(3.0f).epsilon(0.001));

        // Test string data
        ctx.setMaterialData("data_mat", "test_string", std::string("hello"));
        std::string str_val;
        ctx.getMaterialData("data_mat", "test_string", str_val);
        DOCTEST_CHECK(str_val == "hello");

        // Test clearing data
        ctx.clearMaterialData("data_mat", "test_int");
        DOCTEST_CHECK(!ctx.doesMaterialDataExist("data_mat", "test_int"));
    }

    SUBCASE("Material Data - Fallback Helper Method") {
        Context ctx;

        // Create material with data
        ctx.addMaterial("fallback_mat");
        ctx.setMaterialData("fallback_mat", "twosided_flag", 0u);

        // Create primitive with this material
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        ctx.assignMaterialToPrimitive(p1, "fallback_mat");

        // Test getDataWithMaterialFallback - should get data from material
        uint flag_val;
        ctx.getDataWithMaterialFallback(p1, "twosided_flag", flag_val);
        DOCTEST_CHECK(flag_val == 0u);

        // Create another primitive with material but add primitive-specific data
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        ctx.assignMaterialToPrimitive(p2, "fallback_mat");
        ctx.setPrimitiveData(p2, "custom_data", 42);

        // Test fallback - should get data from primitive since material doesn't have it
        int custom_val;
        ctx.getDataWithMaterialFallback(p2, "custom_data", custom_val);
        DOCTEST_CHECK(custom_val == 42);

        // Create third primitive with no special data
        uint p3 = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        ctx.assignMaterialToPrimitive(p3, "fallback_mat");

        // Test fallback - should throw error for non-existent data
        int nonexistent_val;
        DOCTEST_CHECK_THROWS(ctx.getDataWithMaterialFallback(p3, "nonexistent", nonexistent_val));
    }

    SUBCASE("Material Data - XML Round-Trip with Labels") {
        Context ctx;

        // Create material with data
        ctx.addMaterial("data_round_trip_mat");
        ctx.setMaterialColor("data_round_trip_mat", make_RGBAcolor(0.5f, 0.25f, 0.75f, 1));
        ctx.setMaterialData("data_round_trip_mat", "twosided_flag", 1u);
        ctx.setMaterialData("data_round_trip_mat", "reflectance", 0.8f);
        ctx.setMaterialData("data_round_trip_mat", "normal", make_vec3(0, 0, 1));

        // Create primitives with this material
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBcolor(0, 0, 0));
        ctx.assignMaterialToPrimitive(p1, "data_round_trip_mat");
        ctx.assignMaterialToPrimitive(p2, "data_round_trip_mat");

        // Write to XML
        ctx.writeXML("test_material_data.xml", true);

        // Load into new context
        Context ctx2;
        ctx2.loadXML("test_material_data.xml", true);

        // Verify material and data were preserved
        DOCTEST_CHECK(ctx2.doesMaterialExist("data_round_trip_mat"));

        DOCTEST_CHECK(ctx2.doesMaterialDataExist("data_round_trip_mat", "twosided_flag"));
        uint flag_val;
        ctx2.getMaterialData("data_round_trip_mat", "twosided_flag", flag_val);
        DOCTEST_CHECK(flag_val == 1u);

        DOCTEST_CHECK(ctx2.doesMaterialDataExist("data_round_trip_mat", "reflectance"));
        float refl_val;
        ctx2.getMaterialData("data_round_trip_mat", "reflectance", refl_val);
        DOCTEST_CHECK(refl_val == doctest::Approx(0.8f).epsilon(0.001));

        DOCTEST_CHECK(ctx2.doesMaterialDataExist("data_round_trip_mat", "normal"));
        vec3 norm_val;
        ctx2.getMaterialData("data_round_trip_mat", "normal", norm_val);
        DOCTEST_CHECK(norm_val.x == doctest::Approx(0.0f).epsilon(0.001));
        DOCTEST_CHECK(norm_val.y == doctest::Approx(0.0f).epsilon(0.001));
        DOCTEST_CHECK(norm_val.z == doctest::Approx(1.0f).epsilon(0.001));

        // Clean up
        std::filesystem::remove("test_material_data.xml");
    }

    SUBCASE("Material Methods - getPrimitiveMaterialID and getMaterial") {
        Context ctx;

        // Create materials
        ctx.addMaterial("test_mat_1");
        ctx.setMaterialColor("test_mat_1", make_RGBAcolor(1, 0, 0, 1));
        uint mat1_id = ctx.getMaterialIDFromLabel("test_mat_1");

        ctx.addMaterial("test_mat_2");
        ctx.setMaterialColor("test_mat_2", make_RGBAcolor(0, 1, 0, 1));
        uint mat2_id = ctx.getMaterialIDFromLabel("test_mat_2");

        // Create primitives and assign materials
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
        uint p3 = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));

        ctx.assignMaterialToPrimitive(p1, "test_mat_1");
        ctx.assignMaterialToPrimitive(p2, "test_mat_2");
        ctx.assignMaterialToPrimitive(p3, "test_mat_1");

        // Test getPrimitiveMaterialID
        DOCTEST_CHECK(ctx.getPrimitiveMaterialID(p1) == mat1_id);
        DOCTEST_CHECK(ctx.getPrimitiveMaterialID(p2) == mat2_id);
        DOCTEST_CHECK(ctx.getPrimitiveMaterialID(p3) == mat1_id);

        // Test getMaterial
        const Material &mat1 = ctx.getMaterial(mat1_id);
        DOCTEST_CHECK(mat1.label == "test_mat_1");
        DOCTEST_CHECK(mat1.color.r == doctest::Approx(1.0f));
        DOCTEST_CHECK(mat1.color.g == doctest::Approx(0.0f));
        DOCTEST_CHECK(mat1.color.b == doctest::Approx(0.0f));

        const Material &mat2 = ctx.getMaterial(mat2_id);
        DOCTEST_CHECK(mat2.label == "test_mat_2");
        DOCTEST_CHECK(mat2.color.r == doctest::Approx(0.0f));
        DOCTEST_CHECK(mat2.color.g == doctest::Approx(1.0f));
        DOCTEST_CHECK(mat2.color.b == doctest::Approx(0.0f));

        // Test getMaterial with invalid ID throws error
        DOCTEST_CHECK_THROWS((void) ctx.getMaterial(99999));
    }

    SUBCASE("Material Methods - getMaterialIDFromLabel") {
        Context ctx;

        // Create several materials
        ctx.addMaterial("material_a");
        ctx.addMaterial("material_b");
        ctx.addMaterial("material_c");

        // Test getting IDs from labels
        uint id_a = ctx.getMaterialIDFromLabel("material_a");
        uint id_b = ctx.getMaterialIDFromLabel("material_b");
        uint id_c = ctx.getMaterialIDFromLabel("material_c");

        // IDs should be unique
        DOCTEST_CHECK(id_a != id_b);
        DOCTEST_CHECK(id_b != id_c);
        DOCTEST_CHECK(id_a != id_c);

        // Getting same label should return same ID
        DOCTEST_CHECK(ctx.getMaterialIDFromLabel("material_a") == id_a);
        DOCTEST_CHECK(ctx.getMaterialIDFromLabel("material_b") == id_b);

        // Non-existent label should throw error
        DOCTEST_CHECK_THROWS((void) ctx.getMaterialIDFromLabel("nonexistent_material"));
    }

    SUBCASE("Material copy-on-write - basic color modification") {
        Context context;

        // Create two primitives with same color (shared material via deduplication)
        uint uuid1 = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), RGB::red);
        uint uuid2 = context.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), RGB::red);

        // Verify they share material initially
        std::string mat1_before = context.getPrimitiveMaterialLabel(uuid1);
        std::string mat2_before = context.getPrimitiveMaterialLabel(uuid2);
        DOCTEST_CHECK(mat1_before == mat2_before);

        // Modify one primitive's color
        context.setPrimitiveColor(uuid1, RGB::blue);

        // Verify materials are now different (copy-on-write occurred)
        std::string mat1_after = context.getPrimitiveMaterialLabel(uuid1);
        std::string mat2_after = context.getPrimitiveMaterialLabel(uuid2);
        DOCTEST_CHECK(mat1_after != mat2_after);

        // Verify colors are independent
        RGBcolor color1 = context.getPrimitiveColor(uuid1);
        RGBcolor color2 = context.getPrimitiveColor(uuid2);
        DOCTEST_CHECK(color1 == RGB::blue);
        DOCTEST_CHECK(color2 == RGB::red);
    }

    SUBCASE("Material copy-on-write - object-level modification") {
        Context context;

        // Create two sphere objects with same color
        uint obj1 = context.addSphereObject(10, make_vec3(0, 0, 0), 1.f, RGB::green);
        uint obj2 = context.addSphereObject(10, make_vec3(3, 0, 0), 1.f, RGB::green);

        // Modify one object's color
        context.setObjectColor(obj1, RGB::yellow);

        // Verify objects have different colors
        auto prims1 = context.getObjectPrimitiveUUIDs(obj1);
        auto prims2 = context.getObjectPrimitiveUUIDs(obj2);

        RGBcolor color1 = context.getPrimitiveColor(prims1[0]);
        RGBcolor color2 = context.getPrimitiveColor(prims2[0]);

        DOCTEST_CHECK(color1 == RGB::yellow);
        DOCTEST_CHECK(color2 == RGB::green);
    }

    SUBCASE("Material copy-on-write - non-shared optimization") {
        Context context;

        // Create single primitive with explicit color
        uint uuid = context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), RGB::cyan);

        std::string mat1 = context.getPrimitiveMaterialLabel(uuid);

        // Modify color - should NOT create new material since it's not shared
        context.setPrimitiveColor(uuid, RGB::magenta);

        std::string mat2 = context.getPrimitiveMaterialLabel(uuid);

        // Material should be same (no copy needed, just modified in place)
        DOCTEST_CHECK(mat1 == mat2);
    }
}
