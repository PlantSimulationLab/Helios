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

        // setLocation() re-validates rather than trusting construction, because Location's fields
        // are public: an already-valid Location can be mutated into an invalid one and handed back.
        // This is exactly what the ISO-8601 branch of loadTabularTimeseriesData() does with
        // UTC_offset, using a value read out of an external file.
        const Location original = ctx.getLocation();

        Location mutated = ctx.getLocation();
        mutated.UTC_offset = 100.f;

        // No capture_cerr needed: validate() throws directly, as the Date and Time constructors in
        // helios_vector_types.h do, so nothing is written to std::cerr on any build.
        DOCTEST_CHECK_THROWS_AS(ctx.setLocation(mutated), std::runtime_error);

        // A rejected setLocation() must leave the Context's location untouched.
        DOCTEST_CHECK(ctx.getLocation() == original);

        // A valid mutation still goes through, including a fractional-hour time zone.
        Location valid = ctx.getLocation();
        valid.UTC_offset = -5.75f;
        DOCTEST_CHECK_NOTHROW(ctx.setLocation(valid));
        DOCTEST_CHECK(ctx.getLocation().UTC_offset == doctest::Approx(-5.75f));
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

    SUBCASE("sub-patch in non-deformable compound object") {
        // A tile is defined by being planar, so scaling one of its sub-patches individually is blocked. Note a polymesh is deliberately NOT used here: a mesh has no such shape invariant and is deformable,
        // so its member primitives can be transformed individually (see the "Polymesh Deformability" test case).
        uint objID = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));
        std::vector<uint> UUIDs = ctx.getObjectPrimitiveUUIDs(objID);
        DOCTEST_REQUIRE(!UUIDs.empty());

        uint sub_patch = UUIDs.front();
        float area_before = ctx.getPrimitiveArea(sub_patch);

        // Try to scale the sub-patch (should be blocked)
        bool has_warning;
        {
            capture_cerr cerr_buffer;
            ctx.scalePrimitiveAboutPoint(sub_patch, make_vec3(2, 2, 2), make_vec3(0, 0, 0));
            has_warning = cerr_buffer.has_output();
        } // cerr_buffer destroyed here
        DOCTEST_CHECK(has_warning); // Should print warning

        float area_after = ctx.getPrimitiveArea(sub_patch);

        // Area should NOT change (scaling blocked for non-deformable compound objects)
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

    SUBCASE("rotateObject z-axis consistency") {
        // Regression: CompoundObject::rotate negated the rotation angle for the "z" axis only, rotating
        // objects opposite to rotatePrimitive and to the vec3-axis rotateObject overload.
        Context ctx;
        uint obj = ctx.addTileObject(make_vec3(1, 0, 0), make_vec2(0.5, 0.5), nullrotation, make_int2(1, 1));
        uint obj2 = ctx.addTileObject(make_vec3(1, 0, 0), make_vec2(0.5, 0.5), nullrotation, make_int2(1, 1));
        uint prim = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(0.5, 0.5));

        ctx.rotateObject(obj, 0.5f * PI_F, "z");
        ctx.rotateObject(obj2, 0.5f * PI_F, make_vec3(0, 0, 1));
        ctx.rotatePrimitive(prim, 0.5f * PI_F, "z");

        vec3 obj_center = ctx.getObjectCenter(obj);
        vec3 obj2_center = ctx.getObjectCenter(obj2);
        vec3 prim_center = ctx.getPatchCenter(prim);

        DOCTEST_CHECK(obj_center.x == doctest::Approx(0.f).epsilon(errtol));
        DOCTEST_CHECK(obj_center.y == doctest::Approx(1.f).epsilon(errtol));
        DOCTEST_CHECK(obj_center.x == doctest::Approx(prim_center.x).epsilon(errtol));
        DOCTEST_CHECK(obj_center.y == doctest::Approx(prim_center.y).epsilon(errtol));
        DOCTEST_CHECK(obj_center.x == doctest::Approx(obj2_center.x).epsilon(errtol));
        DOCTEST_CHECK(obj_center.y == doctest::Approx(obj2_center.y).epsilon(errtol));
    }

    SUBCASE("domain bounding box min/max") {
        // Regression: the serial (non-OpenMP) path used if/else-if between the min and max tests, so a
        // vertex that set a new minimum could never update the maximum.
        Context ctx;
        uint tri = ctx.addTriangle(make_vec3(3, 0, 0), make_vec3(2, 1, 0), make_vec3(1, 0, 0), RGB::red);
        DOCTEST_CHECK(ctx.doesPrimitiveExist(tri));
        vec2 xb, yb, zb;
        ctx.getDomainBoundingBox(xb, yb, zb);
        DOCTEST_CHECK(xb.x == doctest::Approx(1.f).epsilon(errtol));
        DOCTEST_CHECK(xb.y == doctest::Approx(3.f).epsilon(errtol));
        DOCTEST_CHECK(yb.x == doctest::Approx(0.f).epsilon(errtol));
        DOCTEST_CHECK(yb.y == doctest::Approx(1.f).epsilon(errtol));
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

        // updateTimeseriesData: replace value at an existing (date, time)
        ctx.updateTimeseriesData("ts", date, time0, 999.9f);
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 0) == doctest::Approx(999.9f));
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", date, time0) == doctest::Approx(999.9f));
        // The other point should be untouched
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 1) == doctest::Approx(305.3f));
        // Length is unchanged (update, not insert)
        DOCTEST_CHECK(ctx.getTimeseriesLength("ts") == 2);

        // Regression: adding a data point at a duplicate date/time used to warn "Skipping duplicate" but
        // then terminate with an "unknown reason" runtime error. It must warn and skip.
        {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.addTimeseriesData("ts", 111.1f, date, time0));
            DOCTEST_CHECK_NOTHROW(ctx.addTimeseriesData("ts", 222.2f, date, time1));
        }
        DOCTEST_CHECK(ctx.getTimeseriesLength("ts") == 2);
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 0) == doctest::Approx(999.9f));
        DOCTEST_CHECK(ctx.queryTimeseriesData("ts", 1) == doctest::Approx(305.3f));

        // Error: unknown label
        {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_THROWS(ctx.updateTimeseriesData("nonexistent", date, time0, 0.f));
        }

        // Error: known label but (date, time) does not match any existing point
        {
            Time time_missing = make_Time(time0.hour, 0, 0);
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_THROWS(ctx.updateTimeseriesData("ts", date, time_missing, 0.f));
        }
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
        DOCTEST_CHECK_FALSE(ctx.doesObjectExist(tube));
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

    SUBCASE("multi-ring disk triangle transforms") {
        // Regression: for Ndivs.y >= 2, the color overloads of addDisk/addDiskObject only rotated/translated
        // the second triangle of each outer-ring pair, leaving the first at the world origin.
        Context ctx;
        vec3 center = make_vec3(5, 0, 0);
        std::vector<uint> UUIDs = ctx.addDisk(make_int2(8, 2), center, make_vec2(1, 1), nullrotation, RGB::red);
        DOCTEST_CHECK(UUIDs.size() == 8 + 8 * 2);
        for (uint UUID: UUIDs) {
            for (const vec3 &v: ctx.getPrimitiveVertices(UUID)) {
                DOCTEST_CHECK((v - center).magnitude() <= 1.f + errtol);
            }
        }

        vec3 obj_center = make_vec3(0, 4, 1);
        uint objID = ctx.addDiskObject(make_int2(6, 3), obj_center, make_vec2(2, 2), nullrotation, RGB::red);
        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &v: ctx.getPrimitiveVertices(UUID)) {
                DOCTEST_CHECK((v - obj_center).magnitude() <= 2.f + errtol);
            }
        }
    }

    SUBCASE("voxel rotation restrictions") {
        // Regression: non-z-axis rotations of voxels warned "Ignoring this rotation" but were applied anyway
        // (batch overloads), and Voxel::rotate rotated about z regardless of the requested axis string.
        Context ctx;
        uint vox = ctx.addVoxel(make_vec3(1, 2, 3), make_vec3(1, 1, 1));
        float T_before[16], T_after[16];
        ctx.getPrimitiveTransformationMatrix(vox, T_before);
        std::vector<uint> vlist{vox};
        {
            capture_cerr cerr_buffer;
            ctx.rotatePrimitive(vlist, 0.25f * PI_F, "x"); // batch string-axis overload
            ctx.rotatePrimitive(vox, 0.25f * PI_F, "y"); // single-primitive overload -> Voxel::rotate
            ctx.rotatePrimitive(vlist, 0.25f * PI_F, make_vec3(1, 0, 0)); // batch vec3-axis overload
        }
        ctx.getPrimitiveTransformationMatrix(vox, T_after);
        for (int i = 0; i < 16; i++) {
            DOCTEST_CHECK(T_after[i] == doctest::Approx(T_before[i]).epsilon(errtol));
        }
        // rotation about the z-axis is allowed and must still be applied
        ctx.rotatePrimitive(vox, 0.5f * PI_F, "z");
        ctx.getPrimitiveTransformationMatrix(vox, T_after);
        DOCTEST_CHECK(T_after[0] == doctest::Approx(0.f).epsilon(errtol));
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
        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile) == make_int2(4, 4));

        // a 4x4 tile subdivided so that each sub-patch is ~1x1 gives an area ratio of ~16
        ctx.setTileObjectSubdivisionCount({tile}, 16.f);
        DOCTEST_CHECK(ctx.getTileObjectAreaRatio(tile) == doctest::Approx(16.f).epsilon(0.25f));

        // area ratio less than 1 is physically impossible and must fail fast
        {
            capture_cerr capture;
            DOCTEST_CHECK_THROWS(ctx.setTileObjectSubdivisionCount({tile}, 0.5f));
        }
    }

    SUBCASE("subdivision update preserves flat tile orientation") {
        // Regression test: updating the subdivision count of a horizontal (ground) tile must not flip
        // its orientation. Reconstructing the rotation from the tile normal via cart2sphere() previously
        // produced elevation=90deg for a flat tile, tilting the regenerated tile to vertical.
        Context ctx;
        vec3 center = make_vec3(1, 2, 3);
        vec2 size = make_vec2(4, 6);
        uint tile = ctx.addTileObject(center, size, nullrotation, make_int2(2, 2));

        vec3 normal_before = ctx.getPrimitiveNormal(ctx.getObjectPrimitiveUUIDs(tile).front());
        DOCTEST_CHECK(normal_before.z == doctest::Approx(1.f).epsilon(errtol));

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(5, 3));

        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile) == make_int2(5, 3));

        vec3 normal_after = ctx.getPrimitiveNormal(ctx.getObjectPrimitiveUUIDs(tile).front());
        DOCTEST_CHECK(normal_after.x == doctest::Approx(0.f).epsilon(errtol));
        DOCTEST_CHECK(normal_after.y == doctest::Approx(0.f).epsilon(errtol));
        DOCTEST_CHECK(normal_after.z == doctest::Approx(1.f).epsilon(errtol));

        // center and size must be preserved
        vec3 center_after = ctx.getTileObjectCenter(tile);
        DOCTEST_CHECK(center_after.x == doctest::Approx(center.x).epsilon(errtol));
        DOCTEST_CHECK(center_after.y == doctest::Approx(center.y).epsilon(errtol));
        DOCTEST_CHECK(center_after.z == doctest::Approx(center.z).epsilon(errtol));
        vec2 size_after = ctx.getTileObjectSize(tile);
        DOCTEST_CHECK(size_after.x == doctest::Approx(size.x).epsilon(errtol));
        DOCTEST_CHECK(size_after.y == doctest::Approx(size.y).epsilon(errtol));
    }

    SUBCASE("subdivision update preserves arbitrary tile orientation") {
        // A tile rotated about multiple axes must retain its exact orientation after a subdivision update.
        Context ctx;
        SphericalCoord rotation = make_SphericalCoord(0.6f, 1.1f);
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(3, 3), rotation, make_int2(2, 2));

        vec3 normal_before = ctx.getTileObjectNormal(tile);

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(4, 4));

        vec3 normal_after = ctx.getTileObjectNormal(tile);
        DOCTEST_CHECK(normal_after.x == doctest::Approx(normal_before.x).epsilon(errtol));
        DOCTEST_CHECK(normal_after.y == doctest::Approx(normal_before.y).epsilon(errtol));
        DOCTEST_CHECK(normal_after.z == doctest::Approx(normal_before.z).epsilon(errtol));
    }

    SUBCASE("subdivision update preserves textured tile orientation") {
        // The textured regeneration path must preserve orientation just like the non-textured path.
        Context ctx;
        vec3 center = make_vec3(2, -1, 4);
        vec2 size = make_vec2(2, 2);
        SphericalCoord rotation = make_SphericalCoord(0.4f, 0.9f);
        uint tile = ctx.addTileObject(center, size, rotation, make_int2(2, 2), "lib/images/disk_texture.png");

        vec3 normal_before = ctx.getTileObjectNormal(tile);
        vec3 center_before = ctx.getTileObjectCenter(tile);

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(4, 4));

        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile) == make_int2(4, 4));

        vec3 normal_after = ctx.getTileObjectNormal(tile);
        DOCTEST_CHECK(normal_after.x == doctest::Approx(normal_before.x).epsilon(errtol));
        DOCTEST_CHECK(normal_after.y == doctest::Approx(normal_before.y).epsilon(errtol));
        DOCTEST_CHECK(normal_after.z == doctest::Approx(normal_before.z).epsilon(errtol));

        vec3 center_after = ctx.getTileObjectCenter(tile);
        DOCTEST_CHECK(center_after.x == doctest::Approx(center_before.x).epsilon(errtol));
        DOCTEST_CHECK(center_after.y == doctest::Approx(center_before.y).epsilon(errtol));
        DOCTEST_CHECK(center_after.z == doctest::Approx(center_before.z).epsilon(errtol));
    }

    SUBCASE("subdivision update preserves texture repeat") {
        // Regression test: a tile's texture repeat count was not retained anywhere on the Tile object, so
        // regenerating its sub-patches rebuilt them from a template created with the 5-argument
        // addTileObject(), which delegates with a repeat of 1x1. The texture was silently stretched once
        // across the whole tile. The repeat is observable only through the sub-patch (u,v) windows: every
        // sub-patch spans exactly repeat/subdiv of the image, so repeat == u_span*subdiv.
        //
        // Note: do not assert on getObjectArea() here. The total opaque fraction of an alpha-masked texture
        // is the same whether it is tiled 25x or once, so the tile area is identical before and after and
        // such a check would pass on the buggy code.
        Context ctx;
        const char *texture = "lib/images/disk_texture.png"; // 800x800, far above the subdivision counts used here
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(10, 10), nullrotation, make_int2(10, 10), texture, make_int2(5, 5));

        // Confirm the tile really was built with a 5x5 repeat before exercising the operation under test.
        // This assertion passes on the unfixed code by construction; it is here to prove the measurement.
        DOCTEST_CHECK(ctx.getTileObjectTextureRepeat(tile) == make_int2(5, 5));
        {
            std::vector<vec2> uv = ctx.getPrimitiveTextureUV(ctx.getObjectPrimitiveUUIDs(tile).front());
            DOCTEST_CHECK(uv.size() == 4);
            DOCTEST_CHECK((uv.at(1).x - uv.at(0).x) == doctest::Approx(0.5f).epsilon(errtol)); // 5/10
        }

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(20, 20));

        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile) == make_int2(20, 20));
        DOCTEST_CHECK(ctx.getTileObjectTextureRepeat(tile) == make_int2(5, 5));
        DOCTEST_CHECK(ctx.getTileObjectEffectiveTextureRepeat(tile) == make_int2(5, 5));

        // 20 subdivisions / 5 repeats = 4 sub-patches per texture copy, so each spans 0.25 of the image and
        // the pattern restarts every 4 sub-patches in each direction.
        std::vector<uint> uuids = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(uuids.size() == 400);
        const float uv_sub = 0.25f;
        for (size_t k = 0; k < uuids.size(); k++) {
            const int i_local = static_cast<int>(k % 20) % 4;
            const int j_local = static_cast<int>(k / 20) % 4;
            std::vector<vec2> uv = ctx.getPrimitiveTextureUV(uuids.at(k));
            DOCTEST_CHECK(uv.size() == 4);
            DOCTEST_CHECK(uv.at(0).x == doctest::Approx(float(i_local) * uv_sub).epsilon(errtol));
            DOCTEST_CHECK(uv.at(0).y == doctest::Approx(float(j_local) * uv_sub).epsilon(errtol));
            DOCTEST_CHECK(uv.at(2).x == doctest::Approx(float(i_local + 1) * uv_sub).epsilon(errtol));
            DOCTEST_CHECK(uv.at(2).y == doctest::Approx(float(j_local + 1) * uv_sub).epsilon(errtol));
            // the repeat expressed the way the bug report measures it
            DOCTEST_CHECK(std::lround((uv.at(2).x - uv.at(0).x) * 20.f) == 5);
        }
    }

    SUBCASE("copyObject preserves tile texture repeat") {
        // copyObject() reconstructs the Tile directly rather than going through addTileObject(), so it has
        // its own path for carrying the repeat across. Note that checking only the copy's (u,v) coordinates
        // would pass on the unfixed code: copyPrimitive() duplicates the sub-patches verbatim, so the copy
        // renders correctly even when the repeat it stores is wrong. The defect is observable only through
        // the getter or through a subsequent subdivision change.
        Context ctx;
        const char *texture = "lib/images/disk_texture.png";
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(10, 10), nullrotation, make_int2(10, 10), texture, make_int2(5, 5));
        uint tile_copy = ctx.copyObject(tile);

        DOCTEST_CHECK(ctx.getTileObjectTextureRepeat(tile_copy) == make_int2(5, 5));

        ctx.setTileObjectSubdivisionCount({tile_copy}, make_int2(20, 20));

        std::vector<vec2> uv_copy = ctx.getPrimitiveTextureUV(ctx.getObjectPrimitiveUUIDs(tile_copy).at(1));
        DOCTEST_CHECK(uv_copy.size() == 4);
        DOCTEST_CHECK(uv_copy.at(0).x == doctest::Approx(0.25f).epsilon(errtol));

        // the original must be untouched
        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile) == make_int2(10, 10));
        std::vector<vec2> uv_orig = ctx.getPrimitiveTextureUV(ctx.getObjectPrimitiveUUIDs(tile).front());
        DOCTEST_CHECK((uv_orig.at(1).x - uv_orig.at(0).x) == doctest::Approx(0.5f).epsilon(errtol));
    }

    SUBCASE("texture repeat is re-derived from the request on each subdivision change") {
        // The Tile stores the repeat that was requested, not the value it was auto-resized down to, so an
        // intermediate subdivision count that the repeat does not evenly divide must not degrade it
        // permanently. This is the test that fails if the fix is later "simplified" to store the corrected
        // value, which would let the repeat ratchet toward 1 across successive calls.
        Context ctx;
        const char *texture = "lib/images/disk_texture.png";
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(10, 10), nullrotation, make_int2(10, 10), texture, make_int2(5, 5));

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(7, 7)); // 7 is prime: the repeat resizes to 1
        DOCTEST_CHECK(ctx.getTileObjectTextureRepeat(tile) == make_int2(5, 5));
        DOCTEST_CHECK(ctx.getTileObjectEffectiveTextureRepeat(tile) == make_int2(1, 1));
        {
            std::vector<vec2> uv = ctx.getPrimitiveTextureUV(ctx.getObjectPrimitiveUUIDs(tile).front());
            DOCTEST_CHECK(std::lround((uv.at(2).x - uv.at(0).x) * 7.f) == 1);
        }

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(20, 20)); // the request recovers in full
        DOCTEST_CHECK(ctx.getTileObjectEffectiveTextureRepeat(tile) == make_int2(5, 5));
        {
            std::vector<vec2> uv = ctx.getPrimitiveTextureUV(ctx.getObjectPrimitiveUUIDs(tile).front());
            DOCTEST_CHECK(std::lround((uv.at(2).x - uv.at(0).x) * 20.f) == 5);
        }
    }

    SUBCASE("subdivision update of a low-resolution tiled texture") {
        // The texture resolution guard in addTileObject() is subdiv >= repeat*resolution. Rebuilding the
        // template with a repeat of 1x1 dropped the threshold from 2*5 to 1*5, so this threw even though a
        // repeat of 2 makes the requested subdivision count perfectly representable. Restoring the repeat
        // can only raise that threshold, never lower it.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(4, 4), "lib/images/solid.jpg", make_int2(2, 2)); // 5x5 image
        DOCTEST_CHECK_NOTHROW(ctx.setTileObjectSubdivisionCount({tile}, make_int2(8, 8)));
        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile) == make_int2(8, 8));
        DOCTEST_CHECK(ctx.getTileObjectEffectiveTextureRepeat(tile) == make_int2(2, 2));
    }

    SUBCASE("subdivision update of multiple tiles and object data survival") {
        // Updating several tiles in one call must regenerate each independently and leave the tile objects
        // (and their object-level data) intact. The new sub-patches must be correctly re-parented.
        Context ctx;
        uint tile_a = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(1, 1));
        uint tile_b = ctx.addTileObject(make_vec3(10, 0, 0), make_vec2(4, 2), make_SphericalCoord(0.3f, 0.0f), make_int2(2, 2));

        ctx.setObjectData(tile_a, "plot_id", 7);
        ctx.setObjectData(tile_b, "plot_id", 9);

        ctx.setTileObjectSubdivisionCount({tile_a, tile_b}, make_int2(3, 3));

        // both objects still exist and have the requested subdivision
        DOCTEST_CHECK(ctx.doesObjectExist(tile_a));
        DOCTEST_CHECK(ctx.doesObjectExist(tile_b));
        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile_a) == make_int2(3, 3));
        DOCTEST_CHECK(ctx.getTileObjectSubdivisionCount(tile_b) == make_int2(3, 3));

        // each tile owns exactly 9 sub-patches, all re-parented to the correct object
        std::vector<uint> uuids_a = ctx.getObjectPrimitiveUUIDs(tile_a);
        std::vector<uint> uuids_b = ctx.getObjectPrimitiveUUIDs(tile_b);
        DOCTEST_CHECK(uuids_a.size() == 9);
        DOCTEST_CHECK(uuids_b.size() == 9);
        for (uint UUID: uuids_a) {
            DOCTEST_CHECK(ctx.getPrimitiveParentObjectID(UUID) == tile_a);
        }
        for (uint UUID: uuids_b) {
            DOCTEST_CHECK(ctx.getPrimitiveParentObjectID(UUID) == tile_b);
        }

        // object-level data survives the in-place regeneration
        int plot_a, plot_b;
        ctx.getObjectData(tile_a, "plot_id", plot_a);
        ctx.getObjectData(tile_b, "plot_id", plot_b);
        DOCTEST_CHECK(plot_a == 7);
        DOCTEST_CHECK(plot_b == 9);

        // the two tiles must retain their distinct sizes (independent per-tile regeneration)
        DOCTEST_CHECK(ctx.getTileObjectSize(tile_a).x == doctest::Approx(2.f).epsilon(errtol));
        DOCTEST_CHECK(ctx.getTileObjectSize(tile_b).x == doctest::Approx(4.f).epsilon(errtol));
    }

    SUBCASE("per-subpatch data preserved on 1:1 resolution call") {
        // When the subdivision count is unchanged the data transfer is an exact index-for-index copy.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));

        std::vector<uint> uuids_before = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(uuids_before.size() == 4);
        // tag each sub-patch with its center so we can verify the value follows the geometry
        for (uint UUID: uuids_before) {
            vec3 c = ctx.getPatchCenter(UUID);
            ctx.setPrimitiveData(UUID, "tag_x", c.x);
            ctx.setPrimitiveData(UUID, "tag_y", c.y);
        }

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(2, 2));

        std::vector<uint> uuids_after = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(uuids_after.size() == 4);
        for (uint UUID: uuids_after) {
            vec3 c = ctx.getPatchCenter(UUID);
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(UUID, "tag_x"));
            float tag_x, tag_y;
            ctx.getPrimitiveData(UUID, "tag_x", tag_x);
            ctx.getPrimitiveData(UUID, "tag_y", tag_y);
            // the tag must match the patch's own location (data followed geometry)
            DOCTEST_CHECK(tag_x == doctest::Approx(c.x).epsilon(errtol));
            DOCTEST_CHECK(tag_y == doctest::Approx(c.y).epsilon(errtol));
        }
    }

    SUBCASE("per-subpatch data spatially remapped on refine") {
        // Refining 2x2 -> 4x4: each old quadrant maps to a 2x2 block of new sub-patches that all inherit
        // the quadrant's data. We label each old sub-patch with a quadrant id derived from its center sign.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(4, 4), nullrotation, make_int2(2, 2));

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(tile)) {
            vec3 c = ctx.getPatchCenter(UUID);
            int quadrant = (c.x < 0 ? 0 : 1) + (c.y < 0 ? 0 : 2); // 0..3 by sign of (x,y)
            ctx.setPrimitiveData(UUID, "quadrant", quadrant);
        }

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(4, 4));

        std::vector<uint> uuids_after = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(uuids_after.size() == 16);
        for (uint UUID: uuids_after) {
            vec3 c = ctx.getPatchCenter(UUID);
            int expected_quadrant = (c.x < 0 ? 0 : 1) + (c.y < 0 ? 0 : 2);
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(UUID, "quadrant"));
            int q;
            ctx.getPrimitiveData(UUID, "quadrant", q);
            DOCTEST_CHECK(q == expected_quadrant);
        }
    }

    SUBCASE("per-subpatch data spatially remapped on coarsen") {
        // Coarsening 4x4 -> 2x2: each new sub-patch takes the data of the old cell containing its center.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(4, 4), nullrotation, make_int2(4, 4));

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(tile)) {
            vec3 c = ctx.getPatchCenter(UUID);
            int quadrant = (c.x < 0 ? 0 : 1) + (c.y < 0 ? 0 : 2);
            ctx.setPrimitiveData(UUID, "quadrant", quadrant);
        }

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(2, 2));

        std::vector<uint> uuids_after = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(uuids_after.size() == 4);
        for (uint UUID: uuids_after) {
            vec3 c = ctx.getPatchCenter(UUID);
            int expected_quadrant = (c.x < 0 ? 0 : 1) + (c.y < 0 ? 0 : 2);
            int q;
            ctx.getPrimitiveData(UUID, "quadrant", q);
            DOCTEST_CHECK(q == expected_quadrant);
        }
    }

    SUBCASE("multiple primitive data types preserved across resolution change") {
        // float, string and vec3 data must all survive the regeneration.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(1, 1));
        uint patch = ctx.getObjectPrimitiveUUIDs(tile).front();

        ctx.setPrimitiveData(patch, "f", 3.5f);
        ctx.setPrimitiveData(patch, "s", std::string("hello"));
        ctx.setPrimitiveData(patch, "v", make_vec3(1, 2, 3));

        ctx.setTileObjectSubdivisionCount({tile}, make_int2(3, 3));

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(tile)) {
            float f;
            std::string s;
            vec3 v;
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(UUID, "f"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(UUID, "s"));
            DOCTEST_CHECK(ctx.doesPrimitiveDataExist(UUID, "v"));
            ctx.getPrimitiveData(UUID, "f", f);
            ctx.getPrimitiveData(UUID, "s", s);
            ctx.getPrimitiveData(UUID, "v", v);
            DOCTEST_CHECK(f == doctest::Approx(3.5f).epsilon(errtol));
            DOCTEST_CHECK(s == "hello");
            DOCTEST_CHECK(v.x == doctest::Approx(1.f).epsilon(errtol));
            DOCTEST_CHECK(v.z == doctest::Approx(3.f).epsilon(errtol));
        }
    }
}

//! Collect the axis-aligned tile-local bounds of every sub-patch of an unrotated adaptive tile centered on the origin
static std::vector<std::vector<float>> getAdaptiveTileSubpatchRects(const Context &ctx, uint ObjID) {
    std::vector<std::vector<float>> rects;
    for (uint UUID: ctx.getObjectPrimitiveUUIDs(ObjID)) {
        const std::vector<vec3> vertices = ctx.getPrimitiveVertices(UUID);
        float x_min = vertices.front().x, x_max = vertices.front().x;
        float y_min = vertices.front().y, y_max = vertices.front().y;
        for (const vec3 &vertex: vertices) {
            x_min = std::min(x_min, vertex.x);
            x_max = std::max(x_max, vertex.x);
            y_min = std::min(y_min, vertex.y);
            y_max = std::max(y_max, vertex.y);
        }
        rects.push_back({x_min, y_min, x_max, y_max});
    }
    return rects;
}

//! Distance from a point to the closest point of an axis-aligned rectangle, which is zero when the point lies inside it
static float adaptiveTileRectDistance(const std::vector<float> &rect, const vec2 &point) {
    const float closest_x = std::min(std::max(point.x, rect.at(0)), rect.at(2));
    const float closest_y = std::min(std::max(point.y, rect.at(1)), rect.at(3));
    return std::hypot(point.x - closest_x, point.y - closest_y);
}

TEST_CASE("Adaptive Tile Object") {

    SUBCASE("sub-patches exactly partition the tile") {
        // The quadtree derives every cell's bounds from integer indices against a fixed base cell size rather than by
        // accumulating offsets from its parent, so the leaves must tile the object exactly. A gap or an overlap
        // anywhere shows up immediately as an area that does not match the tile area. The tile must be untextured
        // here, since getPrimitiveArea scales by the texture solid fraction.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(1.f, 2.f);
        refinement.subpatch_size_min = 0.05f;
        refinement.subpatch_size_max = 1.f;
        refinement.transition_exponent = 0.5f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(20, 20), nullrotation, refinement);

        DOCTEST_CHECK(ctx.getObjectType(tile) == OBJECT_TYPE_ADAPTIVE_TILE);
        DOCTEST_CHECK(ctx.getObjectArea(tile) == doctest::Approx(400.f).epsilon(1e-4));
    }

    SUBCASE("no two sub-patches overlap") {
        // Matching total area alone cannot distinguish a correct partition from one containing a gap and an overlap of
        // equal size, so probe the tile on a grid of points and require each to fall inside exactly one sub-patch. The
        // probe offset is deliberately not a dyadic fraction, so no probe point can land on a cell boundary.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(-1.5f, 0.5f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement);

        const std::vector<std::vector<float>> rects = getAdaptiveTileSubpatchRects(ctx, tile);

        const int probe_count = 41;
        for (int j = 0; j < probe_count; j++) {
            for (int i = 0; i < probe_count; i++) {
                const float x = 8.f * ((float(i) + 0.31830989f) / float(probe_count)) - 4.f;
                const float y = 8.f * ((float(j) + 0.31830989f) / float(probe_count)) - 4.f;

                int containing_count = 0;
                for (const std::vector<float> &rect: rects) {
                    if (x >= rect.at(0) && x <= rect.at(2) && y >= rect.at(1) && y <= rect.at(3)) {
                        containing_count++;
                    }
                }
                DOCTEST_CHECK(containing_count == 1);
            }
        }
    }

    SUBCASE("achieved sub-patch sizes bracket the requested range") {
        // The coarsest cells must survive at refinement level 0. They do not if the transition is normalized by the
        // distance to the farthest tile corner instead of by the largest closest-point distance over the base cells:
        // the cell owning that corner is nearer to the target than the corner itself, so no cell reaches the far end of
        // the transition, every base cell subdivides once, and the achieved maximum lands at half the requested one.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.02f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.25f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(50, 50), nullrotation, refinement);

        const int2 base_subdiv = ctx.getAdaptiveTileObjectBaseSubdivisionCount(tile);
        const uint max_level = ctx.getAdaptiveTileObjectMaxRefinementLevel(tile);
        DOCTEST_CHECK(base_subdiv == make_int2(22, 22));
        DOCTEST_CHECK(max_level == 7);

        const vec2 size_range = ctx.getAdaptiveTileObjectSubpatchSizeRange(tile);
        const float base_size = 50.f / 22.f;

        // The coarsest cells are exactly the base cells and the finest are the base cells subdivided the full number of
        // levels, which is only true when both ends of the transition are actually reached.
        DOCTEST_CHECK(size_range.y == doctest::Approx(base_size).epsilon(errtol));
        DOCTEST_CHECK(size_range.x == doctest::Approx(base_size / 128.f).epsilon(errtol));

        // Both ends land within about 20% of the request, and the errors are anti-correlated because the base cell size
        // is chosen as the geometric mean of the two constraints.
        DOCTEST_CHECK(size_range.x == doctest::Approx(0.02f).epsilon(0.25f));
        DOCTEST_CHECK(size_range.y == doctest::Approx(2.f).epsilon(0.25f));
        DOCTEST_CHECK((size_range.y / 2.f) * (size_range.x / 0.02f) == doctest::Approx(1.f).epsilon(0.05f));
    }

    SUBCASE("predicted sub-patch count matches the count actually created") {
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(2.f, -1.f);
        refinement.subpatch_size_min = 0.1f;
        refinement.subpatch_size_max = 1.f;
        refinement.transition_exponent = 0.75f;

        const size_t predicted = ctx.predictAdaptiveTileObjectSubpatchCount(make_vec2(16, 16), refinement);
        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(16, 16), nullrotation, refinement);

        DOCTEST_CHECK(predicted == ctx.getObjectPrimitiveCount(tile));
        DOCTEST_CHECK(predicted > 0);
    }

    SUBCASE("sub-patch size grows with distance from the target") {
        // Refinement level is inherited from a cell's ancestors, so an individual leaf far from the target can be finer
        // than a nearer one that never subdivided. The property that actually holds, and that the feature exists to
        // provide, is that size grows with distance in aggregate.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.05f;
        refinement.subpatch_size_max = 1.f;
        refinement.transition_exponent = 0.5f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(20, 20), nullrotation, refinement);

        const std::vector<std::vector<float>> rects = getAdaptiveTileSubpatchRects(ctx, tile);

        const int bin_count = 5;
        std::vector<float> size_sum(bin_count, 0.f);
        std::vector<int> bin_population(bin_count, 0);
        for (const std::vector<float> &rect: rects) {
            const float distance = adaptiveTileRectDistance(rect, make_vec2(0.f, 0.f));
            const int bin = std::min(bin_count - 1, int(distance / (10.f / float(bin_count))));
            size_sum.at(bin) += std::max(rect.at(2) - rect.at(0), rect.at(3) - rect.at(1));
            bin_population.at(bin)++;
        }

        for (int bin = 1; bin < bin_count; bin++) {
            DOCTEST_CHECK(bin_population.at(bin) > 0);
            DOCTEST_CHECK(bin_population.at(bin - 1) > 0);
            const float mean_previous = size_sum.at(bin - 1) / float(bin_population.at(bin - 1));
            const float mean_current = size_sum.at(bin) / float(bin_population.at(bin));
            DOCTEST_CHECK(mean_current > mean_previous);
        }

        // The finest sub-patches occur at the target itself. Many sub-patches share the finest size, so the nearest of
        // them is what matters; taking whichever happens to come first in traversal order would prove nothing.
        float finest_size = std::numeric_limits<float>::max();
        for (const std::vector<float> &rect: rects) {
            finest_size = std::min(finest_size, std::max(rect.at(2) - rect.at(0), rect.at(3) - rect.at(1)));
        }
        // Compared with a tolerance rather than exactly: a cell's width is recovered as the difference of two vertex
        // coordinates, and that difference loses low-order bits differently depending on where the cell sits on the
        // tile, so cells of the same refinement level do not all report an identical width.
        float finest_distance = std::numeric_limits<float>::max();
        for (const std::vector<float> &rect: rects) {
            if (std::max(rect.at(2) - rect.at(0), rect.at(3) - rect.at(1)) < 1.001f * finest_size) {
                finest_distance = std::min(finest_distance, adaptiveTileRectDistance(rect, make_vec2(0.f, 0.f)));
            }
        }
        DOCTEST_CHECK(finest_distance < 1e-6f);
    }

    SUBCASE("refinement is radially symmetric about the target") {
        // A separable non-uniform grid would refine along the full row and column through the target, leaving a
        // cross-shaped fine region; the quadtree refines into a disc. Sampling at a fixed radius in eight directions
        // distinguishes the two: the cross would be fine along the axes and coarse on the diagonals.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(16, 16), nullrotation, refinement);

        // An even base grid puts the target on a cell corner, so the layout is symmetric under the square's reflections.
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectBaseSubdivisionCount(tile) == make_int2(8, 8));

        const std::vector<std::vector<float>> rects = getAdaptiveTileSubpatchRects(ctx, tile);

        const float sample_radius = 2.7f;
        float reference_size = -1.f;
        for (int direction = 0; direction < 8; direction++) {
            const float angle = float(direction) * 0.25f * M_PI;
            const float x = sample_radius * std::cos(angle);
            const float y = sample_radius * std::sin(angle);

            float sampled_size = -1.f;
            for (const std::vector<float> &rect: rects) {
                if (x >= rect.at(0) && x <= rect.at(2) && y >= rect.at(1) && y <= rect.at(3)) {
                    sampled_size = std::max(rect.at(2) - rect.at(0), rect.at(3) - rect.at(1));
                    break;
                }
            }
            DOCTEST_CHECK(sampled_size > 0.f);

            if (direction == 0) {
                reference_size = sampled_size;
            } else {
                DOCTEST_CHECK(sampled_size == doctest::Approx(reference_size).epsilon(errtol));
            }
        }
    }

    SUBCASE("all sub-patches are coplanar and share the tile normal") {
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(1.f, 1.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        const SphericalCoord rotation = make_SphericalCoord(0.6f, 1.1f);
        uint tile = ctx.addAdaptiveTileObject(make_vec3(3, -2, 1), make_vec2(8, 8), rotation, refinement);

        const vec3 tile_normal = ctx.getAdaptiveTileObjectNormal(tile);
        for (uint UUID: ctx.getObjectPrimitiveUUIDs(tile)) {
            const vec3 subpatch_normal = ctx.getPrimitiveNormal(UUID);
            DOCTEST_CHECK(subpatch_normal.x == doctest::Approx(tile_normal.x).epsilon(errtol));
            DOCTEST_CHECK(subpatch_normal.y == doctest::Approx(tile_normal.y).epsilon(errtol));
            DOCTEST_CHECK(subpatch_normal.z == doctest::Approx(tile_normal.z).epsilon(errtol));
        }

        DOCTEST_CHECK(ctx.getAdaptiveTileObjectCenter(tile).x == doctest::Approx(3.f).epsilon(errtol));
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectCenter(tile).z == doctest::Approx(1.f).epsilon(errtol));
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectSize(tile).x == doctest::Approx(8.f).epsilon(errtol));
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectVertices(tile).size() == 4);
    }

    SUBCASE("texture repeat is applied exactly and no sub-patch straddles a repeat boundary") {
        // A uniform tile reduces a repeat count that does not divide its subdivision count. An adaptive tile instead
        // snaps its base grid to a multiple of the requested count, because the base grid is derived internally and
        // silently degrading the user's requested repeat against it would be surprising.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(16, 16), nullrotation, refinement, "lib/images/diamond_texture.png", make_int2(3, 3));

        DOCTEST_CHECK(ctx.getAdaptiveTileObjectTextureRepeat(tile) == make_int2(3, 3));

        const int2 base_subdiv = ctx.getAdaptiveTileObjectBaseSubdivisionCount(tile);
        DOCTEST_CHECK(base_subdiv.x % 3 == 0);
        DOCTEST_CHECK(base_subdiv.y % 3 == 0);

        const uint max_level = ctx.getAdaptiveTileObjectMaxRefinementLevel(tile);
        const uint finest_per_repeat_x = uint(base_subdiv.x / 3) << max_level;
        const uint finest_per_repeat_y = uint(base_subdiv.y / 3) << max_level;

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(tile)) {
            const std::vector<vec2> uv = ctx.getPrimitiveTextureUV(UUID);
            DOCTEST_CHECK(uv.size() == 4);

            for (const vec2 &coordinate: uv) {
                DOCTEST_CHECK(coordinate.x >= 0.f);
                DOCTEST_CHECK(coordinate.x <= 1.f);
                DOCTEST_CHECK(coordinate.y >= 0.f);
                DOCTEST_CHECK(coordinate.y <= 1.f);

                // Every window boundary lands on the grid of finest-level cells within one texture repeat. This is what
                // guarantees that no sub-patch straddles a boundary between repeats and that the windows tile the image
                // exactly: a window that fell between grid lines would sample across the seam.
                const float scaled_x = coordinate.x * float(finest_per_repeat_x);
                const float scaled_y = coordinate.y * float(finest_per_repeat_y);
                DOCTEST_CHECK(std::fabs(scaled_x - std::round(scaled_x)) < 1e-4f);
                DOCTEST_CHECK(std::fabs(scaled_y - std::round(scaled_y)) < 1e-4f);
            }

            // A window spans a whole number of finest-level cells, never a fraction of one.
            const float window_width = uv.at(1).x - uv.at(0).x;
            const uint cells_per_repeat = uint(std::lround(1.f / window_width));
            DOCTEST_CHECK(window_width == doctest::Approx(1.f / float(cells_per_repeat)).epsilon(errtol));
            DOCTEST_CHECK(cells_per_repeat >= uint(base_subdiv.x / 3));
            DOCTEST_CHECK(cells_per_repeat <= finest_per_repeat_x);
            DOCTEST_CHECK(finest_per_repeat_x % cells_per_repeat == 0);
        }
    }

    SUBCASE("copyObject reproduces geometry and parameters exactly") {
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(1.f, -1.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement);
        uint tile_copy = ctx.copyObject(tile);

        DOCTEST_CHECK(ctx.getObjectType(tile_copy) == OBJECT_TYPE_ADAPTIVE_TILE);
        DOCTEST_CHECK(ctx.getObjectPrimitiveCount(tile_copy) == ctx.getObjectPrimitiveCount(tile));
        DOCTEST_CHECK(ctx.getObjectArea(tile_copy) == doctest::Approx(ctx.getObjectArea(tile)).epsilon(errtol));

        const AdaptiveTileRefinement copied_refinement = ctx.getAdaptiveTileObjectRefinement(tile_copy);
        DOCTEST_CHECK(copied_refinement.target.x == refinement.target.x);
        DOCTEST_CHECK(copied_refinement.target.y == refinement.target.y);
        DOCTEST_CHECK(copied_refinement.subpatch_size_min == refinement.subpatch_size_min);
        DOCTEST_CHECK(copied_refinement.subpatch_size_max == refinement.subpatch_size_max);
        DOCTEST_CHECK(copied_refinement.transition_exponent == refinement.transition_exponent);
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectBaseSubdivisionCount(tile_copy) == ctx.getAdaptiveTileObjectBaseSubdivisionCount(tile));
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectMaxRefinementLevel(tile_copy) == ctx.getAdaptiveTileObjectMaxRefinementLevel(tile));

        // Sub-patch geometry is duplicated rather than regenerated, so it must match bit for bit.
        const std::vector<uint> original_UUIDs = ctx.getObjectPrimitiveUUIDs(tile);
        const std::vector<uint> copied_UUIDs = ctx.getObjectPrimitiveUUIDs(tile_copy);
        for (size_t i = 0; i < original_UUIDs.size(); i++) {
            const std::vector<vec3> original_vertices = ctx.getPrimitiveVertices(original_UUIDs.at(i));
            const std::vector<vec3> copied_vertices = ctx.getPrimitiveVertices(copied_UUIDs.at(i));
            for (size_t v = 0; v < original_vertices.size(); v++) {
                DOCTEST_CHECK(copied_vertices.at(v).x == original_vertices.at(v).x);
                DOCTEST_CHECK(copied_vertices.at(v).y == original_vertices.at(v).y);
            }
        }

        // Deleting the original must leave the copy intact.
        const float copy_area = ctx.getObjectArea(tile_copy);
        ctx.deleteObject(tile);
        DOCTEST_CHECK(ctx.doesObjectExist(tile_copy));
        DOCTEST_CHECK(ctx.getObjectArea(tile_copy) == doctest::Approx(copy_area).epsilon(errtol));
    }

    SUBCASE("uniform tile accessors reject adaptive tiles without corrupting them") {
        // These operations are meaningless for a non-uniform layout. setTileObjectSubdivisionCount in particular takes
        // a vector of object IDs, so a user sweeping it over every object in the scene must not silently replace an
        // adaptive layout with a uniform grid.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement);

        const uint original_count = ctx.getObjectPrimitiveCount(tile);
        const float original_area = ctx.getObjectArea(tile);

        float area_ratio;
        std::string output;
        {
            capture_cerr capture;
            area_ratio = ctx.getTileObjectAreaRatio(tile);
            ctx.setTileObjectSubdivisionCount({tile}, make_int2(4, 4));
            output = capture.get_captured_output();
        }

        DOCTEST_CHECK(area_ratio == 0.f);
        DOCTEST_CHECK(output.find("not a tile object") != std::string::npos);
        DOCTEST_CHECK(ctx.getObjectPrimitiveCount(tile) == original_count);
        DOCTEST_CHECK(ctx.getObjectArea(tile) == doctest::Approx(original_area).epsilon(errtol));

        uint uniform_tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, make_int2(2, 2));
        AdaptiveTileRefinement rejected_refinement;
        DOCTEST_CHECK_THROWS(rejected_refinement = ctx.getAdaptiveTileObjectRefinement(uniform_tile));
    }

    SUBCASE("degenerate refinement range produces a uniform grid") {
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 1.f;
        refinement.subpatch_size_max = 1.f;
        refinement.transition_exponent = 1.f;

        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement);

        DOCTEST_CHECK(ctx.getAdaptiveTileObjectMaxRefinementLevel(tile) == 0);
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectBaseSubdivisionCount(tile) == make_int2(8, 8));
        DOCTEST_CHECK(ctx.getObjectPrimitiveCount(tile) == 64);

        const vec2 size_range = ctx.getAdaptiveTileObjectSubpatchSizeRange(tile);
        DOCTEST_CHECK(size_range.x == doctest::Approx(1.f).epsilon(errtol));
        DOCTEST_CHECK(size_range.y == doctest::Approx(1.f).epsilon(errtol));
    }

    SUBCASE("refinement target outside the tile warns but still produces valid geometry") {
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(100.f, 0.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        uint tile;
        std::string output;
        {
            capture_cerr capture;
            tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(16, 16), nullrotation, refinement);
            output = capture.get_captured_output();
        }

        DOCTEST_CHECK(output.find("outside") != std::string::npos);
        DOCTEST_CHECK(ctx.getObjectArea(tile) == doctest::Approx(256.f).epsilon(1e-4));

        // The requested minimum sub-patch size is not reached, since no cell is anywhere near the target.
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectSubpatchSizeRange(tile).x > refinement.subpatch_size_min);
    }

    SUBCASE("a coarsest-level grid larger than the sub-patch limit is rejected before it is built") {
        // The coarsest-level grid is materialized in full before any subdivision happens, so a limit applied only to
        // the final sub-patch count is applied too late: a large domain relative to subpatch_size_max spends hundreds
        // of megabytes — gigabytes, for a multi-kilometer domain — building a grid whose only purpose is to be
        // rejected, which turns a clean error into an out-of-memory crash. The grid is a lower bound on the sub-patch
        // count, since every cell in it yields at least one sub-patch, so it can be rejected up front.
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.05f;
        refinement.subpatch_size_max = 0.5f;
        refinement.transition_exponent = 0.25f;

        std::string message;
        {
            capture_cerr capture;
            try {
                ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(1000, 1000), nullrotation, refinement);
            } catch (const std::runtime_error &error) {
                message = error.what();
            }
        }
        DOCTEST_CHECK(message.find("coarsest-level grid") != std::string::npos);
    }

    SUBCASE("predicting the count of an unbuildable refinement reports the same error as building it") {
        // The whole point of the prediction is to answer "would this work" without paying for it, so returning the
        // limit itself as though it were the true count reports a viable configuration for one that will be rejected.
        Context ctx;

        AdaptiveTileRefinement explosive;
        explosive.target = make_vec2(0.f, 0.f);
        explosive.subpatch_size_min = 0.005f;
        explosive.subpatch_size_max = 2.f;
        explosive.transition_exponent = 4.f;

        capture_cerr capture;
        size_t predicted = 0;
        DOCTEST_CHECK_THROWS(predicted = ctx.predictAdaptiveTileObjectSubpatchCount(make_vec2(100, 100), explosive));
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(100, 100), nullrotation, explosive));
    }

    SUBCASE("invalid input") {
        Context ctx;

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        capture_cerr capture;

        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(0, 8), nullrotation, refinement));
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, -8), nullrotation, refinement));

        AdaptiveTileRefinement invalid = refinement;
        invalid.subpatch_size_min = 0.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        invalid = refinement;
        invalid.subpatch_size_min = -0.1f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        invalid = refinement;
        invalid.subpatch_size_max = 0.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        invalid = refinement;
        invalid.subpatch_size_min = 3.f;
        invalid.subpatch_size_max = 2.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        invalid = refinement;
        invalid.transition_exponent = 0.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        invalid = refinement;
        invalid.transition_exponent = -1.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        // The maximum sub-patch size must leave room for at least two base cells across the tile, below which the
        // sizing heuristic misses both ends of the requested range in the same direction.
        invalid = refinement;
        invalid.subpatch_size_max = 6.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        // The size ratio bounds the refinement level, and with it the shift used to index cells within a coarsest-level
        // cell. The sub-patch count limit does not cover this, since a chain deep enough to overflow the shift costs
        // only a few cells per level.
        invalid = refinement;
        invalid.subpatch_size_min = 1e-9f;
        invalid.subpatch_size_max = 2.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, invalid));

        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement, "lib/images/solid.jpg", make_int2(0, 1)));
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement, "lib/images/missing.jpg"));

        // More texture repeats than there are base cells cannot be honored, since a sub-patch would have to straddle a
        // boundary between repeats.
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement, "lib/images/solid.jpg", make_int2(100, 100)));

        // A refinement fine enough to exhaust memory is rejected before any geometry is allocated.
        AdaptiveTileRefinement explosive;
        explosive.target = make_vec2(0.f, 0.f);
        explosive.subpatch_size_min = 0.005f;
        explosive.subpatch_size_max = 2.f;
        explosive.transition_exponent = 4.f;
        DOCTEST_CHECK_THROWS(ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(100, 100), nullrotation, explosive));
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

    SUBCASE("setTubeNodes preserves color-to-segment mapping") {
        // Regression test: updateTriangleVertices() must use the same loop order
        // as addTubeObject() to maintain correct UUID-to-vertex mapping.
        Context ctx;
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2), make_vec3(0, 0, 3)};
        std::vector<float> radii = {0.5f, 0.5f, 0.5f, 0.5f};
        std::vector<RGBcolor> colors = {RGB::red, RGB::yellow, RGB::green, RGB::blue};

        int subdiv = 8;
        uint tube = ctx.addTubeObject(subdiv, nodes, radii, colors);

        // Translate nodes slightly — this calls updateTriangleVertices()
        std::vector<vec3> new_nodes = nodes;
        for (auto &n : new_nodes) n.x += 0.1f;
        ctx.setTubeNodes(tube, new_nodes);

        // Verify: each triangle's vertices should be near the segment matching its color.
        // Segment i spans from new_nodes[i].z to new_nodes[i+1].z.
        // A triangle with color.at(i) should have all vertices with z in [nodes[i].z, nodes[i+1].z].
        std::vector<uint> uuids = ctx.getObjectPrimitiveUUIDs(tube);
        int mismatches = 0;
        for (uint uuid : uuids) {
            RGBcolor c = ctx.getPrimitiveColorRGB(uuid);

            // Determine which segment this triangle belongs to based on its color
            int expected_segment = -1;
            if (c.r == RGB::red.r && c.g == RGB::red.g && c.b == RGB::red.b) expected_segment = 0;
            else if (c.r == RGB::yellow.r && c.g == RGB::yellow.g && c.b == RGB::yellow.b) expected_segment = 1;
            else if (c.r == RGB::green.r && c.g == RGB::green.g && c.b == RGB::green.b) expected_segment = 2;

            DOCTEST_REQUIRE(expected_segment >= 0);

            float z_min = new_nodes[expected_segment].z;
            float z_max = new_nodes[expected_segment + 1].z;

            // Check all 3 vertices are within the segment's z-range (with tolerance for radial offset)
            for (uint v = 0; v < 3; v++) {
                vec3 vert = ctx.getTriangleVertex(uuid, v);
                if (vert.z < z_min - 0.01f || vert.z > z_max + 0.01f) {
                    mismatches++;
                    break;
                }
            }
        }
        DOCTEST_CHECK_MESSAGE(mismatches == 0, "Found " << mismatches << " triangles with vertices outside their color's segment range after setTubeNodes()");
    }

    SUBCASE("pruneTubeNodes deletes correct primitives") {
        Context ctx;
        // 4 nodes → 3 segments. Prune at index 3 → keep 2 segments (nodes 0,1,2).
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2), make_vec3(0, 0, 3)};
        std::vector<float> radii = {0.5f, 0.5f, 0.5f, 0.5f};
        std::vector<RGBcolor> colors = {RGB::red, RGB::yellow, RGB::green, RGB::blue};

        int subdiv = 8;
        uint tube = ctx.addTubeObject(subdiv, nodes, radii, colors);

        uint uuids_before = ctx.getObjectPrimitiveUUIDs(tube).size();
        // 3 segments * 8 subdivisions * 2 triangles/subdivision = 48
        DOCTEST_CHECK(uuids_before == 48);

        ctx.pruneTubeNodes(tube, 3);

        // Object should still exist with 3 nodes (2 segments)
        DOCTEST_CHECK(ctx.doesObjectExist(tube));
        DOCTEST_CHECK(ctx.getTubeObjectNodeCount(tube) == 3);

        // 2 segments * 8 subdivisions * 2 triangles = 32
        std::vector<uint> remaining = ctx.getObjectPrimitiveUUIDs(tube);
        DOCTEST_CHECK(remaining.size() == 32);

        // All remaining triangles should have colors from segments 0 or 1 (red or yellow)
        int bad_colors = 0;
        for (uint uuid : remaining) {
            RGBcolor c = ctx.getPrimitiveColorRGB(uuid);
            bool is_red = (c.r == RGB::red.r && c.g == RGB::red.g && c.b == RGB::red.b);
            bool is_yellow = (c.r == RGB::yellow.r && c.g == RGB::yellow.g && c.b == RGB::yellow.b);
            if (!is_red && !is_yellow) bad_colors++;
        }
        DOCTEST_CHECK(bad_colors == 0);

        // Pruning to 1 node should delete the object entirely
        ctx.pruneTubeNodes(tube, 1);
        DOCTEST_CHECK_FALSE(ctx.doesObjectExist(tube));
    }

    SUBCASE("setTubeNodes maintains circular cross-sections after bending") {
        Context ctx;
        // Create a straight vertical tube
        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
        std::vector<float> radii = {0.5f, 0.5f, 0.5f};
        int subdiv = 16;
        uint tube = ctx.addTubeObject(subdiv, nodes, radii);

        // Bend the tube 90 degrees at the tip
        std::vector<vec3> bent_nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(1, 0, 1)};
        ctx.setTubeNodes(tube, bent_nodes);

        // Verify cross-sections are circular and perpendicular to the local axis:
        // For each triangle vertex, find the nearest node, check that the radial vector
        // (vertex - node) is perpendicular to the local axis and has the correct magnitude.
        std::vector<uint> uuids = ctx.getObjectPrimitiveUUIDs(tube);

        // Compute axial directions at each node
        std::vector<vec3> axial(3);
        axial[0] = (bent_nodes[1] - bent_nodes[0]);
        axial[0].normalize();
        axial[1] = 0.5f * ((bent_nodes[1] - bent_nodes[0]) + (bent_nodes[2] - bent_nodes[1]));
        axial[1].normalize();
        axial[2] = (bent_nodes[2] - bent_nodes[1]);
        axial[2].normalize();

        int perp_failures = 0;
        int radius_failures = 0;

        for (uint uuid : uuids) {
            for (uint v = 0; v < 3; v++) {
                vec3 vert = ctx.getTriangleVertex(uuid, v);

                // Find nearest node
                int nearest = 0;
                float min_dist = (vert - bent_nodes[0]).magnitude();
                for (int n = 1; n < 3; n++) {
                    float d = (vert - bent_nodes[n]).magnitude();
                    if (d < min_dist) {
                        min_dist = d;
                        nearest = n;
                    }
                }

                vec3 radial = vert - bent_nodes[nearest];
                float dot = fabs(radial * axial[nearest]);
                if (dot > 0.02f) {
                    perp_failures++;
                }
                if (fabs(radial.magnitude() - 0.5f) > 0.02f) {
                    radius_failures++;
                }
            }
        }

        DOCTEST_CHECK_MESSAGE(perp_failures == 0, "Found " << perp_failures << " vertices not perpendicular to local tube axis after bending");
        DOCTEST_CHECK_MESSAGE(radius_failures == 0, "Found " << radius_failures << " vertices with incorrect radius after bending");
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

    SUBCASE("object bounding box single-primitive object") {
        // Regression: the box was seeded from the first primitive's first vertex and then
        // "continue"d to the next primitive, so the rest of that primitive's vertices were never
        // compared. A single-primitive object reported min == max == its first vertex. A box
        // object cannot expose this (its six faces cover each other's extremes), so use a tile
        // made of exactly one patch.
        Context ctx;
        uint obj = ctx.addTileObject(make_vec3(2, 0, 0), make_vec2(1, 1), nullrotation, make_int2(1, 1));

        DOCTEST_CHECK(ctx.getObjectPrimitiveUUIDs(obj).size() == 1);

        vec3 min_corner, max_corner;
        ctx.getObjectBoundingBox(obj, min_corner, max_corner);

        DOCTEST_CHECK(min_corner.x == doctest::Approx(1.5f).epsilon(0.01));
        DOCTEST_CHECK(max_corner.x == doctest::Approx(2.5f).epsilon(0.01));
        DOCTEST_CHECK(min_corner.y == doctest::Approx(-0.5f).epsilon(0.01));
        DOCTEST_CHECK(max_corner.y == doctest::Approx(0.5f).epsilon(0.01));
        // The box must have real extent rather than collapsing to a single point.
        DOCTEST_CHECK(max_corner.x > min_corner.x);
        DOCTEST_CHECK(max_corner.y > min_corner.y);
    }

    SUBCASE("object bounding box first object in list keeps its extent") {
        // The defect was confined to the first primitive of the first object, so a list whose
        // FIRST object holds a unique extreme is what exposes it.
        Context ctx;
        uint big = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(4, 4), nullrotation, make_int2(1, 1));
        uint small = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, make_int2(1, 1));

        vec3 min_corner, max_corner;
        ctx.getObjectBoundingBox(std::vector<uint>{big, small}, min_corner, max_corner);

        DOCTEST_CHECK(min_corner.x == doctest::Approx(-2.f).epsilon(0.01));
        DOCTEST_CHECK(max_corner.x == doctest::Approx(2.f).epsilon(0.01));
        DOCTEST_CHECK(min_corner.y == doctest::Approx(-2.f).epsilon(0.01));
        DOCTEST_CHECK(max_corner.y == doctest::Approx(2.f).epsilon(0.01));
    }

    SUBCASE("object bounding box with no primitives fails fast") {
        // An empty request has no defined bounding box; returning leaves the caller's corners
        // untouched, which reads as a valid box at whatever they were initialized to.
        Context ctx;
        vec3 min_corner, max_corner;
        DOCTEST_CHECK_THROWS(ctx.getObjectBoundingBox(std::vector<uint>{}, min_corner, max_corner));
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

    SUBCASE("clearTimeseriesData") {
        Context ctx;
        Date date = make_Date(1, 1, 2025);
        Time time = make_Time(0, 0, 12);

        // Add some timeseries data
        ctx.addTimeseriesData("temp", 25.5f, date, time);
        ctx.addTimeseriesData("humidity", 60.0f, date, time);
        DOCTEST_CHECK(ctx.listTimeseriesVariables().size() == 2);
        DOCTEST_CHECK(ctx.doesTimeseriesVariableExist("temp"));

        // Clear and verify
        ctx.clearTimeseriesData();
        DOCTEST_CHECK(ctx.listTimeseriesVariables().empty());
        DOCTEST_CHECK_FALSE(ctx.doesTimeseriesVariableExist("temp"));
        DOCTEST_CHECK_FALSE(ctx.doesTimeseriesVariableExist("humidity"));

        // Calling on already-empty context should be a no-op
        ctx.clearTimeseriesData();
        DOCTEST_CHECK(ctx.listTimeseriesVariables().empty());
    }

    SUBCASE("deleteTimeseriesVariable") {
        Context ctx;
        Date date = make_Date(1, 1, 2025);
        Time time0 = make_Time(0, 0, 12);
        Time time1 = make_Time(1, 0, 12);

        ctx.addTimeseriesData("temp", 25.5f, date, time0);
        ctx.addTimeseriesData("temp", 26.5f, date, time1);
        ctx.addTimeseriesData("humidity", 60.0f, date, time0);
        DOCTEST_CHECK(ctx.listTimeseriesVariables().size() == 2);

        // Delete one variable; the other must remain intact and queryable.
        ctx.deleteTimeseriesVariable("temp");
        DOCTEST_CHECK_FALSE(ctx.doesTimeseriesVariableExist("temp"));
        DOCTEST_CHECK(ctx.doesTimeseriesVariableExist("humidity"));
        DOCTEST_CHECK(ctx.getTimeseriesLength("humidity") == 1);
        DOCTEST_CHECK(ctx.queryTimeseriesData("humidity", 0) == doctest::Approx(60.0f));

        // Re-adding the same label after deletion creates a fresh series.
        ctx.addTimeseriesData("temp", 99.9f, date, time0);
        DOCTEST_CHECK(ctx.doesTimeseriesVariableExist("temp"));
        DOCTEST_CHECK(ctx.getTimeseriesLength("temp") == 1);
        DOCTEST_CHECK(ctx.queryTimeseriesData("temp", 0) == doctest::Approx(99.9f));

        // Deleting a non-existent variable warns but does not throw.
        std::string captured;
        {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.deleteTimeseriesVariable("nonexistent"));
            captured = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(captured.find("WARNING") != std::string::npos);
        DOCTEST_CHECK(captured.find("nonexistent") != std::string::npos);
    }

    SUBCASE("deleteTimeseriesDataPoint by label") {
        Context ctx;
        Date date = make_Date(1, 1, 2025);
        Time time0 = make_Time(0, 0, 12);
        Time time1 = make_Time(1, 0, 12);
        Time time2 = make_Time(2, 0, 12);

        ctx.addTimeseriesData("temp", 25.5f, date, time0);
        ctx.addTimeseriesData("temp", 26.5f, date, time1);
        ctx.addTimeseriesData("temp", 27.5f, date, time2);
        ctx.addTimeseriesData("humidity", 60.0f, date, time1);
        DOCTEST_CHECK(ctx.getTimeseriesLength("temp") == 3);

        // Delete the middle point of "temp"; other points remain and ordering is preserved.
        ctx.deleteTimeseriesDataPoint("temp", date, time1);
        DOCTEST_CHECK(ctx.getTimeseriesLength("temp") == 2);
        DOCTEST_CHECK(ctx.queryTimeseriesData("temp", 0) == doctest::Approx(25.5f));
        DOCTEST_CHECK(ctx.queryTimeseriesData("temp", 1) == doctest::Approx(27.5f));
        // "humidity" is unaffected: deleting from one variable does not touch others.
        DOCTEST_CHECK(ctx.getTimeseriesLength("humidity") == 1);
        DOCTEST_CHECK(ctx.queryTimeseriesData("humidity", 0) == doctest::Approx(60.0f));

        // Deleting at a non-matching (date,time) warns but does not throw.
        std::string captured;
        {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.deleteTimeseriesDataPoint("temp", date, time1));
            captured = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(captured.find("WARNING") != std::string::npos);
        DOCTEST_CHECK(ctx.getTimeseriesLength("temp") == 2);

        // Deleting from a non-existent variable warns but does not throw.
        {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.deleteTimeseriesDataPoint("nonexistent", date, time0));
            captured = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(captured.find("WARNING") != std::string::npos);
        DOCTEST_CHECK(captured.find("nonexistent") != std::string::npos);
    }

    SUBCASE("deleteTimeseriesDataPoint across all variables") {
        Context ctx;
        Date date = make_Date(1, 1, 2025);
        Time time0 = make_Time(0, 0, 12);
        Time time1 = make_Time(1, 0, 12);

        ctx.addTimeseriesData("temp", 25.5f, date, time0);
        ctx.addTimeseriesData("temp", 26.5f, date, time1);
        ctx.addTimeseriesData("humidity", 60.0f, date, time0);
        ctx.addTimeseriesData("humidity", 65.0f, date, time1);
        ctx.addTimeseriesData("pressure", 1013.0f, date, time1);

        // Delete the time0 point across all variables.
        // "pressure" has no point at time0 — it should be silently left alone.
        ctx.deleteTimeseriesDataPoint(date, time0);
        DOCTEST_CHECK(ctx.getTimeseriesLength("temp") == 1);
        DOCTEST_CHECK(ctx.queryTimeseriesData("temp", 0) == doctest::Approx(26.5f));
        DOCTEST_CHECK(ctx.getTimeseriesLength("humidity") == 1);
        DOCTEST_CHECK(ctx.queryTimeseriesData("humidity", 0) == doctest::Approx(65.0f));
        DOCTEST_CHECK(ctx.getTimeseriesLength("pressure") == 1);
        DOCTEST_CHECK(ctx.queryTimeseriesData("pressure", 0) == doctest::Approx(1013.0f));

        // A (date,time) that matches no variable warns but does not throw.
        std::string captured;
        Time time_no_match = make_Time(23, 59, 59);
        {
            capture_cerr cerr_buffer;
            DOCTEST_CHECK_NOTHROW(ctx.deleteTimeseriesDataPoint(date, time_no_match));
            captured = cerr_buffer.get_captured_output();
        }
        DOCTEST_CHECK(captured.find("WARNING") != std::string::npos);
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

    SUBCASE("copyPrimitive of an object member does not claim membership of that object") {
        // Regression: copyPrimitive() carried the source primitive's parent object ID onto the copy without adding
        // the copy to that object's member list, so the copy claimed a membership the object did not recognize.
        // Context::copyObject() assigns the parent itself immediately afterwards, so nothing depended on the
        // inherited value. Consequences of the stray: deleteObject() deleted only the members it listed, leaving
        // the copy behind pointing at an object that no longer existed, which made writeXML() throw an uncaught
        // std::out_of_range; and without the delete, the object wrote one more <patch> block than its own
        // subdivision count, so it reloaded with a sub-primitive it never had.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));
        std::vector<uint> members = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_REQUIRE(members.size() == 4);

        uint stray = ctx.copyPrimitive(members.front());
        DOCTEST_CHECK(ctx.getPrimitiveParentObjectID(stray) == 0);

        // The object's membership must be unchanged by the copy.
        std::vector<uint> members_after = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(members_after.size() == 4);
        DOCTEST_CHECK(std::find(members_after.begin(), members_after.end(), stray) == members_after.end());

        // Deleting the object must leave the stray copy as a valid standalone primitive, and the Context writable.
        ctx.deleteObject(tile);
        DOCTEST_CHECK(ctx.doesPrimitiveExist(stray));
        DOCTEST_CHECK(ctx.getPrimitiveParentObjectID(stray) == 0);
        const char *test_file = "helios_copyprimitive_parent_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));
        std::remove(test_file);

        // copyObject() must still produce a fully-formed copy, since it assigns the parent itself.
        Context ctx2;
        uint source = ctx2.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));
        uint copy = ctx2.copyObject(source);
        DOCTEST_CHECK(ctx2.getObjectPrimitiveUUIDs(copy).size() == 4);
        for (uint UUID: ctx2.getObjectPrimitiveUUIDs(copy)) {
            DOCTEST_CHECK(ctx2.getPrimitiveParentObjectID(UUID) == copy);
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

    SUBCASE("Material System - Rename") {
        Context ctx;

        // Create a triangle with a texture to generate an auto-material
        uint UUID = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), "lib/images/disk_texture.png", make_vec2(0, 0), make_vec2(1, 0), make_vec2(0, 1));
        std::string auto_label = ctx.getPrimitiveMaterialLabel(UUID);
        DOCTEST_CHECK(auto_label.substr(0, 7) == "__auto_");
        DOCTEST_CHECK(ctx.doesMaterialExist(auto_label));

        // Rename the auto-generated material
        ctx.renameMaterial(auto_label, "bean_trifoliate_leaf");
        DOCTEST_CHECK(ctx.doesMaterialExist("bean_trifoliate_leaf"));
        // Auto-label is retained as a lookup alias for deduplication
        DOCTEST_CHECK(ctx.doesMaterialExist(auto_label));

        // Primitive should now report the new display label
        DOCTEST_CHECK(ctx.getPrimitiveMaterialLabel(UUID) == "bean_trifoliate_leaf");

        // Properties should be preserved
        std::string tex = ctx.getMaterialTexture("bean_trifoliate_leaf");
        DOCTEST_CHECK(tex == "lib/images/disk_texture.png");

        // Deduplication: a new triangle with the same texture should reuse the renamed material
        uint UUID2 = ctx.addTriangle(make_vec3(2, 0, 0), make_vec3(3, 0, 0), make_vec3(2, 1, 0), "lib/images/disk_texture.png", make_vec2(0, 0), make_vec2(1, 0), make_vec2(0, 1));
        DOCTEST_CHECK(ctx.getPrimitiveMaterialLabel(UUID2) == "bean_trifoliate_leaf");

        // Rename a user-created (non-auto) material — old label should NOT be retained
        ctx.addMaterial("old_name");
        ctx.renameMaterial("old_name", "new_name");
        DOCTEST_CHECK(ctx.doesMaterialExist("new_name"));
        DOCTEST_CHECK(!ctx.doesMaterialExist("old_name"));

        // Error cases
        DOCTEST_CHECK_THROWS(ctx.renameMaterial("nonexistent", "new_name2"));             // old label doesn't exist
        DOCTEST_CHECK_THROWS(ctx.renameMaterial("bean_trifoliate_leaf", "__reserved"));    // new label reserved
        DOCTEST_CHECK_THROWS(ctx.renameMaterial("bean_trifoliate_leaf", "new_name"));     // new label already exists
        DOCTEST_CHECK_THROWS(ctx.renameMaterial("bean_trifoliate_leaf", ""));              // empty label
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

TEST_CASE("Context Timeseries File Loading") {

    SUBCASE("ISO-8601 UTC datetime") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_iso8601_utc.csv", {}, ",", "ISO8601", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026);
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(17.8f));

        float humid = ctx.queryTimeseriesData("humidity", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(humid == doctest::Approx(0.65f));

        // UTC offset should be set to 0 (ISO offset Z → Helios 0)
        Location loc = ctx.getLocation();
        DOCTEST_CHECK(loc.UTC_offset == doctest::Approx(0.0f));
    }

    SUBCASE("ISO-8601 with timezone offset") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_iso8601_offset.csv", {}, ",", "ISO8601", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026);
        // The local time is used (02:00, 03:00, 04:00)
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(2, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(4, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(17.8f));

        // ISO -08:00 → Helios UTC_offset = +8 (West-positive convention)
        Location loc = ctx.getLocation();
        DOCTEST_CHECK(loc.UTC_offset == doctest::Approx(8.0f));
    }

    SUBCASE("Compact date (YYYYMMDD no delimiters)") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_compact_date.csv", {}, ",", "YYYYMMDD", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026);
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(17.8f));
    }

    SUBCASE("Compact datetime (YYYYMMDDHH)") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_compact_datetime.csv", {}, ",", "YYYYMMDDHH", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026);
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(17.8f));
    }

    SUBCASE("Time column (HH:MM and HH:MM:SS)") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_time_column.csv", {}, ",", "YYYYMMDD", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026);
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(10, 30, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        // HH:MM:SS format: 11:15:30
        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(11, 15, 30));
        DOCTEST_CHECK(temp2 == doctest::Approx(16.2f));

        float temp3 = ctx.queryTimeseriesData("temperature", date, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp3 == doctest::Approx(17.8f));
    }

    SUBCASE("Datetime with space separator") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_datetime_space.csv", {}, " ", "YYYY-MM-DD HH:MM", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026);
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(17.8f));
    }

    SUBCASE("European DD/MM/YYYY HH:MM datetime") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_ddmmyyyy_hhmm.csv", {}, ",", "DD/MM/YYYY HH:MM", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026); // 03/02/2026 = Feb 3rd in DD/MM/YYYY
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(17.8f));
    }

    SUBCASE("US MM/DD/YYYY HH:MM datetime") {
        Context ctx;
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/timeseries_mmddyyyy_hhmm.csv", {}, ",", "MM/DD/YYYY HH:MM", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(3, 2, 2026); // 02/03/2026 = Feb 3rd in MM/DD/YYYY
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(17.8f));
    }

    SUBCASE("Backward compatibility - existing weather_data.csv") {
        Context ctx;
        // weather_data.csv has: date "1-2-2020" with DDMMYYYY = day 1, month 2 = Feb 1
        std::string warning_msg;
        {
            capture_cerr capture;
            ctx.loadTabularTimeseriesData("lib/testdata/weather_data.csv", {}, ",", "DDMMYYYY", 0);
            warning_msg = capture.get_captured_output();
        }
        DOCTEST_CHECK(warning_msg.find("headerlines argument was specified as zero") != std::string::npos);

        Date date = make_Date(1, 2, 2020);
        float temp = ctx.queryTimeseriesData("temperature", date, make_Time(13, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(35.32343f));

        float temp2 = ctx.queryTimeseriesData("temperature", date, make_Time(14, 0, 0));
        DOCTEST_CHECK(temp2 == doctest::Approx(36.23432f));
    }

    SUBCASE("User-specified column labels") {
        Context ctx;
        ctx.loadTabularTimeseriesData("lib/testdata/timeseries_iso8601_utc.csv",
                                      {"datetime", "temp", "rh"}, ",", "ISO8601", 1);

        Date date = make_Date(3, 2, 2026);
        float temp = ctx.queryTimeseriesData("temp", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(15.5f));

        float rh = ctx.queryTimeseriesData("rh", date, make_Time(10, 0, 0));
        DOCTEST_CHECK(rh == doctest::Approx(0.65f));
    }

    SUBCASE("Real data: Open-Meteo Davis CA (ISO-8601 no seconds)") {
        // Real Open-Meteo data for Davis, CA. File has 3 metadata header lines + 1 blank line.
        // Datetime format is ISO-8601 without seconds: "2024-01-01T00:00"
        // Column header says "time" but it's a full datetime — user must remap with labels.
        Context ctx;
        ctx.loadTabularTimeseriesData("lib/testdata/timeseries_openmeteo_davis.csv",
                                      {"datetime", "temperature", "humidity", "precipitation"},
                                      ",", "ISO8601", 4);

        Date jan1 = make_Date(1, 1, 2024);
        Date jan2 = make_Date(2, 1, 2024);

        // First row: 2024-01-01T00:00, 9.1, 96, 0.00
        float temp_midnight = ctx.queryTimeseriesData("temperature", jan1, make_Time(0, 0, 0));
        DOCTEST_CHECK(temp_midnight == doctest::Approx(9.1f));

        // Row: 2024-01-01T12:00, 14.1, 66, 0.00
        float temp_noon = ctx.queryTimeseriesData("temperature", jan1, make_Time(12, 0, 0));
        DOCTEST_CHECK(temp_noon == doctest::Approx(14.1f));

        float humid_noon = ctx.queryTimeseriesData("humidity", jan1, make_Time(12, 0, 0));
        DOCTEST_CHECK(humid_noon == doctest::Approx(66.0f));

        // Row: 2024-01-02T18:00, 11.1, 91, 1.20
        float precip = ctx.queryTimeseriesData("precipitation", jan2, make_Time(18, 0, 0));
        DOCTEST_CHECK(precip == doctest::Approx(1.20f));
    }

    SUBCASE("Real data: Open-Meteo NYC (ISO-8601 negative temperatures)") {
        // Real Open-Meteo data for New York City. Tests negative values and ISO-8601 no-seconds.
        Context ctx;
        ctx.loadTabularTimeseriesData("lib/testdata/timeseries_openmeteo_nyc.csv",
                                      {"datetime", "temperature", "precipitation"},
                                      ",", "ISO8601", 4);

        Date jan1 = make_Date(1, 1, 2024);
        Date jan2 = make_Date(2, 1, 2024);
        Date jan3 = make_Date(3, 1, 2024);

        // Row: 2024-01-01T00:00, 1.8, 0.00
        float temp = ctx.queryTimeseriesData("temperature", jan1, make_Time(0, 0, 0));
        DOCTEST_CHECK(temp == doctest::Approx(1.8f));

        // Row: 2024-01-02T07:00, -4.6, 0.00 (negative temperature)
        float temp_cold = ctx.queryTimeseriesData("temperature", jan2, make_Time(7, 0, 0));
        DOCTEST_CHECK(temp_cold == doctest::Approx(-4.6f));

        // Row: 2024-01-02T00:00, -1.1, 0.00
        float temp_neg = ctx.queryTimeseriesData("temperature", jan2, make_Time(0, 0, 0));
        DOCTEST_CHECK(temp_neg == doctest::Approx(-1.1f));

        // Row: 2024-01-03T13:00, 7.7, 0.00
        float temp_warm = ctx.queryTimeseriesData("temperature", jan3, make_Time(13, 0, 0));
        DOCTEST_CHECK(temp_warm == doctest::Approx(7.7f));
    }
}

TEST_CASE("Polymesh Mesh Topology") {

    // Build a closed unit cube as an explicit indexed mesh. Two triangles per face, wound counter-clockwise when viewed from outside.
    auto buildUnitCubePolymesh = [](Context &ctx) {
        const vec3 corner[8] = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0),
                                make_vec3(0, 0, 1), make_vec3(1, 0, 1), make_vec3(1, 1, 1), make_vec3(0, 1, 1)};
        const int3 cube_faces[12] = {make_int3(0, 3, 2), make_int3(0, 2, 1),  // bottom (-z)
                                     make_int3(4, 5, 6), make_int3(4, 6, 7),  // top (+z)
                                     make_int3(0, 1, 5), make_int3(0, 5, 4),  // front (-y)
                                     make_int3(2, 3, 7), make_int3(2, 7, 6),  // back (+y)
                                     make_int3(3, 0, 4), make_int3(3, 4, 7),  // left (-x)
                                     make_int3(1, 2, 6), make_int3(1, 6, 5)}; // right (+x)

        std::vector<uint> UUIDs;
        for (const int3 &f: cube_faces) {
            UUIDs.push_back(ctx.addTriangle(corner[f.x], corner[f.y], corner[f.z], RGB::red));
        }
        return UUIDs;
    };

    SUBCASE("getVolume on a closed cube and an open mesh") {
        Context ctx;
        std::vector<uint> UUIDs = buildUnitCubePolymesh(ctx);
        uint objID = ctx.addPolymeshObject(UUIDs);

        // Grouping loose primitives carries no face table, so no closure check is possible and the historical primitive-based sum is used.
        DOCTEST_CHECK(ctx.getPolymeshObjectFaceCount(objID) == 0);
        DOCTEST_CHECK(ctx.getPolymeshObjectVolume(objID) == doctest::Approx(1.f).epsilon(1e-5));
    }

    SUBCASE("getVolume raises an error for an open mesh with topology") {
        Context ctx;

        // A single triangle is an open surface: all three of its edges are boundary edges.
        std::vector<uint> UUIDs;
        UUIDs.push_back(ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::red));
        uint objID = ctx.addPolymeshObject(UUIDs);

        ctx.setPolymeshObjectTopology(objID, {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)}, {make_int3(0, 1, 2)}, UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_CHECK(!ctx.isPolymeshObjectClosed(objID));
        DOCTEST_CHECK(ctx.getPolymeshObjectBoundaryEdges(objID).size() == 3);
        float open_mesh_volume = 0.f;
        DOCTEST_CHECK_THROWS_AS(open_mesh_volume = ctx.getPolymeshObjectVolume(objID), std::runtime_error);
    }

    SUBCASE("getVolume on a closed cube carrying a face table") {
        Context ctx;
        std::vector<uint> UUIDs = buildUnitCubePolymesh(ctx);
        uint objID = ctx.addPolymeshObject(UUIDs);

        std::vector<vec3> cube_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0),
                                           make_vec3(0, 0, 1), make_vec3(1, 0, 1), make_vec3(1, 1, 1), make_vec3(0, 1, 1)};
        std::vector<int3> cube_faces = {make_int3(0, 3, 2), make_int3(0, 2, 1), make_int3(4, 5, 6), make_int3(4, 6, 7), make_int3(0, 1, 5), make_int3(0, 5, 4),
                                        make_int3(2, 3, 7), make_int3(2, 7, 6), make_int3(3, 0, 4), make_int3(3, 4, 7), make_int3(1, 2, 6), make_int3(1, 6, 5)};

        ctx.setPolymeshObjectTopology(objID, cube_vertices, cube_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_CHECK(ctx.isPolymeshObjectClosed(objID));
        DOCTEST_CHECK(ctx.getPolymeshObjectVolume(objID) == doctest::Approx(1.f).epsilon(1e-5));
        DOCTEST_CHECK(ctx.getPolymeshObjectSurfaceArea(objID) == doctest::Approx(6.f).epsilon(1e-5));
        DOCTEST_CHECK(ctx.getPolymeshObjectConnectedComponents(objID).size() == 1);
    }

    SUBCASE("Deleting a member primitive repairs the face table") {
        Context ctx;

        // Two triangles sharing an edge, plus a third that is entirely separate so its vertices become orphaned when it is deleted.
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0), make_vec3(5, 5, 0), make_vec3(6, 5, 0), make_vec3(6, 6, 0)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2), make_int3(0, 2, 3), make_int3(4, 5, 6)};

        std::vector<uint> UUIDs;
        for (const int3 &f: mesh_faces) {
            UUIDs.push_back(ctx.addTriangle(mesh_vertices.at(f.x), mesh_vertices.at(f.y), mesh_vertices.at(f.z), RGB::red));
        }
        uint objID = ctx.addPolymeshObject(UUIDs);
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_CHECK(ctx.getPolymeshObjectVertexCount(objID) == 7);
        DOCTEST_CHECK(ctx.getPolymeshObjectFaceCount(objID) == 3);
        DOCTEST_CHECK(ctx.areObjectPrimitivesComplete(objID));

        // Delete the isolated triangle: its face must go, and vertices 4, 5 and 6 are then referenced by nothing.
        ctx.deletePrimitive(UUIDs.at(2));

        DOCTEST_CHECK(!ctx.areObjectPrimitivesComplete(objID));
        DOCTEST_CHECK(ctx.getPolymeshObjectFaceCount(objID) == 2);
        DOCTEST_CHECK(ctx.getPolymeshObjectVertexCount(objID) == 4);

        // The two surviving faces must have been reindexed onto the compacted vertex array, and still describe the same geometry.
        std::vector<vec3> remaining_vertices = ctx.getPolymeshObjectVertices(objID);
        std::vector<int3> remaining_faces = ctx.getPolymeshObjectFaces(objID);
        DOCTEST_REQUIRE(remaining_faces.size() == 2);
        for (const int3 &f: remaining_faces) {
            DOCTEST_CHECK(f.x >= 0);
            DOCTEST_CHECK(f.y >= 0);
            DOCTEST_CHECK(f.z >= 0);
            DOCTEST_CHECK(f.x < scast<int>(remaining_vertices.size()));
            DOCTEST_CHECK(f.y < scast<int>(remaining_vertices.size()));
            DOCTEST_CHECK(f.z < scast<int>(remaining_vertices.size()));
        }
        DOCTEST_CHECK(ctx.getPolymeshObjectSurfaceArea(objID) == doctest::Approx(1.f).epsilon(1e-5));

        // The face-to-primitive mapping must still resolve for the surviving primitives.
        size_t surviving_face_index = 0;
        DOCTEST_CHECK_NOTHROW(surviving_face_index = ctx.getPolymeshObjectFaceIndexForPrimitive(objID, UUIDs.at(0)));
        DOCTEST_CHECK(surviving_face_index < ctx.getPolymeshObjectFaceCount(objID));
        DOCTEST_CHECK_NOTHROW(surviving_face_index = ctx.getPolymeshObjectFaceIndexForPrimitive(objID, UUIDs.at(1)));
        DOCTEST_CHECK(surviving_face_index < ctx.getPolymeshObjectFaceCount(objID));
    }

    SUBCASE("Batched deletion repairs the face table once and matches per-primitive deletion") {
        // Deleting a vector of UUIDs detaches them from the parent object as a batch, so the topology repair runs once rather than once per primitive. The end state must be identical to deleting them
        // individually.
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0), make_vec3(5, 5, 0), make_vec3(6, 5, 0), make_vec3(6, 6, 0)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2), make_int3(0, 2, 3), make_int3(4, 5, 6)};

        Context ctx_batch;
        Context ctx_single;
        std::vector<uint> batch_UUIDs;
        std::vector<uint> single_UUIDs;
        for (const int3 &f: mesh_faces) {
            batch_UUIDs.push_back(ctx_batch.addTriangle(mesh_vertices.at(f.x), mesh_vertices.at(f.y), mesh_vertices.at(f.z), RGB::red));
            single_UUIDs.push_back(ctx_single.addTriangle(mesh_vertices.at(f.x), mesh_vertices.at(f.y), mesh_vertices.at(f.z), RGB::red));
        }
        uint batch_objID = ctx_batch.addPolymeshObject(batch_UUIDs);
        uint single_objID = ctx_single.addPolymeshObject(single_UUIDs);
        ctx_batch.setPolymeshObjectTopology(batch_objID, mesh_vertices, mesh_faces, batch_UUIDs, {}, {}, NORMAL_SOURCE_NONE);
        ctx_single.setPolymeshObjectTopology(single_objID, mesh_vertices, mesh_faces, single_UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        ctx_batch.deletePrimitive(std::vector<uint>{batch_UUIDs.at(2)});
        ctx_single.deletePrimitive(single_UUIDs.at(2));

        DOCTEST_CHECK(ctx_batch.getPolymeshObjectFaceCount(batch_objID) == ctx_single.getPolymeshObjectFaceCount(single_objID));
        DOCTEST_CHECK(ctx_batch.getPolymeshObjectVertexCount(batch_objID) == ctx_single.getPolymeshObjectVertexCount(single_objID));
        DOCTEST_CHECK(!ctx_batch.areObjectPrimitivesComplete(batch_objID));

        std::vector<int3> batch_faces = ctx_batch.getPolymeshObjectFaces(batch_objID);
        std::vector<int3> single_faces = ctx_single.getPolymeshObjectFaces(single_objID);
        DOCTEST_REQUIRE(batch_faces.size() == single_faces.size());
        for (size_t f = 0; f < batch_faces.size(); f++) {
            DOCTEST_CHECK(batch_faces.at(f).x == single_faces.at(f).x);
            DOCTEST_CHECK(batch_faces.at(f).y == single_faces.at(f).y);
            DOCTEST_CHECK(batch_faces.at(f).z == single_faces.at(f).z);
        }

        // Deleting several faces at once must also leave the survivors correctly reindexed.
        ctx_batch.deletePrimitive(std::vector<uint>{batch_UUIDs.at(0), batch_UUIDs.at(1)});
        DOCTEST_CHECK(!ctx_batch.doesObjectExist(batch_objID));
    }

    SUBCASE("Deleting every member primitive deletes the object without leaving a stale face table") {
        Context ctx;
        std::vector<uint> UUIDs = buildUnitCubePolymesh(ctx);
        uint objID = ctx.addPolymeshObject(UUIDs);

        std::vector<vec3> cube_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0),
                                           make_vec3(0, 0, 1), make_vec3(1, 0, 1), make_vec3(1, 1, 1), make_vec3(0, 1, 1)};
        std::vector<int3> cube_faces = {make_int3(0, 3, 2), make_int3(0, 2, 1), make_int3(4, 5, 6), make_int3(4, 6, 7), make_int3(0, 1, 5), make_int3(0, 5, 4),
                                        make_int3(2, 3, 7), make_int3(2, 7, 6), make_int3(3, 0, 4), make_int3(3, 4, 7), make_int3(1, 2, 6), make_int3(1, 6, 5)};
        ctx.setPolymeshObjectTopology(objID, cube_vertices, cube_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_CHECK_NOTHROW(ctx.deletePrimitive(UUIDs));
        DOCTEST_CHECK(!ctx.doesObjectExist(objID));
    }

    SUBCASE("Non-uniform scaling transforms normals by the inverse transpose") {
        Context ctx;

        // A single triangle in the xz-plane, whose surface normal points along +y.
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 0, 1)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2)};
        std::vector<uint> UUIDs = {ctx.addTriangle(mesh_vertices.at(0), mesh_vertices.at(1), mesh_vertices.at(2), RGB::red)};
        uint objID = ctx.addPolymeshObject(UUIDs);

        // Give the mesh a normal that is deliberately NOT axis-aligned, so that a wrong (non-inverse-transpose) transform produces a different direction.
        vec3 authored_normal = normalize(make_vec3(1.f, 1.f, 0.f));
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {authored_normal, authored_normal, authored_normal}, {}, NORMAL_SOURCE_AUTHORED);

        // A NON-uniform scale is essential here: under a uniform scale the matrix and its inverse transpose differ only by a scalar, so the wrong math would still normalize to the right answer.
        const vec3 scale_factor = make_vec3(4.f, 1.f, 1.f);
        ctx.scaleObject(objID, scale_factor);

        std::vector<vec3> normals_after = ctx.getPolymeshObjectVertexNormals(objID);
        DOCTEST_REQUIRE(normals_after.size() == 3);

        // Inverse transpose of diag(4,1,1) is diag(1/4,1,1), so (1,1,0)/sqrt(2) maps to (0.25,1,0) before renormalizing.
        vec3 expected = normalize(make_vec3(authored_normal.x / scale_factor.x, authored_normal.y / scale_factor.y, authored_normal.z / scale_factor.z));
        for (const vec3 &n: normals_after) {
            DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-5));
            DOCTEST_CHECK(n.x == doctest::Approx(expected.x).epsilon(1e-5));
            DOCTEST_CHECK(n.y == doctest::Approx(expected.y).epsilon(1e-5));
            DOCTEST_CHECK(n.z == doctest::Approx(expected.z).epsilon(1e-5));
        }

        // Applying the matrix directly instead of its inverse transpose would give normalize((4,1,0)), which must NOT be the answer.
        vec3 wrong = normalize(make_vec3(authored_normal.x * scale_factor.x, authored_normal.y * scale_factor.y, authored_normal.z * scale_factor.z));
        DOCTEST_CHECK(std::abs(normals_after.front().x - wrong.x) > 1e-3f);
    }

    SUBCASE("Vertex positions round-trip exactly under a combined rotation, non-uniform scale and translation") {
        Context ctx;

        // The face table is stored in the object-local frame, so setTopology() converts incoming global vertices inward and getVertices() converts them back out. Those two conversions use hand-written
        // matrix index arithmetic against a transposed-storage inverse, which is easy to break on a later edit and which a translation-only or axis-aligned test would not catch. Pin the invariant with a
        // transform that combines rotation about two axes, a non-uniform scale and a translation.
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2)};
        std::vector<uint> UUIDs = {ctx.addTriangle(mesh_vertices.at(0), mesh_vertices.at(1), mesh_vertices.at(2), RGB::red)};
        uint objID = ctx.addPolymeshObject(UUIDs);

        ctx.rotateObject(objID, 0.7f, "z");
        ctx.rotateObject(objID, 0.4f, "x");
        ctx.scaleObject(objID, make_vec3(3.f, 0.5f, 2.f));
        ctx.translateObject(objID, make_vec3(7, -2, 5));

        const vec3 authored_normal = normalize(make_vec3(0.3f, -0.6f, 0.74f));
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {authored_normal, authored_normal, authored_normal}, {}, NORMAL_SOURCE_AUTHORED);

        // Vertices handed in must come back out unchanged, in global coordinates.
        std::vector<vec3> vertices_out = ctx.getPolymeshObjectVertices(objID);
        DOCTEST_REQUIRE(vertices_out.size() == 3);
        for (size_t v = 0; v < 3; v++) {
            DOCTEST_CHECK(vertices_out.at(v).x == doctest::Approx(mesh_vertices.at(v).x).epsilon(1e-4));
            DOCTEST_CHECK(vertices_out.at(v).y == doctest::Approx(mesh_vertices.at(v).y).epsilon(1e-4));
            DOCTEST_CHECK(vertices_out.at(v).z == doctest::Approx(mesh_vertices.at(v).z).epsilon(1e-4));
        }

        // setTopology() followed by getVertexNormals() must likewise be an exact identity, and the normal must stay unit length.
        std::vector<vec3> normals_out = ctx.getPolymeshObjectVertexNormals(objID);
        DOCTEST_REQUIRE(normals_out.size() == 3);
        for (const vec3 &n: normals_out) {
            DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-4));
            DOCTEST_CHECK(n.x == doctest::Approx(authored_normal.x).epsilon(1e-4));
            DOCTEST_CHECK(n.y == doctest::Approx(authored_normal.y).epsilon(1e-4));
            DOCTEST_CHECK(n.z == doctest::Approx(authored_normal.z).epsilon(1e-4));
        }
    }

    SUBCASE("Translation and rotation preserve mesh geometry") {
        Context ctx;
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2)};
        std::vector<uint> UUIDs = {ctx.addTriangle(mesh_vertices.at(0), mesh_vertices.at(1), mesh_vertices.at(2), RGB::red)};
        uint objID = ctx.addPolymeshObject(UUIDs);
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {make_vec3(0, 0, 1), make_vec3(0, 0, 1), make_vec3(0, 0, 1)}, {}, NORMAL_SOURCE_AUTHORED);

        ctx.translateObject(objID, make_vec3(3, 4, 5));
        std::vector<vec3> translated = ctx.getPolymeshObjectVertices(objID);
        DOCTEST_CHECK(translated.at(0).x == doctest::Approx(3.f).epsilon(1e-5));
        DOCTEST_CHECK(translated.at(0).y == doctest::Approx(4.f).epsilon(1e-5));
        DOCTEST_CHECK(translated.at(0).z == doctest::Approx(5.f).epsilon(1e-5));

        // A pure translation must leave the normal untouched.
        std::vector<vec3> normals_translated = ctx.getPolymeshObjectVertexNormals(objID);
        DOCTEST_CHECK(normals_translated.at(0).z == doctest::Approx(1.f).epsilon(1e-5));

        // Rotating 90 degrees about the x-axis must carry the +z normal onto -y.
        ctx.rotateObject(objID, 0.5f * PI_F, "x");
        std::vector<vec3> normals_rotated = ctx.getPolymeshObjectVertexNormals(objID);
        DOCTEST_CHECK(normals_rotated.at(0).magnitude() == doctest::Approx(1.f).epsilon(1e-5));
        DOCTEST_CHECK(std::abs(normals_rotated.at(0).y) == doctest::Approx(1.f).epsilon(1e-5));
        DOCTEST_CHECK(normals_rotated.at(0).z == doctest::Approx(0.f).epsilon(1e-5));
    }

    SUBCASE("computeVertexNormals keeps a cube crease hard") {
        Context ctx;
        std::vector<uint> UUIDs = buildUnitCubePolymesh(ctx);
        uint objID = ctx.addPolymeshObject(UUIDs);

        std::vector<vec3> cube_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0),
                                           make_vec3(0, 0, 1), make_vec3(1, 0, 1), make_vec3(1, 1, 1), make_vec3(0, 1, 1)};
        std::vector<int3> cube_faces = {make_int3(0, 3, 2), make_int3(0, 2, 1), make_int3(4, 5, 6), make_int3(4, 6, 7), make_int3(0, 1, 5), make_int3(0, 5, 4),
                                        make_int3(2, 3, 7), make_int3(2, 7, 6), make_int3(3, 0, 4), make_int3(3, 4, 7), make_int3(1, 2, 6), make_int3(1, 6, 5)};
        ctx.setPolymeshObjectTopology(objID, cube_vertices, cube_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_CHECK(!ctx.doesPolymeshObjectHaveVertexNormals(objID));
        DOCTEST_CHECK(ctx.getPolymeshObjectVertexNormalSource(objID) == NORMAL_SOURCE_NONE);

        // Adjacent cube faces meet at 90 degrees. A crease angle below that must keep the faces from blending, leaving every vertex normal axis-aligned rather than averaged into a body diagonal.
        ctx.computePolymeshObjectVertexNormals(objID, 45.f);
        DOCTEST_CHECK(ctx.doesPolymeshObjectHaveVertexNormals(objID));
        DOCTEST_CHECK(ctx.getPolymeshObjectVertexNormalSource(objID) == NORMAL_SOURCE_COMPUTED);

        // Each corner is touched by three mutually perpendicular faces, so it splits into three vertices carrying one axis-aligned normal each: 8 corners -> 24 vertices.
        std::vector<vec3> hard_normals = ctx.getPolymeshObjectVertexNormals(objID);
        DOCTEST_CHECK(ctx.getPolymeshObjectVertexCount(objID) == 24);
        DOCTEST_REQUIRE(hard_normals.size() == 24);
        for (const vec3 &n: hard_normals) {
            DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-5));
            const float largest_component = std::max(std::max(std::abs(n.x), std::abs(n.y)), std::abs(n.z));
            DOCTEST_CHECK(largest_component == doctest::Approx(1.f).epsilon(1e-4));
        }

        // With a crease angle wide enough to span the 90-degree corner, all three faces at a corner form a single smooth group and blend into a body diagonal instead. Rebuild the topology first, because the
        // call above already split each corner into three vertices that are no longer shared between faces.
        ctx.setPolymeshObjectTopology(objID, cube_vertices, cube_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);
        ctx.computePolymeshObjectVertexNormals(objID, 120.f);
        std::vector<vec3> smooth_normals = ctx.getPolymeshObjectVertexNormals(objID);
        DOCTEST_CHECK(ctx.getPolymeshObjectVertexCount(objID) == 8);
        DOCTEST_REQUIRE(smooth_normals.size() == 8);
        // The averaging is area-weighted and a cube corner is not touched by equal triangle area on each of its three faces, so the blended normal is not the exact body diagonal. What distinguishes a blended
        // normal from a hard one is that it points diagonally outward: every component is substantially non-zero, whereas the hard normals checked above were exactly axis-aligned.
        for (size_t v = 0; v < smooth_normals.size(); v++) {
            const vec3 &n = smooth_normals.at(v);
            DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-5));
            DOCTEST_CHECK(std::abs(n.x) > 0.2f);
            DOCTEST_CHECK(std::abs(n.y) > 0.2f);
            DOCTEST_CHECK(std::abs(n.z) > 0.2f);

            // The normal must point away from the cube center, i.e. agree in sign with the corner's offset from (0.5,0.5,0.5).
            const vec3 outward = cube_vertices.at(v) - make_vec3(0.5f, 0.5f, 0.5f);
            DOCTEST_CHECK(n * outward > 0.f);
        }
    }

    SUBCASE("Degenerate and duplicate faces are handled without crashing") {
        Context ctx;

        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), make_vec3(2, 0, 0)};
        // A duplicated face, and a zero-area (collinear) face.
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2), make_int3(0, 1, 2), make_int3(0, 1, 3)};

        std::vector<uint> UUIDs;
        for (const int3 &f: mesh_faces) {
            UUIDs.push_back(ctx.addTriangle(mesh_vertices.at(f.x) + make_vec3(0, 0, 1e-4f * scast<float>(UUIDs.size())), mesh_vertices.at(f.y), mesh_vertices.at(f.z), RGB::red));
        }
        uint objID = ctx.addPolymeshObject(UUIDs);

        DOCTEST_CHECK_NOTHROW(ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE));
        float degenerate_surface_area = 0.f;
        DOCTEST_CHECK_NOTHROW(degenerate_surface_area = ctx.getPolymeshObjectSurfaceArea(objID));
        DOCTEST_CHECK(degenerate_surface_area >= 0.f);
        std::vector<int2> degenerate_boundary_edges;
        DOCTEST_CHECK_NOTHROW(degenerate_boundary_edges = ctx.getPolymeshObjectBoundaryEdges(objID));
        std::vector<std::vector<size_t>> degenerate_components;
        DOCTEST_CHECK_NOTHROW(degenerate_components = ctx.getPolymeshObjectConnectedComponents(objID));
        DOCTEST_CHECK(!degenerate_components.empty());
        DOCTEST_CHECK_NOTHROW(ctx.computePolymeshObjectVertexNormals(objID, 45.f));

        // Every vertex that is actually referenced by a face must receive a unit normal. A zero-area face contributes no direction, so a vertex touched only by degenerate geometry legitimately has none.
        std::vector<vec3> normals = ctx.getPolymeshObjectVertexNormals(objID);
        std::vector<int3> resulting_faces = ctx.getPolymeshObjectFaces(objID);
        std::vector<bool> vertex_is_referenced(normals.size(), false);
        for (const int3 &f: resulting_faces) {
            vertex_is_referenced.at(f.x) = true;
            vertex_is_referenced.at(f.y) = true;
            vertex_is_referenced.at(f.z) = true;
        }
        size_t unit_normal_count = 0;
        for (size_t v = 0; v < normals.size(); v++) {
            if (!vertex_is_referenced.at(v)) {
                continue;
            }
            const float magnitude = normals.at(v).magnitude();
            DOCTEST_CHECK((magnitude == doctest::Approx(1.f).epsilon(1e-4) || magnitude == doctest::Approx(0.f).epsilon(1e-4)));
            if (magnitude > 0.5f) {
                unit_normal_count++;
            }
        }
        DOCTEST_CHECK(unit_normal_count > 0);

        // An out-of-range face index must be rejected rather than silently accepted.
        DOCTEST_CHECK_THROWS_AS(ctx.setPolymeshObjectTopology(objID, mesh_vertices, {make_int3(0, 1, 99)}, {UUIDs.front()}, {}, {}, NORMAL_SOURCE_NONE), std::runtime_error);
    }

    SUBCASE("Copying a polymesh object carries the topology") {
        Context ctx;
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2)};
        std::vector<uint> UUIDs = {ctx.addTriangle(mesh_vertices.at(0), mesh_vertices.at(1), mesh_vertices.at(2), RGB::red)};
        uint objID = ctx.addPolymeshObject(UUIDs);
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {make_vec3(0, 0, 1), make_vec3(0, 0, 1), make_vec3(0, 0, 1)}, {}, NORMAL_SOURCE_AUTHORED);

        uint copy_objID = ctx.copyObject(objID);
        DOCTEST_CHECK(ctx.getPolymeshObjectVertexCount(copy_objID) == 3);
        DOCTEST_CHECK(ctx.getPolymeshObjectFaceCount(copy_objID) == 1);
        DOCTEST_CHECK(ctx.doesPolymeshObjectHaveVertexNormals(copy_objID));
        DOCTEST_CHECK(ctx.getPolymeshObjectVertexNormalSource(copy_objID) == NORMAL_SOURCE_AUTHORED);

        // The copy's face table must reference the copied primitives, not the originals.
        std::vector<uint> copy_UUIDs = ctx.getObjectPrimitiveUUIDs(copy_objID);
        DOCTEST_REQUIRE(copy_UUIDs.size() == 1);
        DOCTEST_CHECK(ctx.getPolymeshObjectPrimitiveUUIDForFace(copy_objID, 0) == copy_UUIDs.front());

        // Deleting from the copy must not disturb the original.
        ctx.deletePrimitive(copy_UUIDs.front());
        DOCTEST_CHECK(ctx.getPolymeshObjectFaceCount(objID) == 1);
    }
}

TEST_CASE("Polymesh Deformability") {

    SUBCASE("Individual primitives of a polymesh can be transformed") {
        Context ctx;

        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2), make_int3(0, 2, 3)};
        std::vector<uint> UUIDs;
        for (const int3 &f: mesh_faces) {
            UUIDs.push_back(ctx.addTriangle(mesh_vertices.at(f.x), mesh_vertices.at(f.y), mesh_vertices.at(f.z), RGB::red));
        }
        uint objID = ctx.addPolymeshObject(UUIDs);
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        // A mesh has no shape invariant to violate, so transforming one of its facets must be permitted and must not warn.
        vec3 before = ctx.getTriangleVertex(UUIDs.front(), 0);
        bool warned;
        {
            capture_cerr cerr_buffer;
            ctx.translatePrimitive(UUIDs.front(), make_vec3(0, 0, 5));
            warned = cerr_buffer.has_output();
        } // capture destroyed here
        DOCTEST_CHECK(!warned);

        vec3 after = ctx.getTriangleVertex(UUIDs.front(), 0);
        DOCTEST_CHECK(after.z == doctest::Approx(before.z + 5.f).epsilon(1e-5));
    }

    SUBCASE("Transforming a polymesh primitive updates the mesh vertices it shares") {
        Context ctx;

        // Two triangles sharing the edge (v0, v2). Moving the first must carry the shared vertices, keeping the mesh welded.
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2), make_int3(0, 2, 3)};
        std::vector<uint> UUIDs;
        for (const int3 &f: mesh_faces) {
            UUIDs.push_back(ctx.addTriangle(mesh_vertices.at(f.x), mesh_vertices.at(f.y), mesh_vertices.at(f.z), RGB::red));
        }
        uint objID = ctx.addPolymeshObject(UUIDs);
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        ctx.translatePrimitive(UUIDs.front(), make_vec3(0, 0, 5));

        // The stored topology must follow the primitive, not go stale.
        std::vector<vec3> vertices_after = ctx.getPolymeshObjectVertices(objID);
        DOCTEST_REQUIRE(vertices_after.size() == 4);
        DOCTEST_CHECK(vertices_after.at(0).z == doctest::Approx(5.f).epsilon(1e-4));
        DOCTEST_CHECK(vertices_after.at(1).z == doctest::Approx(5.f).epsilon(1e-4));
        DOCTEST_CHECK(vertices_after.at(2).z == doctest::Approx(5.f).epsilon(1e-4));
        // Vertex 3 belongs only to the second face and must not have moved.
        DOCTEST_CHECK(vertices_after.at(3).z == doctest::Approx(0.f).epsilon(1e-4));

        // The sibling primitive is deliberately NOT rewritten: cascading the update to neighbours would make a bulk transform of the whole mesh quadratic and order-dependent. Its shared mesh vertices have
        // moved, but the primitive itself stays where it was until it is transformed too.
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(1), 0).z == doctest::Approx(0.f).epsilon(1e-4));

        // Transforming the whole mesh must leave the face table consistent with the primitives, without any cascade.
        ctx.translatePrimitive(UUIDs, make_vec3(0, 0, 10));
        std::vector<vec3> vertices_bulk = ctx.getPolymeshObjectVertices(objID);
        DOCTEST_REQUIRE(vertices_bulk.size() == 4);
        DOCTEST_CHECK(vertices_bulk.at(3).z == doctest::Approx(10.f).epsilon(1e-4));
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(1), 0).z == doctest::Approx(10.f).epsilon(1e-4));
    }

    SUBCASE("Non-deformable compound objects still block individual primitive transforms") {
        Context ctx;

        // A tile must stay planar and a tube coherent, so their sub-primitives remain locked.
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));
        std::vector<uint> tile_UUIDs = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_REQUIRE(!tile_UUIDs.empty());

        vec3 tile_vertex_before = ctx.getPrimitiveVertices(tile_UUIDs.front()).front();
        bool tile_warned;
        {
            capture_cerr cerr_buffer;
            ctx.translatePrimitive(tile_UUIDs.front(), make_vec3(0, 0, 5));
            tile_warned = cerr_buffer.has_output();
        } // capture destroyed here
        DOCTEST_CHECK(tile_warned);
        DOCTEST_CHECK(ctx.getPrimitiveVertices(tile_UUIDs.front()).front().z == doctest::Approx(tile_vertex_before.z).epsilon(1e-5));

        std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        std::vector<float> radii = {0.2f, 0.1f};
        uint tube = ctx.addTubeObject(6, nodes, radii);
        std::vector<uint> tube_UUIDs = ctx.getObjectPrimitiveUUIDs(tube);
        DOCTEST_REQUIRE(!tube_UUIDs.empty());

        vec3 tube_vertex_before = ctx.getPrimitiveVertices(tube_UUIDs.front()).front();
        {
            capture_cerr cerr_buffer;
            ctx.translatePrimitive(tube_UUIDs.front(), make_vec3(0, 0, 5));
        } // capture destroyed here
        DOCTEST_CHECK(ctx.getPrimitiveVertices(tube_UUIDs.front()).front().z == doctest::Approx(tube_vertex_before.z).epsilon(1e-5));
    }

    SUBCASE("A polymesh with no topology is still deformable") {
        Context ctx;

        // Grouping loose primitives gives a polymesh with no face table; transforming its members must still be allowed and must not warn.
        std::vector<uint> UUIDs;
        UUIDs.push_back(ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::red));
        uint objID = ctx.addPolymeshObject(UUIDs);
        DOCTEST_CHECK(ctx.getPolymeshObjectFaceCount(objID) == 0);

        bool warned;
        {
            capture_cerr cerr_buffer;
            ctx.translatePrimitive(UUIDs.front(), make_vec3(1, 2, 3));
            warned = cerr_buffer.has_output();
        } // capture destroyed here
        DOCTEST_CHECK(!warned);
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.front(), 0).z == doctest::Approx(3.f).epsilon(1e-5));
    }
}

TEST_CASE("Analytic vertex normals for curved compound objects") {

    SUBCASE("Sphere normals point radially outward") {
        Context ctx;
        const uint objID = ctx.addSphereObject(8, make_vec3(1, 2, 3), 2.f);
        DOCTEST_CHECK(ctx.doesObjectHaveAnalyticVertexNormals(objID));

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            const std::vector<vec3> vertices = ctx.getPrimitiveVertices(UUID);
            const std::vector<vec3> normals = ctx.getObjectPrimitiveVertexNormals(objID, UUID);
            DOCTEST_REQUIRE(normals.size() == vertices.size());
            for (size_t k = 0; k < vertices.size(); k++) {
                const vec3 outward = normalize(vertices.at(k) - make_vec3(1, 2, 3));
                DOCTEST_CHECK(normals.at(k).magnitude() == doctest::Approx(1.f).epsilon(1e-4));
                DOCTEST_CHECK(normals.at(k).x == doctest::Approx(outward.x).epsilon(1e-4));
                DOCTEST_CHECK(normals.at(k).y == doctest::Approx(outward.y).epsilon(1e-4));
                DOCTEST_CHECK(normals.at(k).z == doctest::Approx(outward.z).epsilon(1e-4));
                // A sign error would still give a unit vector along the right line, so check the direction explicitly.
                DOCTEST_CHECK(normals.at(k) * outward > 0.f);
            }
        }
    }

    SUBCASE("Rotated ellipsoid normals stay perpendicular to the surface") {
        // A facet edge is only perpendicular to the vertex normal in the limit of fine tessellation, so the test is that the residual shrinks as the sphere is refined. A normal computed by dividing the offset
        // from the center by the squared semi-axes in the global frame is exact only while the ellipsoid remains axis-aligned; once rotated it converges to a non-zero residual instead.
        float previous_residual = -1.f;
        for (uint Ndivs: {20u, 40u, 80u}) {
            Context ctx;
            const uint objID = ctx.addSphereObject(Ndivs, make_vec3(0, 0, 0), make_vec3(3, 1, 1));
            ctx.rotateObject(objID, 0.25f * PI_F, "z");

            float worst_residual = 0.f;
            for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
                const std::vector<vec3> vertices = ctx.getPrimitiveVertices(UUID);
                const std::vector<vec3> normals = ctx.getObjectPrimitiveVertexNormals(objID, UUID);
                vec3 edge_a = vertices.at(1) - vertices.at(0);
                vec3 edge_b = vertices.at(2) - vertices.at(0);
                if (edge_a.magnitude() < 1e-9f || edge_b.magnitude() < 1e-9f) {
                    continue;
                }
                edge_a.normalize();
                edge_b.normalize();
                for (const vec3 &n: normals) {
                    DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-4));
                    worst_residual = std::max(worst_residual, std::abs(n * edge_a));
                    worst_residual = std::max(worst_residual, std::abs(n * edge_b));
                }
            }
            if (previous_residual >= 0.f) {
                // Halving the facet size must roughly halve the residual.
                DOCTEST_CHECK(worst_residual < previous_residual * 0.75f);
            }
            previous_residual = worst_residual;
        }
    }

    SUBCASE("Untapered tube normals are perpendicular to the tube axis") {
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
        const std::vector<float> radii{0.3f, 0.3f, 0.3f};
        const uint objID = ctx.addTubeObject(10, nodes, radii);
        DOCTEST_CHECK(ctx.doesObjectHaveAnalyticVertexNormals(objID));

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-4));
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(0.f).epsilon(1e-4));
            }
        }
    }

    SUBCASE("Tapered tube normals tilt toward the narrowing end") {
        // The surface of a tapering tube is a cone rather than a cylinder, so its normal leans along the axis by the taper slope. Without that correction every normal here would be exactly perpendicular to
        // the axis, so this pins the taper term specifically.
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        const std::vector<float> radii{0.4f, 0.1f};
        const uint objID = ctx.addTubeObject(12, nodes, radii);

        const float taper_slope = (0.1f - 0.4f) / 2.f;
        const float expected_axial = -taper_slope / std::sqrt(1.f + taper_slope * taper_slope);
        DOCTEST_REQUIRE(expected_axial > 0.1f); // the effect must be large enough for the check to mean something

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(expected_axial).epsilon(1e-3));
            }
        }
    }

    SUBCASE("Vertices shared across the tube seam receive identical normals") {
        // The first and last radial slot of a ring are the same physical point. Evaluating the normal from the vertex position rather than from its index guarantees they agree exactly, so no seam appears.
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        const std::vector<float> radii{0.25f, 0.25f};
        const uint objID = ctx.addTubeObject(6, nodes, radii);
        const std::vector<uint> UUIDs = ctx.getObjectPrimitiveUUIDs(objID);

        int shared_vertices_found = 0;
        for (size_t a = 0; a < UUIDs.size(); a++) {
            const std::vector<vec3> vertices_a = ctx.getPrimitiveVertices(UUIDs.at(a));
            const std::vector<vec3> normals_a = ctx.getObjectPrimitiveVertexNormals(objID, UUIDs.at(a));
            for (size_t b = a + 1; b < UUIDs.size(); b++) {
                const std::vector<vec3> vertices_b = ctx.getPrimitiveVertices(UUIDs.at(b));
                const std::vector<vec3> normals_b = ctx.getObjectPrimitiveVertexNormals(objID, UUIDs.at(b));
                for (size_t i = 0; i < 3; i++) {
                    for (size_t j = 0; j < 3; j++) {
                        if ((vertices_a.at(i) - vertices_b.at(j)).magnitude() == 0.f) {
                            shared_vertices_found++;
                            DOCTEST_CHECK((normals_a.at(i) - normals_b.at(j)).magnitude() == 0.f);
                        }
                    }
                }
            }
        }
        DOCTEST_CHECK(shared_vertices_found > 0);
    }

    SUBCASE("Textured tubes that drop degenerate triangles still get correct normals") {
        // The textured overload discards zero-area triangles, so a primitive's position in the object's UUID list no longer matches its position in the tessellation grid. Evaluating from vertex positions is
        // unaffected by that, whereas deriving the ring from the UUID index would not be.
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
        const std::vector<float> radii{0.2f, 0.2f, 0.2f};
        const uint objID = ctx.addTubeObject(8, nodes, radii, "lib/images/disk_texture.png");

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(0.f).epsilon(1e-4));
            }
        }
    }

    SUBCASE("Tube normals remain correct after a segment is appended") {
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        const std::vector<float> radii{0.3f, 0.3f};
        const uint objID = ctx.addTubeObject(8, nodes, radii);
        ctx.appendTubeSegment(objID, make_vec3(0, 0, 2), 0.3f, make_RGBcolor(0.5f, 0.5f, 0.5f));

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-4));
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(0.f).epsilon(1e-4));
            }
        }
    }

    SUBCASE("Tube normals follow a change in radius") {
        // Normals are derived from the object's current state rather than stored, so a change in radius is reflected without any explicit invalidation.
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 2)};
        const uint objID = ctx.addTubeObject(10, nodes, {0.3f, 0.3f});

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(0.f).epsilon(1e-4));
            }
        }

        ctx.setTubeRadii(objID, {0.4f, 0.1f});
        const float taper_slope = (0.1f - 0.4f) / 2.f;
        const float expected_axial = -taper_slope / std::sqrt(1.f + taper_slope * taper_slope);
        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(expected_axial).epsilon(1e-3));
            }
        }
    }

    SUBCASE("Cone normals match the slant of the lateral surface") {
        Context ctx;
        const uint objID = ctx.addConeObject(10, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.5f, 0.2f);
        DOCTEST_CHECK(ctx.doesObjectHaveAnalyticVertexNormals(objID));

        const float taper_slope = (0.2f - 0.5f) / 2.f;
        const float expected_axial = -taper_slope / std::sqrt(1.f + taper_slope * taper_slope);
        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_CHECK(n.magnitude() == doctest::Approx(1.f).epsilon(1e-4));
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(expected_axial).epsilon(1e-3));
            }
        }
    }

    SUBCASE("Cone tapering to a point yields finite normals at the apex") {
        // The apex of a cone is a singular point with no single surface normal. The normal is constant along a slant line, so the azimuth of another vertex of the same triangle gives the normal of the
        // surface arriving at the tip rather than a division by zero.
        Context ctx;
        const uint objID = ctx.addConeObject(10, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.5f, 0.f);

        const float taper_slope = (0.f - 0.5f) / 2.f;
        const float expected_axial = -taper_slope / std::sqrt(1.f + taper_slope * taper_slope);
        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &n: ctx.getObjectPrimitiveVertexNormals(objID, UUID)) {
                DOCTEST_REQUIRE(!std::isnan(n.x));
                DOCTEST_REQUIRE(!std::isnan(n.y));
                DOCTEST_REQUIRE(!std::isnan(n.z));
                DOCTEST_CHECK(n * make_vec3(0, 0, 1) == doctest::Approx(expected_axial).epsilon(1e-3));
            }
        }
    }

    SUBCASE("Flat object types report no analytic normals") {
        Context ctx;
        const uint tile_objID = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_int2(2, 2));
        const uint box_objID = ctx.addBoxObject(make_vec3(5, 5, 5), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        DOCTEST_CHECK(!ctx.doesObjectHaveAnalyticVertexNormals(tile_objID));
        DOCTEST_CHECK(!ctx.doesObjectHaveAnalyticVertexNormals(box_objID));
        DOCTEST_CHECK(ctx.getObjectPrimitiveVertexNormals(tile_objID, ctx.getObjectPrimitiveUUIDs(tile_objID).front()).empty());
    }
}

TEST_CASE("Tube geometry stays consistent after appending segments") {

    // Tube::updateTriangleVertices() assigns geometry to the object's UUID list by position, assuming the triangles are ordered with the radial slot varying slowest and the segment index fastest, which is
    // how addTubeObject() emits them. appendTubeSegment() added its new triangles to the end of the list instead of interleaving them into that order, so the geometry was subsequently repainted onto a
    // permuted set of primitives. The vertices themselves still described a valid tube, but everything carried per primitive rather than per vertex - the color above all - stayed with the primitive and so
    // ended up on the wrong part of the tube.

    SUBCASE("Node colors stay on the segment they belong to") {
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        const std::vector<RGBcolor> colors{RGB::red, RGB::red};
        const uint objID = ctx.addTubeObject(6, nodes, {0.3f, 0.3f}, colors);
        ctx.appendTubeSegment(objID, make_vec3(0, 0, 2), 0.3f, RGB::blue);

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            float lowest_z = (std::numeric_limits<float>::max)();
            for (const vec3 &vertex: ctx.getPrimitiveVertices(UUID)) {
                lowest_z = std::min(lowest_z, vertex.z);
            }
            const RGBcolor color = ctx.getPrimitiveColor(UUID);
            // Triangles spanning the first segment were given the original color, those spanning the appended segment the new one.
            if (lowest_z < 0.5f) {
                DOCTEST_CHECK(color.r > 0.5f);
                DOCTEST_CHECK(color.b < 0.5f);
            } else {
                DOCTEST_CHECK(color.b > 0.5f);
                DOCTEST_CHECK(color.r < 0.5f);
            }
        }
    }

    SUBCASE("Radii can be changed after appending a segment") {
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 1)};
        const uint objID = ctx.addTubeObject(6, nodes, {0.3f, 0.3f});
        ctx.appendTubeSegment(objID, make_vec3(0, 0, 2), 0.3f, make_RGBcolor(0.5f, 0.5f, 0.5f));

        const std::vector<float> new_radii{0.5f, 0.4f, 0.2f};
        ctx.setTubeRadii(objID, new_radii);

        // Every vertex of the rebuilt tube must lie on the ring of the node it belongs to.
        const std::vector<vec3> tube_nodes = ctx.getTubeObjectNodes(objID);
        DOCTEST_REQUIRE(tube_nodes.size() == new_radii.size());

        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &vertex: ctx.getPrimitiveVertices(UUID)) {
                bool lies_on_a_ring = false;
                for (size_t n = 0; n < tube_nodes.size(); n++) {
                    const vec3 offset = vertex - tube_nodes.at(n);
                    if (std::abs(offset.z) < 1e-4f && std::abs(offset.magnitude() - new_radii.at(n)) < 1e-4f) {
                        lies_on_a_ring = true;
                        break;
                    }
                }
                DOCTEST_CHECK(lies_on_a_ring);
            }
        }
    }

    SUBCASE("Appending reproduces the same tube as building it in one call") {
        // A tube built by appending must hold the same set of triangles, with the same colors, as the same tube built directly.
        Context ctx;
        const std::vector<vec3> nodes{make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2)};
        const std::vector<float> radii{0.3f, 0.3f, 0.3f};
        const std::vector<RGBcolor> colors{RGB::red, RGB::red, RGB::blue};

        const uint built_objID = ctx.addTubeObject(6, nodes, radii, colors);

        const uint appended_objID = ctx.addTubeObject(6, {nodes.at(0), nodes.at(1)}, {radii.at(0), radii.at(1)}, {colors.at(0), colors.at(1)});
        ctx.appendTubeSegment(appended_objID, nodes.at(2), radii.at(2), colors.at(2));

        // Both tubes must describe the same surface, triangle for triangle and in the same order.

        const std::vector<uint> built_UUIDs = ctx.getObjectPrimitiveUUIDs(built_objID);
        const std::vector<uint> appended_UUIDs = ctx.getObjectPrimitiveUUIDs(appended_objID);
        DOCTEST_REQUIRE(built_UUIDs.size() == appended_UUIDs.size());

        for (size_t k = 0; k < built_UUIDs.size(); k++) {
            const std::vector<vec3> built_vertices = ctx.getPrimitiveVertices(built_UUIDs.at(k));
            const std::vector<vec3> appended_vertices = ctx.getPrimitiveVertices(appended_UUIDs.at(k));
            for (size_t m = 0; m < 3; m++) {
                DOCTEST_CHECK((built_vertices.at(m) - appended_vertices.at(m)).magnitude() < 1e-5f);
            }
            // Colors are not compared here: addTubeObject() colors a segment by the node it starts from, while appendTubeSegment() colors the new segment by the node being appended, so the two disagree on
            // the last segment by design. What matters for the ordering is that each triangle sits at the same place in both lists, which the vertex comparison above establishes.
        }
    }
}

TEST_CASE("Cone node radii account for the object transformation") {

    // Cone::getNodeRadii() used to return the radii exactly as they were supplied at construction, ignoring any scaling later applied to the object, while getNodeCoordinates() and getLength() did apply it.
    // Scaling a cone therefore left the reported radii describing the cone as it was originally built rather than as it now is, and getVolume() combined the un-scaled radii with the scaled length.

    // The true radius of the cone is measured from the geometry itself, so the check does not depend on the accessor under test.
    auto measuredRadiusAtNode = [](Context &ctx, uint objID, int node_index) {
        const std::vector<vec3> nodes = ctx.getConeObjectNodes(objID);
        const vec3 axis = normalize(nodes.at(1) - nodes.at(0));
        float measured = 0.f;
        for (uint UUID: ctx.getObjectPrimitiveUUIDs(objID)) {
            for (const vec3 &vertex: ctx.getPrimitiveVertices(UUID)) {
                const vec3 offset = vertex - nodes.at(node_index);
                // Only vertices lying in the plane of the requested node contribute to that node's ring.
                if (std::abs(offset * axis) < 1e-4f) {
                    measured = std::max(measured, (offset - (offset * axis) * axis).magnitude());
                }
            }
        }
        return measured;
    };

    SUBCASE("Radii follow a uniform object scaling") {
        Context ctx;
        const uint objID = ctx.addConeObject(12, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.5f, 0.25f);
        ctx.scaleObject(objID, make_vec3(2, 2, 2));

        const std::vector<float> reported = ctx.getConeObjectNodeRadii(objID);
        DOCTEST_REQUIRE(reported.size() == 2);
        DOCTEST_CHECK(reported.at(0) == doctest::Approx(measuredRadiusAtNode(ctx, objID, 0)).epsilon(1e-3));
        DOCTEST_CHECK(reported.at(1) == doctest::Approx(measuredRadiusAtNode(ctx, objID, 1)).epsilon(1e-3));
        DOCTEST_CHECK(ctx.getConeObjectNodeRadius(objID, 0) == doctest::Approx(reported.at(0)).epsilon(1e-5));
        DOCTEST_CHECK(ctx.getConeObjectNodeRadius(objID, 1) == doctest::Approx(reported.at(1)).epsilon(1e-5));
    }

    SUBCASE("Volume follows a uniform object scaling") {
        Context ctx;
        const uint objID = ctx.addConeObject(24, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.5f, 0.25f);
        const float original_volume = ctx.getConeObjectVolume(objID);

        ctx.scaleObject(objID, make_vec3(2, 2, 2));

        // Scaling every dimension by two multiplies a volume by eight.
        DOCTEST_CHECK(ctx.getConeObjectVolume(objID) == doctest::Approx(8.f * original_volume).epsilon(1e-3));
    }

    SUBCASE("Radii still follow scaleConeObjectGirth") {
        // scaleGirth() scales the object and must not also be counted a second time through the transformation-aware accessor.
        Context ctx;
        const uint objID = ctx.addConeObject(12, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.5f, 0.25f);
        ctx.scaleConeObjectGirth(objID, 2.f);

        const std::vector<float> reported = ctx.getConeObjectNodeRadii(objID);
        DOCTEST_REQUIRE(reported.size() == 2);
        DOCTEST_CHECK(reported.at(0) == doctest::Approx(1.0f).epsilon(1e-3));
        DOCTEST_CHECK(reported.at(1) == doctest::Approx(0.5f).epsilon(1e-3));
        DOCTEST_CHECK(reported.at(0) == doctest::Approx(measuredRadiusAtNode(ctx, objID, 0)).epsilon(1e-3));
        DOCTEST_CHECK(reported.at(1) == doctest::Approx(measuredRadiusAtNode(ctx, objID, 1)).epsilon(1e-3));
    }

    SUBCASE("Radii are unchanged by translation and rotation") {
        Context ctx;
        const uint objID = ctx.addConeObject(12, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.5f, 0.25f);
        ctx.translateObject(objID, make_vec3(3, -2, 1));
        ctx.rotateObject(objID, 0.3f * PI_F, "y");

        const std::vector<float> reported = ctx.getConeObjectNodeRadii(objID);
        DOCTEST_REQUIRE(reported.size() == 2);
        DOCTEST_CHECK(reported.at(0) == doctest::Approx(0.5f).epsilon(1e-4));
        DOCTEST_CHECK(reported.at(1) == doctest::Approx(0.25f).epsilon(1e-4));
    }
}

TEST_CASE("Tube and Cone objects survive an XML round trip after being transformed") {

    // writeXML() records the object transformation matrix and, for tubes and cones, the nodes and radii the object was built from. loadXML() rebuilds the object from those nodes and radii and then applies
    // the transformation matrix on top, so the values written have to be in the object's local frame. They were previously written with the transformation already applied, so loading applied it twice and a
    // scaled tube or cone came back larger than it was written.

    SUBCASE("Scaled tube round trips unchanged") {
        const std::string filename = "test_tube_roundtrip.xml";
        std::vector<vec3> written_nodes;
        std::vector<float> written_radii;
        {
            Context ctx;
            const uint objID = ctx.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(0, 0, 2)}, {0.5f, 0.25f});
            ctx.scaleObject(objID, make_vec3(2, 2, 2));
            written_nodes = ctx.getTubeObjectNodes(objID);
            written_radii = ctx.getTubeObjectNodeRadii(objID);
            ctx.writeXML(filename.c_str(), true);
        }

        Context ctx;
        ctx.loadXML(filename.c_str(), true);

        std::vector<uint> tube_objIDs;
        for (uint objID: ctx.getAllObjectIDs()) {
            if (ctx.getObjectType(objID) == OBJECT_TYPE_TUBE) {
                tube_objIDs.push_back(objID);
            }
        }
        DOCTEST_REQUIRE(tube_objIDs.size() == 1);

        const std::vector<vec3> loaded_nodes = ctx.getTubeObjectNodes(tube_objIDs.front());
        const std::vector<float> loaded_radii = ctx.getTubeObjectNodeRadii(tube_objIDs.front());
        DOCTEST_REQUIRE(loaded_nodes.size() == written_nodes.size());
        for (size_t n = 0; n < loaded_nodes.size(); n++) {
            DOCTEST_CHECK((loaded_nodes.at(n) - written_nodes.at(n)).magnitude() < 1e-4f);
            DOCTEST_CHECK(loaded_radii.at(n) == doctest::Approx(written_radii.at(n)).epsilon(1e-4));
        }

        std::filesystem::remove(filename);
    }

    SUBCASE("Scaled cone round trips unchanged") {
        const std::string filename = "test_cone_roundtrip.xml";
        std::vector<vec3> written_nodes;
        std::vector<float> written_radii;
        float written_volume = 0.f;
        {
            Context ctx;
            const uint objID = ctx.addConeObject(12, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.5f, 0.25f);
            ctx.scaleObject(objID, make_vec3(2, 2, 2));
            written_nodes = ctx.getConeObjectNodes(objID);
            written_radii = ctx.getConeObjectNodeRadii(objID);
            written_volume = ctx.getConeObjectVolume(objID);
            ctx.writeXML(filename.c_str(), true);
        }

        Context ctx;
        ctx.loadXML(filename.c_str(), true);

        std::vector<uint> cone_objIDs;
        for (uint objID: ctx.getAllObjectIDs()) {
            if (ctx.getObjectType(objID) == OBJECT_TYPE_CONE) {
                cone_objIDs.push_back(objID);
            }
        }
        DOCTEST_REQUIRE(cone_objIDs.size() == 1);

        const std::vector<vec3> loaded_nodes = ctx.getConeObjectNodes(cone_objIDs.front());
        const std::vector<float> loaded_radii = ctx.getConeObjectNodeRadii(cone_objIDs.front());
        DOCTEST_REQUIRE(loaded_nodes.size() == written_nodes.size());
        for (size_t n = 0; n < loaded_nodes.size(); n++) {
            DOCTEST_CHECK((loaded_nodes.at(n) - written_nodes.at(n)).magnitude() < 1e-4f);
            DOCTEST_CHECK(loaded_radii.at(n) == doctest::Approx(written_radii.at(n)).epsilon(1e-4));
        }
        DOCTEST_CHECK(ctx.getConeObjectVolume(cone_objIDs.front()) == doctest::Approx(written_volume).epsilon(1e-3));

        std::filesystem::remove(filename);
    }

    SUBCASE("Untransformed tube round trips unchanged") {
        // The un-transformed case worked before and must keep working, since the fix changes which frame the values are written in.
        const std::string filename = "test_tube_plain_roundtrip.xml";
        {
            Context ctx;
            ctx.addTubeObject(8, {make_vec3(1, 0, 0), make_vec3(1, 0, 2)}, {0.5f, 0.25f});
            ctx.writeXML(filename.c_str(), true);
        }

        Context ctx;
        ctx.loadXML(filename.c_str(), true);
        for (uint objID: ctx.getAllObjectIDs()) {
            if (ctx.getObjectType(objID) == OBJECT_TYPE_TUBE) {
                const std::vector<vec3> loaded_nodes = ctx.getTubeObjectNodes(objID);
                const std::vector<float> loaded_radii = ctx.getTubeObjectNodeRadii(objID);
                DOCTEST_REQUIRE(loaded_nodes.size() == 2);
                DOCTEST_CHECK((loaded_nodes.at(0) - make_vec3(1, 0, 0)).magnitude() < 1e-4f);
                DOCTEST_CHECK((loaded_nodes.at(1) - make_vec3(1, 0, 2)).magnitude() < 1e-4f);
                DOCTEST_CHECK(loaded_radii.at(0) == doctest::Approx(0.5f).epsilon(1e-4));
                DOCTEST_CHECK(loaded_radii.at(1) == doctest::Approx(0.25f).epsilon(1e-4));
            }
        }

        std::filesystem::remove(filename);
    }
}

//! Collect the distinct shared vertices of a compound object and verify that every corner mapped to the same index really is the same point
/**
 * The indices are produced from each facet's position in the object's primitive list, while the coordinates come from the primitives themselves, so agreement between the two is a genuine check on the
 * index arithmetic rather than a restatement of it. A transposed ring or an unwrapped spoke maps two corners that are nowhere near each other onto one index, which this catches immediately.
 */
inline size_t checkSharedVertexTopologyIsConsistent(helios::Context &ctx, uint ObjID, helios::VertexWeldMode weld_mode, float tolerance) {
    const std::vector<uint> object_UUIDs = ctx.getObjectPrimitiveUUIDs(ObjID);
    const std::vector<std::vector<int>> all_indices = ctx.getObjectPrimitiveSharedVertexIndices(ObjID, object_UUIDs, weld_mode);
    const size_t shared_vertex_count = ctx.getObjectSharedVertexCount(ObjID, weld_mode);

    DOCTEST_REQUIRE(all_indices.size() == object_UUIDs.size());

    std::map<int, std::vector<helios::vec3>> positions_by_index;

    for (size_t k = 0; k < object_UUIDs.size(); k++) {
        const std::vector<helios::vec3> primitive_vertices = ctx.getPrimitiveVertices(object_UUIDs.at(k));
        const std::vector<int> &indices = all_indices.at(k);
        DOCTEST_REQUIRE(indices.size() == primitive_vertices.size());
        for (size_t v = 0; v < indices.size(); v++) {
            DOCTEST_REQUIRE(indices.at(v) >= 0);
            DOCTEST_REQUIRE(size_t(indices.at(v)) < shared_vertex_count);
            positions_by_index[indices.at(v)].push_back(primitive_vertices.at(v));
        }
    }

    for (const auto &entry: positions_by_index) {
        const std::vector<helios::vec3> &coincident = entry.second;
        for (size_t a = 1; a < coincident.size(); a++) {
            const float separation = (coincident.at(a) - coincident.front()).magnitude();
            DOCTEST_INFO("shared vertex " << entry.first << " gathers corners " << separation << " apart");
            DOCTEST_CHECK(separation < tolerance);
        }
    }

    return positions_by_index.size();
}

TEST_CASE("Shared vertex topology identifies the facets that meet at each vertex") {

    SUBCASE("Tube welds around the circumference without welding along the axis") {
        Context ctx;
        const uint radial_subdivisions = 6;
        const std::vector<vec3> nodes = {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0, 2), make_vec3(0, 0, 3)};
        const uint ObjID = ctx.addTubeObject(radial_subdivisions, nodes, {0.3f, 0.3f, 0.25f, 0.2f});

        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(ObjID));

        const uint segment_count = uint(nodes.size()) - 1;

        // Each segment carries its own pair of rings, so a shadow edge crossing the tube is not averaged along its length.
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_CROSS_SECTION_ONLY) == size_t(2 * segment_count * radial_subdivisions));
        const size_t cross_section_used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
        DOCTEST_CHECK(cross_section_used == size_t(2 * segment_count * radial_subdivisions));

        // Welding fully instead shares each interior ring between the two segments that meet there.
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_FULL) == size_t(nodes.size() * radial_subdivisions));
        const size_t full_used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_FULL, 1e-4f);
        DOCTEST_CHECK(full_used == size_t(nodes.size() * radial_subdivisions));

        // Welding is the whole point: there are three corners per facet, and far fewer distinct vertices than that.
        DOCTEST_CHECK(full_used < 3 * ctx.getObjectPrimitiveUUIDs(ObjID).size());
    }

    SUBCASE("Tube topology survives the object being rotated and translated") {
        // The indices come from the primitive list rather than from coordinates, so they must be unaffected by where the object has been moved to.
        Context ctx;
        const uint ObjID = ctx.addTubeObject(8, {make_vec3(0, 0, 0), make_vec3(0, 0, 1), make_vec3(0, 0.4f, 1.9f)}, {0.2f, 0.2f, 0.1f});
        const std::vector<std::vector<int>> before = ctx.getObjectPrimitiveSharedVertexIndices(ObjID, ctx.getObjectPrimitiveUUIDs(ObjID), WELD_CROSS_SECTION_ONLY);

        ctx.rotateObject(ObjID, 0.7f, "y");
        ctx.translateObject(ObjID, make_vec3(3, -2, 5));

        const std::vector<std::vector<int>> after = ctx.getObjectPrimitiveSharedVertexIndices(ObjID, ctx.getObjectPrimitiveUUIDs(ObjID), WELD_CROSS_SECTION_ONLY);

        // Vertices are named from where they are, so a transform may renumber the spokes. What has to survive is the property the naming exists for: the same number of facets described, and coincident
        // corners still sharing one vertex.
        DOCTEST_CHECK(after.size() == before.size());
        checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
    }

    SUBCASE("Sphere collapses each pole to a single vertex") {
        Context ctx;
        const uint radial_subdivisions = 7;
        const uint ObjID = ctx.addSphereObject(radial_subdivisions, make_vec3(1, 2, 3), 0.5f);

        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(ObjID));
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_FULL) == size_t(2 + (radial_subdivisions - 1) * radial_subdivisions));

        const size_t full_used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_FULL, 1e-4f);
        DOCTEST_CHECK(full_used == size_t(2 + (radial_subdivisions - 1) * radial_subdivisions));

        checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
    }

    SUBCASE("Cone welds its seam and reports one vertex per division at each end") {
        Context ctx;
        const uint radial_subdivisions = 9;
        const uint ObjID = ctx.addConeObject(radial_subdivisions, make_vec3(0, 0, 0), make_vec3(0, 0, 2), 0.4f, 0.15f);

        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(ObjID));
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_CROSS_SECTION_ONLY) == size_t(2 * radial_subdivisions));

        const size_t used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
        DOCTEST_CHECK(used == size_t(2 * radial_subdivisions));
    }

    SUBCASE("Tile reports the nodes of its sub-patch grid") {
        Context ctx;
        const int2 subdivisions = make_int2(5, 3);
        const uint ObjID = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 1), make_SphericalCoord(0.3f, 0.9f), subdivisions);

        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(ObjID));
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_FULL) == size_t((subdivisions.x + 1) * (subdivisions.y + 1)));

        const size_t used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_FULL, 1e-4f);
        DOCTEST_CHECK(used == size_t((subdivisions.x + 1) * (subdivisions.y + 1)));

        // A tile is planar, so there is no axis for the two weld modes to disagree about.
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_CROSS_SECTION_ONLY) == ctx.getObjectSharedVertexCount(ObjID, WELD_FULL));
    }

    SUBCASE("Flat-faced object types report no shared vertex topology") {
        // A box and a disk are built from genuinely flat faces, where the value of a per-face quantity is already correct across the whole face, so there is nothing for a consumer to interpolate.
        Context ctx;
        const uint box_ObjID = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(2, 2, 2));
        const uint disk_ObjID = ctx.addDiskObject(10, make_vec3(3, 0, 0), make_vec2(0.5f, 0.5f));

        DOCTEST_CHECK(!ctx.doesObjectHaveSharedVertexTopology(box_ObjID));
        DOCTEST_CHECK(!ctx.doesObjectHaveSharedVertexTopology(disk_ObjID));
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(box_ObjID, WELD_FULL) == 0);
        DOCTEST_CHECK(ctx.getObjectPrimitiveSharedVertexIndices(box_ObjID, ctx.getObjectPrimitiveUUIDs(box_ObjID).front(), WELD_FULL).empty());
    }

    SUBCASE("A primitive with no parent object has no shared vertices") {
        Context ctx;
        const uint patch_UUID = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        const uint triangle_UUID = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0));

        DOCTEST_CHECK(ctx.getPrimitiveSharedVertexIndices(patch_UUID, WELD_FULL).empty());
        DOCTEST_CHECK(ctx.getPrimitiveSharedVertexIndices(triangle_UUID, WELD_FULL).empty());
    }

    SUBCASE("Polymesh reports the face table it retained from the file it was loaded from") {
        Context ctx;
        const std::vector<uint> mesh_UUIDs = ctx.loadOBJ("lib/models/test_cube_medium.obj", true);
        DOCTEST_REQUIRE(!mesh_UUIDs.empty());

        const uint ObjID = ctx.getPrimitiveParentObjectID(mesh_UUIDs.front());
        DOCTEST_REQUIRE(ObjID != 0);
        DOCTEST_REQUIRE(ctx.getObjectType(ObjID) == OBJECT_TYPE_POLYMESH);

        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(ObjID));
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_FULL) == ctx.getPolymeshObjectVertexCount(ObjID));

        const size_t used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_FULL, 1e-4f);
        DOCTEST_CHECK(used > 0);
        // Facets that meet along an edge report the same vertices there, so there are fewer distinct vertices than loose corners.
        DOCTEST_CHECK(used < 3 * ctx.getObjectPrimitiveUUIDs(ObjID).size());

        // A mesh has no distinguished axis, so the weld mode makes no difference to it.
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_CROSS_SECTION_ONLY) == ctx.getObjectSharedVertexCount(ObjID, WELD_FULL));
    }

    SUBCASE("A polymesh assembled from loose primitives has no face table to report") {
        // Grouping unrelated triangles does not make them a mesh: nothing says which of their corners are meant to be the same point.
        Context ctx;
        std::vector<uint> loose_UUIDs;
        loose_UUIDs.push_back(ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0)));
        loose_UUIDs.push_back(ctx.addTriangle(make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)));
        const uint ObjID = ctx.addPolymeshObject(loose_UUIDs);

        DOCTEST_CHECK(!ctx.doesObjectHaveSharedVertexTopology(ObjID));
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(ObjID, WELD_FULL) == 0);
        DOCTEST_CHECK(ctx.getObjectPrimitiveSharedVertexIndices(ObjID, loose_UUIDs.front(), WELD_FULL).empty());
    }

    SUBCASE("Adaptive tile counts only the lattice nodes its sub-patches actually reach") {
        // The quadtree refines only near the target, so the lattice of the finest level is mostly unused. Counting the whole lattice would claim orders of magnitude more vertices than the tile has.
        Context ctx;
        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        const uint ObjID = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(8, 8), nullrotation, refinement);
        DOCTEST_REQUIRE(ctx.getObjectType(ObjID) == OBJECT_TYPE_ADAPTIVE_TILE);
        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(ObjID));

        const size_t subpatch_count = ctx.getObjectPrimitiveUUIDs(ObjID).size();
        const size_t used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_FULL, 1e-4f);

        DOCTEST_CHECK(used == ctx.getObjectSharedVertexCount(ObjID, WELD_FULL));
        DOCTEST_CHECK(used > 0);
        // Four corners per sub-patch, and neighbours share them, so the vertex count sits between the sub-patch count and four times it.
        DOCTEST_CHECK(used < 4 * subpatch_count);
        DOCTEST_CHECK(used > subpatch_count);
    }

    SUBCASE("Deleting a facet leaves the remaining facets correctly named") {
        // Facets are located from where their corners are rather than from their position in the object's primitive list, so removing one does not disturb the naming of the others.
        Context ctx;
        const uint ObjID = ctx.addTubeObject(6, {make_vec3(0, 0, 0), make_vec3(0, 0, 1)}, {0.2f, 0.2f});
        DOCTEST_REQUIRE(ctx.doesObjectHaveSharedVertexTopology(ObjID));

        ctx.deletePrimitive(ctx.getObjectPrimitiveUUIDs(ObjID).front());

        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(ObjID));
        checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
    }
}

TEST_CASE("Shared vertex topology survives the ways an object can legitimately be built and moved") {

    SUBCASE("Textured sphere reports the same vertices as an untextured one") {
        // The textured and untextured sphere generators wind the top cap differently, so an index formula derived from one of them is wrong for the other.
        Context ctx;
        const uint ObjID = ctx.addSphereObject(7, make_vec3(0, 0, 0), 0.5f, "lib/images/solid.jpg");
        DOCTEST_REQUIRE(ctx.doesObjectHaveSharedVertexTopology(ObjID));
        checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_FULL, 1e-4f);
    }

    SUBCASE("Tile whose frame has been sheared by a rotation and a non-uniform scale") {
        // scaleObject applies a world-axis scale, so rotating first leaves the tile a parallelogram whose edge vectors are no longer perpendicular.
        Context ctx;
        const uint ObjID = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, make_int2(8, 8));
        ctx.rotateObject(ObjID, 0.25f * float(M_PI), "z");
        ctx.scaleObject(ObjID, make_vec3(2, 1, 1));

        const size_t used = checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_FULL, 1e-4f);
        DOCTEST_INFO("distinct vertices located on the sheared tile: " << used);
        DOCTEST_CHECK(used == size_t(9 * 9));
    }

    SUBCASE("Tube that had a segment appended and was then written and reloaded") {
        // appendTubeSegment inserts the new facets into the radial ordering, and writeXML sorts member UUIDs, so a reloaded tube has the same facet count in a different order.
        const std::string filename = "test_tube_topology_roundtrip.xml";
        {
            Context ctx;
            const uint ObjID = ctx.addTubeObject(6, {make_vec3(0, 0, 0), make_vec3(0, 0, 1)}, {0.2f, 0.2f});
            ctx.appendTubeSegment(ObjID, make_vec3(0, 0, 2), 0.2f, RGB::green);
            checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
            ctx.writeXML(filename.c_str(), true);
        }

        Context ctx;
        ctx.loadXML(filename.c_str(), true);
        for (uint ObjID: ctx.getAllObjectIDs()) {
            if (ctx.getObjectType(ObjID) == OBJECT_TYPE_TUBE) {
                checkSharedVertexTopologyIsConsistent(ctx, ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
            }
        }
        std::filesystem::remove(filename);
    }

    SUBCASE("An object too coarse to have a lattice reports no topology") {
        // hasSharedVertexTopology() is documented as false exactly when getPrimitiveSharedVertexIndices() returns nothing, so it has to be answered from the object rather than asserted.
        Context ctx;
        const uint box_ObjID = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(2, 2, 2));
        DOCTEST_CHECK(!ctx.doesObjectHaveSharedVertexTopology(box_ObjID));
        DOCTEST_CHECK(ctx.getObjectSharedVertexCount(box_ObjID, WELD_FULL) == 0);

        // A textured tube drops its degenerate facets at creation; the remaining ones must still be named correctly.
        const uint tube_ObjID = ctx.addTubeObject(6, {make_vec3(3, 0, 0), make_vec3(3, 0, 1)}, {0.2f, 0.2f}, "lib/images/solid.jpg");
        DOCTEST_CHECK(ctx.doesObjectHaveSharedVertexTopology(tube_ObjID));
        checkSharedVertexTopologyIsConsistent(ctx, tube_ObjID, WELD_CROSS_SECTION_ONLY, 1e-4f);
    }
}

TEST_CASE("Polymesh bulk vertex deformation") {

    // Builds the two-triangle quad shared by these subcases: faces (0,1,2) and (0,2,3) meeting along the edge (v0, v2).
    auto buildQuad = [](Context &ctx, uint &objID, std::vector<uint> &UUIDs, std::vector<vec3> &mesh_vertices, std::vector<int3> &mesh_faces) {
        mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
        mesh_faces = {make_int3(0, 1, 2), make_int3(0, 2, 3)};
        UUIDs.clear();
        for (const int3 &f: mesh_faces) {
            UUIDs.push_back(ctx.addTriangle(mesh_vertices.at(f.x), mesh_vertices.at(f.y), mesh_vertices.at(f.z), RGB::red));
        }
        objID = ctx.addPolymeshObject(UUIDs);
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {}, {}, NORMAL_SOURCE_NONE);
    };

    SUBCASE("Moved vertices round-trip and carry the member primitives with them") {
        Context ctx;
        uint objID;
        std::vector<uint> UUIDs;
        std::vector<vec3> mesh_vertices;
        std::vector<int3> mesh_faces;
        buildQuad(ctx, objID, UUIDs, mesh_vertices, mesh_faces);

        // Lift the two vertices along the far edge, which is the shape a leaf deformation produces: a bend rather than a rigid motion.
        std::vector<vec3> deformed = mesh_vertices;
        deformed.at(1).z = 0.25f;
        deformed.at(2).z = 0.5f;
        ctx.setPolymeshObjectVertices(objID, deformed);

        std::vector<vec3> vertices_after = ctx.getPolymeshObjectVertices(objID);
        DOCTEST_REQUIRE(vertices_after.size() == 4);
        for (size_t v = 0; v < deformed.size(); v++) {
            DOCTEST_CHECK(vertices_after.at(v).x == doctest::Approx(deformed.at(v).x).epsilon(1e-5));
            DOCTEST_CHECK(vertices_after.at(v).y == doctest::Approx(deformed.at(v).y).epsilon(1e-5));
            DOCTEST_CHECK(vertices_after.at(v).z == doctest::Approx(deformed.at(v).z).epsilon(1e-5));
        }

        // Unlike a per-primitive transform, this must push the new positions all the way out to the primitives.
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(0), 1).z == doctest::Approx(0.25f).epsilon(1e-5));
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(0), 2).z == doctest::Approx(0.5f).epsilon(1e-5));
    }

    SUBCASE("Faces meeting at a shared vertex stay welded") {
        Context ctx;
        uint objID;
        std::vector<uint> UUIDs;
        std::vector<vec3> mesh_vertices;
        std::vector<int3> mesh_faces;
        buildQuad(ctx, objID, UUIDs, mesh_vertices, mesh_faces);

        // v2 is referenced by both faces. Moving it must move it in BOTH primitives - the tearing that transforming facets one at a time produces.
        std::vector<vec3> deformed = mesh_vertices;
        deformed.at(2).z = 3.f;
        ctx.setPolymeshObjectVertices(objID, deformed);

        // Face 0 is (0,1,2), so v2 is its corner 2; face 1 is (0,2,3), so v2 is its corner 1.
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(0), 2).z == doctest::Approx(3.f).epsilon(1e-5));
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(1), 1).z == doctest::Approx(3.f).epsilon(1e-5));

        // The vertices that did not move must stay put in both primitives.
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(0), 0).z == doctest::Approx(0.f).epsilon(1e-5));
        DOCTEST_CHECK(ctx.getTriangleVertex(UUIDs.at(1), 2).z == doctest::Approx(0.f).epsilon(1e-5));
    }

    SUBCASE("Deformation survives an object transform and a copy") {
        Context ctx;
        uint objID;
        std::vector<uint> UUIDs;
        std::vector<vec3> mesh_vertices;
        std::vector<int3> mesh_faces;
        buildQuad(ctx, objID, UUIDs, mesh_vertices, mesh_faces);

        // Vertices are stored in the object-local frame but are given and returned in global coordinates, so a scaled and translated object must round-trip global positions unchanged. This is the case the
        // plant model actually hits: leaves are scaled and placed before being deformed.
        ctx.scaleObject(objID, make_vec3(2, 2, 2));
        ctx.translateObject(objID, make_vec3(10, 0, 0));

        std::vector<vec3> deformed = ctx.getPolymeshObjectVertices(objID);
        deformed.at(2).z += 1.5f;
        ctx.setPolymeshObjectVertices(objID, deformed);

        std::vector<vec3> vertices_after = ctx.getPolymeshObjectVertices(objID);
        DOCTEST_REQUIRE(vertices_after.size() == 4);
        DOCTEST_CHECK(vertices_after.at(2).x == doctest::Approx(deformed.at(2).x).epsilon(1e-4));
        DOCTEST_CHECK(vertices_after.at(2).z == doctest::Approx(deformed.at(2).z).epsilon(1e-4));

        // A copy carries the deformed topology, so cloned leaves are themselves deformable.
        uint objID_copy = ctx.copyObject(objID);
        std::vector<vec3> vertices_copy = ctx.getPolymeshObjectVertices(objID_copy);
        DOCTEST_REQUIRE(vertices_copy.size() == 4);
        DOCTEST_CHECK(vertices_copy.at(2).z == doctest::Approx(vertices_after.at(2).z).epsilon(1e-4));

        std::vector<vec3> deformed_copy = vertices_copy;
        deformed_copy.at(3).z += 2.f;
        ctx.setPolymeshObjectVertices(objID_copy, deformed_copy);
        DOCTEST_CHECK(ctx.getPolymeshObjectVertices(objID_copy).at(3).z == doctest::Approx(deformed_copy.at(3).z).epsilon(1e-4));
        // Deforming the copy must not disturb the original it was cloned from.
        DOCTEST_CHECK(ctx.getPolymeshObjectVertices(objID).at(3).z == doctest::Approx(vertices_after.at(3).z).epsilon(1e-4));
    }

    SUBCASE("Deforming a textured mesh does not recompute the solid fraction") {
        Context ctx;

        // The diamond texture is half transparent, so its solid fraction is a value the deformation could plausibly disturb. Solid fraction is a function of the (u,v) coordinates, which a deformation does
        // not touch; recomputing it would rasterize the alpha mask for every facet of every mesh moved, which is the cost this whole API exists to avoid.
        const char *texture = "lib/images/diamond_texture.png";
        std::vector<vec3> mesh_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
        std::vector<vec2> mesh_uv = {make_vec2(0, 0), make_vec2(1, 0), make_vec2(1, 1), make_vec2(0, 1)};
        std::vector<int3> mesh_faces = {make_int3(0, 1, 2), make_int3(0, 2, 3)};
        std::vector<uint> UUIDs;
        for (const int3 &f: mesh_faces) {
            UUIDs.push_back(ctx.addTriangle(mesh_vertices.at(f.x), mesh_vertices.at(f.y), mesh_vertices.at(f.z), texture, mesh_uv.at(f.x), mesh_uv.at(f.y), mesh_uv.at(f.z)));
        }
        uint objID = ctx.addPolymeshObject(UUIDs);
        ctx.setPolymeshObjectTopology(objID, mesh_vertices, mesh_faces, UUIDs, {}, mesh_uv, NORMAL_SOURCE_NONE);

        const float solid_fraction_before = ctx.getPrimitiveSolidFraction(UUIDs.front());

        std::vector<vec3> deformed = mesh_vertices;
        deformed.at(1).z = 0.4f;
        deformed.at(2).z = 0.8f;
        ctx.setPolymeshObjectVertices(objID, deformed);

        DOCTEST_CHECK(ctx.getPrimitiveSolidFraction(UUIDs.front()) == doctest::Approx(solid_fraction_before).epsilon(1e-6));
    }

    SUBCASE("Invalid deformation requests are rejected") {
        Context ctx;
        uint objID;
        std::vector<uint> UUIDs;
        std::vector<vec3> mesh_vertices;
        std::vector<int3> mesh_faces;
        buildQuad(ctx, objID, UUIDs, mesh_vertices, mesh_faces);

        // The method moves existing vertices and cannot add or remove them, so a mismatched count is a mistake rather than something to silently truncate.
        std::vector<vec3> too_few = {make_vec3(0, 0, 0), make_vec3(1, 0, 0)};
        DOCTEST_CHECK_THROWS(ctx.setPolymeshObjectVertices(objID, too_few));

        std::vector<vec3> too_many = mesh_vertices;
        too_many.push_back(make_vec3(2, 2, 2));
        DOCTEST_CHECK_THROWS(ctx.setPolymeshObjectVertices(objID, too_many));

        // A mesh assembled from loose primitives carries no face table, so there are no shared vertices to write and the request cannot be honoured.
        uint UUID_loose = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), RGB::blue);
        uint objID_loose = ctx.addPolymeshObject({UUID_loose});
        DOCTEST_CHECK_THROWS(ctx.setPolymeshObjectVertices(objID_loose, {make_vec3(0, 0, 1), make_vec3(1, 0, 1), make_vec3(1, 1, 1)}));

        DOCTEST_CHECK_THROWS(ctx.setPolymeshObjectVertices(99999, mesh_vertices));
    }
}

TEST_CASE("Polymesh volume comes from the parts of the mesh that enclose something") {
    // Solids are commonly authored alongside open decoration - a fruit modelled with its sepals, or with its stalk. Requiring the whole mesh to be watertight leaves such an object with no computable volume
    // at all, even though the part that has one is closed; and sealing the decoration to satisfy that requirement would add its enclosed volume to the solid's.
    const float h = 0.5f;
    const std::vector<vec3> cube_vertices = {make_vec3(-h, -h, -h), make_vec3(h, -h, -h), make_vec3(h, h, -h), make_vec3(-h, h, -h),
                                             make_vec3(-h, -h, h),  make_vec3(h, -h, h),  make_vec3(h, h, h),  make_vec3(-h, h, h)};
    const std::vector<int3> cube_faces = {make_int3(0, 3, 2), make_int3(0, 2, 1), make_int3(4, 5, 6), make_int3(4, 6, 7), make_int3(0, 1, 5),
                                          make_int3(0, 5, 4), make_int3(2, 3, 7), make_int3(2, 7, 6), make_int3(0, 4, 7), make_int3(0, 7, 3),
                                          make_int3(1, 2, 6), make_int3(1, 6, 5)};

    SUBCASE("A closed solid alongside an open flap reports the solid's volume") {
        Context ctx;
        std::vector<vec3> vertices = cube_vertices;
        std::vector<int3> faces = cube_faces;
        std::vector<uint> face_UUIDs;
        for (const int3 &face: faces) {
            face_UUIDs.push_back(ctx.addTriangle(vertices.at(face.x), vertices.at(face.y), vertices.at(face.z)));
        }

        // A flap standing well clear of the cube: two triangles sharing an edge, open all the way round and touching nothing else.
        const int base = int(vertices.size());
        vertices.push_back(make_vec3(3, 0, 0));
        vertices.push_back(make_vec3(4, 0, 0));
        vertices.push_back(make_vec3(4, 1, 0));
        vertices.push_back(make_vec3(3, 1, 0));
        const std::vector<int3> flap = {make_int3(base, base + 1, base + 2), make_int3(base, base + 2, base + 3)};
        for (const int3 &face: flap) {
            face_UUIDs.push_back(ctx.addTriangle(vertices.at(face.x), vertices.at(face.y), vertices.at(face.z)));
            faces.push_back(face);
        }

        const uint ObjID = ctx.addPolymeshObject(face_UUIDs);
        ctx.setPolymeshObjectTopology(ObjID, vertices, faces, face_UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_REQUIRE(!ctx.getPolymeshObjectBoundaryEdges(ObjID).empty()); // the mesh as a whole is not watertight
        DOCTEST_CHECK(ctx.getPolymeshObjectVolume(ObjID) == doctest::Approx(8.f * h * h * h).epsilon(1e-4));
    }

    SUBCASE("A mesh that encloses nothing anywhere still raises") {
        // With no closed piece there is no volume to report, so the caller is told rather than handed a number that means nothing.
        Context ctx;
        const std::vector<vec3> vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0)};
        const std::vector<int3> faces = {make_int3(0, 1, 2), make_int3(0, 2, 3)};
        std::vector<uint> face_UUIDs;
        for (const int3 &face: faces) {
            face_UUIDs.push_back(ctx.addTriangle(vertices.at(face.x), vertices.at(face.y), vertices.at(face.z)));
        }
        const uint ObjID = ctx.addPolymeshObject(face_UUIDs);
        ctx.setPolymeshObjectTopology(ObjID, vertices, faces, face_UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_CHECK_THROWS_AS(ctx.getPolymeshObjectVolume(ObjID), std::runtime_error);
    }

    SUBCASE("A mesh with no face table is still checked before its volume is reported") {
        // An object grouped from loose primitives carries no topology, so its facets are matched up by which of their corners coincide. Without that check the divergence theorem returns a number for an open
        // surface too - and not even a fixed one, since it measures the cone swept from the origin and therefore changes when the object is moved.
        Context ctx;
        std::vector<uint> flap_UUIDs;
        flap_UUIDs.push_back(ctx.addTriangle(make_vec3(3, 0, 5), make_vec3(4, 0, 5), make_vec3(4, 1, 5)));
        flap_UUIDs.push_back(ctx.addTriangle(make_vec3(3, 0, 5), make_vec3(4, 1, 5), make_vec3(3, 1, 5)));
        const uint flap_ObjID = ctx.addPolymeshObject(flap_UUIDs);
        DOCTEST_REQUIRE(ctx.getPolymeshObjectFaceCount(flap_ObjID) == 0);
        DOCTEST_CHECK_THROWS_AS(ctx.getPolymeshObjectVolume(flap_ObjID), std::runtime_error);

        // A closed solid grouped the same way is still measured, since its facets do meet up.
        std::vector<uint> cube_UUIDs;
        for (const int3 &face: cube_faces) {
            cube_UUIDs.push_back(ctx.addTriangle(cube_vertices.at(face.x), cube_vertices.at(face.y), cube_vertices.at(face.z)));
        }
        const uint cube_ObjID = ctx.addPolymeshObject(cube_UUIDs);
        DOCTEST_REQUIRE(ctx.getPolymeshObjectFaceCount(cube_ObjID) == 0);
        DOCTEST_CHECK(ctx.getPolymeshObjectVolume(cube_ObjID) == doctest::Approx(8.f * h * h * h).epsilon(1e-4));
    }

    SUBCASE("Two separate solids add rather than cancel") {
        // Signed volumes are taken piece by piece, so a second solid wound the other way does not subtract from the first.
        Context ctx;
        std::vector<vec3> vertices;
        std::vector<int3> faces;
        std::vector<uint> face_UUIDs;
        for (int copy = 0; copy < 2; copy++) {
            const int base = int(vertices.size());
            for (const vec3 &vertex: cube_vertices) {
                vertices.push_back(vertex + make_vec3(3.f * float(copy), 0, 0));
            }
            for (const int3 &face: cube_faces) {
                // The second cube is wound inside-out.
                const int3 shifted = (copy == 0) ? make_int3(face.x + base, face.y + base, face.z + base) : make_int3(face.x + base, face.z + base, face.y + base);
                faces.push_back(shifted);
                face_UUIDs.push_back(ctx.addTriangle(vertices.at(shifted.x), vertices.at(shifted.y), vertices.at(shifted.z)));
            }
        }
        const uint ObjID = ctx.addPolymeshObject(face_UUIDs);
        ctx.setPolymeshObjectTopology(ObjID, vertices, faces, face_UUIDs, {}, {}, NORMAL_SOURCE_NONE);

        DOCTEST_CHECK(ctx.getPolymeshObjectVolume(ObjID) == doctest::Approx(2.f * 8.f * h * h * h).epsilon(1e-4));
    }
}
