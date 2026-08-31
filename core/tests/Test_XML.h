#pragma once
// =================================================================================
// Suite 3: XML Parsing Utilities
//
// Tests for functions related to loading data from XML files and nodes.
// =================================================================================
TEST_CASE("XMLoad functions") {
    pugi::xml_document doc;
    pugi::xml_node node = doc.append_child("node");
    node.append_child("float").text().set("1.23");
    node.append_child("int").text().set("123");
    node.append_child("string").text().set("hello");
    node.append_child("vec2").text().set("1.1 2.2");
    node.append_child("vec3").text().set("1.1 2.2 3.3");
    node.append_child("vec4").text().set("1.1 2.2 3.3 4.4");
    node.append_child("int2").text().set("1 2");
    node.append_child("int3").text().set("1 2 3");
    node.append_child("int4").text().set("1 2 3 4");
    node.append_child("rgb").text().set("0.1 0.2 0.3");
    node.append_child("rgba").text().set("0.1 0.2 0.3 0.4");

    SUBCASE("XMLloadfloat") {
        float f = XMLloadfloat(node, "float");
        DOCTEST_CHECK(f == doctest::Approx(1.23f));
    }
    SUBCASE("XMLloadint") {
        int i = XMLloadint(node, "int");
        DOCTEST_CHECK(i == 123);
    }
    SUBCASE("XMLloadstring") {
        std::string s = XMLloadstring(node, "string");
        DOCTEST_CHECK(s == "hello");
    }
    SUBCASE("XMLloadvec2") {
        vec2 v = XMLloadvec2(node, "vec2");
        DOCTEST_CHECK(v == vec2(1.1f, 2.2f));
    }
    SUBCASE("XMLloadvec3") {
        vec3 v = XMLloadvec3(node, "vec3");
        DOCTEST_CHECK(v == vec3(1.1f, 2.2f, 3.3f));
    }
    SUBCASE("XMLloadvec4") {
        vec4 v = XMLloadvec4(node, "vec4");
        DOCTEST_CHECK(v == vec4(1.1f, 2.2f, 3.3f, 4.4f));
    }
    SUBCASE("XMLloadint2") {
        int2 i = XMLloadint2(node, "int2");
        DOCTEST_CHECK(i == int2(1, 2));
    }
    SUBCASE("XMLloadint3") {
        int3 i = XMLloadint3(node, "int3");
        DOCTEST_CHECK(i == int3(1, 2, 3));
    }
    SUBCASE("XMLloadint4") {
        int4 i = XMLloadint4(node, "int4");
        DOCTEST_CHECK(i == int4(1, 2, 3, 4));
    }
    SUBCASE("XMLoad* for missing nodes") {
        pugi::xml_document doc;
        pugi::xml_node node = doc.append_child("node");
        // The XMLoad functions throw when the node is not found, so we check for that.
        capture_cerr cerr_buffer;
        DOCTEST_CHECK(XMLloadfloat(node, "non_existent") == 99999);
        DOCTEST_CHECK(XMLloadint(node, "non_existent") == 99999);
        DOCTEST_CHECK(XMLloadstring(node, "non_existent") == "99999");
        DOCTEST_CHECK(XMLloadvec2(node, "non_existent") == vec2(99999, 99999));
        DOCTEST_CHECK(XMLloadvec3(node, "non_existent") == vec3(99999, 99999, 99999));
        DOCTEST_CHECK(XMLloadvec4(node, "non_existent") == vec4(99999, 99999, 99999, 99999));
        DOCTEST_CHECK(XMLloadint2(node, "non_existent") == int2(99999, 99999));
        DOCTEST_CHECK(XMLloadint3(node, "non_existent") == int3(99999, 99999, 99999));
        DOCTEST_CHECK(XMLloadint4(node, "non_existent") == int4(99999, 99999, 99999, 99999));
        DOCTEST_CHECK(XMLloadrgb(node, "non_existent") == RGBcolor(99999, 99999, 99999));
        DOCTEST_CHECK(XMLloadrgba(node, "non_existent") == RGBAcolor(99999, 99999, 99999, 99999));
    }
    SUBCASE("XMLoad* for missing fields") {
        pugi::xml_document doc;
        pugi::xml_node node = doc.append_child("tag");
        DOCTEST_CHECK(XMLloadfloat(node, "non_existent") == 99999);
        DOCTEST_CHECK(XMLloadint(node, "non_existent") == 99999);
        DOCTEST_CHECK(XMLloadstring(node, "non_existent") == "99999");
        DOCTEST_CHECK(XMLloadvec2(node, "non_existent") == make_vec2(99999, 99999));
        DOCTEST_CHECK(XMLloadvec3(node, "non_existent") == make_vec3(99999, 99999, 99999));
    }

    SUBCASE("parse_xml_tag_*") {
        pugi::xml_document doc;
        pugi::xml_node node = doc.append_child("tag");
        node.text().set(" 123 ");
        int i = parse_xml_tag_int(node, "tag", "test");
        DOCTEST_CHECK(i == 123);
        node.text().set(" 1.23 ");
        float f = parse_xml_tag_float(node, "tag", "test");
        DOCTEST_CHECK(f == doctest::Approx(1.23f));
        node.text().set(" 1.1 2.2 ");
        vec2 v2 = parse_xml_tag_vec2(node, "tag", "test");
        DOCTEST_CHECK(v2 == vec2(1.1f, 2.2f));
        node.text().set(" 1.1 2.2 3.3 ");
        vec3 v3 = parse_xml_tag_vec3(node, "tag", "test");
        DOCTEST_CHECK(v3 == vec3(1.1f, 2.2f, 3.3f));
        node.text().set(" hello ");
        std::string s = parse_xml_tag_string(node, "tag", "test");
        DOCTEST_CHECK(s == "hello");

        capture_cerr cerr_buffer;
        node.text().set("abc");
        int result_int;
        DOCTEST_CHECK_THROWS(result_int = parse_xml_tag_int(node, "tag", "test"));
        float result_float;
        DOCTEST_CHECK_THROWS(result_float = parse_xml_tag_float(node, "tag", "test"));
        vec2 result_vec2;
        DOCTEST_CHECK_THROWS(result_vec2 = parse_xml_tag_vec2(node, "tag", "test"));
        vec3 result_vec3;
        DOCTEST_CHECK_THROWS(result_vec3 = parse_xml_tag_vec3(node, "tag", "test"));
    }

    SUBCASE("open_xml_file") {
        pugi::xml_document doc;
        std::string error_string;
        DOCTEST_CHECK(!open_xml_file("non_existent.xml", doc, error_string));
        DOCTEST_CHECK(!error_string.empty());

        // Create a dummy invalid xml file
        std::ofstream outfile("invalid.xml");
        outfile << "<helios><data>blah</helios>";
        outfile.close();
        DOCTEST_CHECK(!open_xml_file("invalid.xml", doc, error_string));

        // Create a dummy valid xml file
        outfile.open("valid.xml");
        outfile << "<helios><data>blah</data></helios>";
        outfile.close();
        DOCTEST_CHECK(open_xml_file("valid.xml", doc, error_string));
        std::remove("invalid.xml");
        std::remove("valid.xml");
    }
}

TEST_CASE("Context XML I/O Functions") {
    SUBCASE("writeXML and loadXML") {
        Context ctx;

        // Create some test data
        uint patch = ctx.addPatch(make_vec3(1, 2, 3), make_vec2(2, 4), nullrotation, RGB::red);
        uint triangle = ctx.addTriangle(make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(0, 1, 0), RGB::blue);
        uint box = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        // Add some data
        ctx.setPrimitiveData(patch, "test_data", 42.0f);
        ctx.setObjectData(box, "object_data", "test_string");
        ctx.setGlobalData("global_test", 123);

        // Write to XML (use quiet parameter to avoid output)
        const char *test_file = "helios_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        // Create new context and load (use quiet parameter)
        Context ctx2;
        std::vector<uint> loaded_uuids;
        DOCTEST_CHECK_NOTHROW(loaded_uuids = ctx2.loadXML(test_file, true));
        DOCTEST_CHECK(loaded_uuids.size() >= 2);

        // Verify loaded data
        DOCTEST_CHECK(ctx2.getPrimitiveCount() >= 2);
        DOCTEST_CHECK(ctx2.getObjectCount() >= 1);

        // Check primitive data was loaded
        bool found_data = false;
        for (uint uuid: loaded_uuids) {
            if (ctx2.doesPrimitiveDataExist(uuid, "test_data")) {
                float data;
                ctx2.getPrimitiveData(uuid, "test_data", data);
                DOCTEST_CHECK(data == doctest::Approx(42.0f));
                found_data = true;
                break;
            }
        }
        DOCTEST_CHECK(found_data);

        // Check global data
        int global_val;
        ctx2.getGlobalData("global_test", global_val);
        DOCTEST_CHECK(global_val == 123);

        // Clean up
        std::remove(test_file);
    }

    SUBCASE("writeXML and loadXML round-trip int4 and vector-valued data") {
        // Regressions: (1) writeXML wrote int4 data with mismatched tags (<data_int3 ... </data_int4>),
        // producing malformed XML that could not be reloaded; (2) object member primitive data arrays were
        // written with no separator between consecutive elements, silently corrupting vector-valued data.
        Context ctx;

        uint patch = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        ctx.setPrimitiveData(patch, "prim_int4", make_int4(1, -2, 3, -4));
        ctx.setGlobalData("glob_int4", make_int4(5, 6, 7, 8));

        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, make_int2(2, 2));
        std::vector<uint> tileUUIDs = ctx.getObjectPrimitiveUUIDs(tile);
        std::vector<float> float_array{1.5f, 2.5f, -3.5f};
        ctx.setPrimitiveData(tileUUIDs.front(), "member_float_array", float_array);
        std::vector<int2> int2_array{make_int2(1, 2), make_int2(3, 4)};
        ctx.setPrimitiveData(tileUUIDs.front(), "member_int2_array", int2_array);

        const char *test_file = "helios_test_int4_roundtrip.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        bool found_int4 = false;
        bool found_arrays = false;
        for (uint uuid: ctx2.getAllUUIDs()) {
            if (ctx2.doesPrimitiveDataExist(uuid, "prim_int4")) {
                int4 v;
                ctx2.getPrimitiveData(uuid, "prim_int4", v);
                DOCTEST_CHECK(v == make_int4(1, -2, 3, -4));
                found_int4 = true;
            }
            if (ctx2.doesPrimitiveDataExist(uuid, "member_float_array")) {
                std::vector<float> vf;
                ctx2.getPrimitiveData(uuid, "member_float_array", vf);
                DOCTEST_CHECK(vf.size() == 3);
                if (vf.size() == 3) {
                    DOCTEST_CHECK(vf.at(0) == doctest::Approx(1.5f));
                    DOCTEST_CHECK(vf.at(1) == doctest::Approx(2.5f));
                    DOCTEST_CHECK(vf.at(2) == doctest::Approx(-3.5f));
                }
                std::vector<int2> vi;
                ctx2.getPrimitiveData(uuid, "member_int2_array", vi);
                DOCTEST_CHECK(vi.size() == 2);
                if (vi.size() == 2) {
                    DOCTEST_CHECK(vi.at(0) == make_int2(1, 2));
                    DOCTEST_CHECK(vi.at(1) == make_int2(3, 4));
                }
                found_arrays = true;
            }
        }
        DOCTEST_CHECK(found_int4);
        DOCTEST_CHECK(found_arrays);

        int4 gv;
        ctx2.getGlobalData("glob_int4", gv);
        DOCTEST_CHECK(gv == make_int4(5, 6, 7, 8));

        std::remove(test_file);
    }

    SUBCASE("writeXML and loadXML round-trip voxels") {
        // Regression: the voxel branch of loadXML constructed the voxel with a size of
        // make_vec3(0,0,0), which addVoxel() rejects with "Voxel has size of zero". Any file
        // containing a <voxel> element therefore threw on load, so a scene containing a voxel
        // could be written by writeXML() but never read back.
        Context ctx;

        vec3 voxel_center = make_vec3(1.f, 2.f, 3.f);
        vec3 voxel_size = make_vec3(0.5f, 1.5f, 2.5f);
        uint voxel = ctx.addVoxel(voxel_center, voxel_size);
        ctx.setPrimitiveData(voxel, "voxel_label", 7);

        const char *test_file = "helios_test_voxel_roundtrip.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        std::vector<uint> loaded = ctx2.getAllUUIDs();
        DOCTEST_REQUIRE(loaded.size() == 1);
        DOCTEST_CHECK(ctx2.getPrimitiveType(loaded.front()) == PRIMITIVE_TYPE_VOXEL);

        // Geometry must survive the round-trip, not merely load without throwing.
        vec3 loaded_center = ctx2.getVoxelCenter(loaded.front());
        vec3 loaded_size = ctx2.getVoxelSize(loaded.front());
        DOCTEST_CHECK(loaded_center.x == doctest::Approx(voxel_center.x));
        DOCTEST_CHECK(loaded_center.y == doctest::Approx(voxel_center.y));
        DOCTEST_CHECK(loaded_center.z == doctest::Approx(voxel_center.z));
        DOCTEST_CHECK(loaded_size.x == doctest::Approx(voxel_size.x));
        DOCTEST_CHECK(loaded_size.y == doctest::Approx(voxel_size.y));
        DOCTEST_CHECK(loaded_size.z == doctest::Approx(voxel_size.z));

        int label = 0;
        DOCTEST_CHECK_NOTHROW(ctx2.getPrimitiveData(loaded.front(), "voxel_label", label));
        DOCTEST_CHECK(label == 7);

        std::remove(test_file);
    }

    SUBCASE("loadXML polymesh reload") {
        // Regression: loading a file containing a polymesh object into a Context where that object ID was
        // already in use remapped the ID before the primitive-map lookup and crashed with std::out_of_range.
        Context ctx;
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint p2 = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));
        uint poly = ctx.addPolymeshObject({p1, p2});
        DOCTEST_CHECK(ctx.doesObjectExist(poly));

        const char *test_file = "helios_test_polymesh_reload.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));
        DOCTEST_CHECK(ctx2.getObjectCount() == 2);
        DOCTEST_CHECK(ctx2.getPrimitiveCount() == 4);

        std::remove(test_file);
    }

    SUBCASE("loadXML does not remap the file-scoped object ID key") {
        // Regression: the <sphere>/<tube>/<box>/<disk>/<cone> loops in loadXML() reassigned objID when it
        // collided with a Context object ID already handed out earlier in the same load. objID is only ever
        // the key into the file-scoped map of primitives belonging to each object, so the remap made the
        // object claim a *different* object's primitive list. Here the tube is loaded first (the tube loop
        // precedes the cone loop) and takes Context ID 1, so the cone -- whose file objID is 1 -- collided
        // and stole the tube's primitives, leaving one object instead of two.
        Context ctx;
        uint cone = ctx.addConeObject(5, make_vec3(0, 0, 0), make_vec3(0, 0, 1), 0.1f, 0.05f);
        uint tube = ctx.addTubeObject(5, {make_vec3(1, 0, 0), make_vec3(1, 0, 1)}, {0.1f, 0.05f});
        DOCTEST_CHECK(ctx.getObjectType(cone) == OBJECT_TYPE_CONE);
        DOCTEST_CHECK(ctx.getObjectType(tube) == OBJECT_TYPE_TUBE);

        std::vector<uint> source_objects = ctx.getAllObjectIDs();
        DOCTEST_CHECK(source_objects.size() == 2);

        const char *test_file = "helios_objid_remap_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        std::vector<uint> loaded_objects = ctx2.getAllObjectIDs();
        DOCTEST_CHECK(loaded_objects.size() == source_objects.size());
        DOCTEST_CHECK(ctx2.getPrimitiveCount() == ctx.getPrimitiveCount());

        // Each object must come back with the same type and the same number of sub-primitives. The loaded
        // object IDs are not the source IDs, so the comparison is over the sorted (type, count) signature.
        std::vector<std::pair<int, size_t>> source_signature;
        for (uint objID: source_objects) {
            source_signature.emplace_back(static_cast<int>(ctx.getObjectType(objID)), ctx.getObjectPrimitiveUUIDs(objID).size());
        }
        std::vector<std::pair<int, size_t>> loaded_signature;
        for (uint objID: loaded_objects) {
            loaded_signature.emplace_back(static_cast<int>(ctx2.getObjectType(objID)), ctx2.getObjectPrimitiveUUIDs(objID).size());
        }
        std::sort(source_signature.begin(), source_signature.end());
        std::sort(loaded_signature.begin(), loaded_signature.end());
        DOCTEST_CHECK(loaded_signature == source_signature);

        // Type and count alone cannot distinguish two objects that swapped primitive lists, so also check that
        // each object came back at the position it was written from. Each object type appears once here, so the
        // type is enough to pair a loaded object with its source.
        std::map<int, vec3> source_centers;
        for (uint objID: source_objects) {
            source_centers[static_cast<int>(ctx.getObjectType(objID))] = ctx.getObjectCenter(objID);
        }
        for (uint objID: loaded_objects) {
            int type = static_cast<int>(ctx2.getObjectType(objID));
            DOCTEST_REQUIRE(source_centers.find(type) != source_centers.end());
            vec3 loaded_center = ctx2.getObjectCenter(objID);
            vec3 source_center = source_centers.at(type);
            DOCTEST_CHECK(loaded_center.x == doctest::Approx(source_center.x).epsilon(1e-4));
            DOCTEST_CHECK(loaded_center.y == doctest::Approx(source_center.y).epsilon(1e-4));
            DOCTEST_CHECK(loaded_center.z == doctest::Approx(source_center.z).epsilon(1e-4));
        }

        // Ownership must also be self-consistent: every sub-primitive an object lists must point back at it.
        size_t mismatched_parents = 0;
        for (uint objID: loaded_objects) {
            for (uint UUID: ctx2.getObjectPrimitiveUUIDs(objID)) {
                if (ctx2.getPrimitiveParentObjectID(UUID) != objID) {
                    mismatched_parents++;
                }
            }
        }
        DOCTEST_CHECK(mismatched_parents == 0);

        std::remove(test_file);
    }

    SUBCASE("loadXML preserves object ownership across a mixed-type scene") {
        // Same regression as above, exercised across every loop that carried the remap. loadXML() creates
        // objects grouped by type (tile, adaptive tile, sphere, tube, box, disk, cone, polymesh) while file
        // objIDs run in the writing Context's creation order, so the two orderings disagree and file objIDs
        // collide with Context IDs already assigned during the same load. The primitive count survives the
        // corruption intact -- the stolen primitives are simply reparented -- so it is the object count and
        // the per-object type/size signature that reveal it.
        Context ctx;
        uint patch1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint patch2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
        uint polymesh = ctx.addPolymeshObject({patch1, patch2});
        uint cone = ctx.addConeObject(6, make_vec3(0, 3, 0), make_vec3(0, 3, 1), 0.1f, 0.05f);
        uint disk = ctx.addDiskObject(7, make_vec3(0, 6, 0), make_vec2(0.5f, 0.5f));
        uint box = ctx.addBoxObject(make_vec3(0, 9, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint tube = ctx.addTubeObject(8, {make_vec3(0, 12, 0), make_vec3(0, 12, 1)}, {0.1f, 0.05f});
        uint sphere = ctx.addSphereObject(9, make_vec3(0, 15, 0), 0.5f);
        uint tile = ctx.addTileObject(make_vec3(0, 18, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));

        DOCTEST_CHECK(ctx.getObjectType(polymesh) == OBJECT_TYPE_POLYMESH);
        DOCTEST_CHECK(ctx.getObjectType(cone) == OBJECT_TYPE_CONE);
        DOCTEST_CHECK(ctx.getObjectType(disk) == OBJECT_TYPE_DISK);
        DOCTEST_CHECK(ctx.getObjectType(box) == OBJECT_TYPE_BOX);
        DOCTEST_CHECK(ctx.getObjectType(tube) == OBJECT_TYPE_TUBE);
        DOCTEST_CHECK(ctx.getObjectType(sphere) == OBJECT_TYPE_SPHERE);
        DOCTEST_CHECK(ctx.getObjectType(tile) == OBJECT_TYPE_TILE);

        std::vector<uint> source_objects = ctx.getAllObjectIDs();
        DOCTEST_CHECK(source_objects.size() == 7);

        std::vector<std::pair<int, size_t>> source_signature;
        std::map<int, vec3> source_centers;
        for (uint objID: source_objects) {
            source_signature.emplace_back(static_cast<int>(ctx.getObjectType(objID)), ctx.getObjectPrimitiveUUIDs(objID).size());
            source_centers[static_cast<int>(ctx.getObjectType(objID))] = ctx.getObjectCenter(objID);
        }
        std::sort(source_signature.begin(), source_signature.end());

        const char *test_file = "helios_objid_remap_mixed_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        std::vector<uint> loaded_objects = ctx2.getAllObjectIDs();
        DOCTEST_CHECK(loaded_objects.size() == source_objects.size());
        DOCTEST_CHECK(ctx2.getPrimitiveCount() == ctx.getPrimitiveCount());

        std::vector<std::pair<int, size_t>> loaded_signature;
        for (uint objID: loaded_objects) {
            loaded_signature.emplace_back(static_cast<int>(ctx2.getObjectType(objID)), ctx2.getObjectPrimitiveUUIDs(objID).size());
        }
        std::sort(loaded_signature.begin(), loaded_signature.end());
        DOCTEST_CHECK(loaded_signature == source_signature);

        // Type and count alone cannot distinguish two objects that swapped primitive lists. The objects are built
        // at distinct, well-separated y-coordinates precisely so that a swap moves an object's center; each type
        // appears once, so the type pairs a loaded object with its source.
        for (uint objID: loaded_objects) {
            int type = static_cast<int>(ctx2.getObjectType(objID));
            DOCTEST_REQUIRE(source_centers.find(type) != source_centers.end());
            vec3 loaded_center = ctx2.getObjectCenter(objID);
            vec3 source_center = source_centers.at(type);
            DOCTEST_CHECK(loaded_center.x == doctest::Approx(source_center.x).epsilon(1e-4));
            DOCTEST_CHECK(loaded_center.y == doctest::Approx(source_center.y).epsilon(1e-4));
            DOCTEST_CHECK(loaded_center.z == doctest::Approx(source_center.z).epsilon(1e-4));
        }

        // Ownership must also be self-consistent. Note this is not the assertion that catches the remap:
        // setPrimitiveParentObjectID() detaches each primitive from its previous parent and destroys that
        // object once it is left empty, so the corrupted Context is internally consistent -- it is simply
        // missing objects, which the count and signature checks above are what detect.
        size_t mismatched_parents = 0;
        for (uint objID: loaded_objects) {
            for (uint UUID: ctx2.getObjectPrimitiveUUIDs(objID)) {
                if (ctx2.getPrimitiveParentObjectID(UUID) != objID) {
                    mismatched_parents++;
                }
            }
        }
        DOCTEST_CHECK(mismatched_parents == 0);

        std::remove(test_file);
    }

    SUBCASE("loadXML rejects a file in which two object blocks claim the same object ID") {
        // An objID in the file identifies exactly one object. If two object blocks claim the same one, both are
        // handed the same primitive list and reparenting them to the second silently destroys the first -- the
        // same symptom as the objID-remap bug above, reachable through a hand-edited file or by concatenating
        // the bodies of two writeXML() outputs to merge two scenes (both number their objects from 1).
        Context ctx;
        uint sphere1 = ctx.addSphereObject(4, make_vec3(0, 0, 0), 0.5f);
        uint sphere2 = ctx.addSphereObject(4, make_vec3(0, 5, 0), 0.5f);
        DOCTEST_CHECK(ctx.getAllObjectIDs().size() == 2);

        const char *source_file = "helios_dup_objid_source.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(source_file, true));

        // Point the second <sphere> block at the first sphere's objID. writeXML emits every primitive block before
        // the object blocks, so the final <objID> tag naming sphere2 is the one inside its object block.
        std::string xml;
        {
            std::ifstream in(source_file);
            std::stringstream buffer;
            buffer << in.rdbuf();
            xml = buffer.str();
        }
        std::string second_tag = "<objID>" + std::to_string(sphere2) + "</objID>";
        std::string first_tag = "<objID>" + std::to_string(sphere1) + "</objID>";
        size_t object_block_tag = xml.rfind(second_tag);
        DOCTEST_REQUIRE(object_block_tag != std::string::npos);
        DOCTEST_REQUIRE(object_block_tag > xml.rfind("<sphere>"));
        xml.replace(object_block_tag, second_tag.size(), first_tag);

        const char *duplicate_file = "helios_dup_objid.xml";
        {
            std::ofstream out(duplicate_file);
            out << xml;
        }

        // Must fail fast naming the duplicated objID, rather than silently returning a Context short one object.
        Context ctx2;
        std::string message;
        bool threw = false;
        try {
            ctx2.loadXML(duplicate_file, true);
        } catch (const std::runtime_error &e) {
            threw = true;
            message = e.what();
        }
        DOCTEST_CHECK(threw);
        DOCTEST_CHECK(message.find(std::to_string(sphere1)) != std::string::npos);

        std::remove(source_file);
        std::remove(duplicate_file);
    }
    SUBCASE("writeXML and loadXML round-trip sparse object sub-primitive data") {
        // Regression: writeXML() records each datum's position within the object's sub-primitive list in the
        // <data> label attribute and omits sub-primitives that do not carry the label, so the data is sparse.
        // loadOsubPData() ignored that index and consumed the entries positionally, so data set on only some
        // sub-primitives was reassigned to the first N of them -- and because it runs after the per-primitive
        // blocks have already restored the data correctly, it overwrote correct values with shifted ones.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));
        std::vector<uint> sub = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_REQUIRE(sub.size() == 4);

        // Only sub-patches 1 and 3 carry the label.
        ctx.setPrimitiveData(sub.at(1), "sparse", 300);
        ctx.setPrimitiveData(sub.at(3), "sparse", 700);

        const char *test_file = "helios_osubp_sparse_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));
        std::vector<uint> loaded_objects = ctx2.getAllObjectIDs();
        DOCTEST_REQUIRE(loaded_objects.size() == 1);
        std::vector<uint> loaded_sub = ctx2.getObjectPrimitiveUUIDs(loaded_objects.at(0));
        DOCTEST_REQUIRE(loaded_sub.size() == 4);

        // The two sub-patches that carried no data must still carry none, and the two that did must keep
        // their own values rather than having them shifted onto their neighbours.
        DOCTEST_CHECK(!ctx2.doesPrimitiveDataExist(loaded_sub.at(0), "sparse"));
        DOCTEST_CHECK(!ctx2.doesPrimitiveDataExist(loaded_sub.at(2), "sparse"));
        DOCTEST_REQUIRE(ctx2.doesPrimitiveDataExist(loaded_sub.at(1), "sparse"));
        DOCTEST_REQUIRE(ctx2.doesPrimitiveDataExist(loaded_sub.at(3), "sparse"));
        int value1, value3;
        ctx2.getPrimitiveData(loaded_sub.at(1), "sparse", value1);
        ctx2.getPrimitiveData(loaded_sub.at(3), "sparse", value3);
        DOCTEST_CHECK(value1 == 300);
        DOCTEST_CHECK(value3 == 700);

        std::remove(test_file);
    }

    SUBCASE("writeXML and loadXML round-trip object sub-primitive data with a hidden sub-primitive") {
        // Regression: getAllUUIDs() omits hidden primitives, but the object's data block was written from the
        // object's full member list, so a tile with one hidden sub-patch wrote three <patch> blocks and four
        // <data> entries. The surplus entry shifted the values and tripped an 'osubpdata_length_mismatch'
        // warning -- and the shifted data was written anyway.
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));
        std::vector<uint> sub = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_REQUIRE(sub.size() == 4);
        for (size_t i = 0; i < sub.size(); i++) {
            ctx.setPrimitiveData(sub.at(i), "slot", static_cast<int>(10 * i));
        }
        ctx.hidePrimitive(sub.at(1));

        const char *test_file = "helios_osubp_hidden_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        std::string cerr_text;
        {
            capture_cerr capture;
            DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));
            cerr_text = capture.get_captured_output();
        }
        DOCTEST_CHECK(cerr_text.find("osubpdata_length_mismatch") == std::string::npos);

        std::vector<uint> loaded_objects = ctx2.getAllObjectIDs();
        DOCTEST_REQUIRE(loaded_objects.size() == 1);
        std::vector<uint> loaded_sub = ctx2.getObjectPrimitiveUUIDs(loaded_objects.at(0));
        DOCTEST_REQUIRE(loaded_sub.size() == 3);

        // The written sub-patches are the visible ones, in order: slots 0, 20 and 30.
        std::vector<int> expected = {0, 20, 30};
        for (size_t i = 0; i < loaded_sub.size(); i++) {
            DOCTEST_REQUIRE(ctx2.doesPrimitiveDataExist(loaded_sub.at(i), "slot"));
            int value;
            ctx2.getPrimitiveData(loaded_sub.at(i), "slot", value);
            DOCTEST_CHECK(value == expected.at(i));
        }

        std::remove(test_file);
    }
    SUBCASE("writeXML and loadXML round-trip polymesh topology") {
        // Regression: the OBJECT_TYPE_POLYMESH branch of writeXML() emitted nothing but the closing tag -- no
        // transformation matrix and none of the indexed face set -- so a mesh that had retained its connectivity
        // came back from a round-trip as a triangle soup: getPolymeshObjectFaceCount() returned 0,
        // getPolymeshObjectFaceIndexForPrimitive() threw, the mesh was no longer closed so its volume could not
        // be computed, and writePLY()/writeOBJ() fell back to independent per-triangle vertices.
        Context ctx;
        std::vector<vec3> cube_vertices = {make_vec3(0, 0, 0), make_vec3(1, 0, 0), make_vec3(1, 1, 0), make_vec3(0, 1, 0), make_vec3(0, 0, 1), make_vec3(1, 0, 1), make_vec3(1, 1, 1), make_vec3(0, 1, 1)};
        std::vector<int3> cube_faces = {make_int3(0, 3, 2), make_int3(0, 2, 1), make_int3(4, 5, 6), make_int3(4, 6, 7), make_int3(0, 1, 5), make_int3(0, 5, 4),
                                        make_int3(2, 3, 7), make_int3(2, 7, 6), make_int3(3, 0, 4), make_int3(3, 4, 7), make_int3(1, 2, 6), make_int3(1, 6, 5)};

        std::vector<uint> UUIDs;
        for (const int3 &face: cube_faces) {
            UUIDs.push_back(ctx.addTriangle(cube_vertices.at(face.x), cube_vertices.at(face.y), cube_vertices.at(face.z)));
        }
        uint polymesh = ctx.addPolymeshObject(UUIDs);

        // Authored per-vertex normals and texture coordinates exercise the optional halves of the face set.
        std::vector<vec3> vertex_normals;
        std::vector<vec2> vertex_uv;
        for (const vec3 &vertex: cube_vertices) {
            vec3 outward = vertex - make_vec3(0.5f, 0.5f, 0.5f);
            outward.normalize();
            vertex_normals.push_back(outward);
            vertex_uv.push_back(make_vec2(vertex.x, vertex.y));
        }
        ctx.setPolymeshObjectTopology(polymesh, cube_vertices, cube_faces, UUIDs, vertex_normals, vertex_uv, NORMAL_SOURCE_AUTHORED);

        DOCTEST_REQUIRE(ctx.getPolymeshObjectFaceCount(polymesh) == 12);
        DOCTEST_REQUIRE(ctx.isPolymeshObjectClosed(polymesh));

        const char *test_file = "helios_polymesh_topology_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));
        std::vector<uint> loaded_objects = ctx2.getAllObjectIDs();
        DOCTEST_REQUIRE(loaded_objects.size() == 1);
        uint loaded = loaded_objects.at(0);
        DOCTEST_REQUIRE(ctx2.getObjectType(loaded) == OBJECT_TYPE_POLYMESH);

        // The shared-vertex mesh must survive, not degrade to one independent vertex triple per triangle.
        DOCTEST_CHECK(ctx2.getPolymeshObjectVertexCount(loaded) == ctx.getPolymeshObjectVertexCount(polymesh));
        DOCTEST_CHECK(ctx2.getPolymeshObjectFaceCount(loaded) == ctx.getPolymeshObjectFaceCount(polymesh));
        DOCTEST_CHECK(ctx2.getPolymeshObjectFaces(loaded) == ctx.getPolymeshObjectFaces(polymesh));
        DOCTEST_CHECK(ctx2.isPolymeshObjectClosed(loaded));
        DOCTEST_CHECK(ctx2.getPolymeshObjectBoundaryEdges(loaded).empty());
        DOCTEST_CHECK(ctx2.getPolymeshObjectVolume(loaded) == doctest::Approx(ctx.getPolymeshObjectVolume(polymesh)).epsilon(1e-4));

        std::vector<vec3> loaded_vertices = ctx2.getPolymeshObjectVertices(loaded);
        DOCTEST_REQUIRE(loaded_vertices.size() == cube_vertices.size());
        for (size_t i = 0; i < loaded_vertices.size(); i++) {
            DOCTEST_CHECK(loaded_vertices.at(i).x == doctest::Approx(cube_vertices.at(i).x).epsilon(1e-5));
            DOCTEST_CHECK(loaded_vertices.at(i).y == doctest::Approx(cube_vertices.at(i).y).epsilon(1e-5));
            DOCTEST_CHECK(loaded_vertices.at(i).z == doctest::Approx(cube_vertices.at(i).z).epsilon(1e-5));
        }

        DOCTEST_CHECK(ctx2.getPolymeshObjectVertexNormalSource(loaded) == NORMAL_SOURCE_AUTHORED);
        std::vector<vec3> loaded_normals = ctx2.getPolymeshObjectVertexNormals(loaded);
        DOCTEST_REQUIRE(loaded_normals.size() == vertex_normals.size());
        for (size_t i = 0; i < loaded_normals.size(); i++) {
            DOCTEST_CHECK(loaded_normals.at(i).x == doctest::Approx(vertex_normals.at(i).x).epsilon(1e-5));
            DOCTEST_CHECK(loaded_normals.at(i).y == doctest::Approx(vertex_normals.at(i).y).epsilon(1e-5));
            DOCTEST_CHECK(loaded_normals.at(i).z == doctest::Approx(vertex_normals.at(i).z).epsilon(1e-5));
        }

        std::vector<vec2> loaded_uv = ctx2.getPolymeshObjectVertexUV(loaded);
        DOCTEST_REQUIRE(loaded_uv.size() == vertex_uv.size());
        for (size_t i = 0; i < loaded_uv.size(); i++) {
            DOCTEST_CHECK(loaded_uv.at(i).x == doctest::Approx(vertex_uv.at(i).x).epsilon(1e-5));
            DOCTEST_CHECK(loaded_uv.at(i).y == doctest::Approx(vertex_uv.at(i).y).epsilon(1e-5));
        }

        // Each face must still resolve to the member primitive it describes.
        std::vector<uint> loaded_sub = ctx2.getObjectPrimitiveUUIDs(loaded);
        DOCTEST_REQUIRE(loaded_sub.size() == UUIDs.size());
        for (size_t i = 0; i < loaded_sub.size(); i++) {
            DOCTEST_CHECK(ctx2.getPolymeshObjectFaceIndexForPrimitive(loaded, loaded_sub.at(i)) == ctx.getPolymeshObjectFaceIndexForPrimitive(polymesh, UUIDs.at(i)));
        }

        std::remove(test_file);
    }
    SUBCASE("loadXML rejects a file whose primitives reference an object block that is not present") {
        // Regression: primitives carrying an objID were added to the Context but pushed onto the vector that
        // loadXML() returns only when objID was 0. If no object block claimed that objID -- a truncated or
        // hand-edited file -- the primitives ended up in the scene with no parent object and no handle returned
        // to the caller: geometry that still renders and ray-traces but that the caller cannot address.
        Context ctx;
        uint sphere = ctx.addSphereObject(4, make_vec3(0, 0, 0), 0.5f);
        DOCTEST_CHECK(ctx.doesObjectExist(sphere));

        const char *source_file = "helios_orphan_objid_source.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(source_file, true));

        std::string xml;
        {
            std::ifstream in(source_file);
            std::stringstream buffer;
            buffer << in.rdbuf();
            xml = buffer.str();
        }

        // Strip the <sphere> object block, leaving its member primitives behind still tagged with its objID.
        size_t block_start = xml.find("<sphere>");
        size_t block_end = xml.find("</sphere>");
        DOCTEST_REQUIRE(block_start != std::string::npos);
        DOCTEST_REQUIRE(block_end != std::string::npos);
        xml.erase(block_start, block_end + std::string("</sphere>").size() - block_start);

        const char *orphan_file = "helios_orphan_objid.xml";
        {
            std::ofstream out(orphan_file);
            out << xml;
        }

        Context ctx2;
        std::string message;
        bool threw = false;
        try {
            ctx2.loadXML(orphan_file, true);
        } catch (const std::runtime_error &e) {
            threw = true;
            message = e.what();
        }
        DOCTEST_CHECK(threw);
        DOCTEST_CHECK(message.find(std::to_string(sphere)) != std::string::npos);

        std::remove(source_file);
        std::remove(orphan_file);
    }
    SUBCASE("loadXML keeps legacy numeric material IDs from different files apart") {
        // Regression: the v2 format identifies a material by a file-scoped number, which loadXML() mapped onto
        // the Context-global label "__auto_material_<N>". Two v2 files both number their materials from 1, so
        // loading a second one found the label already taken and mutated the existing material in place --
        // silently recoloring every primitive loaded from the first file. loadOBJ() already renames on
        // collision and keeps a remap; loadXML() now does the same.
        auto write_v2_file = [](const char *filename, const char *color) {
            std::ofstream out(filename);
            out << "<?xml version=\"1.0\"?>\n<helios>\n";
            out << "   <materials>\n";
            out << "\t<material id=\"1\">\n";
            out << "\t   <color>" << color << "</color>\n";
            out << "\t</material>\n";
            out << "   </materials>\n";
            out << "   <patch>\n";
            out << "\t<material_id>1</material_id>\n";
            out << "   </patch>\n";
            out << "</helios>\n";
        };

        const char *red_file = "helios_v2_material_red.xml";
        const char *blue_file = "helios_v2_material_blue.xml";
        write_v2_file(red_file, "1 0 0");
        write_v2_file(blue_file, "0 0 1");

        Context ctx;
        std::vector<uint> red_UUIDs;
        std::vector<uint> blue_UUIDs;
        DOCTEST_CHECK_NOTHROW(red_UUIDs = ctx.loadXML(red_file, true));
        DOCTEST_CHECK_NOTHROW(blue_UUIDs = ctx.loadXML(blue_file, true));
        DOCTEST_REQUIRE(red_UUIDs.size() == 1);
        DOCTEST_REQUIRE(blue_UUIDs.size() == 1);

        // The first file's patch must still be red; loading the second file must not have recolored it.
        RGBcolor red = ctx.getPrimitiveColor(red_UUIDs.front());
        RGBcolor blue = ctx.getPrimitiveColor(blue_UUIDs.front());
        DOCTEST_CHECK(red.r == doctest::Approx(1.f));
        DOCTEST_CHECK(red.b == doctest::Approx(0.f));
        DOCTEST_CHECK(blue.r == doctest::Approx(0.f));
        DOCTEST_CHECK(blue.b == doctest::Approx(1.f));

        std::remove(red_file);
        std::remove(blue_file);
    }
    SUBCASE("writeXML succeeds after primitive data is renamed on a member of an object") {
        // Regression: renaming or duplicating primitive data registered the new label on the primitive but not in
        // the Context-wide type registry, and writeXML() looks the type up by label for every label carried by an
        // object's members -- so writing a Context in which any object member's data had been renamed failed with
        // "Primitive data ... does not exist".
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2));
        std::vector<uint> sub = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_REQUIRE(sub.size() == 4);
        ctx.setPrimitiveData(sub, "flux", 1.f);
        ctx.renamePrimitiveData(sub.front(), "flux", "flux_renamed");

        const char *test_file = "helios_renamed_data_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        std::remove(test_file);
    }
    SUBCASE("writeXML and loadXML preserve texture UV coordinates") {
        // Regression test: textured primitives written with the v3 <material> format must
        // retain their texture (u,v) coordinates across a writeXML()/loadXML() round-trip.
        const char *texture = "lib/images/disk_texture.png";
        Context ctx;

        // Textured tile object (its sub-patches carry the UVs)
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(2, 2), nullrotation, make_int2(2, 2), texture);
        std::vector<uint> tile_uuids = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(tile_uuids.size() == 4);
        std::vector<std::vector<vec2>> tile_uv_before;
        for (uint uuid: tile_uuids) {
            std::vector<vec2> uv = ctx.getPrimitiveTextureUV(uuid);
            DOCTEST_CHECK(uv.size() == 4);
            tile_uv_before.push_back(uv);
        }

        // Standalone textured patch and triangle exercise the same material-format read branch
        vec2 uv0 = make_vec2(0.2f, 0.3f);
        vec2 uv2 = make_vec2(0.8f, 0.9f);
        uint patch = ctx.addPatch(make_vec3(3, 0, 0), make_vec2(1, 1), nullrotation, texture, 0.5f * (uv0 + uv2), uv2 - uv0);
        std::vector<vec2> patch_uv_before = ctx.getPrimitiveTextureUV(patch);
        DOCTEST_CHECK(patch_uv_before.size() == 4);

        uint triangle = ctx.addTriangle(make_vec3(5, 0, 0), make_vec3(6, 0, 0), make_vec3(6, 1, 0), texture, make_vec2(0, 0), make_vec2(1, 0), make_vec2(1, 1));
        std::vector<vec2> tri_uv_before = ctx.getPrimitiveTextureUV(triangle);
        DOCTEST_CHECK(tri_uv_before.size() == 3);

        const char *test_file = "helios_texture_uv_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        // Locate the reloaded tile object and verify its sub-patch UVs survived
        std::vector<uint> object_ids = ctx2.getAllObjectIDs();
        uint loaded_tile = 0;
        bool found_tile = false;
        for (uint objID: object_ids) {
            if (ctx2.getObjectType(objID) == OBJECT_TYPE_TILE) {
                loaded_tile = objID;
                found_tile = true;
                break;
            }
        }
        DOCTEST_CHECK(found_tile);

        std::vector<uint> loaded_tile_uuids = ctx2.getObjectPrimitiveUUIDs(loaded_tile);
        DOCTEST_CHECK(loaded_tile_uuids.size() == tile_uuids.size());

        // Each sub-patch must retain its 4 UVs and texture file, and the sub-patch ordering must be
        // preserved across the round-trip (writeXML() emits primitives in ascending UUID order, which
        // matches the tile's row-major sub-patch creation order).
        for (size_t i = 0; i < loaded_tile_uuids.size(); i++) {
            std::vector<vec2> uv = ctx2.getPrimitiveTextureUV(loaded_tile_uuids.at(i));
            DOCTEST_CHECK(uv.size() == 4);
            DOCTEST_CHECK(ctx2.getPrimitiveTextureFile(loaded_tile_uuids.at(i)) == std::string(texture));
            for (size_t k = 0; k < uv.size() && k < tile_uv_before.at(i).size(); k++) {
                DOCTEST_CHECK(uv.at(k).x == doctest::Approx(tile_uv_before.at(i).at(k).x));
                DOCTEST_CHECK(uv.at(k).y == doctest::Approx(tile_uv_before.at(i).at(k).y));
            }
        }

        // Verify the standalone patch and triangle UVs survived
        std::vector<uint> loaded_uuids = ctx2.getAllUUIDs();
        int checked_patch = 0;
        int checked_triangle = 0;
        for (uint uuid: loaded_uuids) {
            if (ctx2.getPrimitiveParentObjectID(uuid) != 0) {
                continue; // skip tile sub-patches already checked above
            }
            if (ctx2.getPrimitiveType(uuid) == PRIMITIVE_TYPE_PATCH) {
                std::vector<vec2> uv = ctx2.getPrimitiveTextureUV(uuid);
                DOCTEST_CHECK(uv.size() == 4);
                checked_patch++;
            } else if (ctx2.getPrimitiveType(uuid) == PRIMITIVE_TYPE_TRIANGLE) {
                std::vector<vec2> uv = ctx2.getPrimitiveTextureUV(uuid);
                DOCTEST_CHECK(uv.size() == 3);
                checked_triangle++;
            }
        }
        DOCTEST_CHECK(checked_patch == 1);
        DOCTEST_CHECK(checked_triangle == 1);

        std::remove(test_file);
    }

    SUBCASE("writeXML and loadXML preserve tile texture repeat") {
        // Regression test: the sub-patch (u,v) coordinates of a textured tile already survived a
        // round-trip, so a reloaded scene rendered correctly. What was lost was the repeat count itself,
        // which was not written at all and which the reader could not recover -- so changing the
        // subdivision count of a reloaded tile stretched the texture across the whole tile.
        const char *texture = "lib/images/disk_texture.png";
        const char *test_file = "helios_tile_texture_repeat_test.xml";

        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(10, 10), nullrotation, make_int2(10, 10), texture, make_int2(5, 5));
        DOCTEST_CHECK(ctx.getTileObjectTextureRepeat(tile) == make_int2(5, 5));
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        uint loaded_tile = 0;
        bool found = false;
        for (uint objID: ctx2.getAllObjectIDs()) {
            if (ctx2.getObjectType(objID) == OBJECT_TYPE_TILE) {
                loaded_tile = objID;
                found = true;
                break;
            }
        }
        DOCTEST_CHECK(found);
        DOCTEST_CHECK(ctx2.getTileObjectSubdivisionCount(loaded_tile) == make_int2(10, 10));
        DOCTEST_CHECK(ctx2.getTileObjectTextureRepeat(loaded_tile) == make_int2(5, 5));

        ctx2.setTileObjectSubdivisionCount({loaded_tile}, make_int2(20, 20));

        std::vector<uint> uuids = ctx2.getObjectPrimitiveUUIDs(loaded_tile);
        DOCTEST_CHECK(uuids.size() == 400);
        for (size_t k = 0; k < uuids.size(); k++) {
            const int i_local = static_cast<int>(k % 20) % 4;
            const int j_local = static_cast<int>(k / 20) % 4;
            std::vector<vec2> uv = ctx2.getPrimitiveTextureUV(uuids.at(k));
            DOCTEST_CHECK(uv.size() == 4);
            DOCTEST_CHECK(uv.at(0).x == doctest::Approx(float(i_local) * 0.25f).epsilon(errtol));
            DOCTEST_CHECK(uv.at(2).y == doctest::Approx(float(j_local + 1) * 0.25f).epsilon(errtol));
        }

        std::remove(test_file);
    }

    SUBCASE("writeXML and loadXML round-trip an adaptive tile") {
        // An adaptive tile is not regenerated from its parameters on load; the sub-patches written to the file are
        // adopted directly. Regenerating would be both wasteful and unreliable, since the refinement predicate uses
        // transcendental functions that are not identically rounded across platforms.
        const char *test_file = "helios_adaptive_tile_test.xml";

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(1.5f, -0.5f);
        refinement.subpatch_size_min = 0.25f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        Context ctx;
        uint tile = ctx.addAdaptiveTileObject(make_vec3(1, 2, 0), make_vec2(8, 8), nullrotation, refinement);

        const uint original_count = ctx.getObjectPrimitiveCount(tile);
        const float original_area = ctx.getObjectArea(tile);
        const int2 original_base_subdiv = ctx.getAdaptiveTileObjectBaseSubdivisionCount(tile);
        const uint original_max_level = ctx.getAdaptiveTileObjectMaxRefinementLevel(tile);
        const vec2 original_size_range = ctx.getAdaptiveTileObjectSubpatchSizeRange(tile);
        const vec3 original_normal = ctx.getAdaptiveTileObjectNormal(tile);

        // Deliberately dense rather than sparse. The writer emits a sub-patch data element only where the data exists
        // while the reader consumes them positionally, so sparse per-sub-patch data is shifted on reload. That is a
        // pre-existing defect affecting uniform tiles too, and testing around it here would hide it.
        std::vector<uint> original_UUIDs = ctx.getObjectPrimitiveUUIDs(tile);
        for (size_t k = 0; k < original_UUIDs.size(); k++) {
            ctx.setPrimitiveData(original_UUIDs.at(k), "cell_index", float(k));
        }

        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        uint loaded_tile = 0;
        bool found = false;
        for (uint objID: ctx2.getAllObjectIDs()) {
            if (ctx2.getObjectType(objID) == OBJECT_TYPE_ADAPTIVE_TILE) {
                loaded_tile = objID;
                found = true;
                break;
            }
        }
        DOCTEST_CHECK(found);

        DOCTEST_CHECK(ctx2.getObjectPrimitiveCount(loaded_tile) == original_count);
        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectBaseSubdivisionCount(loaded_tile) == original_base_subdiv);
        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectMaxRefinementLevel(loaded_tile) == original_max_level);
        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectSubpatchSizeRange(loaded_tile).x == doctest::Approx(original_size_range.x).epsilon(1e-4));
        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectSubpatchSizeRange(loaded_tile).y == doctest::Approx(original_size_range.y).epsilon(1e-4));

        const AdaptiveTileRefinement loaded_refinement = ctx2.getAdaptiveTileObjectRefinement(loaded_tile);
        DOCTEST_CHECK(loaded_refinement.target.x == doctest::Approx(refinement.target.x).epsilon(1e-4));
        DOCTEST_CHECK(loaded_refinement.target.y == doctest::Approx(refinement.target.y).epsilon(1e-4));
        DOCTEST_CHECK(loaded_refinement.subpatch_size_min == doctest::Approx(refinement.subpatch_size_min).epsilon(1e-4));
        DOCTEST_CHECK(loaded_refinement.subpatch_size_max == doctest::Approx(refinement.subpatch_size_max).epsilon(1e-4));
        DOCTEST_CHECK(loaded_refinement.transition_exponent == doctest::Approx(refinement.transition_exponent).epsilon(1e-4));

        // Coordinates are written at the default stream precision of six significant digits, so the sub-patch corners
        // are quantized on the way out and the partition is no longer exact. The exact area check belongs in
        // Test_context.h, before anything has been written to a file.
        DOCTEST_CHECK(ctx2.getObjectArea(loaded_tile) == doctest::Approx(original_area).epsilon(1e-3));

        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectCenter(loaded_tile).x == doctest::Approx(1.f).epsilon(1e-4));
        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectCenter(loaded_tile).y == doctest::Approx(2.f).epsilon(1e-4));
        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectSize(loaded_tile).x == doctest::Approx(8.f).epsilon(1e-4));

        const vec3 loaded_normal = ctx2.getAdaptiveTileObjectNormal(loaded_tile);
        DOCTEST_CHECK(loaded_normal.z == doctest::Approx(original_normal.z).epsilon(1e-4));

        std::vector<uint> loaded_UUIDs = ctx2.getObjectPrimitiveUUIDs(loaded_tile);
        DOCTEST_CHECK(loaded_UUIDs.size() == original_UUIDs.size());
        for (size_t k = 0; k < loaded_UUIDs.size(); k++) {
            float cell_index;
            DOCTEST_CHECK(ctx2.doesPrimitiveDataExist(loaded_UUIDs.at(k), "cell_index"));
            ctx2.getPrimitiveData(loaded_UUIDs.at(k), "cell_index", cell_index);
            DOCTEST_CHECK(cell_index == doctest::Approx(float(k)).epsilon(errtol));
        }

        std::remove(test_file);
    }

    SUBCASE("writeXML and loadXML preserve adaptive tile texture coordinates") {
        const char *texture = "lib/images/diamond_texture.png";
        const char *test_file = "helios_adaptive_tile_texture_test.xml";

        AdaptiveTileRefinement refinement;
        refinement.target = make_vec2(0.f, 0.f);
        refinement.subpatch_size_min = 0.5f;
        refinement.subpatch_size_max = 2.f;
        refinement.transition_exponent = 0.5f;

        Context ctx;
        uint tile = ctx.addAdaptiveTileObject(make_vec3(0, 0, 0), make_vec2(16, 16), nullrotation, refinement, texture, make_int2(2, 2));
        DOCTEST_CHECK(ctx.getAdaptiveTileObjectTextureRepeat(tile) == make_int2(2, 2));

        std::vector<std::vector<vec2>> original_uv;
        for (uint UUID: ctx.getObjectPrimitiveUUIDs(tile)) {
            original_uv.push_back(ctx.getPrimitiveTextureUV(UUID));
        }

        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        uint loaded_tile = 0;
        bool found = false;
        for (uint objID: ctx2.getAllObjectIDs()) {
            if (ctx2.getObjectType(objID) == OBJECT_TYPE_ADAPTIVE_TILE) {
                loaded_tile = objID;
                found = true;
                break;
            }
        }
        DOCTEST_CHECK(found);
        DOCTEST_CHECK(ctx2.getAdaptiveTileObjectTextureRepeat(loaded_tile) == make_int2(2, 2));

        std::vector<uint> loaded_UUIDs = ctx2.getObjectPrimitiveUUIDs(loaded_tile);
        DOCTEST_CHECK(loaded_UUIDs.size() == original_uv.size());
        for (size_t k = 0; k < loaded_UUIDs.size(); k++) {
            std::vector<vec2> uv = ctx2.getPrimitiveTextureUV(loaded_UUIDs.at(k));
            DOCTEST_CHECK(uv.size() == original_uv.at(k).size());
            for (size_t v = 0; v < uv.size(); v++) {
                DOCTEST_CHECK(uv.at(v).x == doctest::Approx(original_uv.at(k).at(v).x).epsilon(1e-4));
                DOCTEST_CHECK(uv.at(v).y == doctest::Approx(original_uv.at(k).at(v).y).epsilon(1e-4));
            }
        }

        std::remove(test_file);
    }

    SUBCASE("loadXML of an adaptive tile block referencing no primitives fails fast") {
        // The primitives in the file are the authority for an adaptive tile's geometry, so a block without them cannot
        // be reconstructed. Silently producing an empty or regenerated object would hide a corrupt file.
        const char *test_file = "helios_adaptive_tile_orphan_test.xml";

        std::ofstream outfile(test_file);
        outfile << "<helios>" << std::endl;
        outfile << "   <adaptive_tile>" << std::endl;
        outfile << "\t<objID>1</objID>" << std::endl;
        outfile << "\t<adaptive_refinement>0 0 0.25 2 0.5</adaptive_refinement>" << std::endl;
        outfile << "\t<subdivisions>4 4</subdivisions>" << std::endl;
        outfile << "\t<max_refinement_level>3</max_refinement_level>" << std::endl;
        outfile << "\t<subpatch_size_range>0.25 2</subpatch_size_range>" << std::endl;
        outfile << "\t<transform> 8 0 0 0 0 8 0 0 0 0 1 0 0 0 0 1 </transform>" << std::endl;
        outfile << "   </adaptive_tile>" << std::endl;
        outfile << "</helios>" << std::endl;
        outfile.close();

        Context ctx;
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(ctx.loadXML(test_file, true));

        std::remove(test_file);
    }

    SUBCASE("loadXML of an adaptive tile block with nonsensical parameters fails fast") {
        // The generation path validates these parameters thoroughly, but the load path adopts whatever the file says.
        // Without matching validation a hand-edited or corrupted file produces an object that reports a negative
        // sub-patch size and a zero-cell base grid through the public accessors as though they were legitimate.
        const char *test_file = "helios_adaptive_tile_invalid_test.xml";

        std::ofstream outfile(test_file);
        outfile << "<helios>" << std::endl;
        outfile << "   <patch>" << std::endl;
        outfile << "\t<UUID>0</UUID>" << std::endl;
        outfile << "\t<objID>1</objID>" << std::endl;
        outfile << "\t<transform> 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1 </transform>" << std::endl;
        outfile << "   </patch>" << std::endl;
        outfile << "   <adaptive_tile>" << std::endl;
        outfile << "\t<objID>1</objID>" << std::endl;
        outfile << "\t<adaptive_refinement>0 0 -5 -2 -1</adaptive_refinement>" << std::endl;
        outfile << "\t<subdivisions>0 0</subdivisions>" << std::endl;
        outfile << "\t<max_refinement_level>3</max_refinement_level>" << std::endl;
        outfile << "\t<subpatch_size_range>0.25 2</subpatch_size_range>" << std::endl;
        outfile << "\t<transform> 8 0 0 0 0 8 0 0 0 0 1 0 0 0 0 1 </transform>" << std::endl;
        outfile << "   </adaptive_tile>" << std::endl;
        outfile << "</helios>" << std::endl;
        outfile.close();

        Context ctx;
        capture_cerr capture;
        DOCTEST_CHECK_THROWS(ctx.loadXML(test_file, true));

        std::remove(test_file);
    }

    SUBCASE("loadXML of a file without a texture repeat element defaults silently") {
        // Backward compatibility: every Helios XML file written before <texture_repeat> existed lacks the
        // element, so its absence must default to 1x1 without any warning. A tile with the default repeat
        // must also not emit the element at all, keeping existing scene files byte-identical.
        const char *texture = "lib/images/disk_texture.png";
        const char *test_file = "helios_tile_default_repeat_test.xml";

        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(4, 4), nullrotation, make_int2(4, 4), texture);
        DOCTEST_CHECK(ctx.getTileObjectTextureRepeat(tile) == make_int2(1, 1));
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        std::ifstream infile(test_file);
        std::stringstream buffer;
        buffer << infile.rdbuf();
        infile.close();
        DOCTEST_CHECK(buffer.str().find("texture_repeat") == std::string::npos);

        std::string captured;
        Context ctx2;
        {
            capture_cerr capture;
            ctx2.loadXML(test_file, true);
            captured = capture.get_captured_output();
        } // capture destroyed before the assertions below
        DOCTEST_CHECK(captured.find("texture_repeat") == std::string::npos);

        uint loaded_tile = 0;
        bool found = false;
        for (uint objID: ctx2.getAllObjectIDs()) {
            if (ctx2.getObjectType(objID) == OBJECT_TYPE_TILE) {
                loaded_tile = objID;
                found = true;
                break;
            }
        }
        DOCTEST_CHECK(found);
        DOCTEST_CHECK(ctx2.getTileObjectTextureRepeat(loaded_tile) == make_int2(1, 1));

        std::remove(test_file);
    }

    SUBCASE("writeXML and loadXML preserve object sub-primitive ordering after deletion") {
        // Deleting a sub-primitive from a compound object must keep the remaining sub-primitives'
        // ordering intact across a writeXML()/loadXML() round-trip (deleteChildPrimitive is
        // order-preserving, and writeXML writes primitives in ascending UUID order).
        Context ctx;
        uint tile = ctx.addTileObject(make_vec3(0, 0, 0), make_vec2(3, 3), nullrotation, make_int2(3, 3));
        std::vector<uint> uuids = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(uuids.size() == 9);

        // Delete a middle sub-patch, then capture the remaining ordering
        ctx.deletePrimitive(uuids.at(3));
        std::vector<uint> remaining = ctx.getObjectPrimitiveUUIDs(tile);
        DOCTEST_CHECK(remaining.size() == 8);
        // The kept sub-primitives must still be in ascending (creation) order in memory
        for (size_t i = 1; i < remaining.size(); i++) {
            DOCTEST_CHECK(remaining.at(i) > remaining.at(i - 1));
        }

        const char *test_file = "helios_order_delete_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, true));

        Context ctx2;
        DOCTEST_CHECK_NOTHROW(ctx2.loadXML(test_file, true));

        std::vector<uint> object_ids = ctx2.getAllObjectIDs();
        DOCTEST_CHECK(object_ids.size() == 1);
        std::vector<uint> loaded = ctx2.getObjectPrimitiveUUIDs(object_ids.at(0));
        DOCTEST_CHECK(loaded.size() == remaining.size());
        // Loaded sub-primitives must also be in ascending order (same relative order as before write)
        for (size_t i = 1; i < loaded.size(); i++) {
            DOCTEST_CHECK(loaded.at(i) > loaded.at(i - 1));
        }

        std::remove(test_file);
    }

    SUBCASE("writeXML_byobject") {
        Context ctx;
        uint box1 = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint box2 = ctx.addBoxObject(make_vec3(2, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));

        std::vector<uint> selected_objects = {box1};
        const char *test_file = "helios_partial_test.xml";

        // Use quiet parameter to avoid output
        DOCTEST_CHECK_NOTHROW(ctx.writeXML_byobject(test_file, selected_objects, true));

        Context ctx2;
        std::vector<uint> loaded_uuids = ctx2.loadXML(test_file, true);

        // Should only have primitives from one box (6 faces)
        DOCTEST_CHECK(loaded_uuids.size() == 6);
        DOCTEST_CHECK(ctx2.getObjectCount() == 1);

        std::remove(test_file);
    }

    SUBCASE("writeXML with specific UUIDs") {
        Context ctx;
        uint p1 = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        uint p2 = ctx.addPatch(make_vec3(1, 0, 0), make_vec2(1, 1));
        uint p3 = ctx.addPatch(make_vec3(2, 0, 0), make_vec2(1, 1));

        std::vector<uint> selected_uuids = {p1, p3};
        const char *test_file = "helios_uuid_test.xml";

        // Use quiet parameter to avoid output
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, selected_uuids, true));

        Context ctx2;
        std::vector<uint> loaded_uuids = ctx2.loadXML(test_file, true);

        DOCTEST_CHECK(loaded_uuids.size() == 2);

        std::remove(test_file);
    }

    SUBCASE("getLoadedXMLFiles") {
        Context ctx;

        // Initially should be empty
        std::vector<std::string> files = ctx.getLoadedXMLFiles();
        size_t initial_count = files.size();

        // Create and load a test file using current working directory (cross-platform)
        uint patch = ctx.addPatch();
        std::string test_file = "helios_loaded_files_test.xml";
        // Use quiet parameter to avoid output
        ctx.writeXML(test_file.c_str(), true);

        Context ctx2;
        ctx2.loadXML(test_file.c_str(), true);

        std::vector<std::string> loaded_files = ctx2.getLoadedXMLFiles();
        DOCTEST_CHECK(loaded_files.size() == initial_count + 1);

        // Compare against the resolved path, not the original path
        std::filesystem::path resolved_test_file = resolveProjectFile(test_file);
        std::string resolved_test_file_str = resolved_test_file.string();
        DOCTEST_CHECK(std::find(loaded_files.begin(), loaded_files.end(), resolved_test_file_str) != loaded_files.end());

        std::remove(test_file.c_str());
    }

    SUBCASE("XML I/O quiet parameter") {
        Context ctx;
        uint patch = ctx.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        const char *test_file = "helios_quiet_test.xml";

        // Test writeXML with quiet=false (should produce output)
        {
            capture_cout cout_buffer;
            ctx.writeXML(test_file, false);
            DOCTEST_CHECK(cout_buffer.get_captured_output().find("Writing XML file") != std::string::npos);
        }

        // Test loadXML with quiet=false (should produce output)
        Context ctx2;
        {
            capture_cout cout_buffer;
            ctx2.loadXML(test_file, false);
            DOCTEST_CHECK(cout_buffer.get_captured_output().find("Loading XML file") != std::string::npos);
        }

        // Test writeXML with quiet=true (should not produce output)
        {
            capture_cout cout_buffer;
            ctx.writeXML(test_file, true);
            DOCTEST_CHECK(cout_buffer.get_captured_output().find("Writing XML file") == std::string::npos);
        }

        // Test loadXML with quiet=true (should not produce output)
        Context ctx3;
        {
            capture_cout cout_buffer;
            ctx3.loadXML(test_file, true);
            DOCTEST_CHECK(cout_buffer.get_captured_output().find("Loading XML file") == std::string::npos);
        }

        std::remove(test_file);
    }

    SUBCASE("vector global data XML formatting") {
        Context ctx;

        // Add various vector type global data
        std::vector<vec2> vec2_data = {make_vec2(400, 0.0666747f), make_vec2(401, 0.0640556f), make_vec2(402, 0.0619332f)};
        std::vector<vec3> vec3_data = {make_vec3(1.1f, 2.2f, 3.3f), make_vec3(4.4f, 5.5f, 6.6f)};
        std::vector<vec4> vec4_data = {make_vec4(1.1f, 2.2f, 3.3f, 4.4f), make_vec4(5.5f, 6.6f, 7.7f, 8.8f)};
        std::vector<int2> int2_data = {make_int2(1, 2), make_int2(3, 4)};
        std::vector<int3> int3_data = {make_int3(1, 2, 3), make_int3(4, 5, 6)};
        std::vector<int4> int4_data = {make_int4(1, 2, 3, 4), make_int4(5, 6, 7, 8)};

        ctx.setGlobalData("test_vec2", vec2_data);
        ctx.setGlobalData("test_vec3", vec3_data);
        ctx.setGlobalData("test_vec4", vec4_data);
        ctx.setGlobalData("test_int2", int2_data);
        ctx.setGlobalData("test_int3", int3_data);
        ctx.setGlobalData("test_int4", int4_data);

        const char *test_file = "helios_vector_test.xml";
        ctx.writeXML(test_file, true);

        // Load and verify all vector data
        Context ctx2;
        ctx2.loadXML(test_file, true);

        // Verify vec2
        std::vector<vec2> loaded_vec2;
        ctx2.getGlobalData("test_vec2", loaded_vec2);
        DOCTEST_CHECK(loaded_vec2.size() == 3);
        DOCTEST_CHECK(loaded_vec2[0].x == doctest::Approx(400.0f));
        DOCTEST_CHECK(loaded_vec2[0].y == doctest::Approx(0.0666747f));
        DOCTEST_CHECK(loaded_vec2[2].x == doctest::Approx(402.0f));

        // Verify vec3
        std::vector<vec3> loaded_vec3;
        ctx2.getGlobalData("test_vec3", loaded_vec3);
        DOCTEST_CHECK(loaded_vec3.size() == 2);
        DOCTEST_CHECK(loaded_vec3[0] == vec3(1.1f, 2.2f, 3.3f));
        DOCTEST_CHECK(loaded_vec3[1] == vec3(4.4f, 5.5f, 6.6f));

        // Verify vec4
        std::vector<vec4> loaded_vec4;
        ctx2.getGlobalData("test_vec4", loaded_vec4);
        DOCTEST_CHECK(loaded_vec4.size() == 2);
        DOCTEST_CHECK(loaded_vec4[0] == vec4(1.1f, 2.2f, 3.3f, 4.4f));
        DOCTEST_CHECK(loaded_vec4[1] == vec4(5.5f, 6.6f, 7.7f, 8.8f));

        // Verify int2
        std::vector<int2> loaded_int2;
        ctx2.getGlobalData("test_int2", loaded_int2);
        DOCTEST_CHECK(loaded_int2.size() == 2);
        DOCTEST_CHECK(loaded_int2[0] == int2(1, 2));
        DOCTEST_CHECK(loaded_int2[1] == int2(3, 4));

        // Verify int3
        std::vector<int3> loaded_int3;
        ctx2.getGlobalData("test_int3", loaded_int3);
        DOCTEST_CHECK(loaded_int3.size() == 2);
        DOCTEST_CHECK(loaded_int3[0] == int3(1, 2, 3));
        DOCTEST_CHECK(loaded_int3[1] == int3(4, 5, 6));

        // Verify int4
        std::vector<int4> loaded_int4;
        ctx2.getGlobalData("test_int4", loaded_int4);
        DOCTEST_CHECK(loaded_int4.size() == 2);
        DOCTEST_CHECK(loaded_int4[0] == int4(1, 2, 3, 4));
        DOCTEST_CHECK(loaded_int4[1] == int4(5, 6, 7, 8));

        std::remove(test_file);
    }

    // Note: scanXMLForTag testing removed due to complex XML tag structure requirements
}
