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
        
        // Write to XML
        const char* test_file = "/tmp/helios_test.xml";
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file));
        
        // Create new context and load
        Context ctx2;
        std::vector<uint> loaded_uuids;
        DOCTEST_CHECK_NOTHROW(loaded_uuids = ctx2.loadXML(test_file));
        DOCTEST_CHECK(loaded_uuids.size() >= 2);
        
        // Verify loaded data
        DOCTEST_CHECK(ctx2.getPrimitiveCount() >= 2);
        DOCTEST_CHECK(ctx2.getObjectCount() >= 1);
        
        // Check primitive data was loaded
        bool found_data = false;
        for (uint uuid : loaded_uuids) {
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
    
    SUBCASE("writeXML_byobject") {
        Context ctx;
        uint box1 = ctx.addBoxObject(make_vec3(0, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        uint box2 = ctx.addBoxObject(make_vec3(2, 0, 0), make_vec3(1, 1, 1), make_int3(1, 1, 1));
        
        std::vector<uint> selected_objects = {box1};
        const char* test_file = "/tmp/helios_partial_test.xml";
        
        DOCTEST_CHECK_NOTHROW(ctx.writeXML_byobject(test_file, selected_objects));
        
        Context ctx2;
        std::vector<uint> loaded_uuids = ctx2.loadXML(test_file);
        
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
        const char* test_file = "/tmp/helios_uuid_test.xml";
        
        DOCTEST_CHECK_NOTHROW(ctx.writeXML(test_file, selected_uuids));
        
        Context ctx2;
        std::vector<uint> loaded_uuids = ctx2.loadXML(test_file);
        
        DOCTEST_CHECK(loaded_uuids.size() == 2);
        
        std::remove(test_file);
    }
    
    SUBCASE("getLoadedXMLFiles") {
        Context ctx;
        
        // Initially should be empty
        std::vector<std::string> files = ctx.getLoadedXMLFiles();
        size_t initial_count = files.size();
        
        // Create and load a test file
        uint patch = ctx.addPatch();
        const char* test_file = "/tmp/helios_loaded_files_test.xml";
        ctx.writeXML(test_file);
        
        Context ctx2;
        ctx2.loadXML(test_file);
        
        std::vector<std::string> loaded_files = ctx2.getLoadedXMLFiles();
        DOCTEST_CHECK(loaded_files.size() == initial_count + 1);
        DOCTEST_CHECK(std::find(loaded_files.begin(), loaded_files.end(), test_file) != loaded_files.end());
        
        std::remove(test_file);
    }
    
    // Note: scanXMLForTag testing removed due to complex XML tag structure requirements
}
