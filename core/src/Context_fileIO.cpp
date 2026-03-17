/**
 * \file "Context_fileIO.cpp" Filesystem input/output functions within the Context.
 *
 * Copyright (C) 2016-2026 Brian Bailey
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 2
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 */

#include "Context.h"

using namespace helios;

// Geometric tolerance for triangle area validation
// Triangles with area below this threshold are considered degenerate and skipped
static constexpr float MIN_TRIANGLE_AREA_THRESHOLD = 1e-8f;

int XMLparser::parse_data_float(const pugi::xml_node &node_data, std::vector<float> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::string tmp_s;
        float tmp_f;
        while (data_stream >> tmp_s) {
            if (parse_float(tmp_s, tmp_f)) {
                data.push_back(tmp_f);
            } else {
                return 2;
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_double(const pugi::xml_node &node_data, std::vector<double> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::string tmp_s;
        double tmp_f;
        while (data_stream >> tmp_s) {
            if (parse_double(tmp_s, tmp_f)) {
                data.push_back(tmp_f);
            } else {
                return 2;
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_int(const pugi::xml_node &node_data, std::vector<int> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::string tmp_s;
        int tmp_f;
        while (data_stream >> tmp_s) {
            if (parse_int(tmp_s, tmp_f)) {
                data.push_back(tmp_f);
            } else {
                return 2;
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_uint(const pugi::xml_node &node_data, std::vector<uint> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::string tmp_s;
        uint tmp_f;
        while (data_stream >> tmp_s) {
            if (parse_uint(tmp_s, tmp_f)) {
                data.push_back(tmp_f);
            } else {
                return 2;
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_string(const pugi::xml_node &node_data, std::vector<std::string> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::string tmp_s;
        while (data_stream >> tmp_s) {
            data.push_back(tmp_s);
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_vec2(const pugi::xml_node &node_data, std::vector<vec2> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::vector<std::string> tmp_s(2);
        vec2 tmp;
        while (data_stream >> tmp_s[0]) {
            data_stream >> tmp_s[1];
            if (!parse_float(tmp_s[0], tmp.x) || !parse_float(tmp_s[1], tmp.y)) {
                return 2;
            } else {
                data.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_vec3(const pugi::xml_node &node_data, std::vector<vec3> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::vector<std::string> tmp_s(3);
        vec3 tmp;
        while (data_stream >> tmp_s[0]) {
            data_stream >> tmp_s[1];
            data_stream >> tmp_s[2];
            if (!parse_float(tmp_s[0], tmp.x) || !parse_float(tmp_s[1], tmp.y) || !parse_float(tmp_s[2], tmp.z)) {
                return 2;
            } else {
                data.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_vec4(const pugi::xml_node &node_data, std::vector<vec4> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::vector<std::string> tmp_s(4);
        vec4 tmp;
        while (data_stream >> tmp_s[0]) {
            data_stream >> tmp_s[1];
            data_stream >> tmp_s[2];
            data_stream >> tmp_s[3];
            if (!parse_float(tmp_s[0], tmp.x) || !parse_float(tmp_s[1], tmp.y) || !parse_float(tmp_s[2], tmp.z) || !parse_float(tmp_s[3], tmp.w)) {
                return 2;
            } else {
                data.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_int2(const pugi::xml_node &node_data, std::vector<int2> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::vector<std::string> tmp_s(2);
        int2 tmp;
        while (data_stream >> tmp_s[0]) {
            data_stream >> tmp_s[1];
            if (!parse_int(tmp_s[0], tmp.x) || !parse_int(tmp_s[1], tmp.y)) {
                return 2;
            } else {
                data.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_int3(const pugi::xml_node &node_data, std::vector<int3> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::vector<std::string> tmp_s(3);
        int3 tmp;
        while (data_stream >> tmp_s[0]) {
            data_stream >> tmp_s[1];
            data_stream >> tmp_s[2];
            if (!parse_int(tmp_s[0], tmp.x) || !parse_int(tmp_s[1], tmp.y) || !parse_int(tmp_s[2], tmp.z)) {
                return 2;
            } else {
                data.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_data_int4(const pugi::xml_node &node_data, std::vector<int4> &data) {
    std::string data_str = node_data.child_value();
    data.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::vector<std::string> tmp_s(4);
        int4 tmp;
        while (data_stream >> tmp_s[0]) {
            data_stream >> tmp_s[1];
            data_stream >> tmp_s[2];
            data_stream >> tmp_s[3];
            if (!parse_int(tmp_s[0], tmp.x) || !parse_int(tmp_s[1], tmp.y) || !parse_int(tmp_s[2], tmp.z) || !parse_int(tmp_s[3], tmp.w)) {
                return 2;
            } else {
                data.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_objID(const pugi::xml_node &node, uint &objID) {
    pugi::xml_node objID_node = node.child("objID");
    std::string oid = trim_whitespace(objID_node.child_value());
    objID = 0;
    if (!oid.empty()) {
        if (!parse_uint(oid, objID)) {
            return 2;
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_transform(const pugi::xml_node &node, float (&transform)[16]) {
    pugi::xml_node transform_node = node.child("transform");

    std::string transform_str = transform_node.child_value();
    if (transform_str.empty()) {
        makeIdentityMatrix(transform);
        return 1;
    } else {
        std::istringstream stream(transform_str);
        std::string tmp_s;
        float tmp;
        int i = 0;
        while (stream >> tmp_s) {
            if (parse_float(tmp_s, tmp)) {
                transform[i] = tmp;
                i++;
            } else {
                return 2;
            }
        }
        if (i != 16) {
            return 3;
        }
    }
    return 0;
}

int XMLparser::parse_texture(const pugi::xml_node &node, std::string &texture) {
    pugi::xml_node texture_node = node.child("texture");
    std::string texfile = trim_whitespace(texture_node.child_value());
    if (texfile.empty()) {
        texture = "none";
        return 1;
    } else {
        texture = texfile;
        return 0;
    }
}

int XMLparser::parse_textureUV(const pugi::xml_node &node, std::vector<vec2> &uvs) {
    pugi::xml_node uv_node = node.child("textureUV");
    std::string texUV = uv_node.child_value();
    if (!texUV.empty()) {
        std::istringstream uv_stream(texUV);
        std::vector<std::string> tmp_s(2);
        vec2 tmp;
        while (uv_stream >> tmp_s[0]) {
            uv_stream >> tmp_s[1];
            if (!parse_float(tmp_s[0], tmp.x) || !parse_float(tmp_s[1], tmp.y)) {
                return 2;
            } else {
                uvs.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_solid_fraction(const pugi::xml_node &node, float &solid_fraction) {
    pugi::xml_node sfrac_node = node.child("solid_fraction");
    std::string sfrac = trim_whitespace(sfrac_node.child_value());
    if (!sfrac.empty()) {
        if (!parse_float(sfrac, solid_fraction)) {
            return 2;
        }
    } else {
        return 1;
    }
    return 0;
}

int XMLparser::parse_vertices(const pugi::xml_node &node, std::vector<float> &vertices) {
    vertices.resize(12);

    pugi::xml_node vertices_node = node.child("vertices");

    std::string vertices_str = vertices_node.child_value();
    if (!vertices_str.empty()) {
        std::istringstream stream(vertices_str);
        std::string tmp_s;
        float tmp;
        int i = 0;
        while (stream >> tmp_s) {
            if (i > 11) {
                return 3;
            } else if (parse_float(tmp_s, tmp)) {
                vertices.at(i) = tmp;
                i++;
            } else {
                return 2;
            }
        }
        vertices.resize(i);
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_subdivisions(const pugi::xml_node &node, uint &subdivisions) {
    pugi::xml_node subdiv_node = node.child("subdivisions");
    std::string subdiv = trim_whitespace(subdiv_node.child_value());
    if (!subdiv.empty()) {
        if (!parse_uint(subdiv, subdivisions)) {
            return 2;
        }
    } else {
        return 1;
    }
    return 0;
}

int XMLparser::parse_subdivisions(const pugi::xml_node &node, int2 &subdivisions) {
    pugi::xml_node subdiv_node = node.child("subdivisions");
    std::string subdiv = trim_whitespace(subdiv_node.child_value());
    if (!subdiv.empty()) {
        std::istringstream data_stream(subdiv);
        std::vector<std::string> tmp_s(2);
        data_stream >> tmp_s[0];
        data_stream >> tmp_s[1];
        if (!parse_int(tmp_s[0], subdivisions.x) || !parse_int(tmp_s[1], subdivisions.y)) {
            return 2;
        }
    } else {
        return 1;
    }
    return 0;
}

int XMLparser::parse_subdivisions(const pugi::xml_node &node, int3 &subdivisions) {
    pugi::xml_node subdiv_node = node.child("subdivisions");
    std::string subdiv = trim_whitespace(subdiv_node.child_value());
    if (!subdiv.empty()) {
        std::istringstream data_stream(subdiv);
        std::vector<std::string> tmp_s(3);
        data_stream >> tmp_s[0];
        data_stream >> tmp_s[1];
        data_stream >> tmp_s[2];
        if (!parse_int(tmp_s[0], subdivisions.x) || !parse_int(tmp_s[1], subdivisions.y) || !parse_int(tmp_s[2], subdivisions.z)) {
            return 2;
        }
    } else {
        return 1;
    }
    return 0;
}

int XMLparser::parse_nodes(const pugi::xml_node &node, std::vector<vec3> &nodes) {
    pugi::xml_node node_data = node.child("nodes");
    std::string data_str = node_data.child_value();
    nodes.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::vector<std::string> tmp_s(3);
        vec3 tmp;
        while (data_stream >> tmp_s[0]) {
            data_stream >> tmp_s[1];
            data_stream >> tmp_s[2];
            if (!parse_float(tmp_s[0], tmp.x) || !parse_float(tmp_s[1], tmp.y) || !parse_float(tmp_s[2], tmp.z)) {
                return 2;
            } else {
                nodes.push_back(tmp);
            }
        }
    } else {
        return 1;
    }

    return 0;
}

int XMLparser::parse_radius(const pugi::xml_node &node, std::vector<float> &radius) {
    pugi::xml_node node_data = node.child("radius");
    std::string data_str = node_data.child_value();
    radius.resize(0);
    if (!data_str.empty()) {
        std::istringstream data_stream(data_str);
        std::string tmp_s;
        float tmp_f;
        while (data_stream >> tmp_s) {
            if (parse_float(tmp_s, tmp_f)) {
                radius.push_back(tmp_f);
            } else {
                return 2;
            }
        }
    } else {
        return 1;
    }

    return 0;
}

void Context::loadMaterialData(pugi::xml_node mat_node, const std::string &material_label) {
    // Load uint data
    for (pugi::xml_node data = mat_node.child("data_uint"); data; data = data.next_sibling("data_uint")) {
        const char *label = data.attribute("label").value();
        std::vector<uint> datav;
        if (XMLparser::parse_data_uint(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_uint> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load int data
    for (pugi::xml_node data = mat_node.child("data_int"); data; data = data.next_sibling("data_int")) {
        const char *label = data.attribute("label").value();
        std::vector<int> datav;
        if (XMLparser::parse_data_int(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_int> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load float data
    for (pugi::xml_node data = mat_node.child("data_float"); data; data = data.next_sibling("data_float")) {
        const char *label = data.attribute("label").value();
        std::vector<float> datav;
        if (XMLparser::parse_data_float(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_float> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load double data
    for (pugi::xml_node data = mat_node.child("data_double"); data; data = data.next_sibling("data_double")) {
        const char *label = data.attribute("label").value();
        std::vector<double> datav;
        if (XMLparser::parse_data_double(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_double> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load vec2 data
    for (pugi::xml_node data = mat_node.child("data_vec2"); data; data = data.next_sibling("data_vec2")) {
        const char *label = data.attribute("label").value();
        std::vector<vec2> datav;
        if (XMLparser::parse_data_vec2(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_vec2> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load vec3 data
    for (pugi::xml_node data = mat_node.child("data_vec3"); data; data = data.next_sibling("data_vec3")) {
        const char *label = data.attribute("label").value();
        std::vector<vec3> datav;
        if (XMLparser::parse_data_vec3(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_vec3> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load vec4 data
    for (pugi::xml_node data = mat_node.child("data_vec4"); data; data = data.next_sibling("data_vec4")) {
        const char *label = data.attribute("label").value();
        std::vector<vec4> datav;
        if (XMLparser::parse_data_vec4(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_vec4> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load int2 data
    for (pugi::xml_node data = mat_node.child("data_int2"); data; data = data.next_sibling("data_int2")) {
        const char *label = data.attribute("label").value();
        std::vector<int2> datav;
        if (XMLparser::parse_data_int2(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_int2> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load int3 data
    for (pugi::xml_node data = mat_node.child("data_int3"); data; data = data.next_sibling("data_int3")) {
        const char *label = data.attribute("label").value();
        std::vector<int3> datav;
        if (XMLparser::parse_data_int3(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_int3> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load int4 data
    for (pugi::xml_node data = mat_node.child("data_int4"); data; data = data.next_sibling("data_int4")) {
        const char *label = data.attribute("label").value();
        std::vector<int4> datav;
        if (XMLparser::parse_data_int4(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_int4> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }

    // Load string data
    for (pugi::xml_node data = mat_node.child("data_string"); data; data = data.next_sibling("data_string")) {
        const char *label = data.attribute("label").value();
        std::vector<std::string> datav;
        if (XMLparser::parse_data_string(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Material data tag <data_string> with label " + std::string(label) + " contained invalid data.");
        }
        if (datav.size() == 1) {
            setMaterialData(material_label, label, datav.front());
        } else if (datav.size() > 1) {
            setMaterialData(material_label, label, datav);
        }
    }
}

void Context::loadPData(pugi::xml_node p, uint UUID) {
    for (pugi::xml_node data = p.child("data_int"); data; data = data.next_sibling("data_int")) {
        const char *label = data.attribute("label").value();

        std::vector<int> datav;
        if (XMLparser::parse_data_int(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_int> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_uint"); data; data = data.next_sibling("data_uint")) {
        const char *label = data.attribute("label").value();

        std::vector<uint> datav;
        if (XMLparser::parse_data_uint(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_uint> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_float"); data; data = data.next_sibling("data_float")) {
        const char *label = data.attribute("label").value();

        std::vector<float> datav;
        if (XMLparser::parse_data_float(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_float> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_double"); data; data = data.next_sibling("data_double")) {
        const char *label = data.attribute("label").value();

        std::vector<double> datav;
        if (XMLparser::parse_data_double(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_double> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_vec2"); data; data = data.next_sibling("data_vec2")) {
        const char *label = data.attribute("label").value();

        std::vector<vec2> datav;
        if (XMLparser::parse_data_vec2(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_vec2> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_vec3"); data; data = data.next_sibling("data_vec3")) {
        const char *label = data.attribute("label").value();

        std::vector<vec3> datav;
        if (XMLparser::parse_data_vec3(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_vec3> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_vec4"); data; data = data.next_sibling("data_vec4")) {
        const char *label = data.attribute("label").value();

        std::vector<vec4> datav;
        if (XMLparser::parse_data_vec4(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_vec4> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_int2"); data; data = data.next_sibling("data_int2")) {
        const char *label = data.attribute("label").value();

        std::vector<int2> datav;
        if (XMLparser::parse_data_int2(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_int2> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_int3"); data; data = data.next_sibling("data_int3")) {
        const char *label = data.attribute("label").value();

        std::vector<int3> datav;
        if (XMLparser::parse_data_int3(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_int3> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_int4"); data; data = data.next_sibling("data_int4")) {
        const char *label = data.attribute("label").value();

        std::vector<int4> datav;
        if (XMLparser::parse_data_int4(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_int4> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_string"); data; data = data.next_sibling("data_string")) {
        const char *label = data.attribute("label").value();

        std::vector<std::string> datav;
        if (XMLparser::parse_data_string(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Primitive data tag <data_string> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setPrimitiveData(UUID, label, datav.front());
        } else if (datav.size() > 1) {
            setPrimitiveData(UUID, label, datav);
        }
    }
}

void Context::loadOData(pugi::xml_node p, uint ID) {
    assert(doesObjectExist(ID));

    for (pugi::xml_node data = p.child("data_int"); data; data = data.next_sibling("data_int")) {
        const char *label = data.attribute("label").value();

        std::vector<int> datav;
        if (XMLparser::parse_data_int(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_int> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_uint"); data; data = data.next_sibling("data_uint")) {
        const char *label = data.attribute("label").value();

        std::vector<uint> datav;
        if (XMLparser::parse_data_uint(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_uint> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_float"); data; data = data.next_sibling("data_float")) {
        const char *label = data.attribute("label").value();

        std::vector<float> datav;
        if (XMLparser::parse_data_float(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_float> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_double"); data; data = data.next_sibling("data_double")) {
        const char *label = data.attribute("label").value();

        std::vector<double> datav;
        if (XMLparser::parse_data_double(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_double> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_vec2"); data; data = data.next_sibling("data_vec2")) {
        const char *label = data.attribute("label").value();

        std::vector<vec2> datav;
        if (XMLparser::parse_data_vec2(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_vec2> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_vec3"); data; data = data.next_sibling("data_vec3")) {
        const char *label = data.attribute("label").value();

        std::vector<vec3> datav;
        if (XMLparser::parse_data_vec3(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_vec3> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_vec4"); data; data = data.next_sibling("data_vec4")) {
        const char *label = data.attribute("label").value();

        std::vector<vec4> datav;
        if (XMLparser::parse_data_vec4(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_vec4> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_int2"); data; data = data.next_sibling("data_int2")) {
        const char *label = data.attribute("label").value();

        std::vector<int2> datav;
        if (XMLparser::parse_data_int2(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_int2> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_int3"); data; data = data.next_sibling("data_int3")) {
        const char *label = data.attribute("label").value();

        std::vector<int3> datav;
        if (XMLparser::parse_data_int3(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_int3> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_int4"); data; data = data.next_sibling("data_int4")) {
        const char *label = data.attribute("label").value();

        std::vector<int4> datav;
        if (XMLparser::parse_data_int4(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_int4> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }

    for (pugi::xml_node data = p.child("data_string"); data; data = data.next_sibling("data_string")) {
        const char *label = data.attribute("label").value();

        std::vector<std::string> datav;
        if (XMLparser::parse_data_string(data, datav) != 0 || datav.empty()) {
            helios_runtime_error("ERROR (Context::loadXML): Object data tag <data_string> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setObjectData(ID, label, datav.front());
        } else if (datav.size() > 1) {
            setObjectData(ID, label, datav);
        }
    }
}

void Context::loadOsubPData(pugi::xml_node p, uint ID) {
    assert(doesObjectExist(ID));

    std::vector<uint> prim_UUIDs = getObjectPointer_private(ID)->getPrimitiveUUIDs();

    int u;

    for (pugi::xml_node prim_data = p.child("primitive_data_int"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<int> datav;
            if (XMLparser::parse_data_int(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_int> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_uint"); prim_data; prim_data = prim_data.next_sibling("primitive_data_uint")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<uint> datav;
            if (XMLparser::parse_data_uint(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_uint> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_float"); prim_data; prim_data = prim_data.next_sibling("primitive_data_float")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<float> datav;
            if (XMLparser::parse_data_float(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_float> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_double"); prim_data; prim_data = prim_data.next_sibling("primitive_data_double")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<double> datav;
            if (XMLparser::parse_data_double(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_double> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_vec2"); prim_data; prim_data = prim_data.next_sibling("primitive_data_vec2")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<vec2> datav;
            if (XMLparser::parse_data_vec2(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_vec2> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_vec3"); prim_data; prim_data = prim_data.next_sibling("primitive_data_vec3")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<vec3> datav;
            if (XMLparser::parse_data_vec3(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_vec3> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_vec4"); prim_data; prim_data = prim_data.next_sibling("primitive_data_vec4")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<vec4> datav;
            if (XMLparser::parse_data_vec4(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_vec4> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_int2"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int2")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<int2> datav;
            if (XMLparser::parse_data_int2(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_int2> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_int3"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int3")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<int3> datav;
            if (XMLparser::parse_data_int3(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_int3> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_int4"); prim_data; prim_data = prim_data.next_sibling("primitive_data_int4")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<int4> datav;
            if (XMLparser::parse_data_int4(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_int4> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }

    for (pugi::xml_node prim_data = p.child("primitive_data_string"); prim_data; prim_data = prim_data.next_sibling("primitive_data_string")) {
        const char *label = prim_data.attribute("label").value();

        u = 0;
        for (pugi::xml_node data = prim_data.child("data"); data; data = data.next_sibling("data")) {
            if (u >= prim_UUIDs.size()) {
                std::cerr << "WARNING (Context::loadXML): There was a problem with reading object primitive data \"" << label
                          << "\". The number of data values provided does not match the number of primitives contained in this object. Skipping remaining data values." << std::endl;
                break;
            }

            std::vector<std::string> datav;
            if (XMLparser::parse_data_string(data, datav) != 0 || datav.empty()) {
                helios_runtime_error("ERROR (Context::loadXML): Object member primitive data tag <primitive_data_string> with label " + std::string(label) + " contained invalid data.");
            }

            if (doesPrimitiveExist(prim_UUIDs.at(u))) {
                if (datav.size() == 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav.front());
                } else if (datav.size() > 1) {
                    setPrimitiveData(prim_UUIDs.at(u), label, datav);
                }
            }
            u++;
        }
    }
}

std::vector<uint> Context::loadXML(const char *filename, bool quiet) {
    if (!quiet) {
        std::cout << "Loading XML file: " << filename << "..." << std::flush;
    }

    std::string fn = filename;
    std::string ext = getFileExtension(filename);
    if (ext != ".xml" && ext != ".XML") {
        helios_runtime_error("failed.\n File " + fn + " is not XML format.");
    }

    // Resolve file path using unified resolution
    std::filesystem::path resolved_path = resolveFilePath(filename);
    std::string resolved_filename = resolved_path.string();

    XMLfiles.emplace_back(resolved_filename);

    uint ID;
    std::vector<uint> UUID;

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    // load file
    pugi::xml_parse_result load_result = xmldoc.load_file(resolved_filename.c_str());

    // error checking
    if (!load_result) {
        helios_runtime_error("failed.\n XML [" + std::string(filename) + "] parsed with errors, attr value: [" + xmldoc.child("node").attribute("attr").value() + "]\nError description: " + load_result.description() +
                             "\nError offset: " + std::to_string(load_result.offset) + " (error at [..." + (filename + load_result.offset) + "]\n");
    }

    pugi::xml_node helios = xmldoc.child("helios");

    if (helios.empty()) {
        if (!quiet) {
            std::cout << "failed." << std::endl;
        }
        helios_runtime_error("ERROR (Context::loadXML): XML file must have tag '<helios> ... </helios>' bounding all other tags.");
    }

    // if primitives are added that belong to an object, store their UUIDs here so that we can make sure their UUIDs are consistent
    std::map<uint, std::vector<uint>> object_prim_UUIDs;

    //-------------- TIME/DATE ---------------//

    for (pugi::xml_node p = helios.child("date"); p; p = p.next_sibling("date")) {
        pugi::xml_node year_node = p.child("year");
        const char *year_str = year_node.child_value();
        int year;
        if (!parse_int(year_str, year)) {
            helios_runtime_error("ERROR (Context::loadXML): Year given in 'date' block must be an integer value.");
        }

        pugi::xml_node month_node = p.child("month");
        const char *month_str = month_node.child_value();
        int month;
        if (!parse_int(month_str, month)) {
            helios_runtime_error("ERROR (Context::loadXML): Month given in 'date' block must be an integer value.");
        }

        pugi::xml_node day_node = p.child("day");
        const char *day_str = day_node.child_value();
        int day;
        if (!parse_int(day_str, day)) {
            helios_runtime_error("ERROR (Context::loadXML): Day given in 'date' block must be an integer value.");
        }

        setDate(day, month, year);
    }

    for (pugi::xml_node p = helios.child("time"); p; p = p.next_sibling("time")) {
        pugi::xml_node hour_node = p.child("hour");
        const char *hour_str = hour_node.child_value();
        int hour;
        if (!parse_int(hour_str, hour)) {
            helios_runtime_error("ERROR (Context::loadXML): Hour given in 'time' block must be an integer value.");
        }

        pugi::xml_node minute_node = p.child("minute");
        const char *minute_str = minute_node.child_value();
        int minute;
        if (!parse_int(minute_str, minute)) {
            helios_runtime_error("ERROR (Context::loadXML): Minute given in 'time' block must be an integer value.");
        }

        pugi::xml_node second_node = p.child("second");
        const char *second_str = second_node.child_value();
        int second;
        if (!parse_int(second_str, second)) {
            helios_runtime_error("ERROR (Context::loadXML): Second given in 'time' block must be an integer value.");
        }

        setTime(second, minute, hour);
    }

    //-------------- MATERIALS ---------------//
    // Map to track legacy numeric material IDs to labels for backward compatibility
    std::map<uint, std::string> legacy_material_id_to_label;

    for (pugi::xml_node m = helios.child("materials"); m; m = m.next_sibling("materials")) {
        for (pugi::xml_node mat = m.child("material"); mat; mat = mat.next_sibling("material")) {
            std::string material_label;
            RGBAcolor color = make_RGBAcolor(0, 0, 0, 1);
            std::string texture_file;
            bool texture_override = false;

            // Check for v3 format (label="...") first
            pugi::xml_attribute label_attr = mat.attribute("label");
            if (!label_attr.empty()) {
                material_label = label_attr.value();
            } else {
                // Check for v2 format (id="N")
                pugi::xml_attribute id_attr = mat.attribute("id");
                if (!id_attr.empty()) {
                    uint matID = 0;
                    const char *id_str = id_attr.value();
                    if (!parse_uint(id_str, matID)) {
                        helios_runtime_error("ERROR (Context::loadXML): Material ID must be an unsigned integer value.");
                    }
                    // Generate label from numeric ID for backward compatibility
                    material_label = "__auto_material_" + std::to_string(matID);
                    legacy_material_id_to_label[matID] = material_label;
                } else {
                    helios_runtime_error("ERROR (Context::loadXML): Material must have either a 'label' or 'id' attribute.");
                }
            }

            // Color
            pugi::xml_node color_node = mat.child("color");
            if (!color_node.empty()) {
                const char *color_str = color_node.child_value();
                std::istringstream color_stream(color_str);
                std::vector<float> color_vec;
                float tmp;
                while (color_stream >> tmp) {
                    color_vec.push_back(tmp);
                }
                if (color_vec.size() == 3) {
                    color = make_RGBAcolor(color_vec.at(0), color_vec.at(1), color_vec.at(2), 1.f);
                } else if (color_vec.size() == 4) {
                    color = make_RGBAcolor(color_vec.at(0), color_vec.at(1), color_vec.at(2), color_vec.at(3));
                }
            }

            // Texture
            pugi::xml_node texture_node = mat.child("texture");
            if (!texture_node.empty()) {
                texture_file = deblank(texture_node.child_value());
                if (!texture_file.empty()) {
                    addTexture(texture_file.c_str());
                }
            }

            // Texture override
            pugi::xml_node override_node = mat.child("texture_override");
            if (!override_node.empty()) {
                const char *override_str = override_node.child_value();
                int override_val;
                if (parse_int(override_str, override_val)) {
                    texture_override = (override_val != 0);
                }
            }

            // Twosided flag
            uint twosided = 1; // default: two-sided
            pugi::xml_node twosided_node = mat.child("twosided_flag");
            if (!twosided_node.empty()) {
                const char *twosided_str = twosided_node.child_value();
                int twosided_val;
                if (parse_int(twosided_str, twosided_val) && twosided_val >= 0) {
                    twosided = (uint) twosided_val;
                }
            }

            // Create the material using the new label-based API
            // Use internal method to bypass reserved label check for __auto_ labels
            if (!doesMaterialExist(material_label)) {
                uint newID = currentMaterialID++;
                Material loaded_mat(newID, material_label, color, texture_file, texture_override, twosided);
                materials[newID] = loaded_mat;
                material_label_to_id[material_label] = newID;
            } else {
                // Material already exists, update its properties
                setMaterialColor(material_label, color);
                if (!texture_file.empty()) {
                    setMaterialTexture(material_label, texture_file);
                }
                setMaterialTextureColorOverride(material_label, texture_override);
                setMaterialTwosidedFlag(material_label, twosided);
            }

            // Load material data
            loadMaterialData(mat, material_label);
        }
    }

    //-------------- PATCHES ---------------//
    for (pugi::xml_node p = helios.child("patch"); p; p = p.next_sibling("patch")) {
        // * Patch Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'patch' block must be a non-negative integer value.");
        }

        // * Patch Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Patch <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Patch <transform> node contains invalid data.");
        }

        // * Patch Texture * //
        std::string texture_file;
        XMLparser::parse_texture(p, texture_file);

        // * Patch Texture (u,v) Coordinates * //
        std::vector<vec2> uv;
        if (XMLparser::parse_textureUV(p, uv) == 2) {
            helios_runtime_error("ERROR (Context::loadXML): (u,v) coordinates given in 'patch' block contain invalid data.");
        }

        // * Patch Solid Fraction * //
        float solid_fraction = -1;
        if (XMLparser::parse_solid_fraction(p, solid_fraction) == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Solid fraction given in 'patch' block contains invalid data.");
        }

        // * Check for v3 material format (string label) vs v2 (numeric ID) vs legacy (color/texture) * //
        pugi::xml_node material_node = p.child("material");
        pugi::xml_node material_id_node = p.child("material_id");
        std::string material_label_from_xml;
        bool has_material = false;

        if (!material_node.empty()) {
            // v3 format: <material>label</material>
            material_label_from_xml = deblank(material_node.child_value());
            if (!material_label_from_xml.empty() && doesMaterialExist(material_label_from_xml)) {
                has_material = true;
                ID = addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBAcolor(0, 0, 0, 1));
            }
        } else if (!material_id_node.empty()) {
            // v2 format: <material_id>N</material_id>
            uint materialID_from_xml = 0;
            const char *mat_id_str = material_id_node.child_value();
            if (parse_uint(mat_id_str, materialID_from_xml)) {
                // Look up the label for this legacy numeric ID
                auto it = legacy_material_id_to_label.find(materialID_from_xml);
                if (it != legacy_material_id_to_label.end()) {
                    material_label_from_xml = it->second;
                    has_material = true;
                    ID = addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), make_RGBAcolor(0, 0, 0, 1));
                }
            }
        }

        if (!has_material) {
            // Legacy format: parse color and texture
            RGBAcolor color;
            pugi::xml_node color_node = p.child("color");

            const char *color_str = color_node.child_value();
            if (strlen(color_str) == 0) {
                color = make_RGBAcolor(0, 0, 0, 1); // assume default color of black
            } else {
                color = string2RGBcolor(color_str);
            }

            // * Add the Patch * //
            if (strcmp(texture_file.c_str(), "none") == 0) { // no texture file was given
                ID = addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), color);
            } else { // has a texture file
                std::string texture_file_copy;
                if (solid_fraction < 1.f && solid_fraction >= 0.f) { // solid fraction was given in the XML, and is not equal to 1.0
                    texture_file_copy = texture_file;
                    texture_file = "lib/images/solid.jpg"; // load dummy solid texture to avoid re-calculating the solid fraction
                }
                if (uv.empty()) { // custom (u,v) coordinates were not given
                    ID = addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), texture_file.c_str());
                } else {
                    ID = addPatch(make_vec3(0, 0, 0), make_vec2(1, 1), make_SphericalCoord(0, 0), texture_file.c_str(), 0.5 * (uv.at(2) + uv.at(0)), uv.at(2) - uv.at(0));
                }
                if (solid_fraction < 1.f && solid_fraction >= 0.f) { // replace dummy texture and set the solid fraction
                    getPrimitivePointer_private(ID)->setTextureFile(texture_file_copy.c_str());
                    addTexture(texture_file_copy.c_str());
                    getPrimitivePointer_private(ID)->setSolidFraction(solid_fraction);
                }
            }
        }

        getPrimitivePointer_private(ID)->setTransformationMatrix(transform);

        // Assign material if using material format
        if (has_material && !material_label_from_xml.empty()) {
            assignMaterialToPrimitive(ID, material_label_from_xml);
        }

        if (objID > 0) {
            object_prim_UUIDs[objID].push_back(ID);
        }

        if (objID == 0) {
            UUID.push_back(ID);
        }

        // * Primitive Data * //

        loadPData(p, ID);
    } // end patches

    //-------------- TRIANGLES ---------------//

    // looping over any triangles specified in XML file
    for (pugi::xml_node tri = helios.child("triangle"); tri; tri = tri.next_sibling("triangle")) {
        // * Triangle Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(tri, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'triangle' block must be a non-negative integer value.");
        }

        // * Triangle Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(tri, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Triangle <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Triangle <transform> node contains invalid data.");
        }

        // * Triangle Texture * //
        std::string texture_file;
        XMLparser::parse_texture(tri, texture_file);

        // * Triangle Texture (u,v) Coordinates * //
        std::vector<vec2> uv;
        if (XMLparser::parse_textureUV(tri, uv) == 2) {
            helios_runtime_error("ERROR (Context::loadXML): (u,v) coordinates given in 'triangle' block contain invalid data.");
        }

        // * Triangle Solid Fraction * //
        float solid_fraction = -1;
        if (XMLparser::parse_solid_fraction(tri, solid_fraction) == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Solid fraction given in 'triangle' block contains invalid data.");
        }

        // * Check for v3 material format (string label) vs v2 (numeric ID) vs legacy (color/texture) * //
        pugi::xml_node material_node_tri = tri.child("material");
        pugi::xml_node material_id_node_tri = tri.child("material_id");
        std::string material_label_from_xml_tri;
        bool has_material_tri = false;

        if (!material_node_tri.empty()) {
            // v3 format: <material>label</material>
            material_label_from_xml_tri = deblank(material_node_tri.child_value());
            if (!material_label_from_xml_tri.empty() && doesMaterialExist(material_label_from_xml_tri)) {
                has_material_tri = true;
            }
        } else if (!material_id_node_tri.empty()) {
            // v2 format: <material_id>N</material_id>
            uint materialID_from_xml_tri = 0;
            const char *mat_id_str = material_id_node_tri.child_value();
            if (parse_uint(mat_id_str, materialID_from_xml_tri)) {
                // Look up the label for this legacy numeric ID
                auto it = legacy_material_id_to_label.find(materialID_from_xml_tri);
                if (it != legacy_material_id_to_label.end()) {
                    material_label_from_xml_tri = it->second;
                    has_material_tri = true;
                }
            }
        }

        std::vector<vec3> vert_pos;
        vert_pos.resize(3);
        vert_pos.at(0) = make_vec3(0.f, 0.f, 0.f);
        vert_pos.at(1) = make_vec3(0.f, 1.f, 0.f);
        vert_pos.at(2) = make_vec3(1.f, 1.f, 0.f);

        if (has_material_tri) {
            // Material format: create triangle with default color, will assign material below
            ID = addTriangle(vert_pos.at(0), vert_pos.at(1), vert_pos.at(2), make_RGBAcolor(0, 0, 0, 1));
        } else {
            // Legacy format: parse color and texture
            RGBAcolor color;
            pugi::xml_node color_node = tri.child("color");

            const char *color_str = color_node.child_value();
            if (strlen(color_str) == 0) {
                color = make_RGBAcolor(0, 0, 0, 1); // assume default color of black
            } else {
                color = string2RGBcolor(color_str);
            }

            // * Add the Triangle * //
            if (strcmp(texture_file.c_str(), "none") == 0 || uv.empty()) {
                ID = addTriangle(vert_pos.at(0), vert_pos.at(1), vert_pos.at(2), color);
            } else {
                std::string texture_file_copy;
                if (solid_fraction < 1.f && solid_fraction >= 0.f) { // solid fraction was given in the XML, and is not equal to 1.0
                    texture_file_copy = texture_file;
                    texture_file = "lib/images/solid.jpg"; // load dummy solid texture to avoid re-calculating the solid fraction
                }
                ID = addTriangle(vert_pos.at(0), vert_pos.at(1), vert_pos.at(2), texture_file.c_str(), uv.at(0), uv.at(1), uv.at(2));
                if (solid_fraction < 1.f && solid_fraction >= 0.f) {
                    getPrimitivePointer_private(ID)->setTextureFile(texture_file_copy.c_str());
                    addTexture(texture_file_copy.c_str());
                    getPrimitivePointer_private(ID)->setSolidFraction(solid_fraction);
                }
            }
        }

        getPrimitivePointer_private(ID)->setTransformationMatrix(transform);

        // Assign material if using material format
        if (has_material_tri && !material_label_from_xml_tri.empty()) {
            assignMaterialToPrimitive(ID, material_label_from_xml_tri);
        }

        if (objID > 0) {
            object_prim_UUIDs[objID].push_back(ID);
        }

        if (objID == 0) {
            UUID.push_back(ID);
        }

        // * Primitive Data * //

        loadPData(tri, ID);
    }

    //-------------- VOXELS ---------------//
    for (pugi::xml_node p = helios.child("voxel"); p; p = p.next_sibling("voxel")) {
        // * Voxel Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'voxel' block must be a non-negative integer value.");
        }

        // * Voxel Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Voxel <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Voxel <transform> node contains invalid data.");
        }

        // * Voxel Solid Fraction * //
        float solid_fraction = 1;
        if (XMLparser::parse_solid_fraction(p, solid_fraction) == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Solid fraction given in 'voxel' block contains invalid data.");
        }

        // * Check for v3 material format (string label) vs v2 (numeric ID) vs legacy (color/texture) * //
        pugi::xml_node material_node_vox = p.child("material");
        pugi::xml_node material_id_node_vox = p.child("material_id");
        std::string material_label_from_xml_vox;
        bool has_material_vox = false;

        if (!material_node_vox.empty()) {
            // v3 format: <material>label</material>
            material_label_from_xml_vox = deblank(material_node_vox.child_value());
            if (!material_label_from_xml_vox.empty() && doesMaterialExist(material_label_from_xml_vox)) {
                has_material_vox = true;
            }
        } else if (!material_id_node_vox.empty()) {
            // v2 format: <material_id>N</material_id>
            uint materialID_from_xml_vox = 0;
            const char *mat_id_str = material_id_node_vox.child_value();
            if (parse_uint(mat_id_str, materialID_from_xml_vox)) {
                // Look up the label for this legacy numeric ID
                auto it = legacy_material_id_to_label.find(materialID_from_xml_vox);
                if (it != legacy_material_id_to_label.end()) {
                    material_label_from_xml_vox = it->second;
                    has_material_vox = true;
                }
            }
        }

        if (has_material_vox) {
            // Material format: create voxel with default color, will assign material below
            ID = addVoxel(make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0, make_RGBAcolor(0, 0, 0, 1));
        } else {
            // Legacy format: parse color
            RGBAcolor color;
            pugi::xml_node color_node = p.child("color");

            const char *color_str = color_node.child_value();
            if (strlen(color_str) == 0) {
                color = make_RGBAcolor(0, 0, 0, 1); // assume default color of black
            } else {
                color = string2RGBcolor(color_str);
            }

            // * Add the Voxel * //
            ID = addVoxel(make_vec3(0, 0, 0), make_vec3(0, 0, 0), 0, color);
        }

        getPrimitivePointer_private(ID)->setTransformationMatrix(transform);

        // Assign material if using material format
        if (has_material_vox && !material_label_from_xml_vox.empty()) {
            assignMaterialToPrimitive(ID, material_label_from_xml_vox);
        }

        if (objID > 0) {
            object_prim_UUIDs[objID].push_back(ID);
        }

        if (objID == 0) {
            UUID.push_back(ID);
        }

        // * Primitive Data * //

        loadPData(p, ID);
    }

    //-------------- COMPOUND OBJECTS ---------------//

    //-------------- TILES ---------------//
    for (pugi::xml_node p = helios.child("tile"); p; p = p.next_sibling("tile")) {
        // * Tile Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'tile' block must be a non-negative integer value.");
        }

        // * Tile Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Tile <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Tile <transform> node contains invalid data.");
        }

        // * Tile Texture * //
        std::string texture_file;
        XMLparser::parse_texture(p, texture_file);

        // * Tile Texture (u,v) Coordinates * //
        std::vector<vec2> uv;
        if (XMLparser::parse_textureUV(p, uv) == 2) {
            helios_runtime_error("ERROR (Context::loadXML): (u,v) coordinates given in 'tile' block contain invalid data.");
        }

        // * Tile Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if (strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Tile Subdivisions * //
        int2 subdiv;
        int result_subdiv = XMLparser::parse_subdivisions(p, subdiv);
        if (result_subdiv == 1) {
            std::cerr << "WARNING (Context::loadXML): Number of subdivisions for tile was not provided. Assuming 1x1." << std::endl;
            subdiv = make_int2(1, 1);
        } else if (result_subdiv == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Tile <subdivisions> node contains invalid data. ");
        }

        // Create a dummy patch in order to get the center, size, and rotation based on transformation matrix
        Patch patch(make_RGBAcolor(0, 0, 0, 0), 0, 0);
        patch.setTransformationMatrix(transform);
        //    SphericalCoord rotation = cart2sphere(patch.getNormal());
        //    rotation.elevation = rotation.zenith;

        // * Add the Tile * //
        //    if (strcmp(texture_file.c_str(), "none") == 0) {
        //      if( strlen(color_str) == 0 ){
        //        ID = addTileObject(patch.getCenter(), patch.getSize(), rotation, subdiv );
        //      }else {
        //        ID = addTileObject(patch.getCenter(), patch.getSize(), rotation, subdiv, make_RGBcolor(color.r, color.g, color.b));
        //      }
        //    } else {
        //      ID = addTileObject(patch.getCenter(), patch.getSize(), rotation, subdiv, texture_file.c_str());
        //    }

        if (strcmp(texture_file.c_str(), "none") == 0) {
            if (strlen(color_str) == 0) {
                ID = addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, subdiv);
            } else {
                ID = addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, subdiv, make_RGBcolor(color.r, color.g, color.b));
            }
        } else {
            ID = addTileObject(make_vec3(0, 0, 0), make_vec2(1, 1), nullrotation, subdiv, texture_file.c_str());
        }

        getTileObjectPointer_private(ID)->setTransformationMatrix(transform);

        setPrimitiveTransformationMatrix(getObjectPrimitiveUUIDs(ID), transform);

        // if primitives exist that were assigned to this object, delete all primitives that were just created
        if (objID > 0 && !object_prim_UUIDs.empty() && object_prim_UUIDs.find(objID) != object_prim_UUIDs.end()) {
            std::vector<uint> uuids_to_delete = getObjectPrimitiveUUIDs(ID);
            getObjectPointer_private(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));
            deletePrimitive(uuids_to_delete);
            // \todo This is fairly inefficient, it would be nice to have a way to do this without having to create and delete a bunch of primitives
        }

        setPrimitiveParentObjectID(getObjectPrimitiveUUIDs(ID), ID);

        // * Tile Sub-Patch Data * //

        loadOsubPData(p, ID);

        // * Tile Object Data * //

        loadOData(p, ID);

        std::vector<uint> childUUIDs = getObjectPrimitiveUUIDs(ID);
        UUID.insert(UUID.end(), childUUIDs.begin(), childUUIDs.end());
    } // end tiles

    //-------------- SPHERES ---------------//
    for (pugi::xml_node p = helios.child("sphere"); p; p = p.next_sibling("sphere")) {
        // * Sphere Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'sphere' block must be a non-negative integer value.");
        }

        if (doesObjectExist(objID)) { // if this object ID is already in use, assign a new one
            objID = currentObjectID;
            currentObjectID++;
        }

        // * Sphere Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Sphere <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Sphere <transform> node contains invalid data.");
        }

        // * Sphere Texture * //
        std::string texture_file;
        XMLparser::parse_texture(p, texture_file);

        // * Sphere Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if (strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Sphere Subdivisions * //
        uint subdiv;
        int result_subdiv = XMLparser::parse_subdivisions(p, subdiv);
        if (result_subdiv == 1) {
            std::cerr << "WARNING (Context::loadXML): Number of subdivisions for sphere was not provided. Assuming 1x1." << std::endl;
            subdiv = 5;
        } else if (result_subdiv == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Sphere <subdivisions> node contains invalid data. ");
        }

        // Create a dummy sphere in order to get the center and radius based on transformation matrix
        std::vector<uint> empty;
        Sphere sphere(0, empty, 3, "", this);
        sphere.setTransformationMatrix(transform);

        // * Add the Sphere * //
        if (strcmp(texture_file.c_str(), "none") == 0) {
            if (strlen(color_str) == 0) {
                ID = addSphereObject(subdiv, sphere.getCenter(), sphere.getRadius());
            } else {
                ID = addSphereObject(subdiv, sphere.getCenter(), sphere.getRadius(), make_RGBcolor(color.r, color.g, color.b));
            }
        } else {
            ID = addSphereObject(subdiv, sphere.getCenter(), sphere.getRadius(), texture_file.c_str());
        }

        // if primitives exist that were assigned to this object, delete all primitives that were just created
        if (objID > 0 && object_prim_UUIDs.find(objID) != object_prim_UUIDs.end()) {
            std::vector<uint> uuids_to_delete = getObjectPrimitiveUUIDs(ID);
            getObjectPointer_private(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));
            deletePrimitive(uuids_to_delete);
            //          if( !doesObjectExist(ID) ){ //if the above method deleted all primitives for this object, move on
            //            continue;
            //          }
        }

        setPrimitiveParentObjectID(getObjectPrimitiveUUIDs(ID), ID);

        // * Sphere Sub-Triangle Data * //

        loadOsubPData(p, ID);

        // * Sphere Object Data * //

        loadOData(p, ID);

        std::vector<uint> childUUIDs = getObjectPrimitiveUUIDs(ID);
        UUID.insert(UUID.end(), childUUIDs.begin(), childUUIDs.end());
    } // end spheres

    //-------------- TUBES ---------------//
    for (pugi::xml_node p = helios.child("tube"); p; p = p.next_sibling("tube")) {
        // * Tube Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'tube' block must be a non-negative integer value.");
        }

        if (doesObjectExist(objID)) { // if this object ID is already in use, assign a new one
            objID = currentObjectID;
            currentObjectID++;
        }

        // * Tube Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Tube <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Tube <transform> node contains invalid data.");
        }

        // * Tube Texture * //
        std::string texture_file;
        XMLparser::parse_texture(p, texture_file);

        // * Tube Subdivisions * //
        uint subdiv;
        int result_subdiv = XMLparser::parse_subdivisions(p, subdiv);
        if (result_subdiv == 1) {
            std::cerr << "WARNING (Context::loadXML): Number of subdivisions for tube was not provided. Assuming 1x1." << std::endl;
            subdiv = 5;
        } else if (result_subdiv == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Tube <subdivisions> node contains invalid data. ");
        }

        // * Tube Nodes * //
        std::vector<vec3> nodes;
        pugi::xml_node nodes_node = p.child("nodes");
        if (XMLparser::parse_data_vec3(nodes_node, nodes) != 0 || nodes.size() < 2) {
            helios_runtime_error("ERROR (Context::loadXML): Tube <nodes> node contains invalid data. ");
        }

        // * Tube Radius * //
        std::vector<float> radii;
        pugi::xml_node radii_node = p.child("radius");
        if (XMLparser::parse_data_float(radii_node, radii) != 0 || radii.size() < 2) {
            helios_runtime_error("ERROR (Context::loadXML): Tube <radius> node contains invalid data. ");
        }

        // * Tube Color * //

        pugi::xml_node color_node = p.child("color");
        const char *color_str = color_node.child_value();

        std::vector<RGBcolor> colors;
        if (strlen(color_str) > 0) {
            std::istringstream data_stream(color_str);
            std::vector<float> tmp;
            tmp.resize(3);
            int c = 0;
            while (data_stream >> tmp.at(c)) {
                c++;
                if (c == 3) {
                    colors.push_back(make_RGBcolor(tmp.at(0), tmp.at(1), tmp.at(2)));
                    c = 0;
                }
            }
        }

        // * Add the Tube * //
        if (texture_file == "none") {
            ID = addTubeObject(subdiv, nodes, radii, colors);
        } else {
            ID = addTubeObject(subdiv, nodes, radii, texture_file.c_str());
        }

        getObjectPointer_private(ID)->setTransformationMatrix(transform);

        // if primitives exist that were assigned to this object, delete all primitives that were just created
        if (objID > 0 && object_prim_UUIDs.find(objID) != object_prim_UUIDs.end()) {
            std::vector<uint> uuids_to_delete = getObjectPrimitiveUUIDs(ID);
            getObjectPointer_private(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));
            deletePrimitive(uuids_to_delete);
            //            if( !doesObjectExist(ID) ){ //if the above method deleted all primitives for this object, move on
            //              continue;
            //            }
        }

        setPrimitiveParentObjectID(getObjectPrimitiveUUIDs(ID), ID);

        // * Tube Sub-Triangle Data * //

        loadOsubPData(p, ID);

        // * tube Object Data * //

        loadOData(p, ID);

        std::vector<uint> childUUIDs = getObjectPrimitiveUUIDs(ID);
        UUID.insert(UUID.end(), childUUIDs.begin(), childUUIDs.end());
    } // end tubes

    //-------------- BOXES ---------------//
    for (pugi::xml_node p = helios.child("box"); p; p = p.next_sibling("box")) {
        // * Box Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'box' block must be a non-negative integer value.");
        }

        if (doesObjectExist(objID)) { // if this object ID is already in use, assign a new one
            objID = currentObjectID;
            currentObjectID++;
        }

        // * Box Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Box <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Box <transform> node contains invalid data.");
        }

        // * Box Texture * //
        std::string texture_file;
        XMLparser::parse_texture(p, texture_file);

        // * Box Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if (strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Box Subdivisions * //
        int3 subdiv;
        int result_subdiv = XMLparser::parse_subdivisions(p, subdiv);
        if (result_subdiv == 1) {
            std::cerr << "WARNING (Context::loadXML): Number of subdivisions for box was not provided. Assuming 1x1." << std::endl;
            subdiv = make_int3(1, 1, 1);
        } else if (result_subdiv == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Box <subdivisions> node contains invalid data. ");
        }

        // Create a dummy box in order to get the center and size based on transformation matrix
        std::vector<uint> empty;
        Box box(0, empty, make_int3(1, 1, 1), "", this);
        box.setTransformationMatrix(transform);

        // * Add the box * //
        if (strcmp(texture_file.c_str(), "none") == 0) {
            if (strlen(color_str) == 0) {
                ID = addBoxObject(box.getCenter(), box.getSize(), subdiv);
            } else {
                ID = addBoxObject(box.getCenter(), box.getSize(), subdiv, make_RGBcolor(color.r, color.g, color.b));
            }
        } else {
            ID = addBoxObject(box.getCenter(), box.getSize(), subdiv, texture_file.c_str());
        }

        // if primitives exist that were assigned to this object, delete all primitives that were just created
        if (objID > 0 && object_prim_UUIDs.find(objID) != object_prim_UUIDs.end()) {
            std::vector<uint> uuids_to_delete = getObjectPrimitiveUUIDs(ID);
            getObjectPointer_private(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));
            deletePrimitive(uuids_to_delete);
            //            if( !doesObjectExist(ID) ){ //if the above method deleted all primitives for this object, move on
            //              continue;
            //            }
        }

        setPrimitiveParentObjectID(getObjectPrimitiveUUIDs(ID), ID);

        // * Box Sub-Patch Data * //

        loadOsubPData(p, ID);

        // * Box Object Data * //

        loadOData(p, ID);

        std::vector<uint> childUUIDs = getObjectPrimitiveUUIDs(ID);
        UUID.insert(UUID.end(), childUUIDs.begin(), childUUIDs.end());
    } // end boxes

    //-------------- DISKS ---------------//
    for (pugi::xml_node p = helios.child("disk"); p; p = p.next_sibling("disk")) {
        // * Disk Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'disk' block must be a non-negative integer value.");
        }

        if (doesObjectExist(objID)) { // if this object ID is already in use, assign a new one
            objID = currentObjectID;
            currentObjectID++;
        }

        // * Disk Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Disk <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Disk <transform> node contains invalid data.");
        }

        // * Disk Texture * //
        std::string texture_file;
        XMLparser::parse_texture(p, texture_file);

        // * Disk Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if (strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Disk Subdivisions * //
        int2 subdiv;
        int result_subdiv = XMLparser::parse_subdivisions(p, subdiv);
        if (result_subdiv == 1) {
            std::cerr << "WARNING (Context::loadXML): Number of subdivisions for disk was not provided. Assuming 1x1." << std::endl;
            subdiv = make_int2(5, 1);
        } else if (result_subdiv == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Disk <subdivisions> node contains invalid data. ");
        }

        // Create a dummy disk in order to get the center and size based on transformation matrix
        std::vector<uint> empty;
        Disk disk(0, empty, make_int2(1, 1), "", this);
        disk.setTransformationMatrix(transform);

        // * Add the disk * //
        if (strcmp(texture_file.c_str(), "none") == 0) {
            if (strlen(color_str) == 0) {
                ID = addDiskObject(subdiv, disk.getCenter(), disk.getSize(), nullrotation, RGB::red);
            } else {
                ID = addDiskObject(subdiv, disk.getCenter(), disk.getSize(), nullrotation, make_RGBcolor(color.r, color.g, color.b));
            }
        } else {
            ID = addDiskObject(subdiv, disk.getCenter(), disk.getSize(), nullrotation, texture_file.c_str());
        }

        // if primitives exist that were assigned to this object, delete all primitives that were just created
        if (objID > 0 && object_prim_UUIDs.find(objID) != object_prim_UUIDs.end()) {
            std::vector<uint> uuids_to_delete = getObjectPrimitiveUUIDs(ID);
            getObjectPointer_private(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));
            deletePrimitive(uuids_to_delete);
            //            if( !doesObjectExist(ID) ){ //if the above method deleted all primitives for this object, move on
            //              continue;
            //            }
        }

        setPrimitiveParentObjectID(getObjectPrimitiveUUIDs(ID), ID);

        // * Disk Sub-Triangle Data * //

        loadOsubPData(p, ID);

        // * Disk Object Data * //

        loadOData(p, ID);

        std::vector<uint> childUUIDs = getObjectPrimitiveUUIDs(ID);
        UUID.insert(UUID.end(), childUUIDs.begin(), childUUIDs.end());
    } // end disks

    //-------------- CONES ---------------//
    for (pugi::xml_node p = helios.child("cone"); p; p = p.next_sibling("cone")) {
        // * Cone Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'cone' block must be a non-negative integer value.");
        }

        if (doesObjectExist(objID)) { // if this object ID is already in use, assign a new one
            objID = currentObjectID;
            currentObjectID++;
        }

        // * Cone Transformation Matrix * //
        float transform[16];
        int result = XMLparser::parse_transform(p, transform);
        if (result == 3) {
            helios_runtime_error("ERROR (Context::loadXML): Cone <transform> node contains less than 16 data values.");
        } else if (result == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Cone <transform> node contains invalid data.");
        }

        // * Cone Texture * //
        std::string texture_file;
        XMLparser::parse_texture(p, texture_file);

        // * Cone Diffuse Colors * //
        RGBAcolor color;
        pugi::xml_node color_node = p.child("color");

        const char *color_str = color_node.child_value();
        if (strlen(color_str) != 0) {
            color = string2RGBcolor(color_str);
        }

        // * Cone Subdivisions * //
        uint subdiv;
        int result_subdiv = XMLparser::parse_subdivisions(p, subdiv);
        if (result_subdiv == 1) {
            std::cerr << "WARNING (Context::loadXML): Number of subdivisions for cone was not provided. Assuming 1x1." << std::endl;
            subdiv = 5;
        } else if (result_subdiv == 2) {
            helios_runtime_error("ERROR (Context::loadXML): Cone <subdivisions> node contains invalid data. ");
        }

        // * Cone Nodes * //
        std::vector<vec3> nodes;
        pugi::xml_node nodes_node = p.child("nodes");
        if (XMLparser::parse_data_vec3(nodes_node, nodes) != 0 || nodes.size() != 2) {
            helios_runtime_error("ERROR (Context::loadXML): Cone <nodes> node contains invalid data. ");
        }

        // * Cone Radius * //
        std::vector<float> radii;
        pugi::xml_node radii_node = p.child("radius");
        if (XMLparser::parse_data_float(radii_node, radii) != 0 || radii.size() != 2) {
            helios_runtime_error("ERROR (Context::loadXML): Cone <radius> node contains invalid data. ");
        }

        // * Add the Cone * //
        if (texture_file == "none") {
            ID = addConeObject(subdiv, nodes.at(0), nodes.at(1), radii.at(0), radii.at(1), make_RGBcolor(color.r, color.g, color.b));
        } else {
            ID = addConeObject(subdiv, nodes.at(0), nodes.at(1), radii.at(0), radii.at(1), texture_file.c_str());
        }

        getObjectPointer_private(ID)->setTransformationMatrix(transform);

        // if primitives exist that were assigned to this object, delete all primitives that were just created
        if (objID > 0 && object_prim_UUIDs.find(objID) != object_prim_UUIDs.end()) {
            std::vector<uint> uuids_to_delete = getObjectPrimitiveUUIDs(ID);
            getObjectPointer_private(ID)->setPrimitiveUUIDs(object_prim_UUIDs.at(objID));
            deletePrimitive(uuids_to_delete);
            //          if( !doesObjectExist(ID) ){ //if the above method deleted all primitives for this object, move on
            //            continue;
            //          }
        }

        setPrimitiveParentObjectID(getObjectPrimitiveUUIDs(ID), ID);

        // * Cone Sub-Triangle Data * //

        loadOsubPData(p, ID);

        // * Cone Object Data * //

        loadOData(p, ID);

        std::vector<uint> childUUIDs = getObjectPrimitiveUUIDs(ID);
        UUID.insert(UUID.end(), childUUIDs.begin(), childUUIDs.end());
    } // end cones

    //-------------- POLYMESH ---------------//
    for (pugi::xml_node p = helios.child("polymesh"); p; p = p.next_sibling("polymesh")) {
        // * Polymesh Object ID * //
        uint objID = 0;
        if (XMLparser::parse_objID(p, objID) > 1) {
            helios_runtime_error("ERROR (Context::loadXML): Object ID (objID) given in 'polymesh' block must be a non-negative integer value.");
        }

        if (doesObjectExist(objID)) { // if this object ID is already in use, assign a new one
            objID = currentObjectID;
            currentObjectID++;
        }

        ID = addPolymeshObject(object_prim_UUIDs.at(objID));

        setPrimitiveParentObjectID(object_prim_UUIDs.at(objID), ID);

        // * Polymesh Sub-Primitive Data * //

        loadOsubPData(p, ID);

        // * Polymesh Object Data * //

        loadOData(p, ID);

        std::vector<uint> childUUIDs = object_prim_UUIDs.at(objID);
        UUID.insert(UUID.end(), childUUIDs.begin(), childUUIDs.end());
    } // end polymesh

    //-------------- GLOBAL DATA ---------------//

    for (pugi::xml_node data = helios.child("globaldata_int"); data; data = data.next_sibling("globaldata_int")) {
        const char *label = data.attribute("label").value();

        std::vector<int> datav;
        if (XMLparser::parse_data_int(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_int> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_uint"); data; data = data.next_sibling("globaldata_uint")) {
        const char *label = data.attribute("label").value();

        std::vector<uint> datav;
        if (XMLparser::parse_data_uint(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_uint> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_float"); data; data = data.next_sibling("globaldata_float")) {
        const char *label = data.attribute("label").value();

        std::vector<float> datav;
        if (XMLparser::parse_data_float(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_float> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_double"); data; data = data.next_sibling("globaldata_double")) {
        const char *label = data.attribute("label").value();

        std::vector<double> datav;
        if (XMLparser::parse_data_double(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_double> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_vec2"); data; data = data.next_sibling("globaldata_vec2")) {
        const char *label = data.attribute("label").value();

        std::vector<vec2> datav;
        if (XMLparser::parse_data_vec2(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_vec2> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_vec3"); data; data = data.next_sibling("globaldata_vec3")) {
        const char *label = data.attribute("label").value();

        std::vector<vec3> datav;
        if (XMLparser::parse_data_vec3(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_vec3> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_vec4"); data; data = data.next_sibling("globaldata_vec4")) {
        const char *label = data.attribute("label").value();

        std::vector<vec4> datav;
        if (XMLparser::parse_data_vec4(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_vec4> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_int2"); data; data = data.next_sibling("globaldata_int2")) {
        const char *label = data.attribute("label").value();

        std::vector<int2> datav;
        if (XMLparser::parse_data_int2(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_int2> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_int3"); data; data = data.next_sibling("globaldata_int3")) {
        const char *label = data.attribute("label").value();

        std::vector<int3> datav;
        if (XMLparser::parse_data_int3(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_int3> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_int4"); data; data = data.next_sibling("globaldata_int4")) {
        const char *label = data.attribute("label").value();

        std::vector<int4> datav;
        if (XMLparser::parse_data_int4(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_int4> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    for (pugi::xml_node data = helios.child("globaldata_string"); data; data = data.next_sibling("globaldata_string")) {
        const char *label = data.attribute("label").value();

        std::vector<std::string> datav;
        if (XMLparser::parse_data_string(data, datav) != 0) {
            helios_runtime_error("ERROR (Context::loadXML): Global data tag <globaldata_string> with label " + std::string(label) + " contained invalid data.");
        }

        if (datav.size() == 1) {
            setGlobalData(label, datav.front());
        } else if (datav.size() > 1) {
            setGlobalData(label, datav);
        }
    }

    //-------------- TIMESERIES DATA ---------------//
    for (pugi::xml_node p = helios.child("timeseries"); p; p = p.next_sibling("timeseries")) {
        const char *label = p.attribute("label").value();

        for (pugi::xml_node d = p.child("datapoint"); d; d = d.next_sibling("datapoint")) {
            Time time;
            pugi::xml_node time_node = d.child("time");
            const char *time_str = time_node.child_value();
            if (strlen(time_str) > 0) {
                int3 time_ = string2int3(time_str);
                if (time_.x < 0 || time_.x > 23) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid hour of " + std::to_string(time_.x) + " given in timeseries. Hour must be positive and not greater than 23.");
                } else if (time_.y < 0 || time_.y > 59) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid minute of " + std::to_string(time_.y) + " given in timeseries. Minute must be positive and not greater than 59.");
                } else if (time_.z < 0 || time_.z > 59) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid second of " + std::to_string(time_.z) + " given in timeseries. Second must be positive and not greater than 59.");
                }
                time = make_Time(time_.x, time_.y, time_.z);
            } else {
                helios_runtime_error("ERROR (Context::loadXML): No time was specified for timeseries datapoint.");
            }

            Date date;
            bool date_flag = false;

            pugi::xml_node date_node = d.child("date");
            const char *date_str = date_node.child_value();
            if (strlen(date_str) > 0) {
                int3 date_ = string2int3(date_str);
                if (date_.x < 1 || date_.x > 31) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid day of month " + std::to_string(date_.x) + " given in timeseries. Day must be greater than zero and not greater than 31.");
                } else if (date_.y < 1 || date_.y > 12) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid month of " + std::to_string(date_.y) + " given in timeseries. Month must be greater than zero and not greater than 12.");
                } else if (date_.z < 1000 || date_.z > 10000) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid year of " + std::to_string(date_.z) + " given in timeseries. Year should be in YYYY format.");
                }
                date = make_Date(date_.x, date_.y, date_.z);
                date_flag = true;
            }

            pugi::xml_node Jdate_node = d.child("dateJulian");
            const char *Jdate_str = Jdate_node.child_value();
            if (strlen(Jdate_str) > 0) {
                int2 date_ = string2int2(Jdate_str);
                if (date_.x < 1 || date_.x > 366) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid Julian day of year " + std::to_string(date_.x) + " given in timeseries. Julian day must be greater than zero and not greater than 366.");
                } else if (date_.y < 1000 || date_.y > 10000) {
                    helios_runtime_error("ERROR (Context::loadXML): Invalid year of " + std::to_string(date_.y) + " given in timeseries. Year should be in YYYY format.");
                }
                date = Julian2Calendar(date_.x, date_.y);
                date_flag = true;
            }

            if (!date_flag) {
                helios_runtime_error("ERROR (Context::loadXML): No date was specified for timeseries datapoint.");
            }

            float value;
            pugi::xml_node value_node = d.child("value");
            const char *value_str = value_node.child_value();
            if (strlen(value_str) > 0) {
                if (!parse_float(value_str, value)) {
                    helios_runtime_error("ERROR (Context::loadXML): Datapoint value in 'timeseries' block must be a float value.");
                }
            } else {
                helios_runtime_error("ERROR (Context::loadXML): No value was specified for timeseries datapoint.");
            }

            addTimeseriesData(label, value, date, time);
        }
    }

    if (!quiet) {
        std::cout << "done." << std::endl;
    }

    return UUID;
}

std::vector<std::string> Context::getLoadedXMLFiles() {
    return XMLfiles;
}

bool Context::scanXMLForTag(const std::string &filename, const std::string &tag, const std::string &label) {
    const std::string &fn = filename;
    std::string ext = getFileExtension(filename);
    if (ext != ".xml" && ext != ".XML") {
        helios_runtime_error("failed.\n File " + fn + " is not XML format.");
    }

    // Using "pugixml" parser.  See pugixml.org
    pugi::xml_document xmldoc;

    // load file
    pugi::xml_parse_result load_result = xmldoc.load_file(filename.c_str());

    // error checking
    if (!load_result) {
        helios_runtime_error("failed.\n XML [" + filename + "] parsed with errors, attr value: [" + xmldoc.child("node").attribute("attr").value() + "]\nError description: " + load_result.description() +
                             "\nError offset: " + std::to_string(load_result.offset) + " (error at [..." + (filename.c_str() + load_result.offset) + "]\n");
    }

    pugi::xml_node helios = xmldoc.child("helios");

    if (helios.empty()) {
        return false;
    }

    for (pugi::xml_node p = helios.child(tag.c_str()); p; p = p.next_sibling(tag.c_str())) {
        const char *labelquery = p.attribute("label").value();

        if (labelquery == label || label.empty()) {
            return true;
        }
    }

    return false;
}

void Context::writeDataToXMLstream(const char *data_group, const std::vector<std::string> &data_labels, void *ptr, std::ofstream &outfile) const {
    for (const auto &label: data_labels) {
        HeliosDataType dtype = HELIOS_TYPE_INT;

        if (strcmp(data_group, "primitive") == 0) {
            dtype = ((Primitive *) ptr)->getPrimitiveDataType(label.c_str());
        } else if (strcmp(data_group, "object") == 0) {
            dtype = ((CompoundObject *) ptr)->getObjectDataType(label.c_str());
        } else if (strcmp(data_group, "material") == 0) {
            dtype = ((Material *) ptr)->getMaterialDataType(label.c_str());
        } else if (strcmp(data_group, "global") == 0) {
            dtype = getGlobalDataType(label.c_str());
        } else {
            helios_runtime_error("ERROR (Context::writeDataToXMLstream): unknown data group argument of " + std::string(data_group) + ". Must be one of primitive, object, material, or global.");
        }

        if (dtype == HELIOS_TYPE_UINT) {
            outfile << "\t<data_uint label=\"" << label << "\">" << std::flush;
            std::vector<uint> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_uint>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT) {
            outfile << "\t<data_int label=\"" << label << "\">" << std::flush;
            std::vector<int> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int>" << std::endl;
        } else if (dtype == HELIOS_TYPE_FLOAT) {
            outfile << "\t<data_float label=\"" << label << "\">" << std::flush;
            std::vector<float> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_float>" << std::endl;
        } else if (dtype == HELIOS_TYPE_DOUBLE) {
            outfile << "\t<data_double label=\"" << label << "\">" << std::flush;
            std::vector<double> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_double>" << std::endl;
        } else if (dtype == HELIOS_TYPE_VEC2) {
            outfile << "\t<data_vec2 label=\"" << label << "\">" << std::flush;
            std::vector<vec2> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_vec2>" << std::endl;
        } else if (dtype == HELIOS_TYPE_VEC3) {
            outfile << "\t<data_vec3 label=\"" << label << "\">" << std::flush;
            std::vector<vec3> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_vec3>" << std::endl;
        } else if (dtype == HELIOS_TYPE_VEC4) {
            outfile << "\t<data_vec4 label=\"" << label << "\">" << std::flush;
            std::vector<vec4> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << " " << data.at(j).w << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_vec4>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT2) {
            outfile << "\t<data_int2 label=\"" << label << "\">" << std::flush;
            std::vector<int2> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int2>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT3) {
            outfile << "\t<data_int3 label=\"" << label << "\">" << std::flush;
            std::vector<int3> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int3>" << std::endl;
        } else if (dtype == HELIOS_TYPE_INT4) {
            outfile << "\t<data_int3 label=\"" << label << "\">" << std::flush;
            std::vector<int4> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j).x << " " << data.at(j).y << " " << data.at(j).z << " " << data.at(j).w << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_int4>" << std::endl;
        } else if (dtype == HELIOS_TYPE_STRING) {
            outfile << "\t<data_string label=\"" << label << "\">" << std::flush;
            std::vector<std::string> data;
            if (strcmp(data_group, "primitive") == 0) {
                ((Primitive *) ptr)->getPrimitiveData(label.c_str(), data);
            } else if (strcmp(data_group, "object") == 0) {
                ((CompoundObject *) ptr)->getObjectData(label.c_str(), data);
            } else if (strcmp(data_group, "material") == 0) {
                ((Material *) ptr)->getMaterialData(label.c_str(), data);
            } else {
                getGlobalData(label.c_str(), data);
            }
            for (int j = 0; j < data.size(); j++) {
                outfile << data.at(j) << std::flush;
                if (j != data.size() - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</data_string>" << std::endl;
        }
    }
}

void Context::writeXML(const char *filename, bool quiet) const {
    writeXML(filename, getAllUUIDs(), quiet);
}

void Context::writeXML_byobject(const char *filename, const std::vector<uint> &objIDs, bool quiet) const {
    for (uint objID: objIDs) {
        if (!doesObjectExist(objID)) {
            helios_runtime_error("ERROR (Context::writeXML_byobject): Object with ID of " + std::to_string(objID) + " does not exist.");
        }
    }
    writeXML(filename, getObjectPrimitiveUUIDs(objIDs), quiet);
}

void Context::writeXML(const char *filename, const std::vector<uint> &UUIDs, bool quiet) const {
    if (!quiet) {
        std::cout << "Writing XML file " << filename << "..." << std::flush;
    }

    std::string xmlfilename = filename;

    if (!validateOutputPath(xmlfilename)) {
        helios_runtime_error("ERROR (Context::writeXML): Invalid output file " + xmlfilename + ".");
    }

    if (getFileName(xmlfilename).empty()) {
        helios_runtime_error("ERROR (Context::writeXML): Invalid output file " + xmlfilename + ". No file name was provided.");
    }

    auto file_extension = getFileExtension(filename);
    if (file_extension != ".xml" && file_extension != ".XML") { // append xml to file name
        xmlfilename.append(".xml");
    }

    std::vector<uint> objectIDs = getUniquePrimitiveParentObjectIDs(UUIDs, false);

    std::ofstream outfile;
    outfile.open(xmlfilename);

    outfile << "<?xml version=\"1.0\"?>\n\n";

    outfile << "<helios>\n\n";

    // -- materials -- //

    // Collect unique material labels used by the primitives being written
    std::set<std::string> material_labels_used;
    for (uint UUID: UUIDs) {
        if (doesPrimitiveExist(UUID)) {
            uint matID = getPrimitivePointer_private(UUID)->materialID;
            if (materials.find(matID) != materials.end()) {
                material_labels_used.insert(materials.at(matID).label);
            }
        }
    }

    if (!material_labels_used.empty()) {
        outfile << "   <materials>" << std::endl;
        for (const std::string &label: material_labels_used) {
            if (doesMaterialExist(label)) {
                uint matID = getMaterialIDFromLabel(label);
                const Material &mat = materials.at(matID);
                outfile << "\t<material label=\"" << mat.label << "\">" << std::endl;
                outfile << "\t\t<color>" << mat.color.r << " " << mat.color.g << " " << mat.color.b << " " << mat.color.a << "</color>" << std::endl;
                if (!mat.texture_file.empty()) {
                    outfile << "\t\t<texture>" << mat.texture_file << "</texture>" << std::endl;
                }
                if (mat.texture_color_overridden) {
                    outfile << "\t\t<texture_override>1</texture_override>" << std::endl;
                }
                if (mat.twosided_flag != 1) { // Only write if non-default
                    outfile << "\t\t<twosided_flag>" << mat.twosided_flag << "</twosided_flag>" << std::endl;
                }
                // Write material data
                std::vector<std::string> mdata = mat.listMaterialData();
                if (!mdata.empty()) {
                    writeDataToXMLstream("material", mdata, const_cast<Material *>(&mat), outfile);
                }
                outfile << "\t</material>" << std::endl;
            }
        }
        outfile << "   </materials>\n" << std::endl;
    }

    // -- time/date -- //

    Date date = getDate();

    outfile << "   <date>" << std::endl;

    outfile << "\t<day>" << date.day << "</day>" << std::endl;
    outfile << "\t<month>" << date.month << "</month>" << std::endl;
    outfile << "\t<year>" << date.year << "</year>" << std::endl;

    outfile << "   </date>" << std::endl;

    Time time = getTime();

    outfile << "   <time>" << std::endl;

    outfile << "\t<hour>" << time.hour << "</hour>" << std::endl;
    outfile << "\t<minute>" << time.minute << "</minute>" << std::endl;
    outfile << "\t<second>" << time.second << "</second>" << std::endl;

    outfile << "   </time>" << std::endl;

    // -- primitives -- //

    for (uint UUID: UUIDs) {
        uint p = UUID;

        if (!doesPrimitiveExist(p)) {
            if (doesObjectExist(p)) {
                helios_runtime_error("ERROR (Context::writeXML): Primitive with UUID of " + std::to_string(p) + " does not exist. There is a compound object with this ID - did you mean to call Context::writeXML_byobject()?");
            } else {
                helios_runtime_error("ERROR (Context::writeXML): Primitive with UUID of " + std::to_string(p) + " does not exist.");
            }
        }

        Primitive *prim = getPrimitivePointer_private(p);

        uint parent_objID = prim->getParentObjectID();

        RGBAcolor color = prim->getColorRGBA();

        std::string texture_file = prim->getTextureFile();

        std::vector<std::string> pdata = prim->listPrimitiveData();

        // if this primitive is a member of a compound object that is "complete", don't write it to XML
        //\todo This was included to make the XML files more efficient and avoid writing all object primitives to file. However, it doesn't work in some cases because it makes it hard to figure out the primitive transformations.
        //     if( parent_objID>0 && areObjectPrimitivesComplete(parent_objID) ){
        //       continue;
        //     }

        if (prim->getType() == PRIMITIVE_TYPE_PATCH) {
            outfile << "   <patch>" << std::endl;
        } else if (prim->getType() == PRIMITIVE_TYPE_TRIANGLE) {
            outfile << "   <triangle>" << std::endl;
        } else if (prim->getType() == PRIMITIVE_TYPE_VOXEL) {
            outfile << "   <voxel>" << std::endl;
        }

        outfile << "\t<UUID>" << p << "</UUID>" << std::endl;

        if (parent_objID > 0) {
            outfile << "\t<objID>" << parent_objID << "</objID>" << std::endl;
        }

        // Write material label (v3 format)
        if (materials.find(prim->materialID) != materials.end()) {
            outfile << "\t<material>" << materials.at(prim->materialID).label << "</material>" << std::endl;
        }

        if (!pdata.empty()) {
            writeDataToXMLstream("primitive", pdata, prim, outfile);
        }

        // Patches
        if (prim->getType() == PRIMITIVE_TYPE_PATCH) {
            Patch *patch = getPatchPointer_private(p);
            float transform[16];
            prim->getTransformationMatrix(transform);

            outfile << "\t<transform>";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;
            std::vector<vec2> uv = patch->getTextureUV();
            if (!uv.empty()) {
                outfile << "\t<textureUV>" << std::flush;
                for (int i = 0; i < uv.size(); i++) {
                    outfile << uv.at(i).x << " " << uv.at(i).y << std::flush;
                    if (i != uv.size() - 1) {
                        outfile << " " << std::flush;
                    }
                }
                outfile << "</textureUV>" << std::endl;
            }
            if (primitiveTextureHasTransparencyChannel(p)) {
                outfile << "\t<solid_fraction>" << getPrimitiveSolidFraction(p) << "</solid_fraction>\n";
            }
            outfile << "   </patch>" << std::endl;

            // Triangles
        } else if (prim->getType() == PRIMITIVE_TYPE_TRIANGLE) {
            float transform[16];
            prim->getTransformationMatrix(transform);

            outfile << "\t<transform>";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            std::vector<vec2> uv = getTrianglePointer_private(p)->getTextureUV();
            if (!uv.empty()) {
                outfile << "\t<textureUV>" << std::flush;
                for (int i = 0; i < uv.size(); i++) {
                    outfile << uv.at(i).x << " " << uv.at(i).y << std::flush;
                    if (i != uv.size() - 1) {
                        outfile << " " << std::flush;
                    }
                }
                outfile << "</textureUV>" << std::endl;
            }
            if (primitiveTextureHasTransparencyChannel(p)) {
                outfile << "\t<solid_fraction>" << getPrimitiveSolidFraction(p) << "</solid_fraction>\n";
            }
            outfile << "   </triangle>" << std::endl;

            // Voxels
        } else if (prim->getType() == PRIMITIVE_TYPE_VOXEL) {
            float transform[16];
            prim->getTransformationMatrix(transform);

            outfile << "\t<transform>";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;
            if (primitiveTextureHasTransparencyChannel(p)) {
                outfile << "\t<solid_fraction>" << getPrimitiveSolidFraction(p) << "</solid_fraction>\n";
            }

            outfile << "   </voxel>" << std::endl;
        }
    }

    // -- objects -- //

    for (auto o: objectIDs) {
        CompoundObject *obj = objects.at(o);

        std::string texture_file = obj->getTextureFile();

        std::vector<std::string> odata = obj->listObjectData();

        if (obj->getObjectType() == OBJECT_TYPE_TILE) {
            outfile << "   <tile>" << std::endl;
        } else if (obj->getObjectType() == OBJECT_TYPE_BOX) {
            outfile << "   <box>" << std::endl;
        } else if (obj->getObjectType() == OBJECT_TYPE_CONE) {
            outfile << "   <cone>" << std::endl;
        } else if (obj->getObjectType() == OBJECT_TYPE_DISK) {
            outfile << "   <disk>" << std::endl;
        } else if (obj->getObjectType() == OBJECT_TYPE_SPHERE) {
            outfile << "   <sphere>" << std::endl;
        } else if (obj->getObjectType() == OBJECT_TYPE_TUBE) {
            outfile << "   <tube>" << std::endl;
        } else if (obj->getObjectType() == OBJECT_TYPE_POLYMESH) {
            outfile << "   <polymesh>" << std::endl;
        }

        outfile << "\t<objID>" << o << "</objID>" << std::endl;
        if (obj->hasTexture()) {
            outfile << "\t<texture>" << texture_file << "</texture>" << std::endl;
        }

        if (!odata.empty()) {
            writeDataToXMLstream("object", odata, obj, outfile);
        }

        std::vector<std::string> pdata_labels;
        std::vector<HeliosDataType> pdata_types;
        std::vector<uint> primitiveUUIDs = obj->getPrimitiveUUIDs();
        for (uint UUID: primitiveUUIDs) {
            std::vector<std::string> labels = getPrimitivePointer_private(UUID)->listPrimitiveData();
            for (const auto &label: labels) {
                if (find(pdata_labels.begin(), pdata_labels.end(), label) == pdata_labels.end()) {
                    pdata_labels.push_back(label);
                    pdata_types.push_back(getPrimitiveDataType(label.c_str()));
                }
            }
        }
        for (size_t l = 0; l < pdata_labels.size(); l++) {
            if (pdata_types.at(l) == HELIOS_TYPE_FLOAT) {
                outfile << "\t<primitive_data_float " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<float> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (float i: data) {
                            outfile << i << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_float>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_DOUBLE) {
                outfile << "\t<primitive_data_double " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<double> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (double i: data) {
                            outfile << i << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_double>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_UINT) {
                outfile << "\t<primitive_data_uint " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<uint> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (unsigned int i: data) {
                            outfile << i << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_uint>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_INT) {
                outfile << "\t<primitive_data_int " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (int i: data) {
                            outfile << i << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_INT2) {
                outfile << "\t<primitive_data_int2 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int2> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (auto &i: data) {
                            outfile << i.x << " " << i.y << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int2>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_INT3) {
                outfile << "\t<primitive_data_int3 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int3> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (auto &i: data) {
                            outfile << i.x << " " << i.y << " " << i.z << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int3>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_INT4) {
                outfile << "\t<primitive_data_int4 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<int4> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (auto &i: data) {
                            outfile << i.x << " " << i.y << " " << i.z << " " << i.w << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_int4>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_VEC2) {
                outfile << "\t<primitive_data_vec2 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<vec2> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (auto &i: data) {
                            outfile << i.x << " " << i.y << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_vec2>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_VEC3) {
                outfile << "\t<primitive_data_vec3 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<vec3> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (auto &i: data) {
                            outfile << i.x << " " << i.y << " " << i.z << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_vec3>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_VEC4) {
                outfile << "\t<primitive_data_vec4 " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<vec4> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (auto &i: data) {
                            outfile << i.x << " " << i.y << " " << i.z << " " << i.w << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_vec4>" << std::endl;
            } else if (pdata_types.at(l) == HELIOS_TYPE_STRING) {
                outfile << "\t<primitive_data_string " << "label=\"" << pdata_labels.at(l) << "\">" << std::endl;
                for (size_t p = 0; p < primitiveUUIDs.size(); p++) {
                    if (doesPrimitiveDataExist(primitiveUUIDs.at(p), pdata_labels.at(l).c_str())) {
                        std::vector<std::string> data;
                        getPrimitiveData(primitiveUUIDs.at(p), pdata_labels.at(l).c_str(), data);
                        outfile << "\t\t<data label=\"" << p << "\"> " << std::flush;
                        for (const auto &i: data) {
                            outfile << i << std::flush;
                        }
                        outfile << " </data>" << std::endl;
                    }
                }
                outfile << "\t</primitive_data_string>" << std::endl;
            }
        }

        // Tiles
        if (obj->getObjectType() == OBJECT_TYPE_TILE) {
            Tile *tile = getTileObjectPointer_private(o);

            float transform[16];
            tile->getTransformationMatrix(transform);

            int2 subdiv = tile->getSubdivisionCount();
            outfile << "\t<subdivisions>" << subdiv.x << " " << subdiv.y << "</subdivisions>" << std::endl;

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            outfile << "   </tile>" << std::endl;

            // Spheres
        } else if (obj->getObjectType() == OBJECT_TYPE_SPHERE) {
            Sphere *sphere = getSphereObjectPointer_private(o);

            float transform[16];
            sphere->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            uint subdiv = sphere->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv << " </subdivisions>" << std::endl;

            outfile << "   </sphere>" << std::endl;

            // Tubes
        } else if (obj->getObjectType() == OBJECT_TYPE_TUBE) {
            Tube *tube = getTubeObjectPointer_private(o);

            float transform[16];
            tube->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            uint subdiv = tube->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv << " </subdivisions>" << std::endl;

            std::vector<vec3> nodes = tube->getNodes();
            std::vector<float> radius = tube->getNodeRadii();

            assert(nodes.size() == radius.size());
            outfile << "\t<nodes> " << std::endl;
            for (auto &node: nodes) {
                outfile << "\t\t" << node.x << " " << node.y << " " << node.z << std::endl;
            }
            outfile << "\t</nodes> " << std::endl;
            outfile << "\t<radius> " << std::endl;
            for (float radiu: radius) {
                outfile << "\t\t" << radiu << std::endl;
            }
            outfile << "\t</radius> " << std::endl;

            if (texture_file.empty()) {
                std::vector<RGBcolor> colors = tube->getNodeColors();

                outfile << "\t<color> " << std::endl;
                for (auto &color: colors) {
                    outfile << "\t\t" << color.r << " " << color.g << " " << color.b << std::endl;
                }
                outfile << "\t</color> " << std::endl;
            }

            outfile << "   </tube>" << std::endl;

            // Boxes
        } else if (obj->getObjectType() == OBJECT_TYPE_BOX) {
            Box *box = getBoxObjectPointer_private(o);

            float transform[16];
            box->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            int3 subdiv = box->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv.x << " " << subdiv.y << " " << subdiv.z << " </subdivisions>" << std::endl;

            outfile << "   </box>" << std::endl;

            // Disks
        } else if (obj->getObjectType() == OBJECT_TYPE_DISK) {
            Disk *disk = getDiskObjectPointer_private(o);

            float transform[16];
            disk->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            int2 subdiv = disk->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv.x << " " << subdiv.y << " </subdivisions>" << std::endl;

            outfile << "   </disk>" << std::endl;

            // Cones
        } else if (obj->getObjectType() == OBJECT_TYPE_CONE) {
            Cone *cone = getConeObjectPointer_private(o);

            float transform[16];
            cone->getTransformationMatrix(transform);

            outfile << "\t<transform> ";
            for (float i: transform) {
                outfile << i << " ";
            }
            outfile << "</transform>" << std::endl;

            uint subdiv = cone->getSubdivisionCount();
            outfile << "\t<subdivisions> " << subdiv << " </subdivisions>" << std::endl;

            std::vector<vec3> nodes = cone->getNodeCoordinates();
            std::vector<float> radius = cone->getNodeRadii();

            assert(nodes.size() == radius.size());
            outfile << "\t<nodes> " << std::endl;
            for (auto &node: nodes) {
                outfile << "\t\t" << node.x << " " << node.y << " " << node.z << std::endl;
            }
            outfile << "\t</nodes> " << std::endl;
            outfile << "\t<radius> " << std::endl;
            for (float radiu: radius) {
                outfile << "\t\t" << radiu << std::endl;
            }
            outfile << "\t</radius> " << std::endl;

            outfile << "   </cone>" << std::endl;

            // Polymesh
        } else if (obj->getObjectType() == OBJECT_TYPE_POLYMESH) {
            outfile << "   </polymesh>" << std::endl;
        }
    }


    // -- global data -- //

    for (const auto &iter: globaldata) {
        std::string label = iter.first;
        GlobalData data = iter.second;
        HeliosDataType type = data.type;
        if (type == HELIOS_TYPE_UINT) {
            outfile << "   <globaldata_uint label=\"" << label << "\">" << std::flush;
            for (size_t i = 0; i < data.size; i++) {
                outfile << data.global_data_uint.at(i) << std::flush;
                if (i != data.size - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</globaldata_uint>" << std::endl;
        } else if (type == HELIOS_TYPE_INT) {
            outfile << "   <globaldata_int label=\"" << label << "\">" << std::flush;
            for (size_t i = 0; i < data.size; i++) {
                outfile << data.global_data_int.at(i) << std::flush;
                if (i != data.size - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</globaldata_int>" << std::endl;
        } else if (type == HELIOS_TYPE_FLOAT) {
            outfile << "   <globaldata_float label=\"" << label << "\">" << std::flush;
            for (size_t i = 0; i < data.size; i++) {
                outfile << data.global_data_float.at(i) << std::flush;
                if (i != data.size - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</globaldata_float>" << std::endl;
        } else if (type == HELIOS_TYPE_DOUBLE) {
            outfile << "   <globaldata_double label=\"" << label << "\">" << std::flush;
            for (size_t i = 0; i < data.size; i++) {
                outfile << data.global_data_double.at(i) << std::flush;
                if (i != data.size - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</globaldata_double>" << std::endl;
        } else if (type == HELIOS_TYPE_VEC2) {
            outfile << "   <globaldata_vec2 label=\"" << label << "\">" << std::endl;
            for (size_t i = 0; i < data.size; i++) {
                outfile << "      " << data.global_data_vec2.at(i).x << " " << data.global_data_vec2.at(i).y << std::endl;
            }
            outfile << "   </globaldata_vec2>" << std::endl;
        } else if (type == HELIOS_TYPE_VEC3) {
            outfile << "   <globaldata_vec3 label=\"" << label << "\">" << std::endl;
            for (size_t i = 0; i < data.size; i++) {
                outfile << "      " << data.global_data_vec3.at(i).x << " " << data.global_data_vec3.at(i).y << " " << data.global_data_vec3.at(i).z << std::endl;
            }
            outfile << "   </globaldata_vec3>" << std::endl;
        } else if (type == HELIOS_TYPE_VEC4) {
            outfile << "   <globaldata_vec4 label=\"" << label << "\">" << std::endl;
            for (size_t i = 0; i < data.size; i++) {
                outfile << "      " << data.global_data_vec4.at(i).x << " " << data.global_data_vec4.at(i).y << " " << data.global_data_vec4.at(i).z << " " << data.global_data_vec4.at(i).w << std::endl;
            }
            outfile << "   </globaldata_vec4>" << std::endl;
        } else if (type == HELIOS_TYPE_INT2) {
            outfile << "   <globaldata_int2 label=\"" << label << "\">" << std::endl;
            for (size_t i = 0; i < data.size; i++) {
                outfile << "      " << data.global_data_int2.at(i).x << " " << data.global_data_int2.at(i).y << std::endl;
            }
            outfile << "   </globaldata_int2>" << std::endl;
        } else if (type == HELIOS_TYPE_INT3) {
            outfile << "   <globaldata_int3 label=\"" << label << "\">" << std::endl;
            for (size_t i = 0; i < data.size; i++) {
                outfile << "      " << data.global_data_int3.at(i).x << " " << data.global_data_int3.at(i).y << " " << data.global_data_int3.at(i).z << std::endl;
            }
            outfile << "   </globaldata_int3>" << std::endl;
        } else if (type == HELIOS_TYPE_INT4) {
            outfile << "   <globaldata_int4 label=\"" << label << "\">" << std::endl;
            for (size_t i = 0; i < data.size; i++) {
                outfile << "      " << data.global_data_int4.at(i).x << " " << data.global_data_int4.at(i).y << " " << data.global_data_int4.at(i).z << " " << data.global_data_int4.at(i).w << std::endl;
            }
            outfile << "   </globaldata_int4>" << std::endl;
        } else if (type == HELIOS_TYPE_STRING) {
            outfile << "   <globaldata_string label=\"" << label << "\">" << std::flush;
            for (size_t i = 0; i < data.size; i++) {
                outfile << data.global_data_string.at(i) << std::flush;
                if (i != data.size - 1) {
                    outfile << " " << std::flush;
                }
            }
            outfile << "</globaldata_string>" << std::endl;
        }
    }

    // -- timeseries -- //

    for (const auto &iter: timeseries_data) {
        std::string label = iter.first;

        std::vector<float> data = iter.second;
        std::vector<double> dateval = timeseries_datevalue.at(label);

        assert(data.size() == dateval.size());

        outfile << "   <timeseries label=\"" << label << "\">" << std::endl;

        for (size_t i = 0; i < data.size(); i++) {
            Date a_date = queryTimeseriesDate(label.c_str(), i);
            Time a_time = queryTimeseriesTime(label.c_str(), i);

            outfile << "\t<datapoint>" << std::endl;

            outfile << "\t   <date>" << a_date.day << " " << a_date.month << " " << a_date.year << "</date>" << std::endl;

            outfile << "\t   <time>" << a_time.hour << " " << a_time.minute << " " << a_time.second << "</time>" << std::endl;

            outfile << "\t   <value>" << data.at(i) << "</value>" << std::endl;

            outfile << "\t</datapoint>" << std::endl;
        }

        outfile << "   </timeseries>" << std::endl;
    }

    // ----------------- //

    outfile << "\n</helios>\n";

    outfile.close();

    if (!quiet) {
        std::cout << "done." << std::endl;
    }
}

std::vector<uint> Context::loadPLY(const char *filename, bool silent) {
    return loadPLY(filename, nullorigin, 0, nullrotation, RGB::blue, "YUP", silent);
}

std::vector<uint> Context::loadPLY(const char *filename, const vec3 &origin, float height, const std::string &upaxis, bool silent) {
    return loadPLY(filename, origin, height, make_SphericalCoord(0, 0), RGB::blue, upaxis, silent);
}

std::vector<uint> Context::loadPLY(const char *filename, const vec3 &origin, float height, const SphericalCoord &rotation, const std::string &upaxis, bool silent) {
    return loadPLY(filename, origin, height, rotation, RGB::blue, upaxis, silent);
}

std::vector<uint> Context::loadPLY(const char *filename, const vec3 &origin, float height, const RGBcolor &default_color, const std::string &upaxis, bool silent) {
    return loadPLY(filename, origin, height, make_SphericalCoord(0, 0), default_color, upaxis, silent);
}

std::vector<uint> Context::loadPLY(const char *filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color, const std::string &upaxis, bool silent) {
    if (!silent) {
        std::cout << "Reading PLY file " << filename << "..." << std::flush;
    }

    std::string fn = filename;
    std::string ext = getFileExtension(filename);
    if (ext != ".ply" && ext != ".PLY") {
        helios_runtime_error("ERROR (Context::loadPLY): File " + fn + " is not PLY format.");
    }

    if (upaxis != "XUP" && upaxis != "YUP" && upaxis != "ZUP") {
        helios_runtime_error("ERROR (Context::loadPLY): " + upaxis + " is not a valid up-axis. Please specify a value of XUP, YUP, or ZUP.");
    }

    std::string line, prop;

    uint vertexCount = 0, faceCount = 0;

    std::vector<vec3> vertices;
    std::vector<std::vector<int>> faces;
    std::vector<RGBcolor> colors;
    std::vector<std::string> properties;

    bool ifColor = false;

    // Resolve file path using unified resolution
    std::filesystem::path resolved_path = resolveFilePath(filename);
    std::string resolved_filename = resolved_path.string();

    std::ifstream inputPly;
    inputPly.open(resolved_filename);

    if (!inputPly.is_open()) {
        helios_runtime_error("ERROR (Context::loadPLY): Couldn't open " + std::string(filename));
    }

    //--- read header info -----//

    // first line should always be 'ply'
    inputPly >> line;
    if ("ply" != line) {
        helios_runtime_error("ERROR (Context::loadPLY): " + std::string(filename) + " is not a PLY file.");
    }

    // read format
    inputPly >> line;
    if ("format" != line) {
        helios_runtime_error("ERROR (Context::loadPLY): could not determine data format of " + std::string(filename));
    }

    inputPly >> line;
    if ("ascii" != line) {
        helios_runtime_error("ERROR (Context::loadPLY): Only ASCII data types are supported.");
    }

    std::string temp_string;

    while ("end_header" != line) {
        inputPly >> line;

        if ("comment" == line) {
            getline(inputPly, line);
        } else if ("element" == line) {
            inputPly >> line;

            if ("vertex" == line) {
                inputPly >> temp_string;
                if (!parse_uint(temp_string, vertexCount)) {
                    helios_runtime_error("ERROR (Context::loadPLY): PLY file read failed. Vertex count value should be a non-negative integer.");
                }
            } else if ("face" == line) {
                inputPly >> temp_string;
                if (!parse_uint(temp_string, faceCount)) {
                    helios_runtime_error("ERROR (Context::loadPLY): PLY file read failed. Face count value should be a non-negative integer.");
                }
            }
        } else if ("property" == line) {
            inputPly >> line; // type

            if ("list" != line) {
                inputPly >> prop; // value
                properties.push_back(prop);
            }
        }
    }

    for (auto &property: properties) {
        if (property == "red") {
            ifColor = true;
        }
    }
    if (!silent) {
        std::cout << "forming " << faceCount << " triangles..." << std::flush;
    }

    vertices.resize(vertexCount);
    colors.resize(vertexCount);
    faces.resize(faceCount);


    //--- read vertices ----//

    for (uint row = 0; row < vertexCount; row++) {
        for (auto &property: properties) {
            if (property == "x") {
                inputPly >> temp_string;
                float x;
                if (!parse_float(temp_string, x)) {
                    helios_runtime_error("ERROR (Context::loadPLY): X value for vertex " + std::to_string(row) + " is invalid and could not be read.");
                }
                if (upaxis == "XUP") {
                    vertices.at(row).z = x;
                } else if (upaxis == "YUP") {
                    vertices.at(row).y = x;
                } else if (upaxis == "ZUP") {
                    vertices.at(row).x = x;
                }
            } else if (property == "y") {
                inputPly >> temp_string;
                float y;
                if (!parse_float(temp_string, y)) {
                    helios_runtime_error("ERROR (Context::loadPLY): Y value for vertex " + std::to_string(row) + " is invalid and could not be read.");
                }
                if (upaxis == "XUP") {
                    vertices.at(row).x = y;
                } else if (upaxis == "YUP") {
                    vertices.at(row).z = y;
                } else if (upaxis == "ZUP") {
                    vertices.at(row).y = y;
                }
            } else if (property == "z") {
                inputPly >> temp_string;
                float z;
                if (!parse_float(temp_string, z)) {
                    helios_runtime_error("ERROR (Context::loadPLY): Z value for vertex " + std::to_string(row) + " is invalid and could not be read.");
                }
                if (upaxis == "XUP") {
                    vertices.at(row).y = z;
                } else if (upaxis == "YUP") {
                    vertices.at(row).x = z;
                } else if (upaxis == "ZUP") {
                    vertices.at(row).z = z;
                }
            } else if (property == "red") {
                inputPly >> temp_string;
                if (!parse_float(temp_string, colors.at(row).r)) {
                    helios_runtime_error("ERROR (Context::loadPLY): red color value for vertex " + std::to_string(row) + " is invalid and could not be read.");
                }
                colors.at(row).r /= 255.f;
            } else if (property == "green") {
                inputPly >> temp_string;
                if (!parse_float(temp_string, colors.at(row).g)) {
                    helios_runtime_error("ERROR (Context::loadPLY): green color value for vertex " + std::to_string(row) + " is invalid and could not be read.");
                }
                colors.at(row).g /= 255.f;
            } else if (property == "blue") {
                inputPly >> temp_string;
                if (!parse_float(temp_string, colors.at(row).b)) {
                    helios_runtime_error("ERROR (Context::loadPLY): blue color value for vertex " + std::to_string(row) + " is invalid and could not be read.");
                }
                colors.at(row).b /= 255.f;
            } else {
                inputPly >> line;
            }
        }

        if (inputPly.eof()) {
            helios_runtime_error("ERROR (Context::loadPLY): Read past end of file while reading vertices. Vertex count specified in header may be incorrect.");
        }
    }

    // determine bounding box

    vec3 boxmin = make_vec3(10000, 10000, 10000);
    vec3 boxmax = make_vec3(-10000, -10000, -10000);

    for (uint row = 0; row < vertexCount; row++) {
        if (vertices.at(row).x < boxmin.x) {
            boxmin.x = vertices.at(row).x;
        }
        if (vertices.at(row).y < boxmin.y) {
            boxmin.y = vertices.at(row).y;
        }
        if (vertices.at(row).z < boxmin.z) {
            boxmin.z = vertices.at(row).z;
        }

        if (vertices.at(row).x > boxmax.x) {
            boxmax.x = vertices.at(row).x;
        }
        if (vertices.at(row).y > boxmax.y) {
            boxmax.y = vertices.at(row).y;
        }
        if (vertices.at(row).z > boxmax.z) {
            boxmax.z = vertices.at(row).z;
        }
    }

    // center PLY object at `origin' and scale to have height `height'
    float scl = 1.f;
    if (height > 0.f) {
        scl = height / (boxmax.z - boxmin.z);
    }
    for (uint row = 0; row < vertexCount; row++) {
        vertices.at(row).z -= boxmin.z;

        vertices.at(row).x *= scl;
        vertices.at(row).y *= scl;
        vertices.at(row).z *= scl;

        vertices.at(row) = rotatePoint(vertices.at(row), rotation) + origin;
    }

    //--- read faces ----//

    uint v, ID;
    std::vector<uint> UUID;
    for (uint row = 0; row < faceCount; row++) {
        inputPly >> temp_string;

        if (!parse_uint(temp_string, v)) {
            helios_runtime_error("ERROR (Context::loadPLY): Vertex count for face " + std::to_string(row) + " should be a non-negative integer.");
        }

        faces.at(row).resize(v);

        for (uint i = 0; i < v; i++) {
            inputPly >> temp_string;
            if (!parse_int(temp_string, faces.at(row).at(i))) {
                helios_runtime_error("ERROR (Context::loadPLY): Vertex index for face " + std::to_string(row) + " is invalid and could not be read.");
            }
        }

        // Add triangles to context

        for (uint t = 2; t < v; t++) {
            RGBcolor color;
            if (ifColor) {
                color = colors.at(faces.at(row).front());
            } else {
                color = default_color;
            }

            vec3 v0 = vertices.at(faces.at(row).front());
            vec3 v1 = vertices.at(faces.at(row).at(t - 1));
            vec3 v2 = vertices.at(faces.at(row).at(t));

            if ((v0 - v1).magnitude() < 1e-10f || (v0 - v2).magnitude() < 1e-10f || (v1 - v2).magnitude() < 1e-10f) {
                continue;
            }

            // Additional check for triangle area to avoid near-degenerate triangles
            float triangle_area = calculateTriangleArea(v0, v1, v2);
            if (triangle_area < MIN_TRIANGLE_AREA_THRESHOLD) {
                continue;
            }

            ID = addTriangle(v0, v1, v2, color);

            UUID.push_back(ID);
        }

        if (inputPly.eof()) {
            helios_runtime_error("ERROR (Context::loadPLY): Read past end of file while reading faces. Face count specified in header may be incorrect.");
        }
    }

    if (!silent) {
        std::cout << "done." << std::endl;
    }

    return UUID;
}

void Context::writePLY(const char *filename) const {
    writePLY(filename, getAllUUIDs());
}

void Context::writePLY(const char *filename, const std::vector<uint> &UUIDs) const {
    // Validate file name / extension
    std::string fname{filename ? filename : ""};

    const auto dotPos = fname.find_last_of('.');
    const std::string ext = (dotPos != std::string::npos) ? fname.substr(dotPos) : "";

    auto ciEqual = [](const char a, const char b) { return std::tolower(a) == std::tolower(b); };
    bool isPly = (ext.size() == 4) && ciEqual(ext[1], 'p') && ciEqual(ext[2], 'l') && ciEqual(ext[3], 'y');

    if (!isPly) {
        helios_runtime_error("ERROR (Context::writePLY) Invalid file extension for " + fname + ". Expected a file ending in '.ply'.");
    }

    // Try to open the output file
    std::ofstream PLYfile;
    PLYfile.open(fname, std::ios::out | std::ios::trunc);

    if (!PLYfile.is_open()) {
        helios_runtime_error("ERROR (Context::writePLY) Unable to open " + fname + " for writing.");
    }

    PLYfile << "ply" << std::endl << "format ascii 1.0" << std::endl << "comment Helios generated" << std::endl;

    std::vector<int3> faces;
    std::vector<vec3> verts;
    std::vector<RGBcolor> colors;

    size_t vertex_count = 0;

    for (auto UUID: UUIDs) {
        std::vector<vec3> vertices = getPrimitivePointer_private(UUID)->getVertices();
        PrimitiveType type = getPrimitivePointer_private(UUID)->getType();
        RGBcolor C = getPrimitivePointer_private(UUID)->getColor();
        C.scale(255.f);

        if (type == PRIMITIVE_TYPE_TRIANGLE) {
            faces.push_back(make_int3((int) vertex_count, (int) vertex_count + 1, (int) vertex_count + 2));
            for (int i = 0; i < 3; i++) {
                verts.push_back(vertices.at(i));
                colors.push_back(C);
                vertex_count++;
            }
        } else if (type == PRIMITIVE_TYPE_PATCH) {
            faces.push_back(make_int3((int) vertex_count, (int) vertex_count + 1, (int) vertex_count + 2));
            faces.push_back(make_int3((int) vertex_count, (int) vertex_count + 2, (int) vertex_count + 3));
            for (int i = 0; i < 4; i++) {
                verts.push_back(vertices.at(i));
                colors.push_back(C);
                vertex_count++;
            }
        }
    }

    PLYfile << "element vertex " << verts.size() << std::endl;
    PLYfile << "property float x" << std::endl << "property float y" << std::endl << "property float z" << std::endl;
    PLYfile << "property uchar red" << std::endl << "property uchar green" << std::endl << "property uchar blue" << std::endl;
    PLYfile << "element face " << faces.size() << std::endl;
    PLYfile << "property list uchar int vertex_indices" << std::endl << "end_header" << std::endl;

    for (size_t v = 0; v < verts.size(); v++) {
        PLYfile << verts.at(v).x << " " << verts.at(v).y << " " << verts.at(v).z << " " << round(colors.at(v).r) << " " << round(colors.at(v).g) << " " << round(colors.at(v).b) << std::endl;
    }

    for (auto &face: faces) {
        PLYfile << "3 " << face.x << " " << face.y << " " << face.z << std::endl;
    }

    PLYfile.close();
}

std::vector<uint> Context::loadOBJ(const char *filename, bool silent) {
    return loadOBJ(filename, nullorigin, 0, nullrotation, RGB::blue, "ZUP", silent);
}

std::vector<uint> Context::loadOBJ(const char *filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color, bool silent) {
    return loadOBJ(filename, origin, make_vec3(0, 0, height), rotation, default_color, "ZUP", silent);
}

std::vector<uint> Context::loadOBJ(const char *filename, const vec3 &origin, float height, const SphericalCoord &rotation, const RGBcolor &default_color, const char *upaxis, bool silent) {
    return loadOBJ(filename, origin, make_vec3(0, 0, height), rotation, default_color, upaxis, silent);
}

std::vector<uint> Context::loadOBJ(const char *filename, const vec3 &origin, const helios::vec3 &scale, const SphericalCoord &rotation, const RGBcolor &default_color, const char *upaxis, bool silent) {

    if (!silent) {
        std::cout << "Reading OBJ file " << filename << "..." << std::flush;
    }

    std::string fn = filename;
    std::string ext = getFileExtension(filename);
    if (ext != ".obj" && ext != ".OBJ") {
        helios_runtime_error("ERROR (Context::loadOBJ): File " + fn + " is not OBJ format.");
    }

    if (strcmp(upaxis, "XUP") != 0 && strcmp(upaxis, "YUP") != 0 && strcmp(upaxis, "ZUP") != 0) {
        helios_runtime_error("ERROR (Context::loadOBJ): Up axis of " + std::string(upaxis) + " is not valid.  Should be one of 'XUP', 'YUP', or 'ZUP'.");
    }

    std::string line, prop;

    std::vector<vec3> vertices;
    std::vector<std::string> objects;
    std::vector<vec2> texture_uv;
    std::map<std::string, std::vector<std::vector<int>>> face_inds, texture_inds;

    std::map<std::string, OBJmaterial> materials;

    std::vector<uint> UUID;

    // Resolve file path using unified resolution
    std::filesystem::path resolved_path = resolveFilePath(filename);
    std::string resolved_filename = resolved_path.string();

    std::ifstream inputOBJ, inputMTL;
    inputOBJ.open(resolved_filename);

    if (!inputOBJ.is_open()) {
        helios_runtime_error("ERROR (Context::loadOBJ): Couldn't open " + std::string(filename));
    }

    // determine the base file path for resolved filename
    std::string filebase = getFilePath(resolved_filename);

    // determine bounding box
    float boxmin = 100000;
    float boxmax = -100000;

    std::string current_material = "none";
    std::string current_object = "none";

    size_t lineno = 0;
    while (inputOBJ.good()) {
        lineno++;

        inputOBJ >> line;

        // ------- COMMENTS --------- //
        if (line == "#") {
            getline(inputOBJ, line);

            // ------- MATERIAL LIBRARY ------- //
        } else if (line == "mtllib") {
            getline(inputOBJ, line);
            std::string material_file = trim_whitespace(line);
            materials = loadMTL(filebase, material_file, default_color);

            // ------- OBJECT ------- //
        } else if (line == "o") {
            getline(inputOBJ, line);
            current_object = trim_whitespace(line);

            // ------- VERTICES --------- //
        } else if (line == "v") {
            getline(inputOBJ, line);
            // parse vertices into points
            vec3 verts(string2vec3(line.c_str()));
            vertices.emplace_back(verts);
            objects.emplace_back(current_object);

            if (verts.z < boxmin) {
                boxmin = verts.z;
            }
            if (verts.z > boxmax) {
                boxmax = verts.z;
            }

            // ------- TEXTURE COORDINATES --------- //
        } else if (line == "vt") {
            getline(inputOBJ, line);
            line = trim_whitespace(line);
            // parse coordinates into uv
            vec2 uv(string2vec2(line.c_str()));
            texture_uv.emplace_back(uv);

            // ------- MATERIALS --------- //
        } else if (line == "usemtl") {
            getline(inputOBJ, line);
            current_material = trim_whitespace(line);

            // ------- FACES --------- //
        } else if (line == "f") {
            getline(inputOBJ, line);
            // parse face vertices
            std::istringstream stream(line);
            std::string tmp, digitf, digitu;
            std::vector<int> f, u;
            while (stream.good()) {
                stream >> tmp;

                digitf = "";
                int ic = 0;
                for (char i: tmp) {
                    if (isdigit(i)) {
                        digitf.push_back(i);
                        ic++;
                    } else {
                        break;
                    }
                }

                digitu = "";
                for (int i = ic + 1; i < tmp.size(); i++) {
                    if (isdigit(tmp[i])) {
                        digitu.push_back(tmp[i]);
                    } else {
                        break;
                    }
                }

                if (!digitf.empty()) {
                    int face;
                    if (!parse_int(digitf, face)) {
                        helios_runtime_error("ERROR (Context::loadOBJ): Face index on line " + std::to_string(lineno) + " must be a non-negative integer value.");
                    }
                    // Add bounds checking for face indices
                    if (face <= 0 || face > vertices.size()) {
                        helios_runtime_error("ERROR (Context::loadOBJ): Face vertex index " + std::to_string(face) + " on line " + std::to_string(lineno) + " is out of range. Valid range is 1-" + std::to_string(vertices.size()) +
                                             ". Check that vertex indices in face definitions reference existing vertices.");
                    }
                    f.push_back(face);
                }
                if (!digitu.empty()) {
                    int uv;
                    if (!parse_int(digitu, uv)) {
                        helios_runtime_error("ERROR (Context::loadOBJ): u,v index on line " + std::to_string(lineno) + " must be a non-negative integer value.");
                    }
                    // Add bounds checking for UV indices
                    if (uv <= 0 || uv > texture_uv.size()) {
                        helios_runtime_error("ERROR (Context::loadOBJ): Texture coordinate index " + std::to_string(uv) + " on line " + std::to_string(lineno) + " is out of range. Valid range is 1-" + std::to_string(texture_uv.size()) +
                                             ". Check that texture coordinate indices in face definitions reference existing texture coordinates.");
                    }
                    u.push_back(uv);
                }
            }
            face_inds[current_material].push_back(f);
            texture_inds[current_material].push_back(u);

            // ------ OTHER STUFF --------- //
        } else {
            getline(inputOBJ, line);
        }
    }

    vec3 scl = scale;
    if (scl.x == 0 && scl.y == 0 && scl.z > 0) {
        if (boxmax - boxmin > 1e-6f) {
            scl = make_vec3(scale.z / (boxmax - boxmin), scale.z / (boxmax - boxmin), scale.z / (boxmax - boxmin));
        } else {
            // Object is flat or has zero height - use uniform scaling of requested height
            scl = make_vec3(scale.z, scale.z, scale.z);
        }
    } else {
        if (scl.x == 0 && (scl.y != 0 || scl.z != 0)) {
            std::cout << "WARNING (Context::loadOBJ): Scaling factor given for x-direction is zero. Setting scaling factor to 1" << std::endl;
        }
        if (scl.y == 0 && (scl.x != 0 || scl.z != 0)) {
            std::cout << "WARNING (Context::loadOBJ): Scaling factor given for y-direction is zero. Setting scaling factor to 1" << std::endl;
        }
        if (scl.z == 0 && (scl.x != 0 || scl.y != 0)) {
            std::cout << "WARNING (Context::loadOBJ): Scaling factor given for z-direction is zero. Setting scaling factor to 1" << std::endl;
        }

        if (scl.x == 0) {
            scl.x = 1;
        }
        if (scl.y == 0) {
            scl.y = 1;
        }
        if (scl.z == 0) {
            scl.z = 1;
        }
    }

    // Structure to hold triangle data for parallel processing
    struct TriangleData {
        vec3 vert0, vert1, vert2;
        std::string texture;
        vec2 uv0, uv1, uv2;
        RGBcolor color;
        bool hasTexture;
        bool textureColorIsOverridden;
        std::string object;
    };

    std::vector<TriangleData> triangleDataList;

    // First pass: Parallel data preparation - compute all triangle vertex data
    for (auto iter = face_inds.begin(); iter != face_inds.end(); ++iter) {
        std::string materialname = iter->first;

        std::string texture;
        RGBcolor color = default_color;
        bool textureColorIsOverridden = false;

        if (materials.find(materialname) != materials.end()) {
            const OBJmaterial &mat = materials.at(materialname);

            texture = mat.texture;
            color = mat.color;
            textureColorIsOverridden = mat.textureColorIsOverridden;
        }


        const auto &material_faces = face_inds.at(materialname);
        const auto &material_texture_inds = texture_inds.count(materialname) ? texture_inds.at(materialname) : std::vector<std::vector<int>>();

        // Exception handling for OpenMP - capture exceptions and rethrow after parallel region
        std::string exception_message;
        bool exception_occurred = false;

#ifdef USE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < static_cast<int>(material_faces.size()); i++) {
            try {
                for (uint t = 2; t < material_faces[i].size(); t++) {
                    vec3 v0 = vertices.at(material_faces[i][0] - 1);
                    vec3 v1 = vertices.at(material_faces[i][t - 1] - 1);
                    vec3 v2 = vertices.at(material_faces[i][t] - 1);

                    if ((v0 - v1).magnitude() == 0 || (v0 - v2).magnitude() == 0 || (v1 - v2).magnitude() == 0) {
                        continue;
                    }

                    if (strcmp(upaxis, "YUP") == 0) {
                        v0 = rotatePointAboutLine(v0, make_vec3(0, 0, 0), make_vec3(1, 0, 0), 0.5 * M_PI);
                        v1 = rotatePointAboutLine(v1, make_vec3(0, 0, 0), make_vec3(1, 0, 0), 0.5 * M_PI);
                        v2 = rotatePointAboutLine(v2, make_vec3(0, 0, 0), make_vec3(1, 0, 0), 0.5 * M_PI);
                    }

                    v0 = rotatePoint(v0, rotation);
                    v1 = rotatePoint(v1, rotation);
                    v2 = rotatePoint(v2, rotation);

                    // Calculate final triangle vertices after transformations
                    vec3 vert0 = origin + make_vec3(v0.x * scl.x, v0.y * scl.y, v0.z * scl.z);
                    vec3 vert1 = origin + make_vec3(v1.x * scl.x, v1.y * scl.y, v1.z * scl.z);
                    vec3 vert2 = origin + make_vec3(v2.x * scl.x, v2.y * scl.y, v2.z * scl.z);

                    // Check if triangle has sufficient area to avoid zero-area triangles
                    float triangle_area = calculateTriangleArea(vert0, vert1, vert2);

                    if (triangle_area > MIN_TRIANGLE_AREA_THRESHOLD) { // Only process triangle if area is not negligible
                        TriangleData triangleData;
                        triangleData.vert0 = vert0;
                        triangleData.vert1 = vert1;
                        triangleData.vert2 = vert2;
                        triangleData.texture = texture;
                        triangleData.color = color;
                        triangleData.textureColorIsOverridden = textureColorIsOverridden;
                        triangleData.object = objects.at(material_faces[i][0] - 1);

                        // Handle texture coordinates if present
                        // First check if material has texture file
                        triangleData.hasTexture = !texture.empty();


                        // If texture exists, try to get UV coordinates
                        if (triangleData.hasTexture && i < material_texture_inds.size() && !material_texture_inds[i].empty() && t < material_texture_inds[i].size()) {

                            int iuv0 = material_texture_inds[i][0] - 1;
                            int iuv1 = material_texture_inds[i][t - 1] - 1;
                            int iuv2 = material_texture_inds[i][t] - 1;

                            if (iuv0 >= 0 && iuv0 < texture_uv.size() && iuv1 >= 0 && iuv1 < texture_uv.size() && iuv2 >= 0 && iuv2 < texture_uv.size()) {
                                triangleData.uv0 = texture_uv.at(iuv0);
                                triangleData.uv1 = texture_uv.at(iuv1);
                                triangleData.uv2 = texture_uv.at(iuv2);
                            } else {
                                helios_runtime_error("ERROR (Context::loadOBJ): Invalid texture coordinate indices in face for material '" + materialname + "'. " + "UV indices [" + std::to_string(iuv0 + 1) + ", " + std::to_string(iuv1 + 1) + ", " +
                                                     std::to_string(iuv2 + 1) + "] " + "exceed available UV coordinates (1-" + std::to_string(texture_uv.size()) + "). " +
                                                     "Check that all face texture coordinate references in the OBJ file are valid.");
                            }
                        } else if (triangleData.hasTexture) {
                            helios_runtime_error("ERROR (Context::loadOBJ): Material '" + materialname + "' specifies texture file '" + texture + "' " + "but face has no texture coordinates. Either remove the texture from the material " +
                                                 "or add texture coordinates (vt) and face texture indices (f v1/vt1 v2/vt2 v3/vt3) to the OBJ file.");
                        }

#ifdef USE_OPENMP
#pragma omp critical
#endif
                        {
                            triangleDataList.push_back(triangleData);
                        }
                    }
                }
            } catch (const std::exception &e) {
                // Capture exception in OpenMP-safe way
#ifdef USE_OPENMP
#pragma omp critical
#endif
                {
                    if (!exception_occurred) {
                        exception_message = e.what();
                        exception_occurred = true;
                    }
                }
            }
        }

        // Rethrow captured exception after parallel region
        if (exception_occurred) {
            helios_runtime_error(exception_message);
        }
    }

    // Second pass: Sequential triangle creation to maintain thread safety
    for (const auto &triangleData: triangleDataList) {
        uint ID = 0;

        if (triangleData.hasTexture) {
            ID = addTriangle(triangleData.vert0, triangleData.vert1, triangleData.vert2, triangleData.texture.c_str(), triangleData.uv0, triangleData.uv1, triangleData.uv2);

            if (triangleData.textureColorIsOverridden) {
                setPrimitiveColor(ID, triangleData.color);
                overridePrimitiveTextureColor(ID);
            }
        } else {
            ID = addTriangle(triangleData.vert0, triangleData.vert1, triangleData.vert2, triangleData.color);
        }

        UUID.push_back(ID);

        if (triangleData.object != "none" && doesPrimitiveExist(ID)) {
            setPrimitiveData(ID, "object_label", triangleData.object);
        }
    }

    if (!silent) {
        std::cout << "done." << std::endl;
    }

    return UUID;
}

std::map<std::string, Context::OBJmaterial> Context::loadMTL(const std::string &filebase, const std::string &material_file, const RGBcolor &default_color) {
    std::ifstream inputMTL;

    std::string file = material_file;

    // For relative paths, resolve relative to the OBJ file's directory (filebase)
    // For absolute paths, use unified file resolution
    std::filesystem::path resolved_path;

    if (std::filesystem::path(file).is_absolute()) {
        // Absolute path - use unified resolution
        resolved_path = resolveFilePath(file);
    } else {
        // Relative path - resolve relative to OBJ file directory
        std::filesystem::path mtl_path = std::filesystem::path(filebase) / file;
        resolved_path = resolveFilePath(mtl_path.string());
    }

    std::string resolved_file = resolved_path.string();
    inputMTL.open(resolved_file.c_str());

    if (!inputMTL.is_open()) {
        helios_runtime_error("ERROR (Context::loadMTL): Could not open material file " + resolved_file + " after successful path resolution.");
    }

    std::map<std::string, OBJmaterial> materials;

    std::string line;

    inputMTL >> line;

    while (inputMTL.good()) {
        if (strcmp("#", line.c_str()) == 0) { // comments
            getline(inputMTL, line);
            inputMTL >> line;
        } else if (line == "newmtl") { // material library
            getline(inputMTL, line);
            std::string material_name = trim_whitespace(line);
            OBJmaterial mat(default_color, "", 0);
            materials.emplace(material_name, mat);

            std::string map_Kd, map_d;

            while (line != "newmtl" && inputMTL.good()) {
                inputMTL >> line;

                if (line == "newmtl") {
                    break;
                } else if (line == "map_a" || line == "map_Ka" || line == "Ks" || line == "Ka" || line == "map_Ks") {
                    getline(inputMTL, line);
                } else if (line == "map_Kd" || line == "map_d") {
                    std::string maptype = line;
                    getline(inputMTL, line);
                    line = trim_whitespace(line);
                    std::istringstream stream(line);
                    std::string tmp;
                    while (stream.good()) {
                        stream >> tmp;
                        std::string ext = getFileExtension(tmp);
                        if (ext == ".png" || ext == ".PNG" || ext == ".jpg" || ext == ".JPG" || ext == ".jpeg" || ext == ".JPEG") {
                            std::string texturefile = tmp;

                            // Check for texture file existence using filesystem operations (more efficient)
                            std::filesystem::path texture_path = texturefile;
                            bool texture_exists = false;

                            // First try the path as given in MTL file
                            if (std::filesystem::exists(texture_path)) {
                                texture_exists = true;
                            } else {
                                // Try looking in the same directory where OBJ file is located
                                texture_path = std::filesystem::path(filebase) / tmp;
                                texturefile = texture_path.string();
                                if (std::filesystem::exists(texture_path)) {
                                    texture_exists = true;
                                }
                            }

                            if (!texture_exists) {
                                helios_runtime_error("ERROR (Context::loadOBJ): Texture file '" + tmp + "' referenced in .mtl file cannot be found. " + "Searched in current directory and OBJ file directory (" + filebase + "). " +
                                                     "Ensure texture file exists or remove texture reference from material.");
                            }

                            if (maptype == "map_d") {
                                map_d = texturefile;
                            } else {
                                map_Kd = texturefile;
                            }
                        }
                    }
                } else if (line == "Kd") {
                    getline(inputMTL, line);
                    std::string color_str = trim_whitespace(line);
                    RGBAcolor color = string2RGBcolor(color_str.c_str());
                    materials.at(material_name).color = make_RGBcolor(color.r, color.g, color.b);
                } else {
                    getline(inputMTL, line);
                }
            }

            if (!map_Kd.empty()) {
                materials.at(material_name).texture = map_Kd;
                if (!map_d.empty() && map_d != map_Kd) {
                    materials.at(material_name).textureHasTransparency = true;
                }
            } else if (!map_d.empty()) {
                materials.at(material_name).texture = map_d;
                materials.at(material_name).textureColorIsOverridden = true;
            }
        } else {
            getline(inputMTL, line);
            inputMTL >> line;
        }
    }

    return materials;
}

void Context::writeOBJ(const std::string &filename, bool write_normals, bool silent) const {
    writeOBJ(filename, getAllUUIDs(), {}, write_normals, silent);
}

void Context::writeOBJ(const std::string &filename, const std::vector<uint> &UUIDs, bool write_normals, bool silent) const {
    writeOBJ(filename, UUIDs, {}, write_normals, silent);
}

void Context::writeOBJ(const std::string &filename, const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_dat_fields, bool write_normals, bool silent) const {

    if (UUIDs.empty()) {
        std::cout << "WARNING (Context::writeOBJ): No primitives found to write - OBJ file " << filename << " will not be written." << std::endl;
        return;
    }
    if (filename.empty()) {
        std::cout << "WARNING (Context::writeOBJ): Filename was empty - OBJ file " << filename << " will not be written." << std::endl;
        return;
    }

    std::string objfilename = filename;
    std::string mtlfilename = filename;

    auto file_extension = getFileExtension(filename);
    auto file_stem = getFileStem(filename);
    auto file_path = getFilePath(filename);

    if (file_extension != ".obj" && file_extension != ".OBJ") { // append obj to file name
        objfilename.append(".obj");
        mtlfilename.append(".mtl");
    } else {
        if (!file_path.empty()) {
            std::filesystem::path mtl_path = std::filesystem::path(file_path) / (file_stem + ".mtl");
            mtlfilename = mtl_path.string();
        } else {
            mtlfilename = file_stem + ".mtl";
        }
    }

    if (!file_path.empty() && !std::filesystem::exists(file_path)) {
        if (!std::filesystem::create_directory(file_path)) {
            std::cerr << "failed. Directory " << file_path << " does not exist and it could not be created - OBJ file will not be written." << std::endl;
            return;
        }
    }

    if (!silent) {
        std::cout << "Writing OBJ file " << objfilename << "..." << std::flush;
    }

    std::vector<OBJmaterial> materials;
    std::unordered_map<std::string, uint> material_cache;
    const size_t primitive_count = UUIDs.size();
    const size_t estimated_vertices = primitive_count * 4;

    std::vector<vec3> verts;
    verts.reserve(estimated_vertices);
    std::vector<vec3> normals;
    if (write_normals) {
        normals.reserve(primitive_count);
    }
    std::vector<vec2> uv;
    uv.reserve(estimated_vertices);

    std::map<uint, std::vector<int3>> faces;
    std::map<uint, std::vector<int>> normal_inds;
    std::map<uint, std::vector<int3>> uv_inds;
    size_t vertex_count = 1;
    size_t normal_count = 0;
    size_t uv_count = 1;
    std::map<uint, std::vector<uint>> UUIDs_write;

    std::map<std::string, std::map<uint, std::vector<int3>>> object_faces;
    std::map<std::string, std::map<uint, std::vector<int>>> object_normal_inds;
    std::map<std::string, std::map<uint, std::vector<int3>>> object_uv_inds;
    std::vector<std::string> object_order;
    object_order.reserve(primitive_count / 10);
    bool object_groups_found = false;

    for (size_t p: UUIDs) {
        if (!doesPrimitiveExist(p)) {
            std::ostringstream err_stream;
            err_stream << "ERROR (Context::writeOBJ): Primitive with UUID " << p << " does not exist. "
                       << "Ensure all UUIDs in the input vector correspond to valid primitives before calling writeOBJ.";
            helios_runtime_error(err_stream.str());
        }

        const Primitive *prim_ptr = getPrimitivePointer_private(p);

        if (prim_ptr->getType() == PRIMITIVE_TYPE_VOXEL) {
            std::ostringstream err_stream;
            err_stream << "ERROR (Context::writeOBJ): Voxel primitives (UUID " << p << ") cannot be written to OBJ format. "
                       << "OBJ format only supports surface primitives (triangles, patches). "
                       << "Filter out voxel primitives before calling writeOBJ.";
            helios_runtime_error(err_stream.str());
        }

        std::vector<vec3> vertices = prim_ptr->getVertices();
        PrimitiveType type = prim_ptr->getType();
        RGBcolor C = prim_ptr->getColor();
        std::string texturefile = prim_ptr->getTextureFile();
        bool texture_color_overridden = prim_ptr->isTextureColorOverridden();

        std::string obj_label = "default";
        if (doesPrimitiveDataExist(p, "object_label")) {
            getPrimitiveData(p, "object_label", obj_label);
            object_groups_found = true;
        }
        if (object_faces.find(obj_label) == object_faces.end()) {
            object_faces[obj_label] = {};
            object_normal_inds[obj_label] = {};
            object_uv_inds[obj_label] = {};
            object_order.push_back(obj_label);
        }

        std::string material_key = texturefile + "|" + std::to_string(C.r) + "," + std::to_string(C.g) + "," + std::to_string(C.b) + "|" + std::to_string(texture_color_overridden);

        uint material_ID;
        auto material_iter = material_cache.find(material_key);

        if (material_iter != material_cache.end()) {
            // Material exists in cache
            material_ID = material_iter->second;
        } else {
            // Create new material
            OBJmaterial mat(C, texturefile, materials.size());
            materials.emplace_back(mat);
            material_ID = mat.materialID;

            if (primitiveTextureHasTransparencyChannel(p)) {
                materials.back().textureHasTransparency = true;
            }
            if (texture_color_overridden) {
                materials.back().textureColorIsOverridden = true;
            }

            material_cache[material_key] = material_ID;
        }

        if (!primitive_dat_fields.empty()) {
            UUIDs_write[material_ID].push_back(p);
        }

        if (write_normals) {
            vec3 normal = getPrimitiveNormal(p);
            normals.push_back(normal);
            normal_count++;
        }

        if (type == PRIMITIVE_TYPE_TRIANGLE) {
            int3 ftmp = make_int3((int) vertex_count, (int) vertex_count + 1, (int) vertex_count + 2);
            faces[material_ID].push_back(ftmp);
            object_faces[obj_label][material_ID].push_back(ftmp);
            for (int i = 0; i < 3; i++) {
                verts.push_back(vertices.at(i));
                vertex_count++;
            }

            if (write_normals) {
                normal_inds[material_ID].push_back(static_cast<int>(normal_count));
                object_normal_inds[obj_label][material_ID].push_back(static_cast<int>(normal_count));
            }

            std::vector<vec2> uv_v = getTrianglePointer_private(p)->getTextureUV();
            if (getTrianglePointer_private(p)->hasTexture()) {
                int3 tuv = make_int3((int) uv_count, (int) uv_count + 1, (int) uv_count + 2);
                uv_inds[material_ID].push_back(tuv);
                object_uv_inds[obj_label][material_ID].push_back(tuv);
                for (int i = 0; i < 3; i++) {
                    uv.push_back(uv_v.at(i));
                    uv_count++;
                }
            } else {
                int3 tuv = make_int3(-1, -1, -1);
                uv_inds[material_ID].push_back(tuv);
                object_uv_inds[obj_label][material_ID].push_back(tuv);
            }
        } else if (type == PRIMITIVE_TYPE_PATCH) {
            int3 ftmp1 = make_int3((int) vertex_count, (int) vertex_count + 1, (int) vertex_count + 2);
            int3 ftmp2 = make_int3((int) vertex_count, (int) vertex_count + 2, (int) vertex_count + 3);
            faces[material_ID].push_back(ftmp1);
            faces[material_ID].push_back(ftmp2);
            object_faces[obj_label][material_ID].push_back(ftmp1);
            object_faces[obj_label][material_ID].push_back(ftmp2);
            for (int i = 0; i < 4; i++) {
                verts.push_back(vertices.at(i));
                vertex_count++;
            }
            std::vector<vec2> uv_v;
            uv_v = getPatchPointer_private(p)->getTextureUV();

            if (write_normals) {
                normal_inds[material_ID].push_back(static_cast<int>(normal_count));
                normal_inds[material_ID].push_back(static_cast<int>(normal_count));
                object_normal_inds[obj_label][material_ID].push_back(static_cast<int>(normal_count));
                object_normal_inds[obj_label][material_ID].push_back(static_cast<int>(normal_count));
            }

            if (getPatchPointer_private(p)->hasTexture()) {
                int3 tuv1 = make_int3((int) uv_count, (int) uv_count + 1, (int) uv_count + 2);
                int3 tuv2 = make_int3((int) uv_count, (int) uv_count + 2, (int) uv_count + 3);
                uv_inds[material_ID].push_back(tuv1);
                uv_inds[material_ID].push_back(tuv2);
                object_uv_inds[obj_label][material_ID].push_back(tuv1);
                object_uv_inds[obj_label][material_ID].push_back(tuv2);
                if (uv_v.empty()) { // default (u,v)
                    uv.push_back(make_vec2(0, 1));
                    uv.push_back(make_vec2(1, 1));
                    uv.push_back(make_vec2(1, 0));
                    uv.push_back(make_vec2(0, 0));
                    uv_count += 4;
                } else { // custom (u,v)
                    for (int i = 0; i < 4; i++) {
                        uv.push_back(uv_v.at(i));
                        uv_count++;
                    }
                }
            } else {
                int3 tuv = make_int3(-1, -1, -1);
                uv_inds[material_ID].push_back(tuv);
                uv_inds[material_ID].push_back(tuv);
                object_uv_inds[obj_label][material_ID].push_back(tuv);
                object_uv_inds[obj_label][material_ID].push_back(tuv);
            }
        }
    }

    if (write_normals)
        assert(normal_inds.size() == faces.size());
    //  assert(verts.size() == faces.size());
    assert(uv_inds.size() == faces.size());
    for (int i = 0; i < faces.size(); i++) {
        assert(uv_inds.at(i).size() == faces.at(i).size());
    }

    // copy material textures to new directory and edit old file paths
    std::filesystem::path output_path = std::filesystem::path(file_path);
    std::filesystem::path texture_dir = output_path.parent_path();

    // If no parent path (filename only), use current directory
    if (texture_dir.empty()) {
        texture_dir = ".";
    }

    for (auto &material: materials) {
        std::string texture = material.texture;
        if (!texture.empty()) {
            std::error_code ec;
            std::filesystem::path source_path = std::filesystem::absolute(texture, ec);

            // If we can't resolve the absolute path, try the original path
            if (ec) {
                source_path = std::filesystem::path(texture);
            }

            if (!std::filesystem::exists(source_path)) {
                // Skip missing texture files silently (maintain original behavior)
                continue;
            }

            auto filename = source_path.filename();
            std::filesystem::path dest_path = texture_dir / filename;

            // Skip copying if source and destination are the same file
            bool same_file = false;
            try {
                same_file = std::filesystem::equivalent(source_path, dest_path, ec);
                if (ec)
                    same_file = false; // If we can't determine equivalence, assume different
            } catch (...) {
                same_file = false;
            }

            if (same_file) {
                material.texture = filename.string();
                continue;
            }

            // Attempt to copy file, but don't fail if it doesn't work
            try {
                std::filesystem::copy_file(source_path, dest_path, std::filesystem::copy_options::overwrite_existing, ec);
                if (!ec) {
                    material.texture = filename.string();
                } // else keep original texture path
            } catch (...) {
                // If copy fails for any reason, keep original texture path
                // This maintains backward compatibility
            }
        }
    }

    std::ofstream objfstream;
    objfstream.open(objfilename);
    std::ofstream mtlfstream;
    mtlfstream.open(mtlfilename);

    objfstream << "# Helios auto-generated OBJ File" << std::endl;
    objfstream << "# baileylab.ucdavis.edu/software/helios" << std::endl;
    objfstream << "mtllib " << getFileName(mtlfilename) << std::endl;

    // Parallel string formatting for vertices, normals, and UV coordinates
    std::vector<std::string> vertex_chunks;
    const int num_threads = std::min(static_cast<int>(verts.size() / 1000 + 1), std::max(1, static_cast<int>(std::thread::hardware_concurrency())));
    vertex_chunks.resize(num_threads);

#ifdef USE_OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
    {
        int tid = 0;
#ifdef USE_OPENMP
        tid = omp_get_thread_num();
#endif
        std::ostringstream vertex_stream;
        vertex_stream.precision(8);

        const size_t chunk_size = (verts.size() + num_threads - 1) / num_threads;
        const size_t start_idx = tid * chunk_size;
        const size_t end_idx = std::min(start_idx + chunk_size, verts.size());

        for (size_t i = start_idx; i < end_idx; i++) {
            vertex_stream << "v " << verts[i].x << " " << verts[i].y << " " << verts[i].z << "\n";
        }

        vertex_chunks[tid] = vertex_stream.str();
    }

    for (const auto &chunk: vertex_chunks) {
        objfstream << chunk;
    }

    if (write_normals) {
        std::vector<std::string> normal_chunks;
        normal_chunks.resize(num_threads);

#ifdef USE_OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
        {
            int tid = 0;
#ifdef USE_OPENMP
            tid = omp_get_thread_num();
#endif
            std::ostringstream normal_stream;
            normal_stream.precision(8);

            const size_t chunk_size = (normals.size() + num_threads - 1) / num_threads;
            const size_t start_idx = tid * chunk_size;
            const size_t end_idx = std::min(start_idx + chunk_size, normals.size());

            const float epsilon = 1e-7;
            for (size_t i = start_idx; i < end_idx; i++) {
                vec3 n = normals[i];
                if (std::abs(n.x) < epsilon)
                    n.x = 0;
                if (std::abs(n.y) < epsilon)
                    n.y = 0;
                if (std::abs(n.z) < epsilon)
                    n.z = 0;
                normal_stream << "vn " << n.x << " " << n.y << " " << n.z << "\n";
            }

            normal_chunks[tid] = normal_stream.str();
        }

        for (const auto &chunk: normal_chunks) {
            objfstream << chunk;
        }
    }

    if (!uv.empty()) {
        std::vector<std::string> uv_chunks;
        uv_chunks.resize(num_threads);

#ifdef USE_OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
        {
            int tid = 0;
#ifdef USE_OPENMP
            tid = omp_get_thread_num();
#endif
            std::ostringstream uv_stream;
            uv_stream.precision(8);

            const size_t chunk_size = (uv.size() + num_threads - 1) / num_threads;
            const size_t start_idx = tid * chunk_size;
            const size_t end_idx = std::min(start_idx + chunk_size, uv.size());

            for (size_t i = start_idx; i < end_idx; i++) {
                uv_stream << "vt " << uv[i].x << " " << uv[i].y << "\n";
            }

            uv_chunks[tid] = uv_stream.str();
        }

        for (const auto &chunk: uv_chunks) {
            objfstream << chunk;
        }
    }

    // Parallel face string generation

    if (object_groups_found) {
        // Process object groups sequentially (maintain OBJ structure)
        // but parallelize face generation within each material group
        for (const auto &obj_label: object_order) {
            objfstream << "o " << obj_label << "\n";

            for (int mat = 0; mat < materials.size(); mat++) {
                auto fit = object_faces[obj_label].find(mat);
                if (fit == object_faces[obj_label].end())
                    continue;

                objfstream << "usemtl material" << mat << "\n";

                const auto &current_faces = fit->second;
                if (current_faces.size() > 100) { // Only parallelize if enough faces
                    // Parallel face string generation for this material
                    std::vector<std::string> face_chunks;
                    face_chunks.resize(num_threads);

#ifdef USE_OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
                    {
                        int tid = 0;
#ifdef USE_OPENMP
                        tid = omp_get_thread_num();
#endif
                        std::ostringstream face_stream;

                        const size_t chunk_size = (current_faces.size() + num_threads - 1) / num_threads;
                        const size_t start_idx = tid * chunk_size;
                        const size_t end_idx = std::min(start_idx + chunk_size, current_faces.size());

                        for (size_t f = start_idx; f < end_idx; f++) {
                            if (uv.empty()) {
                                if (write_normals) {
                                    face_stream << "f " << current_faces[f].x << "//" << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].y << "//" << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].z << "//"
                                                << object_normal_inds[obj_label][mat][f] << "\n";
                                } else {
                                    face_stream << "f " << current_faces[f].x << " " << current_faces[f].y << " " << current_faces[f].z << "\n";
                                }
                            } else if (object_uv_inds[obj_label][mat][f].x < 0) {
                                face_stream << "f " << current_faces[f].x << "/1 " << current_faces[f].y << "/1 " << current_faces[f].z << "/1\n";
                            } else {
                                if (write_normals) {
                                    face_stream << "f " << current_faces[f].x << "/" << object_uv_inds[obj_label][mat][f].x << "/" << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].y << "/" << object_uv_inds[obj_label][mat][f].y
                                                << "/" << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].z << "/" << object_uv_inds[obj_label][mat][f].z << "/" << object_normal_inds[obj_label][mat][f] << "\n";
                                } else {
                                    face_stream << "f " << current_faces[f].x << "/" << object_uv_inds[obj_label][mat][f].x << " " << current_faces[f].y << "/" << object_uv_inds[obj_label][mat][f].y << " " << current_faces[f].z << "/"
                                                << object_uv_inds[obj_label][mat][f].z << "\n";
                                }
                            }
                        }

                        face_chunks[tid] = face_stream.str();
                    }

                    // Sequential write of face chunks
                    for (const auto &chunk: face_chunks) {
                        objfstream << chunk;
                    }
                } else {
                    // For small face counts, use original sequential approach
                    for (size_t f = 0; f < current_faces.size(); ++f) {
                        if (uv.empty()) {
                            if (write_normals) {
                                objfstream << "f " << current_faces[f].x << "//" << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].y << "//" << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].z << "//"
                                           << object_normal_inds[obj_label][mat][f] << std::endl;
                            } else {
                                objfstream << "f " << current_faces[f].x << " " << current_faces[f].y << " " << current_faces[f].z << std::endl;
                            }
                        } else if (object_uv_inds[obj_label][mat][f].x < 0) {
                            objfstream << "f " << current_faces[f].x << "/1 " << current_faces[f].y << "/1 " << current_faces[f].z << "/1" << std::endl;
                        } else {
                            if (write_normals) {
                                objfstream << "f " << current_faces[f].x << "/" << object_uv_inds[obj_label][mat][f].x << "/" << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].y << "/" << object_uv_inds[obj_label][mat][f].y << "/"
                                           << object_normal_inds[obj_label][mat][f] << " " << current_faces[f].z << "/" << object_uv_inds[obj_label][mat][f].z << "/" << object_normal_inds[obj_label][mat][f] << std::endl;
                            } else {
                                objfstream << "f " << current_faces[f].x << "/" << object_uv_inds[obj_label][mat][f].x << " " << current_faces[f].y << "/" << object_uv_inds[obj_label][mat][f].y << " " << current_faces[f].z << "/"
                                           << object_uv_inds[obj_label][mat][f].z << std::endl;
                            }
                        }
                    }
                }
            }
        }
    } else {
        // No object groups - simpler structure, better parallelization opportunity
        for (int mat = 0; mat < materials.size(); mat++) {
            assert(materials.at(mat).materialID == mat);
            objfstream << "usemtl material" << mat << "\n";

            const auto &current_faces = faces.at(mat);
            if (current_faces.size() > 100) { // Only parallelize if enough faces
                // Parallel face string generation for this material
                std::vector<std::string> face_chunks;
                face_chunks.resize(num_threads);

#ifdef USE_OPENMP
#pragma omp parallel num_threads(num_threads)
#endif
                {
                    int tid = 0;
#ifdef USE_OPENMP
                    tid = omp_get_thread_num();
#endif
                    std::ostringstream face_stream;

                    const size_t chunk_size = (current_faces.size() + num_threads - 1) / num_threads;
                    const size_t start_idx = tid * chunk_size;
                    const size_t end_idx = std::min(start_idx + chunk_size, current_faces.size());

                    for (size_t f = start_idx; f < end_idx; f++) {
                        if (uv.empty()) {
                            if (write_normals) {
                                face_stream << "f " << current_faces[f].x << "//" << normal_inds.at(mat)[f] << " " << current_faces[f].y << "//" << normal_inds.at(mat)[f] << " " << current_faces[f].z << "//" << normal_inds.at(mat)[f] << "\n";
                            } else {
                                face_stream << "f " << current_faces[f].x << " " << current_faces[f].y << " " << current_faces[f].z << "\n";
                            }
                        } else if (uv_inds.at(mat)[f].x < 0) {
                            face_stream << "f " << current_faces[f].x << "/1 " << current_faces[f].y << "/1 " << current_faces[f].z << "/1\n";
                        } else {
                            if (write_normals) {
                                face_stream << "f " << current_faces[f].x << "/" << uv_inds.at(mat)[f].x << "/" << normal_inds.at(mat)[f] << " " << current_faces[f].y << "/" << uv_inds.at(mat)[f].y << "/" << normal_inds.at(mat)[f] << " "
                                            << current_faces[f].z << "/" << uv_inds.at(mat)[f].z << "/" << normal_inds.at(mat)[f] << "\n";
                            } else {
                                face_stream << "f " << current_faces[f].x << "/" << uv_inds.at(mat)[f].x << " " << current_faces[f].y << "/" << uv_inds.at(mat)[f].y << " " << current_faces[f].z << "/" << uv_inds.at(mat)[f].z << "\n";
                            }
                        }
                    }

                    face_chunks[tid] = face_stream.str();
                }

                // Sequential write of face chunks
                for (const auto &chunk: face_chunks) {
                    objfstream << chunk;
                }
            } else {
                // For small face counts, use original sequential approach
                for (int f = 0; f < current_faces.size(); f++) {
                    if (uv.empty()) {
                        if (write_normals) {
                            objfstream << "f " << current_faces[f].x << "//" << normal_inds.at(mat)[f] << " " << current_faces[f].y << "//" << normal_inds.at(mat)[f] << " " << current_faces[f].z << "//" << normal_inds.at(mat)[f] << std::endl;
                        } else {
                            objfstream << "f " << current_faces[f].x << " " << current_faces[f].y << " " << current_faces[f].z << std::endl;
                        }
                    } else if (uv_inds.at(mat)[f].x < 0) {
                        objfstream << "f " << current_faces[f].x << "/1 " << current_faces[f].y << "/1 " << current_faces[f].z << "/1" << std::endl;
                    } else {
                        if (write_normals) {
                            objfstream << "f " << current_faces[f].x << "/" << uv_inds.at(mat)[f].x << "/" << normal_inds.at(mat)[f] << " " << current_faces[f].y << "/" << uv_inds.at(mat)[f].y << "/" << normal_inds.at(mat)[f] << " "
                                       << current_faces[f].z << "/" << uv_inds.at(mat)[f].z << "/" << normal_inds.at(mat)[f] << std::endl;
                        } else {
                            objfstream << "f " << current_faces[f].x << "/" << uv_inds.at(mat)[f].x << " " << current_faces[f].y << "/" << uv_inds.at(mat)[f].y << " " << current_faces[f].z << "/" << uv_inds.at(mat)[f].z << std::endl;
                        }
                    }
                }
            }
        }
    }

    for (int mat = 0; mat < materials.size(); mat++) {
        if (materials.at(mat).texture.empty()) {
            RGBcolor current_color = materials.at(mat).color;
            mtlfstream << "newmtl material" << mat << std::endl;
            mtlfstream << "Ka " << current_color.r << " " << current_color.g << " " << current_color.b << std::endl;
            mtlfstream << "Kd " << current_color.r << " " << current_color.g << " " << current_color.b << std::endl;
            mtlfstream << "Ks 0.0 0.0 0.0" << std::endl;
            mtlfstream << "illum 2 " << std::endl;
        } else {
            std::string current_texture = materials.at(mat).texture;
            mtlfstream << "newmtl material" << mat << std::endl;
            if (materials.at(mat).textureColorIsOverridden) {
                RGBcolor current_color = materials.at(mat).color;
                mtlfstream << "Ka " << current_color.r << " " << current_color.g << " " << current_color.b << std::endl;
                mtlfstream << "Kd " << current_color.r << " " << current_color.g << " " << current_color.b << std::endl;
            } else {
                mtlfstream << "map_Kd " << current_texture << std::endl;
            }
            if (materials.at(mat).textureHasTransparency) {
                mtlfstream << "map_d " << current_texture << std::endl;
            }
            mtlfstream << "Ks 0.0 0.0 0.0" << std::endl;
            mtlfstream << "illum 2 " << std::endl;
        }
    }

    objfstream.close();
    mtlfstream.close();

    if (!primitive_dat_fields.empty()) {
        bool uuidexistswarning = false;
        bool dataexistswarning = false;
        bool datatypewarning = false;

        for (const std::string &label: primitive_dat_fields) {
            std::filesystem::path dat_path = std::filesystem::path(file_path) / (file_stem + "_" + std::string(label) + ".dat");
            std::string datfilename = dat_path.string();
            std::ofstream datout(datfilename);

            for (int mat = 0; mat < materials.size(); mat++) {
                for (uint UUID: UUIDs_write.at(mat)) {
                    if (!doesPrimitiveExist(UUID)) {
                        uuidexistswarning = true;
                        continue;
                    }

                    // a patch is converted to 2 triangles, so need to write 2 data values for patches
                    int Nprims = 1;
                    if (getPrimitiveType(UUID) == PRIMITIVE_TYPE_PATCH) {
                        Nprims = 2;
                    }

                    if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                        dataexistswarning = true;
                        for (int i = 0; i < Nprims; i++) {
                            datout << 0 << std::endl;
                        }
                        continue;
                    }

                    HeliosDataType type = getPrimitiveDataType(label.c_str());
                    if (type == HELIOS_TYPE_INT) {
                        int data;
                        getPrimitiveData(UUID, label.c_str(), data);
                        for (int i = 0; i < Nprims; i++) {
                            datout << data << std::endl;
                        }
                    } else if (type == HELIOS_TYPE_UINT) {
                        uint data;
                        getPrimitiveData(UUID, label.c_str(), data);
                        for (int i = 0; i < Nprims; i++) {
                            datout << data << std::endl;
                        }
                    } else if (type == HELIOS_TYPE_FLOAT) {
                        float data;
                        getPrimitiveData(UUID, label.c_str(), data);
                        for (int i = 0; i < Nprims; i++) {
                            datout << data << std::endl;
                        }
                    } else if (type == HELIOS_TYPE_DOUBLE) {
                        double data;
                        getPrimitiveData(UUID, label.c_str(), data);
                        for (int i = 0; i < Nprims; i++) {
                            datout << data << std::endl;
                        }
                    } else if (type == HELIOS_TYPE_STRING) {
                        std::string data;
                        getPrimitiveData(UUID, label.c_str(), data);
                        for (int i = 0; i < Nprims; i++) {
                            datout << data << std::endl;
                        }
                    } else {
                        datatypewarning = true;
                        for (int i = 0; i < Nprims; i++) {
                            datout << 0 << std::endl;
                        }
                    }
                }
            }

            datout.close();
        }

        if (uuidexistswarning) {
            helios_runtime_error("Context::writeOBJ: One or more UUIDs do not exist in the Context. Cannot write OBJ file with invalid primitives.");
        }
        if (dataexistswarning) {
            helios_runtime_error("Context::writeOBJ: Primitive data requested did not exist for one or more primitives. Cannot write incomplete data to OBJ file.");
        }
        if (datatypewarning) {
            helios_runtime_error("Context::writeOBJ: Only scalar primitive data types (uint, int, float, double, and string) are supported for primitive data export.");
        }
    }
}

void Context::writePrimitiveData(const std::string &filename, const std::vector<std::string> &column_format, bool print_header) const {
    writePrimitiveData(filename, column_format, getAllUUIDs(), print_header);
}

void Context::writePrimitiveData(const std::string &filename, const std::vector<std::string> &column_format, const std::vector<uint> &UUIDs, bool print_header) const {
    std::ofstream file(filename);

    if (print_header) {
        for (const auto &label: column_format) {
            file << label << " ";
        }
        file.seekp(-1, std::ios_base::end);
        file << "\n";
    }

    bool uuidexistswarning = false;
    bool dataexistswarning = false;
    bool datatypewarning = false;

    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            uuidexistswarning = true;
            continue;
        }
        for (const auto &label: column_format) {
            if (label == "UUID") {
                file << UUID << " ";
                continue;
            }
            if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                dataexistswarning = true;
                file << 0 << " ";
                continue;
            }
            HeliosDataType type = getPrimitiveDataType(label.c_str());
            if (type == HELIOS_TYPE_INT) {
                int data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            } else if (type == HELIOS_TYPE_UINT) {
                uint data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            } else if (type == HELIOS_TYPE_FLOAT) {
                float data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            } else if (type == HELIOS_TYPE_DOUBLE) {
                double data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            } else if (type == HELIOS_TYPE_STRING) {
                std::string data;
                getPrimitiveData(UUID, label.c_str(), data);
                file << data << " ";
            } else {
                datatypewarning = true;
                file << 0 << " ";
            }
        }
        file.seekp(-1, std::ios_base::end);
        file << "\n";
    }

    if (uuidexistswarning) {
        std::cerr << "WARNING (Context::writePrimitiveData): Vector of UUIDs passed to writePrimitiveData() function contained UUIDs that do not exist, which were skipped." << std::endl;
    }
    if (dataexistswarning) {
        std::cerr << "WARNING (Context::writePrimitiveData): Primitive data requested did not exist for one or more primitives. A default value of 0 was written in these cases." << std::endl;
    }
    if (datatypewarning) {
        std::cerr << "WARNING (Context::writePrimitiveData): Only scalar primitive data types (uint, int, float, and double) are supported for this function. A column of 0's was written in these cases." << std::endl;
    }

    file.close();
}

namespace {

    // Parse a date string with '-' or '/' delimiters, or compact 8-digit YYYYMMDD format.
    Date parseDateString(const std::string &datestr, const std::string &date_string_format, size_t row, const std::string &data_file) {

        // Check for compact 8-digit format (no delimiters)
        if (datestr.find('-') == std::string::npos && datestr.find('/') == std::string::npos) {
            if (datestr.size() == 8) {
                // Compact 8-digit date: parse according to format
                int year, month, day;
                if (date_string_format == "YYYYMMDD" || date_string_format == "YYYY-MM-DD") {
                    year = std::stoi(datestr.substr(0, 4));
                    month = std::stoi(datestr.substr(4, 2));
                    day = std::stoi(datestr.substr(6, 2));
                } else if (date_string_format == "DDMMYYYY" || date_string_format == "DD-MM-YYYY" || date_string_format == "DD/MM/YYYY") {
                    day = std::stoi(datestr.substr(0, 2));
                    month = std::stoi(datestr.substr(2, 2));
                    year = std::stoi(datestr.substr(4, 4));
                } else if (date_string_format == "MMDDYYYY" || date_string_format == "MM-DD-YYYY" || date_string_format == "MM/DD/YYYY") {
                    month = std::stoi(datestr.substr(0, 2));
                    day = std::stoi(datestr.substr(2, 2));
                    year = std::stoi(datestr.substr(4, 4));
                } else if (date_string_format == "YYYYDDMM") {
                    year = std::stoi(datestr.substr(0, 4));
                    day = std::stoi(datestr.substr(4, 2));
                    month = std::stoi(datestr.substr(6, 2));
                } else {
                    helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Invalid date string format '" + date_string_format + "' for compact date on line " + std::to_string(row) + " of file " + data_file + ".");
                }
                if (year < 1000 || month < 1 || month > 12 || day < 1 || day > 31) {
                    helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse compact date string on line " + std::to_string(row) + " of file " + data_file + ".");
                }
                return make_Date(day, month, year);
            }
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse date string on line " + std::to_string(row) + " of file " + data_file +
                                 ". Expected a delimited date (e.g., YYYY-MM-DD) or an 8-digit compact date (e.g., 20260203).");
        }

        // Delimited date: try '-' then '/'
        std::vector<std::string> thisdatestr = separate_string_by_delimiter(datestr, "-");
        if (thisdatestr.size() != 3) {
            thisdatestr = separate_string_by_delimiter(datestr, "/");
        }
        if (thisdatestr.size() != 3) {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse date string on line " + std::to_string(row) + " of file " + data_file +
                                 ". It should be in the format YYYY-MM-DD, delimited by either '-' or '/'.");
        }

        std::vector<int> thisdate(3);
        for (int i = 0; i < 3; i++) {
            if (!parse_int(thisdatestr.at(i), thisdate.at(i))) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse date string on line " + std::to_string(row) + " of file " + data_file +
                                     ". It should be in the format YYYY-MM-DD, delimited by either '-' or '/'.");
            }
        }

        int year, month, day;
        if (date_string_format == "YYYYMMDD" || date_string_format == "YYYY-MM-DD") {
            year = thisdate.at(0);
            month = thisdate.at(1);
            day = thisdate.at(2);
        } else if (date_string_format == "YYYYDDMM") {
            year = thisdate.at(0);
            month = thisdate.at(2);
            day = thisdate.at(1);
        } else if (date_string_format == "DDMMYYYY" || date_string_format == "DD-MM-YYYY" || date_string_format == "DD/MM/YYYY") {
            year = thisdate.at(2);
            month = thisdate.at(1);
            day = thisdate.at(0);
        } else if (date_string_format == "MMDDYYYY" || date_string_format == "MM-DD-YYYY" || date_string_format == "MM/DD/YYYY") {
            year = thisdate.at(2);
            month = thisdate.at(0);
            day = thisdate.at(1);
        } else {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Invalid date string format in file " + data_file + ": " + date_string_format +
                                 ". Must be one of YYYYMMDD, YYYYDDMM, DDMMYYYY, MMDDYYYY (or with - or / delimiters, e.g. YYYY-MM-DD, DD/MM/YYYY).");
        }

        if (year < 1000 || month < 1 || month > 12 || day < 1 || day > 31) {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse date string on line " + std::to_string(row) + " of file " + data_file + ".");
        }

        return make_Date(day, month, year);
    }

    // Parse a time string: "HH", "HH:MM", or "HH:MM:SS"
    // Note: may return hour=24 (via direct struct assignment) for midnight rollover; caller must handle.
    Time parseTimeString(const std::string &timestr, size_t row, const std::string &data_file) {
        std::string trimmed = trim_whitespace(timestr);

        std::vector<std::string> parts = separate_string_by_delimiter(trimmed, ":");
        int hour = 0, minute = 0, second = 0;

        if (parts.size() == 1) {
            // Integer hour
            if (!parse_int(parts.at(0), hour)) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse time string '" + timestr + "' on line " + std::to_string(row) + " of file " + data_file + ".");
            }
            // Handle HHMM format (e.g., 1300)
            if (hour > 24) {
                int hr_min = hour;
                hour = hr_min / 100;
                minute = hr_min - hour * 100;
            }
        } else if (parts.size() == 2) {
            if (!parse_int(parts.at(0), hour) || !parse_int(parts.at(1), minute)) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse time string '" + timestr + "' on line " + std::to_string(row) + " of file " + data_file + ".");
            }
        } else if (parts.size() == 3) {
            if (!parse_int(parts.at(0), hour) || !parse_int(parts.at(1), minute)) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse time string '" + timestr + "' on line " + std::to_string(row) + " of file " + data_file + ".");
            }
            // Handle fractional seconds by truncating at '.'
            std::string sec_str = parts.at(2);
            size_t dot_pos = sec_str.find('.');
            if (dot_pos != std::string::npos) {
                sec_str = sec_str.substr(0, dot_pos);
            }
            if (!parse_int(sec_str, second)) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse time string '" + timestr + "' on line " + std::to_string(row) + " of file " + data_file + ".");
            }
        } else {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse time string '" + timestr + "' on line " + std::to_string(row) + " of file " + data_file + ".");
        }

        // Handle hour=24 by directly setting struct fields (make_Time validates hour < 24)
        if (hour == 24) {
            Time t;
            t.hour = 24;
            t.minute = minute;
            t.second = second;
            return t;
        }

        return make_Time(hour, minute, second);
    }

    // Parse an ISO-8601 datetime string (e.g., "2026-02-03T10:00:00Z" or "2026-02-03T02:00:00-08:00")
    void parseISO8601(const std::string &datetimestr, Date &date, Time &time, float &utc_offset, size_t row, const std::string &data_file) {
        utc_offset = NAN;

        size_t t_pos = datetimestr.find('T');
        if (t_pos == std::string::npos) {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): ISO-8601 datetime string '" + datetimestr + "' on line " + std::to_string(row) + " of file " + data_file + " does not contain 'T' separator.");
        }

        // Parse date part (always YYYY-MM-DD)
        std::string date_part = datetimestr.substr(0, t_pos);
        std::vector<std::string> date_parts = separate_string_by_delimiter(date_part, "-");
        if (date_parts.size() != 3) {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse date portion of ISO-8601 string '" + datetimestr + "' on line " + std::to_string(row) + " of file " + data_file + ".");
        }
        int year, month, day;
        if (!parse_int(date_parts.at(0), year) || !parse_int(date_parts.at(1), month) || !parse_int(date_parts.at(2), day)) {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse date portion of ISO-8601 string '" + datetimestr + "' on line " + std::to_string(row) + " of file " + data_file + ".");
        }
        date = make_Date(day, month, year);

        // Parse time part + optional timezone
        std::string time_tz = datetimestr.substr(t_pos + 1);

        // Strip and parse timezone suffix
        std::string time_part;
        if (time_tz.back() == 'Z' || time_tz.back() == 'z') {
            time_part = time_tz.substr(0, time_tz.size() - 1);
            utc_offset = 0.0f; // UTC → Helios convention: +West, so UTC = 0
        } else {
            // Look for +/- timezone offset (e.g., +05:30, -08:00)
            // Search from after the hour portion to avoid matching a negative hour (shouldn't happen in ISO-8601 time)
            size_t tz_pos = std::string::npos;
            for (size_t i = 1; i < time_tz.size(); i++) {
                if (time_tz[i] == '+' || time_tz[i] == '-') {
                    tz_pos = i;
                    // Keep searching — we want the last +/- that's part of timezone, not inside time
                    // Actually for ISO-8601, the timezone offset is always at the end, so we want the last occurrence
                }
            }
            if (tz_pos != std::string::npos) {
                time_part = time_tz.substr(0, tz_pos);
                std::string tz_str = time_tz.substr(tz_pos); // e.g., "-08:00" or "+05:30"
                char tz_sign = tz_str[0];
                std::string tz_num = tz_str.substr(1);
                std::vector<std::string> tz_parts = separate_string_by_delimiter(tz_num, ":");
                int tz_hours = 0, tz_minutes = 0;
                if (!tz_parts.empty()) parse_int(tz_parts.at(0), tz_hours);
                if (tz_parts.size() > 1) parse_int(tz_parts.at(1), tz_minutes);
                float iso_offset_hours = static_cast<float>(tz_hours) + static_cast<float>(tz_minutes) / 60.0f;
                if (tz_sign == '-') iso_offset_hours = -iso_offset_hours;
                // Helios convention: UTC_offset is +West. ISO convention: +East.
                // So ISO -08:00 (Pacific) → Helios +8, ISO +05:30 (India) → Helios -5.5
                utc_offset = -iso_offset_hours;
            } else {
                time_part = time_tz; // No timezone info
            }
        }

        // Truncate fractional seconds
        size_t dot_pos = time_part.find('.');
        if (dot_pos != std::string::npos) {
            time_part = time_part.substr(0, dot_pos);
        }

        // Parse the time portion
        time = parseTimeString(time_part, row, data_file);
    }

    // Dispatch combined datetime string parsing based on format
    void parseDatetimeString(const std::string &datetimestr, const std::string &date_string_format,
                             Date &date, Time &time, float &utc_offset, size_t row, const std::string &data_file) {
        utc_offset = NAN;

        if (date_string_format == "ISO8601") {
            parseISO8601(datetimestr, date, time, utc_offset, row, data_file);
            return;
        }

        if (date_string_format == "YYYYMMDDHH") {
            if (datetimestr.size() < 10) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): YYYYMMDDHH datetime string '" + datetimestr + "' on line " + std::to_string(row) + " of file " + data_file + " is too short.");
            }
            int year = std::stoi(datetimestr.substr(0, 4));
            int month = std::stoi(datetimestr.substr(4, 2));
            int day = std::stoi(datetimestr.substr(6, 2));
            int hour = std::stoi(datetimestr.substr(8, 2));
            date = make_Date(day, month, year);
            time = make_Time(hour, 0, 0);
            return;
        }

        if (date_string_format == "YYYYMMDDHHMM") {
            if (datetimestr.size() < 12) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): YYYYMMDDHHMM datetime string '" + datetimestr + "' on line " + std::to_string(row) + " of file " + data_file + " is too short.");
            }
            int year = std::stoi(datetimestr.substr(0, 4));
            int month = std::stoi(datetimestr.substr(4, 2));
            int day = std::stoi(datetimestr.substr(6, 2));
            int hour = std::stoi(datetimestr.substr(8, 2));
            int minute = std::stoi(datetimestr.substr(10, 2));
            date = make_Date(day, month, year);
            time = make_Time(hour, minute, 0);
            return;
        }

        // Formats with space separator: "YYYY-MM-DD HH:MM", "DD/MM/YYYY HH:MM", etc.
        // The space has already been rejoined by the caller, so split at space
        size_t space_pos = datetimestr.find(' ');
        if (space_pos != std::string::npos) {
            std::string date_part = datetimestr.substr(0, space_pos);
            std::string time_part = datetimestr.substr(space_pos + 1);

            // Determine the date format portion (strip the time portion from format)
            std::string date_format;
            size_t fmt_space = date_string_format.find(' ');
            if (fmt_space != std::string::npos) {
                date_format = date_string_format.substr(0, fmt_space);
            } else {
                date_format = date_string_format;
            }

            // Normalize date format: "YYYY-MM-DD" → "YYYYMMDD", "DD/MM/YYYY" → "DDMMYYYY", etc.
            // parseDateString handles both delimited and synonym formats
            date = parseDateString(date_part, date_format, row, data_file);
            time = parseTimeString(time_part, row, data_file);
            return;
        }

        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse datetime string '" + datetimestr + "' with format '" + date_string_format + "' on line " + std::to_string(row) + " of file " + data_file + ".");
    }

    // Check if a datetime format string contains a space (i.e., date and time parts separated by space)
    bool datetimeFormatHasSpace(const std::string &format) {
        return format.find(' ') != std::string::npos;
    }

} // anonymous namespace

void Context::loadTabularTimeseriesData(const std::string &data_file, const std::vector<std::string> &col_labels, const std::string &a_delimeter, const std::string &a_date_string_format, uint headerlines) {
    // Resolve file path using project-based resolution
    std::filesystem::path resolved_path = resolveProjectFile(data_file);
    std::string resolved_filename = resolved_path.string();

    std::ifstream datafile(resolved_filename); // open the file

    if (!datafile.is_open()) { // check that file exists
        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Weather data file '" + data_file + "' does not exist.");
    }

    int yearcol = -1;
    int DOYcol = -1;
    int datestrcol = -1;
    int datetimecol = -1;
    int hourcol = -1;
    int minutecol = -1;
    int secondcol = -1;
    int timecol = -1;
    std::map<std::string, int> datacols;

    size_t Ncolumns = 0;

    size_t row = headerlines;

    std::vector<std::string> column_labels = col_labels;
    std::string delimiter = a_delimeter;
    std::string date_string_format = a_date_string_format;

    // pre-defined labels for CIMIS weather data files
    if (col_labels.size() == 1 && (col_labels.front() == "CIMIS" || col_labels.front() == "cimis")) {
        column_labels = {
                "", "", "", "date", "hour", "DOY", "ETo", "", "precipitation", "", "net_radiation", "", "vapor_pressure", "", "air_temperature", "", "air_humidity", "", "dew_point", "", "wind_speed", "", "wind_direction", "", "soil_temperature", ""};
        headerlines = 1;
        delimiter = ",";
        date_string_format = "MMDDYYYY";
    }

    // If user specified column labels as an argument, parse them
    if (!column_labels.empty()) {
        int col = 0;
        for (auto &label: column_labels) {
            if (label == "year" || label == "Year") {
                yearcol = col;
            } else if (label == "DOY" || label == "Jul") {
                DOYcol = col;
            } else if (label == "date" || label == "Date") {
                datestrcol = col;
            } else if (label == "datetime" || label == "Datetime" || label == "DateTime") {
                datetimecol = col;
            } else if (label == "hour" || label == "Hour") {
                hourcol = col;
            } else if (label == "minute" || label == "Minute") {
                minutecol = col;
            } else if (label == "second" || label == "Second") {
                secondcol = col;
            } else if (label == "time" || label == "Time") {
                timecol = col;
            } else if (!label.empty()) {
                if (datacols.find(label) == datacols.end()) {
                    datacols[label] = col;
                } else {
                    datacols[label + "_dup"] = col;
                }
            }

            col++;
        }

        Ncolumns = column_labels.size();

        // If column labels were not provided, read the first line of the text file and parse it for labels
    } else {
        if (headerlines == 0) {
            std::cerr << "WARNING (Context::loadTabularTimeseriesData): "
                         "headerlines"
                         " argument was specified as zero, and no column label information was given. Attempting to read the first line to see if it contains label information."
                      << std::endl;
            headerlines++;
        }

        std::string line;
        if (std::getline(datafile, line)) {
            std::vector<std::string> line_parsed = separate_string_by_delimiter(line, delimiter);

            if (line_parsed.empty()) {
                helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Attempted to parse first line of file for column labels, but it did not contain the specified delimiter.");
            }

            Ncolumns = line_parsed.size();

            for (int col = 0; col < Ncolumns; col++) {
                const std::string &label = line_parsed.at(col);

                if (label == "year" || label == "Year") {
                    yearcol = col;
                } else if (label == "DOY" || label == "Jul") {
                    DOYcol = col;
                } else if (label == "date" || label == "Date") {
                    datestrcol = col;
                } else if (label == "datetime" || label == "Datetime" || label == "DateTime") {
                    datetimecol = col;
                } else if (label == "hour" || label == "Hour") {
                    hourcol = col;
                } else if (label == "minute" || label == "Minute") {
                    minutecol = col;
                } else if (label == "second" || label == "Second") {
                    secondcol = col;
                } else if (label == "time" || label == "Time") {
                    timecol = col;
                } else if (!label.empty()) {
                    if (datacols.find(label) == datacols.end()) {
                        datacols[label] = col;
                    } else {
                        datacols[label + "_dup"] = col;
                    }
                }
            }

            headerlines--;
        } else {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Attempted to parse first line of file for column labels, but read failed.");
        }

        if (yearcol == -1 && DOYcol == -1 && datestrcol == -1 && datetimecol == -1) {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Attempted to parse first line of file for column labels, but could not find valid label information.");
        }
    }

    // Validate column combinations
    bool has_date = (datestrcol >= 0) || (yearcol >= 0 && DOYcol >= 0);
    bool has_time = (hourcol >= 0) || (timecol >= 0);
    bool has_datetime = (datetimecol >= 0);

    if (has_datetime && datestrcol >= 0) {
        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Cannot specify both 'datetime' and 'date' columns. Use 'datetime' for combined date+time, or 'date' + 'hour'/'time' for separate columns.");
    }
    if (has_datetime && hourcol >= 0) {
        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Cannot specify both 'datetime' and 'hour' columns. Use 'datetime' for combined date+time, or 'date' + 'hour' for separate columns.");
    }
    if (has_datetime && timecol >= 0) {
        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Cannot specify both 'datetime' and 'time' columns. The 'datetime' column already includes time information.");
    }
    if (!has_datetime && !has_date) {
        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): The date must be specified by a column labeled 'datetime', 'date', or by two columns labeled 'year' and 'DOY'.");
    }
    if (!has_datetime && !has_time) {
        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): The time must be specified by a column labeled 'datetime', 'hour', or 'time'.");
    }
    if (datacols.empty()) {
        helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): No columns were found containing data variables (e.g., temperature, humidity, wind speed).");
    }

    // Check if datetime format has a space — we may need to rejoin split columns
    bool datetime_format_has_space = has_datetime && datetimeFormatHasSpace(date_string_format);

    std::string line;

    // skip header lines
    // note: if we read labels from the first header line above, we don't need to skip another line
    for (int i = 0; i < headerlines; i++) {
        std::getline(datafile, line);
    }

    bool utc_offset_set = false;

    while (std::getline(datafile, line)) { // loop through file to read data
        row++;

        if (trim_whitespace(line).empty() && row > 1) {
            break;
        }

        // separate the line by delimiter
        std::vector<std::string> line_separated = separate_string_by_delimiter(line, delimiter);

        // Handle space-delimited datetime auto-rejoin: if the datetime format contains a space
        // (e.g., "YYYY-MM-DD HH:MM"), the space delimiter will split the datetime into two columns.
        // Rejoin them here.
        if (datetime_format_has_space && datetimecol >= 0 && line_separated.size() == Ncolumns + 1 && datetimecol + 1 < static_cast<int>(line_separated.size())) {
            line_separated[datetimecol] = line_separated[datetimecol] + " " + line_separated[datetimecol + 1];
            line_separated.erase(line_separated.begin() + datetimecol + 1);
        }

        if (line_separated.size() != Ncolumns) {
            helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Line " + std::to_string(row) + " had " + std::to_string(line_separated.size()) + " columns, but was expecting " + std::to_string(Ncolumns));
        }

        Date date;
        Time time;
        float parsed_utc_offset = NAN;

        if (datetimecol >= 0) {
            // Combined datetime column
            parseDatetimeString(line_separated.at(datetimecol), date_string_format,
                                date, time, parsed_utc_offset, row, data_file);
        } else {
            // Separate date + time columns
            if (yearcol >= 0 && DOYcol >= 0) {
                int DOY;
                parse_int(line_separated.at(DOYcol), DOY);
                if (DOY < 1 || DOY > 366) {
                    helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Invalid date specified on line " + std::to_string(row) + ".");
                }
                int year;
                parse_int(line_separated.at(yearcol), year);
                if (year < 1000) {
                    helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Invalid year specified on line " + std::to_string(row) + ".");
                }
                date = make_Date(DOY, year);
            } else if (datestrcol >= 0) {
                date = parseDateString(line_separated.at(datestrcol), date_string_format, row, data_file);
            }

            if (timecol >= 0) {
                time = parseTimeString(line_separated.at(timecol), row, data_file);
            } else if (hourcol >= 0) {
                int hour = 0;
                int minute = 0;
                int second = 0;

                if (!parse_int(line_separated.at(hourcol), hour)) {
                    helios_runtime_error("ERROR (Context::loadTabularTimeseriesData): Could not parse hour string on line " + std::to_string(row) + " of file " + data_file + ".");
                }
                if (hour > 24 && minutecol < 0 && secondcol < 0) {
                    int hr_min = hour;
                    hour = hr_min / 100;
                    minute = hr_min - hour * 100;
                }
                if (hour == 24) {
                    hour = 0;
                    date.incrementDay();
                }
                if (minutecol >= 0) {
                    if (!parse_int(line_separated.at(minutecol), minute)) {
                        minute = 0;
                        std::cout << "WARNING (Context::loadTabularTimeseriesData): Could not parse minute string on line " << row << " of file " << data_file << ". Setting minute equal to 0." << std::endl;
                    }
                }
                if (secondcol >= 0) {
                    if (!parse_int(line_separated.at(secondcol), second)) {
                        second = 0;
                        std::cout << "WARNING (Context::loadTabularTimeseriesData): Could not parse second string on line " << row << " of file " << data_file << ". Setting second equal to 0." << std::endl;
                    }
                }
                time = make_Time(hour, minute, second);
            }
        }

        // Handle hour=24 rollover
        if (time.hour == 24) {
            time = make_Time(0, time.minute, time.second);
            date.incrementDay();
        }

        // Set UTC offset from ISO-8601 if parsed
        if (!std::isnan(parsed_utc_offset)) {
            if (!utc_offset_set) {
                Location loc = getLocation();
                loc.UTC_offset = parsed_utc_offset;
                setLocation(loc);
                utc_offset_set = true;
            }
        }

        // compile data values
        for (auto &dat: datacols) {
            std::string label = dat.first;
            int col = dat.second;

            float dataval;
            if (!parse_float(line_separated.at(col), dataval)) {
                std::cout << "WARNING (Context::loadTabularTimeseriesData): Failed to parse data value as float on line "
                          << row << ", column " << col + 1 << " of file " << data_file << ". Skipping this value..." << std::endl;
                continue;
            }

            if (label == "air_humidity" && col_labels.size() == 1 && (col_labels.front() == "CIMIS" || col_labels.front() == "cimis")) {
                dataval = dataval / 100.f;
            }

            addTimeseriesData(label.c_str(), dataval, date, time);
        }
    }

    datafile.close();
}
