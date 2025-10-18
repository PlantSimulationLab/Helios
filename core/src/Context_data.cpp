/**
 * \file "Context_data.cpp" Context primitive data, object data, and global data declarations.
 *
 * Copyright (C) 2016-2025 Brian Bailey
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

void Context::incrementPrimitiveDataLabelCounter(const std::string &primitive_data_label) {
    primitive_data_label_counts[primitive_data_label]++;
}

void Context::decrementPrimitiveDataLabelCounter(const std::string &primitive_data_label) {
    auto it = primitive_data_label_counts.find(primitive_data_label);
    if (it != primitive_data_label_counts.end() && it->second > 0) {
        it->second--;
        if (it->second == 0) {
            primitive_data_label_counts.erase(it);
        }
    }
}

//----------- VALUE-LEVEL CACHING CONFIGURATION ----------//

void Context::enablePrimitiveDataValueCaching(const std::string &label) {
    cached_primitive_data_labels.insert(label);
}

void Context::disablePrimitiveDataValueCaching(const std::string &label) {
    cached_primitive_data_labels.erase(label);
    // Clear cached values for this label
    primitive_string_value_registry.erase(label);
    primitive_int_value_registry.erase(label);
    primitive_uint_value_registry.erase(label);
}

bool Context::isPrimitiveDataValueCachingEnabled(const std::string &label) const {
    return cached_primitive_data_labels.find(label) != cached_primitive_data_labels.end();
}

void Context::enableObjectDataValueCaching(const std::string &label) {
    cached_object_data_labels.insert(label);
}

void Context::disableObjectDataValueCaching(const std::string &label) {
    cached_object_data_labels.erase(label);
    // Clear cached values for this label
    object_string_value_registry.erase(label);
    object_int_value_registry.erase(label);
    object_uint_value_registry.erase(label);
}

bool Context::isObjectDataValueCachingEnabled(const std::string &label) const {
    return cached_object_data_labels.find(label) != cached_object_data_labels.end();
}


void Context::incrementObjectDataLabelCounter(const std::string &object_data_label) {
    object_data_label_counts[object_data_label]++;
}

void Context::decrementObjectDataLabelCounter(const std::string &object_data_label) {
    auto it = object_data_label_counts.find(object_data_label);
    if (it != object_data_label_counts.end() && it->second > 0) {
        it->second--;
        if (it->second == 0) {
            object_data_label_counts.erase(it);
        }
    }
}

// ------ Primitive Data -------- //

HeliosDataType Primitive::getPrimitiveDataType(const char *label) const {

#ifdef HELIOS_DEBUG
    if (!doesPrimitiveDataExist(label)) {
        helios_runtime_error("ERROR (Primitive::getPrimitiveDataType): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID));
    }
#endif

    return primitive_data_types.at(label);
}

HeliosDataType Context::getPrimitiveDataType(const char *label) const {
    const auto it = primitive_data_type_registry.find(label);
    if (it != primitive_data_type_registry.end()) {
        return it->second;
    }
    helios_runtime_error("ERROR (Context::getPrimitiveDataType): Primitive data " + std::string(label) + " does not exist.");
    return HELIOS_TYPE_UNKNOWN; // Should never reach here, but added to avoid compiler warning
}

uint Primitive::getPrimitiveDataSize(const char *label) const {

#ifdef HELIOS_DEBUG
    if (!doesPrimitiveDataExist(label)) {
        helios_runtime_error("ERROR (Primitive::getPrimitiveDataSize): Primitive data " + std::string(label) + " does not exist for primitive " + std::to_string(UUID));
    }
#endif

    const HeliosDataType &type = primitive_data_types.at(label);

    if (type == HELIOS_TYPE_INT) {
        return primitive_data_int.at(label).size();
    } else if (type == HELIOS_TYPE_UINT) {
        return primitive_data_uint.at(label).size();
    } else if (type == HELIOS_TYPE_FLOAT) {
        return primitive_data_float.at(label).size();
    } else if (type == HELIOS_TYPE_DOUBLE) {
        return primitive_data_double.at(label).size();
    } else if (type == HELIOS_TYPE_VEC2) {
        return primitive_data_vec2.at(label).size();
    } else if (type == HELIOS_TYPE_VEC3) {
        return primitive_data_vec3.at(label).size();
    } else if (type == HELIOS_TYPE_VEC4) {
        return primitive_data_vec4.at(label).size();
    } else if (type == HELIOS_TYPE_INT2) {
        return primitive_data_int2.at(label).size();
    } else if (type == HELIOS_TYPE_INT3) {
        return primitive_data_int3.at(label).size();
    } else if (type == HELIOS_TYPE_INT4) {
        return primitive_data_int4.at(label).size();
    } else if (type == HELIOS_TYPE_STRING) {
        return primitive_data_string.at(label).size();
    } else {
        assert(false);
    }

    return 0;
}

void Primitive::clearPrimitiveData(const char *label) {

    if (!doesPrimitiveDataExist(label)) {
        return;
    }

    HeliosDataType type = primitive_data_types.at(label);

    if (type == HELIOS_TYPE_INT) {
        primitive_data_int.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_UINT) {
        primitive_data_uint.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_FLOAT) {
        primitive_data_float.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_DOUBLE) {
        primitive_data_double.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_VEC2) {
        primitive_data_vec2.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_VEC3) {
        primitive_data_vec3.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_VEC4) {
        primitive_data_vec4.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_INT2) {
        primitive_data_int2.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_INT3) {
        primitive_data_int3.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_INT4) {
        primitive_data_int4.erase(label);
        primitive_data_types.erase(label);
    } else if (type == HELIOS_TYPE_STRING) {
        primitive_data_string.erase(label);
        primitive_data_types.erase(label);
    } else {
        assert(false);
    }
    dirty_flag = true;
}

bool Primitive::doesPrimitiveDataExist(const char *label) const {
    if (primitive_data_types.find(std::string(label)) == primitive_data_types.end()) {
        return false;
    }
    return true;
}

std::vector<std::string> Primitive::listPrimitiveData() const {

    std::vector<std::string> labels;
    labels.reserve(primitive_data_types.size());

    for (const auto &[label, type]: primitive_data_types) {
        labels.push_back(label);
    }

    return labels;
}

HeliosDataType Context::getPrimitiveDataType(uint UUID, const char *label) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getPrimitiveDataType): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return primitives.at(UUID)->getPrimitiveDataType(label);
}

uint Context::getPrimitiveDataSize(uint UUID, const char *label) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getPrimitiveDataSize): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return primitives.at(UUID)->getPrimitiveDataSize(label);
}

bool Context::doesPrimitiveDataExist(uint UUID, const char *label) const {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::doesPrimitiveDataExist): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    return primitives.at(UUID)->doesPrimitiveDataExist(label);
}

void Context::clearPrimitiveData(uint UUID, const char *label) {
#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    }
#endif
    // Handle value registry before clearing if caching is enabled
    std::string label_str = std::string(label);
    if (isPrimitiveDataValueCachingEnabled(label_str) && primitives.at(UUID)->doesPrimitiveDataExist(label)) {
        HeliosDataType data_type = primitives.at(UUID)->getPrimitiveDataType(label);
        if (data_type == HELIOS_TYPE_STRING) {
            std::string cached_value;
            primitives.at(UUID)->getPrimitiveData(label, cached_value);
            decrementPrimitiveValueRegistry(label_str, cached_value);
        } else if (data_type == HELIOS_TYPE_INT) {
            int cached_value;
            primitives.at(UUID)->getPrimitiveData(label, cached_value);
            decrementPrimitiveValueRegistry(label_str, cached_value);
        } else if (data_type == HELIOS_TYPE_UINT) {
            uint cached_value;
            primitives.at(UUID)->getPrimitiveData(label, cached_value);
            decrementPrimitiveValueRegistry(label_str, cached_value);
        }
    }

    if (primitives.at(UUID)->doesPrimitiveDataExist(label)) {
        decrementPrimitiveDataLabelCounter(label);
    }
    primitives.at(UUID)->clearPrimitiveData(label);
}

void Context::clearPrimitiveData(const std::vector<uint> &UUIDs, const char *label) {
    for (unsigned int UUID: UUIDs) {
#ifdef HELIOS_DEBUG
        if (primitives.find(UUID) == primitives.end()) {
            helios_runtime_error("ERROR (Context::getPrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
        }
#endif
        // Handle value registry before clearing if caching is enabled
        std::string label_str = std::string(label);
        if (isPrimitiveDataValueCachingEnabled(label_str) && primitives.at(UUID)->doesPrimitiveDataExist(label)) {
            HeliosDataType data_type = primitives.at(UUID)->getPrimitiveDataType(label);
            if (data_type == HELIOS_TYPE_STRING) {
                std::string cached_value;
                primitives.at(UUID)->getPrimitiveData(label, cached_value);
                decrementPrimitiveValueRegistry(label_str, cached_value);
            } else if (data_type == HELIOS_TYPE_INT) {
                int cached_value;
                primitives.at(UUID)->getPrimitiveData(label, cached_value);
                decrementPrimitiveValueRegistry(label_str, cached_value);
            } else if (data_type == HELIOS_TYPE_UINT) {
                uint cached_value;
                primitives.at(UUID)->getPrimitiveData(label, cached_value);
                decrementPrimitiveValueRegistry(label_str, cached_value);
            }
        }

        if (primitives.at(UUID)->doesPrimitiveDataExist(label)) {
            decrementPrimitiveDataLabelCounter(label);
        }
        primitives.at(UUID)->clearPrimitiveData(label);
    }
}

void Context::copyPrimitiveData(uint sourceUUID, uint destinationUUID) {

#ifdef HELIOS_DEBUG
    if (primitives.find(sourceUUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::copyPrimitiveData): Source UUID of " + std::to_string(sourceUUID) + " does not exist in the Context.");
    } else if (primitives.find(destinationUUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::copyPrimitiveData): Destination UUID of " + std::to_string(destinationUUID) + " does not exist in the Context.");
    }
#endif

    const auto &dest_labels = primitives.at(destinationUUID)->primitive_data_types;
    for (const auto &[label, type]: dest_labels) {
        decrementPrimitiveDataLabelCounter(label);
    }

    primitives.at(destinationUUID)->primitive_data_types = primitives.at(sourceUUID)->primitive_data_types;

    primitives.at(destinationUUID)->primitive_data_int = primitives.at(sourceUUID)->primitive_data_int;
    primitives.at(destinationUUID)->primitive_data_uint = primitives.at(sourceUUID)->primitive_data_uint;
    primitives.at(destinationUUID)->primitive_data_float = primitives.at(sourceUUID)->primitive_data_float;
    primitives.at(destinationUUID)->primitive_data_double = primitives.at(sourceUUID)->primitive_data_double;
    primitives.at(destinationUUID)->primitive_data_vec2 = primitives.at(sourceUUID)->primitive_data_vec2;
    primitives.at(destinationUUID)->primitive_data_vec3 = primitives.at(sourceUUID)->primitive_data_vec3;
    primitives.at(destinationUUID)->primitive_data_vec4 = primitives.at(sourceUUID)->primitive_data_vec4;
    primitives.at(destinationUUID)->primitive_data_int2 = primitives.at(sourceUUID)->primitive_data_int2;
    primitives.at(destinationUUID)->primitive_data_int3 = primitives.at(sourceUUID)->primitive_data_int3;
    primitives.at(destinationUUID)->primitive_data_int4 = primitives.at(sourceUUID)->primitive_data_int4;
    primitives.at(destinationUUID)->primitive_data_string = primitives.at(sourceUUID)->primitive_data_string;

    for (const auto &[label, type]: primitives.at(destinationUUID)->primitive_data_types) {
        incrementPrimitiveDataLabelCounter(label);
    }

    primitives.at(destinationUUID)->dirty_flag = true;
}

void Context::renamePrimitiveData(uint UUID, const char *old_label, const char *new_label) {

#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::renamePrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (!primitives.at(UUID)->doesPrimitiveDataExist(old_label)) {
        helios_runtime_error("ERROR (Context::renamePrimitiveData): Primitive data of " + std::string(old_label) + " does not exist for primitive " + std::to_string(UUID) + ".");
    }
#endif

    duplicatePrimitiveData(UUID, old_label, new_label);
    clearPrimitiveData(UUID, old_label);
    primitives.at(UUID)->dirty_flag = true;
}

void Context::duplicatePrimitiveData(uint UUID, const char *old_label, const char *new_label) {

#ifdef HELIOS_DEBUG
    if (primitives.find(UUID) == primitives.end()) {
        helios_runtime_error("ERROR (Context::duplicatePrimitiveData): UUID of " + std::to_string(UUID) + " does not exist in the Context.");
    } else if (!primitives.at(UUID)->doesPrimitiveDataExist(old_label)) {
        helios_runtime_error("ERROR (Context::duplicatePrimitiveData): Primitive data of " + std::string(old_label) + " does not exist for primitive " + std::to_string(UUID) + ".");
    }
#endif

    HeliosDataType type = getPrimitiveDataType(old_label);

    if (!primitives.at(UUID)->doesPrimitiveDataExist(new_label)) {
        incrementPrimitiveDataLabelCounter(new_label);
    }
    primitives.at(UUID)->primitive_data_types[new_label] = type;
    if (type == HELIOS_TYPE_INT) {
        primitives.at(UUID)->primitive_data_int[new_label] = primitives.at(UUID)->primitive_data_int.at(old_label);
    } else if (type == HELIOS_TYPE_UINT) {
        primitives.at(UUID)->primitive_data_uint[new_label] = primitives.at(UUID)->primitive_data_uint.at(old_label);
    } else if (type == HELIOS_TYPE_FLOAT) {
        primitives.at(UUID)->primitive_data_float[new_label] = primitives.at(UUID)->primitive_data_float.at(old_label);
    } else if (type == HELIOS_TYPE_DOUBLE) {
        primitives.at(UUID)->primitive_data_double[new_label] = primitives.at(UUID)->primitive_data_double.at(old_label);
    } else if (type == HELIOS_TYPE_VEC2) {
        primitives.at(UUID)->primitive_data_vec2[new_label] = primitives.at(UUID)->primitive_data_vec2.at(old_label);
    } else if (type == HELIOS_TYPE_VEC3) {
        primitives.at(UUID)->primitive_data_vec3[new_label] = primitives.at(UUID)->primitive_data_vec3.at(old_label);
    } else if (type == HELIOS_TYPE_VEC4) {
        primitives.at(UUID)->primitive_data_vec4[new_label] = primitives.at(UUID)->primitive_data_vec4.at(old_label);
    } else if (type == HELIOS_TYPE_INT2) {
        primitives.at(UUID)->primitive_data_int2[new_label] = primitives.at(UUID)->primitive_data_int2.at(old_label);
    } else if (type == HELIOS_TYPE_INT3) {
        primitives.at(UUID)->primitive_data_int3[new_label] = primitives.at(UUID)->primitive_data_int3.at(old_label);
    } else if (type == HELIOS_TYPE_INT4) {
        primitives.at(UUID)->primitive_data_int4[new_label] = primitives.at(UUID)->primitive_data_int4.at(old_label);
    } else if (type == HELIOS_TYPE_STRING) {
        primitives.at(UUID)->primitive_data_string[new_label] = primitives.at(UUID)->primitive_data_string.at(old_label);
    } else {
        assert(false);
    }

    primitives.at(UUID)->dirty_flag = true;
}

std::vector<std::string> Context::listPrimitiveData(uint UUID) const {
    return getPrimitivePointer_private(UUID)->listPrimitiveData();
}

void Context::duplicatePrimitiveData(const char *existing_data_label, const char *copy_data_label) {
    for (auto &[UUID, primitive]: primitives) {
        if (primitive->doesPrimitiveDataExist(existing_data_label)) {
            const HeliosDataType type = primitive->getPrimitiveDataType(existing_data_label);
            if (!primitive->doesPrimitiveDataExist(copy_data_label)) {
                incrementPrimitiveDataLabelCounter(copy_data_label);
            }
            primitive->primitive_data_types[copy_data_label] = type;
            if (type == HELIOS_TYPE_FLOAT) {
                primitive->primitive_data_float[copy_data_label] = primitive->primitive_data_float.at(existing_data_label);
            } else if (type == HELIOS_TYPE_DOUBLE) {
                primitive->primitive_data_double[copy_data_label] = primitive->primitive_data_double.at(existing_data_label);
            } else if (type == HELIOS_TYPE_INT) {
                primitive->primitive_data_int[copy_data_label] = primitive->primitive_data_int.at(existing_data_label);
            } else if (type == HELIOS_TYPE_UINT) {
                primitive->primitive_data_uint[copy_data_label] = primitive->primitive_data_uint.at(existing_data_label);
            } else if (type == HELIOS_TYPE_VEC2) {
                primitive->primitive_data_vec2[copy_data_label] = primitive->primitive_data_vec2.at(existing_data_label);
            } else if (type == HELIOS_TYPE_VEC3) {
                primitive->primitive_data_vec3[copy_data_label] = primitive->primitive_data_vec3.at(existing_data_label);
            } else if (type == HELIOS_TYPE_VEC4) {
                primitive->primitive_data_vec4[copy_data_label] = primitive->primitive_data_vec4.at(existing_data_label);
            } else if (type == HELIOS_TYPE_INT2) {
                primitive->primitive_data_int2[copy_data_label] = primitive->primitive_data_int2.at(existing_data_label);
            } else if (type == HELIOS_TYPE_INT3) {
                primitive->primitive_data_int3[copy_data_label] = primitive->primitive_data_int3.at(existing_data_label);
            } else if (type == HELIOS_TYPE_STRING) {
                primitive->primitive_data_string[copy_data_label] = primitive->primitive_data_string.at(existing_data_label);
            }
            primitive->dirty_flag = true;
        }
    }
}

void Context::calculatePrimitiveDataMean(const std::vector<uint> &UUIDs, const std::string &label, float &mean) const {
    float value;
    float sum = 0.f;
    size_t count = 0;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_FLOAT) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum += value;
            count++;
        }
    }

    if (count == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = 0;
    } else {
        mean = sum / float(count);
    }
}

void Context::calculatePrimitiveDataMean(const std::vector<uint> &UUIDs, const std::string &label, double &mean) const {
    double value;
    double sum = 0.f;
    size_t count = 0;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_DOUBLE) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum += value;
            count++;
        }
    }

    if (count == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = 0;
    } else {
        mean = sum / float(count);
    }
}

void Context::calculatePrimitiveDataMean(const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &mean) const {
    vec2 value;
    vec2 sum(0.f, 0.f);
    size_t count = 0;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC2) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            count++;
        }
    }

    if (count == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = make_vec2(0, 0);
    } else {
        mean = sum / float(count);
    }
}

void Context::calculatePrimitiveDataMean(const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &mean) const {
    vec3 value;
    vec3 sum(0.f, 0.f, 0.f);
    size_t count = 0;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC3) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            count++;
        }
    }

    if (count == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = make_vec3(0, 0, 0);
    } else {
        mean = sum / float(count);
    }
}

void Context::calculatePrimitiveDataMean(const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &mean) const {
    vec4 value;
    vec4 sum(0.f, 0.f, 0.f, 0.f);
    size_t count = 0;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC4) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            count++;
        }
    }

    if (count == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        mean = make_vec4(0, 0, 0, 0);
    } else {
        mean = sum / float(count);
    }
}

void Context::setObjectDataFromPrimitiveDataMean(uint objID, const std::string &label) {
#ifdef HELIOS_DEBUG
    if (!doesObjectExist(objID)) {
        helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): Object ID of " + std::to_string(objID) + " does not exist in the Context.");
    }
#endif

    // Get primitive UUIDs for this object
    std::vector<uint> UUIDs = getObjectPrimitiveUUIDs(objID);

    if (UUIDs.empty()) {
        helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): Object ID " + std::to_string(objID) + " has no primitive children.");
    }

    // Determine the data type by checking the first primitive that has this data
    HeliosDataType data_type = HELIOS_TYPE_UNKNOWN;
    for (uint UUID : UUIDs) {
        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str())) {
            data_type = getPrimitiveDataType(label.c_str());
            break;
        }
    }

    if (data_type == HELIOS_TYPE_UNKNOWN) {
        helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): Primitive data '" + label + "' does not exist for any primitives in object " + std::to_string(objID) + ".");
    }

    // Validate data type is supported for mean calculation
    if (data_type != HELIOS_TYPE_FLOAT && data_type != HELIOS_TYPE_DOUBLE &&
        data_type != HELIOS_TYPE_VEC2 && data_type != HELIOS_TYPE_VEC3 && data_type != HELIOS_TYPE_VEC4) {
        helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): Cannot calculate mean for primitive data type. Only float, double, vec2, vec3, and vec4 are supported.");
    }

    // Calculate mean based on data type
    if (data_type == HELIOS_TYPE_FLOAT) {
        float value;
        float sum = 0.f;
        size_t count = 0;
        for (uint UUID : UUIDs) {
            if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str())) {
                getPrimitiveData(UUID, label.c_str(), value);
                sum += value;
                count++;
            }
        }
        if (count == 0) {
            helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): No primitives in object " + std::to_string(objID) + " have primitive data '" + label + "'.");
        }
        float mean = sum / float(count);
        setObjectData(objID, label.c_str(), mean);

    } else if (data_type == HELIOS_TYPE_DOUBLE) {
        double value;
        double sum = 0.0;
        size_t count = 0;
        for (uint UUID : UUIDs) {
            if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str())) {
                getPrimitiveData(UUID, label.c_str(), value);
                sum += value;
                count++;
            }
        }
        if (count == 0) {
            helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): No primitives in object " + std::to_string(objID) + " have primitive data '" + label + "'.");
        }
        double mean = sum / double(count);
        setObjectData(objID, label.c_str(), mean);

    } else if (data_type == HELIOS_TYPE_VEC2) {
        vec2 value;
        vec2 sum(0.f, 0.f);
        size_t count = 0;
        for (uint UUID : UUIDs) {
            if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str())) {
                getPrimitiveData(UUID, label.c_str(), value);
                sum = sum + value;
                count++;
            }
        }
        if (count == 0) {
            helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): No primitives in object " + std::to_string(objID) + " have primitive data '" + label + "'.");
        }
        vec2 mean = sum / float(count);
        setObjectData(objID, label.c_str(), mean);

    } else if (data_type == HELIOS_TYPE_VEC3) {
        vec3 value;
        vec3 sum(0.f, 0.f, 0.f);
        size_t count = 0;
        for (uint UUID : UUIDs) {
            if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str())) {
                getPrimitiveData(UUID, label.c_str(), value);
                sum = sum + value;
                count++;
            }
        }
        if (count == 0) {
            helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): No primitives in object " + std::to_string(objID) + " have primitive data '" + label + "'.");
        }
        vec3 mean = sum / float(count);
        setObjectData(objID, label.c_str(), mean);

    } else if (data_type == HELIOS_TYPE_VEC4) {
        vec4 value;
        vec4 sum(0.f, 0.f, 0.f, 0.f);
        size_t count = 0;
        for (uint UUID : UUIDs) {
            if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str())) {
                getPrimitiveData(UUID, label.c_str(), value);
                sum = sum + value;
                count++;
            }
        }
        if (count == 0) {
            helios_runtime_error("ERROR (Context::setObjectDataFromPrimitiveDataMean): No primitives in object " + std::to_string(objID) + " have primitive data '" + label + "'.");
        }
        vec4 mean = sum / float(count);
        setObjectData(objID, label.c_str(), mean);
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean(const std::vector<uint> &UUIDs, const std::string &label, float &awt_mean) const {
    float value, A;
    float sum = 0.f;
    float area = 0;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_FLOAT) {
            getPrimitiveData(UUID, label.c_str(), value);
            A = getPrimitiveArea(UUID);
            if (std::isnan(A)) {
                nan_warning = true;
            }
            sum += value * A;
            area += A;
        }
    }

    if (area == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = 0;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    } else {
        awt_mean = sum / area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean(const std::vector<uint> &UUIDs, const std::string &label, double &awt_mean) const {
    double value;
    float A;
    double sum = 0.f;
    double area = 0;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_DOUBLE) {
            getPrimitiveData(UUID, label.c_str(), value);
            A = getPrimitiveArea(UUID);
            if (std::isnan(A)) {
                nan_warning = true;
            }
            sum += value * double(A);
            area += A;
        }
    }

    if (area == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = 0;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    } else {
        awt_mean = sum / area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean(const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &awt_mean) const {
    vec2 value;
    vec2 sum(0.f, 0.f);
    float area = 0;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC2) {
            getPrimitiveData(UUID, label.c_str(), value);
            float A = getPrimitiveArea(UUID);
            if (std::isnan(A)) {
                nan_warning = true;
            }
            sum = sum + (value * A);
            area += A;
        }
    }

    if (area == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = make_vec2(0, 0);
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    } else {
        awt_mean = sum / area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean(const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &awt_mean) const {
    vec3 value;
    vec3 sum(0.f, 0.f, 0.f);
    float area = 0;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC3) {
            getPrimitiveData(UUID, label.c_str(), value);
            float A = getPrimitiveArea(UUID);
            if (std::isnan(A)) {
                nan_warning = true;
            }
            sum = sum + (value * A);
            area += A;
        }
    }

    if (area == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = make_vec3(0, 0, 0);
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    } else {
        awt_mean = sum / area;
    }
}

void Context::calculatePrimitiveDataAreaWeightedMean(const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &awt_mean) const {
    vec4 value;
    vec4 sum(0.f, 0.f, 0.f, 0.f);
    float area = 0;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC4) {
            getPrimitiveData(UUID, label.c_str(), value);
            float A = getPrimitiveArea(UUID);
            if (std::isnan(A)) {
                nan_warning = true;
            }
            sum = sum + (value * A);
            area += A;
        }
    }

    if (area == 0) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
        awt_mean = make_vec4(0, 0, 0, 0);
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedMean): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    } else {
        awt_mean = sum / area;
    }
}

void Context::calculatePrimitiveDataSum(const std::vector<uint> &UUIDs, const std::string &label, float &sum) const {

    float value;
    sum = 0.f;
    bool added_to_sum = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_FLOAT) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum += value;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }
}

void Context::calculatePrimitiveDataSum(const std::vector<uint> &UUIDs, const std::string &label, double &sum) const {

    double value;
    sum = 0.f;
    bool added_to_sum = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_DOUBLE) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum += value;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }
}

void Context::calculatePrimitiveDataSum(const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &sum) const {

    vec2 value;
    sum = make_vec2(0.f, 0.f);
    bool added_to_sum = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC2) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }
}

void Context::calculatePrimitiveDataSum(const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &sum) const {

    vec3 value;
    sum = make_vec3(0.f, 0.f, 0.f);
    bool added_to_sum = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC3) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }
}

void Context::calculatePrimitiveDataSum(const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &sum) const {

    vec4 value;
    sum = make_vec4(0.f, 0.f, 0.f, 0.f);
    bool added_to_sum = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC4) {
            getPrimitiveData(UUID, label.c_str(), value);
            sum = sum + value;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    }
}

void Context::calculatePrimitiveDataAreaWeightedSum(const std::vector<uint> &UUIDs, const std::string &label, float &awt_sum) const {

    float value;
    awt_sum = 0.f;
    bool added_to_sum = false;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_FLOAT) {
            float area = getPrimitiveArea(UUID);
            if (std::isnan(area)) {
                nan_warning = true;
                continue;
            }
            getPrimitiveData(UUID, label.c_str(), value);
            awt_sum += value * area;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    }
}

void Context::calculatePrimitiveDataAreaWeightedSum(const std::vector<uint> &UUIDs, const std::string &label, double &awt_sum) const {

    double value;
    awt_sum = 0.f;
    bool added_to_sum = false;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_DOUBLE) {
            float area = getPrimitiveArea(UUID);
            if (std::isnan(area)) {
                nan_warning = true;
                continue;
            }
            getPrimitiveData(UUID, label.c_str(), value);
            awt_sum += value * area;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    }
}

void Context::calculatePrimitiveDataAreaWeightedSum(const std::vector<uint> &UUIDs, const std::string &label, helios::vec2 &awt_sum) const {

    vec2 value;
    awt_sum = make_vec2(0.f, 0.f);
    bool added_to_sum = false;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC2) {
            float area = getPrimitiveArea(UUID);
            if (std::isnan(area)) {
                nan_warning = true;
                continue;
            }
            getPrimitiveData(UUID, label.c_str(), value);
            awt_sum = awt_sum + value * area;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    }
}

void Context::calculatePrimitiveDataAreaWeightedSum(const std::vector<uint> &UUIDs, const std::string &label, helios::vec3 &awt_sum) const {

    vec3 value;
    awt_sum = make_vec3(0.f, 0.f, 0.f);
    bool added_to_sum = false;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC3) {
            float area = getPrimitiveArea(UUID);
            if (std::isnan(area)) {
                nan_warning = true;
                continue;
            }
            getPrimitiveData(UUID, label.c_str(), value);
            awt_sum = awt_sum + value * area;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    }
}

void Context::calculatePrimitiveDataAreaWeightedSum(const std::vector<uint> &UUIDs, const std::string &label, helios::vec4 &awt_sum) const {

    vec4 value;
    awt_sum = make_vec4(0.f, 0.f, 0.f, 0.F);
    bool added_to_sum = false;
    bool nan_warning = false;
    for (uint UUID: UUIDs) {

        if (doesPrimitiveExist(UUID) && doesPrimitiveDataExist(UUID, label.c_str()) && getPrimitiveDataType(label.c_str()) == HELIOS_TYPE_VEC4) {
            float area = getPrimitiveArea(UUID);
            if (std::isnan(area)) {
                nan_warning = true;
                continue;
            }
            getPrimitiveData(UUID, label.c_str(), value);
            awt_sum = awt_sum + value * area;
            added_to_sum = true;
        }
    }

    if (!added_to_sum) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): No primitives found with primitive data of '" << label << "'. Returning a value of 0." << std::endl;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::calculatePrimitiveDataAreaWeightedSum): At least one primitive has an area of NaN and was excluded from calculations" << std::endl;
    }
}

void Context::scalePrimitiveData(const std::vector<uint> &UUIDs, const std::string &label, float scaling_factor) {

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;
    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            primitives_not_exist++;
            continue;
        }
        if (!doesPrimitiveDataExist(UUID, label.c_str())) {
            primitive_data_not_exist++;
            continue;
        }
        HeliosDataType data_type = getPrimitiveDataType(label.c_str());
        if (data_type == HELIOS_TYPE_FLOAT) {
            for (float &data: primitives.at(UUID)->primitive_data_float[label]) {
                data *= scaling_factor;
            }
        } else if (data_type == HELIOS_TYPE_DOUBLE) {
            for (double &data: primitives.at(UUID)->primitive_data_double[label]) {
                data *= scaling_factor;
            }
        } else if (data_type == HELIOS_TYPE_VEC2) {
            for (auto &data: primitives.at(UUID)->primitive_data_vec2[label]) {
                data.x *= scaling_factor;
                data.y *= scaling_factor;
            }
        } else if (data_type == HELIOS_TYPE_VEC3) {
            for (auto &data: primitives.at(UUID)->primitive_data_vec3[label]) {
                data.x *= scaling_factor;
                data.y *= scaling_factor;
                data.z *= scaling_factor;
            }
        } else if (data_type == HELIOS_TYPE_VEC4) {
            for (auto &data: primitives.at(UUID)->primitive_data_vec4[label]) {
                data.x *= scaling_factor;
                data.y *= scaling_factor;
                data.z *= scaling_factor;
                data.w *= scaling_factor;
            }
        } else {
            helios_runtime_error("ERROR (Context::scalePrimitiveData): This operation only supports primitive data of type float, double, vec2, vec3, and vec4.");
        }
        primitives.at(UUID)->dirty_flag = true;
    }

    if (primitives_not_exist > 0) {
        std::cerr << "WARNING (Context::scalePrimitiveData): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if (primitive_data_not_exist > 0) {
        std::cerr << "WARNING (Context::scalePrimitiveData): Primitive data did not exist for " << primitive_data_not_exist << " primitives, and thus no scaling was applied." << std::endl;
    }
}

void Context::scalePrimitiveData(const std::string &label, float scaling_factor) {
    scalePrimitiveData(getAllUUIDs(), label, scaling_factor);
}

void Context::incrementPrimitiveData(const std::vector<uint> &UUIDs, const char *label, int increment) {

    for (uint UUID: UUIDs) {

        if (!doesPrimitiveDataExist(UUID, label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID, label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_INT) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_int.at(label).at(i) += increment;
            }
            primitives.at(UUID)->dirty_flag = true;
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment primitive data for type int, but data '" << label << "' does not have type int." << std::endl;
        }
    }
}

void Context::incrementPrimitiveData(const std::vector<uint> &UUIDs, const char *label, uint increment) {

    for (uint UUID: UUIDs) {

        if (!doesPrimitiveDataExist(UUID, label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID, label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_UINT) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_uint.at(label).at(i) += increment;
            }
            primitives.at(UUID)->dirty_flag = true;
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment Primitive data for type uint, but data '" << label << "' does not have type uint." << std::endl;
        }
    }
}

void Context::incrementPrimitiveData(const std::vector<uint> &UUIDs, const char *label, float increment) {

    for (uint UUID: UUIDs) {

        if (!doesPrimitiveDataExist(UUID, label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID, label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_FLOAT) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_float.at(label).at(i) += increment;
            }
            primitives.at(UUID)->dirty_flag = true;
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment Primitive data for type float, but data '" << label << "' does not have type float." << std::endl;
        }
    }
}

void Context::incrementPrimitiveData(const std::vector<uint> &UUIDs, const char *label, double increment) {

    for (uint UUID: UUIDs) {

        if (!doesPrimitiveDataExist(UUID, label)) {
            helios_runtime_error("ERROR (Context::incrementPrimitiveData): Primitive data " + std::string(label) + " does not exist in the Context for primitive " + std::to_string(UUID) + ".");
        }

        uint size = getPrimitiveDataSize(UUID, label);

        if (primitives.at(UUID)->primitive_data_types.at(label) == HELIOS_TYPE_DOUBLE) {
            for (uint i = 0; i < size; i++) {
                primitives.at(UUID)->primitive_data_double.at(label).at(i) += increment;
            }
            primitives.at(UUID)->dirty_flag = true;
        } else {
            std::cerr << "WARNING (Context::incrementPrimitiveData): Attempted to increment Primitive data for type double, but data '" << label << "' does not have type double." << std::endl;
        }
    }
}

void Context::aggregatePrimitiveDataSum(const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label) {

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;

    float data_float = 0;
    double data_double = 0;
    uint data_uint = 0;
    int data_int = 0;
    int2 data_int2;
    int3 data_int3;
    int4 data_int4;
    vec2 data_vec2;
    vec3 data_vec3;
    vec4 data_vec4;

    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            primitives_not_exist++;
            continue;
        }

        HeliosDataType data_type;

        bool init_type = false;
        for (const auto &label: primitive_data_labels) {

            if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                continue;
            }

            HeliosDataType data_type_current = getPrimitiveDataType(label.c_str());
            if (!init_type) {
                data_type = data_type_current;
                init_type = true;
            } else {
                if (data_type != data_type_current) {
                    helios_runtime_error("ERROR (Context::aggregatePrimitiveDataSum): Primitive data types are not consistent for UUID " + std::to_string(UUID));
                }
            }

            if (data_type_current == HELIOS_TYPE_FLOAT) {
                float data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_float += data;
            } else if (data_type_current == HELIOS_TYPE_DOUBLE) {
                double data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_double += data;
            } else if (data_type_current == HELIOS_TYPE_VEC2) {
                vec2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec2 = data_vec2 + data;
            } else if (data_type_current == HELIOS_TYPE_VEC3) {
                vec3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec3 = data_vec3 + data;
            } else if (data_type_current == HELIOS_TYPE_VEC4) {
                vec4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_vec4 = data_vec4 + data;
            } else if (data_type_current == HELIOS_TYPE_INT) {
                int data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int = data_int + data;
            } else if (data_type_current == HELIOS_TYPE_UINT) {
                uint data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_uint = data_uint + data;
            } else if (data_type_current == HELIOS_TYPE_INT2) {
                int2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int2 = data_int2 + data;
            } else if (data_type_current == HELIOS_TYPE_INT3) {
                int3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int3 = data_int3 + data;
            } else if (data_type_current == HELIOS_TYPE_INT4) {
                int4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                data_int4 = data_int4 + data;
            } else {
                helios_runtime_error("ERROR (Context::aggregatePrimitiveDataSum): This operation is not supported for string primitive data types.");
            }
        }

        if (!init_type) {
            primitive_data_not_exist++;
            continue;
        } else if (data_type == HELIOS_TYPE_FLOAT) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_float);
            data_float = 0;
        } else if (data_type == HELIOS_TYPE_DOUBLE) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_double);
            data_double = 0;
        } else if (data_type == HELIOS_TYPE_VEC2) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_vec2);
            data_vec2 = make_vec2(0, 0);
        } else if (data_type == HELIOS_TYPE_VEC3) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_vec3);
            data_vec3 = make_vec3(0, 0, 0);
        } else if (data_type == HELIOS_TYPE_VEC4) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_vec4);
            data_vec4 = make_vec4(0, 0, 0, 0);
        } else if (data_type == HELIOS_TYPE_INT) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int);
            data_int = 0;
        } else if (data_type == HELIOS_TYPE_UINT) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_uint);
            data_uint = 0;
        } else if (data_type == HELIOS_TYPE_INT2) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int2);
            data_int2 = make_int2(0, 0);
        } else if (data_type == HELIOS_TYPE_INT3) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int3);
            data_int3 = make_int3(0, 0, 0);
        } else if (data_type == HELIOS_TYPE_INT4) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int4);
            data_int4 = make_int4(0, 0, 0, 0);
        }
    }

    if (primitives_not_exist > 0) {
        std::cerr << "WARNING (Context::aggregatePrimitiveDataSum): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if (primitive_data_not_exist > 0) {
        std::cerr << "WARNING (Context::aggregatePrimitiveDataSum): Primitive data did not exist for " << primitive_data_not_exist
                  << " primitives, and thus no scaling summation was performed and new primitive data was not created for this primitive." << std::endl;
    }
}

void Context::aggregatePrimitiveDataProduct(const std::vector<uint> &UUIDs, const std::vector<std::string> &primitive_data_labels, const std::string &result_primitive_data_label) {

    uint primitives_not_exist = 0;
    uint primitive_data_not_exist = 0;

    float data_float = 0;
    double data_double = 0;
    uint data_uint = 0;
    int data_int = 0;
    int2 data_int2;
    int3 data_int3;
    int4 data_int4;
    vec2 data_vec2;
    vec3 data_vec3;
    vec4 data_vec4;

    for (uint UUID: UUIDs) {
        if (!doesPrimitiveExist(UUID)) {
            primitives_not_exist++;
            continue;
        }

        HeliosDataType data_type;

        bool init_type = false;
        int i = 0;
        for (const auto &label: primitive_data_labels) {

            if (!doesPrimitiveDataExist(UUID, label.c_str())) {
                continue;
            }

            HeliosDataType data_type_current = getPrimitiveDataType(label.c_str());
            if (!init_type) {
                data_type = data_type_current;
                init_type = true;
            } else {
                if (data_type != data_type_current) {
                    helios_runtime_error("ERROR (Context::aggregatePrimitiveDataProduct): Primitive data types are not consistent for UUID " + std::to_string(UUID));
                }
            }

            if (data_type_current == HELIOS_TYPE_FLOAT) {
                float data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_float = data;
                } else {
                    data_float *= data;
                }
            } else if (data_type_current == HELIOS_TYPE_DOUBLE) {
                double data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_double *= data;
                } else {
                    data_double = data;
                }
            } else if (data_type_current == HELIOS_TYPE_VEC2) {
                vec2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_vec2.x *= data.x;
                    data_vec2.y *= data.y;
                } else {
                    data_vec2 = data;
                }
            } else if (data_type_current == HELIOS_TYPE_VEC3) {
                vec3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_vec3.x *= data.x;
                    data_vec3.y *= data.y;
                    data_vec3.z *= data.z;
                } else {
                    data_vec3 = data;
                }
            } else if (data_type_current == HELIOS_TYPE_VEC4) {
                vec4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_vec4.x *= data.x;
                    data_vec4.y *= data.y;
                    data_vec4.z *= data.z;
                    data_vec4.w *= data.w;
                } else {
                    data_vec4 = data;
                }
            } else if (data_type_current == HELIOS_TYPE_INT) {
                int data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_int = data_int * data;
                } else {
                    data_int = data;
                }
            } else if (data_type_current == HELIOS_TYPE_UINT) {
                uint data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_uint = data_uint * data;
                } else {
                    data_uint = data;
                }
            } else if (data_type_current == HELIOS_TYPE_INT2) {
                int2 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_int2.x *= data.x;
                    data_int2.y *= data.y;
                } else {
                    data_int2 = data;
                }
            } else if (data_type_current == HELIOS_TYPE_INT3) {
                int3 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_int3.x *= data.x;
                    data_int3.y *= data.y;
                    data_int3.z *= data.z;
                } else {
                    data_int3 = data;
                }
            } else if (data_type_current == HELIOS_TYPE_INT4) {
                int4 data;
                primitives.at(UUID)->getPrimitiveData(label.c_str(), data);
                if (i == 0) {
                    data_int4.x *= data.x;
                    data_int4.y *= data.y;
                    data_int4.z *= data.z;
                    data_int4.w *= data.w;
                } else {
                    data_int4 = data;
                }
            } else {
                helios_runtime_error("ERROR (Context::aggregatePrimitiveDataProduct): This operation is not supported for string primitive data types.");
            }
            i++;
        }

        if (!init_type) {
            primitive_data_not_exist++;
            continue;
        } else if (data_type == HELIOS_TYPE_FLOAT) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_float);
        } else if (data_type == HELIOS_TYPE_DOUBLE) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_double);
        } else if (data_type == HELIOS_TYPE_VEC2) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_vec2);
        } else if (data_type == HELIOS_TYPE_VEC3) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_vec3);
        } else if (data_type == HELIOS_TYPE_VEC4) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_vec4);
        } else if (data_type == HELIOS_TYPE_INT) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int);
        } else if (data_type == HELIOS_TYPE_UINT) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_uint);
        } else if (data_type == HELIOS_TYPE_INT2) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int2);
        } else if (data_type == HELIOS_TYPE_INT3) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int3);
        } else if (data_type == HELIOS_TYPE_INT4) {
            setPrimitiveData(UUID, result_primitive_data_label.c_str(), data_int4);
        }
    }

    if (primitives_not_exist > 0) {
        std::cerr << "WARNING (Context::aggregatePrimitiveDataProduct): " << primitives_not_exist << " of " << UUIDs.size() << " from the input UUID vector did not exist." << std::endl;
    }
    if (primitive_data_not_exist > 0) {
        std::cerr << "WARNING (Context::aggregatePrimitiveDataProduct): Primitive data did not exist for " << primitive_data_not_exist
                  << " primitives, and thus no multiplication was performed and new primitive data was not created for this primitive." << std::endl;
    }
}


float Context::sumPrimitiveSurfaceArea(const std::vector<uint> &UUIDs) const {

    bool primitive_warning = false;
    bool nan_warning = false;
    float area = 0;
    for (uint UUID: UUIDs) {

        float A = getPrimitiveArea(UUID);

        if (std::isnan(A)) {
            nan_warning = true;
            continue;
        }

        if (doesPrimitiveExist(UUID)) {
            area += A;
        } else {
            primitive_warning = true;
        }
    }

    if (primitive_warning) {
        std::cerr << "WARNING (Context::sumPrimitiveSurfaceArea): One or more primitives reference in the UUID vector did not exist." << std::endl;
    } else if (nan_warning) {
        std::cerr << "WARNING (Context::sumPrimitiveSurfaceArea): One or more primitives had an area of NaN." << std::endl;
    }

    return area;
}

std::vector<uint> Context::filterPrimitivesByData(const std::vector<uint> &UUIDs, const std::string &primitive_data_label, float filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for (std::size_t p = UUIDs.size(); p-- > 0;) {
        uint UUID = UUIDs_out.at(p);
        if (doesPrimitiveDataExist(UUID, primitive_data_label.c_str()) && getPrimitiveDataType(primitive_data_label.c_str()) == HELIOS_TYPE_FLOAT) {
            float data;
            getPrimitiveData(UUID, primitive_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            }
            if (comparator == ">" && data > filter_value) {
                continue;
            }
            if (comparator == "<" && data < filter_value) {
                continue;
            }
            if (comparator == ">=" && data >= filter_value) {
                continue;
            }
            if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        } else {
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;
}

std::vector<uint> Context::filterPrimitivesByData(const std::vector<uint> &UUIDs, const std::string &primitive_data_label, double filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for (std::size_t p = UUIDs.size(); p-- > 0;) {
        uint UUID = UUIDs_out.at(p);
        if (doesPrimitiveDataExist(UUID, primitive_data_label.c_str()) && getPrimitiveDataType(primitive_data_label.c_str()) == HELIOS_TYPE_DOUBLE) {
            double data;
            getPrimitiveData(UUID, primitive_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            }
            if (comparator == ">" && data > filter_value) {
                continue;
            }
            if (comparator == "<" && data < filter_value) {
                continue;
            }
            if (comparator == ">=" && data >= filter_value) {
                continue;
            }
            if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        } else {
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;
}

std::vector<uint> Context::filterPrimitivesByData(const std::vector<uint> &UUIDs, const std::string &primitive_data_label, int filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for (std::size_t p = UUIDs.size(); p-- > 0;) {
        uint UUID = UUIDs_out.at(p);
        if (doesPrimitiveDataExist(UUID, primitive_data_label.c_str()) && getPrimitiveDataType(primitive_data_label.c_str()) == HELIOS_TYPE_INT) {
            int data;
            getPrimitiveData(UUID, primitive_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            }
            if (comparator == ">" && data > filter_value) {
                continue;
            }
            if (comparator == "<" && data < filter_value) {
                continue;
            }
            if (comparator == ">=" && data >= filter_value) {
                continue;
            }
            if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        } else {
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;
}

std::vector<uint> Context::filterPrimitivesByData(const std::vector<uint> &UUIDs, const std::string &primitive_data_label, uint filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterPrimitivesByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> UUIDs_out = UUIDs;
    for (std::size_t p = UUIDs.size(); p-- > 0;) {
        uint UUID = UUIDs_out.at(p);
        if (doesPrimitiveDataExist(UUID, primitive_data_label.c_str()) && getPrimitiveDataType(primitive_data_label.c_str()) == HELIOS_TYPE_UINT) {
            uint data;
            getPrimitiveData(UUID, primitive_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            }
            if (comparator == ">" && data > filter_value) {
                continue;
            }
            if (comparator == "<" && data < filter_value) {
                continue;
            }
            if (comparator == ">=" && data >= filter_value) {
                continue;
            }
            if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        } else {
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;
}

std::vector<uint> Context::filterPrimitivesByData(const std::vector<uint> &UUIDs, const std::string &primitive_data_label, const std::string &filter_value) const {

    std::vector<uint> UUIDs_out = UUIDs;
    for (std::size_t p = UUIDs.size(); p-- > 0;) {
        uint UUID = UUIDs_out.at(p);
        if (doesPrimitiveDataExist(UUID, primitive_data_label.c_str()) && getPrimitiveDataType(primitive_data_label.c_str()) == HELIOS_TYPE_STRING) {
            std::string data;
            getPrimitiveData(UUID, primitive_data_label.c_str(), data);
            if (data != filter_value) {
                std::swap(UUIDs_out.at(p), UUIDs_out.back());
                UUIDs_out.pop_back();
            }
        } else {
            std::swap(UUIDs_out.at(p), UUIDs_out.back());
            UUIDs_out.pop_back();
        }
    }

    return UUIDs_out;
}

//------ Object Data ------- //

HeliosDataType Context::getObjectDataType(uint objID, const char *label) const {
#ifdef HELIOS_DEBUG
    if (objects.find(objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectDataType): objID of " + std::to_string(objID) + " does not exist in the Context.");
    }
#endif
    return objects.at(objID)->getObjectDataType(label);
}

HeliosDataType Context::getObjectDataType(const char *label) const {
    const auto it = object_data_type_registry.find(label);
    if (it != object_data_type_registry.end()) {
        return it->second;
    }
    helios_runtime_error("ERROR (Context::getObjectDataType): Object data " + std::string(label) + " does not exist.");
    return HELIOS_TYPE_UNKNOWN; // This line will never be reached, but is needed to avoid compiler warnings.
}

uint Context::getObjectDataSize(uint objID, const char *label) const {
#ifdef HELIOS_DEBUG
    if (objects.find(objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::getObjectDataSize): objID of " + std::to_string(objID) + " does not exist in the Context.");
    }
#endif
    return objects.at(objID)->getObjectDataSize(label);
}

bool Context::doesObjectDataExist(uint objID, const char *label) const {
#ifdef HELIOS_DEBUG
    if (objects.find(objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::doesObjectDataExist): objID of " + std::to_string(objID) + " does not exist in the Context.");
    }
#endif
    return objects.at(objID)->doesObjectDataExist(label);
}

void Context::copyObjectData(uint source_objID, uint destination_objID) {

#ifdef HELIOS_DEBUG
    if (objects.find(source_objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::copyObjectData): Source object ID of " + std::to_string(source_objID) + " does not exist in the Context.");
    } else if (objects.find(destination_objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::copyObjectData): Destination object ID of " + std::to_string(destination_objID) + " does not exist in the Context.");
    }
#endif

    const auto &dest_labels = objects.at(destination_objID)->object_data_types;
    for (const auto &[label, type]: dest_labels) {
        decrementObjectDataLabelCounter(label);
    }

    objects.at(destination_objID)->object_data_types = objects.at(source_objID)->object_data_types;

    objects.at(destination_objID)->object_data_int = objects.at(source_objID)->object_data_int;
    objects.at(destination_objID)->object_data_uint = objects.at(source_objID)->object_data_uint;
    objects.at(destination_objID)->object_data_float = objects.at(source_objID)->object_data_float;
    objects.at(destination_objID)->object_data_double = objects.at(source_objID)->object_data_double;
    objects.at(destination_objID)->object_data_vec2 = objects.at(source_objID)->object_data_vec2;
    objects.at(destination_objID)->object_data_vec3 = objects.at(source_objID)->object_data_vec3;
    objects.at(destination_objID)->object_data_vec4 = objects.at(source_objID)->object_data_vec4;
    objects.at(destination_objID)->object_data_int2 = objects.at(source_objID)->object_data_int2;
    objects.at(destination_objID)->object_data_int3 = objects.at(source_objID)->object_data_int3;
    objects.at(destination_objID)->object_data_int4 = objects.at(source_objID)->object_data_int4;
    objects.at(destination_objID)->object_data_string = objects.at(source_objID)->object_data_string;

    for (const auto &[lbl, type]: objects.at(destination_objID)->object_data_types) {
        incrementObjectDataLabelCounter(lbl);
    }
}

void Context::duplicateObjectData(uint objID, const char *old_label, const char *new_label) {

#ifdef HELIOS_DEBUG
    if (objects.find(objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::duplicateObjectData): Object ID of " + std::to_string(objID) + " does not exist in the Context.");
    } else if (!doesObjectDataExist(objID, old_label)) {
        helios_runtime_error("ERROR (Context::duplicateObjectData): Object ID of " + std::to_string(objID) + " does not have data with label " + std::string(old_label) + ".");
    }
#endif

    HeliosDataType type = getObjectDataType(old_label);

    if (!objects.at(objID)->doesObjectDataExist(new_label)) {
        incrementObjectDataLabelCounter(new_label);
    }
    objects.at(objID)->object_data_types[new_label] = type;
    if (type == HELIOS_TYPE_INT) {
        objects.at(objID)->object_data_int[new_label] = objects.at(objID)->object_data_int.at(old_label);
    } else if (type == HELIOS_TYPE_UINT) {
        objects.at(objID)->object_data_uint[new_label] = objects.at(objID)->object_data_uint.at(old_label);
    } else if (type == HELIOS_TYPE_FLOAT) {
        objects.at(objID)->object_data_float[new_label] = objects.at(objID)->object_data_float.at(old_label);
    } else if (type == HELIOS_TYPE_DOUBLE) {
        objects.at(objID)->object_data_double[new_label] = objects.at(objID)->object_data_double.at(old_label);
    } else if (type == HELIOS_TYPE_VEC2) {
        objects.at(objID)->object_data_vec2[new_label] = objects.at(objID)->object_data_vec2.at(old_label);
    } else if (type == HELIOS_TYPE_VEC3) {
        objects.at(objID)->object_data_vec3[new_label] = objects.at(objID)->object_data_vec3.at(old_label);
    } else if (type == HELIOS_TYPE_VEC4) {
        objects.at(objID)->object_data_vec4[new_label] = objects.at(objID)->object_data_vec4.at(old_label);
    } else if (type == HELIOS_TYPE_INT2) {
        objects.at(objID)->object_data_int2[new_label] = objects.at(objID)->object_data_int2.at(old_label);
    } else if (type == HELIOS_TYPE_INT3) {
        objects.at(objID)->object_data_int3[new_label] = objects.at(objID)->object_data_int3.at(old_label);
    } else if (type == HELIOS_TYPE_INT4) {
        objects.at(objID)->object_data_int4[new_label] = objects.at(objID)->object_data_int4.at(old_label);
    } else if (type == HELIOS_TYPE_STRING) {
        objects.at(objID)->object_data_string[new_label] = objects.at(objID)->object_data_string.at(old_label);
    } else {
        assert(false);
    }
}


void Context::renameObjectData(uint objID, const char *old_label, const char *new_label) {

#ifdef HELIOS_DEBUG
    if (objects.find(objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::renameObjectData): Object ID of " + std::to_string(objID) + " does not exist in the Context.");
    } else if (!doesObjectDataExist(objID, old_label)) {
        helios_runtime_error("ERROR (Context::renameObjectData): Object ID of " + std::to_string(objID) + " does not have data with label " + std::string(old_label) + ".");
    }
#endif

    duplicateObjectData(objID, old_label, new_label);
    clearObjectData(objID, old_label);
}

void Context::clearObjectData(uint objID, const char *label) {
#ifdef HELIOS_DEBUG
    if (objects.find(objID) == objects.end()) {
        helios_runtime_error("ERROR (Context::clearObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
    }
#endif
    // Handle value registry before clearing if caching is enabled
    std::string label_str = std::string(label);
    if (isObjectDataValueCachingEnabled(label_str) && objects.at(objID)->doesObjectDataExist(label)) {
        HeliosDataType data_type = objects.at(objID)->getObjectDataType(label);
        if (data_type == HELIOS_TYPE_STRING) {
            std::string cached_value;
            objects.at(objID)->getObjectData(label, cached_value);
            decrementObjectValueRegistry(label_str, cached_value);
        } else if (data_type == HELIOS_TYPE_INT) {
            int cached_value;
            objects.at(objID)->getObjectData(label, cached_value);
            decrementObjectValueRegistry(label_str, cached_value);
        } else if (data_type == HELIOS_TYPE_UINT) {
            uint cached_value;
            objects.at(objID)->getObjectData(label, cached_value);
            decrementObjectValueRegistry(label_str, cached_value);
        }
    }

    if (objects.at(objID)->doesObjectDataExist(label)) {
        decrementObjectDataLabelCounter(label);
    }
    objects.at(objID)->clearObjectData(label);
}

void Context::clearObjectData(const std::vector<uint> &objIDs, const char *label) {
    std::string label_str = std::string(label);
    for (uint objID: objIDs) {
#ifdef HELIOS_DEBUG
        if (objects.find(objID) == objects.end()) {
            helios_runtime_error("ERROR (Context::clearObjectData): objID of " + std::to_string(objID) + " does not exist in the Context.");
        }
#endif
        // Handle value registry before clearing if caching is enabled
        if (isObjectDataValueCachingEnabled(label_str) && objects.at(objID)->doesObjectDataExist(label)) {
            HeliosDataType data_type = objects.at(objID)->getObjectDataType(label);
            if (data_type == HELIOS_TYPE_STRING) {
                std::string cached_value;
                objects.at(objID)->getObjectData(label, cached_value);
                decrementObjectValueRegistry(label_str, cached_value);
            } else if (data_type == HELIOS_TYPE_INT) {
                int cached_value;
                objects.at(objID)->getObjectData(label, cached_value);
                decrementObjectValueRegistry(label_str, cached_value);
            } else if (data_type == HELIOS_TYPE_UINT) {
                uint cached_value;
                objects.at(objID)->getObjectData(label, cached_value);
                decrementObjectValueRegistry(label_str, cached_value);
            }
        }

        if (objects.at(objID)->doesObjectDataExist(label)) {
            decrementObjectDataLabelCounter(label);
        }
        objects.at(objID)->clearObjectData(label);
    }
}

std::vector<std::string> Context::listObjectData(uint ObjID) const {
    return getObjectPointer_private(ObjID)->listObjectData();
}

HeliosDataType CompoundObject::getObjectDataType(const char *label) const {

#ifdef HELIOS_DEBUG
    if (!doesObjectDataExist(label)) {
        helios_runtime_error("ERROR (CompoundObject::getObjectDataType): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
    }
#endif

    return object_data_types.at(label);
}

uint CompoundObject::getObjectDataSize(const char *label) const {

#ifdef HELIOS_DEBUG
    if (!doesObjectDataExist(label)) {
        helios_runtime_error("ERROR (CompoundObject::getObjectDataSize): Object data " + std::string(label) + " does not exist for object " + std::to_string(OID));
    }
#endif

    HeliosDataType qtype = object_data_types.at(label);

    if (qtype == HELIOS_TYPE_INT) {
        return object_data_int.at(label).size();
    } else if (qtype == HELIOS_TYPE_UINT) {
        return object_data_uint.at(label).size();
    } else if (qtype == HELIOS_TYPE_FLOAT) {
        return object_data_float.at(label).size();
    } else if (qtype == HELIOS_TYPE_DOUBLE) {
        return object_data_double.at(label).size();
    } else if (qtype == HELIOS_TYPE_VEC2) {
        return object_data_vec2.at(label).size();
    } else if (qtype == HELIOS_TYPE_VEC3) {
        return object_data_vec3.at(label).size();
    } else if (qtype == HELIOS_TYPE_VEC4) {
        return object_data_vec4.at(label).size();
    } else if (qtype == HELIOS_TYPE_INT2) {
        return object_data_int2.at(label).size();
    } else if (qtype == HELIOS_TYPE_INT3) {
        return object_data_int3.at(label).size();
    } else if (qtype == HELIOS_TYPE_INT4) {
        return object_data_int4.at(label).size();
    } else if (qtype == HELIOS_TYPE_STRING) {
        return object_data_string.at(label).size();
    } else {
        assert(false);
    }

    return 0;
}

void CompoundObject::clearObjectData(const char *label) {

    if (!doesObjectDataExist(label)) {
        return;
    }

    const HeliosDataType &qtype = object_data_types.at(label);

    if (qtype == HELIOS_TYPE_INT) {
        object_data_int.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_UINT) {
        object_data_uint.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_FLOAT) {
        object_data_float.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_DOUBLE) {
        object_data_double.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_VEC2) {
        object_data_vec2.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_VEC3) {
        object_data_vec3.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_VEC4) {
        object_data_vec4.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_INT2) {
        object_data_int2.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_INT3) {
        object_data_int3.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_INT4) {
        object_data_int4.erase(label);
        object_data_types.erase(label);
    } else if (qtype == HELIOS_TYPE_STRING) {
        object_data_string.erase(label);
        object_data_types.erase(label);
    } else {
        assert(false);
    }
}

bool CompoundObject::doesObjectDataExist(const char *label) const {

    if (object_data_types.find(std::string(label)) == object_data_types.end()) {
        return false;
    } else {
        return true;
    }
}

std::vector<std::string> CompoundObject::listObjectData() const {

    std::vector<std::string> labels;
    labels.reserve(object_data_types.size());

    for (const auto &[label, type]: object_data_types) {
        labels.push_back(label);
    }

    return labels;
}

std::vector<uint> Context::filterObjectsByData(const std::vector<uint> &objIDs, const std::string &object_data_label, float filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterObjectsByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> objIDs_out = objIDs;
    for (std::size_t p = objIDs.size(); p-- > 0;) {
        uint objID = objIDs_out.at(p);
        if (doesObjectDataExist(objID, object_data_label.c_str()) && getObjectDataType(object_data_label.c_str()) == HELIOS_TYPE_FLOAT) {
            float data;
            getObjectData(objID, object_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            } else if (comparator == ">" && data > filter_value) {
                continue;
            } else if (comparator == "<" && data < filter_value) {
                continue;
            } else if (comparator == ">=" && data >= filter_value) {
                continue;
            } else if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        } else {
            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        }
    }

    return objIDs_out;
}

std::vector<uint> Context::filterObjectsByData(const std::vector<uint> &objIDs, const std::string &object_data_label, double filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterObjectsByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> objIDs_out = objIDs;
    for (std::size_t p = objIDs.size(); p-- > 0;) {
        uint objID = objIDs_out.at(p);
        if (doesObjectDataExist(objID, object_data_label.c_str()) && getObjectDataType(object_data_label.c_str()) == HELIOS_TYPE_DOUBLE) {
            double data;
            getObjectData(objID, object_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            } else if (comparator == ">" && data > filter_value) {
                continue;
            } else if (comparator == "<" && data < filter_value) {
                continue;
            } else if (comparator == ">=" && data >= filter_value) {
                continue;
            } else if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        } else {
            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        }
    }

    return objIDs_out;
}

std::vector<uint> Context::filterObjectsByData(const std::vector<uint> &objIDs, const std::string &object_data_label, int filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterObjectsByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> objIDs_out = objIDs;
    for (std::size_t p = objIDs.size(); p-- > 0;) {
        uint objID = objIDs_out.at(p);
        if (doesObjectDataExist(objID, object_data_label.c_str()) && getObjectDataType(object_data_label.c_str()) == HELIOS_TYPE_INT) {
            int data;
            getObjectData(objID, object_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            } else if (comparator == ">" && data > filter_value) {
                continue;
            } else if (comparator == "<" && data < filter_value) {
                continue;
            } else if (comparator == ">=" && data >= filter_value) {
                continue;
            } else if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        } else {
            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        }
    }

    return objIDs_out;
}

std::vector<uint> Context::filterObjectsByData(const std::vector<uint> &objIDs, const std::string &object_data_label, uint filter_value, const std::string &comparator) const {

    if (comparator != "==" && comparator != ">" && comparator != "<" && comparator != ">=" && comparator != "<=") {
        helios_runtime_error("ERROR (Context::filterObjectsByData): Invalid comparator. Must be one of '==', '>', '<', '>=', or '<='.");
    }

    std::vector<uint> objIDs_out = objIDs;
    for (std::size_t p = objIDs.size(); p-- > 0;) {
        uint objID = objIDs_out.at(p);
        if (doesObjectDataExist(objID, object_data_label.c_str()) && getObjectDataType(object_data_label.c_str()) == HELIOS_TYPE_UINT) {
            uint data;
            getObjectData(objID, object_data_label.c_str(), data);
            if (comparator == "==" && data == filter_value) {
                continue;
            } else if (comparator == ">" && data > filter_value) {
                continue;
            } else if (comparator == "<" && data < filter_value) {
                continue;
            } else if (comparator == ">=" && data >= filter_value) {
                continue;
            } else if (comparator == "<=" && data <= filter_value) {
                continue;
            }

            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        } else {
            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        }
    }

    return objIDs_out;
}

std::vector<uint> Context::filterObjectsByData(const std::vector<uint> &objIDs, const std::string &object_data_label, const std::string &filter_value) const {

    std::vector<uint> objIDs_out = objIDs;
    for (std::size_t p = objIDs.size(); p-- > 0;) {
        uint objID = objIDs_out.at(p);
        if (doesObjectDataExist(objID, object_data_label.c_str()) && getObjectDataType(object_data_label.c_str()) == HELIOS_TYPE_STRING) {
            std::string data;
            getObjectData(objID, object_data_label.c_str(), data);
            if (data != filter_value) {
                std::swap(objIDs_out.at(p), objIDs_out.back());
                objIDs_out.pop_back();
            }
        } else {
            std::swap(objIDs_out.at(p), objIDs_out.back());
            objIDs_out.pop_back();
        }
    }

    return objIDs_out;
}

// -------- Global Data ---------- //

void Context::renameGlobalData(const char *old_label, const char *new_label) {

    if (!doesGlobalDataExist(old_label)) {
        helios_runtime_error("ERROR (Context::duplicateGlobalData): Global data " + std::string(old_label) + " does not exist in the Context.");
    }

    duplicateGlobalData(old_label, new_label);
    clearGlobalData(old_label);
}

void Context::duplicateGlobalData(const char *old_label, const char *new_label) {

    if (!doesGlobalDataExist(old_label)) {
        helios_runtime_error("ERROR (Context::duplicateGlobalData): Global data " + std::string(old_label) + " does not exist in the Context.");
    }

    HeliosDataType type = getGlobalDataType(old_label);

    if (type == HELIOS_TYPE_INT) {
        std::vector<int> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_UINT) {
        std::vector<uint> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_FLOAT) {
        std::vector<float> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_DOUBLE) {
        std::vector<double> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_VEC2) {
        std::vector<vec2> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_VEC3) {
        std::vector<vec3> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_VEC4) {
        std::vector<vec4> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_INT2) {
        std::vector<int2> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_INT3) {
        std::vector<int3> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_INT4) {
        std::vector<int4> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else if (type == HELIOS_TYPE_STRING) {
        std::vector<std::string> gdata;
        getGlobalData(old_label, gdata);
        setGlobalData(new_label, gdata);
    } else {
        assert(false);
    }
}

void Context::clearGlobalData(const char *label) {

    if (doesGlobalDataExist(label)) {
        globaldata.erase(label);
    }
}

HeliosDataType Context::getGlobalDataType(const char *label) const {

    if (!doesGlobalDataExist(label)) {
        helios_runtime_error("ERROR (Context::getGlobalDataType): Global data " + std::string(label) + " does not exist in the Context.");
    }

    return globaldata.at(label).type;
}

size_t Context::getGlobalDataSize(const char *label) const {

    if (!doesGlobalDataExist(label)) {
        helios_runtime_error("ERROR (Context::getGlobalDataSize): Global data " + std::string(label) + " does not exist in the Context.");
    }

    return globaldata.at(label).size;
}

std::vector<std::string> Context::listGlobalData() const {

    std::vector<std::string> labels;
    labels.reserve(globaldata.size());
    for (const auto &[label, data]: globaldata) {
        labels.push_back(label);
    }

    return labels;
}

std::vector<std::string> Context::listAllPrimitiveDataLabels() const {
    std::vector<std::string> labels;
    labels.reserve(primitive_data_label_counts.size());
    for (const auto &[label, count]: primitive_data_label_counts) {
        if (count > 0) {
            labels.push_back(label);
        }
    }
    return labels;
}


std::vector<std::string> Context::listAllObjectDataLabels() const {
    std::vector<std::string> labels;
    labels.reserve(object_data_label_counts.size());
    for (const auto &[label, count]: object_data_label_counts) {
        if (count > 0) {
            labels.push_back(label);
        }
    }
    return labels;
}

bool Context::doesGlobalDataExist(const char *label) const {
    return globaldata.find(std::string(label)) != globaldata.end();
}

void Context::incrementGlobalData(const char *label, int increment) {

    if (!doesGlobalDataExist(label)) {
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if (globaldata.at(label).type == HELIOS_TYPE_INT) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_int.at(i) += increment;
        }
    } else {
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type int, but data '" << label << "' does not have type int." << std::endl;
    }
}

void Context::incrementGlobalData(const char *label, uint increment) {

    if (!doesGlobalDataExist(label)) {
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if (globaldata.at(label).type == HELIOS_TYPE_UINT) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_uint.at(i) += increment;
        }
    } else {
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type uint, but data '" << label << "' does not have type uint." << std::endl;
    }
}

void Context::incrementGlobalData(const char *label, float increment) {

    if (!doesGlobalDataExist(label)) {
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if (globaldata.at(label).type == HELIOS_TYPE_FLOAT) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_float.at(i) += increment;
        }
    } else {
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type float, but data '" << label << "' does not have type float." << std::endl;
    }
}

void Context::incrementGlobalData(const char *label, double increment) {

    if (!doesGlobalDataExist(label)) {
        helios_runtime_error("ERROR (Context::incrementGlobalData): Global data " + std::string(label) + " does not exist in the Context.");
    }

    uint size = getGlobalDataSize(label);

    if (globaldata.at(label).type == HELIOS_TYPE_DOUBLE) {
        for (uint i = 0; i < size; i++) {
            globaldata.at(label).global_data_double.at(i) += increment;
        }
    } else {
        std::cerr << "WARNING (Context::incrementGlobalData): Attempted to increment global data for type double, but data '" << label << "' does not have type double." << std::endl;
    }
}

std::string Context::dataTypeToString(HeliosDataType type) const {
    switch (type) {
        case HELIOS_TYPE_INT:
            return "int";
        case HELIOS_TYPE_UINT:
            return "uint";
        case HELIOS_TYPE_FLOAT:
            return "float";
        case HELIOS_TYPE_DOUBLE:
            return "double";
        case HELIOS_TYPE_VEC2:
            return "vec2";
        case HELIOS_TYPE_VEC3:
            return "vec3";
        case HELIOS_TYPE_VEC4:
            return "vec4";
        case HELIOS_TYPE_INT2:
            return "int2";
        case HELIOS_TYPE_INT3:
            return "int3";
        case HELIOS_TYPE_INT4:
            return "int4";
        case HELIOS_TYPE_STRING:
            return "string";
        case HELIOS_TYPE_BOOL:
            return "bool";
        case HELIOS_TYPE_UNKNOWN:
            return "unknown";
        default:
            return "undefined";
    }
}

bool Context::isTypeCastingSupported(HeliosDataType from_type, HeliosDataType to_type) const {
    // Support casting between numeric types only
    const std::set<HeliosDataType> numeric_types = {HELIOS_TYPE_INT, HELIOS_TYPE_UINT, HELIOS_TYPE_FLOAT, HELIOS_TYPE_DOUBLE};

    return (numeric_types.count(from_type) > 0 && numeric_types.count(to_type) > 0);
}
