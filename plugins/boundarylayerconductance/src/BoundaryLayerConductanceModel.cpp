/** \file "BoundaryLayerConductanceModel.cpp" Boundary-layer conductance  model plugin declarations.

    Copyright (C) 2016-2026 Brian Bailey

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Lesser General Public
    License as published by the Free Software Foundation; either
    version 2.1 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Lesser General Public License for more details.

    SPDX-License-Identifier: LGPL-2.1-or-later

*/

#include "BoundaryLayerConductanceModel.h"
#include "GraefeConvection.h"

using namespace helios;

namespace {
    // Ratios of the boundary-layer conductance to water vapor to that to heat, and of that to water vapor to that to CO2, used to give the models other than "Graefe" per-face scalar conductances. These are the
    // same ratios the energy balance and photosynthesis plug-ins use when they are given only the conductance to heat.
    constexpr float vapor_to_heat_conductance_ratio = 1.08f;
    constexpr float vapor_to_CO2_conductance_ratio = 1.37f;

    graefe::Vec3 toGraefeVec3(const vec3 &v) {
        return {double(v.x), double(v.y), double(v.z)};
    }
} // namespace

BLConductanceModel::BLConductanceModel(helios::Context *__context) {

    wind_speed_default = 1.f;

    air_temperature_default = 293.f;

    surface_temperature_default = 303.f;

    message_flag = true; // print messages to screen by default

    // Copy pointer to the context
    context = __context;
}

void BLConductanceModel::enableMessages() {
    message_flag = true;
}

void BLConductanceModel::disableMessages() {
    message_flag = false;
}

void BLConductanceModel::setBoundaryLayerModel(const char *gH_model) {
    std::vector<uint> UUIDs = context->getAllUUIDs();
    setBoundaryLayerModel(UUIDs, gH_model);
}

void BLConductanceModel::setBoundaryLayerModel(uint UUID, const char *gH_model) {
    std::vector<uint> UUIDs{UUID};
    setBoundaryLayerModel(UUIDs, gH_model);
}

void BLConductanceModel::setBoundaryLayerModel(const std::vector<uint> &UUIDs, const char *gH_model) {

    uint model = 0;

    if (strcmp(gH_model, "Pohlhausen") == 0 || strcmp(gH_model, "Polhausen") == 0) {
        model = 0;
    } else if (strcmp(gH_model, "InclinedPlate") == 0) {
        model = 1;
    } else if (strcmp(gH_model, "Sphere") == 0) {
        model = 2;
    } else if (strcmp(gH_model, "Ground") == 0) {
        model = 3;
    } else if (strcmp(gH_model, "Graefe") == 0) {
        model = 4;
    } else {
        helios_runtime_error("ERROR (BLConductanceModel::setBoundaryLayerModel): Boundary-layer conductance model \"" + std::string(gH_model) +
                             "\" is unknown. Valid models are \"Pohlhausen\", \"InclinedPlate\", \"Sphere\", \"Ground\" and \"Graefe\".");
    }

    for (uint UUID: UUIDs) {
        boundarylayer_model[UUID] = model;
    }
}

void BLConductanceModel::run() {

    run(context->getAllUUIDs());
}

void BLConductanceModel::run(const std::vector<uint> &UUIDs) {

    // leaf shapes of the parent objects for the "Graefe" model, computed once per object
    std::map<uint, LeafShape> leaf_shape_cache;

    std::vector<float> gH_total(UUIDs.size());
    std::vector<float> heat_front(UUIDs.size()), heat_back(UUIDs.size());
    std::vector<float> vapor_front(UUIDs.size()), vapor_back(UUIDs.size());
    std::vector<float> CO2_front(UUIDs.size()), CO2_back(UUIDs.size());

    for (size_t i = 0; i < UUIDs.size(); i++) {
        uint UUID = UUIDs[i];

        float U;
        if (context->doesPrimitiveDataExist(UUID, "wind_speed")) {
            context->getPrimitiveData(UUID, "wind_speed", U);
        } else {
            U = wind_speed_default;
        }

        float Ta;
        if (context->doesPrimitiveDataExist(UUID, "air_temperature")) {
            context->getPrimitiveData(UUID, "air_temperature", Ta);
        } else {
            Ta = air_temperature_default;
        }

        float T;
        if (context->doesPrimitiveDataExist(UUID, "temperature")) {
            context->getPrimitiveData(UUID, "temperature", T);
        } else {
            T = surface_temperature_default;
        }

        // Number of primitive faces - check material first, then primitive data
        uint twosided_flag = context->getPrimitiveTwosidedFlag(UUID, 1);
        char Nsides = (twosided_flag == 0) ? 1 : 2;

        uint model = boundarylayer_model[UUID];

        FaceConductances faces{};
        if (model == 4) {
            faces = calculateGraefeConductances(UUID, U, T, Ta, Nsides, leaf_shape_cache);
        } else {
            float L;
            if (context->doesPrimitiveDataExist(UUID, "object_length")) {
                context->getPrimitiveData(UUID, "object_length", L);
                if (L == 0) {
                    L = sqrt(context->getPrimitiveArea(UUID));
                }
            } else if (context->getPrimitiveParentObjectID(UUID) > 0) {
                uint objID = context->getPrimitiveParentObjectID(UUID);
                L = sqrt(context->getObjectArea(objID));
            } else {
                L = sqrt(context->getPrimitiveArea(UUID));
            }

            if (model == 1) {
                vec3 normal = context->getPrimitiveNormal(UUID);
                normal.normalize();
                vec3 wind_direction;
                bool wind_direction_given = readWindDirection(UUID, wind_direction);
                faces = calculateInclinedPlateConductances(U, L, T, Ta, normal, wind_direction_given ? &wind_direction : nullptr, Nsides);
            } else {
                float gH = calculateBoundaryLayerConductance(model, U, L, Nsides);

                // these models do not distinguish the faces, so the summed conductance is split equally between them
                faces.heat_front = gH / float(Nsides);
                faces.heat_back = (Nsides == 2) ? gH / float(Nsides) : 0.f;
                faces.vapor_front = vapor_to_heat_conductance_ratio * faces.heat_front;
                faces.vapor_back = vapor_to_heat_conductance_ratio * faces.heat_back;
                faces.CO2_front = faces.vapor_front / vapor_to_CO2_conductance_ratio;
                faces.CO2_back = faces.vapor_back / vapor_to_CO2_conductance_ratio;
            }
        }

        gH_total[i] = faces.heat_front + faces.heat_back;
        heat_front[i] = faces.heat_front;
        heat_back[i] = faces.heat_back;
        vapor_front[i] = faces.vapor_front;
        vapor_back[i] = faces.vapor_back;
        CO2_front[i] = faces.CO2_front;
        CO2_back[i] = faces.CO2_back;
    }

    context->setPrimitiveData(UUIDs, "boundarylayer_conductance", gH_total);
    context->setPrimitiveData(UUIDs, "boundarylayer_conductance_front", heat_front);
    context->setPrimitiveData(UUIDs, "boundarylayer_conductance_back", heat_back);
    context->setPrimitiveData(UUIDs, "boundarylayer_conductance_vapor_front", vapor_front);
    context->setPrimitiveData(UUIDs, "boundarylayer_conductance_vapor_back", vapor_back);
    context->setPrimitiveData(UUIDs, "boundarylayer_conductance_CO2_front", CO2_front);
    context->setPrimitiveData(UUIDs, "boundarylayer_conductance_CO2_back", CO2_back);
}

bool BLConductanceModel::readWindDirection(uint UUID, vec3 &wind_direction) const {
    if (!context->doesPrimitiveDataExist(UUID, "wind_direction")) {
        return false;
    }
    if (context->getPrimitiveDataType("wind_direction") != HELIOS_TYPE_VEC3) {
        helios_runtime_error("ERROR (BLConductanceModel::run): Primitive data \"wind_direction\" must be of type vec3, giving the direction the wind blows toward.");
    }
    vec3 wind;
    context->getPrimitiveData(UUID, "wind_direction", wind);
    wind_direction = make_vec3(wind.x, wind.y, 0.f);
    if (!std::isfinite(wind_direction.magnitude()) || wind_direction.magnitude() < 1e-6f) {
        helios_runtime_error("ERROR (BLConductanceModel::run): Primitive data \"wind_direction\" of primitive " + std::to_string(UUID) +
                             " has no horizontal component. It must give the horizontal direction the wind blows toward; remove it if the wind direction is unknown.");
    }
    wind_direction.normalize();
    return true;
}

BLConductanceModel::FaceConductances BLConductanceModel::calculateInclinedPlateConductances(float U, float L, float TL, float Ta, const vec3 &normal, const vec3 *wind_direction, char Nsides) const {

    // Laminar mixed convection on an isothermal plate after Chen et al. (1986), applied to each face of the primitive. The correlations are for the boundary layer on one face; Ramachandran et al. (1987)
    // found the faces above and below an inclined plate to be essentially the same, except for nearly horizontal plates, where buoyancy assists the flow on the face it rises from and opposes it on the other.

    FaceConductances faces{};
    if (L == 0) {
        return faces;
    }

    const float Pr = 0.7f; // air Prandtl number
    const float nu = 1.568e-5f; // air kinematic viscosity (m^2/sec)
    const float alpha = nu / Pr; // air thermal diffusivity (m^2/sec)
    const float molar_density = 41.4f; // air molar density at standard conditions (mol/m^3), so that Nu*alpha/L (m/s) becomes mol/m^2/s

    // The upper face is the one facing up. gamma is the angle of the plate from the vertical, the inclination angle of Chen et al. (1986), which is also the angle of the normal from the horizontal.
    const bool front_is_upper = normal.z >= 0.f;
    const vec3 normal_up = front_is_upper ? normal : -normal;
    const float gamma = asinf(std::min(1.f, normal_up.z));
    const bool nearly_horizontal = gamma >= 75.f * float(M_PI) / 180.f;

    const bool heated = TL >= Ta;
    const float Re = U * L / nu;
    const float Gr = 9.81f * fabs(TL - Ta) * std::pow(L, 3) / (Ta * nu * nu);

    // Nusselt numbers of the face buoyancy rises from (unstable: the upper face of a heated plate, the lower face of a cooled one) and of the other face (stable)
    float Nu_unstable, Nu_stable;

    if (U <= 1e-6f || Re <= 1e-6f) {
        // Without wind the mixed-convection correlations cannot be evaluated, so pure free convection is used (McAdams 1954): the upper and lower surface of a horizontal plate, and an inclined plate with
        // gravity replaced by its component along the plate.
        if (nearly_horizontal) {
            Nu_unstable = 0.54f * std::pow(Gr * sinf(gamma), 0.25f);
            Nu_stable = 0.27f * std::pow(Gr * sinf(gamma), 0.25f);
        } else {
            Nu_unstable = 0.59f * std::pow(Gr * cosf(gamma), 0.25f);
            Nu_stable = Nu_unstable;
        }
    } else {
        // average Nusselt number for forced convection (Churchill and Ozoe 1973); Chen et al. (1986) print the coefficient as 0.399, but the source they cite gives 0.339
        const float F1 = 0.339f * std::cbrt(Pr) * std::pow(1.f + std::pow(0.0468f / Pr, 2.f / 3.f), -0.25f);
        const float Nu_forced = 2.f * F1 * sqrtf(Re);

        // Mixed convection Nu^3 = Nu_forced^3 +- Nu_free^3, written as Nu = Nu_forced (1 +- X^3)^(1/3) with X = Nu_free/Nu_forced. With the buoyancy opposing the flow the correlation holds only while X < 1; beyond
        // that the flow separates and free convection dominates, so the face takes the larger of the opposing mixed value and its pure free convection value.
        auto mixed = [&](float X, bool assisting, float free_convection_factor) {
            if (assisting) {
                return Nu_forced * std::cbrt(1.f + X * X * X);
            }
            return Nu_forced * std::max(std::cbrt(std::max(0.f, 1.f - X * X * X)), free_convection_factor * X);
        };

        if (!nearly_horizontal) {
            const float F2 = 0.75f * sqrtf(Pr) * std::pow(2.5f * (1.f + 2.f * sqrtf(Pr) + 2.f * Pr), -0.25f);
            const float X = 2.f * F2 * std::pow(Gr * cosf(gamma) / (Re * Re), 0.25f) / (3.f * F1);

            // Buoyancy drives the flow up the slope along a heated plate and down it along a cooled one, on both faces, so it assists a wind blowing that way and opposes one blowing the other way. If the
            // wind direction is unknown or across the slope, the buoyancy is taken to assist.
            bool assisting = true;
            if (wind_direction != nullptr) {
                vec3 upslope = make_vec3(0, 0, 1) - normal_up.z * normal_up;
                float wind_upslope = *wind_direction * upslope;
                if (wind_upslope != 0.f) {
                    assisting = (wind_upslope > 0.f) == heated;
                }
            }
            Nu_unstable = mixed(X, assisting, 1.f);
            Nu_stable = Nu_unstable;
        } else {
            const float C = 0.07f * sqrtf(fabs(cosf(gamma)));
            const float F4 = sqrtf(Pr) / (0.25f + 1.6f * sqrtf(Pr)) * std::pow(Pr / 5.f, 0.2f + C);
            const float X = F4 * std::pow(Gr / std::pow(Re, 2.5f), 0.2f) * std::pow(Gr, C) / (6.f * (0.2f + C) * F1);

            // Buoyancy assists the flow on the face it rises from and opposes it on the other, whose free convection is about half as strong (0.27 vs 0.54 above)
            Nu_unstable = mixed(X, true, 1.f);
            Nu_stable = mixed(X, false, 0.5f);
        }
    }

    const float g_unstable = molar_density * alpha / L * Nu_unstable;
    const float g_stable = molar_density * alpha / L * Nu_stable;
    const float g_upper = heated ? g_unstable : g_stable;
    const float g_lower = heated ? g_stable : g_unstable;

    faces.heat_front = front_is_upper ? g_upper : g_lower;
    faces.heat_back = (Nsides == 2) ? (front_is_upper ? g_lower : g_upper) : 0.f;
    faces.vapor_front = vapor_to_heat_conductance_ratio * faces.heat_front;
    faces.vapor_back = vapor_to_heat_conductance_ratio * faces.heat_back;
    faces.CO2_front = faces.vapor_front / vapor_to_CO2_conductance_ratio;
    faces.CO2_back = faces.vapor_back / vapor_to_CO2_conductance_ratio;
    return faces;
}

BLConductanceModel::FaceConductances BLConductanceModel::calculateGraefeConductances(uint UUID, float U, float TL, float Ta, char Nsides, std::map<uint, LeafShape> &leaf_shape_cache) const {

    const std::string primitive_string = "primitive " + std::to_string(UUID);

    if (context->getPrimitiveType(UUID) == PRIMITIVE_TYPE_VOXEL) {
        helios_runtime_error("ERROR (BLConductanceModel::run): The \"Graefe\" boundary-layer conductance model is for flat leaves, but " + primitive_string + " is a voxel. Use a different model for this primitive.");
    }

    // Leaf shape: equivalent ellipse of the parent object (the whole leaf), or of the primitive itself if it has no parent object
    LeafShape shape{};
    uint objID = context->getPrimitiveParentObjectID(UUID);
    if (objID > 0) {
        auto cached = leaf_shape_cache.find(objID);
        if (cached == leaf_shape_cache.end()) {
            cached = leaf_shape_cache.emplace(objID, calculateLeafShape(context->getObjectPrimitiveUUIDs(objID), context->getObjectArea(objID), UUID)).first;
        }
        shape = cached->second;
    } else {
        shape = calculateLeafShape({UUID}, context->getPrimitiveArea(UUID), UUID);
    }

    bool length_given = context->doesPrimitiveDataExist(UUID, "leaf_length");
    bool width_given = context->doesPrimitiveDataExist(UUID, "leaf_width");
    if (length_given != width_given) {
        helios_runtime_error("ERROR (BLConductanceModel::run): Only one of primitive data \"leaf_length\" and \"leaf_width\" was given for " + primitive_string +
                             ". Give both to override the leaf dimensions of the \"Graefe\" model, or neither to compute them from the geometry.");
    }
    if (length_given) {
        if (context->getPrimitiveDataType("leaf_length") != HELIOS_TYPE_FLOAT || context->getPrimitiveDataType("leaf_width") != HELIOS_TYPE_FLOAT) {
            helios_runtime_error("ERROR (BLConductanceModel::run): Primitive data \"leaf_length\" and \"leaf_width\" must be of type float.");
        }
        float leaf_length, leaf_width;
        context->getPrimitiveData(UUID, "leaf_length", leaf_length);
        context->getPrimitiveData(UUID, "leaf_width", leaf_width);
        if (!std::isfinite(leaf_length) || !std::isfinite(leaf_width) || leaf_width <= 0.f || leaf_length < leaf_width) {
            helios_runtime_error("ERROR (BLConductanceModel::run): Primitive data \"leaf_length\" (" + std::to_string(leaf_length) + ") and \"leaf_width\" (" + std::to_string(leaf_width) + ") of " + primitive_string +
                                 " are invalid. Both must be positive, and the length is the longer of the two dimensions.");
        }
        shape.length = leaf_length;
        shape.width = leaf_width;
    }

    // Orientation. The model's upper side is the one facing up, which is the front face if the normal points up.
    vec3 normal = context->getPrimitiveNormal(UUID);
    normal.normalize();
    const bool front_is_upper = normal.z >= 0.f;
    if (!front_is_upper) {
        normal = -normal;
    }
    graefe::Vec3 normal_up = toGraefeVec3(normal);

    // long axis of the leaf projected onto the plane of this primitive, which can differ from the mean plane of a curved leaf
    vec3 long_axis = shape.long_axis - (shape.long_axis * normal) * normal;
    if (long_axis.magnitude() < 1e-3f) {
        helios_runtime_error("ERROR (BLConductanceModel::run): The long axis of the leaf containing " + primitive_string +
                             " is perpendicular to the primitive, so the leaf orientation needed by the \"Graefe\" model is undefined. This happens for strongly folded or rolled leaves; use a different model for these primitives.");
    }
    long_axis.normalize();
    vec3 short_axis = cross(normal, long_axis);
    short_axis.normalize();

    // Wind direction (optional): the horizontal direction the wind blows toward
    graefe::Vec3 wind_direction;
    const graefe::Vec3 *wind_direction_ptr = nullptr;
    vec3 wind_horizontal;
    if (readWindDirection(UUID, wind_horizontal)) {
        wind_direction = toGraefeVec3(wind_horizontal);
        wind_direction_ptr = &wind_direction;
    }

    float turbulence_intensity = turbulence_intensity_default;
    if (context->doesPrimitiveDataExist(UUID, "turbulence_intensity") && context->getPrimitiveDataType("turbulence_intensity") == HELIOS_TYPE_FLOAT) {
        context->getPrimitiveData(UUID, "turbulence_intensity", turbulence_intensity);
        if (!std::isfinite(turbulence_intensity) || turbulence_intensity < 0.f) {
            helios_runtime_error("ERROR (BLConductanceModel::run): Primitive data \"turbulence_intensity\" of " + primitive_string + " is " + std::to_string(turbulence_intensity) + ", but it must be a non-negative percentage.");
        }
    }

    float pressure = air_pressure_default;
    if (context->doesPrimitiveDataExist(UUID, "air_pressure") && context->getPrimitiveDataType("air_pressure") == HELIOS_TYPE_FLOAT) {
        context->getPrimitiveData(UUID, "air_pressure", pressure);
        if (!std::isfinite(pressure) || pressure <= 0.f) {
            helios_runtime_error("ERROR (BLConductanceModel::run): Primitive data \"air_pressure\" of " + primitive_string + " is " + std::to_string(pressure) + ", but it must be positive (Pa).");
        }
    }

    graefe::ModelOptions options;
    options.turbulence_intensity_percent = turbulence_intensity;

    graefe::Vec3 short_axis_graefe = toGraefeVec3(short_axis);
    graefe::LeafGeometry geometry = graefe::leafGeometry(shape.length, shape.width, normal_up, short_axis_graefe);
    graefe::ConductanceResult result = graefe::leafConductances(TL, Ta, U, pressure, geometry, normal_up, short_axis_graefe, wind_direction_ptr, options);

    // conductances in m/s times the molar density of air give mol/m^2/s
    const double molar_density = result.molar_density;
    FaceConductances faces{};
    faces.heat_front = float(molar_density * (front_is_upper ? result.gb_U : result.gb_L));
    faces.heat_back = float(molar_density * (front_is_upper ? result.gb_L : result.gb_U));
    faces.vapor_front = float(molar_density * (front_is_upper ? result.gv_U : result.gv_L));
    faces.vapor_back = float(molar_density * (front_is_upper ? result.gv_L : result.gv_U));
    faces.CO2_front = float(molar_density * (front_is_upper ? result.gc_U : result.gc_L));
    faces.CO2_back = float(molar_density * (front_is_upper ? result.gc_L : result.gc_U));

    if (Nsides == 1) { // only the front face exchanges heat and mass
        faces.heat_back = 0.f;
        faces.vapor_back = 0.f;
        faces.CO2_back = 0.f;
    }

    return faces;
}

BLConductanceModel::LeafShape BLConductanceModel::calculateLeafShape(const std::vector<uint> &UUIDs, float area, uint UUID_for_messages) const {

    const std::string error_prefix = "ERROR (BLConductanceModel::run): Could not determine the leaf shape for the \"Graefe\" model of primitive " + std::to_string(UUID_for_messages);

    if (!(area > 0.f)) {
        helios_runtime_error(error_prefix + " because its leaf area is zero.");
    }

    // triangles of the leaf
    std::vector<std::array<vec3, 3>> triangles;
    for (uint p: UUIDs) {
        std::vector<vec3> vertices = context->getPrimitiveVertices(p);
        for (size_t v = 1; v + 1 < vertices.size(); v++) {
            triangles.push_back({vertices[0], vertices[v], vertices[v + 1]});
        }
    }

    // mean plane: area-weighted normal, with each triangle oriented into the hemisphere of the first so that inconsistent winding does not cancel
    vec3 reference_normal;
    double normal_sum_x = 0, normal_sum_y = 0, normal_sum_z = 0;
    for (const auto &triangle: triangles) {
        vec3 area_normal = cross(triangle[1] - triangle[0], triangle[2] - triangle[0]);
        if (reference_normal.magnitude() == 0.f) {
            reference_normal = area_normal;
        }
        if (area_normal * reference_normal < 0.f) {
            area_normal = -area_normal;
        }
        normal_sum_x += area_normal.x;
        normal_sum_y += area_normal.y;
        normal_sum_z += area_normal.z;
    }
    double normal_magnitude = std::sqrt(normal_sum_x * normal_sum_x + normal_sum_y * normal_sum_y + normal_sum_z * normal_sum_z);
    if (normal_magnitude == 0.0) {
        helios_runtime_error(error_prefix + " because its geometry has no area.");
    }
    vec3 mean_normal(float(normal_sum_x / normal_magnitude), float(normal_sum_y / normal_magnitude), float(normal_sum_z / normal_magnitude));

    // orthonormal basis of the mean plane
    vec3 helper = (std::fabs(mean_normal.z) < 0.9f) ? make_vec3(0, 0, 1) : make_vec3(1, 0, 0);
    vec3 e1 = cross(helper, mean_normal);
    e1.normalize();
    vec3 e2 = cross(mean_normal, e1);

    // second moments of the area projected onto the mean plane (exact for triangles)
    double A = 0, Sx = 0, Sy = 0, Sxx = 0, Syy = 0, Sxy = 0;
    for (const auto &triangle: triangles) {
        double x[3], y[3];
        for (int k = 0; k < 3; k++) {
            x[k] = triangle[k] * e1;
            y[k] = triangle[k] * e2;
        }
        double triangle_area = 0.5 * std::fabs((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]));
        A += triangle_area;
        Sx += triangle_area * (x[0] + x[1] + x[2]) / 3.0;
        Sy += triangle_area * (y[0] + y[1] + y[2]) / 3.0;
        Sxx += triangle_area / 6.0 * (x[0] * x[0] + x[1] * x[1] + x[2] * x[2] + x[0] * x[1] + x[0] * x[2] + x[1] * x[2]);
        Syy += triangle_area / 6.0 * (y[0] * y[0] + y[1] * y[1] + y[2] * y[2] + y[0] * y[1] + y[0] * y[2] + y[1] * y[2]);
        Sxy += triangle_area / 12.0 * (2.0 * (x[0] * y[0] + x[1] * y[1] + x[2] * y[2]) + x[0] * y[1] + x[1] * y[0] + x[0] * y[2] + x[2] * y[0] + x[1] * y[2] + x[2] * y[1]);
    }
    if (A == 0.0) {
        helios_runtime_error(error_prefix + " because its geometry has no area.");
    }
    double cx = Sx / A, cy = Sy / A;
    double Cxx = Sxx / A - cx * cx;
    double Cyy = Syy / A - cy * cy;
    double Cxy = Sxy / A - cx * cy;

    // principal axes: the long axis is the eigenvector of the larger eigenvalue; for an ellipse or a rectangle the length ratio is the square root of the eigenvalue ratio
    double half_trace = 0.5 * (Cxx + Cyy);
    double discriminant = std::sqrt(0.25 * (Cxx - Cyy) * (Cxx - Cyy) + Cxy * Cxy);
    double lambda_major = half_trace + discriminant;
    double lambda_minor = half_trace - discriminant;
    if (!(lambda_minor > 1e-12 * lambda_major)) {
        helios_runtime_error(error_prefix + " because its geometry is degenerate (all of its area lies along a line).");
    }
    double axis_x, axis_y;
    if (std::fabs(Cxy) > 1e-12 * lambda_major) {
        axis_x = Cxy;
        axis_y = lambda_major - Cxx;
    } else if (Cxx >= Cyy) {
        axis_x = 1;
        axis_y = 0;
    } else {
        axis_x = 0;
        axis_y = 1;
    }

    LeafShape shape{};
    shape.long_axis = float(axis_x) * e1 + float(axis_y) * e2;
    shape.long_axis.normalize();

    // equivalent ellipse with the leaf's aspect ratio and area
    double aspect_ratio = std::sqrt(lambda_major / lambda_minor);
    shape.length = std::sqrt(4.0 * double(area) * aspect_ratio / M_PI);
    shape.width = shape.length / aspect_ratio;

    return shape;
}

float BLConductanceModel::calculateBoundaryLayerConductance(uint gH_model, float U, float L, char Nsides) {

    assert(gH_model == 0 || gH_model == 2 || gH_model == 3); // "InclinedPlate" and "Graefe" give per-face conductances, see calculateInclinedPlateConductances() and calculateGraefeConductances()

    float gH = 0;

    if (L == 0) {
        return 0;
    }

    if (gH_model == 0) { // Pohlhausen equation
        // Pohlhausen's (1921) solution for laminar forced convection along an isothermal flat plate parallel to the flow, infinitely wide in the cross-stream direction and "L" long in the streamwise direction,
        // Nu = 0.664 Re^(1/2) Pr^(1/3) for one face. With the properties of air near 20 C and standard pressure this is the leaf boundary-layer conductance 0.135 sqrt(U/L) mol/m^2/s per face of Campbell and Norman (1998).

        gH = 0.135f * sqrt(U / L) * float(Nsides);

    } else if (gH_model == 2) { // Sphere

        // Laminar flow around a sphere (L is sphere diameter). From Eq. 4 of Smart and Sinclair (1976).

        float Pr = 0.7f; // air Prandtl number
        float nu = 1.568e-5; // air viscosity (m^2/sec)
        float ka = 0.024; // air thermal conductivity (W/m-K)
        float cp = 29.25; // air heat capacity (J/mol-K)

        float Re = U * L / nu;

        gH = (ka / cp) / L * (2.0 + 0.6 * sqrtf(Re) * pow(Pr, 1.f / 3.f));

    } else if (gH_model == 3) { // Ground
        // From Eq. A17 of Kustas and Norman (1999). For the soil-air interface.

        gH = 0.004f + 0.012f * U; // units in (m^3 air)/(m^2-sec.)

        // assuming standard temperature and pressure, (m^3 air)*(1.2041 kg/m^3)/(0.02897 kg/mol) = 41.56 (mol air)/(m^2-sec.)

        gH = gH * 41.56f;
    }

    return gH;
}
