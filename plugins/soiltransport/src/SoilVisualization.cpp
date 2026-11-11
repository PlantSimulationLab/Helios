/** \file "SoilVisualization.cpp" Soil-domain visualization geometry for the Visualizer.

    Copyright (C) 2016-2026 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "SoilTransportModel.h"
#include "global.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace helios;

namespace {
    //! Soil-moisture colormap: t in [0,1] (0 = dry, 1 = wet) -> RGB (tan -> olive -> blue).
    RGBcolor soilColormap(float t) {
        t = std::min(1.f, std::max(0.f, t));
        const RGBcolor dry = make_RGBcolor(0.80f, 0.66f, 0.46f); // dry tan
        const RGBcolor mid = make_RGBcolor(0.40f, 0.55f, 0.45f); // damp olive
        const RGBcolor wet = make_RGBcolor(0.10f, 0.25f, 0.60f); // wet blue
        RGBcolor a, b;
        float u;
        if (t < 0.5f) {
            a = dry;
            b = mid;
            u = t / 0.5f;
        } else {
            a = mid;
            b = wet;
            u = (t - 0.5f) / 0.5f;
        }
        return make_RGBcolor(a.r + (b.r - a.r) * u, a.g + (b.g - a.g) * u, a.b + (b.b - a.b) * u);
    }

    //! Axis-aligned world-space sub-box.
    struct WBox {
        float x0, x1, y0, y1, z0, z1;
    };
} // namespace

// Convenience overloads delegate to the (min, max, region) form.
std::vector<uint> SoilTransportModel::addSoilVisualization(SoilColorMode mode) const {
    return addSoilVisualization(mode, make_vec3(0, 0, 0), make_vec3(1, 1, 1), SoilRegion::KEEP);
}

std::vector<uint> SoilTransportModel::addSoilVisualization(SoilColorMode mode, const vec3 &keep_fraction) const {
    return addSoilVisualization(mode, make_vec3(0, 0, 0), keep_fraction, SoilRegion::KEEP);
}

std::vector<uint> SoilTransportModel::addSoilVisualization(SoilColorMode mode, const vec3 &fraction_min, const vec3 &fraction_max, SoilRegion region) const {
    std::vector<uint> UUIDs;
    if (context_ == nullptr) {
        helios_runtime_error("ERROR (SoilTransportModel::addSoilVisualization): the model was constructed without a Context, so there is nowhere to add the visualization.");
    }
    requireDomainSetUp("addSoilVisualization");

    auto clamp01 = [](float v) { return std::min(1.f, std::max(0.f, v)); };
    vec3 fmin = make_vec3(clamp01(fraction_min.x), clamp01(fraction_min.y), clamp01(fraction_min.z));
    vec3 fmax = make_vec3(clamp01(fraction_max.x), clamp01(fraction_max.y), clamp01(fraction_max.z));
    if (fmax.x < fmin.x)
        std::swap(fmin.x, fmax.x);
    if (fmax.y < fmin.y)
        std::swap(fmin.y, fmax.y);
    if (fmax.z < fmin.z)
        std::swap(fmin.z, fmax.z);

    // Domain world box. Fractions are measured from the origin along the extent;
    // origin.z is the surface, so a larger z-fraction is deeper.
    const float ox = origin_.x, oy = origin_.y, oz = origin_.z;
    const float ex = extent_.x, ey = extent_.y, ez = extent_.z;
    const float cx = ex / float(nx_), cy = ey / float(ny_), cz = ez / float(nz_);
    const float X0 = ox, X1 = ox + ex, Y0 = oy, Y1 = oy + ey, Z0 = oz - ez, Z1 = oz;

    // World box of the selected fractional sub-region.
    const float ax0 = ox + fmin.x * ex, ax1 = ox + fmax.x * ex;
    const float ay0 = oy + fmin.y * ey, ay1 = oy + fmax.y * ey;
    const float az0 = oz - fmax.z * ez, az1 = oz - fmin.z * ez; // az0 (deep) <= az1 (top)

    // ---- Build the list of solid sub-boxes to render ----
    std::vector<WBox> boxes;
    auto nondegen = [](float lo, float hi) { return (hi - lo) > 1e-6f; };
    if (region == SoilRegion::KEEP) {
        boxes.push_back({ax0, ax1, ay0, ay1, az0, az1});
    } else {
        // Cut the corner-anchored box [a*] out of the domain; the remainder
        // partitions (guillotine) into up to three sub-boxes. Internal seams end
        // up inside the solid (occluded); the cut walls are exposed.
        auto outside = [](float rlo, float rhi, float Dlo, float Dhi, float &olo, float &ohi) {
            if (rhi >= Dhi - 1e-6f) {
                olo = Dlo;
                ohi = rlo;
            } // removed anchored at the high end
            else {
                olo = rhi;
                ohi = Dhi;
            } // anchored at the low end
        };
        float oxlo, oxhi, oylo, oyhi, ozlo, ozhi;
        outside(ax0, ax1, X0, X1, oxlo, oxhi);
        outside(ay0, ay1, Y0, Y1, oylo, oyhi);
        outside(az0, az1, Z0, Z1, ozlo, ozhi);
        if (nondegen(oxlo, oxhi))
            boxes.push_back({oxlo, oxhi, Y0, Y1, Z0, Z1});
        if (nondegen(ax0, ax1) && nondegen(oylo, oyhi))
            boxes.push_back({ax0, ax1, oylo, oyhi, Z0, Z1});
        if (nondegen(ax0, ax1) && nondegen(ay0, ay1) && nondegen(ozlo, ozhi))
            boxes.push_back({ax0, ax1, ay0, ay1, ozlo, ozhi});
    }

    const bool textured = (mode == SoilColorMode::TEXTURE);
    const bool head_mode = (mode == SoilColorMode::PRESSURE_HEAD);
    const std::string tex = "plugins/soiltransport/assets/textures/dirt.jpg";
    const RGBcolor placeholder = make_RGBcolor(0.5f, 0.5f, 0.5f);
    const float hp = 0.5f * float(M_PI);
    const float min_cell = std::min(cx, std::min(cy, cz));
    const float nudge = 0.25f * min_cell;

    std::vector<float> values; // field value per (non-textured) face patch, parallel to UUIDs

    // Build the outer faces of one solid sub-box and (for field modes) record the
    // field sampled at each face patch, nudged toward this box's center.
    auto buildBox = [&](const WBox &b) {
        const float sxc = 0.5f * (b.x0 + b.x1), syc = 0.5f * (b.y0 + b.y1), szc = 0.5f * (b.z0 + b.z1);
        const vec3 bcenter = make_vec3(sxc, syc, szc);
        auto nsub = [](float span, float cell) { return std::max(1, (int) std::lround(span / cell)); };
        const int nsx = nsub(b.x1 - b.x0, cx), nsy = nsub(b.y1 - b.y0, cy), nsz = nsub(b.z1 - b.z0, cz);
        const bool thin_x = (b.x1 - b.x0) < 1e-6f, thin_y = (b.y1 - b.y0) < 1e-6f, thin_z = (b.z1 - b.z0) < 1e-6f;
        const size_t start = UUIDs.size();

        auto addFace = [&](int normal_axis, float coord, const SphericalCoord &rot) {
            vec2 size;
            vec3 ctr;
            int2 sub;
            if (normal_axis == 0) {
                size = make_vec2(b.y1 - b.y0, b.z1 - b.z0);
                ctr = make_vec3(coord, syc, szc);
                sub = make_int2(nsy, nsz);
            } else if (normal_axis == 1) {
                size = make_vec2(b.x1 - b.x0, b.z1 - b.z0);
                ctr = make_vec3(sxc, coord, szc);
                sub = make_int2(nsx, nsz);
            } else {
                size = make_vec2(b.x1 - b.x0, b.y1 - b.y0);
                ctr = make_vec3(sxc, syc, coord);
                sub = make_int2(nsx, nsy);
            }
            const size_t face_start = UUIDs.size();
            if (textured) {
                // addPatch's texture overload renders the image; addTile's renders black.
                UUIDs.push_back(context_->addPatch(ctr, size, rot, tex.c_str()));
            } else {
                std::vector<uint> f = context_->addTile(ctr, size, rot, sub, placeholder);
                UUIDs.insert(UUIDs.end(), f.begin(), f.end());
            }
            // Faces off the domain boundary are virtual cross-sections exposed by the slice/cutaway.
            const float lo = (normal_axis == 0) ? X0 : (normal_axis == 1) ? Y0 : Z0;
            const float hi = (normal_axis == 0) ? X1 : (normal_axis == 1) ? Y1 : Z1;
            const bool cut = std::fabs(coord - lo) > 1e-5f && std::fabs(coord - hi) > 1e-5f;
            for (size_t n = face_start; n < UUIDs.size(); n++) {
                context_->setPrimitiveData(UUIDs[n], "soil_cut_face", cut ? 1 : 0);
            }
        };

        if (thin_x || thin_y || thin_z) { // collapsed -> single cross-section plane
            if (thin_x)
                addFace(0, b.x0, make_SphericalCoord(hp, hp));
            else if (thin_y)
                addFace(1, b.y0, make_SphericalCoord(hp, 0));
            else
                addFace(2, b.z1, make_SphericalCoord(0, 0));
        } else { // six outer faces
            addFace(2, b.z1, make_SphericalCoord(0, 0)); // top
            addFace(2, b.z0, make_SphericalCoord(0, 0)); // bottom
            addFace(1, b.y1, make_SphericalCoord(hp, 0)); // +y
            addFace(1, b.y0, make_SphericalCoord(hp, float(M_PI))); // -y
            addFace(0, b.x1, make_SphericalCoord(hp, hp)); // +x
            addFace(0, b.x0, make_SphericalCoord(hp, 1.5f * float(M_PI))); // -x
        }

        if (!textured) {
            for (size_t n = start; n < UUIDs.size(); n++) {
                std::vector<vec3> verts = context_->getPrimitiveVertices(UUIDs[n]);
                vec3 c = make_vec3(0, 0, 0);
                for (const vec3 &v: verts)
                    c = c + v;
                c = c / float(verts.size());
                vec3 dir = bcenter - c;
                float dlen = dir.magnitude();
                if (dlen > 1e-9f)
                    c = c + dir * (nudge / dlen);
                values.push_back(head_mode ? getPressureHead(c) : getWaterContent(c));
                // Stash the sample point so updateSoilVisualization() can re-sample the
                // field on this same face without rebuilding the geometry.
                context_->setPrimitiveData(UUIDs[n], "soil_sample_pos", c);
            }
        }
    };

    for (const WBox &b: boxes)
        buildBox(b);

    if (textured)
        return UUIDs; // faces already carry the dirt texture

    // Color every face on one shared range so the (multi-box) cutaway is consistent.
    float vmin = 1e30f, vmax = -1e30f;
    for (float v: values) {
        vmin = std::min(vmin, v);
        vmax = std::max(vmax, v);
    }
    if (vmax - vmin < 1e-12f)
        vmax = vmin + 1.f;
    for (size_t n = 0; n < UUIDs.size() && n < values.size(); n++) {
        float t = (values[n] - vmin) / (vmax - vmin);
        context_->setPrimitiveColor(UUIDs[n], soilColormap(t));
        context_->setPrimitiveData(UUIDs[n], head_mode ? "pressure_head" : "water_content", values[n]);
    }

    if (message_flag_) {
        std::cout << "SoilTransportModel: soil visualization (" << (head_mode ? "pressure head" : "water content") << ") range [" << vmin << ", " << vmax << "], " << UUIDs.size() << " primitives." << std::endl;
    }
    return UUIDs;
}

void SoilTransportModel::updateSoilVisualization(const std::vector<uint> &UUIDs, SoilColorMode mode) const {
    // Refresh the field data on an existing soil visualization (built earlier by
    // addSoilVisualization) WITHOUT rebuilding any geometry. The face layout is fixed;
    // only the sampled field changes, so for an animation this avoids deleting and
    // recreating thousands of primitives (and the GPU buffer churn) every frame. Each
    // face stores its sample point as "soil_sample_pos" primitive data; re-sample there
    // and rewrite the "pressure_head"/"water_content" data the visualizer colors by.
    if (context_ == nullptr) {
        helios_runtime_error("ERROR (SoilTransportModel::updateSoilVisualization): the model was constructed without a Context, so there is no visualization to update.");
    }
    if (mode == SoilColorMode::TEXTURE)
        return; // a textured visualization has no field data
    const bool head_mode = (mode == SoilColorMode::PRESSURE_HEAD);
    const char *dname = head_mode ? "pressure_head" : "water_content";
    for (uint u: UUIDs) {
        if (!context_->doesPrimitiveDataExist(u, "soil_sample_pos"))
            continue;
        vec3 c;
        context_->getPrimitiveData(u, "soil_sample_pos", c);
        context_->setPrimitiveData(u, dname, head_mode ? getPressureHead(c) : getWaterContent(c));
    }
}
