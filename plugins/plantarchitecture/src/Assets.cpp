/** \file "Assets.cpp" Function definitions for plant organ prototypes plant architecture plug-in.

    Copyright (C) 2016-2025 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#include "PlantArchitecture.h"

using namespace helios;

uint GenericLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {

    // If OBJ model file is specified, load it and return the object ID
    if (!prototype_parameters->OBJ_model_file.empty()) {
        if (!std::filesystem::exists(prototype_parameters->OBJ_model_file)) {
            helios_runtime_error("ERROR (PlantArchitecture): Leaf prototype OBJ file " + prototype_parameters->OBJ_model_file + " does not exist.");
        }
        return context_ptr->addPolymeshObject(context_ptr->loadOBJ(prototype_parameters->OBJ_model_file.c_str(), prototype_parameters->leaf_offset, 0, nullrotation, RGB::black, "ZUP", true));
    }

    std::string leaf_texture;
    if (prototype_parameters->leaf_texture_file.empty()) {
        helios_runtime_error("ERROR (PlantArchitecture): Leaf prototype texture file was not specified.");
    } else if (prototype_parameters->leaf_texture_file.size() == 1) {
        leaf_texture = prototype_parameters->leaf_texture_file.begin()->second;
    } else if (prototype_parameters->leaf_texture_file.find(compound_leaf_index) == prototype_parameters->leaf_texture_file.end()) {
        helios_runtime_error("ERROR (PlantArchitecture): Leaf prototype texture file for compound leaf index " + std::to_string(compound_leaf_index) + " was not found.");
    } else {
        leaf_texture = prototype_parameters->leaf_texture_file[compound_leaf_index];
    }

    // -- main leaf generation code -- //

    std::vector<uint> UUIDs;

    uint Nx = prototype_parameters->subdivisions; // number of leaf subdivisions in the x-direction (longitudinal)
    uint Ny = ceil(prototype_parameters->leaf_aspect_ratio.val() * float(Nx)); // number of leaf subdivisions in the y-direction (lateral)

    if (Ny % 2 != 0) { // Ny must be even
        Ny = Ny + 1;
    }

    const float dx = 1.f / float(Nx); // length of leaf subdivision in the x-direction
    const float dy = prototype_parameters->leaf_aspect_ratio.val() / float(Ny); // length of leaf subdivision in the y-direction

    std::vector<std::vector<vec3>> vertices;
    resize_vector(vertices, Nx + 1, Ny + 1);

    for (int j = 0; j <= Ny; j++) {
        float dtheta = 0;
        for (int i = 0; i <= Nx; i++) {

            const float x = float(i) * dx; // x-coordinate of leaf subdivision
            const float y = float(j) * dy - 0.5f * prototype_parameters->leaf_aspect_ratio.val(); // y-coordinate of leaf subdivision

            // midrib leaf folding
            const float y_fold = cosf(0.5f * prototype_parameters->midrib_fold_fraction.val() * M_PI) * y;
            const float z_fold = sinf(0.5f * prototype_parameters->midrib_fold_fraction.val() * M_PI) * fabs(y);

            // x-curvature
            float z_xcurve = prototype_parameters->longitudinal_curvature.val() * powf(x, 4);

            // y-curvature
            float z_ycurve = prototype_parameters->lateral_curvature.val() * powf(y / prototype_parameters->leaf_aspect_ratio.val(), 4);

            // petiole roll
            float z_petiole = 0;
            if (prototype_parameters->petiole_roll.val() != 0.0f) {
                z_petiole = fmin(0.1f, prototype_parameters->petiole_roll.val() * powf(7.f * y / prototype_parameters->leaf_aspect_ratio.val(), 4) * exp(-70.f * (x))) -
                            0.01 * prototype_parameters->petiole_roll.val() / fabs(prototype_parameters->petiole_roll.val());
            }

            // vertical displacement for leaf wave at each of the four subdivision vertices
            float z_wave = 0;
            if (prototype_parameters->wave_period.val() > 0.0f && prototype_parameters->wave_amplitude.val() > 0.0f) {
                z_wave = (2.f * fabs(y) * prototype_parameters->wave_amplitude.val() * sinf((x + 0.5f * float(j >= 0.5 * Ny)) * M_PI / prototype_parameters->wave_period.val()));
            }

            vertices.at(j).at(i) = make_vec3(x, y_fold, z_fold + z_ycurve + z_wave + z_petiole);

            if (prototype_parameters->longitudinal_curvature.val() != 0.0f && i > 0) {
                dtheta -= atan(4.f * prototype_parameters->longitudinal_curvature.val() * powf(x, 3) * dx);
                vertices.at(j).at(i) = rotatePointAboutLine(vertices.at(j).at(i), nullorigin, make_vec3(0, 1, 0), dtheta);
            }

            if (prototype_parameters->leaf_buckle_angle.val() > 0) {
                const float xf = prototype_parameters->leaf_buckle_length.val();
                if (x <= prototype_parameters->leaf_buckle_length.val() && x + dx > prototype_parameters->leaf_buckle_length.val()) {
                    vertices.at(j).at(i) = rotatePointAboutLine(vertices.at(j).at(i), make_vec3(xf, 0, 0), make_vec3(0, 1, 0), 0.5f * deg2rad(prototype_parameters->leaf_buckle_angle.val()));
                } else if (x + dx > prototype_parameters->leaf_buckle_length.val()) {
                    vertices.at(j).at(i) = rotatePointAboutLine(vertices.at(j).at(i), make_vec3(xf, 0, 0), make_vec3(0, 1, 0), deg2rad(prototype_parameters->leaf_buckle_angle.val()));
                }
            }
        }
    }

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {

            const float x = float(i) * dx;
            const float y = float(j) * dy - 0.5f * prototype_parameters->leaf_aspect_ratio.val();
            vec2 uv0(x, (y + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());
            vec2 uv1(x + dx, (y + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());
            vec2 uv2(x + dx, (y + dy + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());
            vec2 uv3(x, (y + dy + 0.5f * prototype_parameters->leaf_aspect_ratio.val()) / prototype_parameters->leaf_aspect_ratio.val());

            vec3 v0 = vertices.at(j).at(i);
            vec3 v1 = vertices.at(j).at(i + 1);
            vec3 v2 = vertices.at(j + 1).at(i + 1);
            vec3 v3 = vertices.at(j + 1).at(i);

            // Add triangle 1 and check if it has effective area (including texture transparency)
            uint uuid1 = context_ptr->addTriangle(v0, v1, v2, leaf_texture.c_str(), uv0, uv1, uv2);
            if (context_ptr->getPrimitiveArea(uuid1) > 0) {
                UUIDs.push_back(uuid1);
            } else {
                context_ptr->deletePrimitive(uuid1);
            }

            // Add triangle 2 and check if it has effective area (including texture transparency)
            uint uuid2 = context_ptr->addTriangle(v0, v2, v3, leaf_texture.c_str(), uv0, uv2, uv3);
            if (context_ptr->getPrimitiveArea(uuid2) > 0) {
                UUIDs.push_back(uuid2);
            } else {
                context_ptr->deletePrimitive(uuid2);
            }
        }
    }

    context_ptr->translatePrimitive(UUIDs, prototype_parameters->leaf_offset);

    if (prototype_parameters->build_petiolule) {
        std::vector<uint> UUIDs_petiolule = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/PetiolulePrototype.obj", make_vec3(0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
        context_ptr->translatePrimitive(UUIDs, make_vec3(0.07, 0, 0.005));
        UUIDs.insert(UUIDs.end(), UUIDs_petiolule.begin(), UUIDs_petiolule.end());
    }

    prototype_parameters->leaf_aspect_ratio.resample();
    prototype_parameters->midrib_fold_fraction.resample();
    prototype_parameters->longitudinal_curvature.resample();
    prototype_parameters->lateral_curvature.resample();
    prototype_parameters->petiole_roll.resample();
    prototype_parameters->wave_period.resample();
    prototype_parameters->wave_amplitude.resample();
    prototype_parameters->leaf_buckle_length.resample();
    prototype_parameters->leaf_buckle_angle.resample();

    return context_ptr->addPolymeshObject(UUIDs);
}

uint GeneralSphericalFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    return context_ptr->addSphereObject(5, make_vec3(0.5f, 0, 0), 0.5f, RGB::red);
}

uint AlmondFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/AlmondHull.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint AlmondFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/AlmondFlower.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void AlmondPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (phytomer->internode_length_max < 0.01) { // spurs
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->scaleLeafPrototypeScale(0.8);
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->parent_shoot_ptr->shoot_parameters.max_nodes_per_season = 7;
    }

    // blind nodes
    //    if( shoot_node_index<3 ){
    //        phytomer->setVegetativeBudState( BUD_DEAD );
    //        phytomer->setFloralBudState( BUD_DEAD );
    //    }
}

void AlmondPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
}

uint AppleFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/AppleFruit.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint AppleFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/AlmondFlower.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void ApplePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (phytomer->internode_length_max < 0.01) { // spurs
        phytomer->setInternodeMaxRadius(0.005);
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->scaleLeafPrototypeScale(0.8);
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->parent_shoot_ptr->shoot_parameters.max_nodes_per_season = 6;
    }
}

void ApplePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
}

uint AsparagusLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {

    float curve_magnitude = context_ptr->randu(0.f, 0.2f);

    std::vector<vec3> nodes;
    nodes.push_back(make_vec3(0, 0, 0));
    nodes.push_back(make_vec3(context_ptr->randu(0.4f, 0.7f), 0, -0.25f * curve_magnitude));
    nodes.push_back(make_vec3(0.95, 0, -0.9f * curve_magnitude));
    nodes.push_back(make_vec3(1, 0, -curve_magnitude));

    std::vector<float> radius;
    radius.push_back(0.015);
    radius.push_back(0.015);
    radius.push_back(0.015);
    radius.push_back(0.0);

    std::vector<RGBcolor> colors;
    colors.push_back(RGB::forestgreen);
    colors.push_back(RGB::forestgreen);
    colors.push_back(RGB::forestgreen);
    colors.push_back(RGB::forestgreen);

    uint objID = context_ptr->addTubeObject(8, nodes, radius, colors);
    context_ptr->rotateObject(objID, context_ptr->randu(0, 2.f * M_PI), "x");
    return objID;
}

void AsparagusPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // blind nodes
    if (shoot_node_index <= 2) {
        phytomer->scaleLeafPrototypeScale(0.6);
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
}

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs;
    UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanLeaf_unifoliate.obj", true);

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs;
    if (compound_leaf_index == 0) {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanLeaf_tip.obj", true);
    } else if (compound_leaf_index < 0) {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanLeaf_left.obj", true);
    } else {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanLeaf_right.obj", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanPod.obj", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint BeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs;
    if (flower_is_open) {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanFlower_open_white.obj", true);
    } else {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanFlower_closed_white.obj", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void BeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index > 5 || phytomer->rank > 1) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    } else {
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.6 + 0.4 * plant_age / 8.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    if (phytomer->rank == 0) {
        float inode_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 10.f);
        phytomer->scaleInternodeMaxLength(inode_scale);
    }
}

uint BindweedFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BindweedFlower.obj", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CapsicumFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::string OBJ_file;
    if (context_ptr->randn() < 0.4) {
        OBJ_file = "plugins/plantarchitecture/assets/obj/CapsicumFruit_green.obj";
    } else {
        OBJ_file = "plugins/plantarchitecture/assets/obj/CapsicumFruit_red.obj";
    }

    std::vector<uint> UUIDs = context_ptr->loadOBJ(OBJ_file.c_str(), make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

void CapsicumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index < 6 && phytomer->rank == 0) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->removeLeaf();
    }

    if (phytomer->rank >= 2) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = std::min(1.f, 0.6f + 0.4f * shoot_node_index / 5.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    if (phytomer->rank == 0) {
        float inode_scale = std::min(1.f, 0.05f + 0.95f * plant_age / 15.f);
        phytomer->scaleInternodeMaxLength(inode_scale);
    }
}

uint CheeseweedLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CheeseweedLeaf.obj", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CowpeaLeaf_unifoliate.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {
    std::vector<uint> UUIDs;
    if (compound_leaf_index < 0) {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CowpeaLeaf_left_highres.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else if (compound_leaf_index == 0) {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CowpeaLeaf_tip_highres.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CowpeaLeaf_right_highres.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CowpeaPod.obj", make_vec3(0., 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint CowpeaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs;
    if (flower_is_open) {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CowpeaFlower_open_yellow.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/CowpeaFlower_closed_yellow.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void CowpeaPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index > 5 || phytomer->rank > 1) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    } else {
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.6 + 0.4 * plant_age / 8.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    if (phytomer->rank == 0) {
        float inode_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 10.f);
        phytomer->scaleInternodeMaxLength(inode_scale);
    }
}

// Function to generate random float between min and max
float random_float(float min, float max) {
    return min + static_cast<float>(rand()) / (static_cast<float>(RAND_MAX / (max - min)));
}

// Function to check if two spheres overlap
bool spheres_overlap(const helios::vec3 &center1, float radius1, const helios::vec3 &center2, float radius2) {
    float distance = std::sqrt(std::pow(center1.x - center2.x, 2) + std::pow(center1.y - center2.y, 2) + std::pow(center1.z - center2.z, 2));
    return distance < (radius1 + radius2);
}

uint GrapevineFruitPrototype(helios::Context *context_ptr, uint subdivisions) {

    int num_grapes = 60;
    float height = 6.0f; // Height of the cluster
    float base_radius = 2.f; // Base radius of the cluster
    float taper_factor = 0.6f; // Taper factor (higher means more taper)
    float grape_radius = 0.25f; // Fixed radius for each grape

    std::vector<std::pair<helios::vec3, float>> grapes;
    float z_step = height / num_grapes;

    // Place the first grape at the bottom center
    helios::vec3 first_center(0.0f, 0.0f, 0.0f);
    grapes.push_back({first_center, grape_radius});

    // Attempt to place each subsequent grape close to an existing grape
    int max_attempts = 100; // Number of retries to find a tight fit

    for (int i = 1; i < num_grapes; ++i) {
        float z = i * z_step;
        // Tapered radius based on height (denser at the top, sparser at the bottom)
        float taper_radius = base_radius * (1.0f - taper_factor * (z / height));

        bool placed = false;
        int attempts = 0;
        while (!placed && attempts < max_attempts) {
            // Randomly select an existing grape as the reference point
            int reference_idx = rand() % grapes.size();
            const helios::vec3 &reference_center = grapes[reference_idx].first;

            // Pick a random offset direction from the reference grape
            float angle = random_float(0, 2 * M_PI);
            float distance = random_float(1.2 * grape_radius, 1.3 * grape_radius); // Keep grapes close but not overlapping

            // Compute the new potential center for the grape
            helios::vec3 new_center(reference_center.x + distance * cos(angle), reference_center.y + distance * sin(angle), random_float(z - 0.5f * z_step, z + 0.5f * z_step));

            // Check that the new center is within the allowable radius (for tapering)
            float new_center_distance = std::sqrt(new_center.x * new_center.x + new_center.y * new_center.y);
            if (new_center_distance > taper_radius) {
                attempts++;
                continue; // Skip if the new position exceeds the tapered radius
            }

            // Check for collisions with existing grapes
            bool collision = false;
            for (const auto &grape: grapes) {
                if (spheres_overlap(new_center, grape_radius, grape.first, grape.second)) {
                    collision = true;
                    break;
                }
            }

            // If no collision, place the grape
            if (!collision) {
                grapes.push_back({new_center, grape_radius});
                placed = true;
            }

            attempts++;
        }
    }

    std::vector<uint> UUIDs;
    for (const auto &grape: grapes) {
        //        std::vector<uint> UUIDs_tmp = context_ptr->addSphere( 10, grape.first, grape.second, "../../../plugins/plantarchitecture/assets/textures/GrapeBerry.jpg" );
        std::vector<uint> UUIDs_tmp = context_ptr->addSphere(10, grape.first, grape.second, make_RGBcolor(0.053, 0.076, 0.098));
        UUIDs.insert(UUIDs.end(), UUIDs_tmp.begin(), UUIDs_tmp.end());
    }

    context_ptr->rotatePrimitive(UUIDs, -0.5 * M_PI, "y");

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

// uint GrapevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open ){
//     std::vector<uint> UUIDs = context_ptr->loadOBJ( "plugins/plantarchitecture/assets/obj/OliveFlower_open.obj", make_vec3(0.0,0,0), 0,nullrotation, RGB::black, "ZUP", true );
//     uint objID = context_ptr->addPolymeshObject( UUIDs );
//     return objID;
// }

void GrapevinePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // blind nodes
    if (shoot_node_index >= 2) {
        phytomer->setFloralBudState(BUD_DEAD);
    }
}

// void GrapevinePhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer ){
//
//     if( phytomer->isdormant ){
//         if( phytomer->shoot_index.x >= phytomer->shoot_index.y-1  ){
//             phytomer->setVegetativeBudState( BUD_DORMANT ); //first vegetative buds always break
//         }
//         if( phytomer->shoot_index.x <= phytomer->shoot_index.y-4  ){
//             phytomer->setFloralBudState( BUD_DORMANT ); //first vegetative buds always break
//         }
//     }
//
// }

uint MaizeTasselPrototype(helios::Context *context_ptr, uint subdivisions) {

    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/MaizeTassel.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

uint MaizeEarPrototype(helios::Context *context_ptr, uint subdivisions) {

    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/MaizeEar.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

void MaizePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // set leaf scale based on position along the shoot
    float scale;
    if (shoot_node_index <= 5) {
        scale = fmin(1.f, 0.7 + 0.3 * float(shoot_node_index) / 5.f);
        phytomer->scaleInternodeMaxLength(scale);
    } else if (shoot_node_index >= phytomer->shoot_index.z - 5) {
        scale = fmin(1.f, 0.65 + 0.35 * float(phytomer->shoot_index.z - shoot_node_index) / 3.f);
    }

    phytomer->scaleLeafPrototypeScale(scale);

    if (shoot_node_index > 8 && shoot_node_index < 12) {
        phytomer->phytomer_parameters.inflorescence.flowers_per_peduncle = 1;
        phytomer->phytomer_parameters.inflorescence.fruit_prototype_function = MaizeEarPrototype;
        phytomer->phytomer_parameters.inflorescence.fruit_prototype_scale = 0.2;
        phytomer->phytomer_parameters.peduncle.length = 0.05f;
        phytomer->phytomer_parameters.peduncle.radius = 0.01;
        phytomer->phytomer_parameters.peduncle.pitch = 5;
        phytomer->setFloralBudState(BUD_ACTIVE);
    } else {
        phytomer->phytomer_parameters.inflorescence.fruit_prototype_function = MaizeTasselPrototype;
        phytomer->setFloralBudState(BUD_DEAD);
    }

    //    phytomer->setFloralBudState( BUD_DEAD );
}

uint OliveLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index) {

    std::vector<uint> UUIDs_upper =
            context_ptr->addTile(make_vec3(0.5, 0, 0), make_vec2(1, 0.2), nullrotation, make_int2(prototype_parameters->subdivisions, prototype_parameters->subdivisions), "plugins/plantarchitecture/assets/textures/OliveLeaf_upper.png");
    std::vector<uint> UUIDs_lower =
            context_ptr->addTile(make_vec3(0.5, 0, -1e-4), make_vec2(1, 0.2), nullrotation, make_int2(prototype_parameters->subdivisions, prototype_parameters->subdivisions), "plugins/plantarchitecture/assets/textures/OliveLeaf_lower.png");
    context_ptr->rotatePrimitive(UUIDs_lower, M_PI, "x");

    UUIDs_upper.insert(UUIDs_upper.end(), UUIDs_lower.begin(), UUIDs_lower.end());
    uint objID = context_ptr->addPolymeshObject(UUIDs_upper);
    return objID;
}

uint OliveFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/OliveFruit.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    std::vector<uint> UUIDs_fruit = context_ptr->filterPrimitivesByData(context_ptr->getObjectPrimitiveUUIDs(objID), "object_label", "fruit");
    context_ptr->setPrimitiveColor(UUIDs_fruit, make_RGBcolor(0.65, 0.7, 0.4)); // green
    return objID;
}

uint OliveFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/OliveFlower_open.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void OlivePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {
}

void OlivePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    if (phytomer->isdormant) {
        if (phytomer->shoot_index.x < phytomer->shoot_index.y - 8) {
            phytomer->setFloralBudState(BUD_DEAD);
        }
    }
}

uint PistachioFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/PistachioNut.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint PistachioFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/OliveFlower_open.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void PistachioPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // blind nodes
    if (shoot_node_index == 0) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }
}

void PistachioPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    if (phytomer->isdormant) {
        if (phytomer->shoot_index.x <= phytomer->shoot_index.y - 4) {
            phytomer->setFloralBudState(BUD_DORMANT);
        }
    }
}

uint PuncturevineFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/PuncturevineFlower.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint RedbudFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/RedbudFlower_open.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

uint RedbudFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/RedbudPod.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    return context_ptr->addPolymeshObject(UUIDs);
}

void RedbudPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {
}

void RedbudPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    // redbud has the shoot pattern that the first few nodes on the shoot are vegetative, then the rest are floral
    if (phytomer->isdormant) {
        int Nchild_shoots = randu(2, 4);
        if (phytomer->shoot_index.x < phytomer->shoot_index.y - Nchild_shoots) {
            phytomer->setVegetativeBudState(BUD_DEAD);
        } else {
            phytomer->setFloralBudState(BUD_DEAD);
        }
    }
}

uint RiceSpikePrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/RiceGrain.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void RicePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // set leaf scale based on position along the shoot
    float scale = fmin(1.f, 0.7 + 0.3 * float(shoot_node_index) / 5.f);
    phytomer->scaleLeafPrototypeScale(scale);

    // set internode length based on position along the shoot
    phytomer->scaleInternodeMaxLength(scale);
}

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    float fact = float(shoot_max_nodes - shoot_node_index) / float(shoot_max_nodes);

    // set leaf scale based on position along the shoot
    //    float scale = fmin(1.f, 1 + 0.1*fact);
    //    phytomer->scaleLeafPrototypeScale(scale);

    //    phytomer->rotateLeaf( 0, 0, make_AxisRotation(-deg2rad(15)*fact, 0, 0));
    phytomer->rotateLeaf(0, 0, make_AxisRotation(-deg2rad(60) * fact, 0, 0));
}

uint SorghumPaniclePrototype(helios::Context *context_ptr, uint subdivisions) {

    if (subdivisions <= 1) {
        subdivisions = 3;
    }

    float panicle_height = 1;
    float panicle_width = 0.08;
    float width_seed = 0.08;
    float height_seed = 0.25;
    float seed_tilt = 50;
    subdivisions = 6;

    std::string seed_texture_file = "plugins/plantarchitecture/assets/textures/SorghumSeed.jpeg";
    RGBcolor stem_color(0.45, 0.55, 0.42);

    std::vector<uint> UUIDs;

    panicle_height -= 0.8 * height_seed;

    std::vector<vec3> nodes_panicle;
    std::vector<float> radius_panicle;

    for (int n = 0; n < subdivisions; n++) {
        float x = 0;
        float y = 0;
        float z;
        if (n == 0) {
            z = 0.5f * height_seed / float(subdivisions - 1);
        } else if (n == subdivisions - 1) {
            z = (subdivisions - 1.5f) * height_seed / float(subdivisions - 1);
        } else {
            z = n * height_seed / float(subdivisions - 1);
        }

        float angle = float(n) * M_PI / float(subdivisions - 1);
        float dr = std::fmax(0.f, 0.5f * width_seed * sin(angle));

        nodes_panicle.push_back(make_vec3(x, y, z));
        radius_panicle.push_back(dr);
    }

    std::vector<uint> UUIDs_seed_ptype = context_ptr->addTube(subdivisions, nodes_panicle, radius_panicle, seed_texture_file.c_str());

    int Ntheta = ceil(6.f * panicle_height / height_seed);
    int Nphi = ceil(2.f * M_PI * panicle_width / width_seed);

    for (int j = 0; j < Nphi; j++) {
        for (int i = 0; i < Ntheta; i++) {

            if (i == 0 && j == 0) {
                continue;
            }

            std::vector<uint> UUIDs_copy = context_ptr->copyPrimitive(UUIDs_seed_ptype);
            context_ptr->scalePrimitive(UUIDs_copy, make_vec3(1, 1, 1) * context_ptr->randu(0.9f, 1.1f));

            float phi = 2.f * M_PI * float(j + 0.5f * float(i % 2)) / float(Nphi);
            float theta = acos(1 - 2 * float(i + float(j) / float(Nphi)) / float(Ntheta));
            float x = sin(theta) * cos(phi);
            float y = sin(theta) * sin(phi);
            float z = 0.5f + 0.5f * cos(theta);

            x *= 0.5f * panicle_width;
            y *= 0.5f * panicle_width;
            z *= panicle_height;

            float tilt = -deg2rad(seed_tilt) * sqrtf(1.f - z / panicle_height);

            context_ptr->rotatePrimitive(UUIDs_copy, tilt, "x");
            context_ptr->rotatePrimitive(UUIDs_copy, phi - 0.5f * M_PI, "z");

            context_ptr->translatePrimitive(UUIDs_copy, make_vec3(x, y, z));
            UUIDs.insert(UUIDs.end(), UUIDs_copy.begin(), UUIDs_copy.end());
        }
    }

    context_ptr->deletePrimitive(UUIDs_seed_ptype);

    std::vector<uint> UUIDs_sphere = context_ptr->addSphere(10, make_vec3(0, 0, 0.5 * panicle_height), 0.5f, seed_texture_file.c_str());
    context_ptr->scalePrimitiveAboutPoint(UUIDs_sphere, make_vec3(1.9 * panicle_width, 1.9 * panicle_width, 0.8 * panicle_height), make_vec3(0, 0, 0.5 * panicle_height));
    UUIDs.insert(UUIDs.end(), UUIDs_sphere.begin(), UUIDs_sphere.end());

    context_ptr->rotatePrimitive(UUIDs, 0.5f * M_PI, "y");
    context_ptr->translatePrimitive(UUIDs, make_vec3(-0.2, 0, 0));

    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void SorghumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // set leaf scale based on position along the shoot
    float scale = fmin(1.f, 0.7 + 0.3 * float(shoot_node_index) / 5.f);
    phytomer->scaleLeafPrototypeScale(scale);

    // set internode length based on position along the shoot
    phytomer->scaleInternodeMaxLength(scale);
}

uint SoybeanFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/SoybeanPod.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint SoybeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs;
    if (flower_is_open) {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/SoybeanFlower_open_white.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    } else {
        UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/BeanFlower_closed_white.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    }
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void SoybeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index > 5 || phytomer->rank > 1) {
        phytomer->setVegetativeBudState(BUD_DEAD);
    } else {
        phytomer->setFloralBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.2 + 0.8 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.1 + 0.9 * plant_age / 15.f);
    phytomer->scaleInternodeMaxLength(inode_scale);
}

uint StrawberryFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/StrawberryFlower.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint StrawberryFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/StrawberryFruit.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint TomatoFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/TomatoFruit.obj", make_vec3(0., 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint TomatoFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/TomatoFlower.obj", make_vec3(0.0, 0, 0), 0.75, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void TomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index < 8 && phytomer->rank == 0) {
        phytomer->setFloralBudState(BUD_DEAD);
    }
    if (phytomer->rank > 1) {
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->setVegetativeBudState(BUD_DEAD);
    }
    if (phytomer->rank > 1) {
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->setVegetativeBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.5 + 0.5 * plant_age / 10.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 10.f);
    phytomer->scaleInternodeMaxLength(inode_scale);
}

void CherryTomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    if (shoot_node_index < 8 || phytomer->rank > 1) {
        phytomer->setFloralBudState(BUD_DEAD);
        phytomer->setVegetativeBudState(BUD_DEAD);
    }

    // set leaf and internode scale based on position along the shoot
    float leaf_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 15.f);
    phytomer->scaleLeafPrototypeScale(leaf_scale);

    // set internode length based on position along the shoot
    float inode_scale = fmin(1.f, 0.7 + 0.3 * plant_age / 10.f);
    phytomer->scaleInternodeMaxLength(inode_scale);
}

void CherryTomatoPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {

    float pruning_height = 1.f;
    float pruning_day = 101.f;

    float plant_age = phytomer->parent_shoot_ptr->plantarchitecture_ptr->getPlantAge(phytomer->plantID);

    if (phytomer->hasLeaf() && plant_age >= pruning_day) {
        float height = phytomer->getInternodeNodePositions().at(0).z;
        if (height < pruning_height) {
            phytomer->removeLeaf();
        }
    }
}

uint WalnutFruitPrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/WalnutHull.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

uint WalnutFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/AlmondFlower.obj", make_vec3(0.0, 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void WalnutPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // blind nodes
    if (shoot_node_index < 4) {
        phytomer->setVegetativeBudState(BUD_DEAD);
        phytomer->setFloralBudState(BUD_DEAD);
    }
}

void WalnutPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer) {
}

uint WheatSpikePrototype(helios::Context *context_ptr, uint subdivisions) {
    std::vector<uint> UUIDs = context_ptr->loadOBJ("plugins/plantarchitecture/assets/obj/WheatSpike.obj", make_vec3(0., 0, 0), 0, nullrotation, RGB::black, "ZUP", true);
    uint objID = context_ptr->addPolymeshObject(UUIDs);
    return objID;
}

void WheatPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age) {

    // set leaf scale based on position along the shoot
    float scale = std::fmin(1.f, 0.7f + 0.3f * float(shoot_node_index) / 5.f);
    phytomer->scaleLeafPrototypeScale(scale);

    // set internode length based on position along the shoot
    phytomer->scaleInternodeMaxLength(scale);
}
