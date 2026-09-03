
#ifndef HELIOS_ASSETS_H
#define HELIOS_ASSETS_H

//! Deflect a leaf blade lattice under its own weight
/**
 * Treats the blade as a cantilever clamped at its base and loaded by its own distributed weight. The bending moment at each station along the midrib is the weight of everything distal to it acting through
 * its horizontal lever arm; curvature is proportional to that moment and is integrated along the arc to give the deflected centreline, which the rest of the lattice is then carried along with.
 *
 * The deflection is computed from the undeformed rest lattice every time rather than accumulated onto the previous result, so a leaf that stops growing stops drooping instead of creeping downward.
 * Because the moment grows with the cube of blade length while the stiffness does not, a leaf that doubles in length droops roughly eight times as far - growth-driven droop falls out of the mechanics
 * and needs no separate age term.
 * \param[in] rest_vertices Undeformed lattice vertices, row-major over (Nx+1) x (Ny+1) with the midrib running along the local x-axis.
 * \param[in] Nx Number of blade subdivisions along the leaf length.
 * \param[in] Ny Number of blade subdivisions across the leaf width.
 * \param[in] scale Current length scale of the leaf, as applied to the unit-length rest lattice.
 * \param[in] mature_scale Length scale the leaf reaches when fully grown. This is the reference length the dimensionless flexibility is measured against, so it must be the leaf's final size rather than its
 * current one - normalizing by the current size would divide out the growth-driven droop entirely.
 * \param[in] flexibility Dimensionless droopiness of the fully-grown leaf. Zero returns the rest lattice unchanged, which is the path taken by every species that does not droop. Values around 20 give a
 * mature grass blade that arcs over to roughly a third of its length, and the same value means the same shape regardless of how long the leaf actually is.
 * \param[in] taper How much more compliant the blade is at its tip than at its base, as a ratio of the two. A blade of uniform stiffness bends hardest where the bending moment is largest, which is at the
 * clamped base, and is left nearly straight over its outer half; a real grass leaf does the opposite, staying straight near the base and curving hardest toward the tip. The difference is the midrib, whose
 * thickness falls several-fold from base to tip and takes the blade's bending stiffness down with it faster than the moment itself decays. One leaves the stiffness uniform along the blade.
 * \return Deflected lattice vertices, parallel to \p rest_vertices.
 */
std::vector<helios::vec3> deformLeafLattice(const std::vector<helios::vec3> &rest_vertices, uint Nx, uint Ny, float scale, float mature_scale, float flexibility, float taper = 1.f);

uint GenericLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);

uint GeneralSphericalFruitPrototype(helios::Context *context_ptr, uint subdivisions);

uint AlmondFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint AlmondFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void AlmondPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void AlmondPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint AppleFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint AppleFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void ApplePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void ApplePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint AsparagusLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
void AsparagusPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint BeanLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint BeanFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint BeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void BeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint BindweedFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);

uint BougainvilleaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);

uint CapsicumFruitPrototype(helios::Context *context_ptr, uint subdivisions);
void CapsicumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint CheeseweedLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint CowpeaFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint CowpeaFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void CowpeaPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint GrapevineFruitPrototype(helios::Context *context_ptr, uint subdivisions);
// uint GrapevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void GrapevinePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
// void GrapevinePhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint MaizeTasselPrototype(helios::Context *context_ptr, uint subdivisions);
uint MaizeEarPrototype(helios::Context *context_ptr, uint subdivisions);
void MaizePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint OliveLeafPrototype(helios::Context *context_ptr, LeafPrototype *prototype_parameters, int compound_leaf_index);
uint OliveFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint OliveFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void OlivePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void OlivePhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint PistachioFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint PistachioFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void PistachioPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void PistachioPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint PuncturevineFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);

uint RedbudFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
uint RedbudFruitPrototype(helios::Context *context_ptr, uint subdivisions);
void RedbudPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void RedbudPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint RiceSpikePrototype(helios::Context *context_ptr, uint subdivisions);
void RicePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint SorghumPaniclePrototype(helios::Context *context_ptr, uint subdivisions);
void SorghumPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint SoybeanFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint SoybeanFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void SoybeanPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint StrawberryFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint StrawberryFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void StrawberryPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

uint TomatoFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint TomatoFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void TomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);

void CherryTomatoPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void CherryTomatoPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint WalnutFruitPrototype(helios::Context *context_ptr, uint subdivisions);
uint WalnutFlowerPrototype(helios::Context *context_ptr, uint subdivisions, bool flower_is_open = false);
void WalnutPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);
void WalnutPhytomerCallbackFunction(std::shared_ptr<Phytomer> phytomer);

uint WheatSpikePrototype(helios::Context *context_ptr, uint subdivisions);
void WheatPhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age);


#endif // HELIOS_ASSETS_H
