
#ifndef HELIOS_ASSETS_H
#define HELIOS_ASSETS_H

//! Function to procedurally build a generic leaf prototype with features such as midrib fold, curvature, etc.
/**
 * \param[in] context_ptr Pointer to the Helios context
 * \param[in] subdivisions Subdivision discretization level of leaf mesh. Higher values result in smoother leaf surfaces but more primitives.
 * \param[in] leaf_texture Path to image texture file for the leaf
 * \param[in] leaf_aspect_ratio Ratio of leaf lateral dimension (width) to longitudinal dimension (length)
 * \param[in] midrib_fold_fraction Leaf fold fraction along the midrib. 0.0 is no fold, 1.0 is a complete fold.
 * \param[in] longitudinal_curvature Curvature factor in the longitudinal direction (along the midrib). Positive values curve the leaf upwards, negative values curve the leaf downwards.
 * \param[in] lateral_curvature Curvature factor in the lateral direction (perpendicular to the midrib). Positive values curve the leaf upwards, negative values curve the leaf downwards.
 * \param[in] petiole_roll Adds a small roll at the base of the leaf to meet the petiole. 0 is no roll. The larger the value the stronger the roll.
 * \param[in] wave_period Period of wavy ripples along the leaf midrib. 0 is no wave.
 * \param[in] wave_amplitude Amplitude of wavy ripples along the leaf midrib. 0 is no wave.
 * \param[in] build_petiolule Add a petiolule at the leaf base to meet the petiole.
 * @return Object ID for the leaf prototype
 */
uint buildGenericLeafPrototype(helios::Context *context_ptr, uint subdivisions, const std::string &leaf_texture, float leaf_aspect_ratio, float midrib_fold_fraction, float longitudinal_curvature, float lateral_curvature, float petiole_roll, float wave_period,
                               float wave_amplitude, bool build_petiolule);

uint AlmondLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint AlmondFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void AlmondPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void AlmondPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint AsparagusLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
void AsparagusPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint BeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint BeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint BeanLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint BeanLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint BeanFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
uint BeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void BeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint BindweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index0 );
uint BindweedFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );

uint CheeseweedLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );

uint CowpeaLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint CowpeaLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint CowpeaFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
uint CowpeaFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void CowpeaPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint PuncturevineLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint PuncturevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );

uint RedbudLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint RedbudFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
uint RedbudFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
void RedbudPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void RedbudPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint RomaineLettuceLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
void RomaineLettucePhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint SorghumLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint SorghumPaniclePrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
void SorghumPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint SoybeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint SoybeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint SoybeanFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
uint SoybeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void SoybeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint StrawberryLeafPrototype(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint StrawberryFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
uint StrawberryFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void StrawberryPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint SugarbeetLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );

uint TomatoLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint TomatoFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
uint TomatoFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void TomatoPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint WheatLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint WheatSpikePrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
void WheatPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );


#endif //HELIOS_ASSETS_H