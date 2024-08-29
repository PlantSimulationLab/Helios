
#ifndef HELIOS_ASSETS_H
#define HELIOS_ASSETS_H

uint buildGenericLeafPrototype(helios::Context *context_ptr, uint subdivisions, const std::string &leaf_texture, float leaf_aspect_ratio, float midrib_fold_fraction, float x_curvature, float y_curvature, float petiole_roll, float wave_period, float wave_amplitude);

uint AlmondLeafPrototype( helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint AlmondFruitPrototype( helios::Context* context_ptr, uint subdivisions, float time_since_fruit_set );
uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void AlmondPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

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

#endif //HELIOS_ASSETS_H