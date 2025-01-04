
#ifndef HELIOS_ASSETS_H
#define HELIOS_ASSETS_H

uint GenericLeafPrototype(helios::Context *context_ptr, LeafPrototype* prototype_parameters, int compound_leaf_index);

uint AlmondFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void AlmondPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void AlmondPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint AppleFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint AppleFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void ApplePhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void ApplePhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint AsparagusLeafPrototype( helios::Context* context_ptr, LeafPrototype* prototype_parameters, int compound_leaf_index );
void AsparagusPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint BeanLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint BeanLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint BeanFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint BeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void BeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint BindweedFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );

uint CowpeaLeafPrototype_unifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint CowpeaLeafPrototype_trifoliate_OBJ(helios::Context* context_ptr, uint subdivisions, int compound_leaf_index );
uint CowpeaFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint CowpeaFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void CowpeaPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint GrapevineFruitPrototype( helios::Context* context_ptr, uint subdivisions );
//uint GrapevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void GrapevinePhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
//void GrapevinePhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint MaizeTasselPrototype( helios::Context* context_ptr, uint subdivisions );
uint MaizeEarPrototype( helios::Context* context_ptr, uint subdivisions );
void MaizePhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint OliveLeafPrototype( helios::Context* context_ptr, LeafPrototype* prototype_parameters, int compound_leaf_index );
uint OliveFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint OliveFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void OlivePhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void OlivePhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint PistachioFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint PistachioFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void PistachioPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void PistachioPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint PuncturevineFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );

uint RedbudFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
uint RedbudFruitPrototype( helios::Context* context_ptr, uint subdivisions );
void RedbudPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void RedbudPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint RiceSpikePrototype( helios::Context* context_ptr, uint subdivisions );
void RicePhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

void ButterLettucePhytomerCreationFunction(std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint SorghumPaniclePrototype( helios::Context* context_ptr, uint subdivisions );
void SorghumPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint SoybeanFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint SoybeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void SoybeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint StrawberryFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint StrawberryFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void StrawberryPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint TomatoFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint TomatoFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void TomatoPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint WalnutFruitPrototype( helios::Context* context_ptr, uint subdivisions );
uint WalnutFlowerPrototype( helios::Context* context_ptr, uint subdivisions, bool flower_is_open=false );
void WalnutPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );
void WalnutPhytomerCallbackFunction( std::shared_ptr<Phytomer> phytomer );

uint WheatSpikePrototype( helios::Context* context_ptr, uint subdivisions );
void WheatPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );


#endif //HELIOS_ASSETS_H