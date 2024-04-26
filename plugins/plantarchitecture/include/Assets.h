
#ifndef HELIOS_ASSETS_H
#define HELIOS_ASSETS_H

#include "PlantArchitecture.h"

uint BeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint BeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint BeanFruitPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint BeanFlowerPrototype( helios::Context* context_ptr, uint subdivisions=1, bool flower_is_open=false );
void BeanPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );

uint SoybeanLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint SoybeanLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions=1, int flag=0 );

uint CowpeaLeafPrototype_unifoliate(helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint CowpeaLeafPrototype_trifoliate(helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint CowpeaFruitPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint CowpeaFlowerPrototype( helios::Context* context_ptr, uint subdivisions=1, bool flower_is_open=false );

uint TomatoLeafPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint TomatoFruitPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint TomatoFlowerPrototype( helios::Context* context_ptr, uint subdivisions=1, bool flower_is_open=false );

uint AlmondLeafPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint AlmondFruitPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint AlmondFlowerPrototype( helios::Context* context_ptr, uint subdivisions=1, bool flower_is_open=false );

uint CheeseweedLeafPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );

uint BindweedLeafPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint BindweedFlowerPrototype( helios::Context* context_ptr, uint subdivisions=1, bool flower_is_open=false );

uint SorghumLeafPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
uint SorghumPaniclePrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 );
void SorghumPhytomerCreationFunction( std::shared_ptr<Phytomer> phytomer, uint shoot_node_index, uint parent_shoot_node_index, uint shoot_max_nodes, float plant_age );


#endif //HELIOS_ASSETS_H