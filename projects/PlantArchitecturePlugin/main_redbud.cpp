#include "PlantArchitecture.h"
#include "Visualizer.h"

using namespace helios;

uint RedbudLeafPrototype( helios::Context* context_ptr, uint subdivisions=1, int flag=0 ){

    //uint leafID = context_ptr->addDiskObject( 10, make_vec3(0.5,0,0), make_vec2(1,1), nullrotation, RGB::green );
    std::vector<uint> UUIDs = context_ptr->loadOBJ( "../obj/RedbudLeaf.obj", nullorigin, 0, nullrotation, RGB::green, "ZUP", true );
    uint leafID = context_ptr->addPolymeshObject(UUIDs);
    return leafID;

}

int main(){

    float growth_respiration = 0;  //grams CHO respired to produce 1 gram of dry weight
    float maintainance_respiration_rate = 0; //grams CHO per gram dry weight per second

    Context context;

    //context.seedRandomGenerator(60);

    PlantArchitecture plantarchitecture(&context);

    PhytomerParameters phytomer_parameters = plantarchitecture.getPhytomerParametersFromLibrary("almond");
    phytomer_parameters.internode.pitch = deg2rad(20);
    phytomer_parameters.internode.length = 0.03;
    phytomer_parameters.internode.radius = 0.002;
    phytomer_parameters.internode.curvature = -0;
    phytomer_parameters.petiole.pitch.uniformDistribution(-deg2rad(40),-deg2rad(60));
    phytomer_parameters.petiole.yaw = M_PI;
    phytomer_parameters.petiole.radius = 0.0015;
    phytomer_parameters.leaf.prototype_function = RedbudLeafPrototype;
    phytomer_parameters.leaf.prototype_scale = 0.03*make_vec3(1,1,1);
//    phytomer_parameters.internode.petioles_per_internode = 0;
    phytomer_parameters.petiole.length = 0.03;
    phytomer_parameters.petiole.curvature = 0;

    ShootParameters shoot_parameters(context.getRandomGenerator());
    shoot_parameters.phytomer_parameters = phytomer_parameters;
    shoot_parameters.max_nodes = 15;
    shoot_parameters.shoot_internode_taper = 0.2;
    shoot_parameters.growth_rate = 0.02;
    shoot_parameters.bud_time = 1;
    shoot_parameters.growth_requires_dormancy = true;
    shoot_parameters.child_internode_length_max = 0.03;
    shoot_parameters.child_internode_length_decay_rate = 0.0005;
//    shoot_parameters.child_internode_length_min = 0.03;
    shoot_parameters.child_insertion_angle_tip = deg2rad(50);
    shoot_parameters.child_insertion_angle_decay_rate = 0.05;
    shoot_parameters.defineChildShootTypes({"main"},{1.0});

    // shaded
    shoot_parameters.phyllochron = 1;
    shoot_parameters.phytomer_parameters.leaf.pitch.uniformDistribution(-deg2rad(20),-deg2rad(45));
    shoot_parameters.bud_break_probability = 0.65;
    // sunlit
//    shoot_parameters.phyllochron = 1.5;
//    shoot_parameters.phytomer_parameters.leaf.pitch.uniformDistribution(-deg2rad(70),-deg2rad(90));
//    shoot_parameters.bud_break_probability = 0.9;

    plantarchitecture.defineShootType("main", shoot_parameters );
    shoot_parameters.phytomer_parameters.internode.pitch = 0;
    shoot_parameters.phytomer_parameters.internode.radius = 0.015;
    shoot_parameters.phytomer_parameters.internode.petioles_per_internode = 0;
    shoot_parameters.shoot_internode_taper = 0.5;
    shoot_parameters.max_nodes = 25;
    shoot_parameters.bud_break_probability = 0.8;
    plantarchitecture.defineShootType("trunk", shoot_parameters );

    int Nplants = 1;

    uint plant0 = plantarchitecture.addPlantInstance(nullorigin, 0);

    uint uID_trunk = plantarchitecture.addBaseShoot( plant0, 25, make_AxisRotation(deg2rad(5),0,0.5*M_PI), "trunk" );

    for( int node=0; node<8; node++ ){
        plantarchitecture.setPhytomerVegetativeBudState( plant0, uID_trunk, node, BUD_DEAD );
    }

    plantarchitecture.setPlantPhenologicalThresholds(plant0, 100, 20, 3, 5, 10);

    plantarchitecture.breakPlantDormancy(plant0);

    // ---------------- //

    context.addDisk( 20, nullorigin, make_vec2(2.5,2.5), nullrotation, RGB::white );

    bool render = false;
    int Nframes = 20;

    for( int i=0; i<Nframes; i++ ) {

        if( render ) {

            Visualizer vis(1200);
            vis.disableMessages();

//            vis.setCameraPosition(make_SphericalCoord(0.3, 0.1 * M_PI, 0. * M_PI), make_vec3(0, 0, 0.1));
            vis.setCameraPosition(make_SphericalCoord(2.2,0.31416,1.09957), make_vec3(0, 0, 0.4));

            //    vis.colorContextPrimitivesByData( "rank" );
            vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);
            //    vis.setLightDirection(make_vec3(0,0,1));

            vis.buildContextGeometry(&context);

            //        vis.plotInteractive();
            vis.plotUpdate();
            wait(1);

            std::stringstream framefile;
            framefile << "../frames/almond_growth_" << std::setfill('0') << std::setw(3) << i << ".jpeg";
            vis.printWindow(framefile.str().c_str());

            vis.closeWindow();

        }

        plantarchitecture.advanceTime( 1 );

        std::cout << "Frame: " << i << std::endl;

    }

    Visualizer vis(1200);

    vis.setCameraPosition(make_SphericalCoord(2.2,0.31416,1.09957), make_vec3(0, 0, 0.4));
    vis.setLightingModel(Visualizer::LIGHTING_PHONG_SHADOWED);

    vis.buildContextGeometry(&context);

    vis.plotInteractive();


}