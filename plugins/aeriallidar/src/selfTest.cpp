#include "AerialLiDAR.h"
#include "Visualizer.h"


using namespace helios;

int AerialLiDARcloud::selfTest(void){

    float err_tol = 0.1;

    int fail_count = 0;

    //------- dense vegatation test to check "mean dr" method for calculating LAD -------//

    std::cout << "Running aerial LiDAR dense vegetation test..." << std::flush;

    Context context_1;

    vec3 boxsize(10,10,10);

    int3 Nleaves(100,100,100);

    float L = 0.05;

    bool flag;
    float LAD_exact;

    for( int k=0; k<Nleaves.z; k++ ){
        for( int j=0; j<Nleaves.y; j++ ){
            for( int i=0; i<Nleaves.x; i++ ){

                vec3 x( context_1.randu()*boxsize.x, context_1.randu()*boxsize.y, context_1.randu()*boxsize.z );

                float theta = acos(1.f-context_1.randu());
                float phi = context_1.randu()*2.f*M_PI;

                context_1.addPatch( x, make_vec2(L,L), make_SphericalCoord(theta,phi) );

            }
        }
    }

    context_1.addPatch( make_vec3(0.5*boxsize.x,0.5*boxsize.y,-0.001), make_vec2( boxsize.x, boxsize.y ) );

    LAD_exact = float(Nleaves.x*Nleaves.y*Nleaves.z)*L*L/(boxsize.x*boxsize.y*boxsize.z);

    flag = true;
    AerialLiDARcloud lidar_1;
    lidar_1.disableMessages();

    lidar_1.syntheticScan( &context_1, "plugins/aeriallidar/xml/synthetic_aerial_test.xml" );

    helios::vec3 center=lidar_1.getGridCenter();
    if( center.x!=5.0 || center.y!=5.0 || center.z!=5.0 ){
        std::cout << "Grid center was not set correctly." << std::endl;
        flag = false;
    }

    helios::int3 resolution=lidar_1.getGridResolution();
    if( resolution.x!=2 || resolution.y!=2 || resolution.z!=2 ){
        std::cout << "Grid resolution was not set correctly." << std::endl;
        flag = false;
    }

    int v = 6;
    helios::int3 ijk = lidar_1.gridindex2ijk(v);
    if( v!=lidar_1.gridijk2index(ijk) ){
        printf("ijk = (%d,%d,%d) %d\n",ijk.x,ijk.y,ijk.z,lidar_1.gridijk2index(ijk));
        flag = false;
        std::cout << "ijk failed" << std::endl;
    }

    lidar_1.calculateLeafAreaGPU( 0.5, 10 );

    for( int v=4; v<8; v++ ){

        float LAD = lidar_1.getCellLeafAreaDensity( lidar_1.gridindex2ijk(v) );

        if( fabs(LAD-LAD_exact)/LAD_exact > err_tol ){
            flag = false;
        }

    }

    if( flag ){
        std::cout << "passed." << std::endl;
    }else{
        std::cout << "failed." << std::endl;
        fail_count ++;
    }

    //------- sparse vegatation test to check "mean P" method for calculating LAD -------//

    std::cout << "Running aerial LiDAR sparse vegetation test..." << std::flush;

    Context context_2;

    Nleaves = make_int3(25,25,25);

    for( int k=0; k<Nleaves.z; k++ ){
        for( int j=0; j<Nleaves.y; j++ ){
            for( int i=0; i<Nleaves.x; i++ ){

                vec3 x( context_2.randu()*boxsize.x, context_2.randu()*boxsize.y, context_2.randu()*boxsize.z );

                float theta = acos(1.f-context_2.randu());
                float phi = context_2.randu()*2.f*M_PI;

                context_2.addPatch( x, make_vec2(L,L), make_SphericalCoord(theta,phi) );

            }
        }
    }

    context_2.addPatch( make_vec3(0.5*boxsize.x,0.5*boxsize.y,-0.001), make_vec2( boxsize.x, boxsize.y ) );

    LAD_exact = float(Nleaves.x*Nleaves.y*Nleaves.z)*L*L/(boxsize.x*boxsize.y*boxsize.z);

    AerialLiDARcloud lidar_2;
    lidar_2.disableMessages();

    lidar_2.syntheticScan( &context_2, "plugins/aeriallidar/xml/synthetic_aerial_test.xml" );

    lidar_2.calculateLeafAreaGPU( 0.5, 10 );

    flag = true;
    for( int v=0; v<8; v++ ){

        float LAD = lidar_2.getCellLeafAreaDensity( lidar_2.gridindex2ijk(v) );

        if( fabs(LAD-LAD_exact)/LAD_exact > 1.5*err_tol ){
            flag = false;
        }

    }

    if( flag ){
        std::cout << "passed." << std::endl;
    }else{
        std::cout << "failed." << std::endl;
        fail_count ++;
    }

    //------- sparse vegatation ground and canopy height estimation -------//

    std::cout << "Running aerial LiDAR ground and canopy height test..." << std::flush;

    Context context_3;

    Nleaves = make_int3(25,25,35);

    for( int k=0; k<Nleaves.z; k++ ){
        for( int j=0; j<Nleaves.y; j++ ){
            for( int i=0; i<Nleaves.x; i++ ){

                vec3 x( context_3.randu()*boxsize.x, context_3.randu()*boxsize.y, context_3.randu()*boxsize.z );

                float theta = acos(1.f-context_3.randu());
                float phi = context_3.randu()*2.f*M_PI;

                context_3.addPatch( x, make_vec2(L,L), make_SphericalCoord(theta,phi) );

            }
        }
    }

    float zground = 0.2;

    context_3.addPatch( make_vec3(0.5*boxsize.x,0.5*boxsize.y,zground), make_vec2( boxsize.x, boxsize.y ) );

    AerialLiDARcloud lidar_3;
    lidar_3.disableMessages();

    lidar_3.syntheticScan( &context_3, "plugins/aeriallidar/xml/synthetic_aerial_test.xml" );

    for( int r=0; r<lidar_3.getHitCount(); r++ ){

        lidar_3.setHitData( r, "target_index", 1 );
        lidar_3.setHitData( r, "target_count", 1 );

    }

    lidar_3.generateHeightModel( 100, 0.5, 0.1, 0.5, 0.1 );

    flag = true;
    for( int v=0; v<8; v++ ){

        int3 index = lidar_3.gridindex2ijk(v);
        float zg = lidar_3.getCellGroundHeight( make_int2(index.x,index.y) );

        if( fabs(zg-zground)/fabs(zground) > 1.5*err_tol ){
            flag = false;
        }

    }

    for( int r=0; r<lidar_3.getHitCount(); r++ ){

        vec3 xyz = lidar_3.getHitXYZ(r);

        if( fabs(xyz.z-zground)>9 ){
            lidar_3.setHitData( r, "target_index", 1 );
        }else{
            lidar_3.setHitData( r, "target_index", 2 );
        }
        lidar_3.setHitData( r, "target_count", 2 );

    }

    lidar_3.generateHeightModel( 400, 0.5, 0.1, 1.0, 0.2 );

    for( int v=0; v<8; v++ ){

        int3 index = lidar_3.gridindex2ijk(v);
        float zc = lidar_3.getCellVegetationHeight( make_int2(index.x,index.y) );
        float zm = lidar_3.getCellMaximumHitHeight( make_int2(index.x,index.y) );

        if( fabs(zc-(boxsize.z-0.5))/fabs(boxsize.z-0.5) > 1.5*err_tol ){
            flag = false;
        }else if( fabs(zm-boxsize.z)/fabs(boxsize.z) > err_tol ){
            flag = false;
        }

    }

    if( flag ){
        std::cout << "passed." << std::endl;
    }else{
        std::cout << "failed." << std::endl;
        fail_count ++;
    }



    {
        // Test AerialLiDARcloud::enableMessages()
        std::cout << "Running enableMessages..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages(); // Disable first
        lidar.enableMessages();  // Then enable again

        bool flag = true; // The method just toggles a flag; no exception expected

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::addHitPoint(scanID, hit_xyz, ray_origin)
        std::cout << "Running addHitPoint(scanID, hit_xyz, ray_origin)..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 30.0f, 10.0f, 0.1f, 0.05f);
        lidar.addScan(scan_meta);

        try {
            vec3 hit = make_vec3(1,2,3);
            vec3 origin = make_vec3(0,0,0);
            lidar.addHitPoint(0, hit, origin);
            if (lidar.getHitCount() != 1) flag = false;
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::addHitPoint(scanID, hit_xyz, direction)
        std::cout << "Running addHitPoint(scanID, hit_xyz, direction)..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 30.0f, 10.0f, 0.1f, 0.05f);
        lidar.addScan(scan_meta);

        try {
            vec3 hit = make_vec3(1,1,1);
            SphericalCoord dir = make_SphericalCoord(1.0f, 0.5f);
            lidar.addHitPoint(0, hit, dir);
            if (lidar.getHitCount() != 1) flag = false;
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::addHitPoint(scanID, hit_xyz, direction, data)
        std::cout << "Running addHitPoint(scanID, hit_xyz, direction, data)..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 30.0f, 10.0f, 0.1f, 0.05f);
        lidar.addScan(scan_meta);

        try {
            vec3 hit = make_vec3(1,1,1);
            SphericalCoord dir = make_SphericalCoord(1.0f, 0.5f);
            std::map<std::string, float> data = { {"reflectance", 123.45f} };
            lidar.addHitPoint(0, hit, dir, data);
            if (lidar.getHitCount() != 1) flag = false;
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::addHitPoint(scanID, hit_xyz, direction, color)
        std::cout << "Running addHitPoint(scanID, hit_xyz, direction, color)..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 30.0f, 10.0f, 0.1f, 0.05f);
        lidar.addScan(scan_meta);

        try {
            vec3 hit = make_vec3(1,1,1);
            SphericalCoord dir = make_SphericalCoord(1.0f, 0.5f);
            RGBcolor color = make_RGBcolor(0.2f, 0.8f, 0.6f);
            lidar.addHitPoint(0, hit, dir, color);
            if (lidar.getHitCount() != 1) flag = false;
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::deleteHitPoint()
        std::cout << "Running deleteHitPoint..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 10.0f, 1.0f, 0.1f, 0.01f);
        lidar.addScan(scan_meta);

        vec3 hit = make_vec3(1,1,1);
        vec3 origin = make_vec3(0,0,0);
        lidar.addHitPoint(0, hit, origin);

        try {
            lidar.deleteHitPoint(0);
            if (lidar.getHitCount() != 0) flag = false;
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::getHitColor()
        std::cout << "Running getHitColor..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 10.0f, 1.0f, 0.1f, 0.01f);
        lidar.addScan(scan_meta);

        RGBcolor expected_color = make_RGBcolor(0.2f, 0.6f, 0.9f);
        SphericalCoord dir = make_SphericalCoord(1.0f, 1.0f);
        vec3 hit = make_vec3(1,1,1);
        lidar.addHitPoint(0, hit, dir, expected_color);

        try {
            RGBcolor actual_color = lidar.getHitColor(0);
            if (actual_color.r != expected_color.r || actual_color.g != expected_color.g || actual_color.b != expected_color.b) {
                flag = false;
            }
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::coordinateShift()
        std::cout << "Running coordinateShift..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 10.0f, 1.0f, 0.1f, 0.01f);
        lidar.addScan(scan_meta);

        vec3 hit = make_vec3(1,1,1);
        vec3 origin = make_vec3(0,0,0);
        lidar.addHitPoint(0, hit, origin);

        vec3 shift = make_vec3(5,5,5);
        lidar.coordinateShift(shift);
        vec3 new_pos = lidar.getHitXYZ(0);

        if (!(fabs(new_pos.x - 6.0f) < 1e-4 && fabs(new_pos.y - 6.0f) < 1e-4 && fabs(new_pos.z - 6.0f) < 1e-4)) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }


    {
        // Test AerialLiDARcloud::addHitsToVisualizer(visualizer, pointsize)
        std::cout << "Testing addHitsToVisualizer(visualizer, pointsize)..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 10.0f, 1.0f, 0.1f, 0.01f);
        lidar.addScan(scan_meta);

        vec3 hit = make_vec3(1,1,1);
        vec3 origin = make_vec3(0,0,0);
        lidar.addHitPoint(0, hit, origin);

        try {
            Visualizer visualizer_dummy(800, 600);
            lidar.addHitsToVisualizer(&visualizer_dummy, 5);
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed..." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }

    {
        // Test AerialLiDARcloud::addHitsToVisualizer(visualizer, pointsize, color_value)
        std::cout << "Testing addHitsToVisualizer(visualizer, pointsize, color_value)..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata scan_meta(make_vec3(0,0,0), make_vec2(1,1), 10.0f, 1.0f, 0.1f, 0.01f);
        lidar.addScan(scan_meta);

        vec3 hit = make_vec3(1,1,1);
        vec3 origin = make_vec3(0,0,0);
        lidar.addHitPoint(0, hit, origin);
        lidar.setHitData(0, "reflectance", 100.0f);

        try {
            Visualizer visualizer_dummy(800, 600);
            lidar.addHitsToVisualizer(&visualizer_dummy, 5, "reflectance");
        } catch (...) {
            flag = false;
        }

        if( flag ){
            std::cout << "passed..." << std::endl;
        }else{
            std::cout << "failed." << std::endl;
            fail_count ++;
        }
    }


    {
        // Test AerialLiDARcloud::addGridToVisualizer()
        std::cout << "Running addGridToVisualizer..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        try {
            Visualizer vis(800, 600);
            lidar.addGridToVisualizer(&vis); // Should be safe even if empty
        } catch (...) {
            flag = false;
        }

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }

    {
        // Test AerialLiDARcloud::getHitBoundingBox()
        std::cout << "Running getHitBoundingBox..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();
        bool flag = true;

        AerialScanMetadata meta(make_vec3(0,0,0), make_vec2(1,1), 30.0f, 1.0f, 0.1f, 0.01f);
        lidar.addScan(meta);
        lidar.addHitPoint(0, make_vec3(1,2,3), make_vec3(0,0,0));

        vec3 boxmin, boxmax;
        try {
            lidar.getHitBoundingBox(boxmin, boxmax);
            if (boxmin.x > boxmax.x || boxmin.y > boxmax.y || boxmin.z > boxmax.z)
                flag = false;
        } catch (...) {
            flag = false;
        }

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }

    {
        // Test AerialLiDARcloud::getGridExtent()
        std::cout << "Running getGridExtent..." << std::flush;

        AerialLiDARcloud lidar;
        vec3 extent = lidar.getGridExtent();

        bool flag = true;
        if (extent.x != 0 || extent.y != 0 || extent.z != 0)
            flag = false;

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }


    {
        // Test getCellLeafArea()
        std::cout << "Running getCellLeafArea..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();

        Context context;
        context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

        int3 grid_idx = make_int3(0, 0, 0);
        lidar.setCellLeafArea(2.5f, grid_idx);

        bool flag = true;
        try {
            float value = lidar.getCellLeafArea(grid_idx);
            if (fabs(value - 2.5f) > 1e-6) flag = false;
        } catch (...) {
            flag = false;
        }

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }

    {
        // Test getCellTransmissionProbability()
        std::cout << "Running getCellTransmissionProbability..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();

        Context context;
        context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

        int3 grid_idx = make_int3(0, 0, 0);
        lidar.setCellTransmissionProbability(10, 4, grid_idx);

        bool flag = true;
        int denom = -1, trans = -1;
        try {
            lidar.getCellTransmissionProbability(grid_idx, denom, trans);
            if (denom != 10 || trans != 4) flag = false;
        } catch (...) {
            flag = false;
        }

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }



    {
        // Test calculateCoverFraction()
        std::cout << "Running calculateCoverFraction..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();

        Context context;
        context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));  // Just a flat patch for now
        lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

        lidar.calculateHitGridCell();  // Required for grid info

        // Assign 'ground_flag' to all hits
        for (uint i = 0; i < lidar.getHitCount(); ++i) {
            lidar.setHitData(i, "ground_flag", 1.0f);  // Treat all as ground hits
        }

        bool flag = true;
        try {
            lidar.calculateCoverFraction();  // Now won't error on missing ground_flag
        } catch (...) {
            flag = false;
        }

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }


    {
        // Test setCellCoverFraction()
        std::cout << "Running setCellCoverFraction..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();

        Context context;
        context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

        bool flag = true;
        try {
            lidar.calculateHitGridCell(); // Good to ensure grid is ready
            lidar.setCellCoverFraction(0.75f, make_int2(0, 0));
        } catch (...) {
            flag = false;
        }

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }

    {
        // Test getCellCoverFraction()
        std::cout << "Running getCellCoverFraction..." << std::flush;

        AerialLiDARcloud lidar;
        lidar.disableMessages();

        Context context;
        context.addPatch(make_vec3(0, 0, 0), make_vec2(1, 1));
        lidar.syntheticScan(&context, "plugins/aeriallidar/xml/synthetic_aerial_test.xml");

        bool flag = true;
        try {
            lidar.calculateHitGridCell(); // Required for valid cell lookup
            lidar.setCellCoverFraction(0.85f, make_int2(0, 0));
            float val = lidar.getCellCoverFraction(make_int2(0, 0));
            if (fabs(val - 0.85f) > 1e-6) flag = false;
        } catch (...) {
            flag = false;
        }

        if (flag) {
            std::cout << "passed." << std::endl;
        } else {
            std::cout << "failed." << std::endl;
            fail_count++;
        }
    }




    if( fail_count==0 ){
        return 0;
    }else{
        return 1;
    }

}

