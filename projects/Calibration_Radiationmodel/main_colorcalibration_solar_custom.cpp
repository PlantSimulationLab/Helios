#include "Context.h"
#include "Visualizer.h"
#include "RadiationModel.h"
#include "CanopyGenerator.h"
using namespace helios;

int main(void){
    Context context;

    vec2 wavelengthrange = make_vec2(300,2500); // Wavelength range for imaging (to ensure the correct source flux)

    context.loadXML("plugins/radiation/spectral_data/camera_spectral_library.xml" );
    context.loadXML("plugins/radiation/spectral_data/light_spectral_library.xml" );
    context.loadXML("plugins/radiation/spectral_data/solar_spectrum_ASTMG173.xml" );

    std::vector<std::string> sourcelabels = {"solar_spectrum_ASTMG173"}; // Sun intensity spectrum
//    std::vector<std::string> sourcelabels = {"CREE_XLamp_XHP70p2_6500K","CREE_XLamp_XHP70p2_6500K","CREE_XLamp_XHP70p2_6500K"}; // LED intensity spectra
    std::vector<std::string> cameraresponselabels = {"NikonD700_spectral_response_red","NikonD700_spectral_response_green","NikonD700_spectral_response_blue"};

    RadiationModel radiation(&context);
    radiation.disableMessages();

    CameraProperties cameraproperties;
    cameraproperties.HFOV = 10.f;
    cameraproperties.camera_resolution = make_int2(400, 600);
    cameraproperties.focal_plane_distance = 1.8f;
    cameraproperties.lens_diameter = 0.01f;
    cameraproperties.sensor_size = make_vec2(0.04, 0.06);
    vec3 camera_position = make_vec3(0, 0, 2.f);
    std::vector<std::string> bandlabels = {"red","green","blue"};
    vec3 camera_lookat = make_vec3(0, 0, 0);

    // Add sun source
    radiation.addSunSphereRadiationSource(make_SphericalCoord(deg2rad(90-50), deg2rad(40)));

    //Write the color board image using uncalibrated camera responses
    std::string orginalcameralabel = "Nikon_solar_raw";
    radiation.addRadiationCamera(orginalcameralabel, bandlabels, camera_position, camera_lookat, cameraproperties,20);

    CameraCalibration cameracalibration(&context);

    // Size and RGB values (used for virtualization) of default color board
    std::vector<std::vector<helios::RGBcolor>> colorassignment_default=
            {{helios::make_RGBcolor(0.161,0.141,0.169), helios::make_RGBcolor(1,0,0.5647), helios::make_RGBcolor(0.898,0.596,0.4784)},
             {helios::make_RGBcolor(0.31,0.282,0.314), helios::make_RGBcolor(0.1882,0.2314,0.3608),helios::make_RGBcolor(0.9961,0.6667,0.5569)},
             {helios::make_RGBcolor(0.463,0.416,0.463), helios::make_RGBcolor(0,0.5647,1), helios::make_RGBcolor(0.8784,0.0039,0.6039)},
             {helios::make_RGBcolor(0.647,0.573,0.651), helios::make_RGBcolor(0.4118,0.7490,0.2706),helios::make_RGBcolor(0,0.4549,0.7412)},
             {helios::make_RGBcolor(0.828,0.573,0.651), helios::make_RGBcolor(1,1,0), helios::make_RGBcolor(0.9569,0.6,0.0078)},
             {helios::make_RGBcolor(0.984,0.977,0.988), helios::make_RGBcolor(1,0.1255,0), helios::make_RGBcolor(0.7294,0.0118,0.1216)}};
    float patchsize = 0.1;
    vec3 centrelocation =make_vec3(0,0,0.2); // Location of color board
    vec3 rotationrad =make_vec3(0,0,1.5705); // Rotation angle of color board

    std::vector<uint> UUIDs_colorboard = cameracalibration.addColorboard(centrelocation, rotationrad,colorassignment_default,patchsize);

    uint colornumber=colorassignment_default.size()*colorassignment_default.at(0).size();

    std::string numberstr;
    std::string filename;
    std::string labelname;

    // Assign spectra to color board
    for (uint UUID:UUIDs_colorboard){
        numberstr = std::to_string(colornumber);
        if (numberstr.size() < 2) {
            numberstr = "0" + numberstr;
        }
        filename= "plugins/radiation/spectral_data/color_board/ColorReference_"+numberstr+".xml";
        labelname= "ColorReference_"+numberstr;
        cameracalibration.setColorboardReflectivity(UUID, filename, labelname);
        colornumber=colornumber-1;
    }
    radiation.setCameraCalibration(&cameracalibration);
    std::vector<std::vector<float>> truevalues(3);

// red
    truevalues.at(0) = {0.733333, 0.801307, 0.811764, 0.116339, 0.843137, 0.788235, 0.943790, 0.175163, 0.005228, 0.448366, 0.930718, 0.911111, 0.124183, 0.288888, 0.450980, 0.606535, 0.749019, 0.900653};
//green
    truevalues.at(1) = {0.588235, 0.635294, 0.317647, 0.603921, 0.490196, 0.0862745, 0.385621, 0.270588, 0.756863, 0.745098, 0.857516, 0.257516, 0.12549, 0.322876, 0.488889, 0.644444, 0.766013, 0.912418};
//blue
    truevalues.at(2) = {0.482353, 0.575163, 0.605229, 0.840523, 0.0196078, 0.130719, 0.593464, 0.487582, 0.998693, 0.401307, 0.303268, 0.197386, 0.145098, 0.339869, 0.50719, 0.665359, 0.781699, 0.917647};

    // Global data mark of calibrated camera spectral responses
    std::string  calibratedmark = "calibrated_sun";

    // Calibrated camera response labels
    std::vector<std::string> cameralabelscal = {calibratedmark + "_" + "NikonD700_spectral_response_red",
                                                calibratedmark + "_" + "NikonD700_spectral_response_green",
                                                calibratedmark + "_" + "NikonD700_spectral_response_blue"};

    // Update camera responses
    radiation.calibrateCamera(orginalcameralabel, sourcelabels, cameraresponselabels, bandlabels, 1, truevalues, calibratedmark);

    // Write color board image using final calibrated camera responses
    std::string cameralabel3 = "Nikon_solar_cal";
    radiation.addRadiationCamera(cameralabel3, bandlabels, camera_position, camera_lookat, cameraproperties,20);
    radiation.runRadiationImaging(cameralabel3, sourcelabels, bandlabels, cameralabelscal, wavelengthrange, 0.8,0);
    radiation.writeCameraImage(cameralabel3, bandlabels, "Calibrated_nikon");

    Visualizer visualizer(400);
    visualizer.buildContextGeometry(&context);
    visualizer.plotInteractive();

    return 0;
}
