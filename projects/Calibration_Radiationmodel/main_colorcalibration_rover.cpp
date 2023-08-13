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

//    std::vector<std::string> sourcelabels = {"solar_spectrum_ASTMG173"}; // Sun intensity spectrum
    std::vector<std::string> sourcelabels = {"CREE_XLamp_XHP70p2_6500K"}; // LED intensity spectra
    std::vector<std::string> cameraresponselabels = {"NikonD700_spectral_response_red","NikonD700_spectral_response_green","NikonD700_spectral_response_blue"};

    std::vector<uint> UUIDg = context.addTile(make_vec3(0, 0, 0), make_vec2(5, 8), nullrotation, make_int2(2, 2));
    context.setPrimitiveData( UUIDg, "twosided_flag", uint(0) );

    CameraCalibration cameracalibration(&context);
    vec3 centrelocation =make_vec3(0,0,0.3); // Location of color board
    vec3 rotationrad =make_vec3(0,0,1.5705); // Rotation angle of color board
    std::vector<uint> UUIDs = cameracalibration.addDefaultColorboard(centrelocation, rotationrad,0.3);

    RadiationModel radiation(&context);
    radiation.disableMessages();

    CameraProperties cameraproperties;
    cameraproperties.HFOV = 23.f;
    cameraproperties.camera_resolution = make_int2(400, 600);
    cameraproperties.focal_plane_distance = 2.2f;
    cameraproperties.lens_diameter = 0.01f;
    cameraproperties.sensor_size = make_vec2(0.04, 0.06);
    vec3 camera_position = make_vec3(0, 0, 3.5);
    std::vector<std::string> bandlabels = {"red","green","blue"};
    vec3 camera_lookat = make_vec3(0, 0, 0);

    // Add 3 LED sources
//    radiation.addSphereRadiationSource(camera_position + make_vec3(0, 1, 0), 0.15 );
//    radiation.addSphereRadiationSource(camera_position + make_vec3(0, -1, 0), 0.15 );
    radiation.addSphereRadiationSource(camera_position + make_vec3(0, 0, 0), 0.25 );

    // Add sun source
//    radiation.addSunSphereRadiationSource(make_SphericalCoord(deg2rad(90-40), deg2rad(40)));

    //Write the color board image using uncalibrated camera responses
    std::string orginalcameralabel = "Camera_t4";
    radiation.addRadiationCamera(orginalcameralabel, bandlabels, camera_position, camera_lookat, cameraproperties,20);


    std::vector<std::vector<float>> truevalues = {{0.8980392,	0.99607843,	0.878431372549,	0,	0.956862745098,	0.72941176470,
                                                          1,	0.16470588235,	0.0039215686,	0.5176470588,	1,	0.9725490196,
                                                          0.1607843137,	0.30980392,	0.462745098,	0.6470588235,	0.82745098,	0.98431372549},
                                                  {0.59607843137,	0.66666667,	0.00392156862745,	0.45490196,	0.6,	0.01176470588,
                                                          0,	0.16470588235,	0.525490196,	0.807843137,	0.98039215686,	0.019607843137,
                                                          0.141176470588,	0.28235294,	0.4156862745,	0.5725490196,	0.75294117647,	0.976470588},
                                                  {0.478431372549,	0.556862745,	0.6039215686,	0.741176470588,	0.0078431372549,	0.12156862745,
                                                          0.56470588235,	0.3607843137,	0.992156862745,	0.270588235294,	0,	0,
                                                          0.16862745098,	0.31372549,	0.462745098,	0.650980392,	0.82745098039,	0.988235294}};

    // Global data mark of calibrated camera spectral responses
    std::string  calibratedmark = "calibrated_t4";

    // Calibrated camera response labels
    std::vector<std::string> cameralabelscal = {calibratedmark + "_" + "NikonD700_spectral_response_red",
                                                calibratedmark + "_" + "NikonD700_spectral_response_green",
                                                calibratedmark + "_" + "NikonD700_spectral_response_blue"};

    radiation.setCameraCalibration(&cameracalibration);
    // Update camera responses
    radiation.calibrateCamera(orginalcameralabel, sourcelabels, cameraresponselabels, bandlabels, 0.4, truevalues, calibratedmark);


    // Write color board image using final calibrated camera responses
    std::string cameralabel3 = "TestCamera_t4";
    radiation.addRadiationCamera(cameralabel3, bandlabels, camera_position, camera_lookat, cameraproperties,20);
    radiation.runRadiationImaging(cameralabel3, sourcelabels, bandlabels, cameralabelscal, wavelengthrange, 1);
    radiation.writeCameraImage(cameralabel3, bandlabels, "Calibrated_t4");

    Visualizer visualizer(400);
    visualizer.buildContextGeometry(&context);
    visualizer.plotInteractive();

    return 0;
}
