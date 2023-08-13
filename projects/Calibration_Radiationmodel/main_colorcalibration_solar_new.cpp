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
    std::vector<std::string> sourcelabels_led = {"CREE_XLamp_XHP70p2_6500K","CREE_XLamp_XHP70p2_6500K","CREE_XLamp_XHP70p2_6500K"}; // LED intensity spectra
    std::vector<std::string> cameraresponselabels = {"NikonD700_spectral_response_red","NikonD700_spectral_response_green","NikonD700_spectral_response_blue"};




    RadiationModel radiation(&context);
    radiation.disableMessages();

    std::vector<vec2> solorspectest;
    context.getGlobalData(sourcelabels.at(0).c_str(),solorspectest);
    float testsun = radiation.integrateSpectrum(solorspectest,300,800);
    std::cout<< 1/testsun << std::endl;

    std::vector<vec2> ledspectest;
    context.getGlobalData(sourcelabels_led.at(0).c_str(),ledspectest);
    float testled = radiation.integrateSpectrum(ledspectest,300,800);
    std::cout<< 1/testled << std::endl;

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

    float sources_fluxsum = 0;
    std::vector<float> sources_fluxes;
    for (uint ID = 0; ID < sourcelabels.size(); ID++){
        std::vector<vec2> Source_spectrum;
        context.getGlobalData(sourcelabels.at(ID).c_str(), Source_spectrum);
        sources_fluxes.push_back(radiation.integrateSpectrum(Source_spectrum, wavelengthrange.x, wavelengthrange.y));
        radiation.setSourceSpectrum(ID, sourcelabels.at(ID).c_str());
        radiation.setSourceSpectrumIntegral(ID, sources_fluxes.at(ID));
        sources_fluxsum += sources_fluxes.at(ID);
    }

    radiation.addRadiationBand(bandlabels.at(0), wavelengthrange.x, wavelengthrange.y);
    radiation.disableEmission(bandlabels.at(0));
    for( uint ID =0; ID<sourcelabels.size();ID++ ) {
        radiation.setSourceFlux(ID, bandlabels.at(0), (1 - 0) * sources_fluxes.at(ID) * 0.035);
    }
    radiation.setScatteringDepth(bandlabels.at(0), 4);
    radiation.setDiffuseRadiationFlux(bandlabels.at(0), 0 * sources_fluxsum );
    radiation.setDiffuseRadiationExtinctionCoeff(bandlabels.at(0), 1.f, make_vec3(-0.5, 0.5, 1) );

    if (bandlabels.size()>1){
        for (int iband=1;iband<bandlabels.size();iband++){
            radiation.copyRadiationBand(bandlabels.at(iband-1), bandlabels.at(iband), wavelengthrange.x, wavelengthrange.y);
            for( uint ID =0; ID<sourcelabels.size();ID++ ) {
                radiation.setSourceFlux(ID, bandlabels.at(iband), (1 - 0) * sources_fluxes.at(ID) * 0.035);
            }
            radiation.setDiffuseRadiationFlux(bandlabels.at(iband), 0 * sources_fluxsum );
        }
    }

    for (int iband=0;iband<bandlabels.size();iband++){
        radiation.setCameraSpectralResponse(orginalcameralabel, bandlabels.at(iband), cameraresponselabels.at(iband));
    }



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
    radiation.calibrateCamera(orginalcameralabel, 1, truevalues, calibratedmark);

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
