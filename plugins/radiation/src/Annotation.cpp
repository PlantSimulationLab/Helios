#include "Annotation.h"
using namespace helios;

Annotation::Annotation(helios::Context *context, const std::string &cameralabel):context(context){
    //Get image UUID labels
    std::vector<uint> camera_UUIDs;
    std::string global_data_label = "camera_" + cameralabel + "_pixel_UUID";
    context->getGlobalData(global_data_label.c_str(), camera_UUIDs);
    pixel_UUIDs = camera_UUIDs;
}

void Annotation::setCameraResolution(const helios::int2 &camera_resolution){
    this->camera_resolution = camera_resolution;
}

void Annotation::getBasicLabel(const std::string &filename, const std::string &labelname, HeliosDataType datatype, float padvalue){

    //Output label image in ".txt" format
    std::ofstream pixel_data(filename);

    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint UUID =pixel_UUIDs.at(j * camera_resolution.x + i)-1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID,labelname.c_str())){

                if( datatype == HELIOS_TYPE_FLOAT ){
                    float labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
                else if (datatype == HELIOS_TYPE_UINT){
                    uint labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
                else if (datatype == HELIOS_TYPE_INT){
                    int labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
                else if (datatype == HELIOS_TYPE_DOUBLE){
                    double labeldata;
                    context->getPrimitiveData(UUID,labelname.c_str(),labeldata);
                    pixel_data << labeldata << " ";
                }
            }
            else{
                pixel_data << padvalue << " ";
            }
        }
        pixel_data << "\n";
    }
    pixel_data.close();
}

float runNetPhotosynthesisKernal( float sourceflux, float Vcmax, float Jmax, float Ci, float Kc, float Ko, float Oi, float Gamma_star,float Rd, float alpha){

    float Kco = Kc*(1+Oi/Ko);
    float Wc = Vcmax*Ci/(Ci+Kco);

    float J = sourceflux*alpha/sqrt(1+pow(sourceflux*alpha/Jmax,2));
    float Wj = J/4*Ci/(Ci+2*Gamma_star);
    float W;
    if (Wc<Wj){
        W = Wc;
    }
    else{
        W = Wj;
    }

    float A = W*(1-Gamma_star/Ci)-Rd;
    return A;
}

float Annotation::runNetPhotosynthesis( float sourceflux, Annotation::PhotosynthesisParameters photosynthesisParameters){

    float A = runNetPhotosynthesisKernal(sourceflux, photosynthesisParameters.Vcmax, photosynthesisParameters.Jmax, photosynthesisParameters.Ci,
                                         photosynthesisParameters.Kc, photosynthesisParameters.Ko, photosynthesisParameters.Oi,
                                         photosynthesisParameters.Gamma_star, photosynthesisParameters.Rd, photosynthesisParameters.alpha);
    return A;
}

float Annotation::runNetPhotosynthesis( float sourceflux, float chlorophyllcontent, Annotation::PhotosynthesisParameters photosynthesisParameters){

    float avc = photosynthesisParameters.avc;
    float bvc = photosynthesisParameters.bvc;
    float Vcmax25 = avc*chlorophyllcontent+bvc;

    float ajv = photosynthesisParameters.ajv;
    float bjv = photosynthesisParameters.bjv;

    float Jmax25 = pow(e, ajv * log(Vcmax25)+bjv);
    float Vcmax = Vcmax25 * photosynthesisParameters.dVcmax;
    float Jmax = Jmax25 * photosynthesisParameters.dJmax;

    float A = runNetPhotosynthesisKernal(sourceflux, Vcmax, Jmax, photosynthesisParameters.Ci,
                                         photosynthesisParameters.Kc, photosynthesisParameters.Ko, photosynthesisParameters.Oi,
                                         photosynthesisParameters.Gamma_star, photosynthesisParameters.Rd, photosynthesisParameters.alpha);
    return A;
}


void Annotation::getNetPhotosynthesis(const std::string &filename, const std::string &leaflabel, const std::string &chllabel, float sourcefluxscale){

    std::ofstream pixel_data_w(filename);
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint UUID =pixel_UUIDs.at(j * camera_resolution.x + i)-1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID,leaflabel.c_str()) && context->doesPrimitiveDataExist(UUID,chllabel.c_str())){

                float sourceflux;
                context->getPrimitiveData(UUID,"radiation_flux_PAR",sourceflux);
                float chl_content;
                context->getPrimitiveData(UUID, chllabel.c_str(), chl_content);
                float sourcefluxu = sourcefluxscale*sourceflux/0.21739130434f;

                float netphotosynthesis = runNetPhotosynthesis(sourcefluxu,chl_content, photosynthesisParameters);
                pixel_data_w << netphotosynthesis << " ";
            }
            else{
                pixel_data_w << 0 << " ";
            }
        }
        pixel_data_w << "\n";
    }
    pixel_data_w.close();
}

void Annotation::getNetPhotosynthesis(const std::string &filename, const std::string &leaflabel, float sourcefluxscale){

    std::cout << "Performing annotation for net photosynthesis...";
    std::ofstream pixel_data_w(filename);
    for (uint j = 0; j < camera_resolution.y; j++) {
        for (uint i = 0; i < camera_resolution.x; i++) {
            uint UUID =pixel_UUIDs.at(j * camera_resolution.x + i)-1;
            if (context->doesPrimitiveExist(UUID) && context->doesPrimitiveDataExist(UUID,leaflabel.c_str())){

                float sourceflux;
                context->getPrimitiveData(UUID,"radiation_flux_PAR",sourceflux);
                float sourcefluxu = sourcefluxscale*sourceflux/0.21739130434f;

                float netphotosynthesis = runNetPhotosynthesis(sourcefluxu, photosynthesisParameters);
                pixel_data_w << netphotosynthesis << " ";
            }
            else{
                pixel_data_w << 0 << " ";
            }
        }
        pixel_data_w << "\n";
    }
    pixel_data_w.close();
    std::cout<<"done."<<std::endl;
}

void Annotation::getDepthImage(const std::string &filename, const helios::vec3 &camera_position, const helios::vec3 &camera_lookat) {

    // Set depth for all primitives
    float maxdepth = 0;
    float mindepth = -1;
    for( uint UUIDr : pixel_UUIDs ){
        uint UUID = UUIDr-1;
        if (context->doesPrimitiveExist(UUID)) {
            std::vector<vec3> vertices = context->getPrimitiveVertices(UUID);//Centre
            vec3 vertex = vertices.at(0) - camera_position;
            vec3 cameraplane_direct = camera_lookat - camera_position;
            float dotproductvc = vertex * cameraplane_direct;
            float cam_norm = cameraplane_direct.magnitude();
            float depth = dotproductvc / cam_norm;
            if (depth > maxdepth) {
                maxdepth = depth;
            }
            if (mindepth == -1) {
                mindepth = depth;
            } else if (depth < mindepth) {
                mindepth = depth;
            }
            context->setPrimitiveData(UUID, "depth", depth);
        }
    }

    for (uint UUIDr : pixel_UUIDs ){
        uint UUID = UUIDr-1;
        if (context->doesPrimitiveExist(UUID)){
            float depth;
            context->getPrimitiveData(UUID,"depth",depth);
            float depth_norm = (depth-mindepth)/(maxdepth-mindepth);
            context->setPrimitiveData(UUID,"depth_norm",depth_norm);
        }
    }

    // Output depth image in ".txt" format
    Annotation::getBasicLabel(filename, "depth_norm", HELIOS_TYPE_FLOAT,1);
}
