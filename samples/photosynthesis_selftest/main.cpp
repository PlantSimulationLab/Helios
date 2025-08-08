#include "PhotosynthesisModel.h"

using namespace helios;

int main(int argc, char** argv) {
    Context context;
    PhotosynthesisModel photo(&context);
    return photo.selfTest(argc, argv);
}
