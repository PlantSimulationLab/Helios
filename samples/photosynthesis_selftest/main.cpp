#include "PhotosynthesisModel.h"

using namespace helios;

int main() {
    Context context;
    PhotosynthesisModel photosynthesis(&context);

    return photosynthesis.selfTest();
}
