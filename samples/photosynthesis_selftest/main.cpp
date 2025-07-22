#include "PhotosynthesisModel.h"

using namespace helios;

int main() {
    Context context;

    PhotosynthesisModel photo(&context);

    return photo.selfTest();
}
