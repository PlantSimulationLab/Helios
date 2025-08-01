#include "SyntheticAnnotation.h"

using namespace helios;

int main(int argc, char** argv) {
    Context context;
    SyntheticAnnotation syntheticannotation(&context);
    return syntheticannotation.selfTest(argc, argv);
}
