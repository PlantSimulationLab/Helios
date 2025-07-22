#include "WeberPennTree.h"

using namespace helios;

int main() {

    Context context;

    WeberPennTree wpt(&context);

    return wpt.selfTest();
}
