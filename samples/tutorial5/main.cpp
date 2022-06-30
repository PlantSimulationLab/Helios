#include "Context.h"

using namespace helios;  // note that we are using the helios namespace so we can omit 'helios::' before names

int main() {
    Context context;  // Declare the "Context" class

    vec3 center(0, 0, 0);  //(x,y,z) position of sphere center
    float r = 1;           // radius of sphere

    std::vector<uint> UUIDs;  // vector to store UUIDs of sphere

    UUIDs = context.addSphere(10, center, r);  // add a sphere to the Context

    context.setPrimitiveData(UUIDs.at(0), "my_data",
                             10.0);  // add primitive data called my_data to only the first primitive with a value of 10

    return 1;
}
