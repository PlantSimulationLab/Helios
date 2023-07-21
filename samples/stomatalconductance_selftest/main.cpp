#include "StomatalConductanceModel.h"

using namespace helios;

int main(){

    Context context;

    StomatalConductanceModel gs(&context);

    return gs.selfTest();

}
