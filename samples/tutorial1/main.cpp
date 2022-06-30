#include "Context.h"

using namespace helios;  // note that we are using the helios namespace so we can omit 'helios::' before names

int main() {
    //------- vec3 ------- //

    // Declare our vec3, named 'a' and set its values
    vec3 a = make_vec3(0, 0.1, 0.2);

    // Declare another vec3, named 'b' and set its values
    vec3 b = make_vec3(1.5, 1.4, 1.3);

    // Add a and b, and assign it to 'c'
    vec3 c = a + b;  // result is c = (1.5,1.5,1.5)

    // Normalize 'c' to have unit length
    c.normalize();  // result is c = (0.577,0.577,0.577)

    // Compute the dot product of a and b, and assign it to 'd'
    float d = a * b;  // result is d = 0.4

    //------ RGBcolor/RGBAcolor ------- //

    RGBcolor color(1, 0, 0);  // red color (opaque)

    RGBAcolor color_t(1, 0, 0, 0.5);  // red color (semi-transparent)

    color = make_RGBcolor(0, 0, 1);  // change color to blue

    //------ Time ------- //

    Time time = make_Time(12, 30, 00);  // time of 12:30:00

    //------ Date ------- //

    Date date = make_Date(1, 1, 2000);  // Jan 1, 2000

    // Convert to Julian day
    int JD = date.JulianDay();

    // Convert Julian day back to Date
    date = Julian2Calendar(JD, 2000);

    return 0;
}
