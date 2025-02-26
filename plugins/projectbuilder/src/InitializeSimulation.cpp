#include "InitializeSimulation.h"

using namespace helios;

void InitializeSimulation(const std::string &xml_input_file, helios::Context *context_ptr ){

    pugi::xml_document xmldoc;

    std::string xml_error_string;
    if( !open_xml_file(xml_input_file, xmldoc, xml_error_string) ) {
        helios_runtime_error(xml_error_string);
    }

    pugi::xml_node helios = xmldoc.child("helios");
    pugi::xml_node node;

    // ####### SET UP THE LOCATION ####### //

    Location location;

    float latitude;
    node = helios.child("latitude");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'latitude'. Using default value of " << location.latitude_deg << std::endl;
    }else {

        const char *latitude_str = node.child_value();
        if (!parse_float(latitude_str, latitude)) {
            helios_runtime_error("ERROR: Value given for 'latitude' could not be parsed.");
        }else{
            location.latitude_deg = latitude;
        }

    }

    float longitude;
    node = helios.child("longitude");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'longitude'. Using default value of " << location.longitude_deg << std::endl;
    }else {

        const char *longitude_str = node.child_value();
        if (!parse_float(longitude_str, longitude)) {
            helios_runtime_error("ERROR: Value given for 'longitude' could not be parsed.");
        }else{
            location.longitude_deg = longitude;
        }

    }

    float UTC_offset;
    node = helios.child("UTC_offset");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'UTC_offset'. Using default value of " << location.UTC_offset << std::endl;
    }else {

        const char *UTC_offset_str = node.child_value();
        if (!parse_float(UTC_offset_str, UTC_offset)) {
            helios_runtime_error("ERROR: Value given for 'UTC_offset' could not be parsed.");
        }else{
            location.UTC_offset = UTC_offset;
        }

    }

    context_ptr->setLocation(location);

    /*

    // ####### BUILDING THE DATE AND TIME ####### //

    Date date;

    int3 calendar_date;
    bool calendar_date_read = false;
    node = helios.child("calendar_date");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'calendar_date'. Using default value of " << date << std::endl;
    }else {

        const char *calendar_date_str = node.child_value();
        if (!parse_int3(calendar_date_str, calendar_date)) {
            helios_runtime_error("ERROR: Value given for 'calendar_date' could not be parsed.");
        }else{
            if( calendar_date.x < 1 || calendar_date.x > 31 || calendar_date.y < 1 || calendar_date.y > 12 || calendar_date.z < 1000 ) {
                helios_runtime_error("ERROR: Value given for 'calendar_date' is out of range.");
            }
            date.day = calendar_date.x;
            date.month = calendar_date.y;
            date.year = calendar_date.z;
            calendar_date_read = true;
        }

    }

    int2 julian_date;
    bool julian_date_read = false;
    node = helios.child("julian_date");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'julian_date'. Using default value of <" << date.JulianDay() << "," << date.year << ">" << std::endl;
    }else {

        const char *julian_date_str = node.child_value();
        if (!parse_int2(julian_date_str, julian_date)) {
            helios_runtime_error("ERROR: Value given for 'julian_date' could not be parsed.");
        }else{
            if( julian_date.x < 1 || julian_date.x > 366 || julian_date.y < 1000 ) {
                helios_runtime_error("ERROR: Value given for 'julian_date' is out of range.");
            }
            date = make_Date(julian_date.x,julian_date.y);
            julian_date_read = true;
        }

    }

    if( calendar_date_read && julian_date_read ) {
        std::cout << "WARNING: Both 'calendar_date' and 'julian_date' were given. Using the Juilian date provided." << std::endl;
    }

    context_ptr->setDate(date);

    Time time;

    int3 time3;
    bool time_read = false;
    node = helios.child("time");
    if( node.empty() ){
        std::cout << "WARNING: No value given for 'time'. Using default value of " << time << std::endl;
    }else {

        const char *time_str = node.child_value();
        if (!parse_int3(time_str, time3)) {
            helios_runtime_error("ERROR: Value given for 'time' could not be parsed.");
        }else{
            if( time3.x < 0 || time3.x > 59 || time3.y < 0 || time3.y > 59 || time3.z < 0 || time3.z > 23 ) {
                helios_runtime_error("ERROR: Value given for 'time' is out of range.");
            }
            time.second = time3.x;
            time.minute = time3.y;
            time.hour = time3.z;
            time_read = true;
        }

    }

    context_ptr->setTime(time);

     */

    // ####### READ IN WEATHER DATA ####### //

    std::string weather_data_file;
    node = helios.child("csv_weather_file");
    if( !node.empty() ){

        const char *weather_data_file_str = node.child_value();
        weather_data_file = trim_whitespace(std::string(weather_data_file_str));

        if( weather_data_file.empty() ){
            helios_runtime_error("ERROR: Value given for 'weather_data_file' is empty.");
        }else if( !std::filesystem::exists(weather_data_file) ){
            helios_runtime_error("ERROR: File given for 'weather_data_file' does not exist.");
        }

        context_ptr->loadTabularTimeseriesData(weather_data_file, {}, ",", "YYYYMMDD", 1 );

    }

    std::string cimis_data_file;
    node = helios.child("cimis_weather_file");
    if (!node.empty()) {

        if( !weather_data_file.empty() ){
            std::cout << "WARNING: Both 'csv_weather_file' and 'cimis_weather_file' were given, but only one weather data file can be loaded for the simulation. Using the CSV weather data file." << std::endl;
        }else{

            const char *cimis_data_file_str = node.child_value();
            cimis_data_file = trim_whitespace(std::string(cimis_data_file_str));

            if (cimis_data_file.empty()) {
                helios_runtime_error("ERROR: Value given for 'cimis_data_file' is empty.");
            } else if (!std::filesystem::exists(cimis_data_file)) {
                helios_runtime_error("ERROR: File given for 'cimis_data_file' does not exist.");
            }

            context_ptr->loadTabularTimeseriesData(cimis_data_file, {"CIMIS"}, ",");

        }

    }
}