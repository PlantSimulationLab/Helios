/** \file "SolarPosition.h" Primary header file for solar position model plug-in.
    \author Brian Bailey

    Copyright (C) 2018  Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef __SOLARPOSITION__
#define __SOLARPOSITION__

#include "Context.h"

class SolarPosition {
   public:
    //! Solar position model default constructor
    /** \param[in] "context" Pointer to the Helios context
     */
    SolarPosition(helios::Context* context);

    //! Solar position model constructor
    /** \param[in] "context" Pointer to the Helios context
       \param[in] "UTC" Hours from Coordinated Universal Time (UTC) for location of interest.  Convention is that UTC is
       positive moving Westward. \param[in] "latitude" Latitude in degrees for location of interest.  Convention is
       latitude is positive for Northern hemisphere. \param[in] "longitude" Longitude in degrees for location of
       interest. Convention is longitude is positive for Western hemisphere.
    */
    SolarPosition(int UTC, float latitude, float longitude, helios::Context* context);

    //! Function to perform a self-test of model functions
    int selfTest(void) const;

    //! Get the approximate time of sunrise at the current location
    helios::Time getSunriseTime(void) const;

    //! Get the approximate time of sunset at the current location
    helios::Time getSunsetTime(void) const;

    //! Get the current sun elevation angle in radians for the current location. The sun angle is computed based on the
    //! current time and date set in the Helios context
    float getSunElevation(void);

    //! Get the current sun zenithal angle in radians for the current location. The sun angle is computed based on the
    //! current time and date set in the Helios context
    float getSunZenith(void);

    //! Get the current sun azimuthal angle in radians for the current location. The sun angle is computed based on the
    //! current time and date set in the Helios context
    float getSunAzimuth(void);

    //! Get a unit vector pointing toward the sun for the current location. The sun angle is computed based on the
    //! current time and date set in the Helios context
    helios::vec3 getSunDirectionVector(void);

    //! Get a spherical coordinate vector pointing toward the sun for the current location. The sun angle is computed
    //! based on the current time and date set in the Helios context
    helios::SphericalCoord getSunDirectionSpherical(void);

    //! Get the solar radiation flux perpendicular to the sun direction.
    /** \param[in] "pressure" Atmospheric pressure near ground surface in Pascals
       \param[in] "temperature" Air temperature near the ground surface in Kelvin
       \param[in] "humidity" Air relative humidity near the ground surface
       \param[in] "turbidity" Angstrom's aerosol turbidity coefficient
       \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal
       surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith()
       function. \note The solar flux model is based on <a
       href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>. \return Global
       solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
    */
    float getSolarFlux(const float pressure, const float temperature, const float humidity, const float turbidity);

    //! Get the photosynthetically active (PAR) component of solar radiation flux perpendicular to the sun direction.
    /** \param[in] "pressure" Atmospheric pressure near ground surface in Pascals
       \param[in] "temperature" Air temperature near the ground surface in Kelvin
       \param[in] "humidity" Air relative humidity near the ground surface
       \param[in] "turbidity" Angstrom's aerosol turbidity coefficient
       \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal
       surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith()
       function. \note The solar flux model is based on <a
       href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>. \return Global
       solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
    */
    float getSolarFluxPAR(const float pressure, const float temperature, const float humidity, const float turbidity);

    //! Get the near-infrared (NIR) component of solar radiation flux perpendicular to the sun direction.
    /** \param[in] "pressure" Atmospheric pressure near ground surface in Pascals
       \param[in] "temperature" Air temperature near the ground surface in Kelvin
       \param[in] "humidity" Air relative humidity near the ground surface
       \param[in] "turbidity" Angstrom's aerosol turbidity coefficient
       \note The flux given by this function is the flux normal to the sun direction. To get the flux on a horizontal
       surface, multiply the returned value by cos(theta), where theta can be found by calling the \ref getSunZenith()
       function. \note The solar flux model is based on <a
       href="http://www.sciencedirect.com/science/article/pii/S0038092X07000990">Gueymard (2008)</a>. \return Global
       solar radiation flux NORMAL TO THE SUN DIRECTION in W/m^2
    */
    float getSolarFluxNIR(const float pressure, const float temperature, const float humidity, const float turbidity);

    //! Get the fraction of solar radiation flux that is diffuse
    /** \param[in] "pressure" Atmospheric pressure near ground surface in Pascals
       \param[in] "temperature" Air temperature near the ground surface in Kelvin
       \param[in] "humidity" Air relative humidity near the ground surface
       \param[in] "turbidity" Angstrom's aerosol turbidity coefficient
       \return Fraction of global radiation that is diffuse
    */
    float getDiffuseFraction(const float pressure, const float temperature, const float humidity,
                             const float turbidity);

    //! Calculate the ambient (sky) longwave radiation flux
    /** \param[in] "temperature" Air temperature near the ground surface in Kelvin
       \param[in] "humidity" Air relative humidity near the ground surface
       \return Ambient longwave flux in W/m^2
       \note The longwave flux model is based on <a
       href="http://onlinelibrary.wiley.com/doi/10.1002/qj.49712253306/full">Prata (1996)</a>.
    */
    float getAmbientLongwaveFlux(const float temperature, const float humidity);

   private:
    helios::Context* context;

    int UTC;
    float latitude;
    float longitude;

    helios::SphericalCoord calculateSunDirection(helios::Time time, helios::Date date) const;

    void GueymardSolarModel(const float pressure, const float temperature, const float humidity, const float turbidity,
                            float& Eb_PAR, float& Eb_NIR, float& fdiff);
};

#endif
