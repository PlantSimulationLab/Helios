/** \file "BoundaryLayerConductanceModel.h" Primary header file for boundary layer conductance model.

    Copyright (C) 2016-2024 Brian Bailey

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, version 2.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

*/

#ifndef __BOUNDARYLAYERCONDUCTANCEMODEL__
#define __BOUNDARYLAYERCONDUCTANCEMODEL__

#include "Context.h"

//! Boundary-layer conductance model class
/** This model computes boundary-layer conductance values for a number of objects */
class BLConductanceModel{
public:

  //! Constructor
  /** \param[in] "context" Pointer to the Helios context */
  BLConductanceModel( helios::Context* context );

  //! Self-test
  /** \return 0 if test was successful, 1 if test failed. */
  int selfTest( void );

  //! Enable standard output from this plug-in (default)
  void enableMessages( void );

  //! Disable standard output from this plug-in
  void disableMessages( void );

  //! Set the boundary-layer conductance model to be used for all primitives in the Context
  /** \param[in] "gH_model" 
  */
  void setBoundaryLayerModel( const char* gH_model );

  //! Set the boundary-layer conductance model to be used for a set of primitives
  /** \param[in] "UUID"  Unique universal identifier (UUID) for primitive the conductance model should be used.
      \param[in] "gH_model" 
  */
  void setBoundaryLayerModel( const uint UUID, const char* gH_model );

  //! Set the boundary-layer conductance model to be used for a set of primitives
  /** \param[in] "UUIDs"  Unique universal identifiers (UUIDs) for primitives the conductance model should be used.
      \param[in] "gH_model" 
  */
  void setBoundaryLayerModel(const std::vector<uint> &UUIDs, const char* gH_model );

  //! Run boundary-layer conductance calculations for all primitives in the Context
  void run( void );

  //! Run boundary-layer conductance calculations for a subset of primitives in the Context based on a vector of UUIDs
  void run(const std::vector<uint> &UUIDs );

private:

  //! Copy of a pointer to the context
  helios::Context* context;

  //! Default wind speed if it was not specified in the context
  float wind_speed_default;

  //! Default air temperature if it was not specified in the context
  float air_temperature_default;

  //! Default surface temperature if it was not specified in the context
  float surface_temperature_default;

  bool message_flag;

  float calculateBoundaryLayerConductance( uint gH_model, float U, float L, char Nsides, float inclination, float TL, float Ta );

  std::map<uint,uint> boundarylayer_model;

};

#endif
