/** \file "RadiationModel.h" Primary header file for radiation transport model.
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

#ifndef __RADIATIONMODEL__
#define __RADIATIONMODEL__

#include "Context.h"

//NVIDIA OptiX Includes
#include <optix.h>
#include <optixu/optixu_vector_types.h>
#include <optixu/optixu_vector_functions.h>

//! Radiation transport model plugin
class RadiationModel{
 public:

  //! Default constructor
  RadiationModel( helios::Context* context );

  //! Destructor
  ~RadiationModel();

  //! Self-test
  /** \return 0 if test was successful, 1 if test failed */
  int selfTest(void);

  //! Disable/silence status messages
  /** \note Error messages are still displayed. */
  void disableMessages( void );

  //! Enable status messages
  void enableMessages( void );

  //! Sets variable directRayCount, the number of rays to be used in direct radiation model.
  /** 
      \param[in] "N" Number of rays
  */
  void setDirectRayCount( const char* label, size_t N );

  //! Sets variable diffuseRayCount, the number of rays to be used in diffuse (ambient) radiation model.
  /** 
      \param[in] "N" Number of rays
  */
  void setDiffuseRayCount( const char* label, size_t N );

  //! Diffuse (ambient) radiation flux 
  /** Diffuse component of adiation incident on a horizontal surface above all geometry in the domain. 
      \param[in] "label" Label used to reference the band
      \param[in] "flux" Radiative flux
  */
  void setDiffuseRadiationFlux( const char* label, float flux );

  //! Add a spectral radiation band to the model
  /** Runs radiation calculations for a continuous spectral band (e.g., PAR: 400-700nm)
      \param[in] "label" Label used to reference the band
      \param[in] "run_direct" Sets whether the radiation model should run direct radiation calculations for this band (true=run; false=don't run)
      \param[in] "run_diffuse" Sets whether the radiation model should run diffuse radiation calculations for this band (true=run; false=don't run)
      \param[in] "run_emission" Sets whether the radiation model should run radiative emission calculations for this band (true=run; false=don't run)
      \return Unique integer identifier for the band
  */
  uint addRadiationBand( const char* label );

  //! Disable emission calculations for all primitives in this band.
  /** \param[in] "label" Label used to reference the band
   */
  void disableEmission( const char* label );

  //! Add an external source of collimated radiation (i.e., source at infinite distance with parallel rays) assuming the default direction of (0,0,1)
  /** 
      \return Source identifier
   */
  uint addCollimatedRadiationSource( void );

  //! Add an external source of collimated radiation (i.e., source at infinite distance with parallel rays)
  /** \param[in] "direction" Spherical coordinate pointing toward the radiation source
      \return Source identifier
  */
  uint addCollimatedRadiationSource( const helios::SphericalCoord direction );
  
  //! Add an external source of collimated radiation (i.e., source at infinite distance with parallel rays)
  /** \param[in] "direction" unit vector pointing toward the radiation source
      \return Source identifier
   */
  uint addCollimatedRadiationSource( const helios::vec3 direction );

  //! Add an external source of radiation that emits from the surface of a sphere.
  /** \param[in] "position" (x,y,z) position of the center of the sphere radiation source
      \param[in] "direction" Spherical coordinate pointing in the direction of the sphere normal
      \param[in] "radius" Radius of the sphere radiation source
      \return Source identifier
   */
  uint addSphereRadiationSource( const helios::vec3 position, const float radius );

  //! Add a sphere radiation source that models the sun assuming the default direction of (0,0,1)
  /**       \return Source identifier
  */
  uint addSunSphereRadiationSource( void );

  //! Add a sphere radiation source that models the sun
  /** \param[in] "sun_direction" Spherical coordinate pointing towards the sun 
      \return Source identifier
  */
  uint addSunSphereRadiationSource( const helios::SphericalCoord sun_direction );
  
  //! Add a sphere radiation source that models the sun
  /** \param[in] "sun_direction" Unit vector pointing towards the sun 
      \return Source identifier
  */
  uint addSunSphereRadiationSource( const helios::vec3 sun_direction );
  
  //! Set the flux of radiation source for this band.
  /** \param[in] "ID" Identifier of radiation source
      \param[in] "band_label" Label used to reference the band
      \param[in] "flux" Radiative flux normal to the direction of radiation propagation
  */
  void setSourceFlux( const uint ID, const char* band_label, const float flux );

  //! Set the position/direction of radiation source
  /** \param[in] "ID" Identifier of radiation source
      \param[in] "position" If point source - (x,y,z) position of the radiation source. If collimated source - (nx,ny,nz) unit vector pointing toward the source.
   */
  void setSourcePosition( const uint ID, const helios::vec3 position );

  //! Set the number of scattering iterations for a certain band
  /** \param[in] "label" Label used to reference the band
      \param[in] "depth" Number of scattering iterations (depth=0 turns scattering off)
  */
  void setScatteringDepth( const char* label, uint depth );

  //! Set the energy threshold used to terminate scattering iterations. Scattering iterations are terminated when the maximum to-be-scattered energy among all primitives is less than "energy"
  /** \param[in] "label" Label used to reference the band
      \param[in] "energy" Energy threshold
  */
  void setMinScatterEnergy( const char* label, uint energy );

  //! Use a periodic boundary condition in one or more lateral directions
  /** \param[in] "boundary" Lateral direction to enforce periodic boundary - choices are "x" (periodic only in x-direction), "y" (periodic only in y-direction), or "xy" (periodic in both x- and y-directions).
  */ 
  void enforcePeriodicBoundary( const char* boundary );

  //! Adds all geometric primitives from the Context to OptiX
  /** This function should be called anytime Context geometry is created or modified
      \note \ref updateGeometry() must be called before simulation can be run
  */
  void updateGeometry( void );

  //! Adds certain geometric primitives from the Context to OptiX as specified by a list of UUIDs
  /** This function should be called anytime Context geometry is created or modified
      \param[in] "UUIDs" Vector of universal unique identifiers of Context primitives to be updated
      \note \ref updateGeometry() must be called before simulation can be run
  */
  void updateGeometry( const std::vector<uint> UUIDs );
  
  //! Run the simulation for a radiative band
  /* \param[in] "label" Label used to reference the band (e.g., ``PAR")
     \note Before running the band simulation, you must 1) add at least one radiative band to the simulation (see \ref addRadiationBand()), 2) update the Context geometry in the model (see \ref updateGeometry()), and 3) update radiative properties in the model (see \ref updateRadiativeProperties).
  */
  void runBand( const char* label );

  //! Run the simulation for a radiative band USING TRADITIONAL MONTE CARLO RAY TRACING
  /* \param[in] "label" Label used to reference the band (e.g., ``PAR")
     \note Before running the band simulation, you must 1) add at least one radiative band to the simulation (see \ref addRadiationBand()), 2) update the Context geometry in the model (see \ref updateGeometry()), and 3) update radiative properties in the model (see \ref updateRadiativeProperties).
  */
  void runBand_MCRT( const char* label );

  //! Get the total absorbed radiation flux summed over all bands for each primitive
  std::vector<float> getTotalAbsorbedFlux( void );

  //! Get the radiative energy lost to the sky (surroundings)
  float getSkyEnergy( void );

  //! Calculate G(theta) (i.e., projected area fraction) for a group of primitives given a certain viewing direction
  /** \param[in] "context" Pointer to Helios context
      \param[in] "view_direction" Viewing direction for projected area
      \return Projected area fraction G(theta)
  */
  float calculateGtheta( helios::Context* context, const helios::vec3 view_direction );

protected:

  //! Flag to determine if status messages are output to the screen
  bool message_flag;

  //! Pointer to the context
  helios::Context* context;

  //! UUIDs currently added from the Context
  std::vector<uint> context_UUIDs;

  //! Number of rays to be used in direct radiation model for each band.
  std::vector<size_t> directRayCount;

  //! Default number of rays to be used in direct radiation model.
  size_t directRayCount_default;

  //! Number of rays to be used in diffuse radiation model for each band.
  std::vector<size_t> diffuseRayCount;

  //! Default number of rays to be used in diffuse radiation model.
  size_t diffuseRayCount_default;

  //! Diffuse component of radiation flux for each band
  std::vector<float> diffuseFlux;

  //! Default diffuse radiation flux
  float diffuseFlux_default;

  //! Scattering depth for each band
  std::vector<uint> scatteringDepth;

  //! Default scattering depth
  uint scatteringDepth_default;

  //! Minimum energy for scattering for each band
  std::vector<float> minScatterEnergy;
  
  //! Default minimum energy for scattering
  float minScatterEnergy_default;

  //! Flag that determines if emission calculations are performed for each band
  /** \sa addRadiationBand() */
  std::vector<bool> emission_flag;

  //! Positions of all radiation sources
  std::vector<helios::vec3> source_positions;

  //! Source position factors used to scale position in case of a sun sphere source
  std::vector<float> source_position_scaling_factors;
  
  //! Fluxes for each radiation source over all bands
  std::map<std::string,std::vector<float> > source_fluxes;

  //! Widths for each radiation source (N/A for collimated and point sources)
  std::vector<float> source_widths;

  //! Source flux factors used to scale flux in case of a sun sphere source
  std::vector<float> source_flux_scaling_factors;

  //! Possible types of radiation sources
  enum RadiationSourceType{
    RADIATION_SOURCE_TYPE_COLLIMATED = 0,
    RADIATION_SOURCE_TYPE_SPHERE = 1
  };
  
  //! Types of all radiation sources
  std::vector<RadiationSourceType> source_types;

  //! Names for each radiative band
  /** \sa addRadiationBand() */
  std::map<std::string,uint> band_names;

  //! Radius of a sphere encapsulating the entire scene/domain
  float scene_radius;

  //! Default primitive reflectivity
  float rho_default;

  //! Default primitive transmissivity
  float tau_default;

  //! Default primitive emissivity
  float eps_default;

  //! Default primitive attenuation coefficient
  float kappa_default;

  //! Default primitive scattering coefficient
  float sigmas_default;

  //! Default primitive temperature
  float temperature_default;

  //! Creates OptiX context and creates all associated variables, buffers, geometry, acceleration structures, etc. needed for radiation ray tracing. 
  void initializeOptiX( void );

  //! Sets radiative properties for all primitives
  /** This function should be called anytime primitive radiative properties are modified. If radiative properties were not set in the Context, default radiative properties will be applied (black body).
      \note \ref updateRadiativeProperties() must be called before simulation can be run
  */
  void updateRadiativeProperties( const char* label );

  //! Get 1D array of data for an OptiX buffer of floats
  /** 
      \param[in] "buffer" OptiX buffer object corresponding to 1D array of data 
  */
  std::vector<float> getOptiXbufferData( RTbuffer buffer );

  //! Get 1D array of data for an OptiX buffer of doubles
  /** 
      \param[in] "buffer" OptiX buffer object corresponding to 1D array of data 
  */
  std::vector<double> getOptiXbufferData_d( RTbuffer buffer );

  void addBuffer( const char* name, RTbuffer& buffer, RTvariable& variable, RTbuffertype type, RTformat format, size_t dimension );

  //! Set size of 1D buffer and initialize all elements to zero.
  /** \param[inout] "buffer" OptiX buffer to be initialized. 
      \param[in] "bsize" length of buffer. 
  */
  void zeroBuffer1D( RTbuffer &buffer, const size_t bsize  );

  //! Copy contents of one buffer to another
  /** \param[in] "buffer" OptiX buffer to copy FROM.
      \param[out] "buffer_copy" OptiX buffer to copy TO.
  */
  void copyBuffer1D( RTbuffer &buffer, RTbuffer &buffer_copy );

  //! Set size of 1D buffer and initialize all elements based on a 1D array of doubles.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dd( RTbuffer &buffer, std::vector<double> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of floats.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Df( RTbuffer &buffer, std::vector<float> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type float2.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dfloat2( RTbuffer &buffer, std::vector<optix::float2> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type float3.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dfloat3( RTbuffer &buffer, std::vector<optix::float3> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type float4.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dfloat4( RTbuffer &buffer, std::vector<optix::float4> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type int.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Di( RTbuffer &buffer, std::vector<int> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type unsigned int.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dui( RTbuffer &buffer, std::vector<uint> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type int2.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dint2( RTbuffer &buffer, std::vector<optix::int2> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type int3.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dint3( RTbuffer &buffer, std::vector<optix::int3> array );
  //! Set size of 1D buffer and initialize all elements based on a 1D array of type bool.
   /** \param[inout] "buffer" OptiX buffer to be initialized.
       \param[in] "array" 1D array used to initialize buffer. 
   */
  void initializeBuffer1Dbool( RTbuffer &buffer, std::vector<bool> array );
  //! Set size of 2D buffer and initialize all elements to zero.
  /** \param[inout] "buffer" OptiX buffer to be initialized. 
      \param[in] "size" length of buffer. 
  */
  void zeroBuffer2D( RTbuffer &buffer, const optix::int2 bsize  );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of doubles. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dd( RTbuffer &buffer, std::vector<std::vector<double> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Df( RTbuffer &buffer, std::vector<std::vector<float> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dfloat2( RTbuffer &buffer, std::vector<std::vector<optix::float2> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dfloat3( RTbuffer &buffer, std::vector<std::vector<optix::float3> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dfloat4( RTbuffer &buffer, std::vector<std::vector<optix::float4> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Di( RTbuffer &buffer, std::vector<std::vector<int> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dui( RTbuffer &buffer, std::vector<std::vector<uint> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dint2( RTbuffer &buffer, std::vector<std::vector<optix::int2> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dint3( RTbuffer &buffer, std::vector<std::vector<optix::int3> > array );
  //! Set size of 2D buffer and initialize all elements based on a 2D array of floats. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 2D array used to initialize buffer. 
  */
  void initializeBuffer2Dbool( RTbuffer &buffer, std::vector<std::vector<bool> > array );
  
  //! Set size of 3D buffer and initialize all elements based on a 3D array. 
  /** \param[inout] "buffer" OptiX buffer to be initialized.
      \param[in] "array" 3D array used to initialize buffer. 
  */
  template <typename anytype>
  void initializeBuffer3D( RTbuffer &buffer, std::vector<std::vector<std::vector<anytype> > > array );

  /* Primary RT API objects */

  //! OptiX context object 
  RTcontext OptiX_Context;
  //! OptiX ray generation program handle for direct radiation 
  RTprogram direct_raygen;
  //! OptiX ray generation program handle for diffuse radiation 
  RTprogram diffuse_raygen;
  //! OptiX ray generation program handle for direct radiation (MCRT only)
  RTprogram direct_raygen_MCRT;
  //! OptiX ray generation program handle for diffuse radiation (MCRT only)
  RTprogram diffuse_raygen_MCRT;
  //! OptiX ray generation program handle for radiation emission (MCRT only)
  RTprogram emission_raygen_MCRT;

  /* Variables */

  //! Random number generator seed
  RTvariable random_seed_RTvariable;

  //! Primitive offset used for tiling ray launches
  RTvariable launch_offset_RTvariable;

  //! Maximum scattering depth
  RTvariable max_scatters_RTvariable;
  
  //! Number of external radiation sources
  RTvariable Nsources_RTvariable;

  //! (x,y,z) positions of external radiation sources - RTbuffer object
  RTbuffer source_positions_RTbuffer;
  //! (x,y,z) positions of external radiation sources - RTvariable
  RTvariable source_positions_RTvariable;

  //! Types of radiation sources - RTbuffer object
  RTbuffer source_types_RTbuffer;
  //! Types radiation sources - RTvariable
  RTvariable source_types_RTvariable;

  //! Fluxes of external radiation sources - RTbuffer object
  RTbuffer source_fluxes_RTbuffer;
  //! Fluxes of external radiation sources - RTvariable
  RTvariable source_fluxes_RTvariable;

  //! Widths of external radiation sources - RTbuffer object
  RTbuffer source_widths_RTbuffer;
  //! Widths of external radiation sources - RTvariable
  RTvariable source_widths_RTvariable;

  //! Number of radiative bands
  RTvariable Nbands_RTvariable;

  //! Flux of ambient/diffuse radiation
  RTvariable diffuseFlux_RTvariable;

  //! Bounding sphere radius
  RTvariable bound_sphere_radius_RTvariable;
  //! Bounding sphere center
  RTvariable bound_sphere_center_RTvariable;

  //! Periodic boundary condition
  helios::vec2 periodic_flag;
  RTvariable periodic_flag_RTvariable;

  //! Energy absorbed by the "sky"
  RTvariable Rsky_RTvariable;

  //! Primitive reflectivity - RTbuffer
  RTbuffer rho_RTbuffer;
  //! Primitive reflectivity - RTvariable
  RTvariable rho_RTvariable;
  //! Primitive transmissivity - RTbuffer
  RTbuffer tau_RTbuffer;
  //! Primitive transmissivity - RTvariable
  RTvariable tau_RTvariable;
  //! Primitive emissivity - RTbuffer
  RTbuffer eps_RTbuffer;
  //! Primitive emissivity - RTvariable
  RTvariable eps_RTvariable;

  //! Primitive type - RTbuffer object
  RTbuffer primitive_type_RTbuffer;
  //! Primitive type - RTvariable
  RTvariable primitive_type_RTvariable;

  //! Primitive area - RTbuffer object
  RTbuffer primitive_area_RTbuffer;
  //! Primitive area - RTvariable
  RTvariable primitive_area_RTvariable;

  //! Primitive UUIDs - RTbuffer object
  RTbuffer patch_UUID_RTbuffer;
  RTbuffer triangle_UUID_RTbuffer;
  RTbuffer disk_UUID_RTbuffer;
  RTbuffer alphamask_UUID_RTbuffer;
  RTbuffer voxel_UUID_RTbuffer;
  RTbuffer bbox_UUID_RTbuffer;
  //! Primitive UUIDs - RTvariable object
  RTvariable patch_UUID_RTvariable;
  RTvariable triangle_UUID_RTvariable;
  RTvariable disk_UUID_RTvariable;
  RTvariable alphamask_UUID_RTvariable;
  RTvariable voxel_UUID_RTvariable;
  RTvariable bbox_UUID_RTvariable;

  //! Primitive two-sided flag - RTbuffer object
  RTbuffer twosided_flag_RTbuffer;
  //! Primitive two-sided flag - RTvariable
  RTvariable twosided_flag_RTvariable;

  //! Radiative flux lost to the sky - RTbuffer object
  RTbuffer Rsky_RTbuffer;

  //-- Patch Buffers --//
  RTbuffer patch_vertices_RTbuffer;
  RTvariable patch_vertices_RTvariable;

  //-- Triangle Buffers --//
  RTbuffer triangle_vertices_RTbuffer;
  RTvariable triangle_vertices_RTvariable;

  //-- Disk Buffers --//
  RTbuffer disk_centers_RTbuffer;
  RTvariable disk_centers_RTvariable;
  RTbuffer disk_radii_RTbuffer;
  RTvariable disk_radii_RTvariable;
  RTbuffer disk_normals_RTbuffer;
  RTvariable disk_normals_RTvariable;

  //-- AlphaMask Buffers --//
  RTbuffer alphamask_vertices_RTbuffer;
  RTvariable alphamask_vertices_RTvariable;
  
  //-- Voxel Buffers --//
  RTbuffer voxel_vertices_RTbuffer;
  RTvariable voxel_vertices_RTvariable;

  //-- Bounding Box Buffers --//
  RTbuffer bbox_vertices_RTbuffer;
  RTvariable bbox_vertices_RTvariable;

  /* Output Buffers */

  //! Primitive affine transformation matrix - RTbuffer object
  RTbuffer transform_matrix_RTbuffer;
  //! Primitive affine transformation matrix - RTvariable
  RTvariable transform_matrix_RTvariable;
  //! Primitive temperatures - RTbuffer object
  RTbuffer primitive_emission_RTbuffer;
  //! Primitive temperatures - RTvariable
  RTvariable primitive_emission_RTvariable;

  //! Incoming radiative energy for each object - RTbuffer object 
  RTbuffer radiation_in_RTbuffer;
  //! Incoming radiative energy for each object - RTvariable 
  RTvariable radiation_in_RTvariable;
  //! Outgoing radiative energy (reflected/emitted) for top surface of each object - RTbuffer object 
  RTbuffer radiation_out_top_RTbuffer;
  //! Outgoing radiative energy (reflected/emitted) for top surface each object - RTvariable 
  RTvariable radiation_out_top_RTvariable;
  //! Outgoing radiative energy (reflected/emitted) for bottom surface of each object - RTbuffer object 
  RTbuffer radiation_out_bottom_RTbuffer;
  //! Outgoing radiative energy (reflected/emitted) for bottom surface each object - RTvariable 
  RTvariable radiation_out_bottom_RTvariable;
  //! "to-be-scattered" radiative energy (reflected/emitted) for top surface of each object - RTbuffer object 
  RTbuffer scatter_buff_top_RTbuffer;
  //! "to-be-scattered" radiative energy (reflected/emitted) for top surface each object - RTvariable 
  RTvariable scatter_buff_top_RTvariable;
  //! "to-be-scattered" radiative energy (reflected/emitted) for bottom surface of each object - RTbuffer object 
  RTbuffer scatter_buff_bottom_RTbuffer;
  //! "to-be-scattered" radiative energy (reflected/emitted) for bottom surface each object - RTvariable 
  RTvariable scatter_buff_bottom_RTvariable;

  //! Mask data for AlphaMasks - RTbuffer object 
  RTbuffer maskdata_RTbuffer;
  //! Mask data for AlphaMask - RTvariable 
  RTvariable maskdata_RTvariable;
  //! Size of mask data for AlphaMask - RTbuffer object
  RTbuffer masksize_RTbuffer;
  //! Size of mask data for AlphaMask 
  RTvariable masksize_RTvariable;
  //! ID of mask data (0...Nmasks-1) - RTbuffer object
  RTbuffer maskID_RTbuffer;
  //! ID of mask data (0...Nmasks-1) - RTvariable object 
  RTvariable maskID_RTvariable;
  //! uv data for textures - RTbuffer object 
  RTbuffer uvdata_RTbuffer;
  //! uv data for textures - RTvariable
  RTvariable uvdata_RTvariable;
  //! ID of uv data (0...Nuv-1) - RTbuffer object
  RTbuffer uvID_RTbuffer;
  //! ID of uv data (0...Nuv-1) - RTvariable
  RTvariable uvID_RTvariable;
  

  /* Ray Types */

  //! Handle to OptiX ray type for direct radiation rays. 
  RTvariable direct_ray_type_RTvariable;
  //! Handle to OptiX ray type for diffuse radiation rays. 
  RTvariable diffuse_ray_type_RTvariable;
  //! Handle to OptiX ray type for direct radiation rays (MCRT only)
  RTvariable direct_ray_type_MCRT_RTvariable;
  //! Handle to OptiX ray type for diffuse radiation rays (MCRT only) 
  RTvariable diffuse_ray_type_MCRT_RTvariable;
  //! Handle to OptiX ray type for emission radiation rays (MCRT only)
  RTvariable emission_ray_type_MCRT_RTvariable;

  //! OptiX Ray Types
  enum RayType { RAYTYPE_DIRECT=0, RAYTYPE_DIFFUSE=1, RAYTYPE_DIRECT_MCRT=2, RAYTYPE_DIFFUSE_MCRT=3, RAYTYPE_EMISSION_MCRT=4 };

  /* OptiX Geometry Structures */
  RTgeometry patch;
  RTgeometry triangle;
  RTgeometry disk;
  RTgeometry alphamask;
  RTgeometry voxel;
  RTgeometry bbox;
  RTmaterial patch_material;
  RTmaterial triangle_material;
  RTmaterial disk_material;
  RTmaterial alphamask_material;
  RTmaterial voxel_material;
  RTmaterial bbox_material;

  RTgroup         top_level_group;
  RTacceleration  top_level_acceleration;
  RTvariable      top_object;
  RTacceleration  geometry_acceleration;

  /* OptiX Functions */

  //! Flag indicating whether geometry has been built
  /**
      \sa \ref buildGeometry()
  */
  bool isgeometryinitialized;

  std::vector<bool> isbandpropertyinitialized;

};

void sutilHandleError(RTcontext context, RTresult code, const char* file, int line);

void sutilReportError(const char* message);

/* assumes current scope has Context variable named 'OptiX_Context' */
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      sutilHandleError( OptiX_Context, code, __FILE__, __LINE__ );       \
  } while(0)

#endif
