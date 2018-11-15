/** \file "DummyModel.h" Primary header file for dummy model.
    \author Brian Bailey
*/

#include "Context.h"

//! Dummy model class
/** This model simply adds a variable to the context and sets it value for all primitives */
class DummyModel{
public:

  //! Constructor
  DummyModel( helios::Context* context );

  //! Function to run the dummy model
  void run( void );

private:

  //! Copy of a pointer to the context
  helios::Context* context;

};
