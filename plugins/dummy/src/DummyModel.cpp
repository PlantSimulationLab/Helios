/** \file "DummyModel.cpp" Dummy model plugin declarations. 
    \author Brian Bailey
*/

#include "DummyModel.h"

DummyModel::DummyModel( helios::Context* __context ){
  std::cout << "Initializing dummy model..." << std::flush;

  context = __context; //just copying the pointer to the context

  std::cout << "done." << std::endl;
}

void DummyModel::run( void ){

  std::cout << "Running dummy model..." << std::flush;

  //Add a dummy variable to the context
  context->addVariable("dummy_variable");

  //Set the value of dummy variable for all primitives to 5.0
  for( uint i=0; i<context->getPrimitiveCount(); i++ ){
    context->setVariableValue("dummy_variable",i,5.f);
  }

  std::cout << "done." << std::endl;
}

