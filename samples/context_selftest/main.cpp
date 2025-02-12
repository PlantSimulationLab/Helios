#include "Context.h"

using namespace helios;

int main(){

#ifdef USE_OPENMP
    std::cout << "OpenMP is enabled. Running with " << omp_get_max_threads() << " threads.\n";
        #pragma omp parallel
        {
            int id = omp_get_thread_num();
            std::cout << "Hello from thread " << id << std::endl;
        }
#else
    std::cout << "OpenMP is not enabled. Running single-threaded.\n";
#endif

  //Run the self-test
  return Context::selfTest();
	
}
