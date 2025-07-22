set(NVCC_EXECUTABLE "${CUDAToolkit_NVCC_EXECUTABLE}")

# ask nvcc for all supported GPU codes (sm_* and compute_*)
execute_process( COMMAND ${NVCC_EXECUTABLE} --list-gpu-code OUTPUT_VARIABLE _nvcc_gpu_codes OUTPUT_STRIP_TRAILING_WHITESPACE )

# split lines into a CMake list
string(REPLACE "\n" ";" _gpu_codes "${_nvcc_gpu_codes}")

# separate out real (sm_XX) and virtual (compute_XX) architectures
set(_sm_codes "")
set(_virtual_codes "")
foreach(_code IN LISTS _gpu_codes)
    if(_code MATCHES "^sm_[0-9]+$")
        list(APPEND _sm_codes ${_code})
    elseif(_code MATCHES "^compute_[0-9]+$")
        list(APPEND _virtual_codes ${_code})
    endif()
endforeach()

# Extract architecture numbers for CMAKE_CUDA_ARCHITECTURES
set(_cuda_architectures "")
foreach(_sm IN LISTS _sm_codes)
    # Extract the number from sm_XX
    string(REGEX REPLACE "sm_(.+)" "\\1" _arch_num "${_sm}")
    list(APPEND _cuda_architectures ${_arch_num})
endforeach()

# if no architectures were detected, fall back to SM 5.0
if(NOT _cuda_architectures)
    message(WARNING "No CUDA architectures detected; defaulting to 50")
    set(_cuda_architectures "50")
endif()

# Set CMAKE_CUDA_ARCHITECTURES for modern CMake
set(CMAKE_CUDA_ARCHITECTURES "${_cuda_architectures}")

# Also build gencode flags for compatibility (if needed elsewhere)
set(_gencode_flags "")
foreach(_sm IN LISTS _sm_codes)
    # derive compute_XX from sm_XX
    string(REGEX REPLACE "sm_(.+)" "compute_\\1" _compute "${_sm}")

    # always emit native cubin for each SM you found
    list(APPEND _gencode_flags  "-gencode"  "arch=${_compute},code=${_sm}" )
endforeach()

# if no architectures were detected, fall back to SM 5.0
if(NOT _gencode_flags)
    message(WARNING "No CUDA architectures detected; defaulting to compute_50,code=sm_50")
    list(APPEND _gencode_flags  "-gencode" "arch=compute_50,code=sm_50" )
endif()

# append to your CUDA flags (for compatibility)
string (JOIN " " _joined "${_gencode_flags}")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} ${_joined}")