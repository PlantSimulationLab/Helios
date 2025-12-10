set(CMAKE_CUDA_COMPILER "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/bin/nvcc")
set(CMAKE_CUDA_HOST_COMPILER "")
set(CMAKE_CUDA_HOST_LINK_LAUNCHER "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/bin/g++")
set(CMAKE_CUDA_COMPILER_ID "NVIDIA")
set(CMAKE_CUDA_COMPILER_VERSION "12.3.52")
set(CMAKE_CUDA_DEVICE_LINKER "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/bin/nvlink")
set(CMAKE_CUDA_FATBINARY "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/bin/fatbinary")
set(CMAKE_CUDA_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CUDA_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CUDA_COMPILE_FEATURES "cuda_std_03;cuda_std_11;cuda_std_14;cuda_std_17;cuda_std_20")
set(CMAKE_CUDA03_COMPILE_FEATURES "cuda_std_03")
set(CMAKE_CUDA11_COMPILE_FEATURES "cuda_std_11")
set(CMAKE_CUDA14_COMPILE_FEATURES "cuda_std_14")
set(CMAKE_CUDA17_COMPILE_FEATURES "cuda_std_17")
set(CMAKE_CUDA20_COMPILE_FEATURES "cuda_std_20")
set(CMAKE_CUDA23_COMPILE_FEATURES "")

set(CMAKE_CUDA_PLATFORM_ID "Linux")
set(CMAKE_CUDA_SIMULATE_ID "GNU")
set(CMAKE_CUDA_COMPILER_FRONTEND_VARIANT "")
set(CMAKE_CUDA_SIMULATE_VERSION "11.4")



set(CMAKE_CUDA_COMPILER_ENV_VAR "CUDACXX")
set(CMAKE_CUDA_HOST_COMPILER_ENV_VAR "CUDAHOSTCXX")

set(CMAKE_CUDA_COMPILER_LOADED 1)
set(CMAKE_CUDA_COMPILER_ID_RUN 1)
set(CMAKE_CUDA_SOURCE_FILE_EXTENSIONS cu)
set(CMAKE_CUDA_LINKER_PREFERENCE 15)
set(CMAKE_CUDA_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CUDA_LINKER_DEPFILE_SUPPORTED )

set(CMAKE_CUDA_SIZEOF_DATA_PTR "8")
set(CMAKE_CUDA_COMPILER_ABI "ELF")
set(CMAKE_CUDA_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CUDA_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

if(CMAKE_CUDA_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CUDA_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CUDA_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CUDA_COMPILER_ABI}")
endif()

if(CMAKE_CUDA_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
endif()

set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_VERSION "12.3.52")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0")

set(CMAKE_CUDA_ARCHITECTURES_ALL "50-real;52-real;53-real;60-real;61-real;62-real;70-real;72-real;75-real;80-real;86-real;87-real;89-real;90")
set(CMAKE_CUDA_ARCHITECTURES_ALL_MAJOR "50-real;60-real;70-real;80-real;90")
set(CMAKE_CUDA_ARCHITECTURES_NATIVE "89-real")

set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/targets/x86_64-linux/include")

set(CMAKE_CUDA_HOST_IMPLICIT_LINK_LIBRARIES "")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_DIRECTORIES "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/targets/x86_64-linux/lib/stubs;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/targets/x86_64-linux/lib")
set(CMAKE_CUDA_HOST_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/include;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/include;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/openmpi-5.0.5/include;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/include;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/include/c++/11.4.0;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/include/c++/11.4.0/x86_64-pc-linux-gnu;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/include/c++/11.4.0/backward;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include-fixed;/usr/local/include;/usr/include/x86_64-linux-gnu;/usr/include")
set(CMAKE_CUDA_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/targets/x86_64-linux/lib/stubs;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/targets/x86_64-linux/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib/gcc/x86_64-pc-linux-gnu/11.4.0;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib/gcc;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib64;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/lib64;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib64;/lib/x86_64-linux-gnu;/lib64;/usr/lib/x86_64-linux-gnu;/usr/lib64;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/main/view/generic/lcov-2.0/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-timedate-2.33-pdpr62yz5slq7x2kbyhrt6hr6dsnhzry/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-time-hires-1.9758-m54hiwgjydyhpddg7otrs7qctvceycpl/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-memory-process-0.06-5ef6fmqtroikfq6dgzibtreetbzuuwza/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-json-4.10-bc2iipf5mv3wrod5ry774um2cdwcfxjb/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-json-xs-4.03-icessw6bc7xe4fibiohgmwfeyn66ylbq/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-types-serialiser-1.01-bovcrkyc4oap7hvmpaiqsldjv4u2yont/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-common-sense-3.75-cj4o65isw4al3rmdxrcsyhhfbcdkfs52/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-file-spec-0.90-3rojs7jlzosj2faou2v3lc2giezhxo3v/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-digest-md5-2.55-g35blicz57u52lmmidkqou7azqouvv7k/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-devel-cover-1.42-ln22buermrrxc2uecri7sh6w6wqciysl/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-datetime-1.65-bef5ftetlpbqd2v4lbc3xin3t45k35mv/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-datetime-timezone-2.60-bvyswehejhzqty7bpirvfucc4ez3iygw/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-datetime-locale-1.40-ckk7n3npfemx3rshqn2r5xkmolqcv4wf/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-specio-0.48-osggrtxksgl6lkxp4pfmd6r2ax2czp4k/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-sub-quote-2.006008-7r6a3z3viqoynfnpogzowyaontizwfik/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-scalar-list-utils-1.63-h5c2n6du323wki4okbr64ultspq7ooxm/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-role-tiny-2.002004-usqyn5lc65rximuy6sbi2rsbk6xvvfkm/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-exporter-tiny-1.006002-37txk54gopfc6mbvicr6t636curxbaqb/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-mro-compat-0.15-rqmggtpsp5oaun7iq7ucdrrrpbcxlmpc/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-params-validationcompiler-0.31-66qysomeypica264f7rkfhwzhti5ixj4/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-exception-class-1.45-ashse2vthhqg3daszpelsjiirnjnylzc/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-devel-stacktrace-2.05-ttfoh5uqvqzzddy7qo5vykh3mspagoos/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-class-data-inheritable-0.09-xwupgwalimxluumweaftaajmuyo235dp/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-eval-closure-0.14-o2yh2hoakyaq5othekj3qsrlbw4f5sat/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-namespace-autoclean-0.29-r5h526fc766fo2ovvhpsvcvx3s6k2ugw/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-sub-identify-0.14-mbwevo63igc4vcg7ac3e262c7u5betof/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-namespace-clean-0.27-ppjbanye26t5fbatnidgq2u6mdgnjhh3/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-package-stash-0.40-nr63dpdwvplx3chffaazh5tkd3mjeiau/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-file-sharedir-1.118-byamuhebblxmlfb2z7xz2ecinrgienjo/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-dist-checkconflicts-0.11-zrvz2zx4wmsgc2i63vjehhyi7y35n2ld/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-class-singleton-1.6-iim55drdg7ezuyigpjeuev6leclp3fzx/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-class-inspector-1.36-odg2nxkmjoilpi7zpcdaorfv7gldcf6z/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-capture-tiny-0.48-avgxeiwh6xdliywhrlojfla6xk3c6cj2/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-b-hooks-endofscope-0.26-of4f42j7kscrvnym6pmwzkbveft3g6cq/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-sub-exporter-progressive-0.001013-efnpjii4i62jy7fkv4larqntvuox6hns/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-module-implementation-0.09-llwz36dyku446cbosefn6yslaneoobac/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-test-requires-0.11-xe6hihtxxl4pvf3wgty7dv56hjfp2r4v/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-test-fatal-0.017-2n6ftep5ymg4ifra4j4e2ya6i247vyao/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-try-tiny-0.31-voibp7ljfax4qiybk2m3gtda3qhmml2g/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-module-runtime-0.016-icsesgecxrdeasrp2frienemsq6rgpql/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/main/view/generic/perl-5.40.0/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/openmpi-5.0.5/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib")
set(CMAKE_CUDA_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")

set(CMAKE_CUDA_RUNTIME_LIBRARY_DEFAULT "STATIC")

set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_MT "")
