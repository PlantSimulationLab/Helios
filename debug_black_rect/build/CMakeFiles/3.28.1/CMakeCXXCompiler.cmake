set(CMAKE_CXX_COMPILER "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/bin/c++")
set(CMAKE_CXX_COMPILER_ARG1 "")
set(CMAKE_CXX_COMPILER_ID "GNU")
set(CMAKE_CXX_COMPILER_VERSION "11.4.0")
set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")
set(CMAKE_CXX_COMPILER_WRAPPER "")
set(CMAKE_CXX_STANDARD_COMPUTED_DEFAULT "17")
set(CMAKE_CXX_EXTENSIONS_COMPUTED_DEFAULT "ON")
set(CMAKE_CXX_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters;cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates;cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates;cxx_std_17;cxx_std_20;cxx_std_23")
set(CMAKE_CXX98_COMPILE_FEATURES "cxx_std_98;cxx_template_template_parameters")
set(CMAKE_CXX11_COMPILE_FEATURES "cxx_std_11;cxx_alias_templates;cxx_alignas;cxx_alignof;cxx_attributes;cxx_auto_type;cxx_constexpr;cxx_decltype;cxx_decltype_incomplete_return_types;cxx_default_function_template_args;cxx_defaulted_functions;cxx_defaulted_move_initializers;cxx_delegating_constructors;cxx_deleted_functions;cxx_enum_forward_declarations;cxx_explicit_conversions;cxx_extended_friend_declarations;cxx_extern_templates;cxx_final;cxx_func_identifier;cxx_generalized_initializers;cxx_inheriting_constructors;cxx_inline_namespaces;cxx_lambdas;cxx_local_type_template_args;cxx_long_long_type;cxx_noexcept;cxx_nonstatic_member_init;cxx_nullptr;cxx_override;cxx_range_for;cxx_raw_string_literals;cxx_reference_qualified_functions;cxx_right_angle_brackets;cxx_rvalue_references;cxx_sizeof_member;cxx_static_assert;cxx_strong_enums;cxx_thread_local;cxx_trailing_return_types;cxx_unicode_literals;cxx_uniform_initialization;cxx_unrestricted_unions;cxx_user_literals;cxx_variadic_macros;cxx_variadic_templates")
set(CMAKE_CXX14_COMPILE_FEATURES "cxx_std_14;cxx_aggregate_default_initializers;cxx_attribute_deprecated;cxx_binary_literals;cxx_contextual_conversions;cxx_decltype_auto;cxx_digit_separators;cxx_generic_lambdas;cxx_lambda_init_captures;cxx_relaxed_constexpr;cxx_return_type_deduction;cxx_variable_templates")
set(CMAKE_CXX17_COMPILE_FEATURES "cxx_std_17")
set(CMAKE_CXX20_COMPILE_FEATURES "cxx_std_20")
set(CMAKE_CXX23_COMPILE_FEATURES "cxx_std_23")

set(CMAKE_CXX_PLATFORM_ID "Linux")
set(CMAKE_CXX_SIMULATE_ID "")
set(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "GNU")
set(CMAKE_CXX_SIMULATE_VERSION "")




set(CMAKE_AR "/usr/bin/ar")
set(CMAKE_CXX_COMPILER_AR "/usr/bin/gcc-ar-11")
set(CMAKE_RANLIB "/usr/bin/ranlib")
set(CMAKE_CXX_COMPILER_RANLIB "/usr/bin/gcc-ranlib-11")
set(CMAKE_LINKER "/usr/bin/ld")
set(CMAKE_MT "")
set(CMAKE_TAPI "CMAKE_TAPI-NOTFOUND")
set(CMAKE_COMPILER_IS_GNUCXX 1)
set(CMAKE_CXX_COMPILER_LOADED 1)
set(CMAKE_CXX_COMPILER_WORKS TRUE)
set(CMAKE_CXX_ABI_COMPILED TRUE)

set(CMAKE_CXX_COMPILER_ENV_VAR "CXX")

set(CMAKE_CXX_COMPILER_ID_RUN 1)
set(CMAKE_CXX_SOURCE_FILE_EXTENSIONS C;M;c++;cc;cpp;cxx;m;mm;mpp;CPP;ixx;cppm;ccm;cxxm;c++m)
set(CMAKE_CXX_IGNORE_EXTENSIONS inl;h;hpp;HPP;H;o;O;obj;OBJ;def;DEF;rc;RC)

foreach (lang C OBJC OBJCXX)
  if (CMAKE_${lang}_COMPILER_ID_RUN)
    foreach(extension IN LISTS CMAKE_${lang}_SOURCE_FILE_EXTENSIONS)
      list(REMOVE_ITEM CMAKE_CXX_SOURCE_FILE_EXTENSIONS ${extension})
    endforeach()
  endif()
endforeach()

set(CMAKE_CXX_LINKER_PREFERENCE 30)
set(CMAKE_CXX_LINKER_PREFERENCE_PROPAGATES 1)
set(CMAKE_CXX_LINKER_DEPFILE_SUPPORTED TRUE)

# Save compiler ABI information.
set(CMAKE_CXX_SIZEOF_DATA_PTR "8")
set(CMAKE_CXX_COMPILER_ABI "ELF")
set(CMAKE_CXX_BYTE_ORDER "LITTLE_ENDIAN")
set(CMAKE_CXX_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")

if(CMAKE_CXX_SIZEOF_DATA_PTR)
  set(CMAKE_SIZEOF_VOID_P "${CMAKE_CXX_SIZEOF_DATA_PTR}")
endif()

if(CMAKE_CXX_COMPILER_ABI)
  set(CMAKE_INTERNAL_PLATFORM_ABI "${CMAKE_CXX_COMPILER_ABI}")
endif()

if(CMAKE_CXX_LIBRARY_ARCHITECTURE)
  set(CMAKE_LIBRARY_ARCHITECTURE "x86_64-linux-gnu")
endif()

set(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX "")
if(CMAKE_CXX_CL_SHOWINCLUDES_PREFIX)
  set(CMAKE_CL_SHOWINCLUDES_PREFIX "${CMAKE_CXX_CL_SHOWINCLUDES_PREFIX}")
endif()





set(CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/include;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/include;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/openmpi-5.0.5/include;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/include;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/include/c++/11.4.0;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/include/c++/11.4.0/x86_64-pc-linux-gnu;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/include/c++/11.4.0/backward;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib/gcc/x86_64-pc-linux-gnu/11.4.0/include-fixed;/usr/local/include;/usr/include/x86_64-linux-gnu;/usr/include")
set(CMAKE_CXX_IMPLICIT_LINK_LIBRARIES "stdc++;m;gcc_s;gcc;c;gcc_s;gcc")
set(CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES "/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib/gcc/x86_64-pc-linux-gnu/11.4.0;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib/gcc;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib64;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/cuda-12.3.0/lib64;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib64;/lib/x86_64-linux-gnu;/lib64;/usr/lib/x86_64-linux-gnu;/usr/lib64;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/main/view/generic/lcov-2.0/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-timedate-2.33-pdpr62yz5slq7x2kbyhrt6hr6dsnhzry/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-time-hires-1.9758-m54hiwgjydyhpddg7otrs7qctvceycpl/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-memory-process-0.06-5ef6fmqtroikfq6dgzibtreetbzuuwza/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-json-4.10-bc2iipf5mv3wrod5ry774um2cdwcfxjb/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-json-xs-4.03-icessw6bc7xe4fibiohgmwfeyn66ylbq/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-types-serialiser-1.01-bovcrkyc4oap7hvmpaiqsldjv4u2yont/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-common-sense-3.75-cj4o65isw4al3rmdxrcsyhhfbcdkfs52/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-file-spec-0.90-3rojs7jlzosj2faou2v3lc2giezhxo3v/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-digest-md5-2.55-g35blicz57u52lmmidkqou7azqouvv7k/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-devel-cover-1.42-ln22buermrrxc2uecri7sh6w6wqciysl/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-datetime-1.65-bef5ftetlpbqd2v4lbc3xin3t45k35mv/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-datetime-timezone-2.60-bvyswehejhzqty7bpirvfucc4ez3iygw/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-datetime-locale-1.40-ckk7n3npfemx3rshqn2r5xkmolqcv4wf/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-specio-0.48-osggrtxksgl6lkxp4pfmd6r2ax2czp4k/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-sub-quote-2.006008-7r6a3z3viqoynfnpogzowyaontizwfik/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-scalar-list-utils-1.63-h5c2n6du323wki4okbr64ultspq7ooxm/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-role-tiny-2.002004-usqyn5lc65rximuy6sbi2rsbk6xvvfkm/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-exporter-tiny-1.006002-37txk54gopfc6mbvicr6t636curxbaqb/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-mro-compat-0.15-rqmggtpsp5oaun7iq7ucdrrrpbcxlmpc/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-params-validationcompiler-0.31-66qysomeypica264f7rkfhwzhti5ixj4/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-exception-class-1.45-ashse2vthhqg3daszpelsjiirnjnylzc/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-devel-stacktrace-2.05-ttfoh5uqvqzzddy7qo5vykh3mspagoos/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-class-data-inheritable-0.09-xwupgwalimxluumweaftaajmuyo235dp/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-eval-closure-0.14-o2yh2hoakyaq5othekj3qsrlbw4f5sat/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-namespace-autoclean-0.29-r5h526fc766fo2ovvhpsvcvx3s6k2ugw/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-sub-identify-0.14-mbwevo63igc4vcg7ac3e262c7u5betof/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-namespace-clean-0.27-ppjbanye26t5fbatnidgq2u6mdgnjhh3/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-package-stash-0.40-nr63dpdwvplx3chffaazh5tkd3mjeiau/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-file-sharedir-1.118-byamuhebblxmlfb2z7xz2ecinrgienjo/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-dist-checkconflicts-0.11-zrvz2zx4wmsgc2i63vjehhyi7y35n2ld/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-class-singleton-1.6-iim55drdg7ezuyigpjeuev6leclp3fzx/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-class-inspector-1.36-odg2nxkmjoilpi7zpcdaorfv7gldcf6z/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-capture-tiny-0.48-avgxeiwh6xdliywhrlojfla6xk3c6cj2/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-b-hooks-endofscope-0.26-of4f42j7kscrvnym6pmwzkbveft3g6cq/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-sub-exporter-progressive-0.001013-efnpjii4i62jy7fkv4larqntvuox6hns/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-module-implementation-0.09-llwz36dyku446cbosefn6yslaneoobac/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-test-requires-0.11-xe6hihtxxl4pvf3wgty7dv56hjfp2r4v/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-test-fatal-0.017-2n6ftep5ymg4ifra4j4e2ya6i247vyao/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-try-tiny-0.31-voibp7ljfax4qiybk2m3gtda3qhmml2g/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/main/linux-ubuntu22.04-x86_64_v3/gcc-13.2.0/perl-module-runtime-0.016-icsesgecxrdeasrp2frienemsq6rgpql/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/main/view/generic/perl-5.40.0/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/compilers/view/gcc-11.4.0/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/openmpi-5.0.5/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/environments/core/view/generic/slurm/lib;/cvmfs/hpc.ucdavis.edu/sw/spack/spack-ucdavis/environments/hpccf/software.hpc/compilers/._view/bvxuxxurp3aux3xlmic4h4obhdmxlyhl/gcc-11.4.0/lib")
set(CMAKE_CXX_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES "")
