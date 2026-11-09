cmake_minimum_required(VERSION 3.16)
project(HPCGPerformanceAnalyzer
        VERSION 1.0.0
        DESCRIPTION "HPCG-inspired Conjugate Gradient benchmark (educational)"
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HPCG_ENABLE_OPENMP "Build the OpenMP and hybrid solvers" ON)
option(HPCG_ENABLE_MPI    "Build the MPI and hybrid solvers"    ON)
option(HPCG_NATIVE        "Compile with -march=native (non-portable binary)" OFF)
option(HPCG_BUILD_TESTS   "Build the unit tests" ON)
set(HPCG_MPIEXEC_EXTRA_FLAGS "" CACHE STRING "Extra flags for mpiexec in tests (e.g. --oversubscribe)")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(HPCG_WARNINGS -Wall -Wextra -Wpedantic)
  if(HPCG_NATIVE)
    add_compile_options(-march=native)
  endif()
endif()

# ---------------------------------------------------------------- dependencies
set(HPCG_HAVE_OPENMP OFF)
if(HPCG_ENABLE_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    set(HPCG_HAVE_OPENMP ON)
  else()
    message(WARNING "OpenMP not found: openmp/hybrid modes will be disabled")
  endif()
endif()

set(HPCG_HAVE_MPI OFF)
if(HPCG_ENABLE_MPI)
  find_package(MPI COMPONENTS CXX)
  if(MPI_CXX_FOUND)
    set(HPCG_HAVE_MPI ON)
  else()
    message(WARNING "MPI not found: mpi/hybrid modes will be disabled")
  endif()
endif()

# ------------------------------------------------------------------ core library
add_library(hpcg_core STATIC
  src/matrix/csr_matrix.cpp
  src/matrix/matrix_generator.cpp
  src/sequential/cg_sequential.cpp
  src/openmp/cg_openmp.cpp
  src/benchmark/metrics.cpp
  src/benchmark/benchmark.cpp)
if(HPCG_HAVE_MPI)
  target_sources(hpcg_core PRIVATE
    src/mpi/distributed_problem.cpp
    src/mpi/cg_mpi.cpp
    src/hybrid/cg_hybrid.cpp)
endif()

target_include_directories(hpcg_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(hpcg_core PRIVATE ${HPCG_WARNINGS})

string(TOUPPER "${CMAKE_BUILD_TYPE}" _bt)
set(_flags "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${_bt}}")
if(HPCG_NATIVE)
  string(APPEND _flags " -march=native")
endif()
string(STRIP "${_flags}" _flags)
target_compile_definitions(hpcg_core PUBLIC
  HPCG_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
  HPCG_BUILD_FLAGS="${_flags}")

if(HPCG_HAVE_OPENMP)
  target_link_libraries(hpcg_core PUBLIC OpenMP::OpenMP_CXX)
  target_compile_definitions(hpcg_core PUBLIC HPCG_HAVE_OPENMP)
endif()
if(HPCG_HAVE_MPI)
  target_link_libraries(hpcg_core PUBLIC MPI::MPI_CXX)
  target_compile_definitions(hpcg_core PUBLIC HPCG_HAVE_MPI)
endif()

# ------------------------------------------------------------------- executable
add_executable(hpcg_benchmark src/main.cpp)
target_link_libraries(hpcg_benchmark PRIVATE hpcg_core)
target_compile_options(hpcg_benchmark PRIVATE ${HPCG_WARNINGS})

# Memory-bandwidth probe used for the roofline analysis.
add_executable(stream_triad src/tools/stream_triad.cpp)
target_compile_options(stream_triad PRIVATE ${HPCG_WARNINGS})
if(HPCG_HAVE_OPENMP)
  target_link_libraries(stream_triad PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(stream_triad PRIVATE HPCG_HAVE_OPENMP)
endif()

# ------------------------------------------------------------------------ tests
if(HPCG_BUILD_TESTS)
  enable_testing()
  foreach(t test_matrix test_sequential test_openmp)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE hpcg_core)
    target_include_directories(${t} PRIVATE tests)
    target_compile_options(${t} PRIVATE ${HPCG_WARNINGS})
  endforeach()
  add_test(NAME test_matrix     COMMAND test_matrix)
  add_test(NAME test_sequential COMMAND test_sequential)
  if(HPCG_HAVE_OPENMP)
    add_test(NAME test_openmp COMMAND test_openmp)
  endif()

  if(HPCG_HAVE_MPI)
    add_executable(test_mpi tests/test_mpi.cpp)
    target_link_libraries(test_mpi PRIVATE hpcg_core)
    target_include_directories(test_mpi PRIVATE tests)
    target_compile_options(test_mpi PRIVATE ${HPCG_WARNINGS})
    separate_arguments(_extra UNIX_COMMAND "${HPCG_MPIEXEC_EXTRA_FLAGS}")
    # 3 ranks exercises an uneven row distribution.
    foreach(np 1 2 3 4)
      add_test(NAME test_mpi_np${np}
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np}
                       ${_extra} ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(test_mpi_np${np} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2")
    endforeach()
  endif()

  # End-to-end smoke test of the benchmark executable itself.
  add_test(NAME e2e_sequential COMMAND hpcg_benchmark --mode sequential --grid 12 --reps 1 --quiet)
  if(HPCG_HAVE_OPENMP)
    add_test(NAME e2e_openmp COMMAND hpcg_benchmark --mode openmp --threads 2 --grid 12 --reps 1 --quiet)
  endif()
  if(HPCG_HAVE_MPI)
    add_test(NAME e2e_mpi_halo
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 3 ${_extra} ${MPIEXEC_PREFLAGS}
                     $<TARGET_FILE:hpcg_benchmark> --mode mpi --comm halo --grid 12 --reps 1 --quiet)
  endif()
endif()

message(STATUS "HPCG: OpenMP=${HPCG_HAVE_OPENMP} MPI=${HPCG_HAVE_MPI} build=${CMAKE_BUILD_TYPE} flags='${_flags}'")
