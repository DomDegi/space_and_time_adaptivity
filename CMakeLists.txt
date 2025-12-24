cmake_minimum_required(VERSION 3.12.0)
project(lab05 LANGUAGES CXX C)

# ==============================================================================
#  CONFIGURATION
# ==============================================================================

# C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Default build type: Release
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# MPI Configuration
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")

# Boost Configuration
# HINTS is essential on clusters to find libraries in module paths
find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix}
)
message(STATUS "Boost found at: ${Boost_DIR}")
include_directories(${Boost_INCLUDE_DIRS})

# deal.II Configuration
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix}
)
deal_ii_initialize_cached_variables()

# Compiler Flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")

# ==============================================================================
#  PROJECT TARGETS
# ==============================================================================

add_executable(heat-equation
    src/main.cc
    src/heat_equation.cc
    src/utilities.cc
)

deal_ii_setup_target(heat-equation)
