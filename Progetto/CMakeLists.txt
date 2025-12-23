# Set the minimum required CMake version
cmake_minimum_required(VERSION 3.12.0)

# Define the project name and the languages used (C++ and C)
project(lab05 LANGUAGES CXX C)

# Include the common configuration file shared across your lab projects.
# This likely sets up compiler flags and finds the deal.II installation.
include(../common/cmake-common.cmake)

# Create the executable target named "progetto".
# You must list all implementation files (.cc or .cpp) here.
# Note: You do not need to list header files (.h) here; the compiler 
# finds them automatically via the #include directives in your code.
add_executable(progetto
    src/main.cc
    src/heat_equation.cc
    src/utilities.cc
)

# Link the deal.II library to your "progetto" target.
# This macro handles all the necessary include directories and linker flags.
deal_ii_setup_target(progetto)
