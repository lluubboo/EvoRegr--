cmake_minimum_required(VERSION 3.10)

# Set the project name
project(Benchmarks)

# Add the executable
add_executable(${PROJECT_NAME})

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

target_link_libraries(${PROJECT_NAME} PRIVATE Eigen3::Eigen sources_lib includes_lib externals_lib)

# Add the source files
target_sources(${PROJECT_NAME} PRIVATE benchmark.cpp )

