cmake_minimum_required(VERSION 3.5.0)

project(EVOREGR++ VERSION 0.1.0 LANGUAGES C CXX)

# set C++ standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)

set(CMAKE_BUILD_TYPE Release)

if (CMAKE_BUILD_TYPE STREQUAL "Debug") 
    #-Weffc++
    add_compile_options(-Wall -Wpedantic -Wextra -fexceptions)
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    if(WIN32)
        set(CMAKE_EXE_LINKER_FLAGS "-mwindows")
# -flto -funroll-loops -fprefetch-loop-arrays -mavx -mavx2
add_compile_options(-march=native)
    else()
        # do nothing now
    endif()
endif()

add_executable(EVOREGR++ main.cpp)

# append user paths 
list(APPEND CMAKE_PREFIX_PATH "C:/Users/lubomir.balaz/SWLibraries/eigen-3.4.0-install/share/eigen3/cmake")
list(APPEND CMAKE_PREFIX_PATH "C:/Users/lubomir.balaz/SWLibraries/fltk/build")
list(APPEND CMAKE_PREFIX_PATH "C:/Users/lubomir.balaz/SWLibraries/spdlog-1.12.0-install")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# OpenMP
find_package(OpenMP)

# spdlog
find_package(spdlog REQUIRED)

# fltk
find_package(FLTK REQUIRED NO_MODULE)

add_subdirectory(external)
add_subdirectory(include)
add_subdirectory(src)

target_link_libraries (${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog fltk includes_lib sources_lib externals_lib)

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_BINARY_DIR} ${FLTK_INCLUDE_DIRS})

