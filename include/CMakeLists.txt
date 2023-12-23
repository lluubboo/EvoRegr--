add_library(includes_lib INTERFACE)
target_link_libraries(includes_lib INTERFACE externals_lib OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog fltk)
target_include_directories(includes_lib INTERFACE . ${FLTK_INCLUDE_DIRS})