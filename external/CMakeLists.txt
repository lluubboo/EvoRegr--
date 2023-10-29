add_library(externals_lib csv.hpp XoshiroCpp.hpp)
set_target_properties(externals_lib PROPERTIES LINKER_LANGUAGE CXX)
target_include_directories(externals_lib INTERFACE .)