add_library(sources_lib EvoResultPostProcessing.cpp EvoCore.cpp Log.cpp CustomCaches.cpp IOTools.cpp EvoIndividual.cpp EvoGene.cpp RegressionSolver.cpp EvoLibrary.cpp RandomNumberGenerator.cpp Stats.cpp Plotter.cpp EvoGui.cpp)
target_link_libraries(sources_lib PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen spdlog::spdlog fltk includes_lib externals_lib)
target_include_directories(sources_lib INTERFACE . ${FLTK_INCLUDE_DIRS})

