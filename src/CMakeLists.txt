add_library(sources_lib IOTools.cpp EvoAPI.cpp EvoGene.cpp RegressionSolver.cpp EvoIndividual.cpp EvoLibrary.cpp RandomNumberGenerator.cpp Stats.cpp Plotter.cpp)
target_link_libraries(sources_lib PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen Matplot++::matplot spdlog::spdlog includes_lib externals_lib)
target_include_directories(sources_lib INTERFACE .)

