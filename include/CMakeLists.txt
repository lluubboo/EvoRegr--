add_library(includes_lib IOTools.hpp EvoAPI.hpp EvoGene.hpp EvoIndividual.hpp RegressionSolver.hpp RandomChoices.hpp EvoLibrary.hpp CommonTools.hpp RandomNumberGenerator.hpp Stats.hpp emptyfile.cpp)
target_link_libraries(includes_lib PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen externals_lib)
target_include_directories(includes_lib INTERFACE .)