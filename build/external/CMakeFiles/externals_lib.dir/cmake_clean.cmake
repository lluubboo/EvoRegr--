file(REMOVE_RECURSE
  "libexternals_lib.a"
  "libexternals_lib.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/externals_lib.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
