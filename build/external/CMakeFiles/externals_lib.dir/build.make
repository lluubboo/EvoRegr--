# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = C:\Users\lubomir.balaz\SWLibraries\cmake-3.27.4-windows-x86_64\bin\cmake.exe

# The command to remove a file.
RM = C:\Users\lubomir.balaz\SWLibraries\cmake-3.27.4-windows-x86_64\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++\build"

# Include any dependencies generated for this target.
include external/CMakeFiles/externals_lib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include external/CMakeFiles/externals_lib.dir/compiler_depend.make

# Include the progress variables for this target.
include external/CMakeFiles/externals_lib.dir/progress.make

# Include the compile flags for this target's objects.
include external/CMakeFiles/externals_lib.dir/flags.make

# Object files for target externals_lib
externals_lib_OBJECTS =

# External object files for target externals_lib
externals_lib_EXTERNAL_OBJECTS =

external/libexternals_lib.a: external/CMakeFiles/externals_lib.dir/build.make
external/libexternals_lib.a: external/CMakeFiles/externals_lib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir="C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++\build\CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Linking CXX static library libexternals_lib.a"
	cd /d C:\Users\LUBOMI~1.BAL\Desktop\PROJEK~1\EVOREG~2\build\external && $(CMAKE_COMMAND) -P CMakeFiles\externals_lib.dir\cmake_clean_target.cmake
	cd /d C:\Users\LUBOMI~1.BAL\Desktop\PROJEK~1\EVOREG~2\build\external && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\externals_lib.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
external/CMakeFiles/externals_lib.dir/build: external/libexternals_lib.a
.PHONY : external/CMakeFiles/externals_lib.dir/build

external/CMakeFiles/externals_lib.dir/clean:
	cd /d C:\Users\LUBOMI~1.BAL\Desktop\PROJEK~1\EVOREG~2\build\external && $(CMAKE_COMMAND) -P CMakeFiles\externals_lib.dir\cmake_clean.cmake
.PHONY : external/CMakeFiles/externals_lib.dir/clean

external/CMakeFiles/externals_lib.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" "C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++" "C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++\external" "C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++\build" "C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++\build\external" "C:\Users\lubomir.balaz\Desktop\Projekty 2023\EvoRegr++\build\external\CMakeFiles\externals_lib.dir\DependInfo.cmake" "--color=$(COLOR)"
.PHONY : external/CMakeFiles/externals_lib.dir/depend

