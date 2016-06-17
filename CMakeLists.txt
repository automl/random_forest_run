cmake_minimum_required (VERSION 3.1 FATAL_ERROR)
project (RFR CXX)

# add a version number
set (RFR_VERSION_MAJOR 0)
set (RFR_VERSION_MINOR 1)


IF(NOT CMAKE_BUILD_TYPE)
 SET(CMAKE_BUILD_TYPE "DEBUG")
 #SET(CMAKE_BUILD_TYPE "RELESE")
 #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
 #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

# required libaries
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Doxygen)
find_package(PythonInterp)

enable_testing()

# define all include directories within the project for different splits, data containers, node types etc...
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("${Boost_INCLUDE_DIR}")


# the documentation
if(DOXYGEN_FOUND)
	configure_file(${PROJECT_SOURCE_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)


# add all the unit tests in the tests directory
add_subdirectory(tests)


if(PYTHONINTERP_FOUND)
	add_test(NAME BuildPyRFR
			COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/setup.py build_ext --force --inplace
			WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
endif(PYTHONINTERP_FOUND)


IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall")
ELSEIF(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  ADD_DEFINITIONS("-Wall -fPIC -pg -std=c++11")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()
