cmake_minimum_required (VERSION 2.8 FATAL_ERROR)
project (RFR CXX)

# add a version number
set (RFR_VERSION_MAJOR 0)
set (RFR_VERSION_MINOR 8)
set (RFR_VERSION_RELEASE 0)

IF(NOT CMAKE_BUILD_TYPE)
 SET(CMAKE_BUILD_TYPE "DEBUG")
 SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage
 #SET(CMAKE_EXE_LINKER_FLAGS "-fprofile-arcs -ftest-coverage")
 
 #SET(CMAKE_BUILD_TYPE "RELESE")
 #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
 #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()

# required libaries
find_package(Boost COMPONENTS unit_test_framework)
find_package(Doxygen)
find_package(PythonInterp 3)
find_package(SWIG)

enable_testing()

# define all include directories within the project for different splits, data containers, node types etc...
include_directories ("${PROJECT_SOURCE_DIR}/include")
include_directories ("${Boost_INCLUDE_DIR}")


# the documentation
if(DOXYGEN_FOUND)
	configure_file(${PROJECT_SOURCE_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doxygen_doc
		COMMAND	${DOXYGEN_EXECUTABLE} Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM)
endif(DOXYGEN_FOUND)



if(PYTHONINTERP_FOUND AND SWIG_FOUND)
	file(COPY ${PROJECT_SOURCE_DIR}/pyrfr DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
	file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/pyrfr ${CMAKE_CURRENT_BINARY_DIR}/python_package)
	
	configure_file(${PROJECT_SOURCE_DIR}/pyrfr/setup.py ${CMAKE_CURRENT_BINARY_DIR}/python_package/setup.py)
	configure_file(${PROJECT_SOURCE_DIR}/pyrfr/pyrfr/__init__.py ${CMAKE_CURRENT_BINARY_DIR}/python_package/pyrfr/__init__.py)
	
	file(COPY ${PROJECT_SOURCE_DIR}/include DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/python_package)
	file(COPY ${PROJECT_SOURCE_DIR}/LICENSE_cereal-1.2.2 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/python_package)


	add_custom_target(pyrfr_docstrings
		COMMAND ${PYTHON_EXECUTABLE} ${PROJECT_SOURCE_DIR}/lib/doxy2swig/doxy2swig.py -a -t -f -q -c ${CMAKE_CURRENT_BINARY_DIR}/doc/xml/index.xml ${CMAKE_CURRENT_BINARY_DIR}/python_package/pyrfr/docstrings.i
		DEPENDS doxygen_doc
		COMMENT "Generating Python Docstrings" VERBATIM
	)

	#file(COPY documentation DESTINATION ${CMAKE_BINARY_DIR})
	#file(RENAME ${CMAKE_BINARY_DIR}/documentation ${CMAKE_BINARY_DIR}/github_docs)
	#file(COPY include DESTINATION ${CMAKE_BINARY_DIR})
	#configure_file(${PROJECT_SOURCE_DIR}/documentation/conf.py ${CMAKE_CURRENT_BINARY_DIR}/github_docs/conf.py)
	#add_custom_target(github_documentation
	#	COMMAND  make html
	#	DEPENDS pyrfr_build
	#	WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/github_docs
	#	COMMENT "Generating github pages"
	#)

	add_custom_target(pyrfr_build ALL
		COMMAND ${PYTHON_EXECUTABLE} setup.py build_ext --inplace --swig ${SWIG_EXECUTABLE}
		DEPENDS pyrfr_docstrings
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/python_package
	)
endif(PYTHONINTERP_FOUND AND SWIG_FOUND)


# add all the unit tests in the tests directory
add_subdirectory(tests)


IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall -O0 -g")
ELSEIF(${CMAKE_CXX_COMPILER_ID} MATCHES "Clang")
  ADD_DEFINITIONS("-Wall -fPIC -pg -std=c++11")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()
