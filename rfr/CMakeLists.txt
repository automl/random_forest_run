cmake_minimum_required (VERSION 2.8 FATAL_ERROR) # used to be version 3.1, but I don't know if that is really necessary
project (RFR CXX)

# add a version number
set (RFR_VERSION_MAJOR 0)
set (RFR_VERSION_MINOR 1)


IF(NOT CMAKE_BUILD_TYPE)
 #SET(CMAKE_BUILD_TYPE "DEBUG")
 SET(CMAKE_BUILD_TYPE "RELEASE")
 #SET(CMAKE_BUILD_TYPE "RELWITHDEBINFO")
 #SET(CMAKE_BUILD_TYPE "MINSIZEREL")
ENDIF()



enable_testing()


# required libaries
find_package(Boost COMPONENTS unit_test_framework python3 REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(Doxygen)

SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED OFF)
SET(Boost_USE_STATIC_RUNTIME OFF)


# define all include directories within the project for different splits, data containers, node types etc...
include_directories ("${PROJECT_SOURCE_DIR}/src/include")
include_directories ("${PYTHON_INCLUDE_DIRS}")
include_directories ("${Boost_INCLUDE_DIR}")

link_libraries(${Boost_LIBRARIES} ${PYTHON_LIBRARIES}) # Deprecated but so convenient!




# the documentation
if(DOXYGEN_FOUND)
	configure_file(${PROJECT_SOURCE_DIR}/Doxyfile ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
	add_custom_target(doc ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
		COMMENT "Generating API documentation with Doxygen" VERBATIM)

endif(DOXYGEN_FOUND)





# the unit tests
add_executable(ut_data_containers "${PROJECT_SOURCE_DIR}/src/tests/unit_test_data_containers.cpp")
set_property(TARGET ut_data_containers PROPERTY CXX_STANDARD 11)
set_property(TARGET ut_data_containers PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(ut_data_containers ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test (UTDataContainers ut_data_containers "${PROJECT_SOURCE_DIR}/test_data_sets/")
set_tests_properties (UTDataContainers PROPERTIES PASS_REGULAR_EXPRESSION "No errors detected")


add_executable(ut_binary_splits "${PROJECT_SOURCE_DIR}/src/tests/unit_test_binary_splits.cpp")
set_property(TARGET ut_binary_splits PROPERTY CXX_STANDARD 11)
set_property(TARGET ut_binary_splits PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(ut_binary_splits ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test (UTBinarySplits ut_binary_splits "${PROJECT_SOURCE_DIR}/test_data_sets/")
set_tests_properties (UTBinarySplits PROPERTIES PASS_REGULAR_EXPRESSION "No errors detected")


add_executable(ut_binary_nodes "${PROJECT_SOURCE_DIR}/src/tests/unit_test_binary_nodes.cpp")
set_property(TARGET ut_binary_nodes PROPERTY CXX_STANDARD 11)
set_property(TARGET ut_binary_nodes PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(ut_binary_nodes ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test (UTBinaryNodes ut_binary_nodes "${PROJECT_SOURCE_DIR}/test_data_sets/")
set_tests_properties (UTBinaryNodes PROPERTIES PASS_REGULAR_EXPRESSION "No errors detected")


add_executable(ut_binary_trees "${PROJECT_SOURCE_DIR}/src/tests/unit_test_binary_tree.cpp")
set_property(TARGET ut_binary_trees PROPERTY CXX_STANDARD 11)
set_property(TARGET ut_binary_trees PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(ut_binary_trees ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test (UTBinaryTrees ut_binary_trees "${PROJECT_SOURCE_DIR}/test_data_sets/")
set_tests_properties (UTBinaryTrees PROPERTIES PASS_REGULAR_EXPRESSION "No errors detected")


add_executable(ut_regression_forest "${PROJECT_SOURCE_DIR}/src/tests/unit_test_regression_forest.cpp")
set_property(TARGET ut_regression_forest PROPERTY CXX_STANDARD 11)
set_property(TARGET ut_regression_forest PROPERTY CXX_STANDARD_REQUIRED ON)
target_link_libraries(ut_regression_forest ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY})

add_test (UTRegressionForest ut_regression_forest "${PROJECT_SOURCE_DIR}/test_data_sets/")
set_tests_properties (UTRegressionForest PROPERTIES PASS_REGULAR_EXPRESSION "No errors detected")




add_subdirectory("Boost.NumPy")
include_directories("${PROJECT_SOURCE_DIR}/Boost.NumPy")

#PYTHON_ADD_MODULE(rfr "${PROJECT_SOURCE_DIR}/src/python_module/rfr.cpp" boost_numpy)
add_library(rfr SHARED "${PROJECT_SOURCE_DIR}/src/python_module/rfr.cpp")
target_link_libraries( rfr boost_numpy)
set_property(TARGET rfr PROPERTY CXX_STANDARD 11)
set_property(TARGET rfr PROPERTY PREFIX "")
set_property(TARGET rfr PROPERTY CXX_STANDARD_REQUIRED ON)





IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall -fPIC")
  #ADD_DEFINITIONS("-fPIC")
 ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()
