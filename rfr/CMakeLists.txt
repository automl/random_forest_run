cmake_minimum_required (VERSION 3.1 FATAL_ERROR)
project (RFR CXX)

# add a version number
set (RFR_VERSION_MAJOR 0)
set (RFR_VERSION_MINOR 1)


enable_testing()


# required libaries
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(Doxygen)



# define all include directories within the project for different splits, data containers, node types etc...
include_directories ("${PROJECT_SOURCE_DIR}/src/include")



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
