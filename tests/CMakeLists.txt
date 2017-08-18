file(GLOB TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} unit_test_*.cpp)

#colon separated list for tests to skip
set (SKIP_TESTS "unit_test_classification_forest.cpp")

# Build all the unit tests
foreach(TEST_SOURCE ${TESTS})
	# construct target name from file name
	string(REPLACE ".cpp" "" TMP "${TEST_SOURCE}")
	string(REPLACE "unit_test" "ut" TEST_TARGET "${TMP}")

	# only build the tests that are not 
	list(FIND SKIP_TESTS "${TEST_SOURCE}" TO_BE_SKIPPED)
	if (TO_BE_SKIPPED EQUAL -1)
		add_executable(${TEST_TARGET} ${TEST_SOURCE})
		set_target_properties(${TEST_TARGET} PROPERTIES COMPILE_DEFINITIONS "BOOST_TEST_DYN_LINK;BOOST_TEST_MODULE=${TEST_TARGET}")
		set_target_properties(${TEST_TARGET} PROPERTIES CXX_STANDARD 11)
		target_link_libraries(${TEST_TARGET} ${Boost_LIBRARIES})
		add_test("${TEST_TARGET}" "${TEST_TARGET}" "${PROJECT_SOURCE_DIR}/test_data_sets/")
	else()
		message("Skipping ${TEST_SOURCE}")
	endif()
endforeach()


if(PYTHONINTERP_FOUND AND SWIG_FOUND)
	file(GLOB PYRFR_TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} pyrfr_unit_test_*.py)

	foreach(PYRFR_TEST ${PYRFR_TESTS})
		configure_file(${PYRFR_TEST} ${CMAKE_CURRENT_BINARY_DIR}/../python_package/tests/${PYRFR_TEST})
	endforeach()

	add_test(NAME pyrfr_test
			COMMAND ${PYTHON_EXECUTABLE} -m unittest discover -p pyrfr_unit_test*.py -s tests
			WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/../python_package
	)


endif(PYTHONINTERP_FOUND AND SWIG_FOUND)

