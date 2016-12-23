cd build
lcov --directory . --capture --output-file coverage.info
lcov --remove coverage.info 'tests/*' '/usr/*' '/usr/include/*' '*/include/cereal/*' --output-file coverage.info
lcov --list coverage.info
mkdir gcov_html
genhtml coverage.info --output-directory gcov_html
firefox gcov_html/index.html
cd ..
