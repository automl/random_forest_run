/usr/bin/c++ -O3 -pg -I./../include ../tests/profiling_example.cpp  -o profiling_example -rdynamic -std=gnu++11
./profiling_example ../test_data_sets/
gprof profiling_example gmon.out > analysis.txt
