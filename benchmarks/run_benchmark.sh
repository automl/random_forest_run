O_level=2
num_feats=1000
num_datapoints=500000
num_samples=5

g++ -I../include/ -O$O_level -Wall -o benchmark_sorting -std=c++11 benchmark_sorting.cpp
./benchmark_sorting $num_feats $num_datapoints $num_samples

g++ -I../include/ -O$O_level -Wall -o benchmark_sorting_profile -std=c++11 benchmark_sorting.cpp
./benchmark_sorting_profile $num_feats $num_datapoints $num_samples
gprof benchmark_sorting_profile gmon.out > analysis.txt
