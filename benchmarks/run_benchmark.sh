O_level=$1
num_feats=100
num_datapoints=100000
num_samples=50000
num_trees=64

#g++ -I../include/ -O$O_level -Wall -o benchmark_sorting -std=c++11 benchmark_sorting.cpp
#./benchmark_sorting $num_feats $num_datapoints $num_samples

#g++ -I../include/ -O$O_level -Wall -o benchmark_sorting_profile -std=c++11 benchmark_sorting.cpp
#./benchmark_sorting_profile $num_feats $num_datapoints $num_samples
#gprof benchmark_sorting_profile gmon.out > analysis.txt

g++ -I../include/ -O$O_level -Wall -pg -o benchmark_rss_v1 -std=c++11 benchmark_rss_v1.cpp
time ./benchmark_rss_v1 $num_feats $num_datapoints $num_samples $num_trees
gprof benchmark_rss_v1 gmon.out > analysis_v1.txt
gprof benchmark_rss_v1 | python /home/sfalkner/.local/lib/python3.5/site-packages/gprof2dot.py -s | dot -Tpng -o graph_v1.png

g++ -I../include/ -O$O_level -Wall -pg -o benchmark_rss_v2 -std=c++11 benchmark_rss_v2.cpp
time ./benchmark_rss_v2 $num_feats $num_datapoints $num_samples $num_trees
gprof benchmark_rss_v2 gmon.out > analysis_v2.txt
gprof benchmark_rss_v2 | python /home/sfalkner/.local/lib/python3.5/site-packages/gprof2dot.py -s | dot -Tpng -o graph_v2.png
