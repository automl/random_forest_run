//#include <boost/test/unit_test.hpp>

#include <random>

#include <memory>

#include <chrono>
#include <time.h>
#include <fstream>

#include "rfr/data_containers/mostly_continuous_data_container.hpp"
#include "rfr/splits/binary_split_one_feature_rss_loss.hpp"
#include "rfr/trees/k_ary_tree.hpp"
#include "rfr/trees/k_ary_mondrian_tree.hpp"
#include "rfr/forests/regression_forest.hpp"
#include "rfr/forests/mondrian_forest.hpp"
#include "rfr/forests/quantile_regression_forest.hpp"

#include <sstream>
#include <cereal/archives/xml.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
typedef cereal::PortableBinaryInputArchive iarch_type;
typedef cereal::PortableBinaryOutputArchive oarch_type;



typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
typedef std::default_random_engine rng_t;

typedef rfr::data_containers::mostly_continuous_data<num_t, response_t, index_t> data_container_type;

//typedef rfr::splits::binary_split_one_feature_rss_loss<num_t, response_t, index_t, rng_t> split_type;//deletge
typedef rfr::nodes::k_ary_mondrian_node_full<2, num_t, response_t, index_t, rng_t> node_type;

typedef rfr::nodes::temporary_node<num_t, index_t> tmp_node_type;/**/

typedef rfr::trees::k_ary_mondrian_tree<2, node_type, num_t, response_t, index_t, rng_t> tree_type;

typedef rfr::forests::mondrian_forest< tree_type, num_t, response_t, index_t, rng_t> forest_type;

data_container_type load_diabetes_data(){
	data_container_type data;
	
    std::string feature_file, response_file;
    
    feature_file  = "/home/student/Downloads/random_forest_run-master/test_data_sets/diabetes_features.csv";
    response_file = "/home/student/Downloads/random_forest_run-master/test_data_sets/diabetes_responses.csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}

data_container_type load_arm_data(){
	data_container_type data;
	
    std::string feature_file, response_file;
    
    feature_file  = "/home/student/Downloads/random_forest_run-master/test_data_sets/arm_features.csv";
    response_file = "/home/student/Downloads/random_forest_run-master/test_data_sets/arm_responses.csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}

data_container_type load_arm_test_data(){
	data_container_type data;	
    std::string feature_file, response_file;

	feature_file  = "/home/student/Downloads/random_forest_run-master/test_data_sets/arm_features_test.csv";
    response_file = "/home/student/Downloads/random_forest_run-master/test_data_sets/arm_responses_test.csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}

data_container_type load_data(std::string file_name, std::string extra){
	data_container_type data;
	
    std::string feature_file, response_file;
    if(extra != ""){
		extra = "_" + extra;
	}

    feature_file  = "/home/student/Downloads/random_forest_run-master/test_data_sets/" + file_name + "_features" + extra + ".csv";
    response_file = "/home/student/Downloads/random_forest_run-master/test_data_sets/" + file_name + "_responses" + extra + ".csv";

    data.import_csv_files(feature_file, response_file);
    return(data);
}

response_t statistics(std::string name, index_t index, data_container_type data, forest_type forest, double time_diff, rng_t rng){
	response_t oob_error;
	rfr::util::running_statistics<num_t> oob_error_stat;
	std::string prediction = "", responses = "", variances = "", means = "", predicted_means = "", standard_deviations = "",  training_points_per_tree = "", deterministic_prediction = "";
	response_t pred, s_d, pred_mean;
	std::vector<index_t> points;
	auto trees = forest.get_trees();

	int p = 0;
	//for(int p = 0; p<trees.size(); p++){
		points = trees[p].get_used_points();
		for (int i = 0; i<points.size(); i++){
			training_points_per_tree = training_points_per_tree + std::to_string(points[i]) + "\n";
		}
	//}

	std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now(); 
	std::cout << "MONDRIAN_FOREST::beginning prediction of : " << data.num_data_points() << " points" << std::endl;
	for (auto i=0u; i < data.num_data_points(); i++){

		rfr::util::running_statistics<num_t> prediction_stat;
		pred = forest.predict(data.retrieve_data_point(i), s_d, pred_mean, rng);
		prediction_stat.push(pred);
		if(i>0){
			prediction = prediction + "\n" + std::to_string(pred);
			responses = responses + "\n" + std::to_string(data.response(i));
			deterministic_prediction = deterministic_prediction + "\n" + std::to_string(forest.predict_deterministic(data.retrieve_data_point(i)));
			predicted_means = predicted_means + "\n" + std::to_string(pred_mean);
			standard_deviations = standard_deviations + "\n" + std::to_string(s_d);
		}
		else{
			prediction = std::to_string(pred);
			responses = std::to_string(data.response(i));
			deterministic_prediction = std::to_string(forest.predict_deterministic(data.retrieve_data_point(i)));
			predicted_means = std::to_string(pred_mean);
			standard_deviations = std::to_string(s_d);
		}

		
		trees = forest.get_trees();
		for(int p = 0; p<trees.size(); p++){
			variances = variances + std::to_string(trees[p].last_variance) + " ";
			means = means + std::to_string(trees[p].last_mean) + " ";
		}
		variances = variances + "\n";
		means = means + "\n";

		// compute squared error of prediction
		oob_error_stat.push(std::pow(prediction_stat.mean() - data.response(i), 2));
		oob_error = std::sqrt(oob_error_stat.mean());
	}

	std::chrono::high_resolution_clock::time_point prediction_end = std::chrono::high_resolution_clock::now(); 
	
	std::chrono::high_resolution_clock::duration d = prediction_end - beginning;
  	double prediction_time_diff = d.count()/1000000000.0;
	
	std::cout << "MONDRIAN_FOREST::end prediction: " << prediction_time_diff << std::endl;
	std::cout << "MONDRIAN_FOREST::writting files" << std::endl;
	oob_error = std::sqrt(oob_error_stat.mean());
	std::ofstream out_error(name + "_error" + std::to_string(index) + ".txt");
	std::cout << "MONDRIAN_FOREST::error " << oob_error << std::endl;
	std::ofstream out_pred(name + "_pred_sample" + std::to_string(index) + ".txt");
	std::ofstream out_resp(name + "_resp" + std::to_string(index) + ".txt");
	std::ofstream out_var(name + "_var" + std::to_string(index) + ".txt");
	std::ofstream out_mean(name + "_mean" + std::to_string(index) + ".txt");
	std::ofstream out_points(name + "_points" + std::to_string(index) + ".txt");
	std::ofstream out_deterministic_prediction(name + "_deterministic_prediction" + std::to_string(index) + ".txt");

	std::ofstream out_s_d(name + "_pred_sd" + std::to_string(index) + ".txt");
	std::ofstream out_pred_mean(name + "_pred_mean" + std::to_string(index) + ".txt");

	out_error << forest.options.to_string() + "\nObb_error: " + std::to_string(forest.out_of_bag_error()) +
		"\ntest_error:" + std::to_string(oob_error) + "\ntime_diff:" + std::to_string(time_diff);
	
 	out_pred_mean << predicted_means;
 	out_s_d << standard_deviations;
	out_pred << prediction;
	out_resp << responses;
	out_var << variances;
	out_mean << means;
	out_points << training_points_per_tree;
	out_deterministic_prediction << deterministic_prediction;

	out_error.close();
	out_pred_mean.close();
	out_s_d.close();
	out_pred.close();
	out_resp.close();
	out_var.close();
	out_mean.close();
	out_points.close();
	out_deterministic_prediction.close();
	// forest.print_info();

	d = std::chrono::high_resolution_clock::now() - prediction_end;
	time_diff = d.count()/1000000000.0;
	std::cout << "MONDRIAN_FOREST::end of file writting: " << time_diff << std::endl;
	return oob_error;
}

void average_pred(std::string name, index_t max_i){
	std::vector<rfr::util::running_statistics<num_t>> oob_error_stat;
	std::string prediction = "";

	std::string line;
  	std::ifstream myfile(name + "_pred" + std::to_string(0) + ".txt");
	int cont = 0;
	int index_obb_error = 0;

	while (getline (myfile,line))
	{
		int found = line.find("Obb_error:");
		if (found!=std::string::npos){
			line = line.substr (10,line.size());
			index_obb_error = cont;
		}
		if(line.size() < 10){
			oob_error_stat.emplace_back(rfr::util::running_statistics<num_t>());
			response_t value = ::atof(line.c_str());
			oob_error_stat[cont].push(value);
			cont++;
		}
		
	}
	myfile.close();

	for (int i = 1; i<max_i; i++){
		std::ifstream myfile(name + "_pred" + std::to_string(i) + ".txt");
		cont = 0;
		while (getline (myfile,line)){
			int found = line.find("Obb_error:");
			if (found!=std::string::npos){
				line = line.substr (10,line.size());
			}
			if(line.size() < 10){
				response_t value = ::atof(line.c_str());
				oob_error_stat[cont].push(value);
				cont++;
			}
		}
		myfile.close();
	}

	for (int i = 0; i<cont; i++){
		if(i == index_obb_error ){
			prediction = prediction + "\n" + "Obb_error: " + std::to_string(oob_error_stat[i].mean());
		}
		else
			prediction = prediction + "\n" + std::to_string(oob_error_stat[i].mean());
	}
	std::ofstream out_pred(name + "_avg_pred" + ".txt");
	out_pred << prediction;
	out_pred.close();
}



void average_mean_var(std::string name, std::string statistic, index_t max_i){
	std::vector<rfr::util::running_statistics<num_t>> oob_error_stat;
	std::string avgs = "";

	std::string line;
  	std::ifstream myfile(name + "_" + statistic + std::to_string(0) + ".txt");
	int cont = 0;
	int index_obb_error = 0;
  	
	while (getline (myfile,line))
	{			
		oob_error_stat.emplace_back(rfr::util::running_statistics<num_t>());
		response_t value = ::atof(line.c_str());
		oob_error_stat[cont].push(value);
		cont++;
	}
	myfile.close();

	for (int i = 1; i<max_i; i++){
		std::ifstream myfile(name + "_" + statistic + std::to_string(i) + ".txt");
		cont = 0;
		while (getline (myfile,line)){
			response_t value = ::atof(line.c_str());
			oob_error_stat[cont].push(value);
			cont++;
		}
		myfile.close();
	}

	for (int i = 0; i<cont; i++){
		avgs = avgs + "\n" + std::to_string(oob_error_stat[i].mean());
	}
	std::ofstream out_pred(name + "_avg_" + statistic + ".txt");
	out_pred << avgs;
	out_pred.close();
}



void average_runs(std::string name, index_t max_i){
	average_pred(name, max_i);
	average_mean_var(name, "var", max_i);
	average_mean_var(name, "mean", max_i);
}



// show the number of leaves
//average depth
//mean non leaves
//log prob
//mse, rmse

int getPoints() {
  std::string line;
  int points;
  std::ifstream myfile ("points.txt");
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
		points = stoi( line );
      	std::cout << line << '\n';
    }
    myfile.close();
  }

  else std::cout << "Unable to open file"; 

  return points;
}

void execute_test(std::string name, index_t test_number, rng_t seed, bool partial_fit, bool normalizeData){
	auto data = load_data(name, "");
	if(normalizeData){
		data.normalize_data();
	}

	rfr::trees::tree_options<num_t, response_t, index_t> tree_opts;
	tree_opts.min_samples_to_split = 2;
	tree_opts.min_samples_in_leaf = 1;
	tree_opts.hierarchical_smoothing = false;
	tree_opts.max_features = data.num_data_points()*3/4;
	tree_opts.life_time = 1000;
	tree_opts.min_samples_node = 1;
	
	rfr::forests::forest_options<num_t, response_t, index_t> forest_opts(tree_opts);

	//index_t num_data_points_per_tree = getPoints();
	index_t num_data_points_per_tree = data.num_data_points();//10000; //data.num_data_points();
	forest_opts.num_data_points_per_tree = num_data_points_per_tree;
	forest_opts.num_trees = 10;	
	forest_opts.do_bootstrapping = false;
	forest_opts.compute_oob_error= false;//test
	
	forest_type the_forest(forest_opts);
	
	std::cout << "UNIT_TEST::Mondrian Forest created" << std::endl;

	

	rng_t rng(seed);


	the_forest.name = name;
	the_forest.internal_index = test_number;
	std::chrono::high_resolution_clock::time_point beginning = std::chrono::high_resolution_clock::now(); 
	time_t start, end;
  	// struct tm y2k = {0};
  	double seconds;
  	// y2k.tm_hour = 0;y2k.tm_min = 0;y2k.tm_sec = 0;
  	// y2k.tm_year = 100;y2k.tm_mon = 0;y2k.tm_mday = 1;
  	 time(&start);
	// seconds = difftime(timer,mktime(&y2k));

	if(partial_fit){
		index_t last_data_point = data.num_data_points();
		for(index_t i =0; i<num_data_points_per_tree && i<last_data_point; i++){
			the_forest.partial_fit(data, rng, i);
		}
	}
	else
		the_forest.fit(data, rng);
	time(&end);
	std::chrono::high_resolution_clock::duration d = std::chrono::high_resolution_clock::now() - beginning;
  	double time_diff = d.count()/1000000000.0;
	seconds = difftime(end,start);
	std::cout << "UNIT_TEST::Mondrian Forest fitted s " << seconds << std::endl;
	std::cout << "UNIT_TEST::Mondrian Forest fitted time difference " << time_diff << std::endl;
	// std::getchar();
	//auto data_test = load_arm_test_data();
	auto data_test = load_data(name, "test");
	if(normalizeData){
		data_test.normalize_data();
	}
	auto obb_err = statistics(name, test_number, data_test, the_forest, time_diff, rng);
	//the_forest.print_info();
}

int main(int argc, char* argv[]){
	time_t timer;
  	struct tm y2k = {0};
  	double seconds;
  	y2k.tm_hour = 0;y2k.tm_min = 0;y2k.tm_sec = 0;
  	y2k.tm_year = 100;y2k.tm_mon = 0;y2k.tm_mday = 1;
  	time(&timer);
	seconds = difftime(timer,mktime(&y2k));
	rng_t rng(seconds);

	//std::string dataset = "sinx_noise";
	//std::string dataset = "arm";
	std::string dataset = "sinxy";
	bool partial_fit = false;
	bool normalize_data = false;
	std::cout << argv[0] << " " << argv[1] << std::endl;
	int from = std::stoi(argv[1]);
	int max = std::stoi(argv[2]);
	
	for(int i = from; i< max; i++){
		//rng_t seed(i);
		rng_t seed(3);
		//rng_t seed(seconds);
		execute_test(dataset, i, seed, partial_fit, normalize_data);
	}
	average_runs(dataset, max);
}	
