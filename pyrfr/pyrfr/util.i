%module util


%{
#include "rfr/util.hpp"

typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;
%}

%include "docstrings.i"
%include "exception.i" 

%exception {
  try {
    $action
  } catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  } catch (const std::string& e) {
    SWIG_exception(SWIG_RuntimeError, e.c_str());
  }
} 

typedef double num_t;
typedef double response_t;
typedef unsigned int index_t;

%ignore rfr::util::merge_two_vectors;
%include "rfr/util.hpp"
%rename (run_stats) rfr::util::running_statistics<num_t>;


%template(running_statistics) rfr::util::running_statistics<num_t>;
//%template(running_covariance) rfr::util::running_covariance<num_t>;
%template(weighted_running_stats) rfr::util::weighted_running_statistics<num_t>;

