%module regression

%{
#include "std_vector.i"
#include "rfr/data_containers/data_container.hpp"
#include "rfr/data_containers/mostly_continuous_data_container.hpp"
%}

%include "rfr/data_containers/data_container.hpp";
%include "rfr/data_containers/mostly_continuous_data_container.hpp";
%template(regression_data_base) rfr::data_containers::base<double, double, unsigned int>;
%template(regression_data) rfr::data_containers::mostly_continuous_data<double, double, unsigned int>;



