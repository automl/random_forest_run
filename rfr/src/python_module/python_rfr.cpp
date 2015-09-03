//#include "rfr/forest/regression_forest.hpp"
#include <boost/python.hpp>



char const* greet( )
{
    return "Hello world";
}



BOOST_PYTHON_MODULE(rfr){
	boost::python::def("greet", greet);
}
