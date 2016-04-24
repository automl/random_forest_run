#ifndef RFR_UTIL_HPP
#define RFR_UTIL_HPP


namespace rfr{

/* Merges f1 and f2 into dest without copying NaNs. This allows for easy marginalization */
template <typename num_type, typename index_type>
inline void merge_two_vectors ( num_type* f1, num_type* f2, num_type* dest, index_type n){
	// make full copy of first vector
	std::copy_n(f1,n, dest);
			for (index_type j=0u; j <n; ++j){
				// copy everything from the second vector that is not NaN
				if (!std::isnan(f2[j]))
					dest[j] = f2[j];
				else if (std::isnan(dest[j]))
					throw std::runtime_error("Merged feature vector still contains a NaN");
			}
}



template<typename num_type>
class running_statistics{
  private:
	long unsigned int N;
	num_type m, v;
  public:
	running_statistics(): N(0), m(0), v(0) {}
	
	void operator() (num_type x){
		++N;
		num_type delta = x - m;
		// adjust mean
		m += delta/N;
		// adjust variance
		v += delta*(x-m);
	}
	
	long unsigned int number_of_points() {return(N);}
	num_type mean(){ return(m);}
	num_type variance(){return(std::max<double>(0.,v/(N-1)));}
};



template <typename num_type>
class running_covariance{
  private:
	long unsigned int N;
	num_type m1, m2;
	num_type cov;

  public:
	running_covariance(): N(0), m1(0), m2(0), cov(0) {}
	
	void operator() (num_type x1, num_type x2){
		N++;
		num_type delta1 = (x1-m1)/N;
		m1 += delta1;
		num_type delta2 = (x2-m2)/N;
		m2 += delta2;
		
		cov += (N-1) * delta1 * delta2 - cov/N;
	}
	
	long unsigned int number_of_points(){return(N);}
	num_type covariance(){return(num_type(N)/num_type(N-1)*cov);}
};






}//namespace rfr::forests
#endif
