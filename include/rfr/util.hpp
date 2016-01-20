#ifndef RFR_UTIL_HPP
#define RFR_UTIL_HPP


namespace rfr{

/* Merges f1 and f2 into dest without copying NaNs. This allows for easy marginalization */
template <typename num_type, typename index_type>
inline void merge_feature_vectors ( num_type* f1, num_type* f2, num_type* dest, index_type n){
	// make full copy of first vector
	std::copy_n(f1,n, dest);
			for (auto j=0u; j <n; ++j){
				// copy everything from the second vector that is not NaN
				if (!isnan(f2[j]))
					dest[j] = f2[j];
				else if (isnan[dest[j])
					throw std::runtime_error("Merged feature vector still contains a NaN");
			}
}




class running_statistics{
  private:
	long unsigned int N;
	double m, v;
	
	running_statistics(): N = 0, m=0, v=0 {};
	
	void operator() (double x){
		++N;
		// initial setup
		if (N == 1){
			m = x;
			v = 0;
		}
		else{
			double m_old = m;
			// adjust mean
			m = m_old + (x-m_old)/N;
			// adjust variance
			v = v + (x-m_old)*(x-m);
		}
	}
	
	double mean(){ return(m);}
	double variance(){return(v/(N-1));}
};






}//namespace rfr::forests
#endif
