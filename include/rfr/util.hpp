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



}//namespace rfr::forests
#endif
