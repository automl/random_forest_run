#ifndef RFR_UTIL_HPP
#define RFR_UTIL_HPP

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>


#include "cereal/cereal.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/array.hpp>



namespace rfr{ namespace util{


/* Compute (pairwise) disjunction of 2 boolean vectors and store the result in dest.*/
inline void disjunction(const std::vector<bool> &source, std::vector<bool> &dest) {
	if (source.size() > dest.size()) {
		dest.resize(source.size());
	}
	for (size_t idx = 0; idx < dest.size(); ++idx) {
		dest[idx] = source[idx] || dest[idx];
	}
}


template <typename num_t>
std::vector<unsigned int> get_non_NAN_indices(const std::vector<num_t> &vector){
	std::vector<unsigned int> indices;
	for (auto i = 0u; i < vector.size(); ++i)
		if (! std::isnan(vector[i]))
			indices.push_back(i);
	return(indices);
}


bool any_true( const std::vector<bool> & b_vector, const std::vector<unsigned int> indices){
	for (auto &i: indices)
		if (b_vector[i])
			return(true);
	return(false);
}


/* Compute the cardinality of a space given the subspaces. */
template <typename num_t, typename index_t>
inline num_t subspace_cardinality(const std::vector< std::vector<num_t> > &subspace, std::vector<index_t> types) {
	num_t result = 1;
	for (auto i=0u; i < types.size(); ++i){
		if (types[i] == 0){ // numerical feature
			result *= subspace[i][1] - subspace[i][0];
		}
		else{ // categorical feature
			result *= subspace[i].size();
		}
	}
	return result;
}

/* Merges f1 and f2 into dest without copying NaNs. This allows for easy marginalization */
template <typename num_t, typename index_type>
inline void merge_two_vectors ( num_t* f1, num_t* f2, num_t* dest, index_type n){
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

/** \brief simple class to compute mean and variance sequentially one value at a time */
template<typename num_t>
class running_statistics{
  private:
	long unsigned int N;
	num_t avg, sdm; // the mean and the squared distance from the mean


  public:

  	/* serialize function */
  	template<class Archive>
	void serialize(Archive & archive) {
		archive( N, avg, sdm); 
	}

  
	running_statistics(): N(0), avg(0), sdm(0) {}

	running_statistics( long unsigned int n, num_t a, num_t s): N(n), avg(a), sdm(s) {}
	/** \brief adds a value to the statistic
	 * 
	 * \param x the value to add
	 */
	void push(num_t x){
		++N;
		num_t delta = x - avg;
		// adjust mean
		avg += delta/N;
		// adjust variance
		sdm += delta*(x-avg);
	}
	/** \brief removes a value from the statistic
	 * 
	 * Consider this the inverse operation to push. Note: you can create
	 * a scenario where the variance would be negative, so a simple sanity
	 * check is implemented that raises a RuntimeError if that happens.
	 * 
	 * \param x the value to remove
	 */
	void pop (num_t x){
		if (--N <1)
			throw std::runtime_error("Statistic now contains no points anymore!");
		num_t delta = x - avg;
		// adjust mean
		avg -= delta/N;
		// adjust variance
		sdm -= delta*(x-avg);
		if (sdm < 0)
			throw std::runtime_error("Statistic now has a negative variance!");
	}

	/** \brief divides the (summed) squared distance from the mean by the argument
	 *  \param value to divide by
	 *\returns sum([ (x - mean())**2 for x in values])/ value */
	num_t	divide_sdm_by(num_t value)	const	{ return(N>1 ? std::max<num_t>(0.,sdm/value) : NAN);}

	/** \brief returns the number of points
	 *\returns the current number of points added*/
	long unsigned int	number_of_points()		const	{return(N);}
	
	/** \brief the mean of all values added
	 *\returns sum([x for x in values])/number_of_points()*/ 
	num_t	mean()					const	{return(N>0 ? avg : NAN);}
	
	/** \brief the sum of all values added
	 *\returns the sum of all values (equivalent to number_of_points()* mean())	*/
	num_t	sum()					const	{return(avg*N);}
	
	/** \brief the sum of all values squared
	 *\returns sum([x**2 for x in values]) */
	num_t	sum_of_squares()		const	{return(sdm + N*mean()*mean());}
	
	/** \brief the variance of all samples assuming it is the total population
	 *\returns sum([(x-mean())**2 for x in values])/number_of_points*/
	num_t	variance_population()	const	{return(divide_sdm_by(N));}
	
	/** \brief unbiased variance of all samples assuming it is a sample from a population with unknown mean
	 *\returns sum([(x-mean())**2 for x in values])/(number_of_points-1)*/
	num_t	variance_sample()		const	{return(divide_sdm_by(N-1));}

	/** \brief biased estimate variance of all samples with the smalles MSE 
	 *\returns sum([(x-mean())**2 for x in values])/(number_of_points+1)*/
	num_t	variance_MSE()			const	{return(divide_sdm_by(N+1));}

	/** \brief standard deviation based on variance_population
	 *\returns sqrt(variance_population())*/
	num_t	std_population() 		const	{return(std::sqrt(variance_population()));}

	/** \brief (biased) estimate of the standard deviation based on variance_sample
	 *\returns sqrt(variance_sample())*/
	 num_t	std_sample() 			const	{return(std::sqrt(variance_sample()));}


	/** \brief unbiased standard deviation for normally distributed values
	 
	 *  Source: https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
	 *\returns std_sample/correction_value */
	num_t	std_unbiased_gaussian() const{
		num_t n(N);
		num_t c4 = 1 - 1/(4*n) - 7/(32*n*n) - 19/(128*n*n*n);
		return( std_sample()/c4);
	}

	/** \brief operator to combine statistics
	 * 
	 * This is a very stable operation, so it should always be accurate. */
	running_statistics operator+ ( const running_statistics &other) const{

		// total number of points is trivial
		long unsigned int N_total = N + other.N;
		num_t n1(N), n2(other.N), nt(N_total);

		// the total mean is also pretty easy to figure out
		num_t avg_total = avg * (n1/nt) + other.avg * (n2/nt);
		// the total sdm looks a bit tricky, but is straight forward to derive
		num_t sdm_total = sdm + other.sdm + n1*std::pow(avg-avg_total,2) + n2*std::pow(other.avg-avg_total,2);

		return(	running_statistics(	N_total, avg_total, sdm_total));
	}

	/**\brief convenience operator for inplace addition*/
	running_statistics& operator+= ( const running_statistics &other) {

		// total number of points is trivial
		long unsigned int N_total = N + other.N;
		num_t n1(N), n2(other.N), nt(N_total);

		// the total mean is also pretty easy to figure out
		num_t avg_total = avg * (n1/nt) + other.avg * (n2/nt);
		// the total sdm looks a bit tricky, but is straight forward to derive
		num_t sdm_total = sdm + other.sdm + n1*std::pow(avg-avg_total,2) + n2*std::pow(other.avg-avg_total,2);

		N = N_total;
		avg = avg_total;
		sdm = sdm_total;
		return(*this);
	}
	/** \brief operator to remove all points of one statistic from another one
	 * 
	 * This function might suffer from catastrophic cancelations, if the two
	 * statistics are almost the same. Use with caution!
	 */
	running_statistics operator- ( const running_statistics &other) const{
		if (other.N >= N)
			throw std::runtime_error("Second statistics must not contain as many points as first one!");

		// new number of points is trivial
		long unsigned int N1 = N - other.N;

		num_t n1(N1), n2(other.N), nt(N);
		// the total mean is also pretty easy to figure out
		num_t avg1 = avg * (nt/n1) - other.avg * (n2/n1);
		// the sdm looks a bit tricky, but is straight forward to derive
		num_t sdm1 = sdm - other.sdm - n2*std::pow(other.avg - avg, 2) - n1*std::pow(avg1-avg,2);
		
		if (N1 == 1) sdm1 = 0;
		
		return(running_statistics( N1, avg1, sdm1));
	}
	
	/** \brief operator to multiply all values by a number */
	running_statistics operator* (const num_t &a) const{
		return(running_statistics(N, a*avg, a*a*sdm));
	}

	/** \brief operator to add a number to all values*/
	running_statistics operator+ (const num_t &a) const{
		return(running_statistics(N, avg+a, sdm));
	}

	/** \brief operator to subtract a number from all values*/
	running_statistics operator- (const num_t &a) const{
		return(running_statistics(N, avg-a, sdm));
	}
	
	/**\brief convenience operator for inplace subtraction*/
	running_statistics &operator-= ( const running_statistics &other) {

		if (other.N >= N)
			throw std::runtime_error("Second statistics must not contain as many points as first one!");

		// new number of points is trivial
		long unsigned int N1 = N - other.N;

		num_t n1(N1), n2(other.N), nt(N);
		// the total mean is also pretty easy to figure out
		num_t avg1 = avg * (nt/n1) - other.avg * (n2/n1);
		// the sdm looks a bit tricky, but is straight forward to derive
		num_t sdm1 = sdm - other.sdm - n2*std::pow(other.avg - avg, 2) - n1*std::pow(avg1-avg,2);

		N = N1;
		avg = avg1;
		sdm = sdm1;
		if (N == 1) sdm = 0;
		
		return(*this);
	}
	
	/**\brief method to check for numerical equivalency
	 * \param other the other running statistic to compare against
	 * \param rel_error relative tolerance for the mean and variance*/
	bool numerically_equal (const running_statistics other, num_t rel_error){
		if (N != other.N) return(false);

		// inline lambda expression for the relative error
		auto relerror = [] (num_t a, num_t b) {return(std::abs(a-b)/(a+b));};

		// all the numerical values are allowed to be slightly off
		if ( relerror(avg, other.avg) > rel_error) return(false);
		if ( relerror(sdm, other.sdm) > rel_error) return(false);

		return(true);
	}

};

/** \brief simple class to compute weighted mean and variance sequentially one value at a time */
template<typename num_t>
class weighted_running_statistics{
  private:
	num_t avg, sdm;
	running_statistics<num_t> weight_stat;

  public:
	weighted_running_statistics(): avg(0), sdm(0), weight_stat() {}
	weighted_running_statistics( num_t m, num_t s, running_statistics<num_t> w_stat):
		avg(m), sdm(s), weight_stat(w_stat) {}


  	/* serialize function */
  	template<class Archive>
	void serialize(Archive & archive) {
		archive( avg, sdm, weight_stat); 
	}

	void push (num_t x, num_t weight){
		if (weight <= 0)
			throw std::runtime_error("Weights have to be strictly positive.");

		// helper
		num_t delta = x - avg;
		// update the weights' sum
		weight_stat.push(weight);
		// adjust mean
		avg += delta * weight / weight_stat.sum();
		// adjust variance
		sdm += weight*delta*(x-avg);
	}

	void pop (num_t x, num_t weight){
		if (weight <= 0)
			throw std::runtime_error("Weights have to be strictly positive.");

		if (weight > weight_stat.sum())
			throw std::runtime_error("Cannot remove item, weight too large.");

		// helper
		num_t delta = (x - avg);
		// update the weights' sum
		weight_stat.pop(weight);
		// adjust mean
		avg -= delta * weight / weight_stat.sum();
		// adjust variance
		sdm -= weight*delta*(x-avg);

		if (sdm < 0)
			throw std::runtime_error("Squared Distance from the mean is now negative; Abort!");
	}

	num_t	squared_deviations_from_the_mean () 			const {return(divide_sdm_by(1,0));}

	num_t	divide_sdm_by(num_t fraction, num_t min_weight) const { return(weight_stat.sum()>min_weight ? std::max<num_t>(0.,sdm/fraction) : NAN);}

	num_t 	mean() 							const	{return(weight_stat.sum()>0?avg:NAN);}
	num_t	sum_of_weights()				const	{return(weight_stat.sum());}
	num_t	sum_of_squares()				const	{return(sdm + sum_of_weights()*mean()*mean());}
	num_t 	variance_population()			const	{return(divide_sdm_by(weight_stat.sum(),0.));}
	// source: https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
	num_t	variance_unbiased_frequency()	const	{return(divide_sdm_by(weight_stat.sum()-1,1.));}
	num_t	variance_unbiased_importance()	const	{return(divide_sdm_by(weight_stat.sum() - (weight_stat.sum_of_squares() / weight_stat.sum()), 0));}


	weighted_running_statistics operator+ ( const weighted_running_statistics &other) const{

		// total weight statistic is trivial
		running_statistics<num_t> total_weight_stat = weight_stat + other.weight_stat;

		num_t sw1(weight_stat.sum()), sw2(other.weight_stat.sum()), swt(total_weight_stat.sum());
		// the total mean is also pretty easy to figure out
		num_t avg_total = avg * (sw1/swt) + other.avg * (sw2/swt);
		// the total squared deviations from the mean look a bit messy, but are straight forwardly derived
		num_t sdm_total = sdm + other.sdm +
							sw1*std::pow(avg-avg_total,2) + sw2*std::pow(other.avg-avg_total,2);

		return(	weighted_running_statistics( avg_total, sdm_total, total_weight_stat));
	}

	weighted_running_statistics& operator+= ( const weighted_running_statistics &other){

		num_t sw1(this->weight_stat.sum());
		num_t sw2(other.weight_stat.sum());
		num_t swt(sw1+sw2);
		// the total mean is also pretty easy to figure out
		num_t avg_total = this->avg * (sw1/swt) + other.avg * (sw2/swt);
		// the total squared deviations from the mean look a bit messy, but are straight forwardly derived
		num_t sdm_total = this->sdm + other.sdm +
							sw1*std::pow(this->avg-avg_total,2) + sw2*std::pow(other.avg-avg_total,2);

		this->avg = avg_total;
		this->sdm = sdm_total;
		this->weight_stat +=other.weight_stat;

		return(*this);
	}

	weighted_running_statistics operator-  ( const weighted_running_statistics &other) const{

		if (other.weight_stat.sum() >= weight_stat.sum())
			throw std::runtime_error("Second statistics must not have a greater sum of weights!");

		// total weight statistic is trivial
		running_statistics<num_t> weight_stat_total = weight_stat - other.weight_stat;

		num_t sw1(weight_stat.sum()), sw2(other.weight_stat.sum()), swt(weight_stat_total.sum());

		// the total mean is also pretty easy to figure out
		num_t avg_total = avg * (sw1/swt) - other.avg * (sw2/swt);

		// the total squared deviations from the mean look a bit messy, but are straight forwardly derived
		num_t sdm_total = sdm - other.sdm -
							swt*std::pow(avg-avg_total,2) - sw2*std::pow(other.avg-avg,2);

		return	(weighted_running_statistics(avg_total, sdm_total, weight_stat_total));
	}

	weighted_running_statistics& operator-=( const weighted_running_statistics &other) {

		if (other.weight_stat.sum() >= weight_stat.sum())
			throw std::runtime_error("Second statistics must not have a greater sum of weights!");

		num_t sw1(weight_stat.sum()), sw2(other.weight_stat.sum()), swt(sw1-sw2);

		// the total mean is also pretty easy to figure out
		num_t avg_total = avg * (sw1/swt) - other.avg * (sw2/swt);

		// the total squared deviations from the mean look a bit messy, but are straight forwardly derived
		num_t sdm_total = sdm - other.sdm -
							swt*std::pow(avg-avg_total,2) - sw2*std::pow(other.avg-avg,2);

		avg = avg_total;
		sdm = sdm_total;
		weight_stat -= other.weight_stat;
		return(*this);
	}

	weighted_running_statistics operator* (const num_t a) const{
		return(weighted_running_statistics(a*avg, a*a*sdm, weight_stat));
	}

	weighted_running_statistics operator+ (const num_t a) const{
		return(weighted_running_statistics(a+avg, sdm, weight_stat));
	}

	weighted_running_statistics multiply_weights_by ( const num_t a) const{
		return(weighted_running_statistics(avg, a*sdm, weight_stat*a));
	}

	bool numerically_equal (weighted_running_statistics other, num_t rel_error){

		// inline lambda expression for the relative error
		auto relerror = [] (num_t a, num_t b) {return(std::abs(a-b)/(a+b));};

		// all the numerical values are allowed to be slightly off
		if ( relerror(avg, other.avg) > rel_error) return(false);
		if ( relerror(sdm, other.sdm) > rel_error) return(false);

		// finally compare the weight statistics
		return( weight_stat.numerically_equal(other.weight_stat, rel_error));
	}
	
	running_statistics<num_t> get_weight_statistics() const { return(weight_stat);}
	
};

template <typename num_t>
class running_covariance{
  private:
	long unsigned int N;
	num_t m1, m2;
	num_t cov;

  public:
	running_covariance(): N(0), m1(0), m2(0), cov(0) {}

	void push (num_t x1, num_t x2){
		N++;
		num_t delta1 = (x1-m1)/N;
		m1 += delta1;
		num_t delta2 = (x2-m2)/N;
		m2 += delta2;

		cov += (N-1) * delta1 * delta2 - cov/N;
	}

	long unsigned int number_of_points(){return(N);}
	num_t covariance(){return(num_t(N)/num_t(N-1)*cov);}
};






}}//namespace rfr::util
#endif
