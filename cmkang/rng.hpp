#ifndef fincu_rng_hpp
#define fincu_rng_hpp

#include <cuda.h>
#include <curand.h>

#include <thrust\host_vector.h>
#include <thrust\device_vector.h>

namespace FinCu {
	
	template <typename Real>
	class RandomNumberGenerator {
	  public:
		typedef typename thrust::host_vector<Real>::iterator 
									host_vector_iterator_type;

		enum Distribution {Uniform, Normal, LogNormal};

		RandomNumberGenerator(
			curandRngType_t generatorType 
					= CURAND_RNG_PSEUDO_DEFAULT,
			unsigned long long seed = 1234ULL,
			unsigned int bufferSize = 1048576);
		~RandomNumberGenerator();

		Real operator() (
			Distribution dist = Uniform, 
			Real mu = 0.0, Real sigma = 1.0);

		thrust::host_vector<Real> nextRandomSequence(
				unsigned int n, Distribution dist = Uniform,
				Real mu = 0.0, Real sigma = 1.0);

	  private:
		void updateBuffer();
		
		thrust::host_vector<Real> hostBuffer_;
		host_vector_iterator_type iter_;

		unsigned int bufferSize_;
		curandGenerator_t gen_;
	};
}

#endif
