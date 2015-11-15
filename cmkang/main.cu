#include "rng.hpp"

using namespace FinCu;

typedef float Real;

int main(void) { 
	RandomNumberGenerator<Real> rng(CURAND_RNG_PSEUDO_XORWOW, 1234ULL, 16777216);

	thrust::host_vector<Real> rsq 
		= rng.nextRandomSequence(100, RandomNumberGenerator<Real>::Normal);

	//for(unsigned int i = 0; i < 100; ++i)
	//	std::cout<<rsq[i]<<std::endl;

	//std::cin.get();
	return 0; 
}

