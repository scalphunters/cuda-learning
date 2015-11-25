#ifndef wiener_comparison_hpp
#define wiener_comparison_hpp

#include <memory>
#include <cuda_runtime.h>
#include <math_functions.h>

#include <wiener/defines.hpp>

namespace Wiener {

	template <typename Real>
	__host__ __device__
    bool close_enough(Real x, Real y);

	template <typename Real>
	__host__ __device__
    bool close_enough(Real x, Real y, Size n);

	template <typename Real>
	__host__ __device__
    inline bool close_enough(Real x, Real y) {
        return close_enough(x,y,42);
    }

	template <typename Real>
	__host__ __device__
    inline bool close_enough(Real x, Real y, Size n) {
        // Deals with +infinity and -infinity representations etc.
        if (x == y)
            return true;

        Real diff = fabs(x-y), tolerance = n * WI_EPSILON;

        if (x * y == 0.0) // x or y = 0.0
            return diff < (tolerance * tolerance);

        return diff <= tolerance*fabs(x) ||
               diff <= tolerance*fabs(y);
    }

	template <typename Real>
	struct CloseEnough 
		: thrust::binary_function<Real, Real, bool>
	{
		__host__ __device__
		bool operator() (Real x, Real y) {return close_enough(x, y);}
	};
}


#endif
