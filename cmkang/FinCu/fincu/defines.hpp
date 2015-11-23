#ifndef fincu_defines_hpp
#define fincu_defines_hpp

#include <math_constants.h>

#define FC_EPSILON             CUDART_SQRT_HALF_LO_F
#define FC_NULL_INTEGER        4294967295 
#define FC_NULL_REAL           CUDART_NORM_HUGE_F

namespace FinCu {
	typedef size_t Size;
	typedef float Time;
	typedef int Integer;
	typedef long BigInteger;
}

#endif