#ifndef wiener_defines_hpp
#define wiener_defines_hpp

#include <math_constants.h>

#define WI_EPSILON             CUDART_SQRT_HALF_LO_F
#define WI_NULL_INTEGER        4294967295 
#define WI_NULL_REAL           CUDART_NORM_HUGE_F

namespace Wiener {
	typedef size_t Size;
	typedef float Time;
	typedef int Integer;
	typedef long BigInteger;
}

#endif