#ifndef wiener_null_hpp
#define wiener_null_hpp

#include <type_traits>
#include <wiener\defines.hpp>

namespace Wiener {

    //! template class providing a null value for a given type.
    template <class Type>
    class Null;

    namespace detail {

        template <bool>
        struct FloatingPointNull;

        // null value for floating-point types
        template <>
        struct FloatingPointNull<true> {
            static float nullValue() {
                return WI_NULL_REAL;
            }
        };

        // null value for integer types
        template <>
        struct FloatingPointNull<false> {
            static int nullValue() {
                return WI_NULL_INTEGER;
            }
        };

    }

    // default implementation for built-in types
    template <typename T>
    class Null {
      public:
        Null() {}
        operator T() const {
            return T(detail::FloatingPointNull<
                         std::is_floating_point<T>::value>::nullValue());
        }
    };

}


#endif
