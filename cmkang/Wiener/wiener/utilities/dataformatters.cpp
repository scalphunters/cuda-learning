#include <ostream>

#include <wiener/defines.hpp>
#include <wiener/utilities/dataformatters.hpp>

namespace Wiener {

    namespace detail {

        std::ostream& operator<<(std::ostream& out,
                                 const ordinal_holder& holder) {
            Size n = holder.n;
            out << n;
            if (n == Size(11) || n == Size(12) || n == Size(13)) {
                out << "th";
            } else {
                switch (n % 10) {
                  case 1:  out << "st";  break;
                  case 2:  out << "nd";  break;
                  case 3:  out << "rd";  break;
                  default: out << "th";
                }
            }
            return out;
        }

		template <typename Real>
        std::ostream& operator<<(std::ostream& out,
                                 const percent_holder<Real>& holder) {
            std::ios::fmtflags flags = out.flags();
            Size width = (Size)out.width();
            if (width > 2)
                out.width(width-2); // eat space used by percent sign
            out << std::fixed;
            if (holder.value == Null<Real>())
                out << "null";
            else
                out << holder.value * 100.0 << " %";
            out.flags(flags);
            return out;
        }

    }

}

