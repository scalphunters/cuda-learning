#include <wiener/time/timeunit.hpp>
#include <wiener/errors.hpp>

namespace Wiener {

    // timeunit formatting

    std::ostream& operator<<(std::ostream& out, const TimeUnit& timeunit) {
        switch (timeunit) {
            case Years:
                return out << "Years";
            case Months:
                return out << "Months";
            case Weeks:
                return out << "Weeks";
            case Days:
                return out << "Days";
            case Hours:
                return out << "Hours";
            case Minutes:
                return out << "Minutes";
            case Seconds:
                return out << "Seconds";
            case Milliseconds:
                return out << "Milliseconds";
            case Microseconds:
                return out << "Microseconds";
            default:
                WI_FAIL("unknown TimeUnit");
        }
    }

}
