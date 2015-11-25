#include <wiener/time/weekday.hpp>
#include <wiener/errors.hpp>

namespace Wiener {

    // weekday formatting

    std::ostream& operator<<(std::ostream& out, const Weekday& w) {
        return out << io::long_weekday(w);
    }

    namespace detail {

        std::ostream& operator<<(std::ostream& out,
                                 const long_weekday_holder& holder) {
            switch (holder.d) {
              case Sunday:
                return out << "Sunday";
              case Monday:
                return out << "Monday";
              case Tuesday:
                return out << "Tuesday";
              case Wednesday:
                return out << "Wednesday";
              case Thursday:
                return out << "Thursday";
              case Friday:
                return out << "Friday";
              case Saturday:
                return out << "Saturday";
              default:
                WI_FAIL("unknown weekday");
            }
        }

        std::ostream& operator<<(std::ostream& out,
                                 const short_weekday_holder& holder) {
            switch (holder.d) {
              case Sunday:
                return out << "Sun";
              case Monday:
                return out << "Mon";
              case Tuesday:
                return out << "Tue";
              case Wednesday:
                return out << "Wed";
              case Thursday:
                return out << "Thu";
              case Friday:
                return out << "Fri";
              case Saturday:
                return out << "Sat";
              default:
                WI_FAIL("unknown weekday");
            }
        }

        std::ostream& operator<<(std::ostream& out,
                                 const shortest_weekday_holder& holder) {
            switch (holder.d) {
              case Sunday:
                return out << "Su";
              case Monday:
                return out << "Mo";
              case Tuesday:
                return out << "Tu";
              case Wednesday:
                return out << "We";
              case Thursday:
                return out << "Th";
              case Friday:
                return out << "Fr";
              case Saturday:
                return out << "Sa";
              default:
                WI_FAIL("unknown weekday");
            }
        }

    }

    namespace io {

        detail::long_weekday_holder long_weekday(Weekday d) {
            return detail::long_weekday_holder(d);
        }

        detail::short_weekday_holder short_weekday(Weekday d) {
            return detail::short_weekday_holder(d);
        }

        detail::shortest_weekday_holder shortest_weekday(Weekday d) {
            return detail::shortest_weekday_holder(d);
        }

    }

}
