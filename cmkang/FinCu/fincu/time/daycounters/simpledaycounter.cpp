#include <fincu/time/daycounters/simpledaycounter.hpp>
#include <fincu/time/daycounters/thirty360.hpp>

namespace FinCu {

    namespace { DayCounter fallback = Thirty360(); }

    BigInteger SimpleDayCounter::Impl::dayCount(const Date& d1,
                                                const Date& d2) const {
        return fallback.dayCount(d1,d2);
    }

    Time SimpleDayCounter::Impl::yearFraction(const Date& d1,
                                              const Date& d2,
                                              const Date&,
                                              const Date&) const {
        Day dm1 = d1.dayOfMonth(),
            dm2 = d2.dayOfMonth();

        if (dm1 == dm2 ||
            // e.g., Aug 30 -> Feb 28 ?
            (dm1 > dm2 && Date::isEndOfMonth(d2)) ||
            // e.g., Feb 28 -> Aug 30 ?
            (dm1 < dm2 && Date::isEndOfMonth(d1))) {

            return (d2.year()-d1.year()) +
                (Integer(d2.month())-Integer(d1.month()))/12.0f;

        } else {
            return fallback.yearFraction(d1,d2);
        }
    }

}

