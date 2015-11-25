#include <algorithm>
#include <wiener/time/daycounters/thirty360.hpp>

namespace Wiener {

    std::shared_ptr<DayCounter::Impl>
    Thirty360::implementation(Thirty360::Convention c) {
        switch (c) {
          case USA:
          case BondBasis:
            return std::shared_ptr<DayCounter::Impl>(new US_Impl);
          case European:
          case EurobondBasis:
            return std::shared_ptr<DayCounter::Impl>(new EU_Impl);
          case Italian:
            return std::shared_ptr<DayCounter::Impl>(new IT_Impl);
          default:
            WI_FAIL("unknown 30/360 convention");
        }
    }

    BigInteger Thirty360::US_Impl::dayCount(const Date& d1,
                                            const Date& d2) const {
        Day dd1 = d1.dayOfMonth(), dd2 = d2.dayOfMonth();
        Integer mm1 = d1.month(), mm2 = d2.month();
        Year yy1 = d1.year(), yy2 = d2.year();

        if (dd2 == 31 && dd1 < 30) { dd2 = 1; mm2++; }

        return 360*(yy2-yy1) + 30*(mm2-mm1-1) +
            std::max(Integer(0),30-dd1) + std::min(Integer(30),dd2);
    }

    BigInteger Thirty360::EU_Impl::dayCount(const Date& d1,
                                            const Date& d2) const {
        Day dd1 = d1.dayOfMonth(), dd2 = d2.dayOfMonth();
        Month mm1 = d1.month(), mm2 = d2.month();
        Year yy1 = d1.year(), yy2 = d2.year();

        return 360*(yy2-yy1) + 30*(mm2-mm1-1) +
            std::max(Integer(0),30-dd1) + std::min(Integer(30),dd2);
    }

    BigInteger Thirty360::IT_Impl::dayCount(const Date& d1,
                                            const Date& d2) const {
        Day dd1 = d1.dayOfMonth(), dd2 = d2.dayOfMonth();
        Month mm1 = d1.month(), mm2 = d2.month();
        Year yy1 = d1.year(), yy2 = d2.year();

        if (mm1 == 2 && dd1 > 27) dd1 = 30;
        if (mm2 == 2 && dd2 > 27) dd2 = 30;

        return 360*(yy2-yy1) + 30*(mm2-mm1-1) +
            std::max(Integer(0),30-dd1) + std::min(Integer(30),dd2);
    }

}
