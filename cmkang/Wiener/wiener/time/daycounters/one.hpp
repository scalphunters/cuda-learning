#ifndef wiener_one_day_counter_h
#define wiener_one_day_counter_h

#include <wiener/time/daycounter.hpp>

namespace Wiener {

    //! 1/1 day count convention
    /*! \ingroup daycounters */
    class OneDayCounter : public DayCounter {
      private:
        class Impl : public DayCounter::Impl {
          public:
            std::string name() const { return std::string("1/1"); }
            BigInteger dayCount(const Date& d1, const Date& d2) const {
                // the sign is all we need
                return (d2 >= d1 ? 1 : -1);
            };
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date&,
                              const Date&) const {
                return Time(dayCount(d1, d2));
            }
        };
      public:
        OneDayCounter()
        : DayCounter(std::shared_ptr<DayCounter::Impl>(
                                        new OneDayCounter::Impl)) {}
    };

}

#endif
