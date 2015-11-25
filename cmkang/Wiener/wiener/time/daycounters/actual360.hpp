#ifndef wiener_actual360_day_counter_h
#define wiener_actual360_day_counter_h

#include <wiener/time/daycounter.hpp>

namespace Wiener {

    //! Actual/360 day count convention

    /*! Actual/360 day count convention, also known as "Act/360", or "A/360".

        \ingroup daycounters
    */
    class Actual360 : public DayCounter {
      private:
        class Impl : public DayCounter::Impl {
          public:
            std::string name() const { return std::string("Actual/360"); }
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date&,
                              const Date&) const {
                return daysBetween(d1,d2)/360.0;
            }
        };
      public:
        Actual360()
        : DayCounter(std::shared_ptr<DayCounter::Impl>(
                                                      new Actual360::Impl)) {}
    };

}

#endif
