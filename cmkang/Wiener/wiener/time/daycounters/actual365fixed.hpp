#ifndef wiener_actual365fixed_day_counter_h
#define wiener_actual365fixed_day_counter_h

#include <wiener/time/daycounter.hpp>

namespace Wiener {

    //! Actual/365 (Fixed) day count convention
    /*! "Actual/365 (Fixed)" day count convention, also know as
        "Act/365 (Fixed)", "A/365 (Fixed)", or "A/365F".

        \warning According to ISDA, "Actual/365" (without "Fixed") is
                 an alias for "Actual/Actual (ISDA)" (see
                 ActualActual.)  If Actual/365 is not explicitly
                 specified as fixed in an instrument specification,
                 you might want to double-check its meaning.

        \ingroup daycounters
    */
    class Actual365Fixed : public DayCounter {
      private:
        class Impl : public DayCounter::Impl {
          public:
            std::string name() const { return std::string("Actual/365 (Fixed)"); }
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date&,
                              const Date&) const {
                return daysBetween(d1,d2)/365.0;
            }
        };
      public:
        Actual365Fixed()
        : DayCounter(std::shared_ptr<DayCounter::Impl>(
                                                 new Actual365Fixed::Impl)) {}
    };

}

#endif
