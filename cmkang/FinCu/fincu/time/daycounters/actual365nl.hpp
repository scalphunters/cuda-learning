#ifndef fincu_actual365nl_h
#define fincu_actual365nl_h

#include <fincu/time/daycounter.hpp>

namespace FinCu {

    //! Actual/365 (No Leap) day count convention
    /*! "Actual/365 (No Leap)" day count convention, also known as
        "Act/365 (NL)", "NL/365", or "Actual/365 (JGB)".

        \ingroup daycounters
    */
    class Actual365NoLeap : public DayCounter {
    private:
        class Impl : public DayCounter::Impl {
        public:
            std::string name() const { return std::string("Actual/365 (NL)"); }

            // Returns the exact number of days between 2 dates, excluding leap days
            BigInteger dayCount(const Date& d1,
                                const Date& d2) const {

                static const Integer MonthOffset[] = {
                    0,  31,  59,  90, 120, 151,  // Jan - Jun
                  181, 212, 243, 273, 304, 334   // Jun - Dec
                };
                BigInteger s1, s2;

                s1 = d1.dayOfMonth() + MonthOffset[d1.month()-1] + (d1.year() * 365);
                s2 = d2.dayOfMonth() + MonthOffset[d2.month()-1] + (d2.year() * 365);

                if (d1.month() == Feb && d1.dayOfMonth() == 29)
                {
                    --s1;
                }

                if (d2.month() == Feb && d2.dayOfMonth() == 29)
                {
                    --s2;
                }

                return s2 - s1;
            }

            FinCu::Time yearFraction(const Date& d1,
                                        const Date& d2,
                                        const Date& d3,
                                        const Date& d4) const {
                return dayCount(d1, d2)/365.0;
            }
        };
    public:
        Actual365NoLeap()
        : DayCounter(std::shared_ptr<DayCounter::Impl>(
                                                new Actual365NoLeap::Impl)) {}
    };

}

#endif

