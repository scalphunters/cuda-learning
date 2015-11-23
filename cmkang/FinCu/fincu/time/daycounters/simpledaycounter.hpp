#ifndef fincu_simple_day_counter_hpp
#define fincu_simple_day_counter_hpp

#include <fincu/time/daycounter.hpp>

namespace FinCu {

    //! Simple day counter for reproducing theoretical calculations.
    /*! This day counter tries to ensure that whole-month distances
        are returned as a simple fraction, i.e., 1 year = 1.0,
        6 months = 0.5, 3 months = 0.25 and so forth.

        \warning this day counter should be used together with
                 NullCalendar, which ensures that dates at whole-month
                 distances share the same day of month. It is <b>not</b>
                 guaranteed to work with any other calendar.

        \ingroup daycounters

        \test the correctness of the results is checked against known
              good values.
    */
    class SimpleDayCounter : public DayCounter {
      private:
        class Impl : public DayCounter::Impl {
          public:
            std::string name() const { return "Simple"; }
            BigInteger dayCount(const Date& d1,
                                const Date& d2) const;
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date&,
                              const Date&) const;
        };
      public:
        SimpleDayCounter()
        : DayCounter(std::shared_ptr<DayCounter::Impl>(
                                             new SimpleDayCounter::Impl())) {}
    };

}

#endif
