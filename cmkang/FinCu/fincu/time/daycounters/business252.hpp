#ifndef fincu_business252_day_counter_hpp
#define fincu_business252_day_counter_hpp

#include <fincu/time/daycounter.hpp>
#include <fincu/time/calendar.hpp>
#include <fincu/time/calendars/brazil.hpp>

namespace FinCu {

    //! Business/252 day count convention
    /*! \ingroup daycounters */
    class Business252 : public DayCounter {
      private:
        class Impl : public DayCounter::Impl {
          private:
            Calendar calendar_;
          public:
            std::string name() const;
            BigInteger dayCount(const Date& d1,
                                const Date& d2) const;
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date&,
                              const Date&) const;
            Impl(Calendar c) { calendar_ = c; }
        };
      public:
        Business252(Calendar c = Brazil())
        : DayCounter(std::shared_ptr<DayCounter::Impl>(
                                                 new Business252::Impl(c))) {}
    };

}

#endif
