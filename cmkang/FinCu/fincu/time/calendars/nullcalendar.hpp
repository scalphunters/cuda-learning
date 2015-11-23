#ifndef fincu_null_calendar_hpp
#define fincu_null_calendar_hpp

#include <fincu/time/calendar.hpp>

namespace FinCu {

    //! %Calendar for reproducing theoretical calculations.
    /*! This calendar has no holidays. It ensures that dates at
        whole-month distances have the same day of month.

        \ingroup calendars
    */
    class NullCalendar : public Calendar {
      private:
        class Impl : public Calendar::Impl {
          public:
            std::string name() const { return "Null"; }
            bool isWeekend(Weekday) const { return false; }
            bool isBusinessDay(const Date&) const { return true; }
        };
      public:
        NullCalendar() {
            impl_ = std::shared_ptr<Calendar::Impl>(new NullCalendar::Impl);
        }
    };

}


#endif
