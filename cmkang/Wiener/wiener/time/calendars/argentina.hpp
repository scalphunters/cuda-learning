#ifndef wiener_argentinian_calendar_hpp
#define wiener_argentinian_calendar_hpp

#include <wiener/time/calendar.hpp>

namespace Wiener {

    //! Argentinian calendars
    /*! Holidays for the Buenos Aires stock exchange
        (data from <http://www.merval.sba.com.ar/>):
        <ul>
        <li>Saturdays</li>
        <li>Sundays</li>
        <li>New Year's Day, January 1st</li>
        <li>Holy Thursday</li>
        <li>Good Friday</li>
        <li>Labour Day, May 1st</li>
        <li>May Revolution, May 25th</li>
        <li>Death of General Manuel Belgrano, third Monday of June</li>
        <li>Independence Day, July 9th</li>
        <li>Death of General Jos?de San Mart?, third Monday of August</li>
        <li>Columbus Day, October 12th (moved to preceding Monday if
            on Tuesday or Wednesday and to following if on Thursday
            or Friday)</li>
        <li>Immaculate Conception, December 8th</li>
        <li>Christmas Eve, December 24th</li>
        <li>New Year's Eve, December 31th</li>
        </ul>

        \ingroup calendars
    */
    class Argentina : public Calendar {
      private:
        class MervalImpl : public Calendar::WesternImpl {
          public:
            std::string name() const { return "Buenos Aires stock exchange"; }
            bool isBusinessDay(const Date&) const;
        };
      public:
        enum Market { Merval   //!< Buenos Aires stock exchange calendar
        };
        Argentina(Market m = Merval);
    };

}


#endif
