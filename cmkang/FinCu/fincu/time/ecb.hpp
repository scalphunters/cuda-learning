#ifndef fincu_ecb_hpp
#define fincu_ecb_hpp

#include <fincu/time/date.hpp>
#include <set>
#include <vector>

namespace FinCu {

    //! European Central Bank reserve maintenance dates
    struct ECB {

        static const std::set<Date>& knownDates();
        static void addDate(const Date& d);
        static void removeDate(const Date& d);

        //! maintenance period start date in the given month/year
        static Date date(Month m,
                         Year y) { return nextDate(Date(1, m, y) - 1); }

        /*! returns the ECB date for the given ECB code
            (e.g. March xxth, 2013 for MAR10).

            \warning It raises an exception if the input
                     string is not an ECB code
        */
        static Date date(const std::string& ecbCode,
                         const Date& referenceDate = Date());

        /*! returns the ECB code for the given date
            (e.g. MAR10 for March xxth, 2010).

            \warning It raises an exception if the input
                     date is not an ECB date
        */
        static std::string code(const Date& ecbDate);

        //! next maintenance period start date following the given date
        static Date nextDate(const Date& d = Date());

        //! next maintenance period start date following the given ECB code
        static Date nextDate(const std::string& ecbCode,
                             const Date& referenceDate = Date()) {
            return nextDate(date(ecbCode, referenceDate));
        }

        //! next maintenance period start dates following the given date
        static std::vector<Date> nextDates(const Date& d = Date());

        //! next maintenance period start dates following the given code
        static std::vector<Date> nextDates(const std::string& ecbCode,
                                           const Date& referenceDate = Date()) {
            return nextDates(date(ecbCode, referenceDate));
        }

        /*! returns whether or not the given date is
            a maintenance period start date */
        static bool isECBdate(const Date& d) {
            Date date = nextDate(d-1);
            return d==date;
        }

        //! returns whether or not the given string is an ECB code
        static bool isECBcode(const std::string& in);

        //! next ECB code following the given date
        static std::string nextCode(const Date& d = Date()) {
            return code(nextDate(d));
        }

        //! next ECB code following the given code
        static std::string nextCode(const std::string& ecbCode);

    };

}

#endif
