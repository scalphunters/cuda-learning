#ifndef fincu_asx_hpp
#define fincu_asx_hpp

#include <fincu/time/date.hpp>

namespace FinCu {

    //! Main cycle of the Australian Securities Exchange (a.k.a. ASX) months
    struct ASX {
        enum Month { F =  1, G =  2, H =  3,
                     J =  4, K =  5, M =  6,
                     N =  7, Q =  8, U =  9,
                     V = 10, X = 11, Z = 12 };

        //! returns whether or not the given date is an ASX date
        static bool isASXdate(const Date& d,
                              bool mainCycle = true);

        //! returns whether or not the given string is an ASX code
        static bool isASXcode(const std::string& in,
                              bool mainCycle = true);

        /*! returns the ASX code for the given date
            (e.g. M5 for June 12th, 2015).

            \warning It raises an exception if the input
                     date is not an ASX date
        */
        static std::string code(const Date& asxDate);

        /*! returns the ASX date for the given ASX code
            (e.g. June 12th, 2015 for M5).

            \warning It raises an exception if the input
                     string is not an ASX code
        */
        static Date date(const std::string& asxCode,
                         const Date& referenceDate = Date());

        //! next ASX date following the given date
        /*! returns the 1st delivery date for next contract listed in the
            Australian Securities Exchange.
        */
        static Date nextDate(const Date& d = Date(),
                             bool mainCycle = true);

        //! next ASX date following the given ASX code
        /*! returns the 1st delivery date for next contract listed in the
            Australian Securities Exchange
        */
        static Date nextDate(const std::string& asxCode,
                             bool mainCycle = true,
                             const Date& referenceDate = Date());

        //! next ASX code following the given date
        /*! returns the ASX code for next contract listed in the
            Australian Securities Exchange
        */
        static std::string nextCode(const Date& d = Date(),
                                    bool mainCycle = true);

        //! next ASX code following the given code
        /*! returns the ASX code for next contract listed in the
            Australian Securities Exchange
        */
        static std::string nextCode(const std::string& asxCode,
                                    bool mainCycle = true,
                                    const Date& referenceDate = Date());
    };

}

#endif
