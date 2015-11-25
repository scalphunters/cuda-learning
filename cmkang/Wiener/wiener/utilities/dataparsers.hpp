#ifndef wiener_data_parsers_hpp
#define wiener_data_parsers_hpp

#include <wiener/defines.hpp>
#include <wiener/time/date.hpp>
#include <vector>

namespace Wiener {

    namespace io {

        Integer to_integer(const std::string&);

    }

    class PeriodParser {
      public:
        static Period parse(const std::string& str);
      private:
        static Period parseOnePeriod(const std::string& str);
    };

    class DateParser {
      public:
        static std::vector<std::string> split(const std::string& str,
                                              char delim);

        //! Parses a string in a used-defined format.
        /*! This method uses the parsing functions from
            Boost.Date_Time and supports the same formats.
        */
        static Date parseFormatted(const std::string& str,
                                   const std::string& fmt);
        static Date parseISO(const std::string& str);
    };

}


#endif
