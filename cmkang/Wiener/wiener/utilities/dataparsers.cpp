#include <wiener/utilities/dataparsers.hpp>
#include <wiener/utilities/null.hpp>
#include <wiener/time/period.hpp>
#include <wiener/errors.hpp>
//#include <boost/lexical_cast.hpp>
//#include <boost/algorithm/string/case_conv.hpp>
//#include <boost/date_time/gregorian/gregorian.hpp>
#include <locale>
#include <cctype>

namespace Wiener {

    namespace io {

        Integer to_integer(const std::string& str) {
            return std::atoi(str.c_str());
        }

    }

    Period PeriodParser::parse(const std::string& str) {
        WI_REQUIRE(str.length()>1, "period string length must be at least 2");

        std::vector<std::string > subStrings;
        std::string reducedString = str;

        Size iPos, reducedStringDim = 100000, max_iter = 0;
        while (reducedStringDim>0) {
            iPos = reducedString.find_first_of("DdWwMmYy");
            Size subStringDim = iPos+1;
            reducedStringDim = reducedString.length()-subStringDim;
            subStrings.push_back(reducedString.substr(0, subStringDim));
            reducedString = reducedString.substr(iPos+1, reducedStringDim);
            ++max_iter;
            WI_REQUIRE(max_iter<str.length(), "unknown '" << str << "' unit");
        }

        Period result = parseOnePeriod(subStrings[0]);
        for (Size i=1; i<subStrings.size(); ++i)
            result += parseOnePeriod(subStrings[i]);
        return result;
    }

    Period PeriodParser::parseOnePeriod(const std::string& str) {
        WI_REQUIRE(str.length()>1, "single period require a string of at "
                   "least 2 characters");

        Size iPos = str.find_first_of("DdWwMmYy");
        WI_REQUIRE(iPos==str.length()-1, "unknown '" <<
                   str.substr(str.length()-1, str.length()) << "' unit");
        TimeUnit units = Days;
        char abbr = static_cast<char>(std::toupper(str[iPos]));
        if      (abbr == 'D') units = Days;
        else if (abbr == 'W') units = Weeks;
        else if (abbr == 'M') units = Months;
        else if (abbr == 'Y') units = Years;

        Size nPos = str.find_first_of("-+0123456789");
        WI_REQUIRE(nPos<iPos, "no numbers of " << units << " provided");
        Integer n;
        try {
            n = io::to_integer(str.substr(nPos,iPos));
                //boost::lexical_cast<Integer>(str.substr(nPos,iPos));
        } catch (std::exception& e) {
            WI_FAIL("unable to parse the number of units of " << units <<
                    " in '" << str << "'. Error:" << e.what());
        }

        return Period(n, units);
    }

    std::vector<std::string> DateParser::split(const std::string& str,
                                               char delim) {
        std::vector<std::string> list;
        Size sx= str.find(delim), so=0;

        while (sx != std::string::npos) {
            list.push_back(str.substr(so,sx));
            so += sx+1;
            sx = str.substr(so).find(delim);
        }
        list.push_back(str.substr(so));
        return list;
    }

    Date DateParser::parseISO(const std::string& str) {
        WI_REQUIRE(str.size() == 10 && str[4] == '-' && str[7] == '-',
                   "invalid format");
        Integer year = //boost::lexical_cast<Integer>(str.substr(0, 4));
            io::to_integer(str.substr(0, 4));
        Month month =
            //  static_cast<Month>(boost::lexical_cast<Integer>(str.substr(5, 2)));
            static_cast<Month>(io::to_integer(str.substr(5, 2)));
        Integer day = //boost::lexical_cast<Integer>(str.substr(8, 2));
            static_cast<Month>(io::to_integer(str.substr(8, 2)));

        return Date(day, month, year);
    }

}
