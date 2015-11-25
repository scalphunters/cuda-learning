#include <cctype>
#include <algorithm>
#include <wiener/time/asx.hpp>
#include <wiener/errors.hpp>
#include <wiener/utilities/dataparsers.hpp>

using std::string;

namespace Wiener {

    bool ASX::isASXdate(const Date& date, bool mainCycle) {
        if (date.weekday()!=Friday)
            return false;

        Day d = date.dayOfMonth();
        if (d<8 || d>14)
            return false;

        if (!mainCycle) return true;

        switch (date.month()) {
          case March:
          case June:
          case September:
          case December:
            return true;
          default:
            return false;
        }
    }

    bool ASX::isASXcode(const std::string& in, bool mainCycle) {
        if (in.length() != 2)
            return false;

        string str1("0123456789");
        string::size_type loc = str1.find(in.substr(1,1), 0);
        if (loc == string::npos)
            return false;

        if (mainCycle) str1 = "hmzuHMZU";
        else           str1 = "fghjkmnquvxzFGHJKMNQUVXZ";
        loc = str1.find(in.substr(0,1), 0);
        if (loc == string::npos)
            return false;

        return true;
    }

    std::string ASX::code(const Date& date) {
        WI_REQUIRE(isASXdate(date, false),
                   date << " is not an ASX date");

        std::ostringstream ASXcode;
        unsigned int y = date.year() % 10;
        switch(date.month()) {
          case January:
            ASXcode << 'F' << y;
            break;
          case February:
            ASXcode << 'G' << y;
            break;
          case March:
            ASXcode << 'H' << y;
            break;
          case April:
            ASXcode << 'J' << y;
            break;
          case May:
            ASXcode << 'K' << y;
            break;
          case June:
            ASXcode << 'M' << y;
            break;
          case July:
            ASXcode << 'N' << y;
            break;
          case August:
            ASXcode << 'Q' << y;
            break;
          case September:
            ASXcode << 'U' << y;
            break;
          case October:
            ASXcode << 'V' << y;
            break;
          case November:
            ASXcode << 'X' << y;
            break;
          case December:
            ASXcode << 'Z' << y;
            break;
          default:
            WI_FAIL("not an ASX month (and it should have been)");
        }

        return ASXcode.str();
    }

    Date ASX::date(const std::string& asxCode,
                   const Date& refDate) {
        WI_REQUIRE(isASXcode(asxCode, false),
                   asxCode << " is not a valid ASX code");
		WI_REQUIRE(refDate != Date(), refDate << ": null date given");

		Date referenceDate = refDate;

		std::string code = asxCode;
		std::transform(code.begin(), code.end(), code.begin(), std::toupper);

        std::string ms = code.substr(0,1);
        Wiener::Month m;
        if (ms=="F")      m = January;
        else if (ms=="G") m = February;
        else if (ms=="H") m = March;
        else if (ms=="J") m = April;
        else if (ms=="K") m = May;
        else if (ms=="M") m = June;
        else if (ms=="N") m = July;
        else if (ms=="Q") m = August;
        else if (ms=="U") m = September;
        else if (ms=="V") m = October;
        else if (ms=="X") m = November;
        else if (ms=="Z") m = December;
        else WI_FAIL("invalid ASX month letter");

//        Year y = boost::lexical_cast<Year>(); // lexical_cast causes compilation errors with x64

        Year y= io::to_integer(code.substr(1,1));
        /* year<1900 are not valid QuantLib years: to avoid a run-time
           exception few lines below we need to add 10 years right away */
        if (y==0 && referenceDate.year()<=1909) y+=10;
        Year referenceYear = (referenceDate.year() % 10);
        y += referenceDate.year() - referenceYear;
        Date result = ASX::nextDate(Date(1, m, y), false);
        if (result<referenceDate)
            return ASX::nextDate(Date(1, m, y+10), false);

        return result;
    }

    Date ASX::nextDate(const Date& date, bool mainCycle) {
		WI_REQUIRE(date != Date(), date << ": null date given");
        Date refDate = date;

        Year y = refDate.year();
        Wiener::Month m = refDate.month();

        Size offset = mainCycle ? 3 : 1;
        Size skipMonths = offset-(m%offset);
        if (skipMonths != offset || refDate.dayOfMonth() > 14) {
            skipMonths += Size(m);
            if (skipMonths<=12) {
                m = Wiener::Month(skipMonths);
            } else {
                m = Wiener::Month(skipMonths-12);
                y += 1;
            }
        }

        Date result = Date::nthWeekday(2, Friday, m, y);
        if (result<=refDate)
            result = nextDate(Date(15, m, y), mainCycle);
        return result;
    }

    Date ASX::nextDate(const std::string& ASXcode,
                       bool mainCycle,
                       const Date& referenceDate)  {
        Date asxDate = date(ASXcode, referenceDate);
        return nextDate(asxDate+1, mainCycle);
    }

    std::string ASX::nextCode(const Date& d,
                              bool mainCycle) {
        Date date = nextDate(d, mainCycle);
        return code(date);
    }

    std::string ASX::nextCode(const std::string& asxCode,
                              bool mainCycle,
                              const Date& referenceDate) {
        Date date = nextDate(asxCode, mainCycle, referenceDate);
        return code(date);
    }

}
