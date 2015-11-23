#include <algorithm>
#include <cctype>
#include <fincu/errors.hpp>
#include <fincu/time/imm.hpp>
#include <fincu/utilities/dataparsers.hpp>

using std::string;

namespace FinCu {

    bool IMM::isIMMdate(const Date& date, bool mainCycle) {
        if (date.weekday()!=Wednesday)
            return false;

        Day d = date.dayOfMonth();
        if (d<15 || d>21)
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

    bool IMM::isIMMcode(const std::string& in, bool mainCycle) {
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

    std::string IMM::code(const Date& date) {
        FC_REQUIRE(isIMMdate(date, false),
                   date << " is not an IMM date");

        std::ostringstream IMMcode;
        unsigned int y = date.year() % 10;
        switch(date.month()) {
          case January:
            IMMcode << 'F' << y;
            break;
          case February:
            IMMcode << 'G' << y;
            break;
          case March:
            IMMcode << 'H' << y;
            break;
          case April:
            IMMcode << 'J' << y;
            break;
          case May:
            IMMcode << 'K' << y;
            break;
          case June:
            IMMcode << 'M' << y;
            break;
          case July:
            IMMcode << 'N' << y;
            break;
          case August:
            IMMcode << 'Q' << y;
            break;
          case September:
            IMMcode << 'U' << y;
            break;
          case October:
            IMMcode << 'V' << y;
            break;
          case November:
            IMMcode << 'X' << y;
            break;
          case December:
            IMMcode << 'Z' << y;
            break;
          default:
            FC_FAIL("not an IMM month (and it should have been)");
        }

        return IMMcode.str();
    }

    Date IMM::date(const std::string& immCode,
                   const Date& refDate) {
        FC_REQUIRE(isIMMcode(immCode, false),
                   immCode << " is not a valid IMM code");
		FC_REQUIRE(refDate != Date(), "null date given");

        Date referenceDate = refDate;

		std::string code = immCode;
		std::transform(code.begin(), code.end(), code.begin(), std::toupper);

        std::string ms = code.substr(0,1);
        FinCu::Month m;
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
        else FC_FAIL("invalid IMM month letter");

//        Year y = boost::lexical_cast<Year>(); // lexical_cast causes compilation errors with x64

        Year y= io::to_integer(code.substr(1,1));
        /* year<1900 are not valid QuantLib years: to avoid a run-time
           exception few lines below we need to add 10 years right away */
        if (y==0 && referenceDate.year()<=1909) y+=10;
        Year referenceYear = (referenceDate.year() % 10);
        y += referenceDate.year() - referenceYear;
        Date result = IMM::nextDate(Date(1, m, y), false);
        if (result<referenceDate)
            return IMM::nextDate(Date(1, m, y+10), false);

        return result;
    }

    Date IMM::nextDate(const Date& date, bool mainCycle) {
		FC_REQUIRE(date != Date(), "null date given");
        Date refDate = date;
        Year y = refDate.year();
        FinCu::Month m = refDate.month();

        Size offset = mainCycle ? 3 : 1;
        Size skipMonths = offset-(m%offset);
        if (skipMonths != offset || refDate.dayOfMonth() > 21) {
            skipMonths += Size(m);
            if (skipMonths<=12) {
                m = FinCu::Month(skipMonths);
            } else {
                m = FinCu::Month(skipMonths-12);
                y += 1;
            }
        }

        Date result = Date::nthWeekday(3, Wednesday, m, y);
        if (result<=refDate)
            result = nextDate(Date(22, m, y), mainCycle);
        return result;
    }

    Date IMM::nextDate(const std::string& IMMcode,
                       bool mainCycle,
                       const Date& referenceDate)  {
        Date immDate = date(IMMcode, referenceDate);
        return nextDate(immDate+1, mainCycle);
    }

    std::string IMM::nextCode(const Date& d,
                              bool mainCycle) {
        Date date = nextDate(d, mainCycle);
        return code(date);
    }

    std::string IMM::nextCode(const std::string& immCode,
                              bool mainCycle,
                              const Date& referenceDate) {
        Date date = nextDate(immCode, mainCycle, referenceDate);
        return code(date);
    }

}
