#include <iostream>
#include <fincu/time/date.hpp>
#include <fincu/time/calendars/southkorea.hpp>
#include <fincu/time/calendars/unitedstates.hpp>
#include <fincu/utilities/dataparsers.hpp>
#include <fincu/time/daycounters/actual365fixed.hpp>
#include <fincu/time/daycounters/thirty360.hpp>
#include <fincu/time/daycounters/actualactual.hpp>
#include <fincu/time/schedule.hpp>

using namespace FinCu;

int main(void) {

	Date date(23, Nov, 2015);
	Date date2 = DateParser::parseISO("2015-11-23");

	if (date == date2)
		std::cout << date << " and " << date2 << "are same dates\n";

	std::cout << date2 + Period(2, Months) << ": after 2 months from " << date2 << std::endl;

	Calendar calendar = SouthKorea();

	std::cout << date << " is a holiday (" << calendar.isHoliday(date) << ") " << std::endl;
	std::cout << calendar.advance(date, Period(5, Days)) << ": after 5 business day after" << date <<std::endl;

	date2 += Period(6, Months);

	DayCounter actual365Fixed = Actual365Fixed();
	DayCounter european = Thirty360(Thirty360::European);
	DayCounter isda = ActualActual(ActualActual::ISDA);

	std::cout << "Date - Time Conversion: d1 = " << date << " and d2 = " << date2 << std::endl;
	std::cout << actual365Fixed.name() << " : " << actual365Fixed.yearFraction(date, date2) << std::endl;
	std::cout << european.name() << " : " << european.yearFraction(date, date2) << std::endl;
	std::cout << isda.name() << " : " << isda.yearFraction(date, date2) << std::endl;

	std::cout << "Schedule Generation" << std::endl;
	
	Schedule schedule = MakeSchedule()
						.from(Date(31, Jan, 2015))
						.to(Date(31, Dec, 2030))
						.withTenor(Period(3, Months))
						.withCalendar(UnitedStates())
						.withConvention(ModifiedFollowing)
						.withTerminationDateConvention(Preceding)
						.withRule(DateGeneration::Forward)
						.endOfMonth(true);

	std::vector<Date> dates = schedule.dates();

	for(std::vector<Date>::iterator it = dates.begin();
									it != dates.end(); ++it) {
		std::cout << *it << std::endl;
	}

	std::cin.get();
	return 0;
}