#include <iostream>
#include <wiener/time/date.hpp>
#include <wiener/time/calendars/southkorea.hpp>
#include <wiener/time/calendars/unitedstates.hpp>
#include <wiener/utilities/dataparsers.hpp>
#include <wiener/time/daycounters/actual365fixed.hpp>
#include <wiener/time/daycounters/thirty360.hpp>
#include <wiener/time/daycounters/actualactual.hpp>
#include <wiener/time/schedule.hpp>
#include <wiener/methods/montecarlo/timegrid.hpp>
#include <wiener/methods/montecarlo/samplepath.hpp>
#include <curand.h>

using namespace Wiener;

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

	std::vector<Time> mandatory;
	mandatory.push_back(0.2);
	mandatory.push_back(0.5);
	mandatory.push_back(0.1);

	std::shared_ptr<TimeGrid> timeGrid(new TimeGrid(
								mandatory.begin(), mandatory.end(), 100));

	SamplePath<float> samplePath(timeGrid);
	

	std::cin.get();
	return 0;
}