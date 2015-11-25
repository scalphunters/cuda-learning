#include <wiener/time/frequency.hpp>
#include <wiener/errors.hpp>

namespace Wiener {

    std::ostream& operator<<(std::ostream& out, Frequency f) {
        switch (f) {
          case NoFrequency:
            return out << "No-Frequency";
          case Once:
            return out << "Once";
          case Annual:
            return out << "Annual";
          case Semiannual:
            return out << "Semiannual";
          case EveryFourthMonth:
            return out << "Every-Fourth-Month";
          case Quarterly:
            return out << "Quarterly";
          case Bimonthly:
            return out << "Bimonthly";
          case Monthly:
            return out << "Monthly";
          case EveryFourthWeek:
            return out << "Every-fourth-week";
          case Biweekly:
            return out << "Biweekly";
          case Weekly:
            return out << "Weekly";
          case Daily:
            return out << "Daily";
          case OtherFrequency:
            return out << "Unknown frequency";
          default:
            WI_FAIL("unknown frequency (" << Integer(f) << ")");
        }
    }

}
