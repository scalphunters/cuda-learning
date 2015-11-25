#include <wiener/time/dategenerationrule.hpp>
#include <wiener/errors.hpp>

namespace Wiener {

    std::ostream& operator<<(std::ostream& out, DateGeneration::Rule r) {
        switch (r) {
          case DateGeneration::Backward:
            return out << "Backward";
          case DateGeneration::Forward:
            return out << "Forward";
          case DateGeneration::Zero:
            return out << "Zero";
          case DateGeneration::ThirdWednesday:
            return out << "ThirdWednesday";
          case DateGeneration::Twentieth:
            return out << "Twentieth";
          case DateGeneration::TwentiethIMM:
            return out << "TwentiethIMM";
          case DateGeneration::OldCDS:
            return out << "OldCDS";
          case DateGeneration::CDS:
            return out << "CDS";
          default:
            WI_FAIL("unknown DateGeneration::Rule (" << Integer(r) << ")");
        }
    }

}
