#include <fincu/time/dategenerationrule.hpp>
#include <fincu/errors.hpp>

namespace FinCu {

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
            FC_FAIL("unknown DateGeneration::Rule (" << Integer(r) << ")");
        }
    }

}
