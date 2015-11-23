#include <fincu/time/businessdayconvention.hpp>
#include <fincu/errors.hpp>

namespace FinCu {

    std::ostream& operator<<(std::ostream& out,
                             BusinessDayConvention b) {
        switch (b) {
          case Following:
            return out << "Following";
          case ModifiedFollowing:
            return out << "Modified Following";
          case HalfMonthModifiedFollowing:
            return out << "Half-Month Modified Following";
          case Preceding:
            return out << "Preceding";
          case ModifiedPreceding:
            return out << "Modified Preceding";
          case Unadjusted:
            return out << "Unadjusted";
          case Nearest:
            return out << "Nearest";
          default:
            FC_FAIL("unknown BusinessDayConvention (" << Integer(b) << ")");
        }
    }

}
