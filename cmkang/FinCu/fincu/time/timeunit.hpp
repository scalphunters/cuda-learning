#ifndef fincu_timeunit_hpp
#define fincu_timeunit_hpp

#include <fincu/defines.hpp>
#include <iosfwd>

namespace FinCu {

    //! Units used to describe time periods
    /*! \ingroup datetime */
    enum TimeUnit { Days,
                    Weeks,
                    Months,
                    Years,
                    Hours,
                    Minutes,
                    Seconds,
                    Milliseconds,
					Microseconds
    };

    /*! \relates TimeUnit */
    std::ostream& operator<<(std::ostream&,
                             const TimeUnit&);

}

#endif
