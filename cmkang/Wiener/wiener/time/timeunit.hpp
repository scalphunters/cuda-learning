#ifndef wiener_timeunit_hpp
#define wiener_timeunit_hpp

#include <wiener/defines.hpp>
#include <iosfwd>

namespace Wiener {

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
