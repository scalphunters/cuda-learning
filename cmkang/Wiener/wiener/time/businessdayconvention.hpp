#ifndef wiener_business_day_convention_hpp
#define wiener_business_day_convention_hpp

#include <wiener/defines.hpp>
#include <iosfwd>

namespace Wiener {

    //! Business Day conventions
    /*! These conventions specify the algorithm used to adjust a date in case
        it is not a valid business day.

        \ingroup datetime
    */
    enum BusinessDayConvention {
        // ISDA
        Following,                   /*!< Choose the first business day after
                                          the given holiday. */
        ModifiedFollowing,           /*!< Choose the first business day after
                                          the given holiday unless it belongs
                                          to a different month, in which case
                                          choose the first business day before
                                          the holiday. */
        Preceding,                   /*!< Choose the first business
                                          day before the given holiday. */
        // NON ISDA
        ModifiedPreceding,           /*!< Choose the first business day before
                                          the given holiday unless it belongs
                                          to a different month, in which case
                                          choose the first business day after
                                          the holiday. */
        Unadjusted,                  /*!< Do not adjust. */
        HalfMonthModifiedFollowing,  /*!< Choose the first business day after
                                          the given holiday unless that day
                                          crosses the mid-month (15th) or the
                                          end of month, in which case choose
                                          the first business day before the
                                          holiday. */
        Nearest                      /*!< Choose the nearest business day 
                                          to the given holiday. If both the
                                          preceding and following business
                                          days are equally far away, default
                                          to following business day. */
    };

    /*! \relates BusinessDayConvention */
    std::ostream& operator<<(std::ostream&,
                             BusinessDayConvention);

}

#endif
