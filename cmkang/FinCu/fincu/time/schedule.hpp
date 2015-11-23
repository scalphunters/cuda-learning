#ifndef fincu_schedule_hpp
#define fincu_schedule_hpp

#include <vector>
#include <fincu/time/calendars/nullcalendar.hpp>
#include <fincu/utilities/null.hpp>
#include <fincu/time/period.hpp>
#include <fincu/time/dategenerationrule.hpp>
#include <fincu/errors.hpp>

namespace FinCu {

    //! Payment schedule
    /*! \ingroup datetime */
    class Schedule {
      public:
        /*! constructor that takes any list of dates, and optionally
            meta information that can be used by client classes. Note
            that neither the list of dates nor the meta information is
            checked for plausibility in any sense. */
        Schedule(const std::vector<Date>&,
                 const Calendar& calendar = NullCalendar(),
                 const BusinessDayConvention
                                    convention = Unadjusted,
                 const std::vector<bool>& isRegular = std::vector<bool>(0));
        /*! rule based constructor */
        Schedule(Date effectiveDate,
                 const Date& terminationDate,
                 const Period& tenor,
                 const Calendar& calendar,
                 BusinessDayConvention convention,
                 BusinessDayConvention terminationDateConvention,
                 DateGeneration::Rule rule,
                 bool endOfMonth,
                 const Date& firstDate = Date(),
                 const Date& nextToLastDate = Date());
        Schedule() {}
        //! \name Date access
        //@{
        Size size() const { return dates_.size(); }
        const Date& operator[](Size i) const;
        const Date& at(Size i) const;
        const Date& date(Size i) const;
        Date previousDate(const Date& refDate) const;
        Date nextDate(const Date& refDate) const;
        const std::vector<Date>& dates() const { return dates_; }
        bool isRegular(Size i) const;
        const std::vector<bool>& isRegular() const;
        //@}
        //! \name Other inspectors
        //@{
        bool empty() const { return dates_.empty(); }
        const Calendar& calendar() const;
        const Date& startDate() const;
        const Date& endDate() const;
        const Period& tenor() const;
        BusinessDayConvention businessDayConvention() const;
        BusinessDayConvention terminationDateBusinessDayConvention() const;
        DateGeneration::Rule rule() const;
        bool endOfMonth() const;
        //@}
        //! \name Iterators
        //@{
        typedef std::vector<Date>::const_iterator const_iterator;
        const_iterator begin() const { return dates_.begin(); }
        const_iterator end() const { return dates_.end(); }
        const_iterator lower_bound(const Date& d = Date()) const;
        //@}
        //! \name Utilities
        //@{
        //! truncated schedule
        Schedule until(const Date& truncationDate) const;
        //@}
      private:
        Period tenor_;
        Calendar calendar_;
        BusinessDayConvention convention_;
        BusinessDayConvention terminationDateConvention_;
        DateGeneration::Rule rule_;
        bool endOfMonth_;
        Date firstDate_, nextToLastDate_;
        std::vector<Date> dates_;
        std::vector<bool> isRegular_;
    };


    //! helper class
    /*! This class provides a more comfortable interface to the
        argument list of Schedule's constructor.
    */
    class MakeSchedule {
      public:
        MakeSchedule();
        MakeSchedule& from(const Date& effectiveDate);
        MakeSchedule& to(const Date& terminationDate);
        MakeSchedule& withTenor(const Period&);
        MakeSchedule& withFrequency(Frequency);
        MakeSchedule& withCalendar(const Calendar&);
        MakeSchedule& withConvention(BusinessDayConvention);
        MakeSchedule& withTerminationDateConvention(BusinessDayConvention);
        MakeSchedule& withRule(DateGeneration::Rule);
        MakeSchedule& forwards();
        MakeSchedule& backwards();
        MakeSchedule& endOfMonth(bool flag=true);
        MakeSchedule& withFirstDate(const Date& d);
        MakeSchedule& withNextToLastDate(const Date& d);
        operator Schedule() const;
      private:
        Calendar calendar_;
        Date effectiveDate_, terminationDate_;
        Period tenor_;
        BusinessDayConvention convention_;
        BusinessDayConvention terminationDateConvention_;
        DateGeneration::Rule rule_;
        bool endOfMonth_;
        Date firstDate_, nextToLastDate_;
    };



    // inline definitions

    inline const Date& Schedule::date(Size i) const {
        return dates_.at(i);
    }

    inline const Date& Schedule::operator[](Size i) const {
        #if defined(QL_EXTRA_SAFETY_CHECKS)
        return dates_.at(i);
        #else
        return dates_[i];
        #endif
    }

    inline const Date& Schedule::at(Size i) const {
        return dates_.at(i);
    }

    inline const Calendar& Schedule::calendar() const {
        return calendar_;
    }

    inline const Date& Schedule::startDate() const {
        return dates_.front();
    }

    inline const Date &Schedule::endDate() const { return dates_.back(); }

    inline const Period& Schedule::tenor() const {
        return tenor_;
    }

    inline BusinessDayConvention Schedule::businessDayConvention() const {
        return convention_;
    }

    inline BusinessDayConvention
    Schedule::terminationDateBusinessDayConvention() const {
        return terminationDateConvention_;
    }

    inline DateGeneration::Rule Schedule::rule() const {
        return rule_;
    }

    inline bool Schedule::endOfMonth() const {
        return endOfMonth_;
    }

}

#endif
