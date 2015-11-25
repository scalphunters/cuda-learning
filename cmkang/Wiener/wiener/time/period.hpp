#ifndef wiener_period_hpp
#define wiener_period_hpp

#include <wiener/time/frequency.hpp>
#include <wiener/time/timeunit.hpp>

namespace Wiener {

    /*! This class provides a Period (length + TimeUnit) class
        and implements a limited algebra.

        \ingroup datetime

        \test self-consistency of algebra is checked.
    */
    class Period {
      public:
        Period()
        : length_(0), units_(Days) {}
        Period(Integer n, TimeUnit units)
        : length_(n), units_(units) {}
        explicit Period(Frequency f);
        Integer length() const { return length_; }
        TimeUnit units() const { return units_; }
        Frequency frequency() const;
        Period& operator+=(const Period&);
        Period& operator-=(const Period&);
        Period& operator/=(Integer);
        void normalize();
      private:
        Integer length_;
        TimeUnit units_;
    };

    /*! \relates Period */
	template <typename Real>
    Real years(const Period&);
    /*! \relates Period */
	template <typename Real>
    Real months(const Period&);
    /*! \relates Period */
	template <typename Real>
    Real weeks(const Period&);
    /*! \relates Period */
	template <typename Real>
    Real days(const Period&);

    /*! \relates Period */
    template <typename T> Period operator*(T n, TimeUnit units);
    /*! \relates Period */
    template <typename T> Period operator*(TimeUnit units, T n);

    /*! \relates Period */
    Period operator-(const Period&);

    /*! \relates Period */
    Period operator*(Integer n, const Period&);
    /*! \relates Period */
    Period operator*(const Period&, Integer n);

    /*! \relates Period */
    Period operator/(const Period&, Integer n);

    /*! \relates Period */
    Period operator+(const Period&, const Period&);
    /*! \relates Period */
    Period operator-(const Period&, const Period&);

    /*! \relates Period */
    bool operator<(const Period&, const Period&);
    /*! \relates Period */
    bool operator==(const Period&, const Period&);
    /*! \relates Period */
    bool operator!=(const Period&, const Period&);
    /*! \relates Period */
    bool operator>(const Period&, const Period&);
    /*! \relates Period */
    bool operator<=(const Period&, const Period&);
    /*! \relates Period */
    bool operator>=(const Period&, const Period&);

    /*! \relates Period */
    std::ostream& operator<<(std::ostream&, const Period&);

    namespace detail {

        struct long_period_holder {
            long_period_holder(const Period& p) : p(p) {}
            Period p;
        };
        std::ostream& operator<<(std::ostream&, const long_period_holder&);

        struct short_period_holder {
            short_period_holder(Period p) : p(p) {}
            Period p;
        };
        std::ostream& operator<<(std::ostream&, const short_period_holder&);

    }

    namespace io {

        //! output periods in long format (e.g. "2 weeks")
        /*! \ingroup manips */
        detail::long_period_holder long_period(const Period&);

        //! output periods in short format (e.g. "2w")
        /*! \ingroup manips */
        detail::short_period_holder short_period(const Period&);

    }

    // inline definitions

    template <typename T>
    inline Period operator*(T n, TimeUnit units) {
        return Period(Integer(n),units);
    }

    template <typename T>
    inline Period operator*(TimeUnit units, T n) {
        return Period(Integer(n),units);
    }

    inline Period operator-(const Period& p) {
        return Period(-p.length(),p.units());
    }

    inline Period operator*(Integer n, const Period& p) {
        return Period(n*p.length(),p.units());
    }

    inline Period operator*(const Period& p, Integer n) {
        return Period(n*p.length(),p.units());
    }

    inline bool operator==(const Period& p1, const Period& p2) {
        return !(p1 < p2 || p2 < p1);
    }

    inline bool operator!=(const Period& p1, const Period& p2) {
        return !(p1 == p2);
    }

    inline bool operator>(const Period& p1, const Period& p2) {
        return p2 < p1;
    }

    inline bool operator<=(const Period& p1, const Period& p2) {
        return !(p1 > p2);
    }

    inline bool operator>=(const Period& p1, const Period& p2) {
        return !(p1 < p2);
    }

}

#endif
