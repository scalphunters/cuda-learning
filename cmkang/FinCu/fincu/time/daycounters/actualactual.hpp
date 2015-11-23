#ifndef fincu_actualactual_day_counter_h
#define fincu_actualactual_day_counter_h

#include <fincu/time/daycounter.hpp>

namespace FinCu {

    //! Actual/Actual day count
    /*! The day count can be calculated according to:

        - the ISDA convention, also known as "Actual/Actual (Historical)",
          "Actual/Actual", "Act/Act", and according to ISDA also "Actual/365",
          "Act/365", and "A/365";
        - the ISMA and US Treasury convention, also known as
          "Actual/Actual (Bond)";
        - the AFB convention, also known as "Actual/Actual (Euro)".

        For more details, refer to
        http://www.isda.org/publications/pdf/Day-Count-Fracation1999.pdf

        \ingroup daycounters

        \test the correctness of the results is checked against known
              good values.
    */
    class ActualActual : public DayCounter {
      public:
        enum Convention { ISMA, Bond,
                          ISDA, Historical, Actual365,
                          AFB, Euro };
      private:
        class ISMA_Impl : public DayCounter::Impl {
          public:
            std::string name() const {
                return std::string("Actual/Actual (ISMA)");
            }
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date& refPeriodStart,
                              const Date& refPeriodEnd) const;
        };
        class ISDA_Impl : public DayCounter::Impl {
          public:
            std::string name() const {
                return std::string("Actual/Actual (ISDA)");
            }
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date&,
                              const Date&) const;
        };
        class AFB_Impl : public DayCounter::Impl {
          public:
            std::string name() const {
                return std::string("Actual/Actual (AFB)");
            }
            Time yearFraction(const Date& d1,
                              const Date& d2,
                              const Date&,
                              const Date&) const;
        };
        static std::shared_ptr<DayCounter::Impl> implementation(
                                                               Convention c);
      public:
        ActualActual(Convention c = ActualActual::ISDA)
        : DayCounter(implementation(c)) {}
    };

}

#endif
