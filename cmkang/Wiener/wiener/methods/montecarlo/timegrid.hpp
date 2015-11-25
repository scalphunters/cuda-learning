#ifndef wiener_time_grid_hpp
#define wiener_time_grid_hpp

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/adjacent_difference.h>

#include <wiener/errors.hpp>
#include <wiener/math/comparison.hpp>

namespace Wiener {
	
    class TimeGrid {
      public:
		typedef typename thrust::host_vector<Time>::iterator iterator;
        typedef typename thrust::host_vector<Time>::const_iterator const_iterator;
        typedef typename thrust::host_vector<Time>::const_reverse_iterator
                                          const_reverse_iterator;

        TimeGrid() {}

        TimeGrid(Time end, Size steps);
		~TimeGrid();

		template <class Iterator>
        TimeGrid(Iterator begin, Iterator end)
        : mandatoryTimes_(begin, end) {
            thrust::sort(mandatoryTimes_.begin(),mandatoryTimes_.end());
            WI_REQUIRE(mandatoryTimes_.front() >= 0.0,
                       "negative times not allowed");

			iterator e = thrust::unique(
								mandatoryTimes_.begin(), mandatoryTimes_.end(),
								CloseEnough<Time>());
            mandatoryTimes_.resize(e - mandatoryTimes_.begin());

            if (mandatoryTimes_[0] > 0.0)
                times_.push_back(0.0);

            times_.insert(times_.end(),
                          mandatoryTimes_.begin(), mandatoryTimes_.end());

			dt_.resize(times_.size() - 1);

            thrust::adjacent_difference(times_.begin()+1,times_.end(),
										dt_.begin());
			initializeDeviceTimes();
        }

		template <class Iterator>
        TimeGrid(Iterator begin, Iterator end, Size steps)
        : mandatoryTimes_(begin, end) {

            thrust::sort(mandatoryTimes_.begin(),mandatoryTimes_.end());
            WI_REQUIRE(mandatoryTimes_.front() >= 0.0,
                       "negative times not allowed");

            iterator e = thrust::unique(
							mandatoryTimes_.begin(),mandatoryTimes_.end(),
                            CloseEnough<Time>());
            mandatoryTimes_.resize(e - mandatoryTimes_.begin());

            Time last = mandatoryTimes_.back();
            Time dtMax;
            // The resulting timegrid have points at times listed in the input
            // list. Between these points, there are inner-points which are
            // regularly spaced.
            if (steps == 0) {
                thrust::host_vector<Time> diff(mandatoryTimes_.size());
                thrust::adjacent_difference(mandatoryTimes_.begin(),
                                         mandatoryTimes_.end(),
                                         diff.begin());
                if (diff.front()==0.0)
                    diff.erase(diff.begin());
                dtMax = *(thrust::min_element(diff.begin(), diff.end()));
            } else {
                dtMax = last/steps;
            }

            Time periodBegin = 0.0;
            times_.push_back(periodBegin);
			
            for (const_iterator t = mandatoryTimes_.begin(); 
								t < mandatoryTimes_.end(); t++) {
                Time periodEnd = *t;
                if (periodEnd != 0.0) {
                    // the nearest integer
                    Size nSteps = Size((periodEnd - periodBegin)/dtMax+0.5);
                    // at least one time step!
                    nSteps = (nSteps!=0 ? nSteps : 1);
                    Time dt = (periodEnd - periodBegin)/nSteps;
                    times_.reserve(nSteps);
                    for (Size n=1; n<=nSteps; ++n)
                        times_.push_back(periodBegin + n*dt);
                }
                periodBegin = periodEnd;
            }

			dt_.resize(times_.size() - 1);
            thrust::adjacent_difference(times_.begin()+1,times_.end(),
                                     dt_.begin());
			initializeDeviceTimes();
        }

		Size index(Time t) const;

        Size closestIndex(Time t) const;

        Time closestTime(Time t) const {
            return times_[closestIndex(t)];
        }

        const thrust::host_vector<Time>& mandatoryTimes() const {
            return mandatoryTimes_;
        }

        const thrust::host_vector<Time>& times() const {
            return times_;
        }

        const thrust::host_vector<Time>& dt() const {
            return dt_;
        }

        Time dt(Size i) const { return dt_[i]; }

        Time operator[](Size i) const { return times_[i]; }
        Time at(Size i) const { return times_[i]; }
        Size size() const { return times_.size(); }
        bool empty() const { return times_.empty(); }
        const_iterator begin() const { return times_.begin(); }
        const_iterator end() const { return times_.end(); }
        const_reverse_iterator rbegin() const { return times_.rbegin(); }
        const_reverse_iterator rend() const { return times_.rend(); }
        Time front() const { return times_.front(); }
        Time back() const { return times_.back(); }

		Time* data() const { return deviceTimes_; }

      private:
		void initializeDeviceTimes();

        thrust::host_vector<Time> times_;
        thrust::host_vector<Time> dt_;
        thrust::host_vector<Time> mandatoryTimes_;

		Time* deviceTimes_;
    };

}


#endif
