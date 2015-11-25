#include <iomanip>
#include <thrust/binary_search.h>

#include <wiener/methods/montecarlo/timegrid.hpp>

namespace Wiener {

	TimeGrid::TimeGrid(Time end, Size steps) {
        // We seem to assume that the grid begins at 0.
        // Let's enforce the assumption for the time being
        // (even though I'm not sure that I agree.)
        WI_REQUIRE(end > 0.0,
                   "negative times not allowed");
        Time dt = end/steps;
        times_.reserve(steps+1);
        for (Size i=0; i<=steps; i++)
            times_.push_back(dt*i);

        mandatoryTimes_ = thrust::host_vector<Time>(1);
        mandatoryTimes_[0] = end;

        dt_ = thrust::host_vector<Time>(steps,dt);
		initializeDeviceTimes();
    }

    Size TimeGrid::index(Time t) const {
        Size i = closestIndex(t);
        if (close_enough(t,Time(times_[i]))) {
            return i;
        } else {
            if (t < times_.front()) {
                WI_FAIL("using inadequate time grid: all nodes "
                        "are later than the required time t = "
                        << std::setprecision(12) << t
                        << " (earliest node is t1 = "
                        << std::setprecision(12) << times_.front() << ")");
            } else if (t > times_.back()) {
                WI_FAIL("using inadequate time grid: all nodes "
                        "are earlier than the required time t = "
                        << std::setprecision(12) << t
                        << " (latest node is t1 = "
                        << std::setprecision(12) << times_.back() << ")");
            } else {
                Size j, k;
                if (t > times_[i]) {
                    j = i;
                    k = i+1;
                } else {
                    j = i-1;
                    k = i;
                }
                WI_FAIL("using inadequate time grid: the nodes closest "
                        "to the required time t = "
                        << std::setprecision(12) << t
                        << " are t1 = "
                        << std::setprecision(12) << times_[j]
                        << " and t2 = "
                        << std::setprecision(12) << times_[k]);
            }
        }
    }

    Size TimeGrid::closestIndex(Time t) const {
        const_iterator begin = times_.begin(), end = times_.end();
        const_iterator result = thrust::lower_bound(begin, end, t);
        if (result == begin) {
            return 0;
        } else if (result == end) {
            return size()-1;
        } else {
            Time dt1 = *result - t;
            Time dt2 = t - *(result-1);
            if (dt1 < dt2)
                return result-begin;
            else
                return (result-begin)-1;
        }
    }

	void TimeGrid::initializeDeviceTimes() {
		WI_CUDA_CALL(cudaMalloc((Time**)&deviceTimes_, 
								times_.size()*sizeof(Time)));
		WI_CUDA_CALL(cudaMemcpy(deviceTimes_, times_.data(),
								times_.size()*sizeof(Time),
								cudaMemcpyHostToDevice));
	}

	TimeGrid::~TimeGrid() {
		WI_CUDA_CALL(cudaFree(deviceTimes_));
	}
}