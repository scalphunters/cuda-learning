#ifndef wiener_stochastic_process_hpp
#define wiener_stochastic_process_hpp

#include <wiener/defines.hpp>
#include <wiener/methods/montecarlo/samplepath.hpp>

namespace Wiener {
	template <typename StateType, typename Real>
	class StochasticProcess {
	  public:
		virtual StateType initialState() const = 0;
		virtual Size dimension() const = 0;
	  
		virtual std::shared_ptr<SamplePath<Real>> 
		generateSamplePath(const std::shared_ptr<TimeGrid>& timeGrid,
							Real* normalRandomNumbers, Size nSamples = 1) = 0;
	};
}

#endif