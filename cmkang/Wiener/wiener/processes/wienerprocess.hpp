#ifndef wiener_wiener_process_hpp
#define wiener_wiener_process_hpp

#include <wiener/utilities/null.hpp>
#include <wiener/processes/stochasticprocess.hpp>

namespace Wiener {
	namespace {	
		template <typename Real>
		__global__
		void transformKernel(Real* path, 
							Time* time, Size pathLength, 
							Real* normals, Size nSamples,
							Real x0, Real mu, Real sigma);
	}

	template <typename Real>
	class WienerProcess1D 
		: public StochasticProcess<Real, Real> {
	  public:
		virtual Real initialState () const { return x0_;}
		virtual Size dimension() const { return 1; }

		virtual std::shared_ptr<SamplePath<Real>> 
		generateSamplePath(std::shared_ptr<TimeGrid> timeGrid,
							Real* normalRandomNumbers, Size nSamples, 
							Size nBlks = Null<Size>(), Size nThds = Null<Size>());
	  private:
		Real x0_, mu_, sigma_;
	};
}

#endif