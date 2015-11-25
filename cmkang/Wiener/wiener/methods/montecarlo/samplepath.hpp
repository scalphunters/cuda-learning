#ifndef wiener_sample_path_hpp
#define wiener_sample_path_hpp

#include <memory>
#include <wiener/methods/montecarlo/timegrid.hpp>

namespace Wiener {
	template <typename Real>
	class SamplePath {
	  public:
		SamplePath(
			const std::shared_ptr<TimeGrid> & timeGrid,
			const Size size = 1, const Size dimension = 1)
		: timeGrid_(timeGrid), size_(size), dimension_(dimension) {
			Size pathLength = timeGrid_->size();
			Size n = size_*dimension_*pathLength;
			WI_CUDA_CALL(cudaMalloc((Real**)&path_, n*sizeof(Real)));
		}

		~SamplePath() {
			WI_CUDA_CALL(cudaFree(path_));
		}

		Real* data() { return path_; }

	  private:
		std::shared_ptr<TimeGrid> timeGrid_;
		// size_: # of sample paths
		// dimension_: dimension of process
		Size size_, dimension_;

		Real* path_; //device pointer to sample paths
	};
}

#endif