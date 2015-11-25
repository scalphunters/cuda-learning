#include <wiener/processes/wienerprocess.hpp>

namespace Wiener {
	namespace {
		template <typename Real>
		__global__
		void transformKernel(Real* path, 
							Time* time, Size pathLength, 
							Real* normals, Size nSamples,
							Real x0, Real mu, Real sigma) 
		{
			int idx = blockIdx.x*blockDim.x + threadIdx.x;
			int offset = idx*pathLength;
			int normalOffet = idx*(pathLength - 1);

			if (idx < nSamples) {
				path[offset] = x0;
				for (int i = 1; i < pathLength; ++i) {
					Time dt = time[i] - time[i-1];
					path[offset + i] = path[offset + i-1] 
							+ mu*dt + sigma*sqrt(dt)*normals[normalOffet + i-1];
				}
			}
		}	
	}

	template <typename Real>
	std::shared_ptr<SamplePath<Real>> 
	WienerProcess1D<Real>::generateSamplePath(
		std::shared_ptr<TimeGrid> timeGrid,
		Real* normalRandomNumbers, Size nSamples,
		Size nBlks, Size nThds) 
	{
		std::shared_ptr<SamplePath<Real>> samplePath(
				new SamplePath<Real>(timeGrid, nSamples));

		Size nBlocks = (nBlks == Null<Size>()) ? 512 : nBlks;
		Size nThreads = (nThds == Null<Size>()) 
						? (nSamples/nBlocks + 1) : nThds;

		dim3 dimGrid(nBlocks);
		dim3 dimBlock(nThreads);

		transformKernel<<<dimGrid, dimBlock>>>(samplePath->data(), 
											timeGrid->data(), timeGrid->size(),
											normalRandomNumbers, nSamples, 
											x0_, mu_, sigma_);
		return samplePath;
	}

	template class WienerProcess1D<float>;
	template class WienerProcess1D<double>;
}