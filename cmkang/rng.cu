#include <cmath>
#include <string>

#define __CUDA_INTERNAL_COMPILATION__
#include <math_functions.hpp>
#undef __CUDA_INTERNAL_COMPILATION__

#include "rng.hpp"

namespace FinCu {
	
	namespace {
		template <typename Real>
		class NormalTransform 
			: public std::unary_function<Real, Real> 
		{
		  public:
			NormalTransform(Real mu = 0.0, Real sigma = 1.0) 
			: mu_(mu), sigma_(sigma) {}

			Real operator () (Real p) {
				return mu_ + sigma_*normcdfinv(p);
			}

		  private:
			Real mu_, sigma_;
		};

		template <typename Real>
		class LogNormalTransform
			: public std::unary_function<Real, Real>
		{
		  public:
			LogNormalTransform(Real mu = 0.0, Real sigma = 1.0)
			: mu_(mu), sigma_(sigma) {}

			Real operator () (Real p) {
				return exp(mu_ + sigma_*normcdfinv(p));
			}

		  private:
			Real mu_, sigma_;
		};
	}

	template <typename Real>
	RandomNumberGenerator<Real>::RandomNumberGenerator(
		curandRngType_t generatorType,
		unsigned long long seed,
		unsigned int bufferSize)
	: bufferSize_(bufferSize), hostBuffer_(bufferSize), 
	  iter_(hostBuffer_.begin())
	{
		curandStatus_t curandResult;
		curandResult = curandCreateGenerator(&gen_, generatorType);
		if (curandResult != CURAND_STATUS_SUCCESS) {
			std::string msg("Could not create cuda random number generator: ");
			msg += curandResult;
			throw std::runtime_error(msg);
		}

		curandResult = curandSetPseudoRandomGeneratorSeed(gen_, seed);
		if (curandResult != CURAND_STATUS_SUCCESS) {
			std::string msg("Could not initialize cuda rng seed: ");
			msg += curandResult;
			throw std::runtime_error(msg);
		}

		updateBuffer();
	}

	template <typename Real>
	RandomNumberGenerator<Real>::~RandomNumberGenerator()
	{
		curandStatus_t curandResult;
		curandResult = curandDestroyGenerator(gen_);
		if (curandResult != CURAND_STATUS_SUCCESS) {
			std::string msg("Could not destroy cuda random number generator: ");
			msg += curandResult;
			throw std::runtime_error(msg);
		}
	}

	template <typename Real>
	void RandomNumberGenerator<Real>::updateBuffer()
	{
		Real* devicePtr;
		cudaError_t cudaResult;

		cudaResult = cudaMalloc((Real**) &devicePtr, 
								bufferSize_*sizeof(Real));
		if (cudaResult != cudaSuccess) {
			std::string msg("Could not allocate memory on device for random numbers: ");
			msg += cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		curandStatus_t curandResult;
		if (typeid(Real) == typeid(double)) {
			curandResult = curandGenerateUniformDouble(
								gen_, (double*) devicePtr, bufferSize_);		
		} else if (typeid(Real) == typeid(float)) {
			curandResult = curandGenerateUniform(
								gen_, (float*) devicePtr, bufferSize_);		
		}

		if (curandResult != CURAND_STATUS_SUCCESS) {
			std::string msg("Could not generate pseudo-random numbers: ");
			msg += curandResult;
			throw std::runtime_error(msg);
		}

		cudaResult = cudaMemcpy(hostBuffer_.data(), devicePtr, 
							bufferSize_*sizeof(Real),
							cudaMemcpyDeviceToHost);
		if (cudaResult != cudaSuccess) {
			std::string msg("Could not copy partial results to host: ");
			msg += cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}


		cudaResult = cudaFree(devicePtr);
		if (cudaResult != cudaSuccess) {
			std::string msg("Could not free device memory: ");
			msg += cudaGetErrorString(cudaResult);
			throw std::runtime_error(msg);
		}

		iter_ = hostBuffer_.begin();
	}

	template <typename Real>
	Real RandomNumberGenerator<Real>::operator() (
		Distribution dist, Real mu, Real sigma)
	{
		Real sample;

		switch (dist)
		{
		case RandomNumberGenerator::Uniform:
			sample = *iter_++;
			break;
		case RandomNumberGenerator::Normal:
			sample = mu + sigma*normcdfinv(*iter_++);
			break;
		case RandomNumberGenerator::LogNormal:
			sample = exp(mu + sigma*normcdfinv(*iter_++));
			break;
		default:
			std::string msg("RandomNumberGenerator: illegal distribution given");
			throw std::runtime_error(msg);
			break;
		}

		if (iter_ == hostBuffer_.end())
			updateBuffer();

		return sample;
	}
	
	template <typename Real>
	thrust::host_vector<Real> 
	RandomNumberGenerator<Real>::nextRandomSequence(
		unsigned int n, Distribution dist,
		Real mu, Real sigma) 
	{
		unsigned int m, k;
		thrust::host_vector<Real> randomSequence(n);

		k = 0;
		while (k < n) {
			m = thrust::distance(iter_, hostBuffer_.end());
			m = (m < n - k) ? m : n - k;

			switch (dist)
			{
			case RandomNumberGenerator::Uniform:
				thrust::copy(iter_, iter_ + m, randomSequence.begin() + k);
				break;
			case RandomNumberGenerator::Normal:
				thrust::transform(iter_, iter_ + m, 
					randomSequence.begin() + k, NormalTransform<Real>(mu, sigma));
				break;
			case RandomNumberGenerator::LogNormal:
				thrust::transform(iter_, iter_ + m, 
					randomSequence.begin() + k, LogNormalTransform<Real>(mu, sigma));
				break;
			default:
				std::string msg("RandomNumberGenerator: illegal distribution given");
				throw std::runtime_error(msg);
				break;
			}
			
			k += m;
			iter_ += m;
			if (iter_ == hostBuffer_.end())
				updateBuffer();
		}

		return randomSequence;
	}

	template class RandomNumberGenerator<float>;
	template class RandomNumberGenerator<double>;
}