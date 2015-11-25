#ifndef wiener_errors_hpp
#define wiener_errors_hpp

#include <memory>
#include <exception>
#include <sstream>
#include <string>
#include <cuda.h>
#include <curand.h>
#include <wiener/defines.hpp>

namespace Wiener {

    //! Base error class
    class Error : public std::exception {
      public:
        /*! The explicit use of this constructor is not advised.
            Use the WI_FAIL macro instead.
        */
        Error(const std::string& file,
              long line,
              const std::string& message = "");
        /*! the automatically generated destructor would
            not have the throw specifier.
        */
        ~Error() throw() {}
        //! returns the error message.
        const char* what() const throw ();
      private:
        std::shared_ptr<std::string> message_;
    };

}

#define MULTILINE_MACRO_BEGIN do {
#define MULTILINE_MACRO_END } while(false)

#define WI_FAIL(message) \
MULTILINE_MACRO_BEGIN \
    std::ostringstream _WI_msg_stream; \
    _WI_msg_stream << message; \
    throw Wiener::Error(__FILE__,__LINE__, \
						_WI_msg_stream.str()); \
MULTILINE_MACRO_END


#define WI_REQUIRE(condition,message) \
if (!(condition)) { \
    std::ostringstream _WI_msg_stream; \
    _WI_msg_stream << message; \
    throw Wiener::Error(__FILE__,__LINE__, \
                          _WI_msg_stream.str()); \
 } else 

#define WI_CUDA_CALL(cudaResult) \
if ((cudaResult) != cudaSuccess) { \
	std::ostringstream _WI_msg_stream; \
	_WI_msg_stream << cudaGetErrorString((cudaResult)); \
	throw Wiener::Error(__FILE__,__LINE__, \
                          _WI_msg_stream.str()); \
 } else 

#define WI_CURAND_CALL(curandResult) \
if ((curandResult) != CURAND_STATUS_SUCCESS) { \
	std::ostringstream _WI_msg_stream; \
	switch ((curandResult)) \
	{ \
		case CURAND_STATUS_VERSION_MISMATCH: \
			_WI_msg_stream << "CURAND_STATUS_VERSION_MISMATCH"; \
			break; \
		case CURAND_STATUS_NOT_INITIALIZED: \
			_WI_msg_stream << "CURAND_STATUS_NOT_INITIALIZED"; \
			break; \
		case CURAND_STATUS_ALLOCATION_FAILED: \
			_WI_msg_stream << "CURAND_STATUS_ALLOCATION_FAILED"; \
			break; \
		case CURAND_STATUS_TYPE_ERROR: \
			_WI_msg_stream << "CURAND_STATUS_TYPE_ERROR"; \
			break; \
		case CURAND_STATUS_OUT_OF_RANGE: \
			_WI_msg_stream << "CURAND_STATUS_OUT_OF_RANGE"; \
			break; \
		case CURAND_STATUS_LENGTH_NOT_MULTIPLE: \
			_WI_msg_stream << "CURAND_STATUS_LENGTH_NOT_MULTIPLE"; \
			break; \
		case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED: \
			_WI_msg_stream << "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED"; \
			break; \
		case CURAND_STATUS_LAUNCH_FAILURE: \
			_WI_msg_stream << "CURAND_STATUS_LAUNCH_FAILURE"; \
			break; \
		case CURAND_STATUS_PREEXISTING_FAILURE: \
			_WI_msg_stream << "CURAND_STATUS_PREEXISTING_FAILURE"; \
			break; \
		case CURAND_STATUS_INITIALIZATION_FAILED: \
			_WI_msg_stream << "CURAND_STATUS_INITIALIZATION_FAILED"; \
			break; \
		case CURAND_STATUS_ARCH_MISMATCH: \
			_WI_msg_stream << "CURAND_STATUS_ARCH_MISMATCH"; \
			break; \
		case CURAND_STATUS_INTERNAL_ERROR: \
			_WI_msg_stream << "CURAND_STATUS_INTERNAL_ERROR"; \
			break; \
		default: \
			_WI_msg_stream << "CURAND_UNKNOWN_ERROR"; \
			break; \
	} \
	_WI_msg_stream << cudaGetErrorString((cudaResult)); \
	throw Wiener::Error(__FILE__,__LINE__, \
                          _WI_msg_stream.str()); \
 } else 

#endif

