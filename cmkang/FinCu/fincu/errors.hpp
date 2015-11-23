#ifndef fincu_errors_hpp
#define fincu_errors_hpp

#include <memory>
#include <exception>
#include <sstream>
#include <string>

#include <fincu/defines.hpp>

namespace FinCu {

    //! Base error class
    class Error : public std::exception {
      public:
        /*! The explicit use of this constructor is not advised.
            Use the FC_FAIL macro instead.
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

#define FC_FAIL(message) \
MULTILINE_MACRO_BEGIN \
    std::ostringstream _fc_msg_stream; \
    _fc_msg_stream << message; \
    throw FinCu::Error(__FILE__,__LINE__, \
						_fc_msg_stream.str()); \
MULTILINE_MACRO_END


#define FC_REQUIRE(condition,message) \
if (!(condition)) { \
    std::ostringstream _fc_msg_stream; \
    _fc_msg_stream << message; \
    throw FinCu::Error(__FILE__,__LINE__, \
                          _fc_msg_stream.str()); \
 } else 


#endif

