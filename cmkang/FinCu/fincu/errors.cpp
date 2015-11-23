#include <stdexcept>
#include <fincu/errors.hpp>

namespace {

    std::string format(const std::string& file, long line,
                       const std::string& message) {
        std::ostringstream msg;
        msg << "\n" << file << ":" << line << ": \n";
        msg << message;
        return msg.str();
    }
}

namespace FinCu {

    Error::Error(const std::string& file, long line,
                 const std::string& message) {
        message_ = std::shared_ptr<std::string>(new std::string(
                                      format(file, line, message)));
    }

    const char* Error::what() const throw () {
        return message_->c_str();
    }

}

