#pragma once

#include <stdexcept>
#include <string>

namespace mot::test {

inline void expect(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}
