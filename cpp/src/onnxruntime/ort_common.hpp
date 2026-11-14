#pragma once

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace mot::detail {

inline std::string shapeString(const std::vector<std::int64_t>& shape) {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << shape[index];
    }
    stream << ']';
    return stream.str();
}

// One Env shared by every session in the process; it must outlive every session.
inline Ort::Env& sharedOrtEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mot");
    return env;
}

}
