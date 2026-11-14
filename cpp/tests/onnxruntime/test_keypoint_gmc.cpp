#include "mot/onnxruntime/keypoint_gmc.hpp"

#include "support/keypoint_gmc_checks.hpp"

#include <memory>
#include <string>

int main(int argc, char** argv) {
    return mot::test::runKeypointGmcTest(argc, argv, "ONNX Runtime",
        [](const std::string& dpoint, const std::string& matcher) {
            return std::make_unique<mot::OnnxRuntimeKeypointGmc>(dpoint, matcher);
        });
}
