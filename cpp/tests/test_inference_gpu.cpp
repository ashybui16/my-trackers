#include "mot_inference/detector.hpp"

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    // Chỉ chạy smoke test khi có GPU CUDA và đường dẫn engine; sau đó suy luận một frame rỗng.
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "No NVIDIA CUDA device is available; skipping TensorRT test\n";
        return 77;
    }
    const char* path = std::getenv("MOT_ENGINE_PATH");
    if (path == nullptr || std::string(path).empty()) {
        std::cerr << "MOT_ENGINE_PATH is not set; skipping TensorRT test\n";
        return 77;
    }
    try {
        mot_inference::TensorRTDetector detector(path);
        cv::Mat frame = cv::Mat::zeros(480, 640, CV_8UC3);
        const auto detections = detector.infer(frame);
        std::cout << "TensorRT inference succeeded with " << detections.size()
                  << " detections using bindings '" << detector.inputBindingName()
                  << "' -> '" << detector.outputBindingName() << "'\n";
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "TensorRT inference test failed: " << exception.what() << '\n';
        return 1;
    }
}
