#include "mot/onnxruntime/detector.hpp"

#include "detection/yolo/postprocess.hpp"
#include "detection/yolo/preprocess.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mot {
namespace {

std::string shapeString(const std::vector<std::int64_t>& shape) {
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

// One Env shared by every detector in the process; it must outlive every session.
Ort::Env& sharedEnv() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "mot");
    return env;
}

}

struct OnnxRuntimeDetector::Impl {
    Ort::Session session{nullptr};
    std::string input_name;
    std::string output_name;
    int candidate_count{0};

    // Fixed CPU buffers; both Ort::Values point into them, so Run does not allocate outputs.
    std::vector<float> input;
    std::vector<float> output;
    Ort::Value input_tensor{nullptr};
    Ort::Value output_tensor{nullptr};
    detail::OutputWorkspace output_workspace;
    cv::Mat canvas;

    void loadModel(const std::string& model_path, int intra_op_threads);
    void initializeTensors(const DetectorConfig& config);
    void run();
};

void OnnxRuntimeDetector::Impl::loadModel(const std::string& model_path, int intra_op_threads) {
    if (!std::filesystem::is_regular_file(model_path)) {
        throw std::runtime_error("cannot open ONNX model: " + model_path);
    }
    if (intra_op_threads < 0) {
        throw std::invalid_argument("intra_op_threads must be >= 0");
    }
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(intra_op_threads);
    options.SetInterOpNumThreads(1);
    session = Ort::Session(sharedEnv(), model_path.c_str(), options);
}

void OnnxRuntimeDetector::Impl::initializeTensors(const DetectorConfig& config) {
    if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
        throw std::runtime_error("ONNX model must have exactly one input and one output");
    }
    Ort::AllocatorWithDefaultOptions allocator;
    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();

    // TypeInfo owns the data; TensorTypeAndShapeInfo is only a view, so TypeInfo must stay alive.
    const Ort::TypeInfo input_type = session.GetInputTypeInfo(0);
    const Ort::TypeInfo output_type = session.GetOutputTypeInfo(0);
    const auto input_info = input_type.GetTensorTypeAndShapeInfo();
    const auto output_info = output_type.GetTensorTypeAndShapeInfo();
    if (input_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("ONNX model input '" + input_name + "' and output '" + output_name +
                                 "' must be FP32 tensors");
    }

    const std::vector<std::int64_t> input_shape = input_info.GetShape();
    if (input_shape.size() != 4 || input_shape[0] != 1 || input_shape[1] != 3 ||
        input_shape[2] != config.input_height || input_shape[3] != config.input_width) {
        throw std::runtime_error("input '" + input_name + "' must have fixed shape [1,3," +
                                 std::to_string(config.input_height) + "," + std::to_string(config.input_width) +
                                 "], got " + shapeString(input_shape) + " (export with dynamic=False)");
    }
    const std::vector<std::int64_t> output_shape = output_info.GetShape();
    if (output_shape.size() != 3 || output_shape[0] != 1 ||
        output_shape[1] != 4 + config.class_count || output_shape[2] <= 0) {
        throw std::runtime_error("output '" + output_name + "' must have fixed shape [1," +
                                 std::to_string(4 + config.class_count) + ",N], got " +
                                 shapeString(output_shape) + " (export with dynamic=False, nms=False)");
    }
    candidate_count = static_cast<int>(output_shape[2]);

    input.assign(static_cast<std::size_t>(3) * config.input_height * config.input_width, 0.0F);
    output.assign(static_cast<std::size_t>(output_shape[1]) * candidate_count, 0.0F);
    canvas.create(config.input_height, config.input_width, CV_8UC3);

    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    input_tensor = Ort::Value::CreateTensor<float>(memory, input.data(), input.size(),
                                                   input_shape.data(), input_shape.size());
    output_tensor = Ort::Value::CreateTensor<float>(memory, output.data(), output.size(),
                                                    output_shape.data(), output_shape.size());
}

void OnnxRuntimeDetector::Impl::run() {
    const char* input_names[] = {input_name.c_str()};
    const char* output_names[] = {output_name.c_str()};
    session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, &output_tensor, 1);
}

OnnxRuntimeDetector::OnnxRuntimeDetector(const std::string& model_path, DetectorConfig config,
                                         std::vector<std::string> class_names, int intra_op_threads)
    : config_(std::move(config)), model_path_(model_path),
      class_names_(std::move(class_names)), impl_(std::make_unique<Impl>()) {
    config_.validate();
    try {
        impl_->loadModel(model_path_, intra_op_threads);
        impl_->initializeTensors(config_);
    } catch (const Ort::Exception& error) {
        throw std::runtime_error("ONNX Runtime failed to load '" + model_path_ + "': " + error.what());
    }
}

OnnxRuntimeDetector::~OnnxRuntimeDetector() = default;
OnnxRuntimeDetector::OnnxRuntimeDetector(OnnxRuntimeDetector&&) noexcept = default;
OnnxRuntimeDetector& OnnxRuntimeDetector::operator=(OnnxRuntimeDetector&&) noexcept = default;

const std::string& OnnxRuntimeDetector::inputName() const { return impl_->input_name; }

const std::string& OnnxRuntimeDetector::outputName() const { return impl_->output_name; }

std::vector<Detection> OnnxRuntimeDetector::detect(const cv::Mat& frame) {
    return run(frame, nullptr);
}

std::vector<Detection> OnnxRuntimeDetector::detectWithRecovered(const cv::Mat& frame,
                                                                std::vector<Detection>& recovered) {
    return run(frame, &recovered);
}

std::vector<Detection> OnnxRuntimeDetector::run(const cv::Mat& frame, std::vector<Detection>* recovered) {
    const detail::LetterboxInfo letterbox = detail::preprocess(frame, impl_->canvas, impl_->input.data());
    try {
        impl_->run();
    } catch (const Ort::Exception& error) {
        throw std::runtime_error("ONNX Runtime inference failed for '" + model_path_ + "': " + error.what());
    }
    detail::OutputOptions options;
    options.class_count = config_.class_count;
    options.conf_threshold = config_.conf_threshold;
    options.iou_threshold = config_.iou_threshold;
    options.max_det = config_.max_det;
    options.transform = {letterbox.scale, letterbox.pad_x, letterbox.pad_y, frame.cols, frame.rows};
    return detail::processOutput(impl_->output.data(), impl_->candidate_count, options,
                                 impl_->output_workspace, recovered);
}

}
