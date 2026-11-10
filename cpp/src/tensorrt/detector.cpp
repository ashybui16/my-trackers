#include "mot/tensorrt/detector.hpp"

#include "detection/yolo/postprocess.hpp"
#include "detection/yolo/preprocess.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mot {
namespace {

class TensorRTLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT: " << message << '\n';
        }
    }
};

template <typename T>
struct TensorRTDestroyer {
    void operator()(T* object) const noexcept {
        if (object != nullptr) {
            object->destroy();
        }
    }
};

template <typename T>
using TensorRTPtr = std::unique_ptr<T, TensorRTDestroyer<T>>;

void checkCuda(cudaError_t status, const std::string& operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
    }
}

class CudaBuffer {
public:
    CudaBuffer() = default;
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;

    ~CudaBuffer() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    void allocate(std::size_t bytes, const std::string& label) {
        if (data_ != nullptr) {
            throw std::logic_error("CUDA buffer '" + label + "' is already allocated");
        }
        checkCuda(cudaMalloc(&data_, bytes), "cudaMalloc " + label);
    }

    void* data() const { return data_; }

private:
    void* data_{nullptr};
};

class CudaStream {
public:
    CudaStream() = default;
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    ~CudaStream() {
        if (stream_ != nullptr) {
            cudaStreamDestroy(stream_);
        }
    }

    void create() { checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate"); }
    void synchronize() const { checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize"); }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_{nullptr};
};

std::string dimsString(const nvinfer1::Dims& dims) {
    std::ostringstream stream;
    stream << '[';
    for (int index = 0; index < dims.nbDims; ++index) {
        if (index != 0) {
            stream << ',';
        }
        stream << dims.d[index];
    }
    stream << ']';
    return stream.str();
}

using detail::LetterboxInfo;
using detail::preprocess;

// Pinned transfers when available; pageable fallback only for allocation pressure.
class HostFloatBuffer {
public:
    HostFloatBuffer() = default;
    HostFloatBuffer(const HostFloatBuffer&) = delete;
    HostFloatBuffer& operator=(const HostFloatBuffer&) = delete;
    ~HostFloatBuffer() { if (pinned_) cudaFreeHost(data_); }
    void allocate(std::size_t elements) {
        void* memory = nullptr;
        const auto status = cudaMallocHost(&memory, elements * sizeof(float));
        if (status == cudaSuccess) {
            data_ = static_cast<float*>(memory);
            pinned_ = true;
        } else if (status == cudaErrorMemoryAllocation) {
            cudaGetLastError(); // Do not leave this handled allocation error pending.
            fallback_.resize(elements);
            data_ = fallback_.data();
        } else {
            checkCuda(status, "cudaMallocHost");
        }
        size_ = elements;
    }
    float* data() const { return data_; }
    std::size_t size() const { return size_; }
    bool pinned() const { return pinned_; }
private:
    float* data_{nullptr};
    std::size_t size_{0};
    bool pinned_{false};
    std::vector<float> fallback_;
};

}

struct TensorRTDetector::Impl {
    // Runtime and engine must outlive the context.
    TensorRTLogger logger;
    TensorRTPtr<nvinfer1::IRuntime> runtime{nullptr};
    TensorRTPtr<nvinfer1::ICudaEngine> engine{nullptr};
    TensorRTPtr<nvinfer1::IExecutionContext> context{nullptr};
    int input_binding_index{-1};
    int output_binding_index{-1};
    std::string input_binding_name;
    std::string output_binding_name;
    int candidate_count{0};
    std::size_t input_elements{0};
    std::size_t output_elements{0};
    std::array<void*, 2> bindings{};

    // Host buffers are kept alive until the CUDA stream completes.
    HostFloatBuffer host_input;
    HostFloatBuffer host_output;
    detail::OutputWorkspace output_workspace;
    cv::Mat canvas;

    // Release GPU resources before host buffers, even if a transfer throws.
    CudaStream stream;
    CudaBuffer device_output;
    CudaBuffer device_input;

    void loadEngine(const std::string& engine_path);
    void initializeBindings(const DetectorConfig& config);
    void allocateBuffers(const DetectorConfig& config);

    void uploadInput();
    void execute();
    void downloadOutput();
    std::vector<mot::Detection> decodeOutput(const cv::Size& frame_size,
                                                    const LetterboxInfo& letterbox,
                                                    const DetectorConfig& config,
                                                    std::vector<mot::Detection>* recovered);
};

void TensorRTDetector::Impl::loadEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open TensorRT engine: " + engine_path);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + engine_path);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> serialized(static_cast<std::size_t>(size));
    if (!file.read(serialized.data(), size)) {
        throw std::runtime_error("cannot read TensorRT engine: " + engine_path);
    }

    runtime.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        throw std::runtime_error("TensorRT runtime creation failed; check TensorRT/CUDA installation");
    }
    engine.reset(runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!engine) {
        throw std::runtime_error("TensorRT engine deserialization failed; serialized engines are tied to the target GPU and TensorRT version, so rebuild the engine");
    }
    context.reset(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("TensorRT execution context creation failed");
    }
}

void TensorRTDetector::Impl::initializeBindings(const DetectorConfig& config) {
    if (engine->hasImplicitBatchDimension() || engine->getNbBindings() != 2) {
        throw std::runtime_error("engine must have explicit batch and exactly one input and one output binding");
    }
    for (int index = 0; index < 2; ++index) {
        const char* raw_name = engine->getBindingName(index);
        const std::string name = raw_name == nullptr ? "<unnamed>" : raw_name;
        if (engine->getBindingDataType(index) != nvinfer1::DataType::kFLOAT ||
            engine->getBindingFormat(index) != nvinfer1::TensorFormat::kLINEAR ||
            engine->getLocation(index) != nvinfer1::TensorLocation::kDEVICE) {
            throw std::runtime_error("binding '" + name + "' must use linear FP32 device storage");
        }
        if (engine->bindingIsInput(index)) {
            if (input_binding_index >= 0) {
                throw std::runtime_error("engine must contain exactly one input binding");
            }
            input_binding_index = index;
            input_binding_name = name;
        } else {
            if (output_binding_index >= 0) {
                throw std::runtime_error("engine must contain exactly one output binding");
            }
            output_binding_index = index;
            output_binding_name = name;
        }
    }
    if (input_binding_index < 0 || output_binding_index < 0) {
        throw std::runtime_error("engine must contain one input and one output binding");
    }

    const auto input_dims = engine->getBindingDimensions(input_binding_index);
    if (input_dims.nbDims != 4 || input_dims.d[0] != 1 || input_dims.d[1] != 3 ||
        input_dims.d[2] != config.input_height || input_dims.d[3] != config.input_width) {
        throw std::runtime_error("input binding '" + input_binding_name + "' must have fixed shape [1,3," +
                                 std::to_string(config.input_height) + "," + std::to_string(config.input_width) +
                                 "], got " + dimsString(input_dims));
    }
    const auto output_dims = engine->getBindingDimensions(output_binding_index);
    if (output_dims.nbDims != 3 || output_dims.d[0] != 1 ||
        output_dims.d[1] != 4 + config.class_count || output_dims.d[2] <= 0) {
        throw std::runtime_error("output binding '" + output_binding_name + "' must have fixed shape [1," +
                                 std::to_string(4 + config.class_count) + ",N], got " + dimsString(output_dims));
    }
    candidate_count = output_dims.d[2];
    input_elements = static_cast<std::size_t>(3) * config.input_height * config.input_width;
    output_elements = static_cast<std::size_t>(output_dims.d[1]) * candidate_count;
}

void TensorRTDetector::Impl::allocateBuffers(const DetectorConfig& config) {
    host_input.allocate(input_elements);
    host_output.allocate(output_elements);
    if (!host_input.pinned() || !host_output.pinned())
        std::cerr << "TensorRT: pinned host allocation exhausted; using pageable transfer buffer\n";
    canvas.create(config.input_height, config.input_width, CV_8UC3);
    device_input.allocate(input_elements * sizeof(float), "input");
    device_output.allocate(output_elements * sizeof(float), "output");
    stream.create();
    bindings[static_cast<std::size_t>(input_binding_index)] = device_input.data();
    bindings[static_cast<std::size_t>(output_binding_index)] = device_output.data();
}

void TensorRTDetector::Impl::uploadInput() {
    checkCuda(cudaMemcpyAsync(device_input.data(), host_input.data(), host_input.size() * sizeof(float),
                              cudaMemcpyHostToDevice, stream.get()),
              "cudaMemcpyAsync input");
}

void TensorRTDetector::Impl::execute() {
    if (!context->enqueueV2(bindings.data(), stream.get(), nullptr)) {
        throw std::runtime_error("TensorRT enqueue failed for bindings '" + input_binding_name + "' -> '" +
                                 output_binding_name + "'");
    }
}

void TensorRTDetector::Impl::downloadOutput() {
    checkCuda(cudaMemcpyAsync(host_output.data(), device_output.data(), host_output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, stream.get()),
              "cudaMemcpyAsync output");
    stream.synchronize();
}

std::vector<mot::Detection> TensorRTDetector::Impl::decodeOutput(
    const cv::Size& frame_size, const LetterboxInfo& letterbox, const DetectorConfig& config,
    std::vector<mot::Detection>* recovered) {
    detail::OutputOptions options;
    options.class_count = config.class_count;
    options.conf_threshold = config.conf_threshold;
    options.iou_threshold = config.iou_threshold;
    options.max_det = config.max_det;
    options.transform = {letterbox.scale, letterbox.pad_x, letterbox.pad_y, frame_size.width, frame_size.height};
    return detail::processOutput(host_output.data(), candidate_count, options, output_workspace, recovered);
}

TensorRTDetector::TensorRTDetector(const std::string& engine_path, DetectorConfig config,
                                   std::vector<std::string> class_names)
    : config_(std::move(config)), engine_path_(engine_path),
      class_names_(std::move(class_names)), impl_(std::make_unique<Impl>()) {
    config_.validate();
    impl_->loadEngine(engine_path_);
    impl_->initializeBindings(config_);
    impl_->allocateBuffers(config_);
}

TensorRTDetector::~TensorRTDetector() = default;
TensorRTDetector::TensorRTDetector(TensorRTDetector&&) noexcept = default;
TensorRTDetector& TensorRTDetector::operator=(TensorRTDetector&&) noexcept = default;

const std::string& TensorRTDetector::inputBindingName() const {
    return impl_->input_binding_name;
}

const std::string& TensorRTDetector::outputBindingName() const {
    return impl_->output_binding_name;
}

std::vector<mot::Detection> TensorRTDetector::detect(const cv::Mat& frame) {
    const LetterboxInfo letterbox = preprocess(frame, impl_->canvas, impl_->host_input.data());
    impl_->uploadInput();
    impl_->execute();
    impl_->downloadOutput();
    return impl_->decodeOutput(frame.size(), letterbox, config_, nullptr);
}

std::vector<mot::Detection> TensorRTDetector::detectWithRecovered(const cv::Mat& frame,
                                                                  std::vector<mot::Detection>& recovered) {
    const LetterboxInfo letterbox = preprocess(frame, impl_->canvas, impl_->host_input.data());
    impl_->uploadInput();
    impl_->execute();
    impl_->downloadOutput();
    return impl_->decodeOutput(frame.size(), letterbox, config_, &recovered);
}

}
