#include "mot_inference/detector.hpp"

#include "mot_tracker/geometry.hpp"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mot_inference {
namespace {

class TensorRTLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        // Chỉ chuyển cảnh báo/lỗi TensorRT ra stderr.
        if (severity <= Severity::kWARNING) {
            std::cerr << "TensorRT: " << message << '\n';
        }
    }
};

template <typename T>
struct TensorRTDestroyer {
    void operator()(T* object) const noexcept {
        // Hủy tài nguyên TensorRT qua API destroy tương ứng.
        if (object != nullptr) {
            object->destroy();
        }
    }
};

template <typename T>
using TensorRTPtr = std::unique_ptr<T, TensorRTDestroyer<T>>;

void checkCuda(cudaError_t status, const std::string& operation) {
    // Đổi mã lỗi CUDA thành ngoại lệ có tên thao tác gây lỗi.
    if (status != cudaSuccess) {
        throw std::runtime_error(operation + ": " + cudaGetErrorString(status));
    }
}

std::string dimsString(const nvinfer1::Dims& dims) {
    // Ghép các chiều tensor thành chuỗi để hiện trong thông báo lỗi.
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

std::size_t elementCount(const nvinfer1::Dims& dims, const std::string& binding_name) {
    // Nhân các chiều đã xác định để biết số phần tử cần cấp phát.
    if (dims.nbDims <= 0) {
        throw std::runtime_error("binding '" + binding_name + "' has invalid rank " + dimsString(dims));
    }
    std::size_t count = 1;
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] <= 0) {
            throw std::runtime_error("binding '" + binding_name + "' has unresolved dimensions " + dimsString(dims));
        }
        count *= static_cast<std::size_t>(dims.d[index]);
    }
    return count;
}

std::size_t dataTypeSize(nvinfer1::DataType type, const std::string& binding_name) {
    // Chỉ engine FP32/FP16 được hỗ trợ khi tính byte cho binding.
    if (type == nvinfer1::DataType::kFLOAT) {
        return sizeof(float);
    }
    if (type == nvinfer1::DataType::kHALF) {
        return sizeof(__half);
    }
    throw std::runtime_error("binding '" + binding_name + "' must use FP32 or FP16");
}

bool hasDynamicDimension(const nvinfer1::Dims& dims) {
    // Chiều âm báo TensorRT cần gán kích thước runtime.
    for (int index = 0; index < dims.nbDims; ++index) {
        if (dims.d[index] < 0) {
            return true;
        }
    }
    return false;
}

nvinfer1::Dims makeInputDims(int height, int width) {
    // Tạo kích thước NCHW cho một ảnh RGB.
    nvinfer1::Dims dims{};
    dims.nbDims = 4;
    dims.d[0] = 1;
    dims.d[1] = 3;
    dims.d[2] = height;
    dims.d[3] = width;
    return dims;
}

void validateProfile(const nvinfer1::ICudaEngine& engine,
                     int binding_index,
                     const nvinfer1::Dims& target,
                     const std::string& binding_name) {
    // Đảm bảo kích thước ảnh nằm trong miền tối ưu của engine động.
    if (engine.getNbOptimizationProfiles() <= 0) {
        throw std::runtime_error("dynamic input binding '" + binding_name + "' has no optimization profile");
    }
    const auto minimum = engine.getProfileDimensions(
        binding_index, 0, nvinfer1::OptProfileSelector::kMIN);
    const auto maximum = engine.getProfileDimensions(
        binding_index, 0, nvinfer1::OptProfileSelector::kMAX);
    if (minimum.nbDims != target.nbDims || maximum.nbDims != target.nbDims) {
        throw std::runtime_error("binding '" + binding_name + "' profile rank does not match input rank");
    }
    for (int index = 0; index < target.nbDims; ++index) {
        if (target.d[index] < minimum.d[index] || target.d[index] > maximum.d[index]) {
            std::ostringstream message;
            message << "binding '" << binding_name << "' input dimension " << index
                    << "=" << target.d[index] << " is outside optimization profile ["
                    << minimum.d[index] << ',' << maximum.d[index] << ']';
            throw std::runtime_error(message.str());
        }
    }
}

// Thông tin co giãn và đệm để đưa hộp dự đoán về tọa độ ảnh gốc.
struct LetterboxInfo {
    double scale{1.0};
    int pad_x{0};
    int pad_y{0};
};

// Chuẩn hóa frame thành tensor RGB dạng CHW trong khoảng [0, 1].
std::vector<float> preprocess(const cv::Mat& frame,
                              int input_width,
                              int input_height,
                              LetterboxInfo& info) {
    // Đổi sang BGR 8 bit, giữ tỉ lệ bằng letterbox, rồi xếp RGB theo kênh.
    if (frame.empty() || frame.dims != 2 || frame.rows < 1 || frame.cols < 1) {
        throw std::invalid_argument("frame must be a non-empty image");
    }
    if (frame.channels() != 1 && frame.channels() != 3) {
        throw std::invalid_argument("frame must have one grayscale or three BGR channels");
    }
    cv::Mat bgr;
    if (frame.channels() == 1) {
        cv::cvtColor(frame, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = frame;
    }
    if (bgr.depth() != CV_8U) {
        cv::Mat converted;
        double minimum = 0.0;
        double maximum = 0.0;
        cv::minMaxLoc(bgr.reshape(1), &minimum, &maximum);
        if (std::isfinite(minimum) && std::isfinite(maximum) && maximum > minimum) {
            bgr.convertTo(converted, CV_8UC3, 255.0 / (maximum - minimum),
                          -minimum * 255.0 / (maximum - minimum));
        } else {
            bgr.convertTo(converted, CV_8UC3);
        }
        bgr = std::move(converted);
    }

    info.scale = std::min(static_cast<double>(input_width) / static_cast<double>(bgr.cols),
                          static_cast<double>(input_height) / static_cast<double>(bgr.rows));
    const int resized_width = std::max(1, static_cast<int>(std::round(bgr.cols * info.scale)));
    const int resized_height = std::max(1, static_cast<int>(std::round(bgr.rows * info.scale)));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(resized_width, resized_height));
    cv::Mat canvas(input_height, input_width, CV_8UC3, cv::Scalar(114, 114, 114));
    info.pad_x = (input_width - resized_width) / 2;
    info.pad_y = (input_height - resized_height) / 2;
    resized.copyTo(canvas(cv::Rect(info.pad_x, info.pad_y, resized_width, resized_height)));

    std::vector<float> chw(static_cast<std::size_t>(3) * input_height * input_width);
    const std::size_t plane = static_cast<std::size_t>(input_height) * input_width;
    for (int row = 0; row < input_height; ++row) {
        for (int column = 0; column < input_width; ++column) {
            const cv::Vec3b pixel = canvas.at<cv::Vec3b>(row, column);
            const std::size_t offset = static_cast<std::size_t>(row) * input_width + column;
            chw[offset] = static_cast<float>(pixel[2]) / 255.0F;
            chw[plane + offset] = static_cast<float>(pixel[1]) / 255.0F;
            chw[2 * plane + offset] = static_cast<float>(pixel[0]) / 255.0F;
        }
    }
    return chw;
}

// Bỏ phần đệm letterbox và đổi tỉ lệ hộp về kích thước frame ban đầu.
mot_tracker::BBox undoLetterbox(const std::array<float, 4>& box, const LetterboxInfo& info) {
    // Trừ đệm ở hai trục rồi chia tỉ lệ co ảnh.
    mot_tracker::BBox result{
        (static_cast<double>(box[0]) - info.pad_x) / info.scale,
        (static_cast<double>(box[1]) - info.pad_y) / info.scale,
        (static_cast<double>(box[2]) - info.pad_x) / info.scale,
        (static_cast<double>(box[3]) - info.pad_y) / info.scale,
    };
    return result;
}

// Một hộp ứng viên trước khi loại bỏ các hộp chồng lấp.
struct Candidate {
    mot_tracker::BBox bbox{};
    double score{0.0};
    int class_id{0};
    int source_index{0};
};

// Giữ tối đa max_det ứng viên điểm cao, loại hộp cùng lớp trùng IoU.
std::vector<mot_tracker::Detection> nms(std::vector<Candidate> candidates,
                                        double iou_threshold,
                                        int max_det) {
    // Sắp theo điểm giảm dần, rồi triệt hộp cùng lớp có IoU vượt ngưỡng.
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& first, const Candidate& second) {
        if (first.score != second.score) {
            return first.score > second.score;
        }
        return first.source_index < second.source_index;
    });
    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<mot_tracker::Detection> result;
    result.reserve(std::min<std::size_t>(candidates.size(), static_cast<std::size_t>(max_det)));
    for (std::size_t index = 0; index < candidates.size() && static_cast<int>(result.size()) < max_det; ++index) {
        if (suppressed[index]) {
            continue;
        }
        result.emplace_back(candidates[index].bbox,
                            candidates[index].score,
                            candidates[index].class_id);
        for (std::size_t other = index + 1; other < candidates.size(); ++other) {
            if (!suppressed[other] && candidates[other].class_id == candidates[index].class_id &&
                mot_tracker::iou(candidates[index].bbox, candidates[other].bbox) > iou_threshold) {
                suppressed[other] = true;
            }
        }
    }
    return result;
}

}  // namespace

struct TensorRTDetector::Impl {
    // Tài nguyên TensorRT và luồng CUDA sở hữu trong suốt vòng đời detector.
    TensorRTLogger logger;
    TensorRTPtr<nvinfer1::IRuntime> runtime{nullptr};
    TensorRTPtr<nvinfer1::ICudaEngine> engine{nullptr};
    TensorRTPtr<nvinfer1::IExecutionContext> context{nullptr};
    cudaStream_t stream{nullptr};
    // Bộ đệm trên GPU và dung lượng đã cấp để tái sử dụng giữa các frame.
    void* device_input{nullptr};
    void* device_output{nullptr};
    std::size_t device_input_bytes{0};
    std::size_t device_output_bytes{0};
    // Chỉ số, tên, kiểu và kích thước tensor của hai binding engine.
    int input_binding_index{-1};
    int output_binding_index{-1};
    std::string input_binding_name;
    std::string output_binding_name;
    nvinfer1::DataType input_type{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType output_type{nvinfer1::DataType::kFLOAT};
    nvinfer1::Dims input_dims{};
    nvinfer1::Dims output_dims{};
    // Bộ đệm CPU cho dữ liệu FP32/FP16 trước và sau suy luận.
    std::vector<float> host_input;
    std::vector<float> host_output_float;
    std::vector<__half> host_input_half;
    std::vector<__half> host_output_half;
    std::vector<void*> bindings;

    ~Impl() {
        // Giải phóng bộ đệm GPU và stream khi detector kết thúc.
        if (device_input != nullptr) {
            cudaFree(device_input);
        }
        if (device_output != nullptr) {
            cudaFree(device_output);
        }
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
    }

    // Chỉ cấp phát lại bộ đệm GPU khi dung lượng hiện tại không đủ.
    void allocate(void** pointer, std::size_t& current_size, std::size_t required, const std::string& label) {
        // Giữ bộ đệm cũ nếu đủ lớn, nếu không thì giải phóng và cấp lại.
        if (current_size >= required && *pointer != nullptr) {
            return;
        }
        if (*pointer != nullptr) {
            checkCuda(cudaFree(*pointer), "cudaFree " + label);
            *pointer = nullptr;
            current_size = 0;
        }
        checkCuda(cudaMalloc(pointer, required), "cudaMalloc " + label);
        current_size = required;
    }
};

void DetectorConfig::validate() const {
    // Kiểm tra kích thước ảnh, số lớp và các ngưỡng trước khi nạp engine.
    if (input_width <= 0 || input_height <= 0) {
        throw std::invalid_argument("detector input dimensions must be positive");
    }
    if (!std::isfinite(conf_threshold) || conf_threshold < 0.0 || conf_threshold > 1.0) {
        throw std::invalid_argument("conf_threshold must be finite and in [0, 1]");
    }
    if (!std::isfinite(iou_threshold) || iou_threshold < 0.0 || iou_threshold > 1.0) {
        throw std::invalid_argument("iou_threshold must be finite and in [0, 1]");
    }
    if (max_det <= 0 || class_count <= 0) {
        throw std::invalid_argument("max_det and class_count must be positive");
    }
}

TensorRTDetector::TensorRTDetector(const std::string& engine_path, DetectorConfig config)
    : config_(std::move(config)), engine_path_(engine_path), impl_(std::make_unique<Impl>()) {
    // Đọc engine, tạo runtime/context, kiểm tra binding và cấp bộ đệm GPU.
    config_.validate();
    std::ifstream file(engine_path_, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("cannot open TensorRT engine: " + engine_path_);
    }
    const std::streamsize size = file.tellg();
    if (size <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + engine_path_);
    }
    file.seekg(0, std::ios::beg);
    std::vector<char> serialized(static_cast<std::size_t>(size));
    if (!file.read(serialized.data(), size)) {
        throw std::runtime_error("cannot read TensorRT engine: " + engine_path_);
    }

    impl_->runtime.reset(nvinfer1::createInferRuntime(impl_->logger));
    if (!impl_->runtime) {
        throw std::runtime_error("TensorRT runtime creation failed; check TensorRT/CUDA installation");
    }
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
    if (!impl_->engine) {
        throw std::runtime_error("TensorRT engine deserialization failed; serialized engines are tied to the target GPU and TensorRT version, so rebuild the engine");
    }
    impl_->context.reset(impl_->engine->createExecutionContext());
    if (!impl_->context) {
        throw std::runtime_error("TensorRT execution context creation failed");
    }

    // Tìm đúng một binding đầu vào và một binding đầu ra trong profile đầu tiên.
    const int total_bindings = impl_->engine->getNbBindings();
    const int profiles = std::max(1, impl_->engine->getNbOptimizationProfiles());
    const int bindings_per_profile = total_bindings / profiles;
    for (int index = 0; index < bindings_per_profile; ++index) {
        const char* raw_name = impl_->engine->getBindingName(index);
        const std::string name = raw_name == nullptr ? "<unnamed>" : raw_name;
        if (impl_->engine->bindingIsInput(index)) {
            if (impl_->input_binding_index >= 0) {
                throw std::runtime_error("engine must contain exactly one input binding; duplicate input '" + name + "'");
            }
            impl_->input_binding_index = index;
            impl_->input_binding_name = name;
        } else {
            if (impl_->output_binding_index >= 0) {
                throw std::runtime_error("engine must contain exactly one output binding; duplicate output '" + name + "'");
            }
            impl_->output_binding_index = index;
            impl_->output_binding_name = name;
        }
    }
    if (impl_->input_binding_index < 0 || impl_->output_binding_index < 0) {
        throw std::runtime_error("TensorRT engine must contain one input and one output binding");
    }
    impl_->input_type = impl_->engine->getBindingDataType(impl_->input_binding_index);
    impl_->output_type = impl_->engine->getBindingDataType(impl_->output_binding_index);
    dataTypeSize(impl_->input_type, impl_->input_binding_name);
    dataTypeSize(impl_->output_type, impl_->output_binding_name);
    impl_->input_dims = impl_->engine->getBindingDimensions(impl_->input_binding_index);
    if (impl_->input_dims.nbDims != 4) {
        throw std::runtime_error("input binding '" + impl_->input_binding_name + "' must be NCHW rank 4, got " + dimsString(impl_->input_dims));
    }
    if ((impl_->input_dims.d[0] != 1 && impl_->input_dims.d[0] != -1) ||
        (impl_->input_dims.d[1] != 3 && impl_->input_dims.d[1] != -1)) {
        throw std::runtime_error("input binding '" + impl_->input_binding_name + "' must be batch-1 RGB NCHW, got " + dimsString(impl_->input_dims));
    }
    // Áp kích thước yêu cầu cho engine động hoặc kiểm tra engine kích thước cố định.
    const nvinfer1::Dims requested = makeInputDims(config_.input_height, config_.input_width);
    if (hasDynamicDimension(impl_->input_dims)) {
        validateProfile(*impl_->engine, impl_->input_binding_index, requested, impl_->input_binding_name);
        if (!impl_->context->setBindingDimensions(impl_->input_binding_index, requested)) {
            throw std::runtime_error("could not set dynamic dimensions for input binding '" + impl_->input_binding_name + "'");
        }
    } else {
        if (impl_->input_dims.d[0] != requested.d[0] || impl_->input_dims.d[1] != requested.d[1] ||
            impl_->input_dims.d[2] != requested.d[2] || impl_->input_dims.d[3] != requested.d[3]) {
            throw std::runtime_error("static input binding '" + impl_->input_binding_name + "' has shape " +
                                     dimsString(impl_->input_dims) + ", requested [1,3," +
                                     std::to_string(config_.input_height) + "," +
                                     std::to_string(config_.input_width) + "]");
        }
    }
    impl_->output_dims = impl_->context->getBindingDimensions(impl_->output_binding_index);
    const std::size_t input_elements = elementCount(impl_->context->getBindingDimensions(impl_->input_binding_index),
                                                    impl_->input_binding_name);
    const std::size_t output_elements = elementCount(impl_->output_dims, impl_->output_binding_name);
    impl_->allocate(&impl_->device_input, impl_->device_input_bytes,
                    input_elements * dataTypeSize(impl_->input_type, impl_->input_binding_name), "input");
    impl_->allocate(&impl_->device_output, impl_->device_output_bytes,
                    output_elements * dataTypeSize(impl_->output_type, impl_->output_binding_name), "output");
    checkCuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");
    impl_->bindings.assign(static_cast<std::size_t>(total_bindings), nullptr);
}

TensorRTDetector::~TensorRTDetector() = default;
TensorRTDetector::TensorRTDetector(TensorRTDetector&&) noexcept = default;
TensorRTDetector& TensorRTDetector::operator=(TensorRTDetector&&) noexcept = default;

const std::string& TensorRTDetector::inputBindingName() const {
    // Trả tên binding đã tìm khi nạp engine.
    return impl_->input_binding_name;
}

const std::string& TensorRTDetector::outputBindingName() const {
    // Trả tên binding đầu ra đã tìm khi nạp engine.
    return impl_->output_binding_name;
}

std::vector<mot_tracker::Detection> TensorRTDetector::infer(const cv::Mat& frame) {
    // Chuẩn bị ảnh, chép input lên GPU và chạy engine trên CUDA stream.
    LetterboxInfo letterbox;
    const std::vector<float> input = preprocess(frame, config_.input_width, config_.input_height, letterbox);
    const std::size_t input_elements = input.size();
    if (impl_->input_type == nvinfer1::DataType::kFLOAT) {
        impl_->host_input = input;
        checkCuda(cudaMemcpyAsync(impl_->device_input,
                                  impl_->host_input.data(),
                                  input_elements * sizeof(float),
                                  cudaMemcpyHostToDevice,
                                  impl_->stream),
                  "cudaMemcpyAsync input");
    } else {
        impl_->host_input_half.resize(input_elements);
        for (std::size_t index = 0; index < input_elements; ++index) {
            impl_->host_input_half[index] = __float2half(input[index]);
        }
        checkCuda(cudaMemcpyAsync(impl_->device_input,
                                  impl_->host_input_half.data(),
                                  input_elements * sizeof(__half),
                                  cudaMemcpyHostToDevice,
                                  impl_->stream),
                  "cudaMemcpyAsync FP16 input");
    }

    impl_->bindings[static_cast<std::size_t>(impl_->input_binding_index)] = impl_->device_input;
    impl_->bindings[static_cast<std::size_t>(impl_->output_binding_index)] = impl_->device_output;
    if (!impl_->context->enqueueV2(impl_->bindings.data(), impl_->stream, nullptr)) {
        throw std::runtime_error("TensorRT enqueue failed for bindings '" + impl_->input_binding_name + "' -> '" +
                                 impl_->output_binding_name + "'");
    }
    // Chép output về CPU, đổi FP16 sang FP32 khi cần.
    const std::size_t output_elements = elementCount(impl_->output_dims, impl_->output_binding_name);
    if (impl_->output_type == nvinfer1::DataType::kFLOAT) {
        impl_->host_output_float.resize(output_elements);
        checkCuda(cudaMemcpyAsync(impl_->host_output_float.data(),
                                  impl_->device_output,
                                  output_elements * sizeof(float),
                                  cudaMemcpyDeviceToHost,
                                  impl_->stream),
                  "cudaMemcpyAsync output");
        checkCuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
    } else {
        impl_->host_output_half.resize(output_elements);
        checkCuda(cudaMemcpyAsync(impl_->host_output_half.data(),
                                  impl_->device_output,
                                  output_elements * sizeof(__half),
                                  cudaMemcpyDeviceToHost,
                                  impl_->stream),
                  "cudaMemcpyAsync FP16 output");
        checkCuda(cudaStreamSynchronize(impl_->stream), "cudaStreamSynchronize");
        impl_->host_output_float.resize(output_elements);
        for (std::size_t index = 0; index < output_elements; ++index) {
            impl_->host_output_float[index] = __half2float(impl_->host_output_half[index]);
        }
    }
    const std::vector<float>& output = impl_->host_output_float;

    const nvinfer1::Dims dims = impl_->output_dims;
    if (dims.nbDims != 3 || dims.d[0] != 1) {
        throw std::runtime_error("output binding '" + impl_->output_binding_name + "' must be rank-3 batch-1 [1,14,N], [1,N,14], or [1,N,6], got " + dimsString(dims));
    }
    // Engine có thể xuất [1, số đặc trưng, N], [1, N, số đặc trưng] hoặc [1, N, 6].
    const int features_first = dims.d[1];
    const int candidates_first = dims.d[2];
    const bool raw_feature_first = features_first == 4 + config_.class_count;
    const bool raw_candidate_first = candidates_first == 4 + config_.class_count;
    const bool postprocessed = dims.d[2] == 6 && !raw_feature_first;
    if (!raw_feature_first && !raw_candidate_first && !postprocessed) {
        throw std::runtime_error("output binding '" + impl_->output_binding_name + "' has unsupported shape " +
                                 dimsString(dims) + "; expected [1," + std::to_string(4 + config_.class_count) + ",N], [1,N," +
                                 std::to_string(4 + config_.class_count) + "], or [1,N,6]");
    }

    // Đọc một giá trị bất kể thứ tự chiều feature/candidate của tensor.
    auto at = [&output, &dims](int candidate, int feature, bool feature_first) -> float {
        const std::size_t offset = feature_first
                                        ? static_cast<std::size_t>(feature) * dims.d[2] + candidate
                                        : static_cast<std::size_t>(candidate) * dims.d[2] + feature;
        return output[offset];
    };
    // Đọc từng ứng viên, chọn lớp/điểm, bỏ letterbox và lọc hộp không hợp lệ.
    std::vector<Candidate> candidates;
    const int candidate_count = raw_feature_first ? dims.d[2] : dims.d[1];
    candidates.reserve(static_cast<std::size_t>(candidate_count));
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
        std::array<float, 4> input_box{};
        double score = 0.0;
        int class_id = -1;
        if (postprocessed) {
            input_box = {at(candidate, 0, false), at(candidate, 1, false),
                         at(candidate, 2, false), at(candidate, 3, false)};
            score = at(candidate, 4, false);
            class_id = static_cast<int>(std::lround(at(candidate, 5, false)));
        } else {
            const bool feature_first = raw_feature_first;
            const float center_x = at(candidate, 0, feature_first);
            const float center_y = at(candidate, 1, feature_first);
            const float width = at(candidate, 2, feature_first);
            const float height = at(candidate, 3, feature_first);
            input_box = {center_x - width / 2.0F, center_y - height / 2.0F,
                         center_x + width / 2.0F, center_y + height / 2.0F};
            for (int class_index = 0; class_index < config_.class_count; ++class_index) {
                const float class_score = at(candidate, 4 + class_index, feature_first);
                if (std::isfinite(class_score) && (class_id < 0 || class_score > score)) {
                    score = class_score;
                    class_id = class_index;
                }
            }
        }
        if (!std::isfinite(score) || score < config_.conf_threshold || class_id < 0 || class_id >= config_.class_count) {
            continue;
        }
        const mot_tracker::BBox box = undoLetterbox(input_box, letterbox);
        try {
            mot_tracker::validateBBox(box);
        } catch (const std::invalid_argument&) {
            continue;
        }
        candidates.push_back({box, score, class_id, candidate});
    }
    return nms(std::move(candidates), config_.iou_threshold, config_.max_det);
}

}  // namespace mot_inference
