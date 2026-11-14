#include "mot/onnxruntime/keypoint_gmc.hpp"

#include "onnxruntime/ort_common.hpp"

#include <onnxruntime_cxx_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>

namespace mot {
namespace {

using detail::shapeString;

constexpr std::array<const char*, 3> kDpointOutputs{"keypoints", "descriptors", "scores"};
constexpr std::array<const char*, 4> kMatcherInputs{"kpts1", "descs1", "kpts2", "descs2"};
constexpr std::array<const char*, 3> kMatcherOutputs{"matched_kpts1", "matched_kpts2", "valid_mask"};

Ort::Session openSession(const std::string& path, int intra_op_threads) {
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error("cannot open ONNX model: " + path);
    }
    Ort::SessionOptions options;
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    options.SetIntraOpNumThreads(intra_op_threads);
    options.SetInterOpNumThreads(1);
    return Ort::Session(detail::sharedOrtEnv(), path.c_str(), options);
}

std::vector<std::int64_t> floatShape(const Ort::TypeInfo& type, const std::string& what) {
    const auto info = type.GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(what + " must be an FP32 tensor");
    }
    return info.GetShape();
}

// Session-order index of each expected name.
template <std::size_t N>
std::array<std::size_t, N> indexByName(const std::vector<std::string>& names,
                                       const std::array<const char*, N>& expected, const std::string& what) {
    if (names.size() != N) {
        throw std::runtime_error(what + " must have exactly " + std::to_string(N) + " tensors");
    }
    std::array<std::size_t, N> indices{};
    for (std::size_t e = 0; e < N; ++e) {
        bool found = false;
        for (std::size_t index = 0; index < names.size(); ++index) {
            if (names[index] == expected[e]) {
                indices[e] = index;
                found = true;
            }
        }
        if (!found) throw std::runtime_error(what + " is missing tensor '" + expected[e] + "'");
    }
    return indices;
}

void expectShape(const std::vector<std::int64_t>& shape, const std::vector<std::int64_t>& expected,
                 const std::string& what) {
    if (shape != expected) {
        throw std::runtime_error(what + " must have fixed shape " + shapeString(expected) + ", got " +
                                 shapeString(shape));
    }
}

}

struct OnnxRuntimeKeypointGmc::Impl {
    Ort::Session dpoint{nullptr};
    Ort::Session matcher{nullptr};
    std::vector<std::string> dpoint_input_names, dpoint_output_names;
    std::vector<std::string> matcher_input_names, matcher_output_names;
    std::vector<const char*> dpoint_inputs, dpoint_outputs, matcher_inputs, matcher_outputs;

    int width{0};
    int height{0};
    int keypoints{0};
    int descriptor_size{0};

    // Two keypoint/descriptor slots: one for the current frame, one for the previous frame.
    std::vector<float> image;
    std::array<std::vector<float>, 2> kpts;
    std::array<std::vector<float>, 2> descs;
    std::vector<float> scores;
    std::vector<float> matched_previous, matched_current, valid;

    // Fixed views into the buffers above, in session order, so Run does not allocate outputs.
    Ort::Value image_value{nullptr};
    std::array<std::vector<Ort::Value>, 2> dpoint_output_values;   // by slot written
    std::array<std::vector<Ort::Value>, 2> matcher_input_values;   // by current slot
    std::vector<Ort::Value> matcher_output_values;

    int current_slot{0};
    bool has_previous{false};

    static std::vector<std::string> names(Ort::Session& session, bool inputs) {
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> result;
        const std::size_t count = inputs ? session.GetInputCount() : session.GetOutputCount();
        for (std::size_t index = 0; index < count; ++index) {
            result.emplace_back(inputs ? session.GetInputNameAllocated(index, allocator).get()
                                       : session.GetOutputNameAllocated(index, allocator).get());
        }
        return result;
    }

    void initialize() {
        dpoint_input_names = names(dpoint, true);
        dpoint_output_names = names(dpoint, false);
        matcher_input_names = names(matcher, true);
        matcher_output_names = names(matcher, false);
        for (auto [list, out] : {std::pair{&dpoint_input_names, &dpoint_inputs},
                                 std::pair{&dpoint_output_names, &dpoint_outputs},
                                 std::pair{&matcher_input_names, &matcher_inputs},
                                 std::pair{&matcher_output_names, &matcher_outputs}}) {
            for (const auto& name : *list) out->push_back(name.c_str());
        }

        if (dpoint_input_names.size() != 1) throw std::runtime_error("dpoint model must have exactly one input");
        const auto image_shape = floatShape(dpoint.GetInputTypeInfo(0), "dpoint input '" + dpoint_input_names[0] + "'");
        if (image_shape.size() != 4 || image_shape[0] != 1 || image_shape[1] != 1 ||
            image_shape[2] <= 0 || image_shape[3] <= 0) {
            throw std::runtime_error("dpoint input must have fixed shape [1,1,H,W], got " + shapeString(image_shape));
        }
        height = static_cast<int>(image_shape[2]);
        width = static_cast<int>(image_shape[3]);

        const auto dpoint_out = indexByName(dpoint_output_names, kDpointOutputs, "dpoint outputs");
        const auto kpts_shape = floatShape(dpoint.GetOutputTypeInfo(dpoint_out[0]), "dpoint output 'keypoints'");
        if (kpts_shape.size() != 3 || kpts_shape[0] != 1 || kpts_shape[1] <= 0 || kpts_shape[2] != 2) {
            throw std::runtime_error("dpoint output 'keypoints' must have fixed shape [1,N,2], got " +
                                     shapeString(kpts_shape));
        }
        const std::int64_t n = kpts_shape[1];
        const auto descs_shape = floatShape(dpoint.GetOutputTypeInfo(dpoint_out[1]), "dpoint output 'descriptors'");
        if (descs_shape.size() != 3 || descs_shape[0] != 1 || descs_shape[1] != n || descs_shape[2] <= 0) {
            throw std::runtime_error("dpoint output 'descriptors' must have fixed shape [1," + std::to_string(n) +
                                     ",D], got " + shapeString(descs_shape));
        }
        const std::int64_t d = descs_shape[2];
        expectShape(floatShape(dpoint.GetOutputTypeInfo(dpoint_out[2]), "dpoint output 'scores'"), {1, n},
                    "dpoint output 'scores'");

        const auto matcher_in = indexByName(matcher_input_names, kMatcherInputs, "matcher inputs");
        for (std::size_t e = 0; e < kMatcherInputs.size(); ++e) {
            const std::string what = std::string("matcher input '") + kMatcherInputs[e] + "'";
            expectShape(floatShape(matcher.GetInputTypeInfo(matcher_in[e]), what),
                        {1, n, e % 2 == 0 ? 2 : d}, what);
        }
        const auto matcher_out = indexByName(matcher_output_names, kMatcherOutputs, "matcher outputs");
        for (std::size_t e = 0; e < kMatcherOutputs.size(); ++e) {
            const std::string what = std::string("matcher output '") + kMatcherOutputs[e] + "'";
            const auto shape = floatShape(matcher.GetOutputTypeInfo(matcher_out[e]), what);
            expectShape(shape, e < 2 ? std::vector<std::int64_t>{1, n, 2} : std::vector<std::int64_t>{1, n}, what);
        }
        keypoints = static_cast<int>(n);
        descriptor_size = static_cast<int>(d);

        const auto un = static_cast<std::size_t>(n);
        image.assign(static_cast<std::size_t>(width) * height, 0.0F);
        for (int slot = 0; slot < 2; ++slot) {
            kpts[slot].assign(un * 2, 0.0F);
            descs[slot].assign(un * static_cast<std::size_t>(d), 0.0F);
        }
        scores.assign(un, 0.0F);
        matched_previous.assign(un * 2, 0.0F);
        matched_current.assign(un * 2, 0.0F);
        valid.assign(un, 0.0F);

        const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const auto view = [&](std::vector<float>& buffer, std::vector<std::int64_t> shape) {
            return Ort::Value::CreateTensor<float>(memory, buffer.data(), buffer.size(), shape.data(), shape.size());
        };
        image_value = view(image, image_shape);
        for (int slot = 0; slot < 2; ++slot) {
            const int other = 1 - slot;
            auto& out = dpoint_output_values[slot];
            out.resize(3);
            out[dpoint_out[0]] = view(kpts[slot], {1, n, 2});
            out[dpoint_out[1]] = view(descs[slot], {1, n, d});
            out[dpoint_out[2]] = view(scores, {1, n});
            auto& in = matcher_input_values[slot];
            in.resize(4);
            in[matcher_in[0]] = view(kpts[other], {1, n, 2});
            in[matcher_in[1]] = view(descs[other], {1, n, d});
            in[matcher_in[2]] = view(kpts[slot], {1, n, 2});
            in[matcher_in[3]] = view(descs[slot], {1, n, d});
        }
        matcher_output_values.resize(3);
        matcher_output_values[matcher_out[0]] = view(matched_previous, {1, n, 2});
        matcher_output_values[matcher_out[1]] = view(matched_current, {1, n, 2});
        matcher_output_values[matcher_out[2]] = view(valid, {1, n});
    }

    bool run(const cv::Mat& input, std::vector<cv::Point2f>& previous, std::vector<cv::Point2f>& current) {
        previous.clear();
        current.clear();
        if (input.type() != CV_32FC1 || input.cols != width || input.rows != height) {
            throw std::invalid_argument("keypoint GMC input must be CV_32FC1 of " + std::to_string(width) + "x" +
                                        std::to_string(height));
        }
        for (int row = 0; row < height; ++row) {
            std::memcpy(image.data() + static_cast<std::size_t>(row) * width, input.ptr<float>(row),
                        static_cast<std::size_t>(width) * sizeof(float));
        }
        auto& outputs = dpoint_output_values[current_slot];
        dpoint.Run(Ort::RunOptions{nullptr}, dpoint_inputs.data(), &image_value, 1,
                   dpoint_outputs.data(), outputs.data(), outputs.size());
        const bool matched = has_previous;
        if (matched) {
            auto& inputs = matcher_input_values[current_slot];
            matcher.Run(Ort::RunOptions{nullptr}, matcher_inputs.data(), inputs.data(), inputs.size(),
                        matcher_outputs.data(), matcher_output_values.data(), matcher_output_values.size());
            for (std::size_t index = 0; index < valid.size(); ++index) {
                if (valid[index] > 0.5F) {
                    previous.emplace_back(matched_previous[2 * index], matched_previous[2 * index + 1]);
                    current.emplace_back(matched_current[2 * index], matched_current[2 * index + 1]);
                }
            }
        }
        has_previous = true;
        current_slot = 1 - current_slot;
        return matched;
    }
};

OnnxRuntimeKeypointGmc::OnnxRuntimeKeypointGmc(const std::string& dpoint_path, const std::string& matcher_path,
                                               int intra_op_threads)
    : impl_(std::make_unique<Impl>()) {
    if (intra_op_threads < 0) throw std::invalid_argument("intra_op_threads must be >= 0");
    try {
        impl_->dpoint = openSession(dpoint_path, intra_op_threads);
        impl_->matcher = openSession(matcher_path, intra_op_threads);
        impl_->initialize();
    } catch (const Ort::Exception& error) {
        throw std::runtime_error("ONNX Runtime failed to load keypoint GMC models '" + dpoint_path + "', '" +
                                 matcher_path + "': " + error.what());
    } catch (const std::runtime_error& error) {
        throw std::runtime_error("keypoint GMC models '" + dpoint_path + "', '" + matcher_path + "': " +
                                 error.what());
    }
}

OnnxRuntimeKeypointGmc::~OnnxRuntimeKeypointGmc() = default;
OnnxRuntimeKeypointGmc::OnnxRuntimeKeypointGmc(OnnxRuntimeKeypointGmc&&) noexcept = default;
OnnxRuntimeKeypointGmc& OnnxRuntimeKeypointGmc::operator=(OnnxRuntimeKeypointGmc&&) noexcept = default;

cv::Size OnnxRuntimeKeypointGmc::inputSize() const { return {impl_->width, impl_->height}; }
int OnnxRuntimeKeypointGmc::keypointCount() const { return impl_->keypoints; }
int OnnxRuntimeKeypointGmc::descriptorSize() const { return impl_->descriptor_size; }

bool OnnxRuntimeKeypointGmc::extractAndMatch(const cv::Mat& image, std::vector<cv::Point2f>& previous,
                                             std::vector<cv::Point2f>& current) {
    try {
        return impl_->run(image, previous, current);
    } catch (const Ort::Exception& error) {
        impl_->has_previous = false;
        throw std::runtime_error(std::string("ONNX Runtime keypoint GMC failed: ") + error.what());
    }
}

void OnnxRuntimeKeypointGmc::reset() { impl_->has_previous = false; }

}
