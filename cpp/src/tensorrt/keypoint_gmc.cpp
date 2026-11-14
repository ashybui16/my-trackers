#include "mot/tensorrt/keypoint_gmc.hpp"

#include "tensorrt/trt_common.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mot {
namespace {

using detail::checkCuda;
using detail::CudaBuffer;
using detail::CudaStream;
using detail::dimsString;
using detail::HostFloatBuffer;
using detail::TensorRTEngine;

constexpr std::array<const char*, 3> kDpointOutputs{"keypoints", "descriptors", "scores"};
constexpr std::array<const char*, 4> kMatcherInputs{"kpts1", "descs1", "kpts2", "descs2"};
constexpr std::array<const char*, 3> kMatcherOutputs{"matched_kpts1", "matched_kpts2", "valid_mask"};

void checkBindings(const nvinfer1::ICudaEngine& engine, int expected, const std::string& what) {
    if (engine.hasImplicitBatchDimension() || engine.getNbBindings() != expected) {
        throw std::runtime_error(what + " engine must have explicit batch and exactly " +
                                 std::to_string(expected) + " bindings");
    }
    for (int index = 0; index < expected; ++index) {
        const char* raw_name = engine.getBindingName(index);
        const std::string name = raw_name == nullptr ? "<unnamed>" : raw_name;
        if (engine.getBindingDataType(index) != nvinfer1::DataType::kFLOAT ||
            engine.getBindingFormat(index) != nvinfer1::TensorFormat::kLINEAR ||
            engine.getLocation(index) != nvinfer1::TensorLocation::kDEVICE) {
            throw std::runtime_error(what + " binding '" + name + "' must use linear FP32 device storage");
        }
    }
}

int bindingIndex(const nvinfer1::ICudaEngine& engine, const char* name, bool input, const std::string& what) {
    const int index = engine.getBindingIndex(name);
    if (index < 0 || engine.bindingIsInput(index) != input) {
        throw std::runtime_error(what + " engine is missing " + (input ? "input" : "output") + " binding '" +
                                 name + "'");
    }
    return index;
}

void expectDims(const nvinfer1::Dims& dims, const std::vector<int>& expected, const std::string& what) {
    bool same = dims.nbDims == static_cast<int>(expected.size());
    for (int index = 0; same && index < dims.nbDims; ++index) same = dims.d[index] == expected[index];
    if (!same) {
        nvinfer1::Dims wanted{};
        wanted.nbDims = static_cast<int>(expected.size());
        for (int index = 0; index < wanted.nbDims; ++index) wanted.d[index] = expected[index];
        throw std::runtime_error(what + " must have fixed shape " + dimsString(wanted) + ", got " + dimsString(dims));
    }
}

}

struct TensorRTKeypointGmc::Impl {
    TensorRTEngine dpoint;
    TensorRTEngine matcher;

    int width{0};
    int height{0};
    int keypoints{0};
    int descriptor_size{0};

    // Host buffers are kept alive until the CUDA stream completes.
    HostFloatBuffer host_image;
    HostFloatBuffer host_matched_previous;
    HostFloatBuffer host_matched_current;
    HostFloatBuffer host_valid;

    // Release GPU resources before host buffers. Keypoints and descriptors stay on the device in two
    // slots (current and previous frame), so only the image goes up and only the matches come down.
    CudaStream stream;
    CudaBuffer device_image;
    std::array<CudaBuffer, 2> device_kpts;
    std::array<CudaBuffer, 2> device_descs;
    CudaBuffer device_scores;
    CudaBuffer device_matched_previous;
    CudaBuffer device_matched_current;
    CudaBuffer device_valid;

    std::array<std::vector<void*>, 2> dpoint_bindings;    // by slot written
    std::array<std::vector<void*>, 2> matcher_bindings;   // by current slot

    int current_slot{0};
    bool has_previous{false};

    void initialize() {
        checkBindings(*dpoint.engine, 4, "dpoint");
        checkBindings(*matcher.engine, 7, "matcher");

        const nvinfer1::ICudaEngine& de = *dpoint.engine;
        int image_index = -1;
        for (int index = 0; index < de.getNbBindings(); ++index) {
            if (de.bindingIsInput(index)) image_index = index;
        }
        if (image_index < 0) throw std::runtime_error("dpoint engine has no input binding");
        const auto image_dims = de.getBindingDimensions(image_index);
        if (image_dims.nbDims != 4 || image_dims.d[0] != 1 || image_dims.d[1] != 1 || image_dims.d[2] <= 0 ||
            image_dims.d[3] <= 0) {
            throw std::runtime_error("dpoint input must have fixed shape [1,1,H,W], got " + dimsString(image_dims));
        }
        height = image_dims.d[2];
        width = image_dims.d[3];

        std::array<int, 3> dpoint_out{};
        for (std::size_t e = 0; e < kDpointOutputs.size(); ++e) {
            dpoint_out[e] = bindingIndex(de, kDpointOutputs[e], false, "dpoint");
        }
        const auto kpts_dims = de.getBindingDimensions(dpoint_out[0]);
        if (kpts_dims.nbDims != 3 || kpts_dims.d[0] != 1 || kpts_dims.d[1] <= 0 || kpts_dims.d[2] != 2) {
            throw std::runtime_error("dpoint output 'keypoints' must have fixed shape [1,N,2], got " +
                                     dimsString(kpts_dims));
        }
        const int n = kpts_dims.d[1];
        const auto descs_dims = de.getBindingDimensions(dpoint_out[1]);
        if (descs_dims.nbDims != 3 || descs_dims.d[0] != 1 || descs_dims.d[1] != n || descs_dims.d[2] <= 0) {
            throw std::runtime_error("dpoint output 'descriptors' must have fixed shape [1," + std::to_string(n) +
                                     ",D], got " + dimsString(descs_dims));
        }
        const int d = descs_dims.d[2];
        expectDims(de.getBindingDimensions(dpoint_out[2]), {1, n}, "dpoint output 'scores'");

        const nvinfer1::ICudaEngine& me = *matcher.engine;
        std::array<int, 4> matcher_in{};
        for (std::size_t e = 0; e < kMatcherInputs.size(); ++e) {
            matcher_in[e] = bindingIndex(me, kMatcherInputs[e], true, "matcher");
            expectDims(me.getBindingDimensions(matcher_in[e]), {1, n, e % 2 == 0 ? 2 : d},
                       std::string("matcher input '") + kMatcherInputs[e] + "'");
        }
        std::array<int, 3> matcher_out{};
        for (std::size_t e = 0; e < kMatcherOutputs.size(); ++e) {
            matcher_out[e] = bindingIndex(me, kMatcherOutputs[e], false, "matcher");
            expectDims(me.getBindingDimensions(matcher_out[e]), e < 2 ? std::vector<int>{1, n, 2} : std::vector<int>{1, n},
                       std::string("matcher output '") + kMatcherOutputs[e] + "'");
        }
        keypoints = n;
        descriptor_size = d;

        const auto un = static_cast<std::size_t>(n);
        const std::size_t image_elements = static_cast<std::size_t>(width) * height;
        host_image.allocate(image_elements);
        host_matched_previous.allocate(un * 2);
        host_matched_current.allocate(un * 2);
        host_valid.allocate(un);
        device_image.allocate(image_elements * sizeof(float), "dpoint image");
        for (int slot = 0; slot < 2; ++slot) {
            device_kpts[slot].allocate(un * 2 * sizeof(float), "keypoints");
            device_descs[slot].allocate(un * static_cast<std::size_t>(d) * sizeof(float), "descriptors");
        }
        device_scores.allocate(un * sizeof(float), "scores");
        device_matched_previous.allocate(un * 2 * sizeof(float), "matched_kpts1");
        device_matched_current.allocate(un * 2 * sizeof(float), "matched_kpts2");
        device_valid.allocate(un * sizeof(float), "valid_mask");
        stream.create();

        for (int slot = 0; slot < 2; ++slot) {
            const int other = 1 - slot;
            auto& db = dpoint_bindings[slot];
            db.assign(4, nullptr);
            db[static_cast<std::size_t>(image_index)] = device_image.data();
            db[static_cast<std::size_t>(dpoint_out[0])] = device_kpts[slot].data();
            db[static_cast<std::size_t>(dpoint_out[1])] = device_descs[slot].data();
            db[static_cast<std::size_t>(dpoint_out[2])] = device_scores.data();
            auto& mb = matcher_bindings[slot];
            mb.assign(7, nullptr);
            mb[static_cast<std::size_t>(matcher_in[0])] = device_kpts[other].data();
            mb[static_cast<std::size_t>(matcher_in[1])] = device_descs[other].data();
            mb[static_cast<std::size_t>(matcher_in[2])] = device_kpts[slot].data();
            mb[static_cast<std::size_t>(matcher_in[3])] = device_descs[slot].data();
            mb[static_cast<std::size_t>(matcher_out[0])] = device_matched_previous.data();
            mb[static_cast<std::size_t>(matcher_out[1])] = device_matched_current.data();
            mb[static_cast<std::size_t>(matcher_out[2])] = device_valid.data();
        }
    }

    void download(HostFloatBuffer& host, const CudaBuffer& device, const char* label) {
        checkCuda(cudaMemcpyAsync(host.data(), device.data(), host.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                  stream.get()),
                  std::string("cudaMemcpyAsync ") + label);
    }

    bool run(const cv::Mat& input, std::vector<cv::Point2f>& previous, std::vector<cv::Point2f>& current) {
        previous.clear();
        current.clear();
        if (input.type() != CV_32FC1 || input.cols != width || input.rows != height) {
            throw std::invalid_argument("keypoint GMC input must be CV_32FC1 of " + std::to_string(width) + "x" +
                                        std::to_string(height));
        }
        for (int row = 0; row < height; ++row) {
            std::memcpy(host_image.data() + static_cast<std::size_t>(row) * width, input.ptr<float>(row),
                        static_cast<std::size_t>(width) * sizeof(float));
        }
        checkCuda(cudaMemcpyAsync(device_image.data(), host_image.data(), host_image.size() * sizeof(float),
                                  cudaMemcpyHostToDevice, stream.get()),
                  "cudaMemcpyAsync dpoint image");
        if (!dpoint.context->enqueueV2(dpoint_bindings[current_slot].data(), stream.get(), nullptr)) {
            throw std::runtime_error("TensorRT enqueue failed for the dpoint engine");
        }
        const bool matched = has_previous;
        if (matched) {
            if (!matcher.context->enqueueV2(matcher_bindings[current_slot].data(), stream.get(), nullptr)) {
                throw std::runtime_error("TensorRT enqueue failed for the matcher engine");
            }
            download(host_matched_previous, device_matched_previous, "matched_kpts1");
            download(host_matched_current, device_matched_current, "matched_kpts2");
            download(host_valid, device_valid, "valid_mask");
        }
        stream.synchronize();
        if (matched) {
            const float* mp = host_matched_previous.data();
            const float* mc = host_matched_current.data();
            const float* valid = host_valid.data();
            for (std::size_t index = 0; index < host_valid.size(); ++index) {
                if (valid[index] > 0.5F) {
                    previous.emplace_back(mp[2 * index], mp[2 * index + 1]);
                    current.emplace_back(mc[2 * index], mc[2 * index + 1]);
                }
            }
        }
        has_previous = true;
        current_slot = 1 - current_slot;
        return matched;
    }
};

TensorRTKeypointGmc::TensorRTKeypointGmc(const std::string& dpoint_engine_path,
                                         const std::string& matcher_engine_path)
    : impl_(std::make_unique<Impl>()) {
    try {
        impl_->dpoint.load(dpoint_engine_path);
        impl_->matcher.load(matcher_engine_path);
        impl_->initialize();
    } catch (const std::runtime_error& error) {
        throw std::runtime_error("keypoint GMC engines '" + dpoint_engine_path + "', '" + matcher_engine_path +
                                 "': " + error.what());
    }
}

TensorRTKeypointGmc::~TensorRTKeypointGmc() = default;
TensorRTKeypointGmc::TensorRTKeypointGmc(TensorRTKeypointGmc&&) noexcept = default;
TensorRTKeypointGmc& TensorRTKeypointGmc::operator=(TensorRTKeypointGmc&&) noexcept = default;

cv::Size TensorRTKeypointGmc::inputSize() const { return {impl_->width, impl_->height}; }
int TensorRTKeypointGmc::keypointCount() const { return impl_->keypoints; }
int TensorRTKeypointGmc::descriptorSize() const { return impl_->descriptor_size; }

bool TensorRTKeypointGmc::extractAndMatch(const cv::Mat& image, std::vector<cv::Point2f>& previous,
                                          std::vector<cv::Point2f>& current) {
    try {
        return impl_->run(image, previous, current);
    } catch (...) {
        impl_->has_previous = false;
        throw;
    }
}

void TensorRTKeypointGmc::reset() { impl_->has_previous = false; }

}
