#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mot::detail {

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

inline void checkCuda(cudaError_t status, const std::string& operation) {
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

inline std::string dimsString(const nvinfer1::Dims& dims) {
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

inline std::vector<char> readSerializedEngine(const std::string& engine_path) {
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
    return serialized;
}

// Deserialized engine plus one execution context; members are destroyed context first.
struct TensorRTEngine {
    TensorRTLogger logger;
    TensorRTPtr<nvinfer1::IRuntime> runtime{nullptr};
    TensorRTPtr<nvinfer1::ICudaEngine> engine{nullptr};
    TensorRTPtr<nvinfer1::IExecutionContext> context{nullptr};

    void load(const std::string& engine_path) {
        const std::vector<char> serialized = readSerializedEngine(engine_path);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("TensorRT runtime creation failed; check TensorRT/CUDA installation");
        }
        engine.reset(runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
        if (!engine) {
            throw std::runtime_error("TensorRT engine deserialization failed for '" + engine_path +
                                     "'; serialized engines are tied to the target GPU and TensorRT "
                                     "version, so rebuild the engine");
        }
        context.reset(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("TensorRT execution context creation failed");
        }
    }
};

}
