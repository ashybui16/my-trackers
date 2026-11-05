# Output decoding/NMS is CPU-only, so tests and benchmarks can use it without CUDA.
# OBJECT library: compiled once, folded into mot_tensorrt, never installed by itself.
if(NOT (MOT_WITH_TENSORRT OR MOT_BUILD_TESTS OR MOT_BUILD_BENCHMARKS))
    return()
endif()
add_library(mot_tensorrt_postprocess OBJECT postprocess.cpp)
target_include_directories(mot_tensorrt_postprocess PUBLIC $<BUILD_INTERFACE:${MOT_SOURCE_DIR}/src>)
target_link_libraries(mot_tensorrt_postprocess PUBLIC mot_core)
mot_set_warnings(mot_tensorrt_postprocess)

if(MOT_WITH_TENSORRT)
    mot_add_library(mot_tensorrt ALIAS mot::tensorrt
        SOURCES detector.cpp $<TARGET_OBJECTS:mot_tensorrt_postprocess>
        PUBLIC_DEPS mot_core
        PRIVATE_DEPS opencv_imgproc TensorRT::nvinfer CUDA::cudart)
    target_compile_options(mot_tensorrt PRIVATE -Wno-deprecated-declarations -Wno-unused-parameter)
endif()
