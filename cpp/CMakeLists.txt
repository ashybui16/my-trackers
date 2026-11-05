cmake_minimum_required(VERSION 3.18)

project(mot_tracker_cpp VERSION 1.0.0 LANGUAGES CXX)

option(MOT_BUILD_BENCHMARKS "Build optional CPU/video performance benchmarks" OFF)

option(MOT_BUILD_TESTS "Build tracker CTest tests" ON)
option(MOT_BUILD_GPU_TESTS "Build tests that require a TensorRT engine and NVIDIA GPU" OFF)
option(MOT_BUILD_INFERENCE "Build TensorRT detector, pipeline and video CLI" ON)
set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
if(NOT TENSORRT_ROOT AND DEFINED ENV{TENSORRT_ROOT})
    set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}")
endif()
if(NOT TENSORRT_ROOT)
    set(TENSORRT_ROOT "/opt/TensorRT-8.6.1.6")
endif()
set(CUDAToolkit_ROOT "" CACHE PATH "CUDA toolkit root")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc video videoio calib3d features2d)
if(MOT_BUILD_INFERENCE)
    find_package(CUDAToolkit QUIET)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
              HINTS "${TENSORRT_ROOT}/include" "${TENSORRT_ROOT}")
    find_library(TENSORRT_NVINFER_LIBRARY nvinfer
                 HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib")
    if(NOT CUDAToolkit_FOUND OR NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY)
        message(WARNING
            "CUDA/TensorRT was not found; disabling MOT_BUILD_INFERENCE. "
            "Set TENSORRT_ROOT and CUDAToolkit_ROOT or use -DMOT_BUILD_INFERENCE=OFF.")
        set(MOT_BUILD_INFERENCE OFF)
    endif()
endif()

include(GNUInstallDirs)
include(CTest)

set(MOT_TRACKER_SOURCES
    src/tracker/assignment.cpp
    src/tracker/association.cpp
    src/tracker/config.cpp
    src/tracker/geometry.cpp
    src/tracker/gmc.cpp
    src/tracker/kalman_filter.cpp
    src/tracker/track.cpp
    src/tracker/tracker.cpp
    src/tracker/types.cpp
)

add_library(mot_tracker_cpp STATIC ${MOT_TRACKER_SOURCES})
add_library(mot::tracker ALIAS mot_tracker_cpp)
target_include_directories(mot_tracker_cpp
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_link_libraries(mot_tracker_cpp PUBLIC ${OpenCV_LIBS})
target_compile_options(mot_tracker_cpp PRIVATE -Wall -Wextra -Wpedantic)

add_library(mot_pipeline_config_cpp STATIC src/pipeline/pipeline_config.cpp)
add_library(mot::config ALIAS mot_pipeline_config_cpp)
target_link_libraries(mot_pipeline_config_cpp PUBLIC mot_tracker_cpp)
target_compile_options(mot_pipeline_config_cpp PRIVATE -Wall -Wextra -Wpedantic)

add_library(mot_annotator_cpp STATIC src/annotator/annotation.cpp)
add_library(mot::annotator ALIAS mot_annotator_cpp)
target_link_libraries(mot_annotator_cpp PUBLIC mot_tracker_cpp)
target_compile_options(mot_annotator_cpp PRIVATE -Wall -Wextra -Wpedantic)

if(MOT_BUILD_INFERENCE)
    add_library(mot_inference_cpp STATIC
        src/detector/detector.cpp
        src/detector/detector_postprocess.cpp
        src/pipeline/pipeline.cpp
    )
    add_library(mot::inference ALIAS mot_inference_cpp)
    target_include_directories(mot_inference_cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${TENSORRT_INCLUDE_DIR}"
            "${CMAKE_CURRENT_SOURCE_DIR}/internal"
    )
    target_link_libraries(mot_inference_cpp
        PUBLIC mot_tracker_cpp mot_pipeline_config_cpp ${OpenCV_LIBS}
        PRIVATE "${TENSORRT_NVINFER_LIBRARY}" CUDA::cudart
    )
    target_compile_options(mot_inference_cpp PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-unused-parameter)

    add_executable(mot_video src/apps/mot_video.cpp)
    target_link_libraries(mot_video PRIVATE mot_inference_cpp mot_annotator_cpp)
    target_compile_options(mot_video PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MOT_BUILD_TESTS)
    add_executable(mot_performance_regression_tests tests/test_performance_regressions.cpp src/detector/detector_postprocess.cpp)
    target_include_directories(mot_performance_regression_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/internal")
    target_link_libraries(mot_performance_regression_tests PRIVATE mot_tracker_cpp)
    target_compile_options(mot_performance_regression_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_performance_regressions COMMAND mot_performance_regression_tests)
    set_tests_properties(mot_performance_regressions PROPERTIES LABELS "regression;cpu")
    add_executable(mot_config_tests tests/pipeline/test_config.cpp)
    target_link_libraries(mot_config_tests PRIVATE mot_pipeline_config_cpp)
    target_compile_options(mot_config_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_config COMMAND mot_config_tests "${CMAKE_CURRENT_SOURCE_DIR}/../weights/config.yaml")
    set_tests_properties(mot_config PROPERTIES LABELS "config;cpu")

    add_executable(mot_annotator_tests tests/annotator/test_annotation.cpp)
    target_link_libraries(mot_annotator_tests PRIVATE mot_annotator_cpp)
    target_compile_options(mot_annotator_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_annotator COMMAND mot_annotator_tests "${CMAKE_CURRENT_SOURCE_DIR}/../weights/26leaf1.engine")
    set_tests_properties(mot_annotator PROPERTIES LABELS "annotation;cpu")
    add_executable(mot_tracker_tests tests/tracker/test_mot_tracker.cpp)
    target_link_libraries(mot_tracker_tests PRIVATE mot_tracker_cpp)
    target_compile_options(mot_tracker_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_tracker_core COMMAND mot_tracker_tests)
    set_tests_properties(mot_tracker_core PROPERTIES LABELS "core;cpu")

    add_executable(mot_detector_postprocess_tests
        tests/detector/test_detector_postprocess.cpp
        src/detector/detector_postprocess.cpp
    )
    target_include_directories(mot_detector_postprocess_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/internal")
    target_link_libraries(mot_detector_postprocess_tests PRIVATE mot_tracker_cpp)
    target_compile_options(mot_detector_postprocess_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_detector_postprocess COMMAND mot_detector_postprocess_tests)
    set_tests_properties(mot_detector_postprocess PROPERTIES LABELS "inference;cpu")
endif()

if(MOT_BUILD_GPU_TESTS AND MOT_BUILD_INFERENCE)
    add_executable(mot_inference_gpu_tests tests/detector/test_inference_gpu.cpp)
    target_link_libraries(mot_inference_gpu_tests PRIVATE mot_inference_cpp CUDA::cudart)
    target_compile_options(mot_inference_gpu_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_inference_gpu COMMAND mot_inference_gpu_tests)
    set_tests_properties(mot_inference_gpu PROPERTIES
        LABELS "gpu"
        SKIP_RETURN_CODE 77
    )
endif()

if(MOT_BUILD_BENCHMARKS)
    add_executable(mot_benchmark_cpu benchmarks/mot_benchmark.cpp src/detector/detector_postprocess.cpp)
    target_include_directories(mot_benchmark_cpu PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/internal")
    target_link_libraries(mot_benchmark_cpu PRIVATE mot_pipeline_config_cpp mot_annotator_cpp)
    target_compile_options(mot_benchmark_cpu PRIVATE -Wall -Wextra -Wpedantic)
    if(MOT_BUILD_INFERENCE)
        add_executable(mot_benchmark_video benchmarks/mot_benchmark.cpp)
        target_include_directories(mot_benchmark_video PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/internal")
        target_compile_definitions(mot_benchmark_video PRIVATE MOT_BENCHMARK_GPU)
        target_link_libraries(mot_benchmark_video PRIVATE mot_inference_cpp mot_annotator_cpp)
        target_compile_options(mot_benchmark_video PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()

set(MOT_INSTALL_TARGETS mot_tracker_cpp mot_pipeline_config_cpp mot_annotator_cpp)
if(MOT_BUILD_INFERENCE)
    list(APPEND MOT_INSTALL_TARGETS mot_inference_cpp mot_video)
endif()
install(TARGETS ${MOT_INSTALL_TARGETS}
        EXPORT mot_tracker_targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT mot_tracker_targets
        FILE mot_tracker_targets.cmake
        NAMESPACE mot::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/mot_tracker_cpp)
