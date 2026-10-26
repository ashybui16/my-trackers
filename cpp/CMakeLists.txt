cmake_minimum_required(VERSION 3.18)

project(mot_tracker_cpp VERSION 1.0.0 LANGUAGES CXX)

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
    src/assignment.cpp
    src/association.cpp
    src/config.cpp
    src/geometry.cpp
    src/gmc.cpp
    src/kalman_filter.cpp
    src/track.cpp
    src/tracker.cpp
    src/types.cpp
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

if(MOT_BUILD_INFERENCE)
    add_library(mot_inference_cpp STATIC src/detector.cpp src/pipeline.cpp)
    add_library(mot::inference ALIAS mot_inference_cpp)
    target_include_directories(mot_inference_cpp
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            "${TENSORRT_INCLUDE_DIR}"
    )
    target_link_libraries(mot_inference_cpp
        PUBLIC mot_tracker_cpp ${OpenCV_LIBS}
        PRIVATE "${TENSORRT_NVINFER_LIBRARY}" CUDA::cudart
    )
    target_compile_options(mot_inference_cpp PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations -Wno-unused-parameter)

    add_executable(mot_video src/mot_video.cpp)
    target_link_libraries(mot_video PRIVATE mot_inference_cpp)
    target_compile_options(mot_video PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(MOT_BUILD_TESTS)
    add_executable(mot_tracker_tests tests/test_mot_tracker.cpp)
    target_link_libraries(mot_tracker_tests PRIVATE mot_tracker_cpp)
    target_compile_options(mot_tracker_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_tracker_core COMMAND mot_tracker_tests)
    set_tests_properties(mot_tracker_core PROPERTIES LABELS "core;cpu")
endif()

if(MOT_BUILD_GPU_TESTS AND MOT_BUILD_INFERENCE)
    add_executable(mot_inference_gpu_tests tests/test_inference_gpu.cpp)
    target_link_libraries(mot_inference_gpu_tests PRIVATE mot_inference_cpp CUDA::cudart)
    target_compile_options(mot_inference_gpu_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME mot_inference_gpu COMMAND mot_inference_gpu_tests)
    set_tests_properties(mot_inference_gpu PROPERTIES
        LABELS "gpu"
        SKIP_RETURN_CODE 77
    )
endif()

set(MOT_INSTALL_TARGETS mot_tracker_cpp)
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
