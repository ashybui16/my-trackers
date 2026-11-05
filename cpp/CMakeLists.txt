cmake_minimum_required(VERSION 3.18)

project(mot VERSION 1.2.0 LANGUAGES CXX)

# Honor <PACKAGE>_ROOT hints such as TENSORRT_ROOT and ONNXRUNTIME_ROOT.
if(POLICY CMP0144)
    cmake_policy(SET CMP0144 NEW)
endif()

set(MOT_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MOT_SOURCE_DIR}/cmake")

# Apps, tests and benchmarks default ON only when this repository is being built
# directly; an application that uses add_subdirectory(cpp) gets just the libraries.
get_filename_component(_mot_repo_root "${MOT_SOURCE_DIR}/.." ABSOLUTE)
if(CMAKE_SOURCE_DIR STREQUAL MOT_SOURCE_DIR OR CMAKE_SOURCE_DIR STREQUAL _mot_repo_root)
    set(MOT_IS_TOP_LEVEL ON)
else()
    set(MOT_IS_TOP_LEVEL OFF)
endif()

option(MOT_WITH_TENSORRT "Build the TensorRT detector (needs CUDA + TensorRT)" ON)
option(MOT_WITH_ONNXRUNTIME "Build the ONNX Runtime (CPU) detector" ON)
option(MOT_BUILD_APPS "Build the mot_video CLI" ${MOT_IS_TOP_LEVEL})
option(MOT_BUILD_TESTS "Build CPU tests" ${MOT_IS_TOP_LEVEL})
option(MOT_BUILD_GPU_TESTS "Build tests that need a TensorRT engine and an NVIDIA GPU" OFF)
option(MOT_BUILD_BENCHMARKS "Build CPU/video performance benchmarks" OFF)
option(MOT_INSTALL "Generate install and package-config rules" ${MOT_IS_TOP_LEVEL})

if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(MOT_OPENCV_COMPONENTS core imgproc video videoio calib3d features2d)
find_package(OpenCV 4 REQUIRED COMPONENTS ${MOT_OPENCV_COMPONENTS})

if(MOT_WITH_TENSORRT)
    find_package(CUDAToolkit QUIET)
    find_package(TensorRT QUIET)
    if(NOT CUDAToolkit_FOUND OR NOT TensorRT_FOUND)
        message(WARNING
            "CUDA/TensorRT was not found; disabling MOT_WITH_TENSORRT. "
            "Set TENSORRT_ROOT and CUDAToolkit_ROOT, or pass -DMOT_WITH_TENSORRT=OFF.")
        set(MOT_WITH_TENSORRT OFF)
    endif()
endif()

if(MOT_WITH_ONNXRUNTIME)
    find_package(OnnxRuntime QUIET)
    if(NOT OnnxRuntime_FOUND)
        message(WARNING
            "ONNX Runtime was not found; disabling MOT_WITH_ONNXRUNTIME. "
            "Set ONNXRUNTIME_ROOT to an extracted onnxruntime-linux-x64-<version> release, "
            "or pass -DMOT_WITH_ONNXRUNTIME=OFF.")
        set(MOT_WITH_ONNXRUNTIME OFF)
    endif()
endif()

include(GNUInstallDirs)
include(MotHelpers)

add_subdirectory(src/core)
add_subdirectory(src/tracking)
add_subdirectory(src/pipeline)
add_subdirectory(src/visualization)
add_subdirectory(src/detection)
add_subdirectory(src/tensorrt)
add_subdirectory(src/onnxruntime)

add_library(mot INTERFACE)
add_library(mot::mot ALIAS mot)
target_link_libraries(mot INTERFACE mot_core mot_tracking mot_pipeline mot_visualization)
if(MOT_WITH_TENSORRT)
    target_link_libraries(mot INTERFACE mot_tensorrt)
endif()
if(MOT_WITH_ONNXRUNTIME)
    target_link_libraries(mot INTERFACE mot_onnxruntime)
endif()
set_property(GLOBAL APPEND PROPERTY MOT_EXPORTED_TARGETS mot)

if(MOT_BUILD_APPS AND (MOT_WITH_TENSORRT OR MOT_WITH_ONNXRUNTIME))
    add_subdirectory(apps/mot_video)
endif()
if(MOT_BUILD_TESTS)
    include(CTest)
    add_subdirectory(tests)
endif()
if(MOT_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(MOT_INSTALL)
    include(CMakePackageConfigHelpers)
    get_property(_mot_targets GLOBAL PROPERTY MOT_EXPORTED_TARGETS)
    install(TARGETS ${_mot_targets}
            EXPORT motTargets
            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
            LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    set(_mot_header_excludes)
    if(NOT MOT_WITH_TENSORRT)
        list(APPEND _mot_header_excludes PATTERN tensorrt EXCLUDE)
    endif()
    if(NOT MOT_WITH_ONNXRUNTIME)
        list(APPEND _mot_header_excludes PATTERN onnxruntime EXCLUDE)
    endif()
    install(DIRECTORY include/mot DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} ${_mot_header_excludes})
    set(_mot_cmake_dir ${CMAKE_INSTALL_LIBDIR}/cmake/mot)
    install(EXPORT motTargets NAMESPACE mot:: DESTINATION ${_mot_cmake_dir})
    configure_package_config_file(cmake/motConfig.cmake.in
        "${CMAKE_CURRENT_BINARY_DIR}/motConfig.cmake"
        INSTALL_DESTINATION ${_mot_cmake_dir})
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/motConfigVersion.cmake"
        COMPATIBILITY SameMinorVersion)
    install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/motConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/motConfigVersion.cmake"
        cmake/FindTensorRT.cmake
        cmake/FindOnnxRuntime.cmake
        DESTINATION ${_mot_cmake_dir})
    if(TARGET mot_video)
        install(TARGETS mot_video RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
endif()
