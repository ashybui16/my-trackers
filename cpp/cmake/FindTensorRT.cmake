# Finds TensorRT and defines the imported target TensorRT::nvinfer.
# Hints: TENSORRT_ROOT (CMake or environment variable).

set(_tensorrt_hints)
foreach(_root IN ITEMS "${TENSORRT_ROOT}" "$ENV{TENSORRT_ROOT}" "/opt/TensorRT-8.6.1.6")
    if(_root)
        list(APPEND _tensorrt_hints "${_root}")
    endif()
endforeach()

find_path(TensorRT_INCLUDE_DIR NvInfer.h
    HINTS ${_tensorrt_hints} PATH_SUFFIXES include)
find_library(TensorRT_nvinfer_LIBRARY nvinfer
    HINTS ${_tensorrt_hints} PATH_SUFFIXES lib targets/x86_64-linux-gnu/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS TensorRT_nvinfer_LIBRARY TensorRT_INCLUDE_DIR)

if(TensorRT_FOUND AND NOT TARGET TensorRT::nvinfer)
    add_library(TensorRT::nvinfer UNKNOWN IMPORTED)
    set_target_properties(TensorRT::nvinfer PROPERTIES
        IMPORTED_LOCATION "${TensorRT_nvinfer_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TensorRT_INCLUDE_DIR}")
endif()
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_nvinfer_LIBRARY)
