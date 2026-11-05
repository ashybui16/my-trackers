# Finds the ONNX Runtime C/C++ SDK (official prebuilt release tarball) and defines
# the imported target OnnxRuntime::onnxruntime.
# Hints: ONNXRUNTIME_ROOT (CMake or environment variable), then /opt/onnxruntime-linux-x64-*.

set(_onnxruntime_hints)
foreach(_root IN ITEMS "${ONNXRUNTIME_ROOT}" "$ENV{ONNXRUNTIME_ROOT}")
    if(_root)
        list(APPEND _onnxruntime_hints "${_root}")
    endif()
endforeach()
file(GLOB _onnxruntime_opt_roots LIST_DIRECTORIES true "/opt/onnxruntime-linux-x64-*")
list(SORT _onnxruntime_opt_roots COMPARE NATURAL ORDER DESCENDING)
list(APPEND _onnxruntime_hints ${_onnxruntime_opt_roots})

find_path(OnnxRuntime_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${_onnxruntime_hints} PATH_SUFFIXES include include/onnxruntime)
find_library(OnnxRuntime_LIBRARY onnxruntime
    HINTS ${_onnxruntime_hints} PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntime
    REQUIRED_VARS OnnxRuntime_LIBRARY OnnxRuntime_INCLUDE_DIR)

if(OnnxRuntime_FOUND AND NOT TARGET OnnxRuntime::onnxruntime)
    add_library(OnnxRuntime::onnxruntime UNKNOWN IMPORTED)
    set_target_properties(OnnxRuntime::onnxruntime PROPERTIES
        IMPORTED_LOCATION "${OnnxRuntime_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntime_INCLUDE_DIR}")
endif()
mark_as_advanced(OnnxRuntime_INCLUDE_DIR OnnxRuntime_LIBRARY)
