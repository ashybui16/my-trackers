# OBJECT library: compiled once, folded into each backend library, never installed by itself.
if(NOT (MOT_WITH_TENSORRT OR MOT_WITH_ONNXRUNTIME OR MOT_BUILD_TESTS OR MOT_BUILD_BENCHMARKS))
    return()
endif()
add_library(mot_yolo_postprocess OBJECT yolo/postprocess.cpp)
target_include_directories(mot_yolo_postprocess PUBLIC $<BUILD_INTERFACE:${MOT_SOURCE_DIR}/src>)
target_link_libraries(mot_yolo_postprocess PUBLIC mot_core opencv_imgproc)
mot_set_warnings(mot_yolo_postprocess)
