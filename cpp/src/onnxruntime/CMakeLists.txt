if(MOT_WITH_ONNXRUNTIME)
    mot_add_library(mot_onnxruntime ALIAS mot::onnxruntime
        SOURCES detector.cpp keypoint_gmc.cpp $<TARGET_OBJECTS:mot_yolo_postprocess>
        PUBLIC_DEPS mot_core
        PRIVATE_DEPS opencv_imgproc OnnxRuntime::onnxruntime)
endif()
