if(MOT_WITH_TENSORRT)
    mot_add_library(mot_tensorrt ALIAS mot::tensorrt
        SOURCES detector.cpp $<TARGET_OBJECTS:mot_yolo_postprocess>
        PUBLIC_DEPS mot_core
        PRIVATE_DEPS opencv_imgproc TensorRT::nvinfer CUDA::cudart)
    target_compile_options(mot_tensorrt PRIVATE -Wno-deprecated-declarations -Wno-unused-parameter)
endif()
