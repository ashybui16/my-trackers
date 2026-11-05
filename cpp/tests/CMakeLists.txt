set(MOT_REPO_WEIGHTS "${MOT_SOURCE_DIR}/../weights")

mot_add_test(botsort
    SOURCES tracking/test_botsort.cpp
    LIBS mot_tracking opencv_video
    LABELS "tracking;cpu")
mot_add_test(tracker_contract
    SOURCES tracking/test_tracker_contract.cpp
    LIBS mot_tracking
    LABELS "tracking;cpu")
mot_add_test(pipeline
    SOURCES pipeline/test_pipeline.cpp
    LIBS mot_pipeline
    LABELS "pipeline;cpu")
mot_add_test(config
    SOURCES pipeline/test_config.cpp
    LIBS mot_pipeline
    ARGS "${MOT_REPO_WEIGHTS}/config.yaml"
    LABELS "config;cpu")
mot_add_test(annotator
    SOURCES visualization/test_annotator.cpp
    LIBS mot_visualization mot_pipeline
    ARGS "${MOT_REPO_WEIGHTS}/config.yaml"
    LABELS "visualization;cpu")
mot_add_test(postprocess
    SOURCES tensorrt/test_postprocess.cpp
    LIBS mot_tensorrt_postprocess
    LABELS "tensorrt;cpu")
mot_add_test(performance_regressions
    SOURCES regression/test_performance_regressions.cpp
    LIBS mot_tracking mot_tensorrt_postprocess opencv_video opencv_imgproc
    LABELS "regression;cpu")

if(MOT_BUILD_GPU_TESTS AND MOT_WITH_TENSORRT)
    mot_add_test(inference_gpu
        SOURCES tensorrt/test_inference_gpu.cpp
        LIBS mot_tensorrt mot_pipeline CUDA::cudart opencv_videoio
        LABELS "gpu"
        SKIP_RETURN_CODE 77)
endif()
