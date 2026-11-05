add_executable(mot_benchmark_cpu mot_benchmark.cpp)
target_include_directories(mot_benchmark_cpu PRIVATE ${MOT_SOURCE_DIR}/src)
target_link_libraries(mot_benchmark_cpu PRIVATE
    mot_pipeline mot_visualization mot_tensorrt_postprocess opencv_video opencv_videoio)
mot_set_warnings(mot_benchmark_cpu)

if(MOT_WITH_TENSORRT)
    add_executable(mot_benchmark_video mot_benchmark.cpp)
    target_compile_definitions(mot_benchmark_video PRIVATE MOT_BENCHMARK_GPU)
    target_link_libraries(mot_benchmark_video PRIVATE
        mot_pipeline mot_tensorrt mot_visualization opencv_videoio)
    mot_set_warnings(mot_benchmark_video)
endif()
