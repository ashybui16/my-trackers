add_executable(mot_video main.cpp)
target_link_libraries(mot_video PRIVATE mot::pipeline mot::tensorrt mot::visualization opencv_videoio)
mot_set_warnings(mot_video)
