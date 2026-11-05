mot_add_library(mot_visualization ALIAS mot::visualization
    SOURCES annotator.cpp
    PUBLIC_DEPS mot_core
    PRIVATE_DEPS opencv_imgproc)
