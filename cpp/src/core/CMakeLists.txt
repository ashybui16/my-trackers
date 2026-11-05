# Shared data types and detector parameters used by every other module.
mot_add_library(mot_core ALIAS mot::core
    SOURCES types.cpp detector_config.cpp
    PUBLIC_DEPS opencv_core)
