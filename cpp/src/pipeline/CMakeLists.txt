# Backend-agnostic detect → track pipeline and the YAML configuration loader.
mot_add_library(mot_pipeline ALIAS mot::pipeline
    SOURCES pipeline.cpp config.cpp
    PUBLIC_DEPS mot_core mot_tracking)
