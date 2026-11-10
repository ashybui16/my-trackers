mot_add_library(mot_tracking ALIAS mot::tracking
    SOURCES
        factory.cpp
        common/assignment.cpp
        common/association.cpp
        common/geometry.cpp
        common/gmc.cpp
        common/kalman_filter.cpp
        common/track.cpp
        botsort/botsort.cpp
        botsort/botsort_config.cpp
        tracktrack/tracktrack.cpp
        tracktrack/tracktrack_config.cpp
        tracktrack/tracktrack_detail.cpp
        tracktrack/tt_track.cpp
    PUBLIC_DEPS mot_core
    PRIVATE_DEPS opencv_video opencv_calib3d opencv_features2d opencv_imgproc)
