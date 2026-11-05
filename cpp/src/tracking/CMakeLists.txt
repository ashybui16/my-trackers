# Every tracker lives in its own folder and is reachable through createTracker().
# Adding a tracker: see README.md in this folder.
mot_add_library(mot_tracking ALIAS mot::tracking
    SOURCES
        factory.cpp
        # Building blocks shared by SORT-family trackers.
        common/assignment.cpp
        common/association.cpp
        common/geometry.cpp
        common/gmc.cpp
        common/kalman_filter.cpp
        common/track.cpp
        # Trackers.
        botsort/botsort.cpp
        botsort/botsort_config.cpp
    PUBLIC_DEPS mot_core
    PRIVATE_DEPS opencv_video opencv_calib3d opencv_features2d opencv_imgproc)
