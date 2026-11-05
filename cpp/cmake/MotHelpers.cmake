# Shared helpers for the mot C++ project.

function(mot_set_warnings target)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic ${ARGN})
endfunction()

# mot_add_library(<name> ALIAS <alias> SOURCES ... [PUBLIC_DEPS ...] [PRIVATE_DEPS ...])
# Static library with public headers in include/ and private headers in src/.
function(mot_add_library name)
    cmake_parse_arguments(ARG "" "ALIAS" "SOURCES;PUBLIC_DEPS;PRIVATE_DEPS" ${ARGN})
    add_library(${name} STATIC ${ARG_SOURCES})
    if(ARG_ALIAS)
        add_library(${ARG_ALIAS} ALIAS ${name})
    endif()
    target_include_directories(${name}
        PUBLIC
            $<BUILD_INTERFACE:${MOT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
            ${MOT_SOURCE_DIR}/src)
    target_compile_features(${name} PUBLIC cxx_std_17)
    target_link_libraries(${name} PUBLIC ${ARG_PUBLIC_DEPS} PRIVATE ${ARG_PRIVATE_DEPS})
    mot_set_warnings(${name})
    set_property(GLOBAL APPEND PROPERTY MOT_EXPORTED_TARGETS ${name})
endfunction()

# mot_add_test(<name> SOURCES ... LIBS ... LABELS ... [ARGS ...] [SKIP_RETURN_CODE n])
# In-tree tests may include private headers from src/ and helpers from tests/.
function(mot_add_test name)
    cmake_parse_arguments(ARG "" "SKIP_RETURN_CODE" "SOURCES;LIBS;LABELS;ARGS" ${ARGN})
    add_executable(${name}_test ${ARG_SOURCES})
    target_include_directories(${name}_test PRIVATE ${MOT_SOURCE_DIR}/src ${MOT_SOURCE_DIR}/tests)
    target_link_libraries(${name}_test PRIVATE ${ARG_LIBS})
    mot_set_warnings(${name}_test)
    add_test(NAME mot_${name} COMMAND ${name}_test ${ARG_ARGS})
    set_tests_properties(mot_${name} PROPERTIES LABELS "${ARG_LABELS}")
    if(DEFINED ARG_SKIP_RETURN_CODE)
        set_tests_properties(mot_${name} PROPERTIES SKIP_RETURN_CODE ${ARG_SKIP_RETURN_CODE})
    endif()
endfunction()
