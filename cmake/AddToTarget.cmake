function(set_simsycl_target_options TARGET)
    # Standard must be at least 20, allow higher if set by the user
    get_target_property(STANDARD_BEFORE "${TARGET}" CXX_STANDARD)
    if((NOT STANDARD_BEFORE) OR (STANDARD_BEFORE LESS 20))
        set_target_properties("${TARGET}" PROPERTIES CXX_STANDARD 20)
    endif()
    set_target_properties("${TARGET}" PROPERTIES CXX_STANDARD_REQUIRED ON)

    target_compile_options("${TARGET}" PRIVATE
        # 4180 is a false positive in MSVC in group_operation_impl -- "fixing" it actually breaks the code
        $<$<CXX_COMPILER_ID:MSVC>:/W4 /wd4180>
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:$<$<BOOL:${SIMSYCL_ENABLE_ASAN}>:-fsanitize=address>>
    )
    target_link_options("${TARGET}" PRIVATE
        $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:$<$<BOOL:${SIMSYCL_ENABLE_ASAN}>:-fsanitize=address>>
    )

    if(ARGV1 STREQUAL "ALL_WARNINGS")
        target_compile_options("${TARGET}" PRIVATE
            $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -Wpedantic>
        )
    endif()
endfunction()

function(add_sycl_to_target)
    cmake_parse_arguments(
        PARSE_ARGV 0  # no positional arguments
        ARGS  # result variable prefix
        "SIMSYCL_ALL_WARNINGS"  # options
        "TARGET"  # one-value keywords
        "SOURCES"  # multi-value keywords (ignored)
    )

    if (NOT DEFINED ARGS_TARGET)
        message(FATAL_ERROR "add_sycl_to_target(): TARGET not specified")
    endif()

    if (ARGS_SIMSYCL_ALL_WARNINGS)
        set_simsycl_target_options(${ARGS_TARGET} ALL_WARNINGS)
    else()
        set_simsycl_target_options(${ARGS_TARGET})
    endif()
    target_link_libraries(${ARGS_TARGET} PRIVATE ${SIMSYCL_LIBRARY})
endfunction()
