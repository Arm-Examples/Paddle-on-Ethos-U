if (NOT ARM_TARGET_OS STREQUAL "arm_baremetal" )
add_subdirectory(math)
endif()
