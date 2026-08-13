cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)

project(Sequential_EBL_Dwell_Time_Optimization)

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/support/cmake/modules)

if (NOT WIN32)
find_package(EnableCxx11 REQUIRED)
endif (NOT WIN32)

find_path(LIBWB_ROOT NAMES wb.h HINTS libwb ../libwb ../../libwb ../../../libwb)
include_directories(${LIBWB_ROOT})
find_path(LIBWB_LIB_DIR NAMES libwb.so HINTS ${LIBWB_ROOT}/lib ${LIBWB_ROOT}/build)
link_directories(${LIBWB_LIB_DIR})

# Add additional source files below
include_directories(include)
add_executable(
    Sequential_EBL_Dwell_Time_Optimization

    sequential.cpp
)

target_link_libraries(Sequential_EBL_Dwell_Time_Optimization wb rt)