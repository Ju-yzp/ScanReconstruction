cmake_minimum_required(VERSION 3.16...3.26)
project(
  kiss_icp_cpp
  VERSION 1.2.3
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-march=native)
add_compile_options(-msse4.2 -mavx2 -mfma)

add_definitions(-DEIGEN_VECTORIZE_SSE42 -DEIGEN_VECTORIZE_AVX2)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Build type (Debug, Release, RelWithDebInfo, MinSizeRel)"
            FORCE)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")

set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -g -fno-omit-frame-pointer -DNDEBUG")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")

set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -Os -DNDEBUG")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(USE_CCACHE "Build using Ccache if found on the path" ON)
option(USE_SYSTEM_TSL-ROBIN-MAP "Use system pre-installed tsl_robin" ON)
option(USE_SYSTEM_TBB "Use system pre-installed oneAPI/tbb" ON)

option(WITH_PROFILING
       "Build with profiling support (debug symbols, frame pointer)" OFF)

if(WITH_PROFILING)
  message(STATUS "Profiling build enabled")

  add_compile_options(-g -fno-omit-frame-pointer)
  add_link_options(-g -fno-omit-frame-pointer)

  if(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(
      WARNING
        "For better profiling results, consider using RelWithDebInfo build type"
    )
  endif()
endif()

if(USE_CCACHE)
  find_program(CCACHE_PATH ccache)
  if(CCACHE_PATH)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(STATUS "Using ccache: ${CCACHE_PATH}")
  endif()
endif()
set(IPP_DIR "/opt/intel/oneapi/ipp/latest/lib/cmake/ipp")
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Sophus REQUIRED)
find_package(IPP REQUIRED)
find_package(g2o REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)

set(OrbbecSDK_DIR "/home/adrewn/OrbbecSDK")
find_package(OrbbecSDK REQUIRED)

include(third_party/find_dependencies.cmake)
include(cmake/CompilerOptions.cmake)

set(ROS_ROOT_DIR /opt/ros/humble)

set(all_headers
    /opt/ros/humble/include /opt/ros/humble/include/cv_bridge
    /opt/ros/humble/include/tf2_ros /opt/ros/humble/include/tf2_msgs
    /opt/ros/humble/include/sensor_msgs /opt/ros/humble/include/message_filters)

add_executable(visualizer test/test_visualizer.cpp src/Allocator.cpp
                          src/DepthTracker.cpp)

target_compile_definitions(
  visualizer PRIVATE TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION=1)

set_target_properties(visualizer PROPERTIES CXX_STANDARD 20
                                            CXX_STANDARD_REQUIRED ON)

target_include_directories(visualizer PUBLIC include ${OpenCV_INCLUDE_DIRS}
                                             ${all_headers})

target_link_directories(visualizer PUBLIC ${ROS_ROOT_DIR}/lib)

target_link_libraries(
  visualizer
  PUBLIC Eigen3::Eigen
         TBB::tbb
         ${OpenCV_LIBS}
         cv_bridge
         tf2_ros
         tf2_msgs__rosidl_typesupport_cpp
         sensor_msgs__rosidl_typesupport_cpp
         rclcpp::rclcpp
         cv_bridge::cv_bridge
         visualization_msgs::visualization_msgs__rosidl_typesupport_cpp
         openvino::runtime
         stdc++fs)
