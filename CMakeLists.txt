cmake_minimum_required(VERSION 3.16...3.26)
project(
  kiss_icp_cpp
  VERSION 1.2.3
  LANGUAGES CXX)

# Setup build options
option(USE_CCACHE "Build using Ccache if found on the path" ON)
option(USE_SYSTEM_TSL-ROBIN-MAP "Use system pre-installed tsl_robin" ON)
option(USE_SYSTEM_TBB "Use system pre-installed oneAPI/tbb" ON)

# ccache setup
if(USE_CCACHE)
  find_program(CCACHE_PATH ccache)
  if(CCACHE_PATH)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
    message(STATUS "Using ccache: ${CCACHE_PATH}")
  endif()
endif()

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(g2o REQUIRED)
find_package(rclcpp REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(sensor_msgs REQUIRED)
# set(OrbbecSDK_DIR "/home/adrewn/OrbbecSDK") find_package(OrbbecSDK REQUIRED)

# Set build type (repeat here for C++ only consumers)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(third_party/find_dependencies.cmake)
include(cmake/CompilerOptions.cmake)

add_library(tracker SHARED)
target_sources(tracker PRIVATE src/tracker.cpp)
target_link_libraries(tracker PUBLIC Eigen3::Eigen tsl::robin_map TBB::tbb
                                     ${OpenCV_LIBS})

target_include_directories(tracker PUBLIC include)
set_global_target_properties(tracker)

set(ROS_ROOT_DIR /opt/ros/humble)

set(all_headers
    /opt/ros/humble/include /opt/ros/humble/include/cv_bridge
    /opt/ros/humble/include/tf2_ros /opt/ros/humble/include/tf2_msgs
    /opt/ros/humble/include/sensor_msgs /opt/ros/humble/include/message_filters)

add_executable(visualizer test/test_visualizer.cpp)

target_include_directories(visualizer PUBLIC include ${OpenCV_INCLUDE_DIRS}
                                             ${all_headers})

target_link_directories(visualizer PUBLIC ${ROS_ROOT_DIR}/lib)

target_link_libraries(
  visualizer
  PUBLIC cv_bridge
         tf2_ros
         tf2_msgs__rosidl_typesupport_cpp
         sensor_msgs__rosidl_typesupport_cpp
         rclcpp::rclcpp
         cv_bridge::cv_bridge
         tracker)
