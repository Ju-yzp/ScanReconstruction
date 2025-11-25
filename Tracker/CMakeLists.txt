add_library(tracker src/tracker.cpp)

target_include_directories(
  tracker PUBLIC ${PROJECT_SOURCE_DIR}/Tracker/include ${Eigen3_INCLUDE_DIRS}
                 ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

target_link_libraries(tracker PUBLIC ${OpenCV_LIBS} ${G2O_LIBRARIES}
                                     Sophus::Sophus)
