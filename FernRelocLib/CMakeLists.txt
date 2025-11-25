file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cxx")

add_library(FernRelocLib ${SOURCES})

target_include_directories(
  FernRelocLib
  PUBLIC ${PROJECT_SOURCE_DIR}/FernRelocLib/include ${Eigen3_INCLUDE_DIRS}
         ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

target_link_libraries(FernRelocLib PUBLIC ${OpenCV_LIBS} ${G2O_LIBRARIES}
                                          Sophus::Sophus)
