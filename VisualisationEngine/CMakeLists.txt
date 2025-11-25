file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cxx")

add_library(VisualisationEngine ${SOURCES})

target_include_directories(
  VisualisationEngine
  PUBLIC ${PROJECT_SOURCE_DIR}/VisualisationEngine/include
         ${Eigen3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${G2O_INCLUDE_DIRS})

target_link_libraries(
  VisualisationEngine PUBLIC ${OpenCV_LIBS} ${G2O_LIBRARIES} Sophus::Sophus
                             OpenMP::OpenMP_CXX)
