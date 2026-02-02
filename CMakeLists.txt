cmake_minimum_required(VERSION 3.28)
project(yolov3_tiny_mlpack)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ARMADILLO_LIBRARIES "/usr/lib/x86_64-linux-gnu/libarmadillo.so")
# set(ARMADILLO_LIBRARIES "/usr/lib/libarmadillo.so")
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})

# add_executable(yolov3 yolov3/yolov3.cpp)
add_executable(yolov3 example.cpp)
# add_executable(yolov3-tiny yolov3/yolov3-tiny.cpp)
# add_executable(yolov3-spp yolov3/yolov3-spp.cpp)

target_link_libraries(yolov3 ${MLPACK_LIBRARIES})
# target_link_libraries(yolov3-tiny ${MLPACK_LIBRARIES})
# target_link_libraries(yolov3-spp ${MLPACK_LIBRARIES})


