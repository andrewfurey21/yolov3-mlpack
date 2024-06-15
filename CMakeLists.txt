cmake_minimum_required(VERSION 3.28)
project(yolov3_tiny_mlpack)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/models)


set(ARMADILLO_LIBRARIES "/usr/lib/libarmadillo.so")
set(MLPACK_LIBRARIES ${MLPACK_LIBRARIES} ${ARMADILLO_LIBRARIES})

add_executable(main src/main.cpp)

target_link_libraries(main PRIVATE
    ${MLPACK_LIBRARIES}
)


