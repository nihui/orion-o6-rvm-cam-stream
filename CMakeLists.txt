cmake_minimum_required(VERSION 3.10)
project(opencv-test)

set(CMAKE_BUILD_TYPE Release)

set(OpenCV_DIR "/home/radxa/opencv-mobile-4.11.0-debian-bookworm-aarch64/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

set(ncnn_DIR "/home/radxa/ncnn/build/install/lib/cmake/ncnn")
find_package(ncnn REQUIRED)

include_directories("/usr/share/cix/include")
link_directories("/usr/share/cix/lib")

add_executable(test main.cpp)

target_link_libraries(test ncnn ${OpenCV_LIBS} noe)
