# CMakeLists.txt for YOLOv8 Pose Tracking Demo
# Standalone version for pose_track_demo

cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(yolov8-pose-tracking VERSION 1.0.0 LANGUAGES CXX)

# Output system info
message(STATUS "System: ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_VERSION}")

# C++17 required for ByteTrack and ReID
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Library architecture
set(LIB_ARCH "aarch64")
set(DEVICE_NAME "RK3588")

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread")

# ========== RKNN API Configuration ==========
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
set(RKNN_API_INCLUDE_PATH ${RKNN_API_PATH}/include)
set(RKNN_API_LIB_PATH ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)

# ========== RGA Configuration ==========
set(3RDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
set(RGA_DIR ${3RDPARTY_PATH}/rga/${DEVICE_NAME})
set(RGA_LIB ${RGA_DIR}/lib/Linux/${LIB_ARCH}/librga.so)

# ========== Find OpenCV ==========
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")

# ========== Find Eigen3 (Required for ByteTrack Kalman Filter) ==========
find_package(Eigen3 3.3 REQUIRED)
if(Eigen3_FOUND)
    message(STATUS "Eigen3 found: ${EIGEN3_VERSION_STRING}")
else()
    message(FATAL_ERROR "Eigen3 not found - ByteTracker requires Eigen3!")
endif()

# ========== Find OpenSSL (Required for WebSocket) ==========
find_package(OpenSSL REQUIRED)
message(STATUS "OpenSSL found: ${OPENSSL_VERSION}")

# ========== Include Directories ==========
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${RKNN_API_INCLUDE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${RGA_DIR}/include
)

# ========== Build Libraries ==========

# Preprocessing and postprocessing library
add_library(nn_process SHARED
    src/process/preprocess.cpp
    src/process/postprocess.cpp
)
target_link_libraries(nn_process
    ${OpenCV_LIBS}
    ${RGA_LIB}
)

# RKNN engine wrapper library
add_library(rknn_engine SHARED
    src/engine/rknn_engine.cpp
)
target_link_libraries(rknn_engine
    ${RKNN_API_LIB_PATH}
)

# YOLOv8 inference library
add_library(yolov8_lib SHARED
    src/task/yolov8_custom.cpp
)
target_link_libraries(yolov8_lib
    rknn_engine
    nn_process
)

# Drawing library
add_library(draw_lib SHARED
    src/draw/cv_draw.cpp
)
target_link_libraries(draw_lib
    ${OpenCV_LIBS}
)

# Configuration file parser library
add_library(reconfig_lib SHARED
    src/reconfig/ReConfig.cpp
)

# Position estimation library
add_library(position_lib SHARED
    src/position/position_estimator.cpp
)
target_link_libraries(position_lib
    ${OpenCV_LIBS}
)

# WebSocket server library
add_library(websocket_lib SHARED
    src/websocket/position_server.cpp
)
target_link_libraries(websocket_lib
    ${OPENSSL_LIBRARIES}
)

# ByteTrack tracking library
add_library(bytetrack_lib SHARED
    src/tracking/BYTETracker.cpp
    src/tracking/STrack.cpp
    src/tracking/kalmanFilter.cpp
    src/tracking/lapjv.cpp
    src/tracking/utils.cpp
)
target_link_libraries(bytetrack_lib
    Eigen3::Eigen
    ${OpenCV_LIBS}
)

# ReID feature extraction library
add_library(reid_lib SHARED
    src/reid/osnet_reid.cpp
)
target_link_libraries(reid_lib
    rknn_engine
    nn_process
)

# ========== Build Executable ==========

add_executable(pose_track_demo
    src/pose_track_demo.cpp
)

target_link_libraries(pose_track_demo
    draw_lib
    yolov8_lib
    bytetrack_lib
    reid_lib
    reconfig_lib
    position_lib
    websocket_lib
    ${OPENSSL_LIBRARIES}
)

# ========== Installation ==========
install(TARGETS pose_track_demo DESTINATION bin)
install(FILES pose_track_config.ini DESTINATION bin)
install(FILES pose_1_labels_list.txt DESTINATION bin)

message(STATUS "========================================")
message(STATUS "YOLOv8 Pose Tracking Demo Configuration Complete")
message(STATUS "========================================")
