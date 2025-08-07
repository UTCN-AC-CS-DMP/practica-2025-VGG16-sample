cmake_minimum_required(VERSION 3.26 FATAL_ERROR) # USE ANYTHING LESS THAN 3.27!!!!
project(practica-2025-vgg16-inf
        VERSION 1.0.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV version: " ${OpenCV_VERSION})

add_definitions(-DASSETS_DIR="${CMAKE_SOURCE_DIR}/assets/")

add_executable(${PROJECT_NAME}
        src/main.cc)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
