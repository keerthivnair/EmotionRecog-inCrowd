cmake_minimum_required(VERSION 3.12)
project(face_parallel)

set(CMAKE_CXX_STANDARD 17)

# PyBind11
find_package(pybind11 REQUIRED)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Build module
add_library(face_parallel MODULE face_parallel.cpp)
target_link_libraries(face_parallel PRIVATE pybind11::module ${OpenCV_LIBS})

# OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(face_parallel PRIVATE OpenMP::OpenMP_CXX)
endif()

# Remove 'lib' prefix for Python import
set_target_properties(face_parallel PROPERTIES PREFIX "")
