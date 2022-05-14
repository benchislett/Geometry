cmake_minimum_required(VERSION 3.16)
project(example)

add_subdirectory(deps/geometry-central)
add_subdirectory(deps/polyscope)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Libigl
# option(LIBIGL_USE_STATIC_LIBRARY "Use libIGL as static library" ON)
option(LIBIGL_GLFW "Build target igl::glfw" OFF)
include(libigl)

# Add your project files
file(GLOB SRC_FILES *.cpp)
add_executable(${PROJECT_NAME} ${SRC_FILES})
target_link_libraries(${PROJECT_NAME} PUBLIC igl::core)
target_link_libraries(${PROJECT_NAME} PUBLIC geometry-central)
target_link_libraries(${PROJECT_NAME} PUBLIC polyscope)

