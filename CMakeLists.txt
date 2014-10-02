cmake_minimum_required(VERSION 2.6)

project(WiimotePositionTrackingDemo)

add_subdirectory(external/wiiuse)

find_package(SDL REQUIRED)
find_package(OpenGL REQUIRED)
find_package(OpenCV REQUIRED)

ADD_DEFINITIONS(
    -std=c++11
)
set(CMAKE_CXX_FLAGS "-g -Wall")

include_directories(${WIIUSE_INCLUDE_DIR} ${SDL_INCLUDE_DIR} ${OPENGL_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
add_executable(demo demo.cpp)
target_link_libraries(demo ${WIIUSE_LIBRARY} ${SDL_LIBRARY} ${OPENGL_LIBRARIES} ${OpenCV_LIBS})
