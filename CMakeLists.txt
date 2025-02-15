cmake_minimum_required(VERSION 3.30)
project(NBT)

set(CMAKE_OSX_ARCHITECTURES "arm64")

set(CMAKE_CXX_STANDARD 20)

# Add the src directory
add_subdirectory(src)

# Add the include directory for header files
include_directories(include)

# Find GLFW and GLEW packages
find_package(glfw3 3.3 REQUIRED)
find_package(GLEW REQUIRED)

add_executable(NBT main.cpp
        include/fVector3.h
        src/fVector3.cpp
        include/Particle.h
        src/Particle.cpp
        include/OctTree.h
        src/OctTree.cpp
)

# Set the optimization level to O0, for debugging
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")

# Link GLFW and GLEW libraries
target_link_libraries(NBT glfw GLEW::GLEW)