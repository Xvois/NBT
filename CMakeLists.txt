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
find_package(HDF5 COMPONENTS C CXX)

add_executable(NBT main.cpp
        include/fVector3.h
        src/fVector3.cpp
        include/Particle.h
        src/Particle.cpp
        include/OctTree.h
        src/OctTree.cpp
        src/Physics.cpp
        src/GalaxyInitializer.cpp
        include/Diagnostics.h
        src/Diagnostics.cpp
        include/Hdf5SnapshotWriter.h
        src/Hdf5SnapshotWriter.cpp
)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O2")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# Link GLFW and GLEW libraries
target_link_libraries(NBT glfw GLEW::GLEW)

if(HDF5_FOUND)
    target_compile_definitions(NBT PRIVATE NBT_HAVE_HDF5)
    target_include_directories(NBT PRIVATE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(NBT ${HDF5_LIBRARIES})
endif()
