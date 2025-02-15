# src/CMakeLists.txt

# Add source files
set(SOURCES
        Particle.cpp
        OctTree.cpp
        fVector3.cpp
)

# Create a library from the source files
add_library(src ${SOURCES})