# src/CMakeLists.txt

# Add source files
set(SOURCES Particle.h fVector3.h OctTree.h)

# Create a library from the source files
add_library(src ${SOURCES})