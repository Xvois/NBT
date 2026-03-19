# src/CMakeLists.txt

# Add source files
set(SOURCES
        Particle.cpp
        OctTree.cpp
        fVector3.cpp
        Physics.cpp
        GalaxyInitializer.cpp
        Diagnostics.cpp
        Hdf5SnapshotWriter.cpp
)

# Create a library from the source files
add_library(src ${SOURCES})