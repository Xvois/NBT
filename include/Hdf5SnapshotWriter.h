#ifndef NBT_HDF5SNAPSHOTWRITER_H
#define NBT_HDF5SNAPSHOTWRITER_H

#include <memory>
#include <string>
#include <vector>

#include "Diagnostics.h"

struct ExportSnapshot {
    int step = 0;
    double simulationTime = 0.0;
    double treeSeconds = 0.0;
    double forceSeconds = 0.0;
    double iterationSeconds = 0.0;
    float averageDt = 0.0f;
    float maxAcceleration = 0.0f;
};

class Hdf5SnapshotWriter {
public:
    Hdf5SnapshotWriter(const std::string &outputPath, size_t particleCount, float gravityConstant);
    ~Hdf5SnapshotWriter();

    bool isEnabled() const;
    std::string statusMessage() const;

    void writeSnapshot(
        const ExportSnapshot &snapshot,
        const Diagnostics::SystemDiagnostics &diagnostics,
        const std::vector<std::shared_ptr<Particle>> &particles
    );

private:
    bool enabled_ = false;
    std::string statusMessage_;

#ifdef NBT_HAVE_HDF5
    class Impl;
    std::unique_ptr<Impl> impl_;
#endif
};

#endif // NBT_HDF5SNAPSHOTWRITER_H

