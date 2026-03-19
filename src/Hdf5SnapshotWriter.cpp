#include "Hdf5SnapshotWriter.h"

#include <iostream>
#include <stdexcept>

#ifdef NBT_HAVE_HDF5
#include <H5Cpp.h>

class Hdf5SnapshotWriter::Impl {
public:
    Impl(const std::string &outputPath, const size_t particleCount, const float gravityConstant)
        : file(outputPath, H5F_ACC_TRUNC), particleCount(particleCount) {
        H5::Group root = file.createGroup("/metadata");

        writeScalarAttribute(root, "particle_count", static_cast<long long>(particleCount));
        writeScalarAttribute(root, "gravity_constant", gravityConstant);
    }

    void writeSnapshot(
        const ExportSnapshot &snapshot,
        const Diagnostics::SystemDiagnostics &diagnostics,
        const std::vector<std::shared_ptr<Particle>> &particles
    ) {
        if (particles.size() != particleCount) {
            throw std::runtime_error("Particle count mismatch while writing snapshot.");
        }

        const std::string groupPath = "/snapshots/step_" + std::to_string(snapshot.step);
        if (!groupExists("/snapshots")) {
            file.createGroup("/snapshots");
        }
        if (groupExists(groupPath)) {
            return;
        }

        H5::Group stepGroup = file.createGroup(groupPath);

        writeScalarAttribute(stepGroup, "step", snapshot.step);
        writeScalarAttribute(stepGroup, "simulation_time", snapshot.simulationTime);
        writeScalarAttribute(stepGroup, "tree_seconds", snapshot.treeSeconds);
        writeScalarAttribute(stepGroup, "force_seconds", snapshot.forceSeconds);
        writeScalarAttribute(stepGroup, "iteration_seconds", snapshot.iterationSeconds);
        writeScalarAttribute(stepGroup, "average_dt", snapshot.averageDt);
        writeScalarAttribute(stepGroup, "max_acceleration", snapshot.maxAcceleration);

        writeScalarAttribute(stepGroup, "kinetic_energy", diagnostics.kineticEnergy);
        writeScalarAttribute(stepGroup, "potential_energy", diagnostics.potentialEnergy);
        writeScalarAttribute(stepGroup, "total_energy", diagnostics.totalEnergy);

        writeVector3Attribute(stepGroup, "total_momentum", diagnostics.totalMomentum);
        writeVector3Attribute(stepGroup, "center_of_mass", diagnostics.centerOfMass);

        std::vector<float> positions;
        positions.resize(particleCount * 3);

        std::vector<float> velocities;
        velocities.resize(particleCount * 3);

        std::vector<int> masses;
        masses.resize(particleCount);

        for (size_t i = 0; i < particleCount; ++i) {
            const fVector3 p = particles[i]->getPosition();
            const fVector3 v = particles[i]->getVelocity();

            positions[i * 3] = p.x;
            positions[i * 3 + 1] = p.y;
            positions[i * 3 + 2] = p.z;

            velocities[i * 3] = v.x;
            velocities[i * 3 + 1] = v.y;
            velocities[i * 3 + 2] = v.z;

            masses[i] = particles[i]->getMass();
        }

        const hsize_t vecDims[2] = {particleCount, 3};
        H5::DataSpace vecSpace(2, vecDims);
        H5::DataSet posDataset = stepGroup.createDataSet("positions", H5::PredType::NATIVE_FLOAT, vecSpace);
        posDataset.write(positions.data(), H5::PredType::NATIVE_FLOAT);

        H5::DataSet velDataset = stepGroup.createDataSet("velocities", H5::PredType::NATIVE_FLOAT, vecSpace);
        velDataset.write(velocities.data(), H5::PredType::NATIVE_FLOAT);

        const hsize_t massDims[1] = {particleCount};
        H5::DataSpace massSpace(1, massDims);
        H5::DataSet massDataset = stepGroup.createDataSet("masses", H5::PredType::NATIVE_INT, massSpace);
        massDataset.write(masses.data(), H5::PredType::NATIVE_INT);
    }

private:
    template <typename T>
    static void writeScalarAttribute(H5::Group &group, const std::string &name, const T value) {
        H5::DataSpace attrSpace(H5S_SCALAR);
        const auto type = getType<T>();
        H5::Attribute attribute = group.createAttribute(name, type, attrSpace);
        attribute.write(type, &value);
    }

    static void writeVector3Attribute(H5::Group &group, const std::string &name, const fVector3 &value) {
        const hsize_t dims[1] = {3};
        H5::DataSpace attrSpace(1, dims);
        H5::Attribute attribute = group.createAttribute(name, H5::PredType::NATIVE_FLOAT, attrSpace);
        const float data[3] = {value.x, value.y, value.z};
        attribute.write(H5::PredType::NATIVE_FLOAT, data);
    }

    template <typename T>
    static const H5::PredType &getType();

    bool groupExists(const std::string &path) const {
        return H5Lexists(file.getId(), path.c_str(), H5P_DEFAULT) > 0;
    }

    H5::H5File file;
    size_t particleCount;
};

template <>
const H5::PredType &Hdf5SnapshotWriter::Impl::getType<int>() {
    return H5::PredType::NATIVE_INT;
}

template <>
const H5::PredType &Hdf5SnapshotWriter::Impl::getType<long long>() {
    return H5::PredType::NATIVE_LLONG;
}

template <>
const H5::PredType &Hdf5SnapshotWriter::Impl::getType<float>() {
    return H5::PredType::NATIVE_FLOAT;
}

template <>
const H5::PredType &Hdf5SnapshotWriter::Impl::getType<double>() {
    return H5::PredType::NATIVE_DOUBLE;
}
#endif

Hdf5SnapshotWriter::Hdf5SnapshotWriter(const std::string &outputPath, const size_t particleCount, const float gravityConstant) {
#ifdef NBT_HAVE_HDF5
    try {
        impl_ = std::make_unique<Impl>(outputPath, particleCount, gravityConstant);
        enabled_ = true;
        statusMessage_ = "Exporting snapshots to " + outputPath;
    } catch (const H5::Exception &e) {
        enabled_ = false;
        statusMessage_ = "Failed to initialize HDF5 writer: " + std::string(e.getDetailMsg());
    } catch (const std::exception &e) {
        enabled_ = false;
        statusMessage_ = "Failed to initialize HDF5 writer: " + std::string(e.what());
    }
#else
    (void)outputPath;
    (void)particleCount;
    (void)gravityConstant;
    enabled_ = false;
    statusMessage_ = "HDF5 support is disabled at build time; snapshots are not written.";
#endif
}

Hdf5SnapshotWriter::~Hdf5SnapshotWriter() = default;

bool Hdf5SnapshotWriter::isEnabled() const {
#ifdef NBT_HAVE_HDF5
    return enabled_;
#else
    return false;
#endif
}

std::string Hdf5SnapshotWriter::statusMessage() const {
    return statusMessage_;
}

void Hdf5SnapshotWriter::writeSnapshot(
    const ExportSnapshot &snapshot,
    const Diagnostics::SystemDiagnostics &diagnostics,
    const std::vector<std::shared_ptr<Particle>> &particles
) {
#ifdef NBT_HAVE_HDF5
    if (!enabled_ || !impl_) {
        return;
    }

    try {
        impl_->writeSnapshot(snapshot, diagnostics, particles);
    } catch (const H5::Exception &e) {
        enabled_ = false;
        statusMessage_ = "Snapshot write failed; disabling export: " + std::string(e.getDetailMsg());
        std::cerr << statusMessage_ << std::endl;
    } catch (const std::exception &e) {
        enabled_ = false;
        statusMessage_ = "Snapshot write failed; disabling export: " + std::string(e.what());
        std::cerr << statusMessage_ << std::endl;
    }
#else
    (void)snapshot;
    (void)diagnostics;
    (void)particles;
#endif
}

