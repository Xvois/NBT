#ifndef NBT_DIAGNOSTICS_H
#define NBT_DIAGNOSTICS_H

#include <memory>
#include <vector>

#include "Particle.h"

namespace Diagnostics {
struct SystemDiagnostics {
    double kineticEnergy = 0.0;
    double potentialEnergy = 0.0;
    double totalEnergy = 0.0;
    fVector3 totalMomentum = fVector3();
    fVector3 centerOfMass = fVector3();
};

// Sum of 0.5 * m * |v|^2 for all particles.
double computeTotalKineticEnergy(const std::vector<std::shared_ptr<Particle>> &particles);

// Pairwise gravitational potential energy with Plummer-like softening.
double computeTotalPotentialEnergy(
    const std::vector<std::shared_ptr<Particle>> &particles,
    float gravitationalConstant,
    float softeningSquared
);

SystemDiagnostics computeSystemDiagnostics(
    const std::vector<std::shared_ptr<Particle>> &particles,
    float gravitationalConstant,
    float softeningSquared
);
}

#endif // NBT_DIAGNOSTICS_H

