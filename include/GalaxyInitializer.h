#ifndef GALAXY_INITIALIZER_H
#define GALAXY_INITIALIZER_H

#include <vector>
#include <memory>
#include <random>
#include "Particle.h"

namespace Init {
// Populate `particles` (already sized) with a central core at index 0 and
// distribute the remaining particles across `galaxyCount` spiral galaxies.
// The function writes into particles[1..]. It uses Physics::SofteningSquared
// (global) for softened velocity computations.
void generateGalaxies(
    std::vector<std::shared_ptr<Particle>> &particles,
    int armCount,
    int galaxyCount,
    float galaxyRadius,
    float galaxyRadiusSpacing,
    float diskRadius,
    float diskThickness,
    float armTurns,
    float armSpread,
    float interArmFraction,
    float interArmSpread,
    float speedScale,
    int coreMass,
    int particleMass,
    std::mt19937 &gen
);

}

#endif // GALAXY_INITIALIZER_H

