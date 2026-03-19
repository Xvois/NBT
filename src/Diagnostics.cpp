#include "Diagnostics.h"

#include <cmath>

namespace Diagnostics {
double computeTotalKineticEnergy(const std::vector<std::shared_ptr<Particle>> &particles) {
    double total = 0.0;
    for (const auto &particle : particles) {
        if (!particle) continue;
        const fVector3 velocity = particle->getVelocity();
        const double speedSquared = static_cast<double>(fVector3::magnitudeSquare(velocity));
        total += 0.5 * static_cast<double>(particle->getMass()) * speedSquared;
    }
    return total;
}

double computeTotalPotentialEnergy(
    const std::vector<std::shared_ptr<Particle>> &particles,
    const float gravitationalConstant,
    const float softeningSquared
) {
    double total = 0.0;
    const size_t n = particles.size();
    for (size_t i = 0; i < n; ++i) {
        if (!particles[i]) continue;
        const fVector3 pi = particles[i]->getPosition();
        const double mi = static_cast<double>(particles[i]->getMass());
        for (size_t j = i + 1; j < n; ++j) {
            if (!particles[j]) continue;
            const fVector3 pj = particles[j]->getPosition();
            const double mj = static_cast<double>(particles[j]->getMass());
            const fVector3 delta = pj - pi;
            const double r2 = static_cast<double>(fVector3::magnitudeSquare(delta));
            const double softenedDistance = std::sqrt(r2 + static_cast<double>(softeningSquared));
            if (softenedDistance > 0.0) {
                total -= static_cast<double>(gravitationalConstant) * mi * mj / softenedDistance;
            }
        }
    }
    return total;
}

SystemDiagnostics computeSystemDiagnostics(
    const std::vector<std::shared_ptr<Particle>> &particles,
    const float gravitationalConstant,
    const float softeningSquared
) {
    SystemDiagnostics diagnostics;
    diagnostics.kineticEnergy = computeTotalKineticEnergy(particles);
    diagnostics.potentialEnergy = computeTotalPotentialEnergy(particles, gravitationalConstant, softeningSquared);
    diagnostics.totalEnergy = diagnostics.kineticEnergy + diagnostics.potentialEnergy;

    double totalMass = 0.0;
    fVector3 weightedPosition(0.0f, 0.0f, 0.0f);
    fVector3 momentum(0.0f, 0.0f, 0.0f);

    for (const auto &particle : particles) {
        if (!particle) continue;
        const float mass = static_cast<float>(particle->getMass());
        totalMass += static_cast<double>(mass);
        weightedPosition += particle->getPosition() * mass;
        momentum += particle->getVelocity() * mass;
    }

    if (totalMass > 0.0) {
        diagnostics.centerOfMass = weightedPosition / static_cast<float>(totalMass);
    }
    diagnostics.totalMomentum = momentum;

    return diagnostics;
}
}

