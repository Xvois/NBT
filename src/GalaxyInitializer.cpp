#include "../include/GalaxyInitializer.h"
#include "../include/OctTree.h"
#include <cmath>
#include <algorithm>

// Local Sersic sampler used by the initializer
struct SersicSampler {
    std::vector<float> radii;
    std::vector<double> cdf;
    float Re = 1.0f;
    float n = 1.0f;
    double b = 1.0;
    int bins = 1000;

    static double estimate_b(double n) {
        if (n <= 0.0) return 1.0;
        return 2.0 * n - 1.0 / 3.0 + 0.009876 / n;
    }

    void build(float Re_, float n_, float Rmax, int bins_ = 1000) {
        Re = Re_ > 0.0f ? Re_ : 1.0f;
        n = n_ > 0.0f ? n_ : 1.0f;
        bins = std::max(64, bins_);
        b = estimate_b(n);

        radii.assign(bins, 0.0f);
        cdf.assign(bins, 0.0);
        for (int i = 0; i < bins; ++i) radii[i] = Rmax * static_cast<float>(i) / static_cast<float>(bins - 1);

        std::vector<double> integrand(bins);
        for (int i = 0; i < bins; ++i) {
            double R = static_cast<double>(radii[i]);
            double x = (Re > 0.0f) ? (R / static_cast<double>(Re)) : 0.0;
            double expo = (x > 0.0) ? -b * std::pow(x, 1.0 / static_cast<double>(n)) : 0.0;
            double sigma = std::exp(expo);
            integrand[i] = 2.0 * M_PI * R * sigma;
        }

        double cum = 0.0;
        cdf[0] = 0.0;
        for (int i = 1; i < bins; ++i) {
            double dx = static_cast<double>(radii[i]) - static_cast<double>(radii[i - 1]);
            cum += 0.5 * (integrand[i] + integrand[i - 1]) * dx;
            cdf[i] = cum;
        }
        double total = cdf.back();
        if (total <= 0.0) total = 1.0;
        for (int i = 0; i < bins; ++i) cdf[i] /= total;
        cdf.back() = 1.0;
    }

    float sample(std::mt19937 &gen, std::uniform_real_distribution<> &ud) const {
        double u = std::min(0.9999999, std::max(0.0, ud(gen)));
        auto it = std::lower_bound(cdf.begin(), cdf.end(), u);
        int idx = static_cast<int>(std::distance(cdf.begin(), it));
        if (idx == 0) return radii.front();
        if (idx >= static_cast<int>(cdf.size())) return radii.back();
        int i0 = idx - 1;
        int i1 = idx;
        double t = (u - cdf[i0]) / (cdf[i1] - cdf[i0]);
        double r = static_cast<double>(radii[i0]) * (1.0 - t) + static_cast<double>(radii[i1]) * t;
        return static_cast<float>(r);
    }

    double cdfAt(double r) const {
        if (r <= static_cast<double>(radii.front())) return 0.0;
        if (r >= static_cast<double>(radii.back())) return 1.0;
        auto it = std::upper_bound(radii.begin(), radii.end(), static_cast<float>(r));
        int idx = static_cast<int>(std::distance(radii.begin(), it));
        if (idx == 0) return cdf[0];
        int i0 = idx - 1;
        int i1 = idx;
        double t = (r - static_cast<double>(radii[i0])) / (static_cast<double>(radii[i1]) - static_cast<double>(radii[i0]));
        return cdf[i0] * (1.0 - t) + cdf[i1] * t;
    }
};

namespace Init {

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
) {
    const int totalParticles = static_cast<int>(particles.size());
    if (totalParticles < 2) return;

    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> angleJitter(-armSpread, armSpread);
    std::uniform_real_distribution<> interArmJitter(-interArmSpread, interArmSpread);
    std::uniform_real_distribution<> thicknessJitter(-diskThickness * 0.5f, diskThickness * 0.5f);

    // We will create one SMBH per galaxy (no single central static core).

    SersicSampler sersic;
    const float sersic_n = 1.0f;
    const float sersic_Re = diskRadius / 3.0f;
    sersic.build(sersic_Re, sersic_n, diskRadius, 1000);

    const float twoPi = 2.0f * static_cast<float>(M_PI);
    // Reserve one SMBH per galaxy
    if (galaxyCount <= 0) galaxyCount = 1;
    if (galaxyCount > totalParticles) galaxyCount = totalParticles; // clamp
    const int totalDiskParticles = totalParticles - galaxyCount;
    const float globalDiskMassTotal = static_cast<float>(totalDiskParticles * particleMass);
    // Distribute particles as evenly as possible between galaxies. Handle remainder
    // so all particle slots are filled (no null shared_ptrs remain).
    if (galaxyCount <= 0) galaxyCount = 1;
    const int perGalaxyBase = (galaxyCount > 0) ? (totalDiskParticles / galaxyCount) : 0;
    const int remainder = (galaxyCount > 0) ? (totalDiskParticles % galaxyCount) : 0;
    int idx = 0; // start filling at index 0
    for (int g = 0; g < galaxyCount; ++g) {
        int countForThisGalaxy = perGalaxyBase + (g < remainder ? 1 : 0);
        // place each galaxy center at radius galaxyRadius + g*spacing and random orientation
        float centerR = galaxyRadius + g * galaxyRadiusSpacing;
        float centerTheta = static_cast<float>(dis(gen)) * twoPi;
        float cx = centerR * std::cos(centerTheta);
        float cz = centerR * std::sin(centerTheta);
        // Choose a random rotation axis (uniform on sphere) for this galaxy so its
        // disk lies in a random axial plane. This produces different spin axes.
        double uz = 2.0 * dis(gen) - 1.0; // cos(theta) uniform
        double uphi = 2.0 * M_PI * dis(gen);
        double ux = std::sqrt(std::max(0.0, 1.0 - uz * uz)) * std::cos(uphi);
        double uy = std::sqrt(std::max(0.0, 1.0 - uz * uz)) * std::sin(uphi);
        fVector3 axis(static_cast<float>(ux), static_cast<float>(uy), static_cast<float>(uz));

        // Build an orthonormal basis (u,v,axis) where u and v span the galaxy plane
        fVector3 arbitrary(1.0f, 0.0f, 0.0f);
        if (std::abs(axis.x) > 0.9f) arbitrary = fVector3(0.0f, 1.0f, 0.0f);
        fVector3 u = fVector3::cross(axis, arbitrary);
        float umag = fVector3::magnitude(u);
        if (umag < 1e-6f) {
            u = fVector3(1.0f, 0.0f, 0.0f);
            umag = 1.0f;
        }
        u = u / umag;
        fVector3 v = fVector3::cross(axis, u);
        float vmag = fVector3::magnitude(v);
        if (vmag < 1e-6f) {
            v = fVector3(0.0f, 0.0f, 1.0f);
        } else {
            v = v / vmag;
        }

        // --- SMBH for this galaxy ---
        // Place SMBH at galaxy centre and give it a tangential velocity so it is not static.
        // Compute a simple circular speed around origin using the combined mass (SMBHs + disks)
        // as an approximation so SMBHs orbit the system center instead of being static.
        // Approximate total mass as (galaxyCount * coreMass + totalDiskParticles * particleMass).
        const double approxTotalMass = static_cast<double>(galaxyCount) * static_cast<double>(coreMass)
                                      + static_cast<double>(totalDiskParticles) * static_cast<double>(particleMass);

        // SMBH index
        if (idx >= totalParticles) break;
        fVector3 smbhPos(cx, 0.0f, cz);
        // tangential direction around origin
        fVector3 smbhTangent(-cz, 0.0f, cx);
        float smbhTangentMag = fVector3::magnitude(smbhTangent);
        if (smbhTangentMag > 1e-6f) smbhTangent = smbhTangent / smbhTangentMag; else smbhTangent = fVector3(1.0f, 0.0f, 0.0f);
        float smbhR = std::sqrt(cx * cx + cz * cz);
        float smbhSpeed = 0.0f;
        const float eps2 = Physics::SofteningSquared;
        if (smbhR > 1e-6f) {
            const float denom = std::pow(smbhR * smbhR + eps2, 0.75f);
            smbhSpeed = std::sqrt(Physics::GravityConstant * static_cast<float>(approxTotalMass)) * smbhR / denom;
        }
        fVector3 smbhVel = smbhTangent * smbhSpeed;
        particles[idx] = std::make_shared<Particle>(smbhPos, smbhVel, coreMass);
        ++idx;

        float perGalaxyDiskMassTotal = static_cast<float>(countForThisGalaxy * particleMass);
        for (int j = 0; j < countForThisGalaxy && idx < totalParticles; ++j, ++idx) {
            float r = sersic.sample(gen, dis);
            int armIndex = static_cast<int>(dis(gen) * armCount) % armCount;
            float baseAngle = (static_cast<float>(armIndex) / static_cast<float>(armCount)) * twoPi;
            float armSpacing = twoPi / static_cast<float>(armCount);
            bool isInterArm = static_cast<float>(dis(gen)) < interArmFraction;
            float angle;
            if (isInterArm) {
                float midAngle = baseAngle + 0.5f * armSpacing;
                angle = midAngle + (r / diskRadius) * (armTurns * twoPi) + static_cast<float>(interArmJitter(gen));
            } else {
                angle = baseAngle + (r / diskRadius) * (armTurns * twoPi) + static_cast<float>(angleJitter(gen));
            }

            // position in the rotated galaxy plane: center + u*(r*cos) + v*(r*sin) + axis*(thickness)
            float planarX = r * std::cos(angle);
            float planarY = r * std::sin(angle);
            float thickness = static_cast<float>(thicknessJitter(gen));
            fVector3 position = fVector3(cx, 0.0f, cz) + u * planarX + v * planarY + axis * thickness;

            // Tangential direction in rotated plane: tangent = -sin(angle)*u + cos(angle)*v
            fVector3 tangent = (u * (-std::sin(angle))) + (v * std::cos(angle));
            float tangentMag = fVector3::magnitude(tangent);
            if (tangentMag > 1e-6f) tangent = tangent / tangentMag; else tangent = fVector3(1.0f, 0.0f, 0.0f);

            double fracEnclosed = sersic.cdfAt(static_cast<double>(r));
            float diskMassEnclosed = static_cast<float>(fracEnclosed * static_cast<double>(perGalaxyDiskMassTotal));
            float effectiveMass = static_cast<float>(coreMass) + diskMassEnclosed;

            const float eps2 = Physics::SofteningSquared;
            float orbitSpeed = 0.0f;
            if (r > 1e-6f) {
                const float denom = std::pow(r * r + eps2, 0.75f);
                orbitSpeed = std::sqrt(Physics::GravityConstant * effectiveMass) * r / denom;
                orbitSpeed *= speedScale;
            }

            fVector3 velocity = tangent * orbitSpeed;
            particles[idx] = std::make_shared<Particle>(position, velocity, particleMass);
        }
    }
    // If any particle slots remain (due to rounding or other), fill them with
    // diffuse background sampled from the same Sersic distribution around origin.
    while (idx < totalParticles) {
        float r = sersic.sample(gen, dis);
        float theta = static_cast<float>(dis(gen)) * twoPi;
        float x = r * std::cos(theta);
        float z = r * std::sin(theta);
        float y = static_cast<float>(thicknessJitter(gen));
        fVector3 position(x, y, z);
        fVector3 tangent(-z, 0.0f, x);
        float tangentMag = fVector3::magnitude(tangent);
        if (tangentMag > 1e-6f) tangent = tangent / tangentMag; else tangent = fVector3(1.0f, 0.0f, 0.0f);
        double fracEnclosed = sersic.cdfAt(static_cast<double>(r));
        float diskMassEnclosed = static_cast<float>(fracEnclosed * static_cast<double>(globalDiskMassTotal));
        float effectiveMass = static_cast<float>(coreMass) + diskMassEnclosed;
        const float eps2 = Physics::SofteningSquared;
        float orbitSpeed = 0.0f;
        if (r > 1e-6f) {
            const float denom = std::pow(r * r + eps2, 0.75f);
            orbitSpeed = std::sqrt(Physics::GravityConstant * effectiveMass) * r / denom;
            orbitSpeed *= speedScale;
        }
        fVector3 velocity = tangent * orbitSpeed;
        particles[idx] = std::make_shared<Particle>(position, velocity, particleMass);
        ++idx;
    }
}

} // namespace Init

