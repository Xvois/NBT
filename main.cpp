#include <iostream>
#include <random>
#include <vector>
#include <algorithm>
#include <cmath>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>
#include <string>
#include <memory>

#include "OctTree.h"
#include "Particle.h"
#include "Diagnostics.h"
#include "Hdf5SnapshotWriter.h"
#include "GalaxyInitializer.h"

/*
 * 1000 - Base 224.027ms
 * 1000 - Reserve - 187.878ms
 * 1000 - // + QuotientSqr - 166.364ms
 * 1000 - // + Vector class fixes - 152.871ms
 * 1000 - Force resolution in tree - 120.912ms
 * 1000 - // + Hierarchical pseudos - 89.0132ms
 * 1000 - // + minor tweaks - 76.0814ms
 * 1000 - O2 optimisation - 12.7235ms
*/

namespace Graphics {
    float cameraAngle = 0.0f;
    float cameraRadius = 200.0f;
    float cameraSpeed = 0.001f;

    void renderParticles(const std::vector<std::shared_ptr<Particle> > &particles) {
        glEnable(GL_BLEND); // Enable blending
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Set the blend function

        glPointSize(2.0f);
        for (size_t i = 0; i < particles.size(); ++i) {
            if (i == 0) {
                glColor4f(1.0f, 1.0f, 0.2f, 1.0f); // Highlight the core
            } else {
                glColor4f(1.0f, 1.0f, 1.0f, 1.0f); // Set color to white with transparency for the rest
            }
            fVector3 pos = particles[i]->getPosition();
            glBegin(GL_POINTS);
            glVertex3f(pos.x, pos.y, pos.z);
            glEnd();
        }

        glDisable(GL_BLEND); // Disable blending
    }

    GLFWwindow *initGraphics(int width, int height, float projectionExtent) {
        if (!glfwInit()) {
            std::cerr << "Failed to initialize GLFW" << std::endl;
            return nullptr;
        }

        GLFWwindow *window = glfwCreateWindow(width, height, "Particle Simulation", nullptr, nullptr);
        if (!window) {
            std::cerr << "Failed to create GLFW window" << std::endl;
            glfwTerminate();
            return nullptr;
        }

        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;
        if (glewInit() != GLEW_OK) {
            std::cerr << "Failed to initialize GLEW" << std::endl;
            return nullptr;
        }

        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glPointSize(0.25f);
        glEnable(GL_POINT_SMOOTH);
        glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glOrtho(-projectionExtent, projectionExtent, -projectionExtent, projectionExtent, -projectionExtent, projectionExtent);
        glMatrixMode(GL_MODELVIEW);

        return window;
    }

    void updateCamera() {
        Graphics::cameraAngle += Graphics::cameraSpeed;

        float camX = 0.25f * Graphics::cameraRadius * cos(cameraAngle);
        float camZ = 0.25f * Graphics::cameraRadius * sin(cameraAngle);
        float camY = 0.25f * Graphics::cameraRadius * sin(Graphics::cameraAngle); // Slightly tilt the camera
        glLoadIdentity();
        gluLookAt(camX, camY, camZ, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    }
}

struct Config {
    int particleCount = 0;
    int armCount = 3;
    // Number of separate galaxies to generate (spread around central core)
    int galaxyCount = 2;
    // Distance of the first galaxy center from origin (0 = centered)
    float galaxyRadius = 0.0f;
    // Radial spacing between successive galaxy centers
    float galaxyRadiusSpacing = 2000.0f;
    float diskRadius = 1000.0f;
    // Reduce disk thickness and angular jitter for a more stable initial disk
    float diskThickness = 50.0f;
    float armTurns = 0.5f;
    // Make arms only slightly denser than the background:
    // Increase the fraction of inter-arm particles so arms are only a bit
    // denser than the background, and broaden arm spread to avoid tight packing.
    // Aim for roughly ~52% in arms / 48% inter-arm.
    float armSpread = 1.5f;           // wider arms (less tight)
    float interArmFraction = 0.40f;   // ~52% in arms, 48% in inter-arm
    float interArmSpread = 0.8f;      // broader inter-arm jitter
    // Slightly under-speed particles to avoid initially super-virial system
    float speedScale = 0.98f;
    int coreMass = 50000;
    int particleMass = 50;
    // Increase softening length (length units) to avoid large central accelerations
    float orbitSoftening = 100.0f;
    float globalDt = 0.02f;
    float theta = 0.2f;
    float scale = 5000.0f;
    int windowWidth = 800;
    int windowHeight = 600;
    float projectionExtent = 2000.0f;
    float cameraRadius = 200.0f;
    float cameraSpeed = 0.001f;
    bool showTree = true;
    bool exportSnapshots = true;
    int exportEvery = 50;
    std::string exportFile = "simulation.hd5";
    bool useSeed = false;
    unsigned int seed = 0;
};

static bool parseInt(const char *value, int &out) {
    try {
        size_t idx = 0;
        int parsed = std::stoi(value, &idx);
        if (idx != std::string(value).size()) return false;
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

static bool parseUInt(const char *value, unsigned int &out) {
    try {
        size_t idx = 0;
        unsigned long parsed = std::stoul(value, &idx);
        if (idx != std::string(value).size()) return false;
        out = static_cast<unsigned int>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parseFloat(const char *value, float &out) {
    try {
        size_t idx = 0;
        float parsed = std::stof(value, &idx);
        if (idx != std::string(value).size()) return false;
        out = parsed;
        return true;
    } catch (...) {
        return false;
    }
}

static void printUsage(const char *exeName) {
    std::cout << "Usage: " << exeName << " [options]\n"
              << "  --particles N            Number of particles\n"
              << "  --export                 Enable snapshot export\n"
              << "  --no-export              Disable snapshot export\n"
              << "  --export-file PATH       Output .hd5/.h5 file path\n"
              << "  --orbit-softening S      Softening length\n"
              << "  --global-dt S            Global dt\n"
              << "  --theta T                Barnes-Hut opening angle\n"
              << "  --seed N                 Random seed (uint)\n"
              << "  --help                   Show this help\n";
}

static bool parseArgs(int argc, char **argv, Config &config, bool &particleCountProvided, bool &showHelp) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            showHelp = true;
            return true;
        } else if (arg == "--export") {
            config.exportSnapshots = true;
        } else if (arg == "--no-export") {
            config.exportSnapshots = false;
        } else if (arg == "--particles" && i + 1 < argc) {
            if (!parseInt(argv[++i], config.particleCount)) return false;
            particleCountProvided = true;
        } else if (arg == "--export-file" && i + 1 < argc) {
            config.exportFile = argv[++i];
        } else if (arg == "--orbit-softening" && i + 1 < argc) {
            if (!parseFloat(argv[++i], config.orbitSoftening)) return false;
        } else if (arg == "--galaxy-count" && i + 1 < argc) {
            if (!parseInt(argv[++i], config.galaxyCount)) return false;
        } else if (arg == "--galaxy-radius" && i + 1 < argc) {
            if (!parseFloat(argv[++i], config.galaxyRadius)) return false;
        } else if (arg == "--galaxy-radius-spacing" && i + 1 < argc) {
            if (!parseFloat(argv[++i], config.galaxyRadiusSpacing)) return false;
        } else if (arg == "--global-dt" && i + 1 < argc) {
            if (!parseFloat(argv[++i], config.globalDt)) return false;
        } else if (arg == "--theta" && i + 1 < argc) {
            if (!parseFloat(argv[++i], config.theta)) return false;
        } else if (arg == "--seed" && i + 1 < argc) {
            if (!parseUInt(argv[++i], config.seed)) return false;
            config.useSeed = true;
        } else {
            std::cerr << "Unknown or incomplete option: " << arg << "\n";
            return false;
        }
    }

    return true;
}


void applyGravitationalForce(std::shared_ptr<Particle> &p, const OctTree &tree, float theta) {
    tree.resolveForce(p, theta);
}

static float computeTreeScale(const std::vector<std::shared_ptr<Particle>> &particles, float minimumScale) {
    float maxAbsCoordinate = 1.0f;
    for (const auto &particle : particles) {
        const fVector3 position = particle->getPosition();
        maxAbsCoordinate = std::max(maxAbsCoordinate, std::abs(position.x));
        maxAbsCoordinate = std::max(maxAbsCoordinate, std::abs(position.y));
        maxAbsCoordinate = std::max(maxAbsCoordinate, std::abs(position.z));
    }
    return std::max(minimumScale, maxAbsCoordinate * 1.1f + 1.0f);
}

static std::unique_ptr<OctTree> computeForces(
    std::vector<std::shared_ptr<Particle>> &particles,
    float theta,
    float minimumScale,
    float &maxAcceleration,
    double *treeSeconds,
    double *forceSeconds
) {
    for (auto &particle : particles) {
        particle->zeroAcceleration();
    }

    const auto treeStart = std::chrono::high_resolution_clock::now();
    const float treeScale = computeTreeScale(particles, minimumScale);
    auto tree = std::make_unique<OctTree>(
        fVector3(-treeScale, -treeScale, -treeScale),
        fVector3(2.0f * treeScale, 2.0f * treeScale, 2.0f * treeScale)
    );
    for (const auto &particle : particles) {
        tree->insert(particle);
    }
    const auto treeEnd = std::chrono::high_resolution_clock::now();
    if (treeSeconds != nullptr) {
        *treeSeconds = std::chrono::duration<double>(treeEnd - treeStart).count();
    }

    const auto forceStart = std::chrono::high_resolution_clock::now();
    maxAcceleration = 0.0f;
    for (size_t i = 0; i < particles.size(); ++i) {
        if (!particles[i]) continue;
        applyGravitationalForce(particles[i], *tree, theta);
        const float acc = fVector3::magnitude(particles[i]->getAcceleration());
        maxAcceleration = std::max(maxAcceleration, acc);
    }
    const auto forceEnd = std::chrono::high_resolution_clock::now();
    if (forceSeconds != nullptr) {
        *forceSeconds = std::chrono::duration<double>(forceEnd - forceStart).count();
    }

    return tree;
}

int main(int argc, char **argv) {
    Config config;
    bool particleCountProvided = false;
    bool showHelp = false;

    // Parse command-line arguments
    if (!parseArgs(argc, argv, config, particleCountProvided, showHelp)) {
        printUsage(argv[0]);
        return 1;
    }
    if (showHelp) {
        return 0;
    }

    // If particle count was not provided, ask the user
    if (!particleCountProvided) {
        std::cout << "Enter the number of particles: ";
        std::cin >> config.particleCount;
    }
    if (config.particleCount <= 0) {
        std::cerr << "Particle count must be greater than 0.\n";
        return 1;
    }
    if (config.exportEvery <= 0) {
        std::cerr << "Export interval must be greater than 0.\n";
        return 1;
    }
    if (config.armCount <= 0) {
        std::cerr << "Arm count must be greater than 0.\n";
        return 1;
    }
    if (config.diskRadius <= 0.0f || config.scale <= 0.0f) {
        std::cerr << "Disk radius and scale must be greater than 0.\n";
        return 1;
    }
    if (config.globalDt <= 0.0f) {
        std::cerr << "global-dt must be greater than 0.\n";
        return 1;
    }

    Graphics::cameraRadius = config.cameraRadius;
    Graphics::cameraSpeed = config.cameraSpeed;

    GLFWwindow *window = Graphics::initGraphics(config.windowWidth, config.windowHeight, config.projectionExtent);
    if (window == nullptr) {
        return -1;
    }

    double sumChrono = 0.0;
    double sumTreeTime = 0.0;
    double sumForceTime = 0.0;
    int totalIterations = 0;

    const int particleCount = config.particleCount;
    const int armCount = config.armCount;
    const float diskRadius = config.diskRadius;
    const float diskThickness = config.diskThickness;
    const float armTurns = config.armTurns;
    const float armSpread = config.armSpread;
    const float interArmFraction = config.interArmFraction;
    const float interArmSpread = config.interArmSpread;
    const float speedScale = config.speedScale;
    const int coreMass = config.coreMass;
    const int particleMass = config.particleMass;
    const float orbitSoftening = config.orbitSoftening;
    const float globalDt = config.globalDt;
    const float theta = config.theta;


    std::vector<std::shared_ptr<Particle> > particles(particleCount);

    // Randomly generate particle positions
    std::random_device rd;
    std::mt19937 gen(config.useSeed ? config.seed : rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    std::uniform_real_distribution<> angleJitter(-armSpread, armSpread);
    std::uniform_real_distribution<> interArmJitter(-interArmSpread, interArmSpread);
    std::uniform_real_distribution<> thicknessJitter(-diskThickness * 0.5f, diskThickness * 0.5f);

    // Set the runtime softening (Plummer softening length squared) from config
    Physics::SofteningSquared = orbitSoftening * orbitSoftening;

    // Generate galaxies (fills particles including particles[0] core)
    Init::generateGalaxies(
        particles,
        armCount,
        config.galaxyCount,
        config.galaxyRadius,
        config.galaxyRadiusSpacing,
        diskRadius,
        diskThickness,
        armTurns,
        armSpread,
        interArmFraction,
        interArmSpread,
        speedScale,
        coreMass,
        particleMass,
        gen
    );


    std::unique_ptr<Hdf5SnapshotWriter> snapshotWriter;
    const float diagnosticsSofteningSquared = Physics::SofteningSquared;
    if (config.exportSnapshots) {
        snapshotWriter = std::make_unique<Hdf5SnapshotWriter>(
            config.exportFile,
            particles.size(),
            Physics::GravityConstant
        );
        std::cout << snapshotWriter->statusMessage() << std::endl;

        if (snapshotWriter->isEnabled()) {
            const Diagnostics::SystemDiagnostics initialDiagnostics = Diagnostics::computeSystemDiagnostics(
                particles,
                Physics::GravityConstant,
                diagnosticsSofteningSquared
            );
            ExportSnapshot initialSnapshot;
            initialSnapshot.step = 0;
            initialSnapshot.simulationTime = 0.0;
            snapshotWriter->writeSnapshot(initialSnapshot, initialDiagnostics, particles);
        }
    }

    double simulationTime = 0.0;
    float maxAccThisFrame = 0.0f;
    computeForces(particles, theta, config.scale, maxAccThisFrame, nullptr, nullptr);

    while (!glfwWindowShouldClose(window)) {
        totalIterations++;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < particleCount; ++i) {
            if (!particles[i]) continue;
            particles[i]->kick(0.5f * globalDt);
        }

        for (int i = 0; i < particleCount; ++i) {
            if (!particles[i]) continue;
            particles[i]->drift(globalDt);
        }

        double treeElapsedSeconds = 0.0;
        double forceElapsedSeconds = 0.0;
        std::unique_ptr<OctTree> tree = computeForces(
            particles,
            theta,
            config.scale,
            maxAccThisFrame,
            &treeElapsedSeconds,
            &forceElapsedSeconds
        );
        sumTreeTime += treeElapsedSeconds;
        sumForceTime += forceElapsedSeconds;

        for (int i = 0; i < particleCount; ++i) {
            if (!particles[i]) continue;
            particles[i]->kick(0.5f * globalDt);
        }

        Graphics::updateCamera();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Graphics::renderParticles(particles);
        if (config.showTree && tree) {
            tree->drawOutline();
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        const float totalDtThisFrame = globalDt;

        simulationTime += static_cast<double>(globalDt);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        sumChrono += elapsed.count();

        if (snapshotWriter && snapshotWriter->isEnabled() && (totalIterations % config.exportEvery == 0)) {
            const Diagnostics::SystemDiagnostics diagnostics = Diagnostics::computeSystemDiagnostics(
                particles,
                Physics::GravityConstant,
                diagnosticsSofteningSquared
            );

            ExportSnapshot snapshot;
            snapshot.step = totalIterations;
            snapshot.simulationTime = simulationTime;
            snapshot.treeSeconds = treeElapsedSeconds;
            snapshot.forceSeconds = forceElapsedSeconds;
            snapshot.iterationSeconds = elapsed.count();
            snapshot.averageDt = totalDtThisFrame;
            snapshot.maxAcceleration = maxAccThisFrame;

            snapshotWriter->writeSnapshot(snapshot, diagnostics, particles);
        }
    }

    std::cout << "Average tree construction time: " << (sumTreeTime / static_cast<double>(totalIterations)) * 1000.0 << "ms" << std::endl;
    std::cout << "Average force application time: " << (sumForceTime / static_cast<double>(totalIterations)) * 1000.0 << "ms" << std::endl;
    std::cout << "Average time per iteration: " << (sumChrono / static_cast<double>(totalIterations)) * 1000.0 << "ms" << std::endl;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
