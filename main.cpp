#include <iostream>
#include <random>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>

#define SCALE 1000

#include "Particle.h"
#include "OctTree.h"


/*
 * 1000 - Base 224.027ms
 * 1000 - Reserve - 187.878ms
 * 1000 - // + QuotientSqr - 166.364ms
 * 1000 - // + Vector class fixes - 152.871ms
 * 1000 - // + Multithreading - 39.4286ms
*/
float cameraAngle = 0.0f;
const float cameraRadius = 1000.0f;
void renderParticles(const std::vector<std::shared_ptr<Particle>> &particles) {
    glEnable(GL_BLEND); // Enable blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Set the blend function

    glBegin(GL_POINTS);
    for (size_t i = 0; i < particles.size(); ++i) {
        if (i == 0) {
            glColor4f(0.0f, 0.0f, 1.0f, 0.5f); // Set color to blue with transparency for the first particle
        } else {
            glColor4f(1.0f, 1.0f, 1.0f, 0.5f); // Set color to white with transparency for the rest
        }
        fVector3 pos = particles[i]->getPosition();
        glVertex3f(pos.x, pos.y, pos.z);
    }
    glEnd();

    glDisable(GL_BLEND); // Disable blending
}

GLFWwindow *initGraphics() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    GLFWwindow *window = glfwCreateWindow(800, 600, "Particle Simulation", nullptr, nullptr);
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
    glPointSize(1.0f);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2000.0, 2000.0, -2000.0, 2000.0, -2000.0, 2000.0); // Adjusted to see further out
    glMatrixMode(GL_MODELVIEW);

    return window;
}

void applyGravitationalForce(std::shared_ptr<Particle> p1, std::shared_ptr<Particle> p2) {
    fVector3 direction = p2->getPosition() - p1->getPosition();
    float d2 = fVector3::magnitudeSquare(direction);
    fVector3 norm = fVector3::norm(direction);
    float force = 10.0f * p1->getMass() * p2->getMass() / (d2 + 1);
    p1->impulse(norm * force);
    p2->impulse(norm * -force);
}

void applyGravitationalForce(std::shared_ptr<Particle> &p, const OctTree &tree, int reserveSize) {
    constexpr float theta = 0.2;
    std::vector<const OctTree *> nodes;
    nodes.reserve(reserveSize); // Reserve space based on O(n log n)
    tree.recurseToTargets(p->getPosition(), theta, nodes);
    for (const auto *node: nodes) {
        PsuedoParticle psuedoParticle = node->getPsuedoParticle();
        fVector3 direction = psuedoParticle.position - p->getPosition();
        float d2 = fVector3::magnitudeSquare(direction);
        if (d2 > 1) {
            fVector3 norm = fVector3::norm(direction);
            float force = 10.0f * p->getMass() * psuedoParticle.mass / (d2 + 1);
            p->impulse(norm * force);
        }
    }
}

// Function to apply gravitational force in parallel
void applyGravitationalForceParallel(std::vector<std::shared_ptr<Particle>> &particles, const OctTree &tree, int reserveSize, int start, int end) {
    for (int i = start; i < end; ++i) {
        applyGravitationalForce(particles[i], tree, reserveSize);
    }
}



void updateCamera() {
    static float cameraRadius = 1000.0f;
    static float radiusChange = 10.0f;

    cameraAngle += 0.01f; // Adjust the speed of rotation as needed
    cameraRadius += radiusChange; // Vary the camera radius

    // Reverse the direction of radius change if it goes out of bounds
    if (cameraRadius > 1500.0f || cameraRadius < 500.0f) {
        radiusChange = -radiusChange;
    }

    float camX = cameraRadius * cos(cameraAngle);
    float camZ = cameraRadius * sin(cameraAngle);
    float camY = 0.25 * cameraRadius * sin(cameraAngle / 5); // Slightly tilt the camera
    glLoadIdentity();
    gluLookAt(camX, camY, camZ, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
}

int main() {
    GLFWwindow *window = initGraphics();
    if (window == nullptr) {
        return -1;
    }

    float sumChrono = 0;
    float sumTreeTime = 0;
    float sumForceTime = 0;
    int totalIterations = 0;

    int particleCount;
    std::cout << "Enter the number of particles: ";
    std::cin >> particleCount;

    // Tree traversal reserve size
    int reserveSize = static_cast<int>(0.543 * particleCount * std::log(particleCount));


    std::vector<std::shared_ptr<Particle> > particles(particleCount);

    // Randomly generate particle positions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < particleCount; i++) {
        particles[i] = std::make_shared<Particle>();
        particles[i]->setPosition(fVector3::random(gen, dis, 500));
        particles[i]->setVelocity(fVector3::random(gen, dis, 10));
        particles[i]->setMass(1.0f + dis(gen) * 10.0);
    }

    while (!glfwWindowShouldClose(window)) {
        totalIterations++;
        auto start = std::chrono::high_resolution_clock::now();

        auto treeStart = std::chrono::high_resolution_clock::now();
        OctTree tree(fVector3(-SCALE, -SCALE, -SCALE), fVector3(2*SCALE, 2*SCALE, 2*SCALE));

        for (int i = 0; i < particleCount; i++) {
            tree.insert(particles[i]);
        }

        auto treeEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> treeElapsed = treeEnd - treeStart;
        sumTreeTime += treeElapsed.count();

        auto forceStart = std::chrono::high_resolution_clock::now();


        for (int i = 0; i < particleCount; ++i) {
            applyGravitationalForce(particles[i], tree, reserveSize);
        }

        auto forceEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> forceElapsed = forceEnd - forceStart;
        sumForceTime += forceElapsed.count();



        updateCamera();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderParticles(particles);
        //tree.drawOutline();

        glfwSwapBuffers(window);
        glfwPollEvents();

        for (int i = 0; i < particleCount; i++) {
            particles[i]->integrate(0.1f);
            particles[i]->zeroAcceleration();
            fVector3 pos = particles[i]->getPosition();
            if (fVector3::magnitude(pos) > SCALE) {

                std::pair<fVector3, float> closestPlane = tree.closestPlane(pos);
                // SYSTEM IS SYMMETRIC CENTRED AT 0,0,0
                particles[i]->setPosition(pos*-1 + closestPlane.first * closestPlane.second);
            }
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        sumChrono += elapsed.count();

    }

    std::cout << "Average tree construction time: " << (sumTreeTime / totalIterations) * 1000 << "ms" << std::endl;
    std::cout << "Average force application time: " << (sumForceTime / totalIterations) * 1000 << "ms" << std::endl;
    std::cout << "Average time per iteration: " << (sumChrono / totalIterations) * 1000 << "ms" << std::endl;

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
