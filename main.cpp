#include <iostream>
#include <random>
#include <vector>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <chrono>
#include <thread>

#define SCALE 1250

#include "Particle.h"
#include "OctTree.h"


/*
 * 1000 - Base 224.027ms
 * 1000 - Reserve - 187.878ms
 * 1000 - // + QuotientSqr - 166.364ms
 * 1000 - // + Vector class fixes - 152.871ms
 * 1000 - Force resolution in tree - 120.912ms
 * 1000 - // + Hierarchical pseudos - 89.0132ms
*/

namespace Graphics {
    float cameraAngle = 0.0f;
    float cameraRadius = 200.0f;

    void renderParticles(const std::vector<std::shared_ptr<Particle> > &particles) {
        glEnable(GL_BLEND); // Enable blending
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Set the blend function

        for (size_t i = 0; i < particles.size(); ++i) {
            if (i == 0) {
                glColor4f(0.0f, 0.0f, 1.0f, 0.5f); // Set color to blue with transparency for the first particle
            } else {
                glColor4f(1.0f, 1.0f, 1.0f, 0.5f); // Set color to white with transparency for the rest
            }
            fVector3 pos = particles[i]->getPosition();
            float size = std::sqrt(particles[i]->getMass());
            glPointSize(size);
            glBegin(GL_POINTS);
            glVertex3f(pos.x, pos.y, pos.z);
            glEnd();
        }

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

    void updateCamera() {
        Graphics::cameraAngle += 0.01f; // Adjust the speed of rotation as needed

        float camX = 0.25f * Graphics::cameraRadius * cos(cameraAngle);
        float camZ = 0.25f * Graphics::cameraRadius * sin(cameraAngle);
        float camY = 0.25f * Graphics::cameraRadius * sin(Graphics::cameraAngle); // Slightly tilt the camera
        glLoadIdentity();
        gluLookAt(camX, camY, camZ, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
    }
}


void applyGravitationalForce(std::shared_ptr<Particle> &p, const OctTree &tree) {
    constexpr float theta = 0.2f;
    tree.resolveForce(p, theta);
}

int main() {
    GLFWwindow *window = Graphics::initGraphics();
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


    std::vector<std::shared_ptr<Particle> > particles(particleCount);

    // Randomly generate particle positions
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);

    particles[0] = std::make_shared<Particle>(fVector3(0, 0, 0), fVector3(0, 0, 0), 10000);

    // Orbiting particles
    for (int i = 1; i < particleCount; i++) {
        float angle = dis(gen) * 2 * M_PI;
        float distance = 500 + dis(gen) * 200; // Distance from the center
        float height = (dis(gen) - 0.5) * 100; // Small height variation

        fVector3 position(distance * cos(angle), height, distance * sin(angle));
        float speed = std::sqrt(10.0f * static_cast<float>(particles[0]->getMass()) / distance);
        fVector3 velocity(-speed * sin(angle), 0, speed * cos(angle));

        particles[i] = std::make_shared<Particle>(position, velocity, 1.0f);
    }

    while (!glfwWindowShouldClose(window)) {
        totalIterations++;
        auto start = std::chrono::high_resolution_clock::now();

        auto treeStart = std::chrono::high_resolution_clock::now();
        OctTree tree(fVector3(-SCALE, -SCALE, -SCALE), fVector3(2 * SCALE, 2 * SCALE, 2 * SCALE));

        for (int i = 0; i < particleCount; i++) {
            tree.insert(particles[i]);
        }

        auto treeEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> treeElapsed = treeEnd - treeStart;
        sumTreeTime += treeElapsed.count();

        auto forceStart = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < particleCount; i++) {
            applyGravitationalForce(particles[i], tree);
        }

        auto forceEnd = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> forceElapsed = forceEnd - forceStart;
        sumForceTime += forceElapsed.count();

        Graphics::updateCamera();
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        Graphics::renderParticles(particles);
        //tree.drawOutline();

        glfwSwapBuffers(window);
        glfwPollEvents();

        for (int i = 0; i < particleCount; i++) {
            particles[i]->integrate(0.2f);
            particles[i]->zeroAcceleration();
            fVector3 pos = particles[i]->getPosition();
            if (fVector3::magnitude(pos) > SCALE) {
                // SYSTEM IS SYMMETRIC CENTRED AT 0,0,0
                particles[i]->setPosition(pos * -1);
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
