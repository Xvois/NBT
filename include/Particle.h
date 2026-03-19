//
// Created by Sonny Parker on 04/02/2025.
//

#ifndef PARTICLE_H
#define PARTICLE_H
#include "fVector3.h"

/// A struct to represent an imaginary particle that is used to calculate the gravitational force
/// in a tree structure.
struct PseudoParticle {
    fVector3 position;
    int mass;
};

class Particle {
    fVector3 position;
    fVector3 velocity;
    fVector3 acceleration;
    int mass;

public:
    Particle(fVector3 position, fVector3 velocity, fVector3 acceleration, int mass) : position(position), velocity(velocity), acceleration(acceleration), mass(mass) {};
    Particle(fVector3 position, fVector3 velocity, int mass) : position(position), velocity(velocity), acceleration(fVector3()), mass(mass) {};
    Particle() : position(fVector3()), velocity(fVector3()), acceleration(fVector3()), mass(1) {};

    void setPosition(fVector3 position) {
        this->position = position;
    }

    [[nodiscard]] fVector3 getPosition() const {
        return position;
    }

    void setVelocity(fVector3 velocity) {
        this->velocity = velocity;
    }

    [[nodiscard]] fVector3 getVelocity() const {
        return velocity;
    }

    [[nodiscard]] fVector3 getAcceleration() const {
        return acceleration;
    }

    void setMass(int mass) {
        this->mass = mass;
    }

    [[nodiscard]] int getMass() const {
        return mass;
    }

    void kick(float timeStep);
    void drift(float timeStep);
    void zeroAcceleration();
    void impulse(fVector3 force);
};

#endif //PARTICLE_H
