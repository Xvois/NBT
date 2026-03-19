//
// Created by Sonny Parker on 04/02/2025.
//

#include "Particle.h"

void Particle::kick(const float timeStep) {
    velocity = velocity + acceleration * timeStep;
}

void Particle::drift(const float timeStep) {
    position = position + velocity * timeStep;
}

void Particle::zeroAcceleration() {
    acceleration.x = 0; acceleration.y = 0; acceleration.z = 0;
}

void Particle::impulse(const fVector3 force) {
    acceleration += force / static_cast<float>(mass);
}