//
// Created by Sonny Parker on 06/02/2025.
//


#include "../include/OctTree.h"

#include <float.h>
#include <cmath>
#include <GL/glew.h>

void OctTree::computePseudo() const {
    std::vector<PseudoParticle> consideredPseudos;

    if (divided) {
        for (const auto &child: children) {
            PseudoParticle childPsuedo = child->getPseudoParticle();
            if (childPsuedo.mass > 0) {
                consideredPseudos.push_back(childPsuedo);
            }
        }
    }

    if (particle) {
        consideredPseudos.push_back({particle->getPosition(), particle->getMass()});
    }

    PseudoParticle total = {fVector3::NullVector, 0};

    for (const auto &psuedo: consideredPseudos) {
        total.position += psuedo.position * static_cast<float>(psuedo.mass);
        total.mass += psuedo.mass;
    }

    if (total.mass > 0) {
        total.position /= static_cast<float>(total.mass);
    }

    pseudo_ = total;
}

PseudoParticle OctTree::getPseudoParticle() const {
    if (pseudo_.mass == 0 && pseudo_.position == fVector3::NullVector) {
        computePseudo();
    }
    return pseudo_;
}


void OctTree::resolveForce(const std::shared_ptr<Particle>& p, float theta) const {
    const fVector3 position = p->getPosition();
    const float thetaSquare = theta * theta;
    const float quotientSquare = this->quotientSquare(position);
    const bool containsTarget = contains(position);

    if (!containsTarget && quotientSquare < thetaSquare) {
        const PseudoParticle pseudo = getPseudoParticle();
        const fVector3 direction = pseudo.position - position;
        const float d2 = fVector3::magnitudeSquare(direction);
        if (d2 > 0.0f && pseudo.mass > 0) {
            const float softened = d2 + Physics::SofteningSquared;
            const float invDistance = 1.0f / std::sqrt(softened);
            const float invDistanceCubed = invDistance * invDistance * invDistance;
            const float scale = Physics::GravityConstant * static_cast<float>(p->getMass()) * static_cast<float>(pseudo.mass) * invDistanceCubed;
            p->impulse(direction * scale);
        }
        return;
    }

    if (divided) {
        for (const auto &child: children) {
            child->resolveForce(p, theta);
        }
        return;
    }

    if (particle && particle.get() != p.get()) {
        const fVector3 direction = particle->getPosition() - position;
        const float d2 = fVector3::magnitudeSquare(direction);
        if (d2 > 0.0f) {
            const float softened = d2 + Physics::SofteningSquared;
            const float invDistance = 1.0f / std::sqrt(softened);
            const float invDistanceCubed = invDistance * invDistance * invDistance;
            const float scale = Physics::GravityConstant * static_cast<float>(p->getMass()) * static_cast<float>(particle->getMass()) * invDistanceCubed;
            p->impulse(direction * scale);
        }
    }
}

void OctTree::drawOutline() const {
    glEnable(GL_BLEND); // Enable blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Set the blend function

    glColor4f(1.0f, 0.0f, 0.0f, 0.1f); // Set color to red with full opacity
    glBegin(GL_LINES); // Start drawing lines

    // Draw the outline of the node
    glVertex3f(position.x, position.y, position.z);
    glVertex3f(position.x + dimension.x, position.y, position.z);

    glVertex3f(position.x + dimension.x, position.y, position.z);
    glVertex3f(position.x + dimension.x, position.y + dimension.y, position.z);

    glVertex3f(position.x + dimension.x, position.y + dimension.y, position.z);
    glVertex3f(position.x, position.y + dimension.y, position.z);

    glVertex3f(position.x, position.y + dimension.y, position.z);
    glVertex3f(position.x, position.y, position.z);

    glVertex3f(position.x, position.y, position.z + dimension.z);
    glVertex3f(position.x + dimension.x, position.y, position.z + dimension.z);

    glVertex3f(position.x + dimension.x, position.y, position.z + dimension.z);
    glVertex3f(position.x + dimension.x, position.y + dimension.y, position.z + dimension.z);

    glVertex3f(position.x + dimension.x, position.y + dimension.y, position.z + dimension.z);
    glVertex3f(position.x, position.y + dimension.y, position.z + dimension.z);

    glVertex3f(position.x, position.y + dimension.y, position.z + dimension.z);
    glVertex3f(position.x, position.y, position.z + dimension.z);

    glVertex3f(position.x, position.y, position.z);
    glVertex3f(position.x, position.y, position.z + dimension.z);

    glVertex3f(position.x + dimension.x, position.y, position.z);
    glVertex3f(position.x + dimension.x, position.y, position.z + dimension.z);

    glVertex3f(position.x + dimension.x, position.y + dimension.y, position.z);
    glVertex3f(position.x + dimension.x, position.y + dimension.y, position.z + dimension.z);

    glVertex3f(position.x, position.y + dimension.y, position.z);
    glVertex3f(position.x, position.y + dimension.y, position.z + dimension.z);

    glEnd(); // End drawing lines

    glDisable(GL_BLEND); // Disable blending

    // Recursively draw the children
    if (divided) {
        for (const auto &child: children) {
            child->drawOutline();
        }
    }
}
