//
// Created by Sonny Parker on 06/02/2025.
//


#include "../include/OctTree.h"

#include <float.h>
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
    float quotientSquare = this->quotientSquare(p->getPosition());

    if (quotientSquare < theta * theta) {
        // This is a valid approximation
        fVector3 direction = pseudo_.position - p->getPosition();
        float d2 = fVector3::magnitudeSquare(direction);
        fVector3 norm = fVector3::norm(direction);
        if (d2 > 0.1) {
            float force = 10.0f * static_cast<float>(p->getMass()) * static_cast<float>(pseudo_.mass) / (d2 + 1);
            p->impulse(norm * force);
        }
    } else if (divided) {
        // Not a valid approximation, consider children
        for (const auto &child: children) {
            child->resolveForce(p, theta);
        }
    } else if (particle) {
        // Not a valid approximation, but best we can do
        fVector3 direction = particle->getPosition() - p->getPosition();
        float d2 = fVector3::magnitudeSquare(direction);
        fVector3 norm = fVector3::norm(direction);
        if (d2 > 0.1) {
            float force = 10.0f * static_cast<float>(p->getMass()) * static_cast<float>(particle->getMass()) / (d2 + 1);
            p->impulse(norm * force);
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
