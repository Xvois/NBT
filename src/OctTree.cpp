//
// Created by Sonny Parker on 06/02/2025.
//


#include "../include/OctTree.h"

#include <float.h>


void OctTree::computePsuedo() const {
    std::vector<PsuedoParticle> consideredPsuedos;

    if (divided) {
        for (const auto &child: children) {
            consideredPsuedos.push_back(child->getPsuedoParticle());
        }
    } else {
        for (const auto &particle: particles) {
            consideredPsuedos.push_back({particle->getPosition(), particle->getMass()});
        }
    }

    PsuedoParticle total = {fVector3::null(), 0};

    for (const auto &psuedo: consideredPsuedos) {
        total.position += psuedo.position * psuedo.mass;
        total.mass += psuedo.mass;
    }

    total.position /= total.mass;

    psuedo_ = total;
}

PsuedoParticle OctTree::getPsuedoParticle() const {
    if (psuedo_.mass == 0 && psuedo_.position == fVector3::NullVector) {
        computePsuedo();
    }
    return psuedo_;
}


void OctTree::recurseToTargets(const fVector3 &point, float theta, std::vector<const OctTree *> &validNodes) const {
    float quotientSquare = this->quotientSquare(point);
    if (!divided || quotientSquare < theta * theta) {
        validNodes.emplace_back(this);
        return;
    }

    for (const auto &child: children) {
        child->recurseToTargets(point, theta, validNodes);
    }
}

void OctTree::drawOutline() const {
    glEnable(GL_BLEND); // Enable blending
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); // Set the blend function

    glColor4f(1.0f, 0.0f, 0.0f, 0.2f); // Set color to red with full opacity
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