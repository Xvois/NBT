//
// Created by Sonny Parker on 06/02/2025.
//

#ifndef OCTTREE_H
#define OCTTREE_H
#include <GLFW/glfw3.h>

#include "Particle.h"
#include "fVector3.h"
#include <array>
#include <memory>
#include <optional>
#include <vector>

/// Represents a single node in the octree
class OctTree {
    static constexpr int capacity = 1; // Compile-time constant

    fVector3 position;   // The position of the node
    fVector3 dimension;  // The dimensions of the node
    fVector3 centre;   // The centre of the node

    bool divided = false;  // Whether the node has been divided

    std::vector<std::shared_ptr<Particle>> particles; // Optimized to shared_ptr
    std::array<std::unique_ptr<OctTree>, 8> children; // Preallocated array for children

    mutable PsuedoParticle psuedo_ = PsuedoParticle{}; // Cached pseudo particle

    [[nodiscard]] void computePsuedo() const; // Internal method to compute psudo particle

public:
    // Constructor with correct centre initialization
    OctTree(const fVector3& position, const fVector3& dimension)
        : position(position), dimension(dimension), centre(position + dimension / 2) {
        particles.reserve(capacity); // Reserve space to avoid frequent reallocations
    }

    // Remove copy constructor and assignment operator
    OctTree(const OctTree&) = delete;
    OctTree& operator=(const OctTree&) = delete;

    // Getters
    [[nodiscard]] fVector3 getPosition() const { return position; }
    [[nodiscard]] fVector3 getDimension() const { return dimension; }
    [[nodiscard]] fVector3 getCentre() const { return centre; } // Getter for centre
    [[nodiscard]] bool isDivided() const { return divided; }
    [[nodiscard]] const std::array<std::unique_ptr<OctTree>, 8>& getChildren() const { return children; }
    [[nodiscard]] PsuedoParticle getPsuedoParticle() const;


    // Subdivide the node into 8 children
    void subdivide() {
        if (divided) return;

        fVector3 halfSize = dimension / 2;
        fVector3 offsets[8] = {
            {0, 0, 0}, {halfSize.x, 0, 0}, {0, halfSize.y, 0}, {halfSize.x, halfSize.y, 0},
            {0, 0, halfSize.z}, {halfSize.x, 0, halfSize.z}, {0, halfSize.y, halfSize.z}, {halfSize.x, halfSize.y, halfSize.z}
        };

        for (int i = 0; i < 8; ++i) {
            children[i] = std::make_unique<OctTree>(position + offsets[i], halfSize);
        }

        this->divided = true;

        // Redistribute particles among children
        for (const auto& p : particles) {
            for (auto& child : children) {
                if (child->insert(p)) {
                    break;
                }
            }
        }

        particles.clear(); // Clear particles from the current node
    }

    // Efficient insert method
    bool insert(const std::shared_ptr<Particle>& p) {
        if (!contains(p->getPosition())) return false; // Early exit if not contained

        if (particles.size() < capacity) {
            particles.push_back(p);
            return true;
        }

        if (!divided) subdivide();

        for (auto& child : children) {
            if (child->insert(p)) return true;
        }

        return false; // Should never reach here
    }

    // Check if a point is within the node
    [[nodiscard]] bool contains(const fVector3 &point) const {
        return (point.x >= position.x && point.x <= position.x + dimension.x &&
                point.y >= position.y && point.y <= position.y + dimension.y &&
                point.z >= position.z && point.z <= position.z + dimension.z);
    }

    [[nodiscard]] std::pair<fVector3, float> closestPlane(const fVector3 &point) const {
        float minDistance = std::numeric_limits<float>::max();
        fVector3 closestPlaneNormal;

        // Define the normals for the six planes
        std::array<fVector3, 6> normals = {
            fVector3(-1, 0, 0), fVector3(1, 0, 0), // Left and Right
            fVector3(0, -1, 0), fVector3(0, 1, 0), // Bottom and Top
            fVector3(0, 0, -1), fVector3(0, 0, 1)  // Back and Front
        };

        // Define the positions of the six planes
        std::array<fVector3, 6> planePositions = {
            position, position + fVector3(dimension.x, 0, 0), // Left and Right
            position, position + fVector3(0, dimension.y, 0), // Bottom and Top
            position, position + fVector3(0, 0, dimension.z)  // Back and Front
        };

        // Calculate the distance to each plane
        for (int i = 0; i < 6; ++i) {
            float distance = std::abs(fVector3::dot(point - planePositions[i], normals[i]));
            if (distance < minDistance) {
                minDistance = distance;
                closestPlaneNormal = normals[i];
            }
        }

        return {closestPlaneNormal, minDistance};
    }

    [[nodiscard]] float quotientSquare(const fVector3 &point) const {
        float xdiff = (centre.x - point.x);
        float ydiff = (centre.y - point.y);
        float zdiff = (centre.z - point.z);
        float distanceSquared = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff;
        return dimension.x * dimension.x / distanceSquared;
    }


    // Resolve the force acting on a particle
    void resolveForce(std::shared_ptr<Particle> p, float theta) const;


    void drawOutline() const; // Draw the node
    void drawShaded() const;
};

#endif //OCTTREE_H