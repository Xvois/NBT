// fVector3.cpp
#include "../include/fVector3.h"

const fVector3 fVector3::NullVector(0, 0, 0);

// Implement static methods
float fVector3::magnitude(const fVector3 &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

float fVector3::dot(const fVector3 &v1, const fVector3 &v2) {
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

float fVector3::magnitudeSquare(const fVector3 &v) {
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

fVector3 fVector3::norm(const fVector3 &v) {
    float mag = std::sqrt(magnitudeSquare(v));
    return (mag > 0) ? fVector3(v.x / mag, v.y / mag, v.z / mag) : fVector3(0, 0, 0);
}

fVector3 fVector3::random(std::mt19937 &gen, std::uniform_real_distribution<> &dis, float radius) {
    float r = radius * dis(gen);
    float theta = 2 * M_PI * dis(gen);
    float phi = std::acos(2 * dis(gen) - 1);

    float x = r * std::sin(phi) * std::cos(theta);
    float y = r * std::sin(phi) * std::sin(theta);
    float z = r * std::cos(phi);

    return fVector3(x, y, z);
}

fVector3 fVector3::cross(const fVector3 &a, const fVector3 &b) {
    return fVector3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}
