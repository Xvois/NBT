// fVector3.h
#ifndef VECTOR_H
#define VECTOR_H

#include <ctgmath>
#include <random>

class fVector3 {
public:
    float x, y, z;

    // Constructors
    fVector3(float x, float y, float z) : x(x), y(y), z(z) {
    }

    fVector3() : x(0), y(0), z(0) {
    }

    fVector3(const fVector3 &v) = default;

    // Operators

    bool operator==(const fVector3& v) {
        return this->x == v.x && this->y == v.y && this->z == v.z;
    };

    fVector3 &operator*=(float scaler) {
        x *= scaler;
        y *= scaler;
        z *= scaler;
        return *this;
    }

    fVector3 &operator/=(float scaler) {
        x /= scaler;
        y /= scaler;
        z /= scaler;
        return *this;
    }

    fVector3 &operator+=(const fVector3 &v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    fVector3 &operator-=(const fVector3 &v) {
        x -= v.x;
        y -= v.y;
        z -= v.z;
        return *this;
    }

    fVector3 operator+(const fVector3 &v) const {
        return {x + v.x, y + v.y, z + v.z};
    }

    fVector3 operator-(const fVector3 &v) const {
        return {x - v.x, y - v.y, z - v.z};
    }

    fVector3 operator*(float scalar) const {
        return {x * scalar, y * scalar, z * scalar};
    }

    fVector3 operator/(float scalar) const {
        return {x / scalar, y / scalar, z / scalar};
    }

    bool operator==(const fVector3 &v) const {
        return x == v.x && y == v.y && z == v.z;
    }

    // Static methods
    static float magnitude(const fVector3 &v);

    static float dot(const fVector3 &v1, const fVector3 &v2);

    static float magnitudeSquare(const fVector3 &v);

    static fVector3 norm(const fVector3 &v);

    static fVector3 random(std::mt19937 &gen, std::uniform_real_distribution<> &dis, float radius);

    // Declare static variable (no definition here)
    static const fVector3 NullVector;
};


#endif // VECTOR_H
