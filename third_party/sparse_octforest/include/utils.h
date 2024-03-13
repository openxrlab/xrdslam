#ifndef _UTILS_H
#define _UTILS_H

#include <iostream>
#include <eigen3/Eigen/Dense>

#define MAX_BITS 21
typedef Eigen::Matrix<uint64_t, 1, 3> Vector3lu;

constexpr uint64_t MASK[] = {
    0x7000000000000000, // 0111 0000 0000 0000 0000 ...
    0x7e00000000000000, // 0111 1110 0000 0000 0000 ...
    0x7fc0000000000000, // 0111 1111 1100 0000 0000 ...
    0x7ff8000000000000, // 0111 1111 1111 1000 0000 ...
    0x7fff000000000000,
    0x7fffe00000000000,
    0x7ffffc0000000000,
    0x7fffff8000000000,
    0x7ffffff000000000,
    0x7ffffffe00000000,
    0x7fffffffc0000000,
    0x7ffffffff8000000,
    0x7fffffffff000000,
    0x7fffffffffe00000,
    0x7ffffffffffc0000,
    0x7fffffffffff8000,
    0x7ffffffffffff000,
    0x7ffffffffffffe00,
    0x7fffffffffffffc0,
    0x7ffffffffffffff8,
    0x7fffffffffffffff
    };

inline uint64_t expand(unsigned long long value)
{
    uint64_t x = value & 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

inline uint64_t compact(uint64_t value)
{
    uint64_t x = value & 0x1249249249249249; // 0001 0010 0100 1001 0010 ...
    x = (x | x >> 2) & 0x10c30c30c30c30c3; // 0001 0000 1100 0011 0000 ...
    x = (x | x >> 4) & 0x100f00f00f00f00f; // 0001 0000 0000 1111 0000
    x = (x | x >> 8) & 0x1f0000ff0000ff;
    x = (x | x >> 16) & 0x1f00000000ffff;
    x = (x | x >> 32) & 0x1fffff;
    return x;
}

inline uint64_t compute_morton(uint64_t x, uint64_t y, uint64_t z)
{
    uint64_t code = 0;

    x = expand(x);
    y = expand(y) << 1;
    z = expand(z) << 2;

    code = x | y | z;
    return code;
}

inline Vector3lu decode(const uint64_t code)
{
    return {
        compact(code >> 0ull),
        compact(code >> 1ull),
        compact(code >> 2ull)};
}

inline uint64_t encode(const int x, const int y, const int z)
{
    return (compute_morton(x, y, z) & MASK[MAX_BITS - 1]);
}

#endif
