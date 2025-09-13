#include <cmath>
#include <cstdint>

struct mwc128xxa32 {
    using u32 = uint32_t;
    using u64 = uint64_t;

    // Default constructor: fixed seed initial state
    u32 x2 = 12345u;
    u32 x3 = 0xcafef00du;
    u64 c_x1 = (u64(0xd15ea5e5) << 32) | 23456u;
    mwc128xxa32() = default;

    // Internal constants: multiplier for MWC and reciprocal of 2^32 for [0,1) scaling
    static constexpr u32 MULT = 3487286589u;
    static constexpr double INV_2POW32 = 1.0 / 4294967296.0;

    // Generate next 32-bit random number
    __attribute__((always_inline)) inline u32 next() {
        u64 x = u64(x3) * MULT;
        u32 result = (x3 ^ x2) + (static_cast<u32>(c_x1) ^ static_cast<u32>(x >> 32));
        // shift state
        x3 = x2;
        x2 = static_cast<u32>(c_x1);
        c_x1 = x + (c_x1 >> 32);
        return result;
    }

    // Integer in [0, m)
    __attribute__((always_inline)) inline int nextInt(int m) { return int((u64(next()) * u64(m)) >> 32); }

    // Double in [0, 1)
    __attribute__((always_inline)) inline double nextDouble() { return next() * INV_2POW32; }

    // Log of uniform [0,1)
    inline double nextLog() { return std::log(nextDouble()); }
};
static thread_local mwc128xxa32 rnd32;