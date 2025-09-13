#include <cmath>
#include <cstdint>

struct mwc256xxa64 {
    using u64 = uint64_t;
    using u128 = unsigned __int128;

    // Default constructor: fixed seed initial state
    u64 x2 = 12345ull;
    u64 x3 = 0xcafef00dd15ea5e5ull;
    u128 c_x1 = (u128(0x1405'7B7E'F767'814Full) << 64) | 23456ull;
    mwc256xxa64() = default;

    // Internal constants: multiplier for MWC and reciprocal of 2^64 for [0,1) scaling
    static constexpr u64 MULT = 0xfeb3'4465'7c0af413ull;
    static constexpr double INV_2POW64 = 1.0 / 18446744073709551616.0;

    // Generate next 64-bit random number
    __attribute__((always_inline)) inline u64 next() {
        u128 x = u128(x3) * MULT;
        u64 result = (x3 ^ x2) + (static_cast<u64>(c_x1) ^ static_cast<u64>(x >> 64));
        // shift state
        x3 = x2;
        x2 = static_cast<u64>(c_x1);
        c_x1 = x + (c_x1 >> 64);
        return result;
    }

    // Integer in [0, m)
    __attribute__((always_inline)) inline u64 nextInt(u64 m) { return u64((u128(next()) * u128(m)) >> 64); }

    // Double in [0, 1)
    __attribute__((always_inline)) inline double nextDouble() { return next() * INV_2POW64; }

    // Log of uniform [0,1)
    inline double nextLog() { return std::log(nextDouble()); }
};
static thread_local mwc256xxa64 rnd64;