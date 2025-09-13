#include <cstdlib>
#include <sstream>
#include <string>
#include <iostream>

namespace hp {

#ifndef HP_ENV_PREFIX
#define HP_ENV_PREFIX "HP_"
#endif

template <typename T>
inline T from_env(const char* name, T fallback) {
    const char* s = std::getenv(name);
    if (!s) return fallback;
    std::stringstream ss(s);
    T v; ss >> v; return ss.fail() ? fallback : v;
}

template <typename T>
inline T clamp_range(const char* name, T v, T lo, T hi) {
    if (v < lo) { std::cerr << "[hp] " << name << " clamped to " << lo << " from " << v << "\n"; return lo; }
    if (v > hi) { std::cerr << "[hp] " << name << " clamped to " << hi << " from " << v << "\n"; return hi; }
    return v;
}

} // namespace hp

#if defined(ONLINE_JUDGE)
#define HP_PARAM(type, name, def, lo, hi) \
    static_assert((def) >= (lo) && (def) <= (hi), "default out of range"); \
    constexpr type name = (def)
#else
#define HP_PARAM(type, name, def, lo, hi) \
    static_assert((def) >= (lo) && (def) <= (hi), "default out of range"); \
    static type name = hp::clamp_range<type>(#name, hp::from_env<type>(HP_ENV_PREFIX #name, (def)), (lo), (hi))
#endif

