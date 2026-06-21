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

// HP_PARAM:   Optuna の探索対象になるチューニングパラメータ(範囲付き)
// META_PARAM: 探索対象にはしない切替フラグなど。カテゴリ別展開(meta_params.hpp)では
//             カテゴリごとに値を差し替えられる
//
// USE_META_PARAMS を hp_params.hpp の include 前に定義すると、ONLINE_JUDGE ビルドでも
// 可変グローバルとして宣言され、gen_meta_params.py が生成する meta::apply_params() で
// 実行時にカテゴリ別の値を代入できる。
#if defined(ONLINE_JUDGE) && !defined(USE_META_PARAMS)
#define HP_PARAM(type, name, def, lo, hi) \
    static_assert((def) >= (lo) && (def) <= (hi), "default out of range"); \
    constexpr type name = (def)
#define META_PARAM(type, name, def) constexpr type name = (def)
#elif defined(ONLINE_JUDGE)
#define HP_PARAM(type, name, def, lo, hi) \
    static_assert((def) >= (lo) && (def) <= (hi), "default out of range"); \
    static type name = (def)
#define META_PARAM(type, name, def) static type name = (def)
#else
#define HP_PARAM(type, name, def, lo, hi) \
    static_assert((def) >= (lo) && (def) <= (hi), "default out of range"); \
    static type name = hp::clamp_range<type>(#name, hp::from_env<type>(HP_ENV_PREFIX #name, (def)), (lo), (hi))
#define META_PARAM(type, name, def) \
    static type name = hp::from_env<type>(HP_ENV_PREFIX #name, (def))
#endif
