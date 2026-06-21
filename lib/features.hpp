#pragma once
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <utility>

namespace ahc {

// テストケースの特徴量を C++ 側で定義するためのヘルパー。
//
// 入力を読み終えた直後に呼ぶ:
//     ahc::emit_features({{"N", (double)N}, {"W", (double)W}, {"density", dens}});
//
// 環境変数 AHC_FEATURES が設定されているときだけ
//     feature <name> <value>
// 形式で stdout に出力して exit(0) する(features.py の USE_CPP_EXTRACTOR が利用)。
// 通常実行や ONLINE_JUDGE ビルドでは何もしない。
inline void emit_features(std::initializer_list<std::pair<const char*, double>> feats) {
#if !defined(ONLINE_JUDGE)
    if (std::getenv("AHC_FEATURES") == nullptr) {
        return;
    }
    for (const auto& kv : feats) {
        std::printf("feature %s %.12g\n", kv.first, kv.second);
    }
    std::fflush(stdout);
    std::exit(0);
#else
    (void)feats;
#endif
}

} // namespace ahc
