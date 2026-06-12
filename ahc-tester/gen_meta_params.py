"""meta_params.json から lib/meta_params.hpp を自動生成する。

optuna_manager.py --by-category が集約したカテゴリ別ベストパラメータ
(手動編集も可)を、実行時に入力の特徴量からカテゴリを判定して
HP_PARAM / META_PARAM のグローバル変数へ代入する C++ ヘッダに展開する。

    $ uv run ahc-tester/gen_meta_params.py

main.cpp 側の使い方(生成ヘッダの先頭コメントにも記載):
    #define USE_META_PARAMS
    #include "lib/hp_params.hpp"
    HP_PARAM(double, T0, 2.0, 0.1, 10.0);
    META_PARAM(int, STRATEGY, 0);
    #include "lib/meta_params.hpp"
    int main() { /* 入力読み込み後に */ meta::apply_params(N, W); }
"""

import argparse
import json
import os
import re
import sys

import config_util
import cpp_params
import features as features_mod
import meta_report

IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def fmt_value(value, is_float: bool) -> str:
    if is_float:
        return repr(float(value))
    return str(int(round(float(value))))


def fmt_threshold(value) -> str:
    if isinstance(value, int) or (isinstance(value, float) and value.is_integer()):
        return str(int(value))
    return repr(float(value))


def emit_bin_selector(axis: str, spec: dict, lines: list):
    """軸の値 -> ビン番号の if 連鎖を生成する(Binner.index と同じ割り当て)。"""
    var = f"bi_{axis}"
    if spec["kind"] == "edges":
        edges = spec["edges"]
        thresholds = edges[1:-1]
        n_bins = len(edges) - 1
    else:
        values = spec["values"]
        if any(isinstance(v, str) for v in values):
            raise ValueError(f"axis '{axis}': 文字列の特徴量はコード展開に使えません")
        thresholds = [(values[i] + values[i + 1]) / 2.0 for i in range(len(values) - 1)]
        n_bins = len(values)

    if n_bins == 1:
        lines.append(f"    const int {var} = 0;")
        lines.append(f"    (void){var};")
        return
    lines.append(f"    int {var};")
    for i, th in enumerate(thresholds):
        kw = "if" if i == 0 else "else if"
        lines.append(f"    {kw} ({axis} < {fmt_threshold(th)}) {{ {var} = {i}; }}")
    lines.append(f"    else {{ {var} = {n_bins - 1}; }}")


def main():
    parser = argparse.ArgumentParser(description="Generate lib/meta_params.hpp from meta_params.json.")
    parser.add_argument("--meta", default=None, help="Path to meta_params.json (default: config's meta_params_file).")
    parser.add_argument("--cpp", default=None, help="C++ source to scan for HP_PARAM/META_PARAM (default: config's cpp_file).")
    parser.add_argument("--out", default=os.path.join("lib", "meta_params.hpp"), help="Output header path.")
    parser.add_argument(
        "--in",
        dest="in_dir",
        default=None,
        metavar="DIR",
        help="Input directory for feature/bin reconstruction (default: config's testcase_input_dir).",
    )
    args = parser.parse_args()

    config = config_util.load_config()
    work_dir = config_util.work_dir()

    meta_path = args.meta or os.path.join(work_dir, config["files"].get("meta_params_file", "meta_params.json"))
    if not os.path.isfile(meta_path):
        print(f"Error: {meta_path} not found. Run optuna_manager.py --by-category first (or create it manually).")
        sys.exit(1)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    axes = meta.get("axes") or []
    categories = meta.get("categories") or {}
    if not axes or not categories:
        print(f"Error: {meta_path} has no axes/categories.")
        sys.exit(1)
    for axis in axes:
        if not IDENT_RE.match(axis):
            print(f"Error: axis '{axis}' is not a valid C++ identifier.")
            sys.exit(1)
    if features_mod.AXES and list(features_mod.AXES) != list(axes):
        print(f"Warning: features.py AXES {features_mod.AXES} != meta_params.json axes {axes} (json 側を使用)")

    cpp_file = args.cpp or config["files"]["cpp_file"]
    cpp_path = cpp_file if os.path.isabs(cpp_file) else os.path.join(work_dir, cpp_file)
    if not os.path.isfile(cpp_path):
        print(f"Error: C++ file not found: {cpp_path}")
        sys.exit(1)
    registry = cpp_params.param_registry(cpp_path)
    if not registry:
        print(f"Warning: no HP_PARAM/META_PARAM declarations found in {cpp_file}")

    in_dir = args.in_dir or config["paths"]["testcase_input_dir"]
    input_dir = in_dir if os.path.isabs(in_dir) else os.path.join(work_dir, in_dir)
    feats = features_mod.load_features(input_dir) if os.path.isdir(input_dir) else {}
    binner = meta_report.Binner.build(feats, axes, features_mod.BINS)

    label_index = {axis: {label: i for i, label in enumerate(binner.labels(axis))} for axis in axes}

    body = []
    for axis in axes:
        emit_bin_selector(axis, binner.spec(axis), body)
    body.append("")

    n_emitted = 0
    for cat_key in sorted(categories, key=lambda k: tuple(
        label_index[a].get(l, 10**9) for a, l in zip(axes, k.split(","))
    )):
        entry = categories[cat_key]
        labels = cat_key.split(",")
        if len(labels) != len(axes):
            print(f"Warning: category '{cat_key}' does not match axes {axes}. Skipped.")
            continue
        idxs = []
        for axis, label in zip(axes, labels):
            if label not in label_index[axis]:
                idxs = None
                break
            idxs.append(label_index[axis][label])
        if idxs is None:
            print(f"Warning: category '{cat_key}' has unknown bin labels (BINS 変更後は再チューニング推奨). Skipped.")
            continue

        params = entry.get("params") or {}
        assigns = []
        for name, value in params.items():
            info = registry.get(name)
            if info is None:
                print(f"Warning: param '{name}' (category {cat_key}) is not declared in {cpp_file}. Skipped.")
                continue
            if "lower" in info:
                lo, hi = info["lower"], info["upper"]
                clamped = min(max(value, lo), hi)
                if clamped != value:
                    print(f"Warning: param '{name}' value {value} clamped into [{lo}, {hi}].")
                    value = clamped
            assigns.append(f"        hp_assign({name}, HP_ENV_PREFIX \"{name}\", {fmt_value(value, info['is_float'])});")
        if not assigns:
            continue

        cond = " && ".join(f"bi_{axis} == {idx}" for axis, idx in zip(axes, idxs))
        note = ""
        if entry.get("best_score") is not None:
            note = f"  // best_score={entry['best_score']:.2f}, n_cases={entry.get('n_cases', '?')}"
        body.append(f"    // {cat_key}{note}")
        body.append(f"    if ({cond}) {{")
        body.extend(assigns)
        body.append("        return;")
        body.append("    }")
        n_emitted += 1

    body.append("    // 未チューニングのカテゴリはデフォルト値のまま")

    arg_list = ", ".join(f"double {axis}" for axis in axes)
    header = f"""// このファイルは gen_meta_params.py により自動生成される。手動編集しないこと。
// 元データ: {os.path.relpath(meta_path, work_dir)} (axes: {', '.join(axes)})
//
// 使い方:
//   #define USE_META_PARAMS
//   #include "lib/hp_params.hpp"
//   HP_PARAM(...); META_PARAM(...);   // パラメータ宣言
//   #include "lib/meta_params.hpp"    // 宣言の後に include する
//   int main() {{ /* 入力読み込み後に */ meta::apply_params({', '.join(axes)}); }}
#pragma once

#if defined(ONLINE_JUDGE) && !defined(USE_META_PARAMS)
#error "meta_params.hpp requires USE_META_PARAMS to be defined before including hp_params.hpp"
#endif

#if !defined(ONLINE_JUDGE)
#include <cstdlib>
#endif

namespace meta {{

template <typename T, typename V>
inline void hp_assign(T& target, const char* env_name, V value) {{
#if !defined(ONLINE_JUDGE)
    // ローカルでは環境変数(optuna が注入した値)を優先する
    if (std::getenv(env_name) != nullptr) {{ return; }}
#else
    (void)env_name;
#endif
    target = static_cast<T>(value);
}}

inline void apply_params({arg_list}) {{
"""
    footer = """}

}  // namespace meta
"""

    out_path = args.out if os.path.isabs(args.out) else os.path.join(work_dir, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(body) + "\n")
        f.write(footer)

    print(f"Generated {os.path.relpath(out_path, work_dir)} ({n_emitted}/{len(categories)} categories embedded)")


if __name__ == "__main__":
    main()
