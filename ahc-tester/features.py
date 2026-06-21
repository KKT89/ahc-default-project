"""テストケースの特徴量(メタデータ)抽出。

コンテストごとに下の「編集する設定」ブロックを書き換えて使う。
抽出結果は <入力ディレクトリ>/features.json にキャッシュされ、
run_test / optuna_manager のカテゴリ別集計・絞り込みで参照される。

設定を変更したら CLI で再抽出する:
    $ uv run ahc-tester/features.py
"""

import argparse
import json
import os
import sys

import meta_report

# ==== コンテストごとに編集する設定 ====

# 入力ファイル1行目のトークン名(先頭から順に対応)。例: ["N", "W"]
HEADER_NAMES: list = []

# 分類・集計に使う軸(特徴量のキー)。例: ["N", "W"]
AXES: list = []

# 軸ごとのビン境界(昇順)。例: {"N": [10, 12, 14, 16, 18, 20]}
#   → [10,12) [12,14) [14,16) [16,18) [18,20] (最後のみ上端含む)
# 省略した軸は自動ビン分け(distinct <= 8 なら値ごと、それ以外は5分位)。
BINS: dict = {}

# True にすると extract() の代わりにソリューションバイナリから特徴量を取得する。
# main.cpp 側で入力読み込み直後に ahc::emit_features({...}) を呼んでおくこと
# (lib/features.hpp 参照)。派生統計量を C++ と Python で二重実装せずに済む。
USE_CPP_EXTRACTOR: bool = False


def extract(input_path: str) -> dict:
    """1ケースの特徴量 dict を返す。

    デフォルトは入力1行目のトークンを HEADER_NAMES で名前付けするだけ。
    派生統計量(グリッド密度など)が必要ならここを書き換える。
    ただしカテゴリ別パラメータ展開(gen_meta_params)で使う軸は、
    C++ 側でも実行時に計算できる値に限ること。
    """
    with open(input_path, "r") as f:
        tokens = f.readline().split()
    return {name: _to_number(tok) for name, tok in zip(HEADER_NAMES, tokens)}


# ==== 以下は通常編集不要 ====

FEATURES_FILENAME = "features.json"


def _cpp_solution_path(build_if_missing: bool = True) -> str:
    """C++ 抽出に使うソリューションバイナリのパス。なければビルドする。"""
    import build
    import config_util

    config = config_util.load_config()
    work_dir = config_util.work_dir()
    sol_path = os.path.join(work_dir, config["files"]["sol_file"])
    if build_if_missing and not os.path.exists(sol_path):
        build.compile_program(config)
    return sol_path


def extract_via_cpp(input_path: str, sol_path: str) -> dict:
    """AHC_FEATURES=1 でバイナリを実行し、"feature <name> <value>" 行を回収する。"""
    import subprocess

    env = os.environ.copy()
    env["AHC_FEATURES"] = "1"
    try:
        with open(input_path, "r") as fin:
            res = subprocess.run(
                [sol_path],
                stdin=fin,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=env,
                text=True,
                timeout=30,
            )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"feature extraction timed out: {input_path} "
            "(main.cpp で ahc::emit_features() を呼んでいるか確認)"
        )
    feats = {}
    for line in res.stdout.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "feature":
            feats[parts[1]] = _to_number(parts[2])
    if not feats:
        raise RuntimeError(
            f"no features emitted for {input_path} "
            "(main.cpp で ahc::emit_features() を呼んでいるか確認)"
        )
    return feats


def _to_number(token: str):
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def features_file_path(input_dir: str) -> str:
    return os.path.join(input_dir, FEATURES_FILENAME)


def list_case_strs(input_dir: str) -> list:
    """入力ディレクトリ内のケース名(数字 stem)を seed 順で返す。"""
    stems = [
        os.path.splitext(fname)[0]
        for fname in os.listdir(input_dir)
        if fname.endswith(".txt") and os.path.splitext(fname)[0].isdigit()
    ]
    return sorted(stems, key=int)


def load_features(input_dir: str, case_strs=None, refresh: bool = False) -> dict:
    """ケースごとの特徴量を返す。未抽出分だけ抽出してキャッシュを更新する。"""
    cache_path = features_file_path(input_dir)
    cache = {}
    if not refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except json.JSONDecodeError:
            cache = {}

    if case_strs is None:
        case_strs = list_case_strs(input_dir)

    missing = [c for c in case_strs if c not in cache]
    sol_path = _cpp_solution_path() if (USE_CPP_EXTRACTOR and missing) else None
    extracted = 0
    for case_str in missing:
        input_path = os.path.join(input_dir, case_str + ".txt")
        if not os.path.exists(input_path):
            continue
        if USE_CPP_EXTRACTOR:
            cache[case_str] = extract_via_cpp(input_path, sol_path)
        else:
            cache[case_str] = extract(input_path)
        extracted += 1
    if extracted > 0:
        with open(cache_path, "w") as f:
            json.dump(dict(sorted(cache.items())), f, indent=2)

    return {c: cache[c] for c in case_strs if c in cache}


def build_binner(features_by_case: dict) -> meta_report.Binner | None:
    """AXES 設定から Binner を作る。AXES 未設定なら None。"""
    if not AXES:
        return None
    return meta_report.Binner.build(features_by_case, AXES, BINS)


def main():
    import config_util

    parser = argparse.ArgumentParser(description="Extract testcase features into features.json.")
    parser.add_argument(
        "--in",
        dest="in_dir",
        default=None,
        metavar="DIR",
        help="Input directory (default: config's testcase_input_dir)",
    )
    args = parser.parse_args()

    config = config_util.load_config()
    work_dir = config_util.work_dir()
    if args.in_dir is not None:
        input_dir = args.in_dir if os.path.isabs(args.in_dir) else os.path.join(work_dir, args.in_dir)
    else:
        input_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    if not os.path.isdir(input_dir):
        print(f"Error: input directory not found: {input_dir}")
        sys.exit(1)

    if USE_CPP_EXTRACTOR:
        # 抽出ロジックが変わっている可能性があるので CLI 実行時は必ずビルドし直す
        import build
        build.compile_program(config)
    elif not HEADER_NAMES:
        print("Note: HEADER_NAMES が未設定です。features.py を編集してください。")

    feats = load_features(input_dir, refresh=True)
    print(f"Extracted features for {len(feats)} cases -> {features_file_path(input_dir)}")
    if not feats:
        return

    keys = sorted({k for f in feats.values() for k in f})
    print(f"Feature keys: {', '.join(keys) if keys else '(none)'}")

    binner = build_binner(feats)
    if binner is None:
        print("AXES が未設定のため、カテゴリ分布の表示をスキップしました。")
        return
    groups, unmatched = meta_report.group_by_category(sorted(feats), feats, binner)
    if not groups:
        print("カテゴリに割り当てられたケースがありません(AXES の設定を確認)。")
        return
    print()
    print(f"Case distribution  [axes: {', '.join(binner.axes)}]")
    for line in meta_report.render_matrix(binner, groups, lambda cat, cases: str(len(cases))):
        print("  " + line)
    if unmatched:
        print(f"  (uncategorized: {len(unmatched)} cases)")


if __name__ == "__main__":
    main()
