"""保存済みランの閲覧・比較。

run_test.py が results/ に保存した JSON を読み込み、
解法ごとのサマリ・ケース別スコア・カテゴリ別の勝者を表示する。
run_test.py の複数解法比較モードからも同じレンダリングを使う。

    $ uv run ahc-tester/report.py results/A.json results/B.json
"""

import argparse
import json
import math
import os
import sys

import config_util
import features as features_mod
import meta_report
from meta_report import C_RESET, C_GREEN


def load_run(path: str) -> dict:
    with open(path, "r") as f:
        run = json.load(f)
    if "label" not in run or "results" not in run:
        raise ValueError(f"{path}: run_test の結果ファイルではありません")
    return run


def effective_score(entry: dict | None, fail_score: int) -> int:
    """log 集計用スコア。AC 以外・欠損は fail_score。"""
    if entry is None:
        return fail_score
    status = str(entry.get("status", ""))
    if status.startswith("AC") and entry.get("score") is not None:
        return max(1, int(entry["score"]))
    return fail_score


def compute_fail_score(runs: list, case_strs: list, objective: str) -> int:
    max_known = 1
    for run in runs:
        for case_str in case_strs:
            entry = run["results"].get(case_str)
            if entry and str(entry.get("status", "")).startswith("AC") and entry.get("score") is not None:
                max_known = max(max_known, int(entry["score"]))
    return max_known * 10 if objective == "minimize" else 1


def _is_ac(entry) -> bool:
    return entry is not None and str(entry.get("status", "")).startswith("AC") and entry.get("score") is not None


def render_case_table(runs: list, case_strs: list, objective: str):
    labels = [run["label"] for run in runs]
    col_width = max(12, max(len(l) for l in labels) + 1)

    header = f"{'Case':<5} |"
    for label in labels:
        header += " " + meta_report.pad(label, col_width)
    print(header)
    print("-" * meta_report.visible_len(header))

    for case_str in case_strs:
        entries = [run["results"].get(case_str) for run in runs]
        scores = [int(e["score"]) if _is_ac(e) else None for e in entries]
        valid = [s for s in scores if s is not None]
        best = None
        if valid:
            best = max(valid) if objective == "maximize" else min(valid)
        line = f"{case_str:<5} |"
        for entry, score in zip(entries, scores):
            if score is None:
                cell = str(entry.get("status", "-")) if entry else "-"
            else:
                cell = f"{score:,d}"
                if best is not None and score == best and len(valid) >= 2:
                    cell = f"{C_GREEN}{cell}*{C_RESET}"
            line += " " + meta_report.pad(cell, col_width)
        print(line)


def render_summary(runs: list, case_strs: list, objective: str, fail_score: int):
    label_width = max(12, max(len(run["label"]) for run in runs) + 1)
    print(
        f"{'Label':<{label_width}} {'Total':>16} {'WA':>4} {'MaxTime':>8} "
        f"{'mean(log)':>10} {'vsBase':>10}"
    )
    print("-" * (label_width + 52))

    base_eff = None
    for run in runs:
        results = run["results"]
        total = sum(int(e["score"]) for e in results.values() if _is_ac(e))
        wa = sum(1 for c in case_strs if not _is_ac(results.get(c)))
        max_time = max((float(e.get("time_ms", 0.0)) for e in results.values()), default=0.0)
        eff = {c: effective_score(results.get(c), fail_score) for c in case_strs}
        mean_log = sum(math.log(float(eff[c])) for c in case_strs) / len(case_strs)

        if base_eff is None:
            base_eff = eff
            vs_base = "-"
        else:
            diffs = []
            for c in case_strs:
                d = math.log(float(eff[c])) - math.log(float(base_eff[c]))
                diffs.append(d if objective == "maximize" else -d)
            ratio = math.exp(sum(diffs) / len(diffs))
            text = f"x{ratio:.4f}"
            if ratio > 1 + 1e-9:
                vs_base = f"{C_GREEN}{text}{C_RESET}"
            elif ratio < 1 - 1e-9:
                vs_base = f"\033[31m{text}{C_RESET}"
            else:
                vs_base = text

        print(
            f"{run['label']:<{label_width}} {total:>16,d} {wa:>4d} {max_time:>8.0f} "
            f"{mean_log:>10.4f} {meta_report.pad(vs_base, 10)}"
        )


def render_category_comparison(runs: list, case_strs: list, objective: str, fail_score: int, input_dir: str):
    """カテゴリごとの勝者(と次点との差)を表示。1ランのみなら best(ref) 比。"""
    if not features_mod.AXES:
        return
    try:
        feats = features_mod.load_features(input_dir, case_strs)
        binner = features_mod.build_binner(feats)
    except Exception as exc:
        print(f"Category breakdown skipped: {exc}")
        return
    if binner is None:
        return
    groups, unmatched = meta_report.group_by_category(case_strs, feats, binner)
    if not groups:
        return

    eff_maps = [
        (run["label"], {c: effective_score(run["results"].get(c), fail_score) for c in case_strs})
        for run in runs
    ]

    if len(runs) == 1:
        # 単独ランは score/best_scores.json を基準に表示
        best_file = os.path.join(config_util.work_dir(), "score", "best_scores.json")
        ref_map = {}
        if os.path.exists(best_file):
            try:
                with open(best_file, "r") as f:
                    ref_map = json.load(f)
            except json.JSONDecodeError:
                ref_map = {}
        label, eff = eff_maps[0]

        def cell(cat, cases):
            return meta_report.vs_ref_cell(cases, lambda c: eff.get(c), ref_map, objective)

        title = f"Category Breakdown vs best(ref)  [axes: {', '.join(binner.axes)}]"
    else:
        base_label = eff_maps[0][0]

        def cell(cat, cases):
            means = []
            for label, eff in eff_maps:
                mean_log = sum(math.log(float(eff[c])) for c in cases) / len(cases)
                means.append((label, mean_log))
            reverse = objective == "maximize"
            ranked = sorted(means, key=lambda kv: kv[1], reverse=reverse)
            winner_label, winner_mean = ranked[0]
            margin = math.exp(abs(winner_mean - ranked[1][1])) if len(ranked) >= 2 else 1.0
            text = f"{winner_label} x{margin:.3f}({len(cases)})"
            if winner_label != base_label:
                return f"{C_GREEN}{text}{C_RESET}"
            return text

        title = f"Category Winner (margin vs runner-up)  [axes: {', '.join(binner.axes)}, base: {base_label}]"

    print()
    print(title)
    for line in meta_report.render_matrix(binner, groups, cell):
        print("  " + line)
    if unmatched:
        print(f"  (uncategorized: {len(unmatched)} cases)")


def render_comparison(runs: list, case_strs: list, objective: str, input_dir: str, show_cases: bool = True):
    if not case_strs:
        print("No common cases to compare.")
        return
    fail_score = compute_fail_score(runs, case_strs, objective)
    if show_cases and len(runs) >= 2:
        render_case_table(runs, case_strs, objective)
        print()
    render_summary(runs, case_strs, objective, fail_score)
    render_category_comparison(runs, case_strs, objective, fail_score, input_dir)


def main():
    parser = argparse.ArgumentParser(description="View / compare saved run results.")
    parser.add_argument("files", nargs="+", help="results/*.json files saved by run_test.py")
    parser.add_argument("--no-cases", action="store_true", help="Skip the per-case table.")
    parser.add_argument(
        "--in",
        dest="in_dir",
        default=None,
        metavar="DIR",
        help="Input directory for feature lookup (default: recorded in the result file)",
    )
    args = parser.parse_args()

    config = config_util.load_config()
    work_dir = config_util.work_dir()

    runs = [load_run(path) for path in args.files]

    # ラベルが重複すると比較表が読めないので連番を付けて区別する
    seen = {}
    for run in runs:
        label = run["label"]
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            run["label"] = f"{label}#{seen[label]}"

    # 全ランに共通するケースだけ比較対象にする
    common = set(runs[0]["results"].keys())
    for run in runs[1:]:
        common &= set(run["results"].keys())
    case_strs = sorted(common, key=int)
    if not case_strs:
        print("Error: no common cases across the given result files.")
        sys.exit(1)

    objective = runs[0].get("objective") or config["problem"]["objective"]
    in_dir = args.in_dir or runs[0].get("input_dir") or config["paths"]["testcase_input_dir"]
    input_dir = in_dir if os.path.isabs(in_dir) else os.path.join(work_dir, in_dir)

    for path, run in zip(args.files, runs):
        print(f"Run: {run['label']}  ({os.path.relpath(path, work_dir)}, cpp={run.get('cpp_file', '?')}, at={run.get('timestamp', '?')})")
    print()

    render_comparison(runs, case_strs, objective, input_dir, show_cases=not args.no_cases)


if __name__ == "__main__":
    main()
