import argparse
import build
from build import RELEASE_FLAGS
import config_util
import shutil
import subprocess
import time
import os
import json
import math
from concurrent.futures import ThreadPoolExecutor

SCORE_DIR_NAME = "score"
BEST_SCORES_FILENAME = "best_scores.json"
PREV_SCORES_FILENAME = "prev_scores.json"
BEST_OUTPUT_DIR_NAME = "best"

C_RESET = "\033[0m"
C_RED   = "\033[31m"
C_GREEN = "\033[32m"
C_GOLD  = "\033[33;1m"


def score_file_path(filename: str, work_dir: str | None = None) -> str:
    base_dir = work_dir if work_dir is not None else config_util.work_dir()
    return os.path.join(base_dir, SCORE_DIR_NAME, filename)


def best_output_file_path(case_str: str, work_dir: str | None = None) -> str:
    base_dir = work_dir if work_dir is not None else config_util.work_dir()
    return os.path.join(base_dir, BEST_OUTPUT_DIR_NAME, case_str + ".out")


def load_scores(filepath) -> dict:
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_scores(scores, filepath):
    with open(filepath, "w") as f:
        json.dump(scores, f, indent=4)


def prompt_yes_no(message: str) -> bool:
    while True:
        user_input = input(f"{message} [y/n]: ").strip().lower()
        if user_input in ("y", "yes"):
            return True
        if user_input in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")


def effective_score_from_result(result: dict, fail_score: int) -> int:
    """log計算用のスコア。AC以外は fail_score を返す。"""
    status = str(result.get("status", ""))
    if status.startswith("AC") and result.get("score") is not None:
        return max(1, int(result["score"]))
    return fail_score


def run_test_case(
    case_str,
    input_file,
    output_file,
    solution_file,
    vis_file,
    tester_file,
    score_prefix,
    interactive,
):
    cmd_cpp = [tester_file, solution_file] if interactive else [solution_file]
    start_time = time.perf_counter()
    err_file = os.path.splitext(output_file)[0] + ".err"

    try:
        with open(input_file, "r") as fin, open(output_file, "w") as fout, open(err_file, "w") as ferr:
            subprocess.run(
                cmd_cpp,
                stdin=fin,
                stdout=fout,
                stderr=ferr,
                text=True,
                check=True,
            )
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
        status = "AC"
    except subprocess.CalledProcessError:
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "case": case_str,
            "score": None,
            "elapsed_time": elapsed_time_ms,
            "status": "RE",
            "output_file": output_file,
        }

    score = None
    if interactive:
        parse_targets = []
        try:
            with open(err_file, "r") as ferr:
                parse_targets.append(ferr.read())
        except Exception:
            pass
        try:
            with open(output_file, "r") as fout:
                parse_targets.append(fout.read())
        except Exception:
            pass

        for text in parse_targets:
            for line in text.splitlines():
                line = line.strip()
                if line.startswith(score_prefix):
                    try:
                        score = int(line.split("=")[-1].strip())
                    except Exception:
                        score = None
                    break
            if score is not None:
                break
    else:
        try:
            res = subprocess.run(
                [vis_file, input_file, output_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
        except Exception:
            status = "WA"

        if status == "AC":
            for line in res.stdout.splitlines():
                line = line.strip()
                if line.startswith(score_prefix):
                    try:
                        score = int(line.split("=")[-1].strip())
                    except Exception:
                        score = None
                    break

    if status == "AC" and score is None:
        status = "WA"
    if status == "AC" and score == 0:
        status = "WA"
        score = None

    return {
        "case": case_str,
        "score": score,
        "elapsed_time": elapsed_time_ms,
        "status": status,
        "output_file": output_file,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run local tests for AHC submissions.")
    parser.add_argument("--cases", type=int, default=None, help="Number of test cases to run.")
    parser.add_argument("--range", nargs=2, metavar=("L", "R"), type=int, default=None, help="Run cases with seed IDs in [L, R).")
    parser.add_argument(
        "--in",
        dest="in_dir",
        type=str,
        default=None,
        help="Override testcase input directory. Score tracking (best/prev) is disabled when specified.",
    )
    parser.add_argument("--jobs", type=int, default=8, help="Number of worker threads.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Build solver with -DDEBUG.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help=f"Build solver with online judge flags ({' '.join(RELEASE_FLAGS)}).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        dest="no_save",
        help="Skip prev-score saving prompt.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.range is not None and args.cases is not None:
        raise ValueError("--range and --cases cannot be used together.")

    # --in が指定された場合はスコアトラッキングを無効化する
    score_tracking_enabled = args.in_dir is None

    config = config_util.load_config()
    extra_flags = []
    if args.debug:
        extra_flags.append("-DDEBUG")
        print("Debug build enabled: -DDEBUG")
    if args.release:
        extra_flags += RELEASE_FLAGS
        print(f"Release build enabled: {' '.join(RELEASE_FLAGS)}")
    build.compile_program(config, extra_flags=extra_flags)

    work_dir = config_util.work_dir()
    if score_tracking_enabled:
        os.makedirs(os.path.join(work_dir, SCORE_DIR_NAME), exist_ok=True)
        os.makedirs(os.path.join(work_dir, BEST_OUTPUT_DIR_NAME), exist_ok=True)
    else:
        print("Score tracking disabled (--in was specified).")

    if args.in_dir is None:
        input_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    else:
        in_dir_arg = args.in_dir.strip()
        input_dir = in_dir_arg if os.path.isabs(in_dir_arg) else os.path.join(work_dir, in_dir_arg)
        input_dir = os.path.normpath(input_dir)
        print(f"Input dir override: {os.path.relpath(input_dir, work_dir)}")
        if not os.path.isdir(input_dir):
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir = os.path.join(work_dir, config["paths"]["testcase_output_dir"])
    solution_file = os.path.join(work_dir, config["files"]["sol_file"])
    vis_file = os.path.join(work_dir, config["files"]["vis_file"])
    tester_file = os.path.join(work_dir, config["files"]["tester_file"])
    score_prefix = config["problem"]["score_prefix"]
    objective = config["problem"]["objective"]
    interactive = config["problem"]["interactive"]
    os.makedirs(output_dir, exist_ok=True)

    best_scores_file = score_file_path(BEST_SCORES_FILENAME, work_dir)
    prev_scores_file = score_file_path(PREV_SCORES_FILENAME, work_dir)
    prev_scores_label = os.path.relpath(prev_scores_file, work_dir)

    available_cases = sorted(
        int(os.path.splitext(fname)[0])
        for fname in os.listdir(input_dir)
        if fname.endswith(".txt") and os.path.splitext(fname)[0].isdigit()
    )

    if args.range is not None:
        l_seed, r_seed = args.range
        selected_cases = [c for c in available_cases if l_seed <= c < r_seed]
    else:
        config_case_count = config["problem"]["pretest_count"]
        requested_count = args.cases if args.cases is not None else config_case_count
        selected_cases = available_cases[:min(requested_count, len(available_cases))]

    if score_tracking_enabled:
        best_scores_map = load_scores(best_scores_file)
        best_scores_ref_map = dict(best_scores_map)
        prev_scores_map = load_scores(prev_scores_file)
    else:
        best_scores_map = {}
        best_scores_ref_map = {}
        prev_scores_map = {}
    current_scores_map = {}

    results = []
    total_diff_prev = 0
    total_diff_best = 0
    total_diff_prev_norm = 0.0
    wa_count = 0
    best_updated_count = 0

    cases_to_run = [
        (c, f"{c:03d}", os.path.join(input_dir, f"{c:03d}.txt"), os.path.join(output_dir, f"{c:03d}.txt"))
        for c in selected_cases
    ]
    selected_case_strs = [case_str for _, case_str, _, _ in cases_to_run]

    def format_diff(diff, is_new_record=False, width=8):
        if diff is None:
            return " " * width
        val_str = f"{diff:+d}"
        padded_str = f"{val_str:<{width}}"
        if is_new_record:
            return f"{C_GOLD}{padded_str}{C_RESET}"
        is_improvement = (objective == "maximize" and diff > 0) or (objective == "minimize" and diff < 0)
        is_degradation = (objective == "maximize" and diff < 0) or (objective == "minimize" and diff > 0)
        if is_improvement:
            return f"{C_GREEN}{padded_str}{C_RESET}"
        if is_degradation:
            return f"{C_RED}{padded_str}{C_RESET}"
        return padded_str

    def format_sum(total_val, width=8):
        val_str = f"{total_val:+d}"
        padded_str = f"{val_str:<{width}}"
        is_good = (objective == "maximize" and total_val >= 0) or (objective == "minimize" and total_val <= 0)
        return f"{C_GREEN}{padded_str}{C_RESET}" if is_good else f"{C_RED}{padded_str}{C_RESET}"

    def format_diff_float(diff, width=8):
        if diff is None:
            return " " * width
        val_str = f"{diff:+.3f}"
        padded_str = f"{val_str:<{width}}"
        if diff > 1e-12:
            return f"{C_GREEN}{padded_str}{C_RESET}"
        if diff < -1e-12:
            return f"{C_RED}{padded_str}{C_RESET}"
        return padded_str

    def format_sum_float(total_val, width=8):
        val_str = f"{total_val:+.3f}"
        padded_str = f"{val_str:<{width}}"
        if total_val >= -1e-12:
            return f"{C_GREEN}{padded_str}{C_RESET}"
        return f"{C_RED}{padded_str}{C_RESET}"

    def normalized_ratio(score, norm_ref):
        if score is None or norm_ref is None or norm_ref <= 0:
            return None
        if objective == "minimize":
            if score <= 0:
                return None
            r = float(norm_ref) / float(score)
        else:
            r = float(score) / float(norm_ref)
        return max(0.0, min(1.0, r))

    futures_list = []
    max_workers = max(args.jobs, 1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for case_id, case_str, input_file, output_file in cases_to_run:
            fut = executor.submit(
                run_test_case,
                case_str,
                input_file,
                output_file,
                solution_file,
                vis_file,
                tester_file,
                score_prefix,
                interactive,
            )
            futures_list.append((fut, case_id, case_str))

        print(
            f"{'Case':<5} {'Score':<10} {'Time':<6} {'Stat':<5} | "
            f"{'vsPrev':<8} {'Sum':<8} | "
            f"{'vsBest':<8} {'Sum':<8} | "
            f"{'vsPrevN':<8} {'Sum':<8}"
        )
        print("-" * 99)

        for fut, case_id, case_str in futures_list:
            try:
                result = fut.result()
            except Exception as exc:
                print(f"Error: case {case_str} exception: {exc}")
                continue

            current_score = result["score"]
            is_ac_like = str(result["status"]).startswith("AC") and current_score is not None
            if not is_ac_like:
                wa_count += 1

            if score_tracking_enabled and is_ac_like:
                current_scores_map[case_str] = current_score

            old_prev = prev_scores_map.get(case_str)
            old_best = best_scores_map.get(case_str)

            if score_tracking_enabled and is_ac_like:
                diff_prev = (current_score - old_prev) if old_prev is not None else 0
                diff_best = (current_score - old_best) if old_best is not None else 0

                # ベスト更新時は新スコアを基準にすることで vsPrevN が不利にならないようにする
                if old_best is None:
                    norm_ref = current_score
                elif objective == "maximize":
                    norm_ref = max(old_best, current_score)
                else:
                    norm_ref = min(old_best, current_score)
                cur_norm = normalized_ratio(current_score, norm_ref)
                prev_norm = normalized_ratio(old_prev, norm_ref) if old_prev is not None else None
                diff_prev_norm = (cur_norm - prev_norm) if (prev_norm is not None and cur_norm is not None) else 0.0
            elif score_tracking_enabled:
                diff_prev = None if old_prev is not None else 0
                diff_best = None if old_best is not None else 0
                diff_prev_norm = None if old_prev is not None else 0.0
            else:
                diff_prev = diff_best = diff_prev_norm = None

            if diff_prev is not None:
                total_diff_prev += diff_prev
            if diff_best is not None:
                total_diff_best += diff_best
            if diff_prev_norm is not None:
                total_diff_prev_norm += diff_prev_norm

            is_new_best = score_tracking_enabled and is_ac_like and (
                old_best is None
                or (objective == "maximize" and current_score > old_best)
                or (objective == "minimize" and current_score < old_best)
            )
            if is_new_best:
                best_scores_map[case_str] = current_score
                src_out = result.get("output_file")
                if src_out and os.path.exists(src_out):
                    shutil.copyfile(src_out, best_output_file_path(case_str, work_dir))
                best_updated_count += 1

            p_case   = f"{result['case']:<5}"
            p_score  = f"{result['score']:<10,d}" if is_ac_like else f"{'-':<10}"
            p_time   = f"{result['elapsed_time']:<6.0f}"
            p_status = f"{result['status']:<5}"

            if score_tracking_enabled:
                prev_str      = format_diff(diff_prev if old_prev is not None else None, is_new_record=is_new_best)
                prev_sum_str  = format_sum(total_diff_prev)
                best_str      = format_diff(diff_best if old_best is not None else None, is_new_record=is_new_best)
                best_sum_str  = format_sum(total_diff_best)
                prev_norm_str     = format_diff_float(diff_prev_norm if old_prev is not None else None)
                prev_norm_sum_str = format_sum_float(total_diff_prev_norm)

                if old_prev is None:
                    prev_str      = f"{'NEW':<8}" if is_ac_like else f"{'-':<8}"
                    prev_norm_str = f"{'NEW':<8}" if is_ac_like else f"{'-':<8}"
                if old_best is None:
                    best_str = f"{C_GOLD}{'NEW':<8}{C_RESET}" if is_ac_like else f"{'-':<8}"
            else:
                prev_str = prev_sum_str = best_str = best_sum_str = prev_norm_str = prev_norm_sum_str = f"{'-':<8}"

            print(
                f"{p_case} {p_score} {p_time} {p_status} | "
                f"{prev_str} {prev_sum_str} | "
                f"{best_str} {best_sum_str} | "
                f"{prev_norm_str} {prev_norm_sum_str}"
            )

            results.append(result)

    if not results:
        return

    total_score = sum(
        r["score"] for r in results
        if str(r.get("status", "")).startswith("AC") and r.get("score") is not None
    )
    max_elapsed = max((r["elapsed_time"] for r in results), default=0.0)

    print("-" * 99)
    print(f"Total Score  : {total_score:,d}")
    print(f"WA Count     : {wa_count}")
    print(f"Max Time     : {max_elapsed:.0f} ms")
    if score_tracking_enabled:
        print(f"Best Updated : {best_updated_count}")

    result_by_case = {r["case"]: r for r in results}
    max_known_score = 1
    for case_str in selected_case_strs:
        r = result_by_case.get(case_str)
        if r is not None and str(r.get("status", "")).startswith("AC") and r.get("score") is not None:
            max_known_score = max(max_known_score, int(r["score"]))
        prev_v = prev_scores_map.get(case_str)
        if prev_v is not None:
            max_known_score = max(max_known_score, int(prev_v))
        best_v = best_scores_ref_map.get(case_str)
        if best_v is not None:
            max_known_score = max(max_known_score, int(best_v))
    fail_score = max_known_score * 10 if objective == "minimize" else 1

    cur_eff_list = [
        effective_score_from_result(
            result_by_case.get(case_str, {"status": "WA", "score": None}),
            fail_score,
        )
        for case_str in selected_case_strs
    ]

    if cur_eff_list:
        mean_log_cur = sum(math.log(float(s)) for s in cur_eff_list) / len(cur_eff_list)
        print("Relative Metrics (selected cases)")
        print(f"  mean(log score)      : {mean_log_cur:.6f} (n={len(cur_eff_list)})")

    def print_relative_vs(ref_map: dict, label: str):
        diffs = []
        ratios = []
        for case_str in selected_case_strs:
            ref = ref_map.get(case_str)
            if ref is None:
                continue
            ref_eff = max(1, int(ref))
            cur_eff = effective_score_from_result(
                result_by_case.get(case_str, {"status": "WA", "score": None}),
                fail_score,
            )
            if objective == "minimize":
                diffs.append(math.log(float(ref_eff)) - math.log(float(cur_eff)))
                ratios.append(float(ref_eff) / float(cur_eff))
            else:
                diffs.append(math.log(float(cur_eff)) - math.log(float(ref_eff)))
                ratios.append(float(cur_eff) / float(ref_eff))

        if not diffs:
            print(f"  vs {label:<16}: n/a")
            return

        delta_mean_log = sum(diffs) / len(diffs)
        geom_factor = math.exp(delta_mean_log)
        mean_ratio = sum(ratios) / len(ratios)
        print(
            f"  vs {label:<16}: "
            f"delta_mean_log={delta_mean_log:+.6f}, "
            f"exp(delta)={geom_factor:.6f}, "
            f"mean_ratio={mean_ratio:.6f}, n={len(diffs)}"
        )

    print_relative_vs(prev_scores_map, "prev")
    print_relative_vs(best_scores_ref_map, "best(ref)")

    if not score_tracking_enabled:
        return

    save_scores(best_scores_map, best_scores_file)

    if not current_scores_map:
        print("No AC results to save. Skipped updating prev scores.")
    elif args.no_save:
        print("Skipped saving prev scores (--no-save).")
    else:
        if prompt_yes_no(f"{C_GOLD}Save current results to {prev_scores_label}?{C_RESET}"):
            existing_prev = load_scores(prev_scores_file)
            existing_prev.update(current_scores_map)
            save_scores(existing_prev, prev_scores_file)
            print(f"{C_GOLD}Saved current results to {prev_scores_label}{C_RESET}")
        else:
            print("Skipped saving current results.")


if __name__ == "__main__":
    main()
