import argparse
import build
import config_util as config_util
import subprocess
import time
import os
import json
import math
from concurrent.futures import ThreadPoolExecutor

# ファイルパス定義
SCORE_DIR_NAME = "score"
BEST_SCORES_FILENAME = "best_scores.json" # 理論値（全期間ベスト）
PREV_SCORES_FILENAME = "prev_scores.json" # 直近の提出

def score_file_path(filename: str, work_dir: str | None = None) -> str:
    base_dir = work_dir if work_dir is not None else config_util.work_dir()
    return os.path.join(base_dir, SCORE_DIR_NAME, filename)


def load_scores(filepath):
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
    """Simple Y/N prompt for interactive confirmation."""
    while True:
        user_input = input(f"{message} [y/n]: ").strip().lower()
        if user_input in ("y", "yes"):
            return True
        if user_input in ("n", "no"):
            return False
        print("Please enter 'y' or 'n'.")

def effective_score_from_result(result: dict, objective: str, fail_score_minimize: int) -> int:
    """log計算用のスコア。AC以外は目的関数に応じて悪化方向の値を使う。"""
    status = str(result.get("status", ""))
    if status.startswith("AC") and result.get("score") is not None:
        return max(1, int(result["score"]))
    if objective == "minimize":
        return max(1, int(fail_score_minimize))
    return 1


def aggregate_trial_results(case_str: str, trial_results: list[dict], objective: str) -> dict:
    total_elapsed = sum(float(r.get("elapsed_time", 0.0)) for r in trial_results)
    ac_trials = [r for r in trial_results if str(r.get("status", "")).startswith("AC") and r.get("score") is not None]

    if ac_trials:
        # 乱数ぶれの下振れを見るため、目的関数に対して「悪い側」を代表値に採用する。
        # maximize 問題: score が小さいほど悪い -> min
        # minimize 問題: score が大きいほど悪い -> max
        if objective == "minimize":
            best = max(ac_trials, key=lambda x: int(x["score"]))
        else:
            best = min(ac_trials, key=lambda x: int(x["score"]))
        status = "AC"
        if len(ac_trials) < len(trial_results):
            status = "AC*"
        return {
            "case": case_str,
            "score": int(best["score"]),
            "elapsed_time": total_elapsed,
            "status": status,
        }

    fallback_status = "WA"
    if trial_results:
        fallback_status = str(trial_results[0].get("status", "WA"))
    return {
        "case": case_str,
        "score": None,
        "elapsed_time": total_elapsed,
        "status": fallback_status,
    }

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
    
    # 実行
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
        status = "RE"
        return {
            "case": case_str,
            "score": None,
            "elapsed_time": elapsed_time_ms,
            "status": status,
        }

    score = None
    if interactive:
        # interactive は tester の出力（主に stderr）からスコアを読む
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

        if status == "AC":
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
        # 非interactiveはビジュアライザでスコア算出
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
            score = None

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
    }


def run_test_case_try(
    case_str,
    input_file,
    output_file,
    solution_file,
    vis_file,
    tester_file,
    score_prefix,
    interactive,
    try_count,
    objective,
):
    trials = []
    for _ in range(try_count):
        trials.append(
            run_test_case(
                case_str,
                input_file,
                output_file,
                solution_file,
                vis_file,
                tester_file,
                score_prefix,
                interactive,
            )
        )
    return aggregate_trial_results(case_str, trials, objective)

def parse_args():
    parser = argparse.ArgumentParser(description="Run local tests for AHC submissions.")
    parser.add_argument("--cases", type=int, default=None, help="Number of test cases to run.")
    parser.add_argument("--range", nargs=2, metavar=("L", "R"), type=int, default=None, help="Run cases with seed IDs in [L, R).")
    parser.add_argument("--jobs", type=int, default=8, help="Number of worker threads.")
    parser.add_argument(
        "--try",
        dest="try_count",
        type=int,
        default=1,
        help="Run each case multiple times and pick worst-side score by objective.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Skip prev-score saving prompt and build solver with -DDEBUG.",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    if args.range is not None and args.cases is not None:
        raise ValueError("--range and --cases cannot be used together.")
    if args.try_count <= 0:
        raise ValueError("--try must be >= 1.")

    config = config_util.load_config()
    extra_flags = []
    if args.debug:
        extra_flags.append("-DDEBUG")
        print("Debug build enabled: -DDEBUG")
    build.compile_program(config, extra_flags=extra_flags)

    work_dir = config_util.work_dir()
    score_dir = os.path.join(work_dir, SCORE_DIR_NAME)
    os.makedirs(score_dir, exist_ok=True)

    input_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    output_dir = os.path.join(work_dir, config["paths"]["testcase_output_dir"])
    solution_file = os.path.join(work_dir, config["files"]["sol_file"])
    vis_file = os.path.join(work_dir, config["files"]["vis_file"])
    tester_file = os.path.join(work_dir, config["files"]["tester_file"])
    score_prefix = config["problem"]["score_prefix"]
    objective = config["problem"]["objective"]
    interactive = bool(config["problem"]["interactive"])
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
        testcase_count = len(selected_cases)
    else:
        config_case_count = config["problem"]["pretest_count"]
        requested_count = args.cases if args.cases is not None else config_case_count
        testcase_count = min(requested_count, len(available_cases))
        selected_cases = available_cases[:testcase_count]

    # --- スコア読み込み ---
    best_scores_map = load_scores(best_scores_file)
    # 比較用に更新前のスナップショットを保持
    best_scores_ref_map = dict(best_scores_map)
    prev_scores_map = load_scores(prev_scores_file)
    current_scores_map = {} 

    results = []
    total_diff_prev = 0 # Submit比の累積
    total_diff_best = 0 # 理論値比の累積
    total_diff_prev_norm = 0.0 # 正規化(0~1)のPrev比累積
    wa_count = 0
    
    cases_to_run = []
    for case_id in selected_cases:
        case_str = f"{case_id:03d}"
        input_file = os.path.join(input_dir, case_str + ".txt")
        output_file = os.path.join(output_dir, case_str + ".txt")
        cases_to_run.append((case_id, case_str, input_file, output_file))
    selected_case_strs = [f"{c:03d}" for c in selected_cases]

    # --- カラーコード定義 ---
    C_RESET  = "\033[0m"
    C_RED    = "\033[31m"
    C_GREEN  = "\033[32m"
    C_GOLD   = "\033[33;1m"

    # --- 色付けヘルパー関数 ---
    def format_diff(diff, is_new_record=False, width=8):
        if diff is None:
            return " " * width # 比較対象がない場合
        
        val_str = f"{diff:+d}"
        padded_str = f"{val_str:<{width}}"
        
        if is_new_record:
            return f"{C_GOLD}{padded_str}{C_RESET}"
        
        # 改善判定
        is_improvement = False
        is_degradation = False
        
        if diff != 0:
            if objective == "maximize":
                if diff > 0: is_improvement = True
                else: is_degradation = True
            elif objective == "minimize":
                if diff < 0: is_improvement = True
                else: is_degradation = True
        
        if is_improvement:
            return f"{C_GREEN}{padded_str}{C_RESET}"
        elif is_degradation:
            return f"{C_RED}{padded_str}{C_RESET}"
        else:
            return padded_str

    def format_sum(total_val, width=8):
        val_str = f"{total_val:+d}"
        padded_str = f"{val_str:<{width}}"
        
        is_good = False
        if objective == "maximize":
            is_good = (total_val >= 0)
        else:
            is_good = (total_val <= 0)
            
        if is_good:
            return f"{C_GREEN}{padded_str}{C_RESET}"
        else:
            return f"{C_RED}{padded_str}{C_RESET}"

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

    def normalized_ratio(score, best_ref):
        if score is None or best_ref is None:
            return None
        if best_ref <= 0:
            return None
        if objective == "minimize":
            if score <= 0:
                return None
            r = float(best_ref) / float(score)
        else:
            r = float(score) / float(best_ref)
        return max(0.0, min(1.0, r))


    futures_list = []
    max_workers = args.jobs if args.jobs > 0 else 1

    if args.try_count > 1:
        agg_mode = "max" if objective == "minimize" else "min"
        print(f"Try mode: {args.try_count} runs/case, pick worst-side={agg_mode} by objective={objective}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for case_id, case_str, input_file, output_file in cases_to_run:
            fut = executor.submit(
                run_test_case_try,
                case_str,
                input_file,
                output_file,
                solution_file,
                vis_file,
                tester_file,
                score_prefix,
                interactive,
                args.try_count,
                objective,
            )
            futures_list.append((fut, case_id, case_str))

        # ヘッダー出力（幅調整）
        # vsPrev, SumPrev, vsBest, SumBest の順
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
            
            # 保存用
            if is_ac_like:
                current_scores_map[case_str] = current_score

            # --- Diff計算 ---
            old_prev = prev_scores_map.get(case_str)
            old_best = best_scores_map.get(case_str)
            ref_best = best_scores_ref_map.get(case_str)
            
            if is_ac_like:
                diff_prev = (current_score - old_prev) if old_prev is not None else 0
                diff_best = (current_score - old_best) if old_best is not None else 0
                cur_norm = normalized_ratio(current_score, ref_best)
                prev_norm = normalized_ratio(old_prev, ref_best) if old_prev is not None else None
                diff_prev_norm = (cur_norm - prev_norm) if (prev_norm is not None and cur_norm is not None) else 0.0
            else:
                diff_prev = None if old_prev is not None else 0
                diff_best = None if old_best is not None else 0
                diff_prev_norm = None if old_prev is not None else 0.0
            
            # 初回(記録なし)の場合はDiffを0として計算に含めるか、あるいは表示だけ変えるか
            # ここでは計算には含める（0扱い）
            
            if diff_prev is not None:
                total_diff_prev += diff_prev
            if diff_best is not None:
                total_diff_best += diff_best
            if diff_prev_norm is not None:
                total_diff_prev_norm += diff_prev_norm

            # --- Best更新判定 ---
            is_new_best = False
            should_update_best = False
            
            if is_ac_like:
                if old_best is None:
                    should_update_best = True
                    is_new_best = True
                elif objective == "maximize" and current_score > old_best:
                    should_update_best = True
                    is_new_best = True
                elif objective == "minimize" and current_score < old_best:
                    should_update_best = True
                    is_new_best = True
            
            if should_update_best:
                best_scores_map[case_str] = current_score

            # --- 表示 ---
            p_case   = f"{result['case']:<5}"
            if is_ac_like:
                p_score = f"{result['score']:<10,d}"
            else:
                p_score = f"{'-':<10}"
            p_time   = f"{result['elapsed_time']:<6.0f}" # 小数点なしで短く
            p_status = f"{result['status']:<5}"

            # Diff Prev (NewBestならPrev比較でもGoldにするかは好みだが、Prev比較はGreenでいいかも。今回はBest更新ならPrevもGoldにしてみる)
            prev_str = format_diff(diff_prev if old_prev is not None else None, is_new_record=is_new_best, width=8)
            prev_sum_str = format_sum(total_diff_prev, width=8)
            
            # Diff Best (NewBestならGold)
            best_str = format_diff(diff_best if old_best is not None else None, is_new_record=is_new_best, width=8)
            best_sum_str = format_sum(total_diff_best, width=8)

            # Diff Prev normalized (0~1)
            prev_norm_str = format_diff_float(diff_prev_norm if old_prev is not None else None, width=8)
            prev_norm_sum_str = format_sum_float(total_diff_prev_norm, width=8)
            
            # NEWの文字上書き
            if old_prev is None:
                if is_ac_like:
                    prev_str = f"{'NEW':<8}"
                    prev_norm_str = f"{'NEW':<8}"
                else:
                    prev_str = f"{'-':<8}"
                    prev_norm_str = f"{'-':<8}"
            if old_best is None:
                if is_ac_like:
                    best_str = f"{C_GOLD}{'NEW':<8}{C_RESET}"
                else:
                    best_str = f"{'-':<8}"

            print(
                f"{p_case} {p_score} {p_time} {p_status} | "
                f"{prev_str} {prev_sum_str} | "
                f"{best_str} {best_sum_str} | "
                f"{prev_norm_str} {prev_norm_sum_str}"
            )

            results.append(result)

    if not results:
        return

    total_score = sum(r['score'] for r in results if str(r.get('status', '')).startswith("AC") and r.get('score') is not None)
    
    print("-" * 99)
    print(f"Total Score: {total_score:,d}")
    print(f"WA Count   : {wa_count}")

    # --- 相対評価サマリー ---
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
    fail_score_minimize = max_known_score * 10

    cur_eff_list = []
    for case_str in selected_case_strs:
        r = result_by_case.get(case_str, {"status": "WA", "score": None})
        cur_eff_list.append(effective_score_from_result(r, objective, fail_score_minimize))

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
                objective,
                fail_score_minimize,
            )

            if objective == "minimize":
                # 1より大きいほど改善となるように定義
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

    # Bestは常に保存
    save_scores(best_scores_map, best_scores_file)

    # Prev更新：--range/--cases を指定しない通常実行時のみ確認。--debug のときは保存自体をスキップ。
    ran_default = args.range is None and args.cases is None

    if not current_scores_map:
        print("No AC results to save. Skipped updating prev scores.")
    elif not args.debug and ran_default:
        if prompt_yes_no(f"{C_GOLD}Save current results to {prev_scores_label}?{C_RESET}"):
            existing_prev = load_scores(prev_scores_file)
            existing_prev.update(current_scores_map)
            save_scores(existing_prev, prev_scores_file)
            print(f"{C_GOLD}Saved current results to {prev_scores_label}{C_RESET}")
        else:
            print("Skipped saving current results.")

if __name__ == "__main__":
    main()
