import argparse
import build
import config_util as config_util
import subprocess
import time
import os
import json
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

def run_test_case(
    case_str,
    input_file,
    output_file,
    solution_file,
    vis_file,
    score_prefix,
):
    cmd_cpp = [solution_file]
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

    # ビジュアライザ実行
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

def parse_args():
    parser = argparse.ArgumentParser(description="Run local tests for AHC submissions.")
    parser.add_argument("--cases", type=int, default=None, help="Number of test cases to run.")
    parser.add_argument("--range", nargs=2, metavar=("L", "R"), type=int, default=None, help="Run cases with seed IDs in [L, R).")
    parser.add_argument("--jobs", type=int, default=8, help="Number of worker threads.")
    parser.add_argument("--debug", action="store_true", help="Skip confirmations and overwrite prev scores directly.")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.range is not None and args.cases is not None:
        raise ValueError("--range and --cases cannot be used together.")

    config = config_util.load_config()
    build.compile_program(config)

    work_dir = config_util.work_dir()
    score_dir = os.path.join(work_dir, SCORE_DIR_NAME)
    os.makedirs(score_dir, exist_ok=True)

    input_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    output_dir = os.path.join(work_dir, config["paths"]["testcase_output_dir"])
    solution_file = os.path.join(work_dir, config["files"]["sol_file"])
    vis_file = os.path.join(work_dir, config["files"]["vis_file"])
    score_prefix = config["problem"]["score_prefix"]
    objective = config["problem"]["objective"]
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
    prev_scores_map = load_scores(prev_scores_file)
    current_scores_map = {} 

    results = []
    total_diff_prev = 0 # Submit比の累積
    total_diff_best = 0 # 理論値比の累積
    wa_count = 0
    
    cases_to_run = []
    for case_id in selected_cases:
        case_str = f"{case_id:03d}"
        input_file = os.path.join(input_dir, case_str + ".txt")
        output_file = os.path.join(output_dir, case_str + ".txt")
        cases_to_run.append((case_id, case_str, input_file, output_file))

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


    futures_list = []
    max_workers = args.jobs if args.jobs > 0 else 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for case_id, case_str, input_file, output_file in cases_to_run:
            fut = executor.submit(
                run_test_case,
                case_str,
                input_file,
                output_file,
                solution_file,
                vis_file,
                score_prefix,
            )
            futures_list.append((fut, case_id, case_str))

        # ヘッダー出力（幅調整）
        # vsPrev, SumPrev, vsBest, SumBest の順
        print(f"{'Case':<5} {'Score':<10} {'Time':<6} {'Stat':<5} | {'vsPrev':<8} {'Sum':<8} | {'vsBest':<8} {'Sum':<8}")
        print("-" * 75)

        for fut, case_id, case_str in futures_list:
            try:
                result = fut.result()
            except Exception as exc:
                print(f"Error: case {case_str} exception: {exc}")
                continue

            current_score = result["score"]
            if result["status"] == "WA":
                wa_count += 1
            
            # 保存用
            if result["status"] == "AC" and current_score is not None:
                current_scores_map[case_str] = current_score

            # --- Diff計算 ---
            old_prev = prev_scores_map.get(case_str)
            old_best = best_scores_map.get(case_str)
            
            if result["status"] == "AC" and current_score is not None:
                diff_prev = (current_score - old_prev) if old_prev is not None else 0
                diff_best = (current_score - old_best) if old_best is not None else 0
            else:
                diff_prev = None if old_prev is not None else 0
                diff_best = None if old_best is not None else 0
            
            # 初回(記録なし)の場合はDiffを0として計算に含めるか、あるいは表示だけ変えるか
            # ここでは計算には含める（0扱い）
            
            if diff_prev is not None:
                total_diff_prev += diff_prev
            if diff_best is not None:
                total_diff_best += diff_best

            # --- Best更新判定 ---
            is_new_best = False
            should_update_best = False
            
            if result["status"] == "AC" and current_score is not None:
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
            if result["status"] == "AC":
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
            
            # NEWの文字上書き
            if old_prev is None:
                if result["status"] == "AC":
                    prev_str = f"{'NEW':<8}"
                else:
                    prev_str = f"{'-':<8}"
            if old_best is None:
                if result["status"] == "AC":
                    best_str = f"{C_GOLD}{'NEW':<8}{C_RESET}"
                else:
                    best_str = f"{'-':<8}"

            print(f"{p_case} {p_score} {p_time} {p_status} | {prev_str} {prev_sum_str} | {best_str} {best_sum_str}")

            results.append(result)

    if not results:
        return

    total_score = sum(r['score'] for r in results if r.get('status') == "AC" and r.get('score') is not None)
    
    print("-" * 75)
    print(f"Total Score: {total_score:,d}")
    print(f"WA Count   : {wa_count}")

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
