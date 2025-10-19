import argparse
import build
import config_util as config_util
import subprocess
import time
import os


TLE_FACTOR = 2.5
TLE_MARGIN_RATIO = 0.05  # タイムアウト判定用の緩衝比率（5%余裕）


def failure_score(objective: str) -> int:
    if objective == "maximize":
        return -1
    if objective == "minimize":
        return 10 ** 9
    raise ValueError(f"Unsupported objective: {objective}")


def run_test_case(
    case_str,
    input_file,
    output_file,
    solution_file,
    vis_file,
    score_prefix,
    fail_score,
    tle_limit_ms,
    tle_margin_ratio,
):
    cmd_cpp = [solution_file]

    start_time = time.perf_counter()
    timeout_limit_ms = tle_limit_ms * (1.0 + tle_margin_ratio)
    timeout_sec = timeout_limit_ms / 1000.0
    try:
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            subprocess.run(
                cmd_cpp,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
    except subprocess.TimeoutExpired:
        elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "case": case_str,
            "score": fail_score,
            "elapsed_time": elapsed_time_ms,
            "tle": True,
        }

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000.0

    if elapsed_time_ms > timeout_limit_ms:
        return {
            "case": case_str,
            "score": fail_score,
            "elapsed_time": elapsed_time_ms,
            "tle": True,
        }

    res = subprocess.run(
        [vis_file, input_file, output_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )

    score = fail_score
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith(score_prefix):
            try:
                score = int(line.split("=")[-1].strip())
            except Exception:
                score = fail_score
            break

    return {
        "case": case_str,
        "score": score,
        "elapsed_time": elapsed_time_ms,
        "tle": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Run local tests for AHC submissions.")
    parser.add_argument(
        "--cases",
        type=int,
        default=None,
        help="Number of test cases to run (default: value from config, typically 150).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # コンパイル
    config = config_util.load_config()
    build.compile_program(config)

    work_dir = config_util.work_dir()
    input_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    output_dir = os.path.join(work_dir, config["paths"]["testcase_output_dir"])
    solution_file = os.path.join(work_dir, config["files"]["sol_file"])
    vis_file = os.path.join(work_dir, config["files"]["vis_file"])
    score_prefix = config["problem"]["score_prefix"]
    objective = config["problem"]["objective"]
    fail_score = failure_score(objective)
    tle_limit_ms = config["problem"]["time_limit_ms"] * TLE_FACTOR
    # 非インタラクティブ前提の簡易テスト（interactive は参照のみ）
    is_interactive = config["problem"]["interactive"]

    os.makedirs(output_dir, exist_ok=True)

    available_cases = sorted(
        int(os.path.splitext(fname)[0])
        for fname in os.listdir(input_dir)
        if fname.endswith(".txt") and os.path.splitext(fname)[0].isdigit()
    )
    if not available_cases:
        raise FileNotFoundError(f"No input cases were found under {input_dir}")
    
    # テストケースの実行結果
    config_case_count = config["problem"]["pretest_count"]
    if args.cases is None:
        requested_count = config_case_count
    else:
        if args.cases <= 0:
            raise ValueError("--cases must be positive.")
        requested_count = args.cases

    testcase_count = min(requested_count, len(available_cases))
    if requested_count > len(available_cases):
        print(
            f"Warning: requested {requested_count} cases but only {len(available_cases)} available. "
            f"Using {testcase_count} cases."
        )

    wrong_answer_count = 0
    results = []

    # シンプルな逐次実行に限定
    selected_cases = available_cases[:testcase_count]

    for case_id in selected_cases:
        case_str = f"{case_id:04d}"
        input_file = os.path.join(input_dir, case_str + ".txt")
        output_file = os.path.join(output_dir, case_str + ".txt")
        if not os.path.exists(input_file):
            print(f"Error: {input_file} was not found.")
            continue
        result = run_test_case(
            case_str,
            input_file,
            output_file,
            solution_file,
            vis_file,
            score_prefix,
            fail_score,
            tle_limit_ms,
            TLE_MARGIN_RATIO,
        )
        score = result['score']
        if result['tle']:
            wrong_answer_count += 1
            print(
                f"Error: {result['case']} exceeded TL ({result['elapsed_time']:.2f} ms > {(tle_limit_ms * (1.0 + TLE_MARGIN_RATIO)):.2f} ms)."
            )
            continue
        if score == fail_score:
            wrong_answer_count += 1
            print(f"Error: {result['case']} failed to get score.")
            continue
        results.append(result)
        print(f"seed:{result['case']}  score:{result['score']:,d}  ({result['elapsed_time']:.2f} ms)")

    if len(results) == 0:
        print("No results to display.")
        exit(0)
    
    # スコアの合計と平均を計算
    total_score = sum(result['score'] for result in results)
    avg_score = total_score / len(results)

    # 最大実行時間を取得
    max_time_result = max(results, key=lambda r: r['elapsed_time'])
    max_time = max_time_result['elapsed_time']
    max_time_case = max_time_result['case']

    print(f"----- All test cases finished (total {testcase_count}) -----")
    print(f"Wrong Answers: {wrong_answer_count} / {testcase_count}")
    print(f"Maximum Execution Time: {max_time:.2f} ms (case: {max_time_case})")
    print(f"Total Score: {total_score:,d}")
    print(f"Average Score: {avg_score:.2f}")

    return {
        "wrong_answers": wrong_answer_count,
        "total_score": total_score,
        "avg_score": avg_score,
    }


if __name__ == "__main__":
    main()
