import argparse
import build
import cpp_params
import json
import numpy as np
import os
import optuna
import config_util
import features as features_mod
import meta_report
import shutil
import sys
import time
import uuid
import warnings
import subprocess
from optuna.storages import RDBStorage
from optuna.exceptions import ExperimentalWarning

warnings.filterwarnings("ignore", category=ExperimentalWarning)


def suggest_parameters(trial, json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    params = {}
    # 整数パラメータ
    for p in data.get("integer_params", []):
        if p.get("used", False):
            name = p["name"]
            low, high = p["lower"], p["upper"]
            params[name] = trial.suggest_int(name, low, high)

    # 浮動小数点パラメータ
    for p in data.get("float_params", []):
        if p.get("used", False):
            name = p["name"]
            low, high = p["lower"], p["upper"]
            log = p.get("log", False)
            params[name] = trial.suggest_float(name, low, high, log=log)

    return params


def _write_params_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def objective(trial, case_strs, input_dir, output_dir, sol_file, vis_file, score_prefix, param_json_file, env_prefix: str = "HP_"):
    params = suggest_parameters(trial, param_json_file)

    # 固定順序(Prunerの影響を安定化): 環境変数 OPTUNA_OBJECTIVE_SEED で制御
    seed_env = os.environ.get("OPTUNA_OBJECTIVE_SEED")
    if seed_env is not None:
        try:
            seed_val = int(seed_env)
        except Exception:
            seed_val = 0
        rng = np.random.default_rng(seed_val)
        order = rng.permutation(len(case_strs))
    else:
        order = np.random.permutation(len(case_strs))

    results = []
    for idx in order:
        case_str = case_strs[int(idx)]
        input_file = os.path.join(input_dir, case_str + ".txt")
        uid = uuid.uuid4().hex[:8]
        output_file = os.path.join(output_dir, uid + ".txt")
        if not os.path.exists(input_file):
            print(f"Error: {input_file} was not found.")
            sys.exit(1)

        # Optuna の各試行で得たパラメータを環境変数として子プロセスへ注入
        # Run solution with params injected via environment variables
        env = os.environ.copy()
        for k, v in params.items():
            env[f"{env_prefix}{k}"] = str(v)
        with open(input_file, "r") as fin, open(output_file, "w") as fout:
            subprocess.run(
                [sol_file],
                stdin=fin,
                stdout=fout,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
                env=env,
            )
        # Score via vis
        res = subprocess.run(
            [vis_file, input_file, output_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        score = -1
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith(score_prefix):
                try:
                    score = int(line.split("=")[-1].strip())
                except Exception:
                    score = -1
                break
        # cleanup temporary output
        try:
            if os.path.exists(output_file):
                os.remove(output_file)
        except Exception:
            pass
        if score <= 0:
            results.append(-1)
        else:
            results.append(score)
        trial.report(score, step=int(case_str))
        if trial.should_prune():
            print(f"Trial pruned at case {case_str} with intermediate avg score {sum(results) / len(results):.2f}")
            return sum(results) / len(results)
    avg_score = sum(results) / len(results)
    print(f"Trial finished. Params={params}, avg_score={avg_score:.2f}")
    return avg_score


def parse_filters(filter_args):
    """--filter の KEY=VALUE / KEY=LO..HI(両端含む)形式をパースする。"""
    filters = []
    for spec in filter_args or []:
        key, sep, val = spec.partition("=")
        key, val = key.strip(), val.strip()
        if not sep or not key or not val:
            raise ValueError(f"Invalid --filter: {spec} (expected KEY=VALUE or KEY=LO..HI)")
        if ".." in val:
            lo_s, hi_s = val.split("..", 1)
            lo = features_mod._to_number(lo_s.strip())
            hi = features_mod._to_number(hi_s.strip())
            if isinstance(lo, str) or isinstance(hi, str):
                raise ValueError(f"Invalid --filter range: {spec}")
            filters.append((key, ("range", lo, hi)))
        else:
            filters.append((key, ("eq", features_mod._to_number(val))))
    return filters


def apply_filters(case_strs, features_by_case, filters):
    selected = []
    for case_str in case_strs:
        feats = features_by_case.get(case_str)
        if feats is None:
            continue
        ok = True
        for key, cond in filters:
            if key not in feats:
                ok = False
                break
            v = feats[key]
            if cond[0] == "range":
                if isinstance(v, str) or not (cond[1] <= v <= cond[2]):
                    ok = False
                    break
            else:
                if v != cond[1] and str(v) != str(cond[1]):
                    ok = False
                    break
        if ok:
            selected.append(case_str)
    return selected


def optimize_study(storage, study_name, direction, objective_fn, n_trials, n_jobs):
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction=direction,
        pruner=optuna.pruners.WilcoxonPruner(p_threshold=0.1),
    )
    if n_trials > 0:
        study.optimize(objective_fn, n_trials=n_trials, n_jobs=n_jobs)
    return study


def write_meta_params(config, work_dir, study_dir, axes, category_results):
    """カテゴリ別ベストパラメータを meta_params.json に書き出す(既存分とマージ)。"""
    meta_params_name = config["files"].get("meta_params_file", "meta_params.json")
    root_path = os.path.join(work_dir, meta_params_name)
    data = {"axes": list(axes), "categories": {}}
    if os.path.exists(root_path):
        try:
            with open(root_path, "r") as f:
                old = json.load(f)
            if old.get("axes") == list(axes):
                data["categories"] = old.get("categories", {})
            else:
                print(f"Warning: axes changed ({old.get('axes')} -> {list(axes)}). Existing categories were discarded.")
        except json.JSONDecodeError:
            pass
    data["categories"].update(category_results)
    data["categories"] = dict(sorted(data["categories"].items()))

    for path in (root_path, os.path.join(study_dir, meta_params_name)):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[Done] Wrote per-category best params to {root_path}")
    return root_path


def main():
    # コマンドライン引数をパース
    parser = argparse.ArgumentParser(
        description="Optuna study with parallel trials."
    )
    parser.add_argument(
        "--dir",
        help="Directory to store study results.",
        default=None,
    )
    parser.add_argument(
        "--last",
        help="Use the most recent study directory under optuna work dir.",
        action="store_true",
    )
    parser.add_argument(
        "--zero",
        help="Run with n_trials = 0 (skip optimization).",
        action="store_true",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=500,
        help="Number of trials per study (default: 500).",
    )
    parser.add_argument(
        "--cases",
        type=int,
        default=50,
        help="Max cases per study (0 = all available; default: 50).",
    )
    parser.add_argument(
        "--in",
        dest="in_dir",
        default=None,
        metavar="DIR",
        help="Override testcase input directory (default: config's testcase_input_dir).",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=None,
        metavar="KEY=V|KEY=LO..HI",
        help="Filter cases by features (repeatable, both ends inclusive). e.g. --filter N=10..13 --filter W=0",
    )
    parser.add_argument(
        "--by-category",
        action="store_true",
        dest="by_category",
        help="Run one study per category (features.py AXES/BINS) and write meta_params.json.",
    )
    args = parser.parse_args()

    # 設定読み込み
    config = config_util.load_config()
    work_dir = config_util.work_dir()
    optuna_work_dir = os.path.join(work_dir, config["paths"]["optuna_work_dir"])

    if not os.path.exists(optuna_work_dir):
        os.makedirs(optuna_work_dir, exist_ok=True)

    if args.last:
        # optuna_work_dir 配下のサブディレクトリを辞書順でソートして最新を取得
        subs = [d for d in os.listdir(optuna_work_dir) if os.path.isdir(os.path.join(optuna_work_dir, d))]
        if not subs:
            print(f"Error: no study directories found in {optuna_work_dir}", file=sys.stderr)
            sys.exit(1)
        latest = sorted(subs)[-1]
        study_dir = os.path.join(optuna_work_dir, latest)
    elif args.dir:
        study_dir = args.dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        study_dir = os.path.join(optuna_work_dir, f"study_{timestamp}")

    if not os.path.exists(study_dir):
        build.compile_program(config)
        cpp_file = os.path.join(work_dir, config["files"]["cpp_file"])
        sol_file = os.path.join(work_dir, config["files"]["sol_file"])
        param_json_name = config["files"]["optuna_params_file"]

        if not os.path.isfile(cpp_file):
            print(f"Error: C++ file not found: {cpp_file}", file=sys.stderr)
            sys.exit(1)

        os.makedirs(study_dir, exist_ok=True)
        print(f"Created a new directory {study_dir}.")
        cpp_copy = shutil.copy(cpp_file, study_dir)
        shutil.copy(sol_file, study_dir)

        # Generate params.json by extracting HP_PARAM macros from the copied cpp
        params_data = cpp_params.extract_hp_params(cpp_copy)
        param_json_file = os.path.join(study_dir, param_json_name)
        _write_params_json(params_data, param_json_file)

    if args.in_dir is not None:
        input_dir = args.in_dir if os.path.isabs(args.in_dir) else os.path.join(work_dir, args.in_dir)
        input_dir = os.path.normpath(input_dir)
    else:
        input_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    output_dir = study_dir
    sol_file = os.path.join(study_dir, config["files"]["sol_file"])
    vis_file = os.path.join(work_dir, config["files"]["vis_file"])
    score_prefix = config["problem"]["score_prefix"]
    param_json_file = os.path.join(study_dir, config["files"]["optuna_params_file"])
    direction = config["problem"]["objective"]

    # 対象ケースの決定(features によるフィルタ込み)
    all_case_strs = features_mod.list_case_strs(input_dir)
    if not all_case_strs:
        print(f"Error: no testcases found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    filters = parse_filters(args.filter)
    features_by_case = {}
    if filters or args.by_category:
        features_by_case = features_mod.load_features(input_dir, all_case_strs)

    case_strs = all_case_strs
    if filters:
        case_strs = apply_filters(case_strs, features_by_case, filters)
        print(f"Filter matched {len(case_strs)}/{len(all_case_strs)} cases.")
        if not case_strs:
            print("Error: no cases matched the filter.", file=sys.stderr)
            sys.exit(1)

    def cap_cases(cases):
        if args.cases > 0:
            return cases[:args.cases]
        return cases

    # DBファイルパス(SQLite)
    optuna_db_file = config["files"]["optuna_db_file"]
    db_path = os.path.join(study_dir, optuna_db_file)
    db_url = f"sqlite:///{db_path}?cache=shared&mode=wal"

    # SQLite ストレージの作成
    storage = RDBStorage(
        url=db_url,
        engine_kwargs={
            "connect_args": {
                # ロック待ちを最大20秒まで許容
                "timeout": 20.0,
            }
        },
    )

    n_trials = 0 if args.zero else args.trials

    # 必要なら環境変数名にプレフィックスを付けたい場合はここで設定(例: "HP_")
    # 既定はヘッダのデフォルトに合わせて HP_
    env_prefix = os.environ.get("OPTUNA_PARAM_ENV_PREFIX", "HP_")

    # 並列度は環境変数 OPTUNA_N_JOBS で上書き可能(デフォルト: -1 = 最大)
    n_jobs_env = os.environ.get("OPTUNA_N_JOBS")
    try:
        n_jobs = int(n_jobs_env) if n_jobs_env is not None else -1
    except Exception:
        n_jobs = -1

    def make_objective(target_cases):
        return lambda trial: objective(
            trial, target_cases, input_dir, output_dir, sol_file, vis_file,
            score_prefix, param_json_file, env_prefix=env_prefix,
        )

    if args.by_category:
        # カテゴリごとに study を作り、ベストパラメータを meta_params.json へ集約する
        binner = features_mod.build_binner(features_by_case)
        if binner is None:
            print("Error: --by-category には features.py の AXES 設定が必要です。", file=sys.stderr)
            sys.exit(1)
        groups, unmatched = meta_report.group_by_category(case_strs, features_by_case, binner)
        if not groups:
            print("Error: no cases were assigned to any category.", file=sys.stderr)
            sys.exit(1)
        if unmatched:
            print(f"Note: {len(unmatched)} cases are uncategorized and will be skipped.")

        category_results = {}
        for cat, cases in groups.items():
            cat_key = meta_report.category_key(cat)
            target_cases = cap_cases(cases)
            print()
            print(f"=== Category {cat_key} ({len(target_cases)} cases, {n_trials} trials) ===")
            study = optimize_study(
                storage, cat_key, direction, make_objective(target_cases), n_trials, n_jobs,
            )
            try:
                best_params = study.best_params
                best_score = study.best_value
            except ValueError:
                print(f"Skipped {cat_key}: no completed trials.")
                continue
            print(f"Best for {cat_key}: {best_params} (score={best_score:.2f})")
            category_results[cat_key] = {
                "params": best_params,
                "best_score": best_score,
                "n_cases": len(target_cases),
                "n_trials": len(study.trials),
            }

        if category_results:
            write_meta_params(config, work_dir, study_dir, binner.axes, category_results)
        return

    # 単一 study(従来動作)
    target_cases = cap_cases(case_strs)
    print(f"Optimizing on {len(target_cases)} cases, {n_trials} trials.")
    study = optimize_study(
        storage, optuna_db_file, direction, make_objective(target_cases), n_trials, n_jobs,
    )

    # 最終ベストパラメータで study_dir の JSON の "value" を更新し、ルートの params.json にも反映
    best = study.best_params
    best_score = study.best_value

    def _apply_best_to_json(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key in ("integer_params", "float_params"):
            for p in data.get(key, []):
                name = p.get("name")
                if p.get("used") and name in best:
                    p["value"] = best[name]
        data["best_score"] = best_score
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    # study ディレクトリ側
    _apply_best_to_json(param_json_file)
    print(f"[Done] Updated {param_json_file} with best params: {best}")

    # ルート側
    root_param_json = os.path.join(work_dir, config["files"]["optuna_params_file"])
    if os.path.isfile(root_param_json):
        _apply_best_to_json(root_param_json)
        print(f"[Done] Also updated {root_param_json} with best params.")

    print("Best params:", study.best_params)
    print("Best score:", study.best_value)


if __name__ == "__main__":
    main()
