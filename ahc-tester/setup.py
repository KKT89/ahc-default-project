import copy
import os
import subprocess
import argparse

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_FILE = os.path.join(ROOT_DIR, "config.toml")

DEFAULT_CONFIG = {
    "paths": {
        "tools_dir": "tools",                               # ツールのディレクトリ
        "testcase_input_dir": "in",                         # テストケースの入力ファイルがあるディレクトリ
        "testcase_output_dir": "out",                       # テストケースの出力ファイルを保存するディレクトリ
        "optuna_work_dir": "optuna_work",                   # Optuna 用の作業ディレクトリ
    },
    "files": {
        "cpp_file": "main.cpp",                             # メインのソースファイル
        "combined_file": "combined.cpp",                    # 結合後のソースファイル
        "sol_file": "solution",                             # コンパイルしたプログラムの名前
        "gen_file": "tools/gen",                            # テストケース生成プログラムの名前
        "vis_file": "tools/vis",                            # ビジュアライズプログラムの名前
        "tester_file": "tools/tester",                      # テスタープログラムの名前
        "optuna_db_file": "optuna_study.db",                # Optuna 用のデータベースファイル
        "optuna_params_file": "params.json",                # Optuna 用パラメータ定義ファイル
    },
    "problem": {
        "pretest_count": 150,                               # プレテストの数
        "interactive": False,                               # インタラクティブかどうか
        "objective": "maximize",                            # 最大化 or 最小化
        "score_prefix": "Score =",                          # テスターの出力からスコアを取得するためのプレフィックス
    },
}


def _write_config(cfg, path):
    lines = []
    for section, values in cfg.items():
        lines.append(f"[{section}]\n")
        for k, v in values.items():
            if isinstance(v, bool):
                val = "true" if v else "false"
            elif isinstance(v, (int, float)):
                val = str(v)
            else:
                val = f'"{v}"'
            lines.append(f"{k} = {val}\n")
        lines.append("\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"{path} has been overwritten successfully!\n")


def build_tools_with_cargo(cfg):
    tools_path = os.path.join(ROOT_DIR, cfg["paths"]["tools_dir"])
    cargo_manifest_path = os.path.join(tools_path, "Cargo.toml")

    if not os.path.exists(cargo_manifest_path):
        print(f"Error: {cargo_manifest_path} does not exist.")
        exit(1)

    print("Running rustup update ...")
    result = subprocess.run(["rustup", "update"], capture_output=True, text=True, cwd=ROOT_DIR)
    if result.returncode != 0:
        print("rustup update failed.")
        print(result.stderr)
        exit(1)
    print("rustup update succeeded.\n")

    binary_list = [cfg["files"]["gen_file"], cfg["files"]["vis_file"]]
    if cfg["problem"]["interactive"]:
        binary_list.append(cfg["files"]["tester_file"])

    for binary in binary_list:
        binary_name = os.path.basename(binary)
        cmd = [
            "cargo", "build",
            "--manifest-path", cargo_manifest_path,
            "-r",
            "--bin", binary_name,
        ]
        print(f"Running Cargo build command for {binary_name} ...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR)
        if result.returncode != 0:
            print(f"Cargo build failed for {binary_name}.")
            print(result.stderr)
            exit(1)
        print(f"Cargo build succeeded for {binary_name}.")
        src = os.path.join(tools_path, "target", "release", binary_name)
        dst = os.path.join(ROOT_DIR, binary)
        if os.path.exists(dst):
            os.remove(dst)
        os.rename(src, dst)
        print(f"Moved {binary_name} to {dst}\n")


def _normalize_objective(obj: str) -> str:
    obj = obj.strip().lower()
    if obj in ("max", "maximize"):
        return "maximize"
    if obj in ("min", "minimize"):
        return "minimize"
    raise ValueError("objective must be 'max'/'maximize' or 'min'/'minimize'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Setup ahc-tester: write config.toml and build tools.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "objective",
        choices=["max", "min", "maximize", "minimize"],
        help="Optimization direction: max/min (maximize/minimize)",
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Problem is interactive (default: non-interactive)",
    )
    return parser.parse_args()


def build_config_from_args(args) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["problem"]["objective"] = _normalize_objective(args.objective)
    cfg["problem"]["interactive"] = args.interactive
    return cfg


def main():
    args = parse_args()
    cfg = build_config_from_args(args)
    _write_config(cfg, CONFIG_FILE)
    build_tools_with_cargo(cfg)


if __name__ == "__main__":
    main()
