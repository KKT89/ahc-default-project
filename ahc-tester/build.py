import os
import config_util
import subprocess
import sys

# オンラインジャッジ（AtCoder AHC）と同等のコンパイルオプション
RELEASE_FLAGS = ["-DONLINE_JUDGE"]


def compile_program(config, extra_flags=None):
    work_dir = config_util.work_dir()

    cpp_file_path = os.path.join(work_dir, config["files"]["cpp_file"])
    sol_file_path = os.path.join(work_dir, config["files"]["sol_file"])
    if not os.path.exists(cpp_file_path):
        print(f"Error: {cpp_file_path} was not found.")
        sys.exit(1)

    cmd = ["g++", cpp_file_path, "-O2"]
    if extra_flags:
        cmd += list(extra_flags)
    cmd += ["-o", sol_file_path]
    print(f"Building: {config['files']['cpp_file']} -> {config['files']['sol_file']}")

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=work_dir)
    if result.returncode != 0:
        print("Build failed.")
        print(result.stderr)
        sys.exit(1)
    else:
        print("Build succeeded.")


if __name__ == "__main__":
    config = config_util.load_config()
    compile_program(config)
