import config_util as config_util
import sys
import os
import subprocess
import uuid

def main_with_params(L: int, R: int):
    if L > R:
        print("L must be less than or equal to R.")
        sys.exit(1)

    config = config_util.load_config()
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    work_dir = config_util.work_dir()

    unique_id = uuid.uuid4().hex
    tmp_dir = os.path.join(SCRIPT_DIR, f"tmp_dir_{unique_id}")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_file = os.path.join(SCRIPT_DIR, f"tmp_{unique_id}.txt")
    with open(tmp_file, "w") as f:
        for x in range(L, R):
            f.write(f"{x}\n")

    gen = os.path.join(work_dir, config["files"]["gen_file"])
    cmd = [gen, tmp_file, f"--dir={tmp_dir}"]
    subprocess.run(cmd, check=True, cwd=SCRIPT_DIR)

    in_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    os.makedirs(in_dir, exist_ok=True)
    total_cases = R - L
    for i in range(total_cases):
        src_path = os.path.join(tmp_dir, f"{i:04d}.txt")
        new_file_name = f"{L + i:03d}.txt"
        dst_path = os.path.join(in_dir, new_file_name)
        if os.path.exists(src_path):
            os.rename(src_path, dst_path)

    os.remove(tmp_file)
    os.rmdir(tmp_dir)


def main():
    if len(sys.argv) < 3:
        print("Usage: python make_test.py <L> <R>")
        sys.exit(1)
    L = int(sys.argv[1])
    R = int(sys.argv[2])
    main_with_params(L, R)


if __name__ == "__main__":
    main()
