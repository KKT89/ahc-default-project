import argparse
import config_util
import os
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(
        description="Generate test cases using the gen binary.",
        epilog="Unknown options are forwarded to the gen binary (e.g. --M=4 --U=3).",
    )
    parser.add_argument("L", type=int, help="Start seed (inclusive)")
    parser.add_argument("R", type=int, help="End seed (exclusive)")
    parser.add_argument(
        "--in",
        dest="in_dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: config's testcase_input_dir)",
    )
    args, extra_gen_args = parser.parse_known_args()

    if args.L >= args.R:
        print("Error: L must be less than R.")
        sys.exit(1)

    config = config_util.load_config()
    work_dir = config_util.work_dir()

    if args.in_dir is not None:
        in_dir = os.path.join(work_dir, args.in_dir)
    else:
        in_dir = os.path.join(work_dir, config["paths"]["testcase_input_dir"])
    os.makedirs(in_dir, exist_ok=True)

    gen = os.path.join(work_dir, config["files"]["gen_file"])

    with tempfile.TemporaryDirectory() as tmp_dir:
        seeds_file = os.path.join(tmp_dir, "seeds.txt")
        with open(seeds_file, "w") as f:
            for seed in range(args.L, args.R):
                f.write(f"{seed}\n")

        cmd = [gen, seeds_file, f"--dir={tmp_dir}"] + extra_gen_args
        subprocess.run(cmd, check=True, cwd=work_dir)

        for i in range(args.R - args.L):
            src = os.path.join(tmp_dir, f"{i:04d}.txt")
            dst = os.path.join(in_dir, f"{args.L + i:03d}.txt")
            if os.path.exists(src):
                os.rename(src, dst)


if __name__ == "__main__":
    main()
