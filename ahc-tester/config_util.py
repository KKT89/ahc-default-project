import os
import sys
import tomllib

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CONFIG_FILE = os.path.join(ROOT_DIR, "config.toml")


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: {CONFIG_FILE} was not found. Please run setup.py first.")
        sys.exit(1)
    with open(CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


def work_dir() -> str:
    return ROOT_DIR
