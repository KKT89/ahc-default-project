"""C++ ソースから HP_PARAM / META_PARAM 宣言を抽出する。

optuna_manager.py(探索対象の列挙)と gen_meta_params.py(カテゴリ別展開)で共用。
単純な数値リテラルのみ対応し、それ以外の宣言はスキップする。
"""

import re

_HP_RE = re.compile(r"\bHP_PARAM\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^\)]+)\)")
_META_RE = re.compile(r"\bMETA_PARAM\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^\)]+)\)")


def _to_num(s: str):
    s = s.strip().rstrip(";")
    try:
        if re.match(r"^[+-]?\d+$", s):
            return int(s)
        return float(s)
    except Exception:
        return None


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_].*$", "", name.strip())


def _is_float_type(ty: str) -> bool:
    ty_l = ty.replace("const", "").strip().lower()
    return any(k in ty_l for k in ["double", "float"])


def _read(cpp_path: str) -> str:
    with open(cpp_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def extract_hp_params(cpp_path: str) -> dict:
    """HP_PARAM(type, name, def, low, high) を抽出し params.json 形式の dict を返す。"""
    text = _read(cpp_path)
    ints, floats = [], []
    for m in _HP_RE.finditer(text):
        ty, name, d, lo, hi = (t.strip() for t in m.groups())
        name = _sanitize_name(name)
        v_def, v_lo, v_hi = _to_num(d), _to_num(lo), _to_num(hi)
        if not name or v_def is None or v_lo is None or v_hi is None:
            continue
        rec = {
            "name": name,
            "lower": v_lo,
            "upper": v_hi,
            "value": v_def,
            "used": True,
        }
        if _is_float_type(ty):
            floats.append(rec)
        else:
            ints.append(rec)
    return {"integer_params": ints, "float_params": floats}


def extract_meta_params(cpp_path: str) -> list:
    """META_PARAM(type, name, def) を抽出する。"""
    text = _read(cpp_path)
    out = []
    for m in _META_RE.finditer(text):
        ty, name, d = (t.strip() for t in m.groups())
        name = _sanitize_name(name)
        v_def = _to_num(d)
        if not name or v_def is None:
            continue
        out.append({"name": name, "is_float": _is_float_type(ty), "value": v_def})
    return out


def param_registry(cpp_path: str) -> dict:
    """name -> {is_float, lower?, upper?} の一覧(HP_PARAM + META_PARAM)。"""
    registry = {}
    hp = extract_hp_params(cpp_path)
    for rec in hp["integer_params"]:
        registry[rec["name"]] = {"is_float": False, "lower": rec["lower"], "upper": rec["upper"]}
    for rec in hp["float_params"]:
        registry[rec["name"]] = {"is_float": True, "lower": rec["lower"], "upper": rec["upper"]}
    for rec in extract_meta_params(cpp_path):
        registry[rec["name"]] = {"is_float": rec["is_float"]}
    return registry
