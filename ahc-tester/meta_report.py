"""特徴量のビン分けとカテゴリ別集計表のレンダリング。

features.py が抽出した特徴量を軸ごとのビンに割り当て、
カテゴリ(各軸のビンの組)単位の集計表を描画する。
"""

import math
import re

C_RESET = "\033[0m"
C_RED   = "\033[31m"
C_GREEN = "\033[32m"
C_GOLD  = "\033[33;1m"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def visible_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def pad(s: str, width: int, align: str = "right") -> str:
    fill = width - visible_len(s)
    if fill <= 0:
        return s
    return (" " * fill + s) if align == "right" else (s + " " * fill)


def _fmt_num(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _is_int_like(v) -> bool:
    return isinstance(v, int) or (isinstance(v, float) and v.is_integer())


class Binner:
    """各軸の特徴量をビンに割り当てる。

    ビン定義は2種類:
    - edges:  境界値リスト [e0, e1, ..., ek] → [e0,e1) [e1,e2) ... [e(k-1),ek](最後のみ上端含む)
    - values: 値ごとに1ビン(離散値・文字列向け)
    """

    AUTO_VALUE_BIN_MAX = 8  # distinct がこれ以下なら値ごとのビンにする
    AUTO_QUANTILE_BINS = 5  # それ以外は分位点で分割するビン数

    def __init__(self, axes, specs):
        self.axes = list(axes)
        self._specs = specs
        self._label_index = {
            axis: {label: i for i, label in enumerate(spec["labels"])}
            for axis, spec in specs.items()
        }

    @classmethod
    def build(cls, features_by_case: dict, axes, bins_config: dict | None = None) -> "Binner":
        bins_config = bins_config or {}
        specs = {}
        for axis in axes:
            observed = [f[axis] for f in features_by_case.values() if axis in f]
            if axis in bins_config:
                specs[axis] = cls._edges_spec(axis, list(bins_config[axis]))
            else:
                specs[axis] = cls._auto_spec(axis, observed)
        return cls(axes, specs)

    @staticmethod
    def _edges_spec(axis, edges):
        if len(edges) < 2:
            raise ValueError(f"axis '{axis}': bins には境界値を2つ以上指定してください")
        edges = sorted(edges)
        int_style = all(_is_int_like(e) for e in edges)
        labels = []
        n_bins = len(edges) - 1
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if int_style:
                lo_i = int(lo)
                # 中間ビンは [lo, hi) なので整数なら hi-1 がラベル上の上端
                hi_i = int(hi) if i == n_bins - 1 else int(hi) - 1
                labels.append(f"{axis}{lo_i}" if lo_i == hi_i else f"{axis}{lo_i}-{hi_i}")
            else:
                labels.append(f"{axis}{_fmt_num(lo)}-{_fmt_num(hi)}")
        return {"kind": "edges", "edges": edges, "labels": labels}

    @classmethod
    def _auto_spec(cls, axis, observed):
        if not observed:
            raise ValueError(f"axis '{axis}': 特徴量に値がありません(features.py の設定を確認)")
        if any(isinstance(v, str) for v in observed):
            distinct = sorted({str(v) for v in observed})
            return {
                "kind": "values",
                "values": distinct,
                "labels": [f"{axis}={v}" for v in distinct],
            }
        distinct = sorted(set(observed))
        if len(distinct) <= cls.AUTO_VALUE_BIN_MAX:
            return {
                "kind": "values",
                "values": distinct,
                "labels": [f"{axis}{_fmt_num(v)}" for v in distinct],
            }
        # 分位点ベースの自動ビン分け
        vals = sorted(observed)
        n = len(vals)
        k = cls.AUTO_QUANTILE_BINS
        edges = []
        for i in range(k + 1):
            e = vals[round(i * (n - 1) / k)]
            if not edges or e != edges[-1]:
                edges.append(e)
        if len(edges) < 2:
            return {
                "kind": "values",
                "values": distinct,
                "labels": [f"{axis}{_fmt_num(v)}" for v in distinct],
            }
        return cls._edges_spec(axis, edges)

    def spec(self, axis) -> dict:
        return self._specs[axis]

    def labels(self, axis) -> list:
        return list(self._specs[axis]["labels"])

    def index(self, axis, value):
        spec = self._specs[axis]
        if spec["kind"] == "edges":
            if isinstance(value, str):
                return None
            edges = spec["edges"]
            for i in range(len(edges) - 1):
                if value < edges[i + 1]:
                    return max(0, i)
            return len(edges) - 2  # 上端以上は最後のビンに丸める
        values = spec["values"]
        if any(isinstance(v, str) for v in values):
            return values.index(str(value)) if str(value) in values else None
        if isinstance(value, str):
            return None
        if value in values:
            return values.index(value)
        # 未知の数値は最も近い値のビンへ
        return min(range(len(values)), key=lambda i: abs(values[i] - value))

    def category(self, feats: dict):
        """特徴量 dict をカテゴリ(軸ごとのビンラベルの tuple)へ。割り当て不能なら None。"""
        if feats is None:
            return None
        labels = []
        for axis in self.axes:
            if axis not in feats:
                return None
            idx = self.index(axis, feats[axis])
            if idx is None:
                return None
            labels.append(self._specs[axis]["labels"][idx])
        return tuple(labels)

    def label_order(self, axis, label) -> int:
        return self._label_index[axis].get(label, 10**9)

    def sort_key(self, category: tuple):
        return tuple(self.label_order(axis, label) for axis, label in zip(self.axes, category))


def category_key(category: tuple) -> str:
    """meta_params.json などで使うカテゴリの文字列キー。"""
    return ",".join(category)


def group_by_category(case_strs, features_by_case, binner: Binner):
    """ケースをカテゴリごとにグループ化する。割り当て不能ケースは unmatched へ。"""
    groups = {}
    unmatched = []
    for case_str in case_strs:
        cat = binner.category(features_by_case.get(case_str))
        if cat is None:
            unmatched.append(case_str)
        else:
            groups.setdefault(cat, []).append(case_str)
    ordered = dict(sorted(groups.items(), key=lambda kv: binner.sort_key(kv[0])))
    return ordered, unmatched


def render_matrix(binner: Binner, groups: dict, cell_fn) -> list:
    """カテゴリ別集計のマトリクスを文字列リストとして描画する。

    行 = 先頭側の軸(複数あればラベルを連結)、列 = 最後の軸。
    cell_fn(category, cases) は ANSI カラー込みの文字列を返してよい。
    """
    last_axis = binner.axes[-1]
    col_order = {label: i for i, label in enumerate(binner.labels(last_axis))}
    col_labels = sorted({cat[-1] for cat in groups}, key=lambda l: col_order.get(l, 10**9))
    row_keys = sorted(
        {cat[:-1] for cat in groups},
        key=lambda rk: tuple(binner.label_order(a, l) for a, l in zip(binner.axes[:-1], rk)),
    )

    cells = {}
    for cat, cases in groups.items():
        cells[(cat[:-1], cat[-1])] = cell_fn(cat, cases)

    row_header = ",".join(binner.axes[:-1]) if len(binner.axes) > 1 else ""
    row_names = {rk: (",".join(rk) if rk else "all") for rk in row_keys}
    row_width = max([visible_len(row_header)] + [visible_len(n) for n in row_names.values()])
    col_widths = {
        cl: max([visible_len(cl)] + [visible_len(cells.get((rk, cl), "")) for rk in row_keys])
        for cl in col_labels
    }

    lines = []
    header = pad(row_header, row_width, "left") + " |"
    for cl in col_labels:
        header += " " + pad(cl, col_widths[cl])
    lines.append(header)
    lines.append("-" * visible_len(header))
    for rk in row_keys:
        line = pad(row_names[rk], row_width, "left") + " |"
        for cl in col_labels:
            line += " " + pad(cells.get((rk, cl), ""), col_widths[cl])
        lines.append(line)
    return lines


def vs_ref_cell(cases, eff_of, ref_map, objective) -> str:
    """カテゴリ内ケースの基準スコア比(幾何平均)をセル文字列にする。

    比は「1 より大きいほど良い」に正規化される(minimize では ref/cur)。
    """
    diffs = []
    for case_str in cases:
        ref = ref_map.get(case_str)
        cur = eff_of(case_str)
        if ref is None or cur is None:
            continue
        ref_eff = max(1, int(ref))
        if objective == "minimize":
            diffs.append(math.log(float(ref_eff)) - math.log(float(cur)))
        else:
            diffs.append(math.log(float(cur)) - math.log(float(ref_eff)))
    n = len(cases)
    if not diffs:
        return f"n={n}"
    ratio = math.exp(sum(diffs) / len(diffs))
    text = f"x{ratio:.3f}({n})"
    if ratio > 1 + 1e-9:
        return f"{C_GREEN}{text}{C_RESET}"
    if ratio < 1 - 1e-9:
        return f"{C_RED}{text}{C_RESET}"
    return text
