# ahc-tester

AHC のローカルテスト・パラメータ最適化を行うツール群です。

## セットアップ

`config.toml` をプロジェクトルートに生成し、公式ローカルテストツール（`gen` / `vis` / `tester`）を cargo でビルドします。

```
$ uv run ahc-tester/setup.py <max|min>
$ uv run ahc-tester/setup.py <max|min> -i   # インタラクティブ問題の場合
```

**引数**

| 引数 | 説明 |
|------|------|
| `objective` | 最適化方向。`max` / `maximize` または `min` / `minimize` |
| `-i, --interactive` | インタラクティブ問題のときに指定。`tester` バイナリを追加でビルドします |

**ヘルプ**

```
$ uv run ahc-tester/setup.py --help
```

## テストケース作成

seed `L` 以上 `R` 未満のテストケースを生成し、`in/{seed:03d}.txt` として保存します。

```
$ uv run ahc-tester/make_test.py L R
$ uv run ahc-tester/make_test.py L R --in in2        # 出力先ディレクトリを指定
$ uv run ahc-tester/make_test.py L R --M=4 --U=3     # gen へのオプションを追加
```

**オプション**

| オプション | 説明 |
|------------|------|
| `--in DIR` | 出力ディレクトリを指定（省略時は config の `testcase_input_dir`） |
| その他 | gen バイナリ固有のオプションをそのまま渡せます（例: `--M=4`, `--U=3`） |

## 特徴量（メタデータ）

テストケースを N や W などの特徴量で分類し、カテゴリ別の集計・チューニングに使います。
コンテストごとに `features.py` 冒頭の設定を編集してください。

```python
HEADER_NAMES = ["N", "W"]                 # 入力1行目のトークン名
AXES = ["N", "W"]                         # 分類・集計に使う軸
BINS = {"N": [10, 12, 14, 16, 18, 20]}    # ビン境界（省略した軸は自動ビン分け）
```

- ビン境界 `[10, 12, 14, ...]` は `[10,12) [12,14) ...` を意味します（最後のビンのみ上端含む）
- 派生統計量（グリッド密度など）が必要な場合は `extract()` を書き換えます
- 抽出結果は `<入力ディレクトリ>/features.json` にキャッシュされます

```
$ uv run ahc-tester/features.py           # 全ケース再抽出 + カテゴリ分布の表示
$ uv run ahc-tester/features.py --in in2
```

設定を変更したら再実行してキャッシュを更新してください。

## ビルド

`main.cpp` を `-O2` でコンパイルし、`solution` バイナリを生成します。
`run_test.py` 実行時は自動でビルドが走るため、単独で呼ぶ機会は少ないです。

```
$ uv run ahc-tester/build.py
```

## テスト実行

ビルドしてからテストを実行し、スコアを表示します。

```
$ uv run ahc-tester/run_test.py
$ uv run ahc-tester/run_test.py --cases 10
$ uv run ahc-tester/run_test.py --range 0 50
$ uv run ahc-tester/run_test.py --in in2      # 別ディレクトリのケースで実行（スコア保存なし）
$ uv run ahc-tester/run_test.py --debug
$ uv run ahc-tester/run_test.py --release
$ uv run ahc-tester/run_test.py --no-save
```

複数の cpp ファイルを渡すと**比較モード**になります（後述）。

```
$ uv run ahc-tester/run_test.py main.cpp experiments/beam.cpp
```

**オプション**

| オプション | 説明 |
|------------|------|
| `CPP ...` | 比較モード。指定した cpp をそれぞれビルド・実行して比較（best/prev トラッキングなし） |
| `--cases N` | 実行件数を指定。省略時は `config.toml` の `pretest_count`（デフォルト 150） |
| `--range L R` | seed が `[L, R)` のケースのみ実行。`--cases` と同時指定不可 |
| `--in DIR` | 入力ディレクトリを上書き指定。スコアトラッキング（best/prev の読み書き）は無効化される |
| `--jobs N` | 並列スレッド数（デフォルト 8） |
| `--debug` | `-DDEBUG` 付きでビルド |
| `--release` | `-DONLINE_JUDGE` 付きでビルド |
| `--no-save` | prev スコアの保存プロンプトをスキップ |
| `--tag LABEL` | 保存するラン結果のラベルを指定（比較モードでは cpp ごとに繰り返し指定） |

**出力列**

| 列 | 説明 |
|----|------|
| `vsPrev` | 前回提出スコアとの差分 |
| `vsBest` | 全期間ベストスコアとの差分 |
| `vsPrevN` | 前回提出スコアとの差分（0〜1 に正規化）。ベスト更新時は新スコアを基準に正規化 |
| `Sum` | 選択ケース全体の累積差分 |

改善は緑、悪化は赤、ベスト更新は金色で表示します。
`features.py` の `AXES` が設定されていれば、末尾にカテゴリ別の vsBest 集計表
（セル = ベスト比の幾何平均とケース数）も表示されます。

**スコアの保存**

- `score/best_scores.json`：全期間ベストスコアを自動更新
- `score/prev_scores.json`：スコアトラッキングが有効な実行の終了時に保存するか確認します（`--no-save` 時はスキップ）
- `best/<case>.out`：ベスト更新時にその出力ファイルを自動保存
- `results/<日時>_<ラベル>.json`：毎ランのケース別結果を自動保存（`report.py` で閲覧・比較）

## 複数解法の比較と結果の閲覧

run_test に cpp を複数渡すと、それぞれをビルド・実行して比較表を出します。

```
$ uv run ahc-tester/run_test.py main.cpp experiments/beam.cpp
$ uv run ahc-tester/run_test.py main.cpp main.cpp --tag old --tag new --release
```

- ケース別スコア表（勝者に `*`）
- 解法別サマリ（Total / WA / MaxTime / mean(log) / 先頭解法を基準とした幾何平均比）
- カテゴリ別の勝者表（セル = 勝者ラベルと次点との幾何平均比、先頭解法以外の勝ちは緑）

バイナリは `bin/<ラベル>`、出力は `out/<ラベル>/` に分かれて保存されます。

保存済みのラン結果は `report.py` でいつでも再閲覧・比較できます。

```
$ uv run ahc-tester/report.py results/A.json results/B.json
$ uv run ahc-tester/report.py results/A.json    # 単独ランは vsBest のカテゴリ表を表示
```

**オプション**

| オプション | 説明 |
|------------|------|
| `--no-cases` | ケース別スコア表を省略 |
| `--in DIR` | 特徴量参照用の入力ディレクトリを上書き（省略時は結果ファイルに記録されたもの） |

## optuna

`HP_PARAM` マクロで宣言されたハイパーパラメータを Optuna で最適化します。

```
$ uv run ahc-tester/optuna_manager.py            # 新規 study を作成して最適化
$ uv run ahc-tester/optuna_manager.py --last     # 最新の study を再開
$ uv run ahc-tester/optuna_manager.py --dir <ディレクトリ>  # 指定 study を再開
$ uv run ahc-tester/optuna_manager.py --zero     # 試行を実行せずパラメータだけ更新
$ uv run ahc-tester/optuna_manager.py --filter N=10..13 --filter W=0..4   # ケースを絞って最適化
$ uv run ahc-tester/optuna_manager.py --by-category                       # カテゴリごとに最適化
```

**オプション**

| オプション | 説明 |
|------------|------|
| `--dir <dir>` | 再開する study ディレクトリを指定 |
| `--last` | `optuna_work/` 配下で最新のサブディレクトリを自動選択（`--dir` より優先） |
| `--zero` | `n_trials=0` で実行。パラメータを即時更新したいときに使用 |
| `--trials N` | study あたりの試行数（デフォルト 500） |
| `--cases N` | study あたりの最大ケース数（`0` で全ケース。デフォルト 50） |
| `--in DIR` | 入力ディレクトリを上書き指定（チューニング専用ケースセットなど） |
| `--filter K=V` / `K=LO..HI` | 特徴量でケースを絞り込み（繰り返し指定で AND、両端含む） |
| `--by-category` | カテゴリ（`features.py` の AXES/BINS）ごとに study を作成 |

新規 study 作成時は `main.cpp` から `HP_PARAM` マクロを自動抽出して `params.json` を生成します。
最適化終了後はベストパラメータを `params.json` の `value` フィールドに書き戻します。

### カテゴリ別チューニング

`--by-category` はカテゴリごとに独立した study を作成し（同一 DB 内に study 名 = カテゴリ名で保存）、
各カテゴリのベストパラメータをルートの `meta_params.json` に集約します（既存の内容とマージ）。

```
$ uv run ahc-tester/optuna_manager.py --by-category --trials 100
$ uv run ahc-tester/optuna_manager.py --by-category --last --zero   # 再集計のみ
```

`meta_params.json` は手動編集も想定しています。例えば run_test の比較で決めた解法切り替えを
`META_PARAM` のフラグ値としてカテゴリごとに追記できます。

```json
{
  "axes": ["N", "W"],
  "categories": {
    "N10-11,W0-1": {"params": {"T0": 1.85, "STRATEGY": 1}}
  }
}
```

**環境変数**

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `OPTUNA_N_JOBS` | 並列試行数（`-1` で最大） | `-1` |
| `OPTUNA_PARAM_ENV_PREFIX` | パラメータ注入時の環境変数プレフィックス | `HP_` |
| `OPTUNA_OBJECTIVE_SEED` | テストケースのシャッフル seed | ランダム |

### HP_PARAM / META_PARAM マクロ

`lib/hp_params.hpp` でパラメータを宣言します。

```cpp
HP_PARAM(type, name, default, low, high)   // Optuna の探索対象
META_PARAM(type, name, default)            // 探索対象にしない切替フラグなど
```

- 実行時は環境変数 `HP_{name}` から値を読み込みます（`ONLINE_JUDGE` ビルドでは定数化）
- Optuna 実行時は `HP_PARAM` だけが自動抽出され、`params.json` に記録されます
- `META_PARAM` は探索されませんが、カテゴリ別展開（下記）で値を切り替えられます

**例**

```cpp
HP_PARAM(int,    BEAM_WIDTH,  10,   1,    100)
HP_PARAM(double, TEMP_START,  2.0,  0.1,  10.0)
META_PARAM(int,  STRATEGY,    0)
```

## カテゴリ別パラメータのコード展開

`meta_params.json` の内容を、実行時に入力の特徴量からカテゴリを判定して
パラメータへ代入する C++ ヘッダ `lib/meta_params.hpp` に展開します。

```
$ uv run ahc-tester/gen_meta_params.py
```

**オプション**

| オプション | 説明 |
|------------|------|
| `--meta PATH` | meta_params.json のパスを指定（デフォルトはルートの `meta_params.json`） |
| `--cpp PATH` | パラメータ宣言を読むソースを指定（デフォルトは config の `cpp_file`） |
| `--out PATH` | 出力先（デフォルト `lib/meta_params.hpp`） |
| `--in DIR` | ビン再構築に使う入力ディレクトリ |

**main.cpp 側の使い方**

```cpp
#define USE_META_PARAMS               // hp_params.hpp の include より前に定義
#include "lib/hp_params.hpp"

HP_PARAM(double, TEMP_START, 2.0, 0.1, 10.0);
META_PARAM(int,  STRATEGY,   0);

#include "lib/meta_params.hpp"        // パラメータ宣言の後に include

int main() {
    // 入力を読んだ直後に呼ぶ（引数は AXES の順）
    meta::apply_params(N, W);
    ...
}
```

- ローカルビルドでは環境変数（Optuna が注入した値）がカテゴリ別の値より優先されるため、
  展開後も通常のチューニングがそのまま動きます
- `ONLINE_JUDGE` ビルドではカテゴリ別の値がコードに焼き込まれ、環境変数は読まれません
- 展開に使う軸は C++ 側でも実行時に計算できる特徴量に限ってください
