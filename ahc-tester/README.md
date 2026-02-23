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
```

**例**

```
$ uv run ahc-tester/make_test.py 0 150   # seed 0〜149 を生成
```

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
$ uv run ahc-tester/run_test.py --debug
```

**オプション**

| オプション | 説明 |
|------------|------|
| `--cases N` | 実行件数を指定。省略時は `config.toml` の `pretest_count`（デフォルト 150） |
| `--range L R` | seed が `[L, R)` のケースのみ実行。`--cases` と同時指定不可 |
| `--jobs N` | 並列スレッド数（デフォルト 8） |
| `--debug` | `-DDEBUG` 付きでビルド。prev スコアの保存プロンプトをスキップ |

**出力列**

```
Case  Score      Time   Stat  | vsPrev   Sum      | vsBest   Sum      | vsPrevN  Sum
```

| 列 | 説明 |
|----|------|
| `vsPrev` | 前回提出スコアとの差分 |
| `vsBest` | 全期間ベストスコアとの差分 |
| `vsPrevN` | 前回提出スコアとの差分（0〜1 に正規化） |
| `Sum` | 選択ケース全体の累積差分 |

改善は緑、悪化は赤、ベスト更新は金色で表示します。

**スコアの保存**

- `score/best_scores.json`：全期間ベストスコアを自動更新
- `score/prev_scores.json`：`--range` / `--cases` を指定しない通常実行の終了時に、保存するか確認します（`--debug` 時はスキップ）

## optuna

`HP_PARAM` マクロで宣言されたハイパーパラメータを Optuna で最適化します。

```
$ uv run ahc-tester/optuna_manager.py            # 新規 study を作成して最適化
$ uv run ahc-tester/optuna_manager.py --last     # 最新の study を再開
$ uv run ahc-tester/optuna_manager.py --dir <ディレクトリ>  # 指定 study を再開
$ uv run ahc-tester/optuna_manager.py --zero     # 試行を実行せずパラメータだけ更新
```

**オプション**

| オプション | 説明 |
|------------|------|
| `--dir <dir>` | 再開する study ディレクトリを指定 |
| `--last` | `optuna_work/` 配下で最新のサブディレクトリを自動選択（`--dir` より優先） |
| `--zero` | `n_trials=0` で実行。パラメータを即時更新したいときに使用 |

新規 study 作成時は `main.cpp` から `HP_PARAM` マクロを自動抽出して `params.json` を生成します。
最適化終了後はベストパラメータを `params.json` の `value` フィールドに書き戻します。

**環境変数**

| 変数 | 説明 | デフォルト |
|------|------|-----------|
| `OPTUNA_N_JOBS` | 並列試行数（`-1` で最大） | `-1` |
| `OPTUNA_PARAM_ENV_PREFIX` | パラメータ注入時の環境変数プレフィックス | `HP_` |
| `OPTUNA_OBJECTIVE_SEED` | テストケースのシャッフル seed | ランダム |

### HP_PARAM マクロ

`lib/hp_params.hpp` でハイパーパラメータを宣言します。

```cpp
HP_PARAM(type, name, default, low, high)
```

- 実行時は環境変数 `HP_{name}` から値を読み込みます
- Optuna 実行時はこれらが自動抽出され、`params.json` に記録されます

**例**

```cpp
HP_PARAM(int,    BEAM_WIDTH,  10,   1,    100)
HP_PARAM(double, TEMP_START,  2.0,  0.1,  10.0)
```
