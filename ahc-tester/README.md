# ahc-tester
AtCoder Heuristic Contest (AHC) で使用しているツール群です。

## 使い方

### Python環境
ルートディレクトリで `uv sync` を実行すると、`pyproject.toml` / `uv.lock` に基づいて `.venv` が自動生成されます。

### セットアップ
プロジェクトルートに設定ファイル `config.toml` を作成し、公式ローカルテストツールのビルドを実行します。

`objective` の指定が必須です。インタラクティブ問題の場合のみ `-i` を指定します。

```
$ uv run ahc-tester/setup.py {max|min|maximize|minimize} [-i]
```

**主な引数**
- `objective`：最適化方向 `max|min|maximize|minimize` を受け付け、内部で `maximize|minimize` に正規化します。
- `-i, --interactive`：インタラクティブ問題のときに指定し、この時 `tester` を追加でビルドします。

**使用例**
- 非インタラクティブ・最大化
```
$ uv run ahc-tester/setup.py max
```
- インタラクティブ・最大化
```
$ uv run ahc-tester/setup.py max -i
```
- 非インタラクティブ・最小化
```
$ uv run ahc-tester/setup.py min
```

**ヘルプ表示**
```
$ uv run ahc-tester/setup.py --help
```

### テストケース作成
以下のコマンドで、L 以上 R 未満のシード値のテストケースを作成します。

```
$ uv run ahc-tester/make_test.py L R
```

### ビルド

```
$ uv run ahc-tester/build.py
```

### パラメータ（HP_PARAM）
`lib/hp_params.hpp` の `HP_PARAM(type, name, def, low, high)` でハイパーパラメータを宣言します。
Optuna 実行時は、`main.cpp` からこれらを自動抽出して study ディレクトリに `params.json` を生成します。

### テスト実行
以下のコマンドでテストを実行します。

```
$ uv run ahc-tester/run_test.py
$ uv run ahc-tester/run_test.py --cases 5
$ uv run ahc-tester/run_test.py --range 10 20
```

- `--cases` で実行件数を指定できます（省略時は `config.toml` の `pretest_count`）。
- `--range L R` で seed ID が `[L, R)` のケースだけ実行できます（`--cases` とは同時指定不可）。

### optuna

以下のコマンドで optuna を使ったパラメータ最適化を実行します。新規 study 作成時に `main.cpp` から `HP_PARAM` を抽出し、`params.json` を自動生成します。

```
$ uv run ahc-tester/optuna_manager.py
```

**オプション**
- `--dir <ディレクトリ>`
  - 指定したディレクトリから最適化を再開します。
- `--last`
  - `optuna_work` 配下で最も新しいサブディレクトリを自動的に選択します。(`--dir` より優先されます)
- `--zero`
  - `n_trials = 0` で実行します。パラメータを即時更新したい時に使います。
