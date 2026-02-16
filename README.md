## ディレクトリ構成

```
AHCxxx/                  // ルートディレクトリ
├── ahc-tester/          
├── tools/               // 公式ローカルテストツール
├── task.html            // 問題文（ローカル用）
└── main.cpp             // 解答コード
```

## 使い方

### 事前準備

以下をリポジトリ直下に配置した上で、次の初回セットアップを実行してください。
- `tools/`：公式ローカルテストツール一式
- `task.html`：問題文のローカルコピー

### 初回セットアップ
以下のスクリプトで、`uv sync` による依存解決とツールのセットアップ（プロジェクトルートに `config.toml` を作成）を一括実行します。

```
$ chmod +x scripts/init.sh
$ ./scripts/init.sh -o <max|min|maximize|minimize>
```

- 前提：`uv` がインストール済み
- 必須：
  - `-o, --objective`：`max|min|maximize|minimize`
- 任意：
  - `-i, --interactive`：インタラクティブ問題の場合に付与

**例**

```
$ bash scripts/init.sh -o max -i
```

### ローカルテストの実行

```
$ uv run ahc-tester/run_test.py            # config 上の全ケース（既定 150 件）を実行
$ uv run ahc-tester/run_test.py --cases 5  # 冒頭 5 ケースだけ検証
$ uv run ahc-tester/run_test.py --range 10 20  # seed が [10,20) のケースだけ実行
$ uv run ahc-tester/run_test.py --try 5    # 各ケースを5回実行し、目的関数に対して悪い側のスコアを採用
$ uv run ahc-tester/run_test.py --debug    # ソルバを -DDEBUG 付きでビルド（prevスコア保存はスキップ）
```

- `--cases` を省略すると `config.toml` に設定された件数（デフォルト 150）を実行します。
- 指定数が手元の入力ファイル数を超える場合は、利用可能な件数までに自動調整されます。
- `--range L R` で seed ID が `[L, R)` にあるケースだけを実行できます（`--cases` と同時指定不可）。
- `--try N`（`N >= 1`）を指定すると、各ケースを複数回実行して乱数ぶれを評価します。
- `--debug` を指定すると、`main.cpp` は `-DDEBUG` 付きでコンパイルされます。

### テストケースの生成

```
$ uv run ahc-tester/make_test.py L R
$ uv run ahc-tester/make_test.py 0 10   # 例: seed 0〜9 のケースを生成
```

- seed を `L` 以上 `R` 未満（整数）の範囲でテストケースをまとめて生成します。
- 生成されたファイルは `in/`（入力）と `out/`（出力）ディレクトリに配置されます。
- すべてリポジトリ直下（プロジェクトルート）で実行してください。
