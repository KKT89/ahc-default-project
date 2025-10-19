## ディレクトリ構成

```
AHCxxx/                  // ルートディレクトリ
├── ahc-tester/          
├── tools/               // 公式ローカルテストツール
├── docs/task.html       // 問題文（ローカル用）
└── main.cpp             // 解答コード
```

## 使い方

ahc-tester の使い方は[こちら](https://github.com/KKT89/ahc-tester/blob/main/README.md)

### 事前準備

以下をリポジトリ直下に配置した上で、次の初回セットアップを実行してください。
- `tools/`：公式ローカルテストツール一式
- `docs/task.html`：問題文のローカルコピー

### 初回セットアップ
以下のスクリプトで、仮想環境の作成・依存導入・ツールのセットアップ（プロジェクトルートに `config.toml` を作成）を一括実行します。

```
$ chmod +x scripts/init.sh
$ ./scripts/init.sh -o <max|min|maximize|minimize> -t <秒>
```

- 前提：`uv` がインストール済み
- 必須：
  - `-o, --objective`：`max|min|maximize|minimize`
  - `-t, --tl`：タイムリミット(秒)
- 任意：
  - `-i, --interactive`：インタラクティブ問題の場合に付与

**例**

```
$ bash scripts/init.sh -o max -t 2 -i
```

### ローカルテストの実行

```
$ uv run ahc-tester/run_test.py            # config 上の全ケース（既定 150 件）を実行
$ uv run ahc-tester/run_test.py --cases 5  # 冒頭 5 ケースだけ検証
```

- `--cases` を省略すると `config.toml` に設定された件数（デフォルト 150）を実行します。
- 指定数が手元の入力ファイル数を超える場合は、利用可能な件数までに自動調整されます。
