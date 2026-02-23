## ディレクトリ構成

```
AHCxxx/                  // ルートディレクトリ
├── ahc-tester/
├── tools/               // 公式ローカルテストツール
├── task.html            // 問題文（ローカル用）
└── main.cpp             // 解答コード
```

## クイックスタート

### 事前準備

以下をリポジトリ直下に配置した上で、次のセットアップを実行してください。
- `tools/`：公式ローカルテストツール一式
- `task.html`：問題文のローカルコピー

### セットアップ
以下のスクリプトで、依存解決・ツールのビルド・テストケース生成（seed 0〜149）を一括実行します。

```
$ ./scripts/init.sh -o <max|min>
$ ./scripts/init.sh -o <max|min> -i  # インタラクティブ問題の場合
```

### テスターツール

詳細は [ahc-tester/README.md](ahc-tester/README.md) を参照してください。

#### [テストケース作成](ahc-tester/README.md#テストケース作成)

```
$ uv run ahc-tester/make_test.py L R
```

#### [ビルド](ahc-tester/README.md#ビルド)

```
$ uv run ahc-tester/build.py
```

#### [テスト実行](ahc-tester/README.md#テスト実行)

```
$ uv run ahc-tester/run_test.py
```

#### [optuna](ahc-tester/README.md#optuna)

```
$ uv run ahc-tester/optuna_manager.py
```
