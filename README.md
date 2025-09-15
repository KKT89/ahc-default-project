## ディレクトリ構成

```
AHCxxx/                  // ルートディレクトリ
├── ahc-tester/          
├── tools/               // 公式ローカルテストツール
├── task.html            // 問題文（ローカル用）
└── main.cpp             // 解答コード
```

## 使い方

ahc-tester の使い方は[こちら](https://github.com/KKT89/ahc-tester/blob/main/README.md)

### 事前準備
- `tools/`：公式ローカルテストツール一式をリポジトリ直下に配置してください。
- `task.html`：問題文のローカルコピーをリポジトリ直下に配置してください。
- 上記を配置したうえで、次の「初回セットアップ」を実行します。

### 初回セットアップ
以下のスクリプトで、仮想環境の作成・依存導入・ツールのセットアップ（プロジェクトルートに `config.toml` を作成）を一括実行します。

```
$ bash scripts/init.sh -o <max|min|maximize|minimize> -t <秒>
```

実行権限を付けて直接実行する場合は以下のとおり。

```
$ chmod +x scripts/init.sh && ./scripts/init.sh -o <max|min|maximize|minimize> -t <秒>
```

- 前提：`uv` がインストール済み
- 必須：
  - `-o, --objective`：`max|min|maximize|minimize`
  - `-t, --tl`：タイムリミット(秒)
- 任意：
  - `-i, --interactive`：インタラクティブ問題の場合に付与

例：

```
$ bash scripts/init.sh -o max -t 2 -i
```
