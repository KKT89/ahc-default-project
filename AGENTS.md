# このリポジトリについて

## ハイパーパラメータについて

このテンプレートでは、`lib/hp_params.hpp` のマクロでハイパーパラメータを宣言し、ローカル実行時のみ環境変数で上書きできます。提出時（`ONLINE_JUDGE` 定義時）はコンパイル時定数として固定され、安全に最適化されます。

### 宣言方法（HP_PARAM）
- 形式: `HP_PARAM(type, name, def, low, high)`
- 例: `HP_PARAM(int, BEAM_WIDTH, 100, 1, 10000);`

各引数の意味（def / low / high）
- def: 既定値（提出時はこの値に固定）。ローカルで環境変数が無ければこの値を使用。
- low: 取りうる最小値（閉区間）。ローカル上書き時に下回ると `low` にクランプされます。
- high: 取りうる最大値（閉区間）。ローカル上書き時に上回ると `high` にクランプされます。
  - 範囲外が指定された場合は stderr にクランプ情報を出力します。
  - `def` が `[low, high]` を外れるとコンパイル時に `static_assert` でエラーになります。

### 参照方法
- 宣言直後から通常の変数と同じように参照できます。
- 例:
  - `HP_PARAM(double, TEMP, 1.0, 0.0, 10.0);`
  - `for (int i = 0; i < (int)TEMP; ++i) { /* ... */ }`

### ローカル実行での上書き（環境変数）
- 既定の環境変数プレフィックスは `HP_` です。
- 変数名は `HP_` + 宣言名。
  - 例: `HP_PARAM(int, BEAM_WIDTH, 100, 1, 10000)` → `HP_BEAM_WIDTH=200`

実行例（デフォルトのプレフィックス HP_ を使用）
- `HP_BEAM_WIDTH=200 uv run ahc-tester/run_test.py`
  - `run_test.py` 経由で `solution` を起動していれば、その環境が子プロセスに引き継がれます。
  - 直接 `./solution` を実行する場合も同様に環境変数を付与してください。

### プレフィックスの変更（任意）
- 既定の接頭辞は `HP_`（`lib/hp_params.hpp` 内の `#define HP_ENV_PREFIX "HP_"`）。
- 特別な事情がある場合のみ、コンパイル時マクロ `-DHP_ENV_PREFIX=\"EXP_\"` を付けてビルドすれば変更できます（デフォルトでは未使用）。
- 通常運用では変更不要です。

補足
- 提出環境（`ONLINE_JUDGE` 定義時）は、環境変数は無視され、`def` の定数が使用されます。
- 文字列からのパースは `std::stringstream` で行います。
  - 整数型: `42` のような整数リテラル。
  - 浮動小数: `0.5` などの小数も可。

## 運用メモ（最適化サイクル）

### 背景
- Optuna のハイパーパラメータを環境変数で注入し、並列最適化でも干渉しないようにする。
- 提出時は `ONLINE_JUDGE` で定数埋め込みにして安全に最適化。

### 構成概要
- 追加: `lib/hp_params.hpp`
  - `HP_PARAM(type, name, def, low, high)` で宣言。
  - ローカル: 環境変数で上書き（範囲チェック＋クランプ）。
  - 本番: `-DONLINE_JUDGE` で `constexpr` 固定。
  - 既定プレフィックス: `HP_`（`-DHP_ENV_PREFIX="..."` で変更可）。
- 更新: `ahc-tester/run_test.py`
  - 逐次で `solution` を実行し `vis` で採点するシンプル構成。
- 更新: `ahc-tester/optuna_manager.py`
  - 新規 study 作成時に `main.cpp` の `HP_PARAM(...)` を抽出して `params.json` を自動生成。
  - 各試行の提案値を `HP_...` 環境変数として `solution` に注入し、`vis` でスコア算出。
- 更新: `ahc-tester/build.py`
  - `HP_ENV_PREFIX` をシェルから受け取り `-DHP_ENV_PREFIX` を付与。
  - 旧 `params.cpp` 自動生成は廃止。

### C++ 側の例
```cpp
#include "lib/hp_params.hpp"
HP_PARAM(int,    ITER,  500,  1,   5000);
HP_PARAM(double, ALPHA, 0.30, 0.0, 1.0);
```

### 手動実行例
- ビルド: `uv run ahc-tester/build.py`
- 実行: `HP_ITER=1200 HP_ALPHA=0.55 ./solution < in/0000.txt > out.txt`

### プレフィックス変更
- ビルド側: `HP_ENV_PREFIX=MY_ uv run ahc-tester/build.py`
- Optuna側: `OPTUNA_PARAM_ENV_PREFIX=MY_ uv run ahc-tester/optuna_manager.py`
- 両者を揃える。

### オンラインジャッジ（定数埋め込み）
- `config.toml` の `build.compile_options` に `-DONLINE_JUDGE` を付与（例: `-O2 -DONLINE_JUDGE`）。

### 既存 `params.cpp` からの移行
- 本テンプレートは `params.cpp` を使用しません（`hp_params.hpp` に統一）。
- 過去資産がある場合は `HP_PARAM` へ移植してください。
