# Variant Go Platform

変則ルール対応の五目並べプラットフォームです。複数のAIアルゴリズムと対戦し、棋譜を記録して機械学習の訓練データとして活用できます。

## 機能一覧

- **複数のゲームルール**: 標準五目並べ、重力付き五目並べ（Connect Four風）
- **複数のAI**: Random、Minimax（Alpha-Beta）、MCTS、VCF/VCT
- **GUI**: Fletによるクロスプラットフォーム対応
- **CLI**: コマンドラインからAI対戦を実行
- **棋譜記録**: JSON/JSONL形式で対局を保存

---

## インストール

```bash
# 依存パッケージのインストール
pip install flet

# テスト実行（オプション）
pytest
```

---

## 使い方

### 1. GUI モード（設定画面から開始）

```bash
python main.py
```

設定画面でルール、プレイヤー（人間/AI）、AI設定を選択してゲームを開始します。

### 2. CLI モード（直接ゲーム開始）

```bash
# 人間 vs Minimax AI
python main.py -b human -w minimax

# Minimax vs MCTS
python main.py -b minimax -w mcts

# VCF AI同士（VCT探索有効）
python main.py -b vcf -w vcf --vct

# 重力ルールで対戦
python main.py -b human -w mcts --rule gravity

# 棋譜を記録しながら対戦
python main.py -b minimax -w mcts --record
```

### 3. AI対戦バッチスクリプト

複数のAI対戦を自動実行し、棋譜を記録します。

```bash
# デフォルト: Minimax vs MCTS を3回
python run_ai_battles.py

# VCF vs Minimax を10回
python run_ai_battles.py -b vcf -w minimax -n 10

# 詳細表示モード
python run_ai_battles.py --verbose

# JSONL形式で保存（大量データ向き）
python run_ai_battles.py -n 100 --format jsonl
```

---

## CLI オプション一覧

### main.py

| オプション | 短縮形 | 説明 | デフォルト |
|-----------|--------|------|-----------|
| `--black` | `-b` | 黒プレイヤー (human/random/minimax/mcts/vcf) | - |
| `--white` | `-w` | 白プレイヤー (human/random/minimax/mcts/vcf) | - |
| `--rule` | `-r` | ルール (standard/gravity) | standard |
| `--depth` | `-d` | Minimax探索深度 (1-5) | 3 |
| `--simulations` | `-s` | MCTSシミュレーション回数 | 500 |
| `--iterative` | `-i` | 反復深化を有効化 | False |
| `--vct` | - | VCT探索を有効化（VCF AIのみ） | False |
| `--no-thinking` | - | 思考パネルを非表示 | False |
| `--record` | - | 棋譜記録を有効化 | False |
| `--record-dir` | - | 棋譜保存先ディレクトリ | ./game_logs |
| `--record-format` | - | 保存形式 (json/jsonl/text/csv/seq) | json |

### run_ai_battles.py

| オプション | 短縮形 | 説明 | デフォルト |
|-----------|--------|------|-----------|
| `--black` | `-b` | 黒プレイヤーAI | minimax |
| `--white` | `-w` | 白プレイヤーAI | mcts |
| `--rule` | `-r` | ルール | standard |
| `--games` | `-n` | 対戦回数 | 3 |
| `--depth` | `-d` | Minimax探索深度 | 3 |
| `--simulations` | `-s` | MCTSシミュレーション回数 | 500 |
| `--vct` | - | VCT探索を有効化 | False |
| `--output-dir` | `-o` | 棋譜保存先 | ./game_logs |
| `--format` | `-f` | 保存形式 (json/jsonl/text/csv/seq) | json |
| `--verbose` | `-v` | 詳細表示 | False |

---

## AI アルゴリズム

| AI | 難易度 | 説明 |
|----|--------|------|
| Random | Easy | 合法手からランダム選択 |
| Minimax | Medium | Alpha-Beta探索（深度調整可能、反復深化対応） |
| MCTS | Hard | モンテカルロ木探索（UCB1） |
| VCF | Hard | 詰め探索（Victory by Continuous Fours） |

### Minimax オプション

- **depth**: 探索深度（1〜5）。大きいほど強いが遅い
- **iterative**: 反復深化。時間制限と併用で安定した手を返す

### MCTS オプション

- **simulations**: シミュレーション回数。多いほど強いが遅い

### VCF オプション

- **vct**: VCT（Victory by Continuous Threats）を有効化。より強力だが遅い

---

## ゲームルール

### StandardGomokuRule（標準五目並べ）

- 15×15の盤面
- 先に5つ連続で並べた方が勝ち
- 禁じ手なし

### GravityGomokuRule（重力付き五目並べ）

- 7×6の盤面（Connect Four風）
- 石は下に落ちる
- 4つ連続で勝利

---

## 棋譜形式

5種類のフォーマットに対応しています。

| フォーマット | 拡張子 | 用途 |
|-------------|--------|------|
| json | .json | 構造化データ、詳細分析 |
| jsonl | .jsonl | 大量データ、ストリーム処理 |
| text | .txt | 人間可読、レビュー用 |
| csv | .csv | データ分析、Excel等 |
| seq | .seq | Transformer訓練、LLM用 |

---

### text形式（人間可読）

```
# Variant Go: StandardGomokuRule 15x15
# Black: Minimax AI vs White: MCTS AI
# Result: BLACK_WIN (9 moves)

1. h8
2. c12
3. g7
4. i14
5. f6
...
```

### seq形式（Transformer訓練向け）

純粋なトークン列。1行1対局。先頭に結果トークン。

```
[BLACK_WIN] h8 c12 g7 i14 f6 h6 e5 k4 d4
[WHITE_WIN] h8 i9 g7 j10 f6 ...
```

### csv形式（データ分析向け）

```csv
move,x,y,col,row,stone
1,7,7,h,8,BLACK
2,2,11,c,12,WHITE
3,6,6,g,7,BLACK
```

### json形式（詳細データ）

```json
{
  "game_id": "20260129_120000_000000",
  "rule_id": "StandardGomokuRule",
  "result": "BLACK_WIN",
  "moves": [
    {"move_number": 1, "x": 7, "y": 7, "stone": "BLACK", "thinking_time": 0.3}
  ]
}
```

### jsonl形式（大量データ向け）

```
{"game_id":"20260129_120000","result":"BLACK_WIN","moves":[...]}
{"game_id":"20260129_120001","result":"WHITE_WIN","moves":[...]}
```

---

## ファイル構成

```
testpg/
├── main.py                # GUIエントリーポイント
├── run_ai_battles.py      # AI対戦バッチスクリプト
├── game_logs/             # 棋譜保存先
│   └── .gitkeep
├── ui/                    # UIコンポーネント
│   ├── board_component.py # 盤面表示
│   ├── settings_view.py   # 設定画面
│   ├── game_view.py       # ゲーム画面
│   └── thinking_panel.py  # AI思考可視化
├── game_core.py           # コアロジック（Board, GameEngine, Rules）
├── ai_strategies.py       # AI戦略（Random, Minimax, MCTS）
├── threat_search.py       # VCF/VCT探索
├── game_record.py         # 棋譜記録
├── players.py             # プレイヤー・セッション管理
├── test_*.py              # テストファイル（193件）
├── CLAUDE.md              # 開発者向けドキュメント
└── README.md              # このファイル
```

---

## 開発者向け情報

詳細な設計情報は [CLAUDE.md](./CLAUDE.md) を参照してください。

### テスト実行

```bash
# 全テスト実行
pytest

# 特定のテストファイル
pytest test_ai_strategies.py -v

# カバレッジ付き
pytest --cov=.
```

### 棋譜の読み込み

```python
from game_record import load_game_record, load_game_records_jsonl

# 単一ファイル
record = load_game_record("game_logs/game_20260129_120000.json")
print(f"Result: {record.result}, Moves: {record.total_moves}")

# JSONL形式（複数対局）
records = load_game_records_jsonl("game_logs/games_20260129.jsonl")
for r in records:
    print(f"{r.game_id}: {r.result}")
```

---

## ライセンス

MIT License
