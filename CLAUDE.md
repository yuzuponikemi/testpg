# Variant Go Platform - 開発メモ

このファイルは、今後の実装で重要になる設計決定・注意点・コンテキストをまとめたものです。

---

## プロジェクト概要

**目的:** 変則ルール対応のボードゲームプラットフォーム（最終的にAndroid対応）

**現在の状態:** Milestone 4 完了（Flet GUI実装済み）

---

## アーキテクチャ原則

### 責務分離（厳守）

```
Board       → 盤面状態のみ（ロジックなし）
GameRule    → ルール判定のみ（状態を持たない）
GameEngine  → ゲーム進行管理 + Observer通知
Player      → 手の決定（Human/AI）
UI          → 表示のみ（ロジック呼び出しはEngineのみ経由）
```

**重要:** UIからBoardを直接操作しない。必ずGameEngine経由。

### Observerパターン（重要）

```python
# 正しい使い方
engine.add_listener(callback)  # UIがリスナー登録
engine.play_move(x, y)         # 内部でcallbackが呼ばれる

# やってはいけない
board.set_stone(x, y, stone)   # UIから直接呼ぶとObserverが動かない
```

### 非同期設計（Milestone 2以降）

- AI思考中にUIがフリーズしないよう、`async/await`を使用
- Fletは`page.run_task()`で非同期処理を実行
- AI計算が重い場合は`asyncio.to_thread()`でスレッド分離

---

## 既存コードの重要ポイント

### game_core.py

| クラス | 注意点 |
|--------|--------|
| `Board.copy()` | 外部に渡すときは必ずコピーを返す（直接変更防止） |
| `GameEngine.board` | プロパティがコピーを返す設計（意図的） |
| `GameRule.get_valid_moves()` | AI用。全座標をループするので大きい盤面では遅い可能性 |
| `_notify_listeners()` | 例外をキャッチしてゲーム継続（リスナーの不具合でゲームが壊れない） |

### 拡張予定のフック（Docstring参照）

- `GameRule.is_valid_move()` → 禁じ手ルールで拡張
- `GameRule.check_winner()` → オセロ風など特殊勝利条件で拡張
- `Board` → 異なる盤面形状（トーラス、六角形）で継承

---

## Milestone 2（AI基盤）実装済み

### players.py

| クラス | 説明 |
|--------|------|
| `Player` (ABC) | プレイヤー抽象基底クラス |
| `HumanPlayer` | 人間プレイヤー（asyncio.Eventで入力待ち） |
| `AIPlayer` | AIプレイヤー（asyncio.to_threadで非同期実行） |
| `GameSession` | ゲームセッション管理（非同期ゲームループ） |
| `GameSessionConfig` | セッション設定（タイムアウト、遅延） |
| `SessionEvent` | セッションイベント（GAME_START, TURN_START, GAME_END等） |

### ai_strategies.py

| クラス | 難易度 | 説明 |
|--------|--------|------|
| `AIStrategy` (ABC) | - | AI戦略抽象基底クラス |
| `RandomAI` | Easy | 合法手からランダム選択 |
| `MinimaxAI` | Medium | Alpha-Beta探索（深度調整可能） |
| `MCTSAI` | Hard | モンテカルロ木探索 |
| `AIStrategyFactory` | - | 名前からAIを生成するファクトリ |

### 使用例

```python
import asyncio
from game_core import StandardGomokuRule
from players import AIPlayer, GameSession
from ai_strategies import MinimaxAI, RandomAI

async def main():
    rule = StandardGomokuRule()
    black = AIPlayer(MinimaxAI(depth=3))
    white = AIPlayer(RandomAI())

    session = GameSession(rule, black, white)
    result = await session.run_game()
    print(f"Result: {result}")

asyncio.run(main())
```

### 評価関数（MinimaxAI）

五目並べの評価要素:
1. 連続石数（4連 >> 3連 >> 2連）
2. 両端が開いているか（両開き > 片開き > 閉じ）
3. 候補手絞り込み（既存石の周囲2マスのみ探索）

```python
# EvaluationWeights デフォルト値
SCORES = {
    "five": 100000,      # 勝ち
    "open_four": 10000,  # 止められない
    "four": 1000,        # 要防御
    "open_three": 500,
    "three": 100,
    "open_two": 50,
    "two": 10,
}
```

---

## Milestone 3（ルールエンジン）実装済み

### GameRule 拡張メソッド

```python
class GameRule(ABC):
    @property
    def rule_id(self) -> str:
        """ルール識別子（デフォルト: クラス名）"""

    def apply_move_effects(self, board, x, y, stone) -> list[Position]:
        """着手の副作用（重力、捕獲など）"""
        # デフォルト: 空リスト

    def get_rule_config(self) -> dict:
        """設定のシリアライズ"""

    @classmethod
    def from_config(cls, config) -> GameRule:
        """設定からのデシリアライズ"""
```

### RuleRegistry（ルール管理）

```python
from game_core import RuleRegistry, GravityGomokuRule

# ルール一覧
available = RuleRegistry.list_available()  # ["GravityGomokuRule", "StandardGomokuRule"]

# ルール生成
rule = RuleRegistry.create("GravityGomokuRule", width=10, height=8)

# 設定から復元
config = {"rule_id": "GravityGomokuRule", "width": 10, "height": 8}
rule = RuleRegistry.create_from_config(config)
```

### GravityGomokuRule（重力付き五目並べ）

```python
from game_core import GravityGomokuRule, GameEngine

rule = GravityGomokuRule(width=7, height=6, win_condition=4)
engine = GameEngine(rule)

# 石は自動的に下に落ちる
engine.play_move(3, 0)  # 列3に置く → (3, 5)に落下
```

| プロパティ | デフォルト値 | 説明 |
|-----------|-------------|------|
| width | 7 | 盤面の横幅（列数） |
| height | 6 | 盤面の縦幅（行数） |
| win_condition | 4 | 勝利に必要な連続数 |

### Observerイベント拡張

```python
# 新規追加されたイベントタイプ
"STONE_MOVED"     # 石が移動した（重力落下時）

# 将来追加予定
"STONE_CAPTURED"  # 石が取られた
"TURN_SKIPPED"    # パス
```

### 使用例（AI対戦）

```python
import asyncio
from game_core import GravityGomokuRule
from players import AIPlayer, GameSession
from ai_strategies import RandomAI

async def main():
    rule = GravityGomokuRule()  # 7x6, 4連勝利
    black = AIPlayer(RandomAI())
    white = AIPlayer(RandomAI())

    session = GameSession(rule, black, white)
    result = await session.run_game()
    print(f"Result: {result}")

asyncio.run(main())
```

---

## Milestone 4（GUI）実装済み

### 実行方法

```bash
# Fletアプリとして実行
flet run main.py

# または直接実行
python main.py
```

### ファイル構成

| ファイル | 説明 |
|----------|------|
| `main.py` | エントリーポイント |
| `ui/__init__.py` | UIモジュール初期化 |
| `ui/board_component.py` | 盤面表示コンポーネント |
| `ui/settings_view.py` | 設定画面（ルール・プレイヤー選択） |
| `ui/game_view.py` | ゲーム画面 |

### UIコンポーネント

#### BoardComponent

```python
from ui.board_component import BoardComponent

board = BoardComponent(
    engine=engine,
    on_cell_click=lambda x, y: handle_click(x, y),
    cell_size=40,
)
```

- GameEngineのObserverパターンに自動登録
- MOVE_PLAYED, STONE_MOVED, GAME_RESETイベントで自動更新
- 最後の手をハイライト表示

#### SettingsView

```python
from ui.settings_view import SettingsView, GameConfig

settings = SettingsView(page, on_start_game=lambda config: start_game(config))
```

**GameConfig:**
- `rule_id`: "StandardGomokuRule" or "GravityGomokuRule"
- `black_player_type`: "human", "random", "minimax", "mcts"
- `white_player_type`: 同上
- `ai_depth`: Minimax探索深度（1-5）
- `ai_simulations`: MCTS シミュレーション回数

#### GameView

```python
from ui.game_view import GameView

game = GameView(page, config, on_back=go_to_settings)
```

- 非同期ゲームループを`page.run_task()`で実行
- HumanPlayerの入力待ち中もUIはレスポンシブ
- 勝敗判定後に結果を表示

### 非同期設計

```python
# GameViewでの非同期ゲーム実行
async def _run_game_loop(self):
    result = await self._session.run_game()
    self._show_result(result)

# Fletでの起動
self._game_task = self._page.run_task(self._run_game_loop)
```

### タッチ操作

- セルクリックで石を配置（人間プレイヤーの番のみ）
- 合法手チェック済みの場合のみHumanPlayer.set_move()を呼び出し
- 最後の手を赤枠でハイライト

---

## AI改良（Milestone 5）実装済み

### 新機能一覧

| 機能 | 説明 |
|------|------|
| 反復深化 | MinimaxAIで深さ1から順に探索、時間制限対応 |
| VCF探索 | 四の連続で詰みを探索 |
| VCT探索 | 三も含めた詰み探索（VCFより強力） |
| 思考進捗通知 | AI思考過程をコールバックで通知 |
| ThinkingPanel | 思考過程のUI可視化 |

### 思考進捗通知

```python
from ai_strategies import ThinkingProgress, MinimaxAI

def on_progress(progress: ThinkingProgress):
    print(f"Depth: {progress.current_depth}, Nodes: {progress.nodes_visited}")

ai = MinimaxAI(depth=4, use_iterative_deepening=True)
ai.set_progress_callback(on_progress)
move = ai.select_move(board, rule, stone)
```

**ThinkingProgress フィールド:**
- `ai_type`: "minimax", "mcts", "vcf"
- `elapsed_time`: 経過時間（秒）
- Minimax用: `current_depth`, `max_depth`, `nodes_visited`, `current_best_move`, `current_best_score`
- MCTS用: `simulations_completed`, `total_simulations`, `top_moves`
- VCF用: `vcf_depth`, `is_forced_win`, `win_sequence`

### 反復深化（Iterative Deepening）

```python
# 反復深化有効
ai = MinimaxAI(depth=5, use_iterative_deepening=True, time_limit=3.0)
```

- 深さ1から順に探索
- 各深度完了時に最善手を更新
- 時間切れでも最後に完了した深度の結果を使用

### VCF/VCT探索

```python
from threat_search import VCFSearch, VCTSearch, VCFBasedAI

# VCF探索（四の連続）
vcf = VCFSearch(max_depth=20, time_limit=1.0)
result = vcf.search(board, rule, stone)
if result.is_winning:
    print(f"Win sequence: {result.win_sequence}")

# VCF AI（フォールバック付き）
ai = VCFBasedAI(use_vct=True, fallback=RandomAI())
move = ai.select_move(board, rule, stone)
```

**VCFResult:**
- `is_winning`: 詰みが見つかったか
- `win_sequence`: 詰み手順
- `depth_searched`: 探索深さ
- `nodes_visited`: 訪問ノード数

### ThinkingPanel（UI）

```python
from ui.thinking_panel import ThinkingPanel

panel = ThinkingPanel(show_detailed_log=True)
panel.set_page(page)
panel.update_progress(progress)  # スレッドセーフ
```

- 進捗バー（深度/シミュレーション）
- 統計情報（ノード数、時間、評価値）
- 最善手表示
- 詳細ログ（オプション）

### GameConfig拡張

```python
config = GameConfig(
    rule_id="StandardGomokuRule",
    black_player_type="minimax",
    white_player_type="vcf",
    ai_depth=4,
    ai_simulations=1000,
    use_iterative_deepening=True,  # 新規
    use_vct=False,                  # 新規
    show_thinking=True,             # 新規
    record_games=True,              # 棋譜記録
    record_dir="./game_logs",       # 保存先
    record_format="json",           # "json" or "jsonl"
)
```

---

## 棋譜記録（Game Recording）

### 概要

対局記録をJSON/JSONL形式で保存し、機械学習やGPTの訓練データとして活用できます。

### 使用方法

```python
from game_record import GameRecorder, load_game_record, load_game_records_jsonl

# 記録開始
recorder = GameRecorder(output_dir="./game_logs", format="json", enabled=True)
recorder.start_game(rule, black_player, white_player, config)

# 各手の記録
recorder.record_move(position, stone, thinking_time=1.2, eval_score=0.5)

# 対局終了
recorder.end_game(result)
filepath = recorder.save()

# 棋譜読み込み
record = load_game_record("game_20260129_120000.json")
records = load_game_records_jsonl("games_20260129.jsonl")  # JSONL形式
```

### CLI オプション

```bash
# 棋譜記録を有効にしてAI対戦
python main.py -b minimax -w mcts --record

# 保存先とフォーマットを指定
python main.py -b vcf -w vcf --record --record-dir ./training_data --record-format jsonl
```

### データ構造

**GameRecord:**
```json
{
  "game_id": "20260129_120000_000000",
  "timestamp": "2026-01-29T12:00:00",
  "rule_id": "StandardGomokuRule",
  "board_width": 15,
  "board_height": 15,
  "win_condition": 5,
  "black_player": {"stone": "BLACK", "player_type": "minimax", "name": "Minimax AI", "ai_depth": 3},
  "white_player": {"stone": "WHITE", "player_type": "mcts", "name": "MCTS AI", "ai_simulations": 1000},
  "result": "BLACK_WIN",
  "total_moves": 45,
  "duration": 120.5,
  "moves": [
    {"move_number": 1, "x": 7, "y": 7, "stone": "BLACK", "timestamp": 0.5, "thinking_time": 0.3},
    {"move_number": 2, "x": 8, "y": 8, "stone": "WHITE", "timestamp": 1.2, "eval_score": 0.1}
  ],
  "version": "1.0",
  "platform": "Variant Go Platform"
}
```

### フォーマット比較

| フォーマット | 用途 | 特徴 |
|-------------|------|------|
| JSON | 個別分析、デバッグ | 1ファイル1対局、読みやすい |
| JSONL | 大量データ、訓練 | 1行1対局、追記可能、ストリーム処理向き |

---

## テスト方針

### 必須テストケース

| 領域 | テスト内容 |
|------|-----------|
| ルール | 境界値（盤面端での勝利判定） |
| AI | 1手詰め・2手詰めを逃さない |
| Observer | イベント順序が正しい |
| 非同期 | タイムアウト処理 |

### モック活用

```python
# AIテストでのモック例
mock_strategy = Mock(spec=AIStrategy)
mock_strategy.select_move.return_value = Position(7, 7)
```

---

## パフォーマンス考慮

### 潜在的ボトルネック

1. `get_valid_moves()` - 15x15で225回ループ
2. Minimax探索 - 深さ5で指数的に増加
3. MCTS - シミュレーション回数依存

### 最適化オプション

- Zobrist Hashing（局面のハッシュ化）
- Transposition Table（探索済み局面の再利用）
- 着手候補の絞り込み（既存石の周囲のみ探索）

---

## ファイル構成（現在）

```
testpg/
├── main.py                # GUIエントリーポイント（Milestone 4）
├── ui/                    # UIコンポーネント（Milestone 4-5）
│   ├── __init__.py
│   ├── board_component.py # 盤面表示
│   ├── settings_view.py   # 設定画面
│   ├── game_view.py       # ゲーム画面
│   └── thinking_panel.py  # AI思考可視化パネル（Milestone 5）
├── game_core.py           # コアロジック（Milestone 1 + 3）
│                          #   - Board, GameRule, StandardGomokuRule
│                          #   - GravityGomokuRule, RuleRegistry
│                          #   - GameEngine
├── ai_strategies.py       # AI戦略（Milestone 2 + 5）
│                          #   - ThinkingProgress, 反復深化
├── threat_search.py       # VCF/VCT探索（Milestone 5）
│                          #   - ThreatDetector, VCFSearch, VCTSearch
│                          #   - VCFBasedAI
├── game_record.py         # 棋譜記録
│                          #   - GameRecord, MoveRecord, PlayerRecord
│                          #   - GameRecorder
├── players.py             # プレイヤー・セッション管理（Milestone 2）
├── test_game_core.py      # コアロジックテスト（82件）
├── test_ai_strategies.py  # AI戦略テスト（34件）
├── test_threat_search.py  # VCF/VCTテスト（21件）
├── test_game_record.py    # 棋譜記録テスト（16件）
├── test_players.py        # プレイヤーテスト（28件）
├── test_integration.py    # 統合テスト（12件）
├── conftest.py            # pytest設定
├── pytest.ini             # pytest設定
├── ROADMAP.md             # 実装計画
└── CLAUDE.md              # このファイル
```

**テスト総数:** 193件

---

## 用語集

| 用語 | 説明 |
|------|------|
| 禁じ手 | 三三、四四、長連など特定の形を禁止するルール |
| 両開き | 連続石の両端が空いている状態（防御困難） |
| プレイアウト | MCTSでの終局までのランダム対局 |
| UCB1 | MCTSのノード選択に使うアルゴリズム |
| VCF | Victory by Continuous Fours - 四の連続による詰み |
| VCT | Victory by Continuous Threats - 脅威の連続による詰み |
| 反復深化 | 深さ1から順に探索する手法、時間管理に有効 |
| 四 | あと1手で5連になる形 |
| 活三 | 両端が空いた3連（次に四を2箇所作れる） |
| 棋譜 | 対局の手順記録、ML/GPT訓練データに利用可能 |
| JSONL | JSON Lines形式、1行1レコードで大量データ向き |

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-01-28 | Milestone 1完了、初版作成 |
| 2026-01-28 | Milestone 2完了（AI基盤: RandomAI, MinimaxAI, MCTSAI, GameSession） |
| 2026-01-28 | Milestone 3完了（ルールエンジン: RuleRegistry, GravityGomokuRule, STONE_MOVEDイベント） |
| 2026-01-28 | Milestone 4完了（Flet GUI: BoardComponent, SettingsView, GameView） |
| 2026-01-29 | Milestone 5完了（AI改良: 反復深化, VCF/VCT, ThinkingProgress, ThinkingPanel） |
| 2026-01-29 | 棋譜記録機能追加（GameRecord, GameRecorder, JSON/JSONL形式） |
