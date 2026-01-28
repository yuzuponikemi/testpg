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
├── ui/                    # UIコンポーネント（Milestone 4）
│   ├── __init__.py
│   ├── board_component.py # 盤面表示
│   ├── settings_view.py   # 設定画面
│   └── game_view.py       # ゲーム画面
├── game_core.py           # コアロジック（Milestone 1 + 3）
│                          #   - Board, GameRule, StandardGomokuRule
│                          #   - GravityGomokuRule, RuleRegistry
│                          #   - GameEngine
├── ai_strategies.py       # AI戦略（Milestone 2）
├── players.py             # プレイヤー・セッション管理（Milestone 2）
├── test_game_core.py      # コアロジックテスト（82件）
├── test_ai_strategies.py  # AI戦略テスト（27件）
├── test_players.py        # プレイヤーテスト（28件）
├── test_integration.py    # 統合テスト（12件）
├── conftest.py            # pytest設定
├── pytest.ini             # pytest設定
├── ROADMAP.md             # 実装計画
└── CLAUDE.md              # このファイル
```

**テスト総数:** 149件

---

## 用語集

| 用語 | 説明 |
|------|------|
| 禁じ手 | 三三、四四、長連など特定の形を禁止するルール |
| 両開き | 連続石の両端が空いている状態（防御困難） |
| プレイアウト | MCTSでの終局までのランダム対局 |
| UCB1 | MCTSのノード選択に使うアルゴリズム |

---

## 変更履歴

| 日付 | 内容 |
|------|------|
| 2026-01-28 | Milestone 1完了、初版作成 |
| 2026-01-28 | Milestone 2完了（AI基盤: RandomAI, MinimaxAI, MCTSAI, GameSession） |
| 2026-01-28 | Milestone 3完了（ルールエンジン: RuleRegistry, GravityGomokuRule, STONE_MOVEDイベント） |
| 2026-01-28 | Milestone 4完了（Flet GUI: BoardComponent, SettingsView, GameView） |
