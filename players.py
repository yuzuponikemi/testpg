"""
Variant Go Platform - Player Module (Milestone 2)

プレイヤーの抽象化とゲームセッション管理を提供します。

設計原則:
- Player.get_move() は非同期（async/await）
- AIStrategy は同期メソッド（計算のみ）
- AIPlayer 内で asyncio.to_thread() を使用してスレッド分離

クラス構成:
- Player (ABC): プレイヤーの抽象基底クラス
- HumanPlayer: 人間プレイヤー（GUI入力待ち）
- AIPlayer: AIプレイヤー（AIStrategyを使用）
- GameSession: ゲームセッション管理
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Callable
import asyncio

from game_core import (
    GameEngine, GameRule, Position, Stone,
    GameStatus, GameEvent
)
from ai_strategies import AIStrategy


class Player(ABC):
    """
    プレイヤーの抽象基底クラス

    全てのプレイヤー（Human/AI）はこのインターフェースを実装します。
    get_move() は非同期であり、HumanPlayer は入力待ち、
    AIPlayer は計算処理を別スレッドで実行します。

    将来の拡張:
    - NetworkPlayer: ネットワーク対戦用
    - ReplayPlayer: リプレイ再生用
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """プレイヤー名（表示用）"""
        pass

    @property
    @abstractmethod
    def is_human(self) -> bool:
        """人間プレイヤーかどうか"""
        pass

    @abstractmethod
    async def get_move(self, engine: GameEngine) -> Position:
        """
        次の手を決定する（非同期）

        Args:
            engine: 現在のゲームエンジン（boardプロパティはコピーを返す）

        Returns:
            選択した座標

        Raises:
            asyncio.CancelledError: キャンセルされた場合
            asyncio.TimeoutError: タイムアウトした場合
        """
        pass

    def on_game_start(self, engine: GameEngine, stone: Stone) -> None:
        """
        ゲーム開始時に呼ばれるフック（オプション）

        Args:
            engine: ゲームエンジン
            stone: このプレイヤーに割り当てられた石の色
        """
        pass

    def on_game_end(self, engine: GameEngine, result: GameStatus) -> None:
        """
        ゲーム終了時に呼ばれるフック（オプション）

        Args:
            engine: ゲームエンジン
            result: ゲーム結果
        """
        pass


class HumanPlayer(Player):
    """
    人間プレイヤー

    GUIからの入力をポーリングで待機します。
    set_move() が呼ばれるまで get_move() はブロックします。

    使用例:
        player = HumanPlayer("Player 1")

        # 別タスクでゲームループが動いている場合
        async def game_loop():
            move = await player.get_move(engine)  # ここで待機

        # GUIのクリックハンドラ
        def on_board_click(x, y):
            player.set_move(Position(x, y))  # 待機を解除
    """

    def __init__(self, name: str = "Human"):
        self._name = name
        self._pending_move: Optional[Position] = None
        self._is_cancelled = False
        self._waiting = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_human(self) -> bool:
        return True

    async def get_move(self, engine: GameEngine) -> Position:
        """
        人間の入力を待機

        set_move() が呼ばれるまでブロックします。
        タイムアウトは GameSession 側で管理します。
        """
        self._pending_move = None
        self._is_cancelled = False
        self._waiting = True

        # ポーリングで待機（asyncio.Eventの代わり）
        while self._pending_move is None and not self._is_cancelled:
            await asyncio.sleep(0.05)  # 50msごとにチェック

        self._waiting = False

        if self._is_cancelled or self._pending_move is None:
            raise asyncio.CancelledError("Move input was cancelled")

        return self._pending_move

    def set_move(self, position: Position) -> None:
        """
        GUIから呼ばれる: 人間が選択した手を設定

        Args:
            position: 選択した座標
        """
        if self._waiting:
            self._pending_move = position
            self._is_cancelled = False

    def cancel_move(self) -> None:
        """
        入力待ちをキャンセル（ゲーム中断時など）
        """
        self._is_cancelled = True
        self._pending_move = None


class AIPlayer(Player):
    """
    AIプレイヤー

    AIStrategy を使用して手を決定します。
    計算処理は asyncio.to_thread() で別スレッドで実行し、
    UIをブロックしません。

    使用例:
        from ai_strategies import MinimaxAI

        strategy = MinimaxAI(depth=3)
        player = AIPlayer(strategy)

        move = await player.get_move(engine)
    """

    def __init__(self, strategy: AIStrategy, name: Optional[str] = None):
        """
        Args:
            strategy: 使用するAI戦略
            name: プレイヤー名（省略時は戦略名を使用）
        """
        self._strategy = strategy
        self._name = name or f"AI ({strategy.name})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_human(self) -> bool:
        return False

    @property
    def strategy(self) -> AIStrategy:
        """使用中のAI戦略"""
        return self._strategy

    async def get_move(self, engine: GameEngine) -> Position:
        """
        AI戦略を使用して次の手を決定

        計算は別スレッドで実行されるため、UIはブロックされません。
        """
        # engine.board はコピーを返すので、AIは安全に評価できる
        board = engine.board
        rule = engine.rule
        stone = engine.current_turn

        # 計算処理を別スレッドで実行
        move = await asyncio.to_thread(
            self._strategy.select_move,
            board,
            rule,
            stone
        )

        return move


# ----------------------
# ゲームセッション
# ----------------------

@dataclass
class GameSessionConfig:
    """ゲームセッションの設定"""
    move_timeout: Optional[float] = None  # 1手あたりのタイムアウト（秒）
    game_timeout: Optional[float] = None  # ゲーム全体のタイムアウト（秒）
    delay_between_moves: float = 0.0      # 手の間の遅延（秒）- 観戦用


@dataclass
class SessionEvent:
    """
    ゲームセッションイベント

    GameEventとは別に、セッションレベルのイベントを表します。
    （ターン開始、タイムアウト、セッション開始/終了など）
    """
    event_type: str  # "GAME_START", "TURN_START", "MOVE_TIMEOUT", "GAME_END"
    current_player: Optional[Player] = None
    stone: Optional[Stone] = None
    result: Optional[GameStatus] = None
    message: str = ""


# セッションイベントのコールバック型
SessionEventCallback = Callable[[SessionEvent], None]


class GameSession:
    """
    ゲームセッション管理

    プレイヤーを管理し、非同期ゲームループを実行します。
    Human vs CPU、CPU vs CPU の両方に対応します。

    使用例:
        rule = StandardGomokuRule()
        black = HumanPlayer("You")
        white = AIPlayer(MinimaxAI(depth=3))

        session = GameSession(rule, black, white)
        result = await session.run_game()
    """

    def __init__(
        self,
        rule: GameRule,
        black_player: Player,
        white_player: Player,
        config: Optional[GameSessionConfig] = None
    ):
        """
        Args:
            rule: 使用するゲームルール
            black_player: 黒番プレイヤー
            white_player: 白番プレイヤー
            config: セッション設定（省略時はデフォルト）
        """
        self._engine = GameEngine(rule)
        self._players = {
            Stone.BLACK: black_player,
            Stone.WHITE: white_player
        }
        self._config = config or GameSessionConfig()
        self._listeners: list[SessionEventCallback] = []
        self._is_running = False
        self._is_cancelled = False

    @property
    def engine(self) -> GameEngine:
        """ゲームエンジン"""
        return self._engine

    @property
    def black_player(self) -> Player:
        """黒番プレイヤー"""
        return self._players[Stone.BLACK]

    @property
    def white_player(self) -> Player:
        """白番プレイヤー"""
        return self._players[Stone.WHITE]

    @property
    def current_player(self) -> Player:
        """現在の手番のプレイヤー"""
        return self._players[self._engine.current_turn]

    @property
    def is_running(self) -> bool:
        """ゲームが実行中かどうか"""
        return self._is_running

    def add_listener(self, callback: SessionEventCallback) -> None:
        """セッションイベントリスナーを登録"""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: SessionEventCallback) -> bool:
        """セッションイベントリスナーを解除"""
        if callback in self._listeners:
            self._listeners.remove(callback)
            return True
        return False

    def _notify_listeners(self, event: SessionEvent) -> None:
        """全リスナーにイベントを通知"""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                print(f"Session listener error: {e}")

    def cancel(self) -> None:
        """ゲームをキャンセル"""
        self._is_cancelled = True
        # HumanPlayer の入力待ちをキャンセル
        for player in self._players.values():
            if isinstance(player, HumanPlayer):
                player.cancel_move()

    async def run_game(self) -> GameStatus:
        """
        ゲームループを非同期で実行

        Returns:
            ゲーム結果（GameStatus）

        Raises:
            RuntimeError: ゲームが既に実行中の場合
            asyncio.TimeoutError: ゲーム全体がタイムアウトした場合
        """
        if self._is_running:
            raise RuntimeError("Game is already running")

        self._is_running = True
        self._is_cancelled = False

        try:
            # ゲーム開始通知
            for stone, player in self._players.items():
                player.on_game_start(self._engine, stone)

            self._notify_listeners(SessionEvent(
                event_type="GAME_START",
                message="Game started"
            ))

            # メインゲームループ
            while not self._engine.is_game_over and not self._is_cancelled:
                current_stone = self._engine.current_turn
                current_player = self._players[current_stone]

                # ターン開始通知
                self._notify_listeners(SessionEvent(
                    event_type="TURN_START",
                    current_player=current_player,
                    stone=current_stone
                ))

                # 手を取得（タイムアウト付き）
                try:
                    if self._config.move_timeout:
                        move = await asyncio.wait_for(
                            current_player.get_move(self._engine),
                            timeout=self._config.move_timeout
                        )
                    else:
                        move = await current_player.get_move(self._engine)
                except asyncio.TimeoutError:
                    self._notify_listeners(SessionEvent(
                        event_type="MOVE_TIMEOUT",
                        current_player=current_player,
                        stone=current_stone,
                        message=f"{current_player.name} timed out"
                    ))
                    raise
                except asyncio.CancelledError:
                    # キャンセルされた場合は静かに終了
                    break

                # 手を実行
                if not self._is_cancelled:
                    success = self._engine.play_move(move.x, move.y)
                    if not success:
                        # 不正な手の場合（通常は起こらないはず）
                        raise RuntimeError(
                            f"Invalid move by {current_player.name}: ({move.x}, {move.y})"
                        )

                # 手の間の遅延（観戦モード用）
                if self._config.delay_between_moves > 0 and not self._engine.is_game_over:
                    await asyncio.sleep(self._config.delay_between_moves)

            # ゲーム終了
            result = self._engine.status

            for player in self._players.values():
                player.on_game_end(self._engine, result)

            self._notify_listeners(SessionEvent(
                event_type="GAME_END",
                result=result,
                message=f"Game ended: {result.name}"
            ))

            return result

        finally:
            self._is_running = False

    def reset(self) -> None:
        """ゲームをリセット"""
        if self._is_running:
            self.cancel()
        self._engine.reset()
        self._is_cancelled = False
