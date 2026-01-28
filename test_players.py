"""
Variant Go Platform - Player Tests

プレイヤークラスのユニットテストを提供します。
"""

import pytest
import asyncio

from game_core import GameEngine, StandardGomokuRule, Position, Stone, GameStatus
from players import (
    Player, HumanPlayer, AIPlayer,
    GameSession, GameSessionConfig, SessionEvent
)
from ai_strategies import RandomAI


class TestHumanPlayer:
    """HumanPlayerのテスト"""

    def test_name(self):
        """名前が正しく設定される"""
        player = HumanPlayer("Alice")
        assert player.name == "Alice"

    def test_default_name(self):
        """デフォルト名はHuman"""
        player = HumanPlayer()
        assert player.name == "Human"

    def test_is_human(self):
        """is_humanがTrue"""
        player = HumanPlayer()
        assert player.is_human is True

    @pytest.mark.asyncio
    async def test_waits_for_move(self):
        """set_move()が呼ばれるまで待機する"""
        player = HumanPlayer()
        engine = GameEngine(StandardGomokuRule())

        async def set_move_after_delay():
            await asyncio.sleep(0.05)
            player.set_move(Position(7, 7))

        asyncio.create_task(set_move_after_delay())
        move = await player.get_move(engine)

        assert move == Position(7, 7)

    @pytest.mark.asyncio
    async def test_cancel_move(self):
        """キャンセルで待機解除（CancelledError）"""
        player = HumanPlayer()
        engine = GameEngine(StandardGomokuRule())

        async def cancel_after_delay():
            await asyncio.sleep(0.05)
            player.cancel_move()

        asyncio.create_task(cancel_after_delay())

        with pytest.raises(asyncio.CancelledError):
            await player.get_move(engine)

    @pytest.mark.asyncio
    async def test_multiple_moves(self):
        """複数回の入力待ち"""
        player = HumanPlayer()
        engine = GameEngine(StandardGomokuRule())

        # 1回目
        async def set_move1():
            await asyncio.sleep(0.05)
            player.set_move(Position(0, 0))

        asyncio.create_task(set_move1())
        move1 = await player.get_move(engine)
        assert move1 == Position(0, 0)

        # 2回目
        async def set_move2():
            await asyncio.sleep(0.05)
            player.set_move(Position(1, 1))

        asyncio.create_task(set_move2())
        move2 = await player.get_move(engine)
        assert move2 == Position(1, 1)


class TestAIPlayer:
    """AIPlayerのテスト"""

    def test_name_from_strategy(self):
        """戦略名がプレイヤー名に使われる"""
        strategy = RandomAI()
        player = AIPlayer(strategy)
        assert "Random" in player.name

    def test_custom_name(self):
        """カスタム名を設定できる"""
        strategy = RandomAI()
        player = AIPlayer(strategy, name="Bob")
        assert player.name == "Bob"

    def test_is_human(self):
        """is_humanがFalse"""
        player = AIPlayer(RandomAI())
        assert player.is_human is False

    def test_strategy_property(self):
        """strategy プロパティ"""
        strategy = RandomAI()
        player = AIPlayer(strategy)
        assert player.strategy is strategy

    @pytest.mark.asyncio
    async def test_returns_valid_move(self):
        """AIが合法手を返す"""
        strategy = RandomAI(seed=42)
        player = AIPlayer(strategy)
        engine = GameEngine(StandardGomokuRule())

        move = await player.get_move(engine)

        assert engine.rule.is_valid_move(engine.board, move.x, move.y, Stone.BLACK)

    @pytest.mark.asyncio
    async def test_does_not_block_event_loop(self):
        """イベントループをブロックしない"""
        strategy = RandomAI()
        player = AIPlayer(strategy)
        engine = GameEngine(StandardGomokuRule())

        # タイムアウト付きで実行
        move = await asyncio.wait_for(player.get_move(engine), timeout=5.0)

        assert move is not None


class TestGameSession:
    """GameSessionのテスト"""

    def test_properties(self):
        """プロパティが正しい"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(), name="Black")
        white = AIPlayer(RandomAI(), name="White")

        session = GameSession(rule, black, white)

        assert session.black_player is black
        assert session.white_player is white
        assert session.engine is not None
        assert session.is_running is False

    def test_current_player(self):
        """current_playerが手番に応じて変わる"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(), name="Black")
        white = AIPlayer(RandomAI(), name="White")

        session = GameSession(rule, black, white)

        assert session.current_player is black  # 黒先手

    @pytest.mark.asyncio
    async def test_run_game_completes(self):
        """ゲームが終了まで実行される"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)
        result = await asyncio.wait_for(session.run_game(), timeout=30.0)

        assert result in [GameStatus.BLACK_WIN, GameStatus.WHITE_WIN, GameStatus.DRAW]
        assert session.engine.is_game_over

    @pytest.mark.asyncio
    async def test_already_running_raises(self):
        """実行中に再度run_gameを呼ぶとエラー"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)

        async def run_twice():
            task = asyncio.create_task(session.run_game())
            await asyncio.sleep(0.01)  # 少し待つ
            if session.is_running:
                with pytest.raises(RuntimeError, match="already running"):
                    await session.run_game()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await run_twice()

    @pytest.mark.asyncio
    async def test_cancel_game(self):
        """ゲームをキャンセルできる"""
        rule = StandardGomokuRule()
        black = HumanPlayer("Human")  # 入力待ちになる
        white = AIPlayer(RandomAI())

        session = GameSession(rule, black, white)

        async def cancel_soon():
            await asyncio.sleep(0.05)
            session.cancel()

        asyncio.create_task(cancel_soon())

        # キャンセルされてもエラーにならずに終了
        result = await session.run_game()
        assert result == GameStatus.ONGOING  # まだ終わっていない

    @pytest.mark.asyncio
    async def test_reset(self):
        """リセットで初期状態に戻る"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)
        await session.run_game()

        session.reset()

        assert session.engine.status == GameStatus.ONGOING
        assert session.is_running is False


class TestGameSessionConfig:
    """GameSessionConfigのテスト"""

    def test_default_config(self):
        """デフォルト設定"""
        config = GameSessionConfig()
        assert config.move_timeout is None
        assert config.game_timeout is None
        assert config.delay_between_moves == 0.0

    def test_custom_config(self):
        """カスタム設定"""
        config = GameSessionConfig(
            move_timeout=30.0,
            game_timeout=600.0,
            delay_between_moves=0.5
        )
        assert config.move_timeout == 30.0
        assert config.game_timeout == 600.0
        assert config.delay_between_moves == 0.5


class TestSessionEvent:
    """SessionEventのテスト"""

    def test_game_start_event(self):
        """ゲーム開始イベント"""
        event = SessionEvent(
            event_type="GAME_START",
            message="Game started"
        )
        assert event.event_type == "GAME_START"
        assert event.current_player is None

    def test_turn_start_event(self):
        """ターン開始イベント"""
        player = AIPlayer(RandomAI())
        event = SessionEvent(
            event_type="TURN_START",
            current_player=player,
            stone=Stone.BLACK
        )
        assert event.event_type == "TURN_START"
        assert event.current_player is player
        assert event.stone == Stone.BLACK

    def test_game_end_event(self):
        """ゲーム終了イベント"""
        event = SessionEvent(
            event_type="GAME_END",
            result=GameStatus.BLACK_WIN,
            message="Black wins!"
        )
        assert event.event_type == "GAME_END"
        assert event.result == GameStatus.BLACK_WIN


class TestSessionListeners:
    """セッションリスナーのテスト"""

    @pytest.mark.asyncio
    async def test_listener_called(self):
        """リスナーが呼ばれる"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)

        events: list[SessionEvent] = []

        def listener(event: SessionEvent):
            events.append(event)

        session.add_listener(listener)

        await asyncio.wait_for(session.run_game(), timeout=30.0)

        # GAME_START, 複数のTURN_START, GAME_END
        assert any(e.event_type == "GAME_START" for e in events)
        assert any(e.event_type == "TURN_START" for e in events)
        assert any(e.event_type == "GAME_END" for e in events)

    @pytest.mark.asyncio
    async def test_remove_listener(self):
        """リスナーを解除できる"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)

        events: list[SessionEvent] = []

        def listener(event: SessionEvent):
            events.append(event)

        session.add_listener(listener)
        result = session.remove_listener(listener)
        assert result is True

        await asyncio.wait_for(session.run_game(), timeout=30.0)

        # リスナーが解除されているのでイベントは記録されない
        assert len(events) == 0

    @pytest.mark.asyncio
    async def test_listener_error_does_not_stop_game(self):
        """リスナーがエラーを起こしてもゲームは継続"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)

        def failing_listener(event: SessionEvent):
            raise Exception("Listener error!")

        session.add_listener(failing_listener)

        # エラーが起きてもゲームは完了する
        result = await asyncio.wait_for(session.run_game(), timeout=30.0)
        assert result in [GameStatus.BLACK_WIN, GameStatus.WHITE_WIN, GameStatus.DRAW]


class TestPlayerLifecycleHooks:
    """プレイヤーライフサイクルフックのテスト"""

    @pytest.mark.asyncio
    async def test_on_game_start_called(self):
        """on_game_startが呼ばれる"""
        rule = StandardGomokuRule()

        class TrackingPlayer(AIPlayer):
            def __init__(self, strategy):
                super().__init__(strategy)
                self.started = False
                self.assigned_stone = None

            def on_game_start(self, engine, stone):
                self.started = True
                self.assigned_stone = stone

        black = TrackingPlayer(RandomAI(seed=1))
        white = TrackingPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)
        await asyncio.wait_for(session.run_game(), timeout=30.0)

        assert black.started is True
        assert black.assigned_stone == Stone.BLACK
        assert white.started is True
        assert white.assigned_stone == Stone.WHITE

    @pytest.mark.asyncio
    async def test_on_game_end_called(self):
        """on_game_endが呼ばれる"""
        rule = StandardGomokuRule()

        class TrackingPlayer(AIPlayer):
            def __init__(self, strategy):
                super().__init__(strategy)
                self.ended = False
                self.game_result = None

            def on_game_end(self, engine, result):
                self.ended = True
                self.game_result = result

        black = TrackingPlayer(RandomAI(seed=1))
        white = TrackingPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)
        result = await asyncio.wait_for(session.run_game(), timeout=30.0)

        assert black.ended is True
        assert black.game_result == result
        assert white.ended is True
        assert white.game_result == result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
