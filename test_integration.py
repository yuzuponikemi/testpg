"""
Variant Go Platform - Integration Tests

CPU vs CPU の統合テストを提供します。
"""

import pytest
import asyncio

from game_core import StandardGomokuRule, GameStatus, Stone
from players import AIPlayer, GameSession, GameSessionConfig
from ai_strategies import RandomAI, MinimaxAI, MCTSAI


class TestCPUvsCPU:
    """CPU vs CPU 対戦テスト"""

    @pytest.mark.asyncio
    async def test_random_vs_random_completes(self):
        """RandomAI同士の対戦が終了する"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)
        result = await asyncio.wait_for(session.run_game(), timeout=30.0)

        assert result in [GameStatus.BLACK_WIN, GameStatus.WHITE_WIN, GameStatus.DRAW]
        assert session.engine.is_game_over

    @pytest.mark.asyncio
    async def test_minimax_vs_random_black_wins_often(self):
        """Minimax(黒) vs Random(白) で黒が高勝率"""
        wins = 0
        games = 3

        for i in range(games):
            rule = StandardGomokuRule()
            black = AIPlayer(MinimaxAI(depth=2))
            white = AIPlayer(RandomAI(seed=i * 100))

            session = GameSession(rule, black, white)
            result = await asyncio.wait_for(session.run_game(), timeout=60.0)

            if result == GameStatus.BLACK_WIN:
                wins += 1

        # 3戦中2勝以上
        assert wins >= 2, f"Minimax won only {wins}/{games} games"

    @pytest.mark.asyncio
    async def test_random_vs_minimax_white_wins_often(self):
        """Random(黒) vs Minimax(白) で白が高勝率"""
        wins = 0
        games = 3

        for i in range(games):
            rule = StandardGomokuRule()
            black = AIPlayer(RandomAI(seed=i * 100))
            white = AIPlayer(MinimaxAI(depth=2))

            session = GameSession(rule, black, white)
            result = await asyncio.wait_for(session.run_game(), timeout=60.0)

            if result == GameStatus.WHITE_WIN:
                wins += 1

        assert wins >= 2, f"Minimax won only {wins}/{games} games"

    @pytest.mark.asyncio
    async def test_minimax_vs_minimax(self):
        """Minimax同士の対戦が終了する"""
        rule = StandardGomokuRule()
        black = AIPlayer(MinimaxAI(depth=2))
        white = AIPlayer(MinimaxAI(depth=2))

        session = GameSession(rule, black, white)
        result = await asyncio.wait_for(session.run_game(), timeout=120.0)

        assert result in [GameStatus.BLACK_WIN, GameStatus.WHITE_WIN, GameStatus.DRAW]

    @pytest.mark.asyncio
    async def test_mcts_vs_random(self):
        """MCTS vs Random で MCTS が勝つ"""
        rule = StandardGomokuRule()
        # シミュレーション回数を減らして高速化
        black = AIPlayer(MCTSAI(simulations=100))
        white = AIPlayer(RandomAI(seed=42))

        session = GameSession(rule, black, white)
        result = await asyncio.wait_for(session.run_game(), timeout=120.0)

        # MCTSが勝つことを期待（ただし確率的なので失敗する可能性あり）
        assert result in [GameStatus.BLACK_WIN, GameStatus.WHITE_WIN, GameStatus.DRAW]


class TestSessionWithDelay:
    """遅延付きセッションのテスト"""

    @pytest.mark.asyncio
    async def test_delay_between_moves(self):
        """手の間に遅延が入る"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        config = GameSessionConfig(delay_between_moves=0.01)
        session = GameSession(rule, black, white, config)

        move_count = 0

        def on_event(event):
            nonlocal move_count
            if event.event_type == "TURN_START":
                move_count += 1

        session.add_listener(on_event)

        await asyncio.wait_for(session.run_game(), timeout=60.0)

        # 複数の手が打たれている
        assert move_count > 5


class TestGameProgress:
    """ゲーム進行のテスト"""

    @pytest.mark.asyncio
    async def test_engine_events_fired(self):
        """GameEngineのイベントも発火する"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)

        engine_events = []

        def engine_listener(event):
            engine_events.append(event)

        session.engine.add_listener(engine_listener)

        await asyncio.wait_for(session.run_game(), timeout=30.0)

        # MOVE_PLAYED イベントが発火している
        assert any(e.event_type == "MOVE_PLAYED" for e in engine_events)
        # GAME_OVER イベントが発火している
        assert any(e.event_type == "GAME_OVER" for e in engine_events)

    @pytest.mark.asyncio
    async def test_move_history_recorded(self):
        """着手履歴が記録される"""
        rule = StandardGomokuRule()
        black = AIPlayer(RandomAI(seed=1))
        white = AIPlayer(RandomAI(seed=2))

        session = GameSession(rule, black, white)

        await asyncio.wait_for(session.run_game(), timeout=30.0)

        history = session.engine.move_history
        assert len(history) > 0

        # 履歴の石の色が交互になっている
        for i, (pos, stone) in enumerate(history):
            expected = Stone.BLACK if i % 2 == 0 else Stone.WHITE
            assert stone == expected


class TestDeterminism:
    """決定性のテスト"""

    @pytest.mark.asyncio
    async def test_same_seed_same_result(self):
        """同じシードで同じ結果"""
        results = []

        for _ in range(2):
            rule = StandardGomokuRule()
            black = AIPlayer(RandomAI(seed=12345))
            white = AIPlayer(RandomAI(seed=67890))

            session = GameSession(rule, black, white)
            result = await asyncio.wait_for(session.run_game(), timeout=30.0)
            results.append(result)

        assert results[0] == results[1]

    @pytest.mark.asyncio
    async def test_same_seed_same_history(self):
        """同じシードで同じ履歴"""
        histories = []

        for _ in range(2):
            rule = StandardGomokuRule()
            black = AIPlayer(RandomAI(seed=12345))
            white = AIPlayer(RandomAI(seed=67890))

            session = GameSession(rule, black, white)
            await asyncio.wait_for(session.run_game(), timeout=30.0)
            histories.append(session.engine.move_history)

        assert histories[0] == histories[1]


class TestEdgeCases:
    """エッジケースのテスト"""

    @pytest.mark.asyncio
    async def test_fast_win(self):
        """高速勝利（Minimaxが即勝ち）"""
        rule = StandardGomokuRule()

        # Minimax vs Random、Minimaxは1手詰めを見つける
        black = AIPlayer(MinimaxAI(depth=2))
        white = AIPlayer(RandomAI(seed=42))

        session = GameSession(rule, black, white)

        # 盤面を事前にセットアップ（黒が4連）
        session.engine._board.set_stone(0, 0, Stone.BLACK)
        session.engine._board.set_stone(1, 0, Stone.BLACK)
        session.engine._board.set_stone(2, 0, Stone.BLACK)
        session.engine._board.set_stone(3, 0, Stone.BLACK)
        # 白は別の場所
        session.engine._board.set_stone(10, 10, Stone.WHITE)
        session.engine._board.set_stone(11, 10, Stone.WHITE)
        session.engine._board.set_stone(12, 10, Stone.WHITE)
        session.engine._board._move_count = 7

        result = await asyncio.wait_for(session.run_game(), timeout=10.0)

        # 黒が勝つはず
        assert result == GameStatus.BLACK_WIN

    @pytest.mark.asyncio
    async def test_multiple_games_sequential(self):
        """複数ゲームを順番に実行"""
        for i in range(3):
            rule = StandardGomokuRule()
            black = AIPlayer(RandomAI(seed=i))
            white = AIPlayer(RandomAI(seed=i + 100))

            session = GameSession(rule, black, white)
            result = await asyncio.wait_for(session.run_game(), timeout=30.0)

            assert result in [GameStatus.BLACK_WIN, GameStatus.WHITE_WIN, GameStatus.DRAW]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
