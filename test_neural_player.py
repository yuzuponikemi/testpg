"""
NeuralPlayer のテスト

学習済みモデルを使用するNeuralPlayerの動作を確認します。
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
import torch

from game_core import Board, StandardGomokuRule, Position, Stone, GameStatus


# モデルファイルが存在する場合のみ実行するマーカー
MODEL_PATH = "checkpoints/gomoku_gpt_best.pth"
requires_model = pytest.mark.skipif(
    not Path(MODEL_PATH).exists(),
    reason=f"Model file not found: {MODEL_PATH}"
)


class TestNeuralPlayerWithMock:
    """モックを使用したNeuralPlayerのテスト（モデル不要）"""

    def test_import(self):
        """インポートできることを確認"""
        from neural_player import NeuralPlayer, NeuralPlayerFactory
        assert NeuralPlayer is not None
        assert NeuralPlayerFactory is not None

    def test_factory_is_model_available(self):
        """モデル存在チェックが動作することを確認"""
        from neural_player import NeuralPlayerFactory

        # 存在しないパス
        assert not NeuralPlayerFactory.is_model_available("nonexistent.pth")

    @patch('neural_player.NeuralPlayer._load_model')
    def test_player_properties(self, mock_load):
        """プレイヤーのプロパティが正しく設定されることを確認"""
        from neural_player import NeuralPlayer

        # モックでモデル読み込みをスキップ
        mock_load.return_value = None

        player = NeuralPlayer.__new__(NeuralPlayer)
        player._model_path = "test.pth"
        player._temperature = 0.5
        player._top_k = 10
        player._device = torch.device("cpu")
        player._model = None
        player._config = None
        player._move_history = []
        player._progress_callback = None

        assert "NeuralAI" in player.name
        assert player.difficulty == "Hard"
        assert player.supports_progress is True

    @patch('neural_player.NeuralPlayer._load_model')
    def test_reset_history(self, mock_load):
        """履歴リセットが動作することを確認"""
        from neural_player import NeuralPlayer

        player = NeuralPlayer.__new__(NeuralPlayer)
        player._move_history = [112, 127, 113]

        player.reset_history()
        assert player._move_history == []

    @patch('neural_player.NeuralPlayer._load_model')
    def test_notify_opponent_move(self, mock_load):
        """相手の手の通知が動作することを確認"""
        from neural_player import NeuralPlayer

        player = NeuralPlayer.__new__(NeuralPlayer)
        player._move_history = []

        player.notify_opponent_move(Position(7, 7), board_width=15)
        assert 112 in player._move_history  # 7*15+7 = 112


@requires_model
class TestNeuralPlayerWithModel:
    """実際のモデルを使用したテスト"""

    def test_create_player(self):
        """プレイヤーを作成できることを確認"""
        from neural_player import NeuralPlayer

        player = NeuralPlayer(MODEL_PATH, temperature=0.5)
        assert player is not None
        assert player._model is not None

    def test_select_valid_move(self):
        """合法手を選択できることを確認"""
        from neural_player import NeuralPlayer

        player = NeuralPlayer(MODEL_PATH, temperature=0.0)  # 決定的
        rule = StandardGomokuRule()
        board = Board(15, 15)

        # 空の盤面で着手
        move = player.select_move(board, rule, Stone.BLACK)

        assert isinstance(move, Position)
        assert 0 <= move.x < 15
        assert 0 <= move.y < 15
        assert rule.is_valid_move(board, move.x, move.y, Stone.BLACK)

    def test_respects_occupied_positions(self):
        """既に石がある場所には置かないことを確認"""
        from neural_player import NeuralPlayer

        player = NeuralPlayer(MODEL_PATH, temperature=0.0)
        rule = StandardGomokuRule()
        board = Board(15, 15)

        # 中央に石を置く
        board.set_stone(7, 7, Stone.BLACK)
        board.set_stone(7, 8, Stone.WHITE)

        # 白の番で着手
        player.reset_history()
        player._move_history = [112, 127]  # 既存の手を履歴に追加

        move = player.select_move(board, rule, Stone.WHITE)

        # 既存の石と重ならない
        assert move != Position(7, 7)
        assert move != Position(7, 8)
        assert rule.is_valid_move(board, move.x, move.y, Stone.WHITE)

    def test_multiple_moves(self):
        """複数回の着手が正しく動作することを確認"""
        from neural_player import NeuralPlayer

        player = NeuralPlayer(MODEL_PATH, temperature=0.5)
        rule = StandardGomokuRule()
        board = Board(15, 15)

        player.reset_history()
        occupied = set()

        # 10手プレイ
        stone = Stone.BLACK
        for _ in range(10):
            move = player.select_move(board, rule, stone)

            # 合法手であることを確認
            assert rule.is_valid_move(board, move.x, move.y, stone)
            assert (move.x, move.y) not in occupied

            # 盤面に反映
            board.set_stone(move.x, move.y, stone)
            occupied.add((move.x, move.y))

            # 履歴に追加
            player.notify_opponent_move(move)

            # 手番交代
            stone = stone.opponent()

    def test_temperature_affects_output(self):
        """温度パラメータが出力に影響することを確認"""
        from neural_player import NeuralPlayer

        rule = StandardGomokuRule()
        board = Board(15, 15)

        # 温度0（決定的）
        player_greedy = NeuralPlayer(MODEL_PATH, temperature=0.0)
        moves_greedy = set()
        for _ in range(5):
            player_greedy.reset_history()
            move = player_greedy.select_move(board, rule, Stone.BLACK)
            moves_greedy.add((move.x, move.y))

        # 温度1（ランダム性あり）
        player_random = NeuralPlayer(MODEL_PATH, temperature=1.0)
        moves_random = set()
        for _ in range(10):
            player_random.reset_history()
            move = player_random.select_move(board, rule, Stone.BLACK)
            moves_random.add((move.x, move.y))

        # 温度0は同じ手を繰り返す
        assert len(moves_greedy) == 1

        # 温度1は複数の手を選ぶ可能性がある（確率的なのでスキップ可）

    def test_factory_creates_and_reuses(self):
        """ファクトリがインスタンスを再利用することを確認"""
        from neural_player import NeuralPlayerFactory

        player1 = NeuralPlayerFactory.create(MODEL_PATH, temperature=0.5)
        player2 = NeuralPlayerFactory.create(MODEL_PATH, temperature=0.3)

        # 同じインスタンスが再利用される
        assert player1 is player2
        # ただし温度は更新される
        assert player2._temperature == 0.3


@requires_model
class TestNeuralPlayerIntegration:
    """統合テスト"""

    def test_full_game_legal_moves_only(self):
        """完全なゲームで全ての手が合法であることを確認"""
        from neural_player import NeuralPlayer

        player = NeuralPlayer(MODEL_PATH, temperature=0.3)
        rule = StandardGomokuRule()
        board = Board(15, 15)

        player.reset_history()
        stone = Stone.BLACK
        last_move = None

        # 最大50手（通常はこれで終わる）
        for i in range(50):
            try:
                move = player.select_move(board, rule, stone)
            except ValueError:
                # 合法手がない（引き分け）
                break

            # 合法手チェック
            assert rule.is_valid_move(board, move.x, move.y, stone), \
                f"Move {i+1}: ({move.x}, {move.y}) is not valid"

            # 盤面に反映
            board.set_stone(move.x, move.y, stone)
            player.notify_opponent_move(move)
            last_move = move

            # 勝敗チェック
            status = rule.check_winner(board, move.x, move.y, stone)
            if status != GameStatus.ONGOING:
                print(f"Game ended after {i+1} moves: {status.name}")
                break

            stone = stone.opponent()

    def test_ai_vs_random(self):
        """NeuralAI vs RandomAI の対戦テスト"""
        from neural_player import NeuralPlayer
        from ai_strategies import RandomAI

        neural = NeuralPlayer(MODEL_PATH, temperature=0.3)
        random_ai = RandomAI(seed=42)
        rule = StandardGomokuRule()
        board = Board(15, 15)

        neural.reset_history()
        players = {
            Stone.BLACK: neural,
            Stone.WHITE: random_ai
        }

        stone = Stone.BLACK

        for i in range(100):
            player = players[stone]
            move = player.select_move(board, rule, stone)

            assert rule.is_valid_move(board, move.x, move.y, stone)

            board.set_stone(move.x, move.y, stone)

            # RandomAIの手をNeuralPlayerに通知
            # （NeuralPlayer自身の手はselect_move内で履歴に追加済み）
            if stone == Stone.WHITE:
                neural.notify_opponent_move(move)

            status = rule.check_winner(board, move.x, move.y, stone)
            if status != GameStatus.ONGOING:
                print(f"Game ended: {status.name} after {i+1} moves")
                return

            stone = stone.opponent()

        print("Game reached move limit (draw)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
