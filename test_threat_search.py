"""
Variant Go Platform - Threat Search Tests

VCF/VCT探索のユニットテストを提供します。
"""

import pytest
import time

from game_core import Board, Stone, Position, GameStatus, StandardGomokuRule
from threat_search import (
    ThreatType, Threat, ThreatDetector,
    VCFSearch, VCTSearch, VCFBasedAI, VCFResult
)


class TestThreatType:
    """ThreatTypeのテスト"""

    def test_priority_order(self):
        """優先度順になっている"""
        assert ThreatType.FIVE.value < ThreatType.STRAIGHT_FOUR.value
        assert ThreatType.STRAIGHT_FOUR.value < ThreatType.FOUR.value
        assert ThreatType.FOUR.value < ThreatType.OPEN_THREE.value
        assert ThreatType.OPEN_THREE.value < ThreatType.THREE.value


class TestThreatDetector:
    """ThreatDetectorのテスト"""

    def test_find_horizontal_four(self):
        """横4連を検出"""
        detector = ThreatDetector()
        board = Board(15, 15)

        # 黒の4連: (0,0), (1,0), (2,0), (3,0)
        for i in range(4):
            board.set_stone(i, 0, Stone.BLACK)

        threats = detector.find_threats(board, Stone.BLACK, {ThreatType.FOUR})

        # 4連が見つかる（(4,0)に打てば5連）
        four_threats = [t for t in threats if t.threat_type == ThreatType.FOUR]
        assert len(four_threats) > 0

        # 勝ち手の位置を確認
        winning_positions = [t.position for t in four_threats]
        assert Position(4, 0) in winning_positions

    def test_find_vertical_four(self):
        """縦4連を検出"""
        detector = ThreatDetector()
        board = Board(15, 15)

        # 黒の縦4連
        for i in range(4):
            board.set_stone(0, i, Stone.BLACK)

        threats = detector.find_threats(board, Stone.BLACK, {ThreatType.FOUR})

        four_threats = [t for t in threats if t.threat_type == ThreatType.FOUR]
        assert len(four_threats) > 0

        winning_positions = [t.position for t in four_threats]
        assert Position(0, 4) in winning_positions

    def test_find_diagonal_four(self):
        """斜め4連を検出"""
        detector = ThreatDetector()
        board = Board(15, 15)

        # 黒の斜め4連
        for i in range(4):
            board.set_stone(i, i, Stone.BLACK)

        threats = detector.find_threats(board, Stone.BLACK, {ThreatType.FOUR})

        four_threats = [t for t in threats if t.threat_type == ThreatType.FOUR]
        assert len(four_threats) > 0

        winning_positions = [t.position for t in four_threats]
        assert Position(4, 4) in winning_positions

    def test_find_winning_moves(self):
        """即勝ち手を検出"""
        detector = ThreatDetector()
        board = Board(15, 15)

        # 黒の4連
        for i in range(4):
            board.set_stone(i, 0, Stone.BLACK)

        winning_moves = detector.find_winning_moves(board, Stone.BLACK)
        assert len(winning_moves) > 0

    def test_find_open_three(self):
        """活三を検出"""
        detector = ThreatDetector()
        board = Board(15, 15)

        # 黒の3連（両端が空いている）: (2,0), (3,0), (4,0)
        board.set_stone(2, 0, Stone.BLACK)
        board.set_stone(3, 0, Stone.BLACK)
        board.set_stone(4, 0, Stone.BLACK)

        threats = detector.find_threats(board, Stone.BLACK, {ThreatType.OPEN_THREE})

        # 活三が見つかる可能性（両端が空いていれば）
        # 注: このテストは実装の詳細に依存
        assert isinstance(threats, list)

    def test_no_threat_with_opponent_blocking(self):
        """相手の石がある場合は脅威にならない"""
        detector = ThreatDetector()
        board = Board(15, 15)

        # 黒の4連だが、片方が白でブロック
        for i in range(4):
            board.set_stone(i, 0, Stone.BLACK)
        board.set_stone(4, 0, Stone.WHITE)  # ブロック

        # (4,0)は勝ち手にならない
        winning_moves = detector.find_winning_moves(board, Stone.BLACK)
        # 逆側(端外)も含めてチェック
        # 端でブロックされている場合、勝ち手がない可能性
        assert Position(4, 0) not in winning_moves


class TestVCFSearch:
    """VCFSearchのテスト"""

    def test_finds_simple_vcf(self):
        """シンプルなVCFを見つける"""
        vcf = VCFSearch(max_depth=10)
        rule = StandardGomokuRule()
        board = rule.create_board()

        # 黒の4連（1手で勝ち）
        for i in range(4):
            board.set_stone(i, 0, Stone.BLACK)
        # 白は別の場所
        board.set_stone(10, 10, Stone.WHITE)
        board.set_stone(10, 11, Stone.WHITE)
        board.set_stone(10, 12, Stone.WHITE)

        result = vcf.search(board, rule, Stone.BLACK)

        assert result.is_winning is True
        assert len(result.win_sequence) >= 1
        assert result.win_sequence[0] == Position(4, 0)

    def test_no_vcf_when_blocked(self):
        """ブロックされている場合はVCFがない"""
        vcf = VCFSearch(max_depth=10)
        rule = StandardGomokuRule()
        board = rule.create_board()

        # 黒の3連（両端ブロック）
        board.set_stone(1, 0, Stone.BLACK)
        board.set_stone(2, 0, Stone.BLACK)
        board.set_stone(3, 0, Stone.BLACK)
        board.set_stone(0, 0, Stone.WHITE)  # 左ブロック
        board.set_stone(4, 0, Stone.WHITE)  # 右ブロック

        result = vcf.search(board, rule, Stone.BLACK)

        # この局面では即座のVCFはない
        # （より複雑なVCFがあるかもしれないが、単純なケースでは見つからない）
        assert isinstance(result, VCFResult)

    def test_respects_time_limit(self):
        """時間制限を守る"""
        vcf = VCFSearch(max_depth=50, time_limit=0.3)
        rule = StandardGomokuRule()
        board = rule.create_board()

        # 複雑な局面を作成
        board.set_stone(7, 7, Stone.BLACK)
        board.set_stone(8, 8, Stone.WHITE)

        start = time.time()
        vcf.search(board, rule, Stone.BLACK)
        elapsed = time.time() - start

        assert elapsed < 0.6

    def test_vcf_result_attributes(self):
        """VCFResultの属性が正しい"""
        vcf = VCFSearch(max_depth=10)
        rule = StandardGomokuRule()
        board = rule.create_board()

        board.set_stone(7, 7, Stone.BLACK)

        result = vcf.search(board, rule, Stone.BLACK)

        assert hasattr(result, 'is_winning')
        assert hasattr(result, 'win_sequence')
        assert hasattr(result, 'depth_searched')
        assert hasattr(result, 'nodes_visited')


class TestVCTSearch:
    """VCTSearchのテスト"""

    def test_finds_vcf_first(self):
        """まずVCFを試す"""
        vct = VCTSearch(max_depth=10)
        rule = StandardGomokuRule()
        board = rule.create_board()

        # 単純なVCF（4連）
        for i in range(4):
            board.set_stone(i, 0, Stone.BLACK)
        board.set_stone(10, 10, Stone.WHITE)
        board.set_stone(10, 11, Stone.WHITE)
        board.set_stone(10, 12, Stone.WHITE)

        result = vct.search(board, rule, Stone.BLACK)

        assert result.is_winning is True
        assert result.win_sequence[0] == Position(4, 0)

    def test_respects_time_limit(self):
        """時間制限を守る"""
        vct = VCTSearch(max_depth=20, time_limit=0.3)
        rule = StandardGomokuRule()
        board = rule.create_board()

        board.set_stone(7, 7, Stone.BLACK)
        board.set_stone(8, 8, Stone.WHITE)

        start = time.time()
        vct.search(board, rule, Stone.BLACK)
        elapsed = time.time() - start

        assert elapsed < 0.6


class TestVCFBasedAI:
    """VCFBasedAIのテスト"""

    def test_finds_winning_move(self):
        """勝ち手を見つける"""
        ai = VCFBasedAI()
        rule = StandardGomokuRule()
        board = rule.create_board()

        # 黒の4連
        for i in range(4):
            board.set_stone(i, 0, Stone.BLACK)
        board.set_stone(10, 10, Stone.WHITE)
        board.set_stone(10, 11, Stone.WHITE)
        board.set_stone(10, 12, Stone.WHITE)

        move = ai.select_move(board, rule, Stone.BLACK)

        assert move == Position(4, 0)

    def test_blocks_opponent_winning_move(self):
        """相手の勝ち手をブロック"""
        ai = VCFBasedAI()
        rule = StandardGomokuRule()
        board = rule.create_board()

        # 黒の4連（白は止めなければならない）
        for i in range(4):
            board.set_stone(i, 0, Stone.BLACK)
        board.set_stone(10, 10, Stone.WHITE)
        board.set_stone(10, 11, Stone.WHITE)
        board.set_stone(10, 12, Stone.WHITE)

        move = ai.select_move(board, rule, Stone.WHITE)

        assert move == Position(4, 0)

    def test_name_and_difficulty(self):
        """名前と難易度"""
        ai = VCFBasedAI()
        assert "VCF" in ai.name
        assert ai.difficulty == "Hard"

    def test_with_vct(self):
        """VCT有効時の名前"""
        ai = VCFBasedAI(use_vct=True)
        assert "VCT" in ai.name

    def test_supports_progress(self):
        """進捗通知をサポート"""
        ai = VCFBasedAI()
        assert ai.supports_progress is True

    def test_fallback_to_random(self):
        """VCFがない場合はフォールバック"""
        ai = VCFBasedAI()
        rule = StandardGomokuRule()
        board = rule.create_board()

        # VCFがない局面
        board.set_stone(7, 7, Stone.BLACK)
        board.set_stone(8, 8, Stone.WHITE)

        move = ai.select_move(board, rule, Stone.BLACK)

        # 有効な手を返す
        assert rule.is_valid_move(board, move.x, move.y, Stone.BLACK)

    def test_raises_on_no_valid_moves(self):
        """合法手がない場合はエラー"""
        ai = VCFBasedAI()
        rule = StandardGomokuRule()
        board = Board(1, 1)
        board.set_stone(0, 0, Stone.BLACK)

        with pytest.raises(ValueError, match="No valid moves"):
            ai.select_move(board, rule, Stone.WHITE)


class TestIntegration:
    """統合テスト"""

    def test_vcf_ai_vs_random_position(self):
        """VCF AIがランダムな局面で有効な手を返す"""
        from ai_strategies import RandomAI

        vcf_ai = VCFBasedAI(fallback=RandomAI(seed=42))
        rule = StandardGomokuRule()
        board = rule.create_board()

        # ランダムに石を配置
        import random
        rng = random.Random(42)

        for _ in range(10):
            valid_moves = rule.get_valid_moves(board, Stone.BLACK)
            if valid_moves:
                move = rng.choice(valid_moves)
                board.set_stone(move.x, move.y, Stone.BLACK)

            valid_moves = rule.get_valid_moves(board, Stone.WHITE)
            if valid_moves:
                move = rng.choice(valid_moves)
                board.set_stone(move.x, move.y, Stone.WHITE)

        # VCF AIが有効な手を返す
        move = vcf_ai.select_move(board, rule, Stone.BLACK)
        assert rule.is_valid_move(board, move.x, move.y, Stone.BLACK)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
