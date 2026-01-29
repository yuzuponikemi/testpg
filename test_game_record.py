"""
Variant Go Platform - Game Record Tests

棋譜記録機能のテスト
"""

import pytest
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock

from game_core import Stone, GameStatus, Position, StandardGomokuRule
from game_record import (
    MoveRecord, PlayerRecord, GameRecord,
    GameRecorder, load_game_record, load_game_records_jsonl
)


class TestMoveRecord:
    """MoveRecordのテスト"""

    def test_basic_fields(self):
        """基本フィールドが正しく設定される"""
        move = MoveRecord(
            move_number=1,
            x=7,
            y=7,
            stone="BLACK",
            timestamp=0.5,
        )
        assert move.move_number == 1
        assert move.x == 7
        assert move.y == 7
        assert move.stone == "BLACK"
        assert move.timestamp == 0.5

    def test_optional_fields(self):
        """オプションフィールドが正しく設定される"""
        move = MoveRecord(
            move_number=1,
            x=7,
            y=7,
            stone="BLACK",
            timestamp=0.5,
            thinking_time=1.2,
            eval_score=0.75,
            depth=4,
            nodes=12345,
            pv=[{"x": 8, "y": 8}, {"x": 9, "y": 9}],
        )
        assert move.thinking_time == 1.2
        assert move.eval_score == 0.75
        assert move.depth == 4
        assert move.nodes == 12345
        assert len(move.pv) == 2


class TestPlayerRecord:
    """PlayerRecordのテスト"""

    def test_human_player(self):
        """人間プレイヤーの記録"""
        player = PlayerRecord(
            stone="BLACK",
            player_type="human",
            name="Player 1",
        )
        assert player.stone == "BLACK"
        assert player.player_type == "human"
        assert player.name == "Player 1"

    def test_ai_player(self):
        """AIプレイヤーの記録"""
        player = PlayerRecord(
            stone="WHITE",
            player_type="minimax",
            name="Minimax AI",
            ai_depth=4,
            use_iterative_deepening=True,
        )
        assert player.player_type == "minimax"
        assert player.ai_depth == 4
        assert player.use_iterative_deepening is True


class TestGameRecord:
    """GameRecordのテスト"""

    def test_to_dict(self):
        """辞書変換が正しく動作する"""
        record = GameRecord(
            game_id="20260129_120000_000000",
            timestamp="2026-01-29T12:00:00",
            rule_id="StandardGomokuRule",
            board_width=15,
            board_height=15,
            win_condition=5,
            black_player=PlayerRecord("BLACK", "human", "Black"),
            white_player=PlayerRecord("WHITE", "minimax", "White AI"),
            result="BLACK_WIN",
            total_moves=50,
            duration=120.5,
            moves=[
                MoveRecord(1, 7, 7, "BLACK", 0.5),
                MoveRecord(2, 8, 8, "WHITE", 1.2),
            ],
        )
        d = record.to_dict()
        assert d["game_id"] == "20260129_120000_000000"
        assert d["rule_id"] == "StandardGomokuRule"
        assert d["black_player"]["stone"] == "BLACK"
        assert len(d["moves"]) == 2

    def test_to_json(self):
        """JSON変換が正しく動作する"""
        record = GameRecord(
            game_id="20260129_120000_000000",
            timestamp="2026-01-29T12:00:00",
            rule_id="StandardGomokuRule",
            board_width=15,
            board_height=15,
            win_condition=5,
            black_player=PlayerRecord("BLACK", "human", "Black"),
            white_player=PlayerRecord("WHITE", "minimax", "White AI"),
            result="BLACK_WIN",
            total_moves=50,
            duration=120.5,
        )
        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed["game_id"] == "20260129_120000_000000"

    def test_from_dict(self):
        """辞書からの復元が正しく動作する"""
        data = {
            "game_id": "20260129_120000_000000",
            "timestamp": "2026-01-29T12:00:00",
            "rule_id": "StandardGomokuRule",
            "board_width": 15,
            "board_height": 15,
            "win_condition": 5,
            "black_player": {"stone": "BLACK", "player_type": "human", "name": "Black"},
            "white_player": {"stone": "WHITE", "player_type": "minimax", "name": "White AI"},
            "result": "BLACK_WIN",
            "total_moves": 50,
            "duration": 120.5,
            "moves": [
                {"move_number": 1, "x": 7, "y": 7, "stone": "BLACK", "timestamp": 0.5},
            ],
            "version": "1.0",
            "platform": "Variant Go Platform",
        }
        record = GameRecord.from_dict(data)
        assert record.game_id == "20260129_120000_000000"
        assert record.black_player.stone == "BLACK"
        assert len(record.moves) == 1

    def test_from_json(self):
        """JSONからの復元が正しく動作する"""
        json_str = json.dumps({
            "game_id": "20260129_120000_000000",
            "timestamp": "2026-01-29T12:00:00",
            "rule_id": "StandardGomokuRule",
            "board_width": 15,
            "board_height": 15,
            "win_condition": 5,
            "black_player": {"stone": "BLACK", "player_type": "human", "name": "Black"},
            "white_player": {"stone": "WHITE", "player_type": "minimax", "name": "White AI"},
            "result": "BLACK_WIN",
            "total_moves": 50,
            "duration": 120.5,
            "moves": [],
            "version": "1.0",
            "platform": "Variant Go Platform",
        })
        record = GameRecord.from_json(json_str)
        assert record.game_id == "20260129_120000_000000"

    def test_roundtrip(self):
        """to_json -> from_json のラウンドトリップが正しく動作する"""
        original = GameRecord(
            game_id="20260129_120000_000000",
            timestamp="2026-01-29T12:00:00",
            rule_id="StandardGomokuRule",
            board_width=15,
            board_height=15,
            win_condition=5,
            black_player=PlayerRecord("BLACK", "human", "Black"),
            white_player=PlayerRecord("WHITE", "minimax", "White AI", ai_depth=3),
            result="BLACK_WIN",
            total_moves=2,
            duration=120.5,
            moves=[
                MoveRecord(1, 7, 7, "BLACK", 0.5, thinking_time=0.3),
                MoveRecord(2, 8, 8, "WHITE", 1.2, thinking_time=0.7, eval_score=0.5),
            ],
        )
        json_str = original.to_json()
        restored = GameRecord.from_json(json_str)

        assert restored.game_id == original.game_id
        assert restored.black_player.name == original.black_player.name
        assert restored.white_player.ai_depth == original.white_player.ai_depth
        assert len(restored.moves) == len(original.moves)
        assert restored.moves[1].eval_score == original.moves[1].eval_score


class TestGameRecorder:
    """GameRecorderのテスト"""

    def test_disabled_recorder_does_nothing(self):
        """無効化されたレコーダーは何もしない"""
        recorder = GameRecorder(enabled=False)

        rule = Mock()
        rule.rule_id = "StandardGomokuRule"
        rule.board_width = 15
        rule.board_height = 15
        rule.win_condition = 5

        player = Mock()
        player.name = "Test"

        recorder.start_game(rule, player, player)
        recorder.record_move(Position(7, 7), Stone.BLACK)
        recorder.end_game(GameStatus.BLACK_WIN)

        assert recorder.save() is None
        assert recorder.get_current_record() is None

    def test_record_game_json(self):
        """JSON形式での棋譜記録"""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = GameRecorder(output_dir=tmpdir, format="json", enabled=True)

            rule = Mock()
            rule.rule_id = "StandardGomokuRule"
            rule.board_width = 15
            rule.board_height = 15
            rule.win_condition = 5

            black_player = Mock()
            black_player.name = "Black Player"

            white_player = Mock()
            white_player.name = "White Player"

            recorder.start_game(rule, black_player, white_player)
            recorder.record_move(Position(7, 7), Stone.BLACK)
            recorder.record_move(Position(8, 8), Stone.WHITE)
            recorder.end_game(GameStatus.BLACK_WIN)

            filepath = recorder.save()

            assert filepath is not None
            assert os.path.exists(filepath)

            # ファイルを読み込んで検証
            loaded = load_game_record(filepath)
            assert loaded.rule_id == "StandardGomokuRule"
            assert loaded.result == "BLACK_WIN"
            assert len(loaded.moves) == 2
            assert loaded.moves[0].x == 7
            assert loaded.moves[0].y == 7

    def test_record_game_jsonl(self):
        """JSONL形式での棋譜記録"""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = GameRecorder(output_dir=tmpdir, format="jsonl", enabled=True)

            rule = Mock()
            rule.rule_id = "GravityGomokuRule"
            rule.board_width = 7
            rule.board_height = 6
            rule.win_condition = 4

            player = Mock()
            player.name = "Test Player"

            # 1対局目
            recorder.start_game(rule, player, player)
            recorder.record_move(Position(3, 0), Stone.BLACK)
            recorder.end_game(GameStatus.ONGOING)
            filepath1 = recorder.save()

            # 2対局目
            recorder.start_game(rule, player, player)
            recorder.record_move(Position(4, 0), Stone.BLACK)
            recorder.end_game(GameStatus.BLACK_WIN)
            filepath2 = recorder.save()

            # 同じファイルに追記されているはず
            assert filepath1 == filepath2

            # 複数レコードを読み込んで検証
            records = load_game_records_jsonl(filepath1)
            assert len(records) == 2
            assert records[0].moves[0].x == 3
            assert records[1].moves[0].x == 4

    def test_creates_output_directory(self):
        """出力ディレクトリが自動作成される"""
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "subdir", "logs")
            recorder = GameRecorder(output_dir=new_dir, enabled=True)

            assert os.path.exists(new_dir)


class TestSimpleFormats:
    """シンプル形式のテスト"""

    def test_to_simple_text(self):
        """テキスト形式の変換"""
        record = GameRecord(
            game_id="test",
            timestamp="2026-01-29T12:00:00",
            rule_id="StandardGomokuRule",
            board_width=15,
            board_height=15,
            win_condition=5,
            black_player=PlayerRecord("BLACK", "minimax", "Minimax AI"),
            white_player=PlayerRecord("WHITE", "mcts", "MCTS AI"),
            result="BLACK_WIN",
            total_moves=3,
            duration=10.0,
            moves=[
                MoveRecord(1, 7, 7, "BLACK", 0.5),   # h8
                MoveRecord(2, 8, 8, "WHITE", 1.0),   # i9
                MoveRecord(3, 6, 6, "BLACK", 1.5),   # g7
            ],
        )
        text = record.to_simple_text()

        assert "# Variant Go: StandardGomokuRule 15x15" in text
        assert "# Result: BLACK_WIN (3 moves)" in text
        assert "1. h8" in text
        assert "2. i9" in text
        assert "3. g7" in text

    def test_to_csv(self):
        """CSV形式の変換"""
        record = GameRecord(
            game_id="test",
            timestamp="2026-01-29T12:00:00",
            rule_id="StandardGomokuRule",
            board_width=15,
            board_height=15,
            win_condition=5,
            black_player=PlayerRecord("BLACK", "human", "Black"),
            white_player=PlayerRecord("WHITE", "human", "White"),
            result="BLACK_WIN",
            total_moves=2,
            duration=5.0,
            moves=[
                MoveRecord(1, 7, 7, "BLACK", 0.5),
                MoveRecord(2, 8, 8, "WHITE", 1.0),
            ],
        )
        csv = record.to_csv()
        lines = csv.split("\n")

        assert lines[0] == "move,x,y,col,row,stone"
        assert lines[1] == "1,7,7,h,8,BLACK"
        assert lines[2] == "2,8,8,i,9,WHITE"

    def test_to_sequence(self):
        """シーケンス形式の変換"""
        record = GameRecord(
            game_id="test",
            timestamp="2026-01-29T12:00:00",
            rule_id="StandardGomokuRule",
            board_width=15,
            board_height=15,
            win_condition=5,
            black_player=PlayerRecord("BLACK", "human", "Black"),
            white_player=PlayerRecord("WHITE", "human", "White"),
            result="WHITE_WIN",
            total_moves=4,
            duration=5.0,
            moves=[
                MoveRecord(1, 7, 7, "BLACK", 0.5),   # h8
                MoveRecord(2, 0, 0, "WHITE", 1.0),   # a1
                MoveRecord(3, 14, 14, "BLACK", 1.5), # o15
                MoveRecord(4, 2, 11, "WHITE", 2.0),  # c12
            ],
        )
        seq = record.to_sequence()

        assert seq == "[WHITE_WIN] h8 a1 o15 c12"

    def test_pos_to_notation(self):
        """座標から棋譜表記への変換"""
        # 静的メソッドを直接テスト
        assert GameRecord._pos_to_notation(0, 0) == "a1"
        assert GameRecord._pos_to_notation(7, 7) == "h8"
        assert GameRecord._pos_to_notation(14, 14) == "o15"
        assert GameRecord._pos_to_notation(2, 11) == "c12"


class TestLoadFunctions:
    """棋譜読み込み関数のテスト"""

    def test_load_game_record(self):
        """単一棋譜の読み込み"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as f:
            record = GameRecord(
                game_id="test",
                timestamp="2026-01-29T12:00:00",
                rule_id="StandardGomokuRule",
                board_width=15,
                board_height=15,
                win_condition=5,
                black_player=PlayerRecord("BLACK", "human", "Black"),
                white_player=PlayerRecord("WHITE", "minimax", "White"),
                result="DRAW",
                total_moves=225,
                duration=300.0,
            )
            f.write(record.to_json())
            filepath = f.name

        try:
            loaded = load_game_record(filepath)
            assert loaded.game_id == "test"
            assert loaded.result == "DRAW"
        finally:
            os.unlink(filepath)

    def test_load_game_records_jsonl(self):
        """JSONL形式の複数棋譜の読み込み"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            for i in range(3):
                record = GameRecord(
                    game_id=f"test_{i}",
                    timestamp="2026-01-29T12:00:00",
                    rule_id="StandardGomokuRule",
                    board_width=15,
                    board_height=15,
                    win_condition=5,
                    black_player=PlayerRecord("BLACK", "human", "Black"),
                    white_player=PlayerRecord("WHITE", "minimax", "White"),
                    result="BLACK_WIN",
                    total_moves=50,
                    duration=120.0,
                )
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            filepath = f.name

        try:
            records = load_game_records_jsonl(filepath)
            assert len(records) == 3
            assert records[0].game_id == "test_0"
            assert records[2].game_id == "test_2"
        finally:
            os.unlink(filepath)

    def test_load_game_records_jsonl_empty_lines(self):
        """空行を含むJSONLの読み込み"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False, encoding='utf-8') as f:
            record = GameRecord(
                game_id="test",
                timestamp="2026-01-29T12:00:00",
                rule_id="StandardGomokuRule",
                board_width=15,
                board_height=15,
                win_condition=5,
                black_player=PlayerRecord("BLACK", "human", "Black"),
                white_player=PlayerRecord("WHITE", "minimax", "White"),
                result="BLACK_WIN",
                total_moves=50,
                duration=120.0,
            )
            f.write(json.dumps(record.to_dict()) + "\n")
            f.write("\n")  # 空行
            f.write(json.dumps(record.to_dict()) + "\n")
            filepath = f.name

        try:
            records = load_game_records_jsonl(filepath)
            assert len(records) == 2
        finally:
            os.unlink(filepath)
