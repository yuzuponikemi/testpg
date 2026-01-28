"""
Variant Go Platform - Core Game Logic Tests

このテストモジュールは、game_core.py の機能を検証します。

テストシナリオ:
1. 黒が横一列に5つ並んで勝利するケース
2. すでに石がある場所に置こうとしてエラーが返るケース
3. Observerの動作確認（コールバックが正しく呼ばれるか）
"""

import pytest
from unittest.mock import Mock, call

from game_core import (
    Stone,
    GameStatus,
    Position,
    GameEvent,
    Board,
    GameRule,
    StandardGomokuRule,
    GravityGomokuRule,
    RuleRegistry,
    GameEngine,
)


class TestStone:
    """Stone列挙型のテスト"""

    def test_opponent_of_black_is_white(self):
        """黒の相手は白"""
        assert Stone.BLACK.opponent() == Stone.WHITE

    def test_opponent_of_white_is_black(self):
        """白の相手は黒"""
        assert Stone.WHITE.opponent() == Stone.BLACK

    def test_opponent_of_empty_is_empty(self):
        """空の相手は空"""
        assert Stone.EMPTY.opponent() == Stone.EMPTY


class TestPosition:
    """Position座標クラスのテスト"""

    def test_equality(self):
        """同じ座標は等しい"""
        pos1 = Position(3, 4)
        pos2 = Position(3, 4)
        assert pos1 == pos2

    def test_inequality(self):
        """異なる座標は等しくない"""
        pos1 = Position(3, 4)
        pos2 = Position(4, 3)
        assert pos1 != pos2

    def test_hashable(self):
        """Positionはハッシュ可能（setやdictのキーに使える）"""
        pos1 = Position(3, 4)
        pos2 = Position(3, 4)
        pos_set = {pos1}
        assert pos2 in pos_set


class TestBoard:
    """Boardクラスのテスト"""

    def test_initial_board_is_empty(self):
        """初期盤面は全て空"""
        board = Board(15, 15)
        for y in range(15):
            for x in range(15):
                assert board.get_stone(x, y) == Stone.EMPTY

    def test_set_and_get_stone(self):
        """石を置いて取得できる"""
        board = Board(15, 15)
        board.set_stone(7, 7, Stone.BLACK)
        assert board.get_stone(7, 7) == Stone.BLACK

    def test_is_within_bounds_valid(self):
        """有効な座標は範囲内"""
        board = Board(15, 15)
        assert board.is_within_bounds(0, 0) is True
        assert board.is_within_bounds(14, 14) is True
        assert board.is_within_bounds(7, 7) is True

    def test_is_within_bounds_invalid(self):
        """無効な座標は範囲外"""
        board = Board(15, 15)
        assert board.is_within_bounds(-1, 0) is False
        assert board.is_within_bounds(0, -1) is False
        assert board.is_within_bounds(15, 0) is False
        assert board.is_within_bounds(0, 15) is False

    def test_is_empty(self):
        """空のセルはis_empty=True"""
        board = Board(15, 15)
        assert board.is_empty(7, 7) is True
        board.set_stone(7, 7, Stone.BLACK)
        assert board.is_empty(7, 7) is False

    def test_move_count(self):
        """手数が正しくカウントされる"""
        board = Board(15, 15)
        assert board.move_count == 0
        board.set_stone(7, 7, Stone.BLACK)
        assert board.move_count == 1
        board.set_stone(7, 8, Stone.WHITE)
        assert board.move_count == 2

    def test_board_copy_is_independent(self):
        """コピーは元の盤面と独立している"""
        board = Board(15, 15)
        board.set_stone(7, 7, Stone.BLACK)

        copied = board.copy()
        copied.set_stone(8, 8, Stone.WHITE)

        assert board.get_stone(8, 8) == Stone.EMPTY
        assert copied.get_stone(8, 8) == Stone.WHITE

    def test_clear_board(self):
        """盤面クリアで初期状態に戻る"""
        board = Board(15, 15)
        board.set_stone(7, 7, Stone.BLACK)
        board.set_stone(8, 8, Stone.WHITE)

        board.clear()

        assert board.get_stone(7, 7) == Stone.EMPTY
        assert board.get_stone(8, 8) == Stone.EMPTY
        assert board.move_count == 0


class TestStandardGomokuRule:
    """StandardGomokuRuleクラスのテスト"""

    def test_board_size(self):
        """盤面サイズは15x15"""
        rule = StandardGomokuRule()
        assert rule.board_width == 15
        assert rule.board_height == 15

    def test_win_condition(self):
        """勝利条件は5連"""
        rule = StandardGomokuRule()
        assert rule.win_condition == 5

    def test_valid_move_on_empty_cell(self):
        """空のセルへの着手は合法"""
        rule = StandardGomokuRule()
        board = rule.create_board()
        assert rule.is_valid_move(board, 7, 7, Stone.BLACK) is True

    def test_invalid_move_on_occupied_cell(self):
        """石がある場所への着手は不正"""
        rule = StandardGomokuRule()
        board = rule.create_board()
        board.set_stone(7, 7, Stone.BLACK)
        assert rule.is_valid_move(board, 7, 7, Stone.WHITE) is False

    def test_invalid_move_out_of_bounds(self):
        """盤面外への着手は不正"""
        rule = StandardGomokuRule()
        board = rule.create_board()
        assert rule.is_valid_move(board, -1, 0, Stone.BLACK) is False
        assert rule.is_valid_move(board, 15, 0, Stone.BLACK) is False


class TestGameEngineBasic:
    """GameEngineの基本機能テスト"""

    def test_initial_state(self):
        """初期状態は黒の手番、進行中"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        assert engine.current_turn == Stone.BLACK
        assert engine.status == GameStatus.ONGOING
        assert engine.is_game_over is False

    def test_turn_alternates(self):
        """手番が交互に入れ替わる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        assert engine.current_turn == Stone.BLACK
        engine.play_move(7, 7)
        assert engine.current_turn == Stone.WHITE
        engine.play_move(8, 8)
        assert engine.current_turn == Stone.BLACK

    def test_stone_placed_correctly(self):
        """石が正しく置かれる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(7, 7)
        assert engine.get_stone_at(7, 7) == Stone.BLACK

        engine.play_move(8, 8)
        assert engine.get_stone_at(8, 8) == Stone.WHITE


class TestGameEngineOccupiedCell:
    """既に石がある場所への着手テスト（要件2）"""

    def test_cannot_play_on_occupied_cell(self):
        """すでに石がある場所に置こうとするとFalseが返る"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        # 黒が(7,7)に置く - 成功
        result1 = engine.play_move(7, 7)
        assert result1 is True

        # 白が同じ場所(7,7)に置こうとする - 失敗
        result2 = engine.play_move(7, 7)
        assert result2 is False

        # 手番は変わっていない（白のまま）
        assert engine.current_turn == Stone.WHITE

        # 石は黒のまま
        assert engine.get_stone_at(7, 7) == Stone.BLACK

    def test_cannot_play_after_game_over(self):
        """ゲーム終了後は石を置けない"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        # 黒の5連を作る（横一列）
        moves = [
            (0, 0, Stone.BLACK), (0, 1, Stone.WHITE),
            (1, 0, Stone.BLACK), (1, 1, Stone.WHITE),
            (2, 0, Stone.BLACK), (2, 1, Stone.WHITE),
            (3, 0, Stone.BLACK), (3, 1, Stone.WHITE),
            (4, 0, Stone.BLACK),  # 黒の勝ち
        ]

        for x, y, _ in moves:
            engine.play_move(x, y)

        assert engine.is_game_over is True

        # ゲーム終了後に置こうとしても失敗
        result = engine.play_move(5, 5)
        assert result is False


class TestGameEngineWinCondition:
    """勝利条件のテスト（要件1: 黒が横一列に5つ並んで勝利）"""

    def test_black_wins_horizontal(self):
        """黒が横一列に5つ並んで勝利する"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        # 黒: (0,0), (1,0), (2,0), (3,0), (4,0) - 横一列
        # 白: (0,1), (1,1), (2,1), (3,1) - 別の列（4手）
        moves = [
            (0, 0),  # 黒
            (0, 1),  # 白
            (1, 0),  # 黒
            (1, 1),  # 白
            (2, 0),  # 黒
            (2, 1),  # 白
            (3, 0),  # 黒
            (3, 1),  # 白
            (4, 0),  # 黒 - 勝利！
        ]

        for x, y in moves:
            engine.play_move(x, y)

        assert engine.status == GameStatus.BLACK_WIN
        assert engine.is_game_over is True

    def test_black_wins_vertical(self):
        """黒が縦一列に5つ並んで勝利する"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        moves = [
            (0, 0),  # 黒
            (1, 0),  # 白
            (0, 1),  # 黒
            (1, 1),  # 白
            (0, 2),  # 黒
            (1, 2),  # 白
            (0, 3),  # 黒
            (1, 3),  # 白
            (0, 4),  # 黒 - 勝利！
        ]

        for x, y in moves:
            engine.play_move(x, y)

        assert engine.status == GameStatus.BLACK_WIN

    def test_black_wins_diagonal(self):
        """黒が斜めに5つ並んで勝利する"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        moves = [
            (0, 0),  # 黒
            (1, 0),  # 白
            (1, 1),  # 黒
            (2, 0),  # 白
            (2, 2),  # 黒
            (3, 0),  # 白
            (3, 3),  # 黒
            (4, 0),  # 白
            (4, 4),  # 黒 - 勝利！
        ]

        for x, y in moves:
            engine.play_move(x, y)

        assert engine.status == GameStatus.BLACK_WIN

    def test_white_wins(self):
        """白が5つ並んで勝利する"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        moves = [
            (0, 0),  # 黒
            (5, 0),  # 白
            (0, 1),  # 黒
            (5, 1),  # 白
            (0, 2),  # 黒
            (5, 2),  # 白
            (0, 3),  # 黒
            (5, 3),  # 白
            (1, 4),  # 黒（別の場所）
            (5, 4),  # 白 - 勝利！
        ]

        for x, y in moves:
            engine.play_move(x, y)

        assert engine.status == GameStatus.WHITE_WIN

    def test_more_than_five_still_wins(self):
        """6つ以上並んでも勝ち（標準ルールでは長連も勝ち）"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        # 6連を作る（白は連続しないようにバラバラに配置）
        moves = [
            (0, 0), (14, 0),   # 黒(0,0), 白(14,0)
            (1, 0), (14, 2),   # 黒(1,0), 白(14,2)
            (2, 0), (14, 4),   # 黒(2,0), 白(14,4)
            (3, 0), (14, 6),   # 黒(3,0), 白(14,6)
            (5, 0), (14, 8),   # 黒(5,0)で飛ばす, 白(14,8)
            (4, 0),            # 黒が(4,0)に置いて6連完成
        ]

        for x, y in moves:
            engine.play_move(x, y)

        assert engine.status == GameStatus.BLACK_WIN


class TestGameEngineObserver:
    """Observerパターンのテスト（要件3）"""

    def test_listener_called_on_move(self):
        """石を置いた時にリスナーが呼ばれる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        # モックのコールバックを作成
        mock_callback = Mock()
        engine.add_listener(mock_callback)

        # 石を置く
        engine.play_move(7, 7)

        # コールバックが呼ばれたことを確認
        assert mock_callback.called is True
        assert mock_callback.call_count == 1

        # 呼ばれた引数を確認
        event: GameEvent = mock_callback.call_args[0][0]
        assert event.event_type == "MOVE_PLAYED"
        assert event.position == Position(7, 7)
        assert event.stone == Stone.BLACK

    def test_listener_called_on_game_over(self):
        """ゲーム終了時にリスナーが呼ばれる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        mock_callback = Mock()
        engine.add_listener(mock_callback)

        # 黒の5連を作る
        moves = [
            (0, 0), (0, 1),
            (1, 0), (1, 1),
            (2, 0), (2, 1),
            (3, 0), (3, 1),
            (4, 0),  # 黒の勝ち
        ]

        for x, y in moves:
            engine.play_move(x, y)

        # 最後の手で MOVE_PLAYED と GAME_OVER の2回呼ばれる
        # 全体では 9回の MOVE_PLAYED + 1回の GAME_OVER = 10回
        assert mock_callback.call_count == 10

        # 最後のイベントを確認
        last_event: GameEvent = mock_callback.call_args[0][0]
        assert last_event.event_type == "GAME_OVER"
        assert last_event.status == GameStatus.BLACK_WIN

    def test_multiple_listeners(self):
        """複数のリスナーが全て呼ばれる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        mock1 = Mock()
        mock2 = Mock()
        mock3 = Mock()

        engine.add_listener(mock1)
        engine.add_listener(mock2)
        engine.add_listener(mock3)

        engine.play_move(7, 7)

        assert mock1.call_count == 1
        assert mock2.call_count == 1
        assert mock3.call_count == 1

    def test_remove_listener(self):
        """リスナーを解除できる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        mock_callback = Mock()
        engine.add_listener(mock_callback)

        engine.play_move(7, 7)
        assert mock_callback.call_count == 1

        # リスナーを解除
        result = engine.remove_listener(mock_callback)
        assert result is True

        engine.play_move(8, 8)
        # 解除後は呼ばれない
        assert mock_callback.call_count == 1

    def test_remove_nonexistent_listener(self):
        """存在しないリスナーの解除はFalseを返す"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        mock_callback = Mock()
        result = engine.remove_listener(mock_callback)
        assert result is False

    def test_listener_receives_correct_sequence(self):
        """リスナーが正しい順序でイベントを受け取る"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        events: list[GameEvent] = []

        def recorder(event: GameEvent):
            events.append(event)

        engine.add_listener(recorder)

        engine.play_move(7, 7)  # 黒
        engine.play_move(8, 8)  # 白

        assert len(events) == 2
        assert events[0].stone == Stone.BLACK
        assert events[0].position == Position(7, 7)
        assert events[1].stone == Stone.WHITE
        assert events[1].position == Position(8, 8)

    def test_listener_error_does_not_stop_game(self):
        """リスナーがエラーを起こしてもゲームは継続する"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        def failing_listener(event: GameEvent):
            raise Exception("Listener error!")

        good_listener = Mock()

        engine.add_listener(failing_listener)
        engine.add_listener(good_listener)

        # エラーが起きても石は置ける
        result = engine.play_move(7, 7)
        assert result is True
        assert engine.get_stone_at(7, 7) == Stone.BLACK

        # 他のリスナーは正常に呼ばれる
        assert good_listener.called is True


class TestGameEngineReset:
    """ゲームリセット機能のテスト"""

    def test_reset_clears_board(self):
        """リセットで盤面がクリアされる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(7, 7)
        engine.play_move(8, 8)

        engine.reset()

        assert engine.get_stone_at(7, 7) == Stone.EMPTY
        assert engine.get_stone_at(8, 8) == Stone.EMPTY

    def test_reset_restores_initial_state(self):
        """リセットで初期状態に戻る"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(7, 7)
        engine.reset()

        assert engine.current_turn == Stone.BLACK
        assert engine.status == GameStatus.ONGOING
        assert engine.is_game_over is False

    def test_reset_notifies_listeners(self):
        """リセット時にリスナーが呼ばれる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        mock_callback = Mock()
        engine.add_listener(mock_callback)

        engine.reset()

        event: GameEvent = mock_callback.call_args[0][0]
        assert event.event_type == "GAME_RESET"


class TestGetValidMoves:
    """合法手取得のテスト（AI実装準備）"""

    def test_initial_valid_moves(self):
        """初期盤面では全てのセルが合法手"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        valid_moves = engine.get_valid_moves()
        assert len(valid_moves) == 15 * 15

    def test_valid_moves_decrease_after_play(self):
        """石を置くと合法手が減る"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(7, 7)
        valid_moves = engine.get_valid_moves()

        assert len(valid_moves) == 15 * 15 - 1
        assert Position(7, 7) not in valid_moves


class TestMoveHistory:
    """着手履歴のテスト（Undo機能準備）"""

    def test_move_history_recorded(self):
        """着手履歴が記録される"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(7, 7)
        engine.play_move(8, 8)

        history = engine.move_history
        assert len(history) == 2
        assert history[0] == (Position(7, 7), Stone.BLACK)
        assert history[1] == (Position(8, 8), Stone.WHITE)

    def test_history_cleared_on_reset(self):
        """リセットで履歴がクリアされる"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(7, 7)
        engine.reset()

        assert len(engine.move_history) == 0


class TestGravityGomokuRule:
    """GravityGomokuRuleのテスト"""

    def test_default_board_size(self):
        """デフォルト盤面サイズは7x6"""
        rule = GravityGomokuRule()
        assert rule.board_width == 7
        assert rule.board_height == 6

    def test_custom_board_size(self):
        """カスタム盤面サイズを設定できる"""
        rule = GravityGomokuRule(width=10, height=8, win_condition=5)
        assert rule.board_width == 10
        assert rule.board_height == 8
        assert rule.win_condition == 5

    def test_default_win_condition(self):
        """デフォルト勝利条件は4連"""
        rule = GravityGomokuRule()
        assert rule.win_condition == 4

    def test_rule_name(self):
        """ルール名が正しい"""
        rule = GravityGomokuRule()
        assert "Gravity" in rule.rule_name
        assert "7x6" in rule.rule_name
        assert "4-in-a-row" in rule.rule_name

    def test_is_valid_move_empty_column(self):
        """空の列への着手は合法"""
        rule = GravityGomokuRule()
        board = rule.create_board()
        assert rule.is_valid_move(board, 0, 0, Stone.BLACK) is True
        assert rule.is_valid_move(board, 3, 0, Stone.BLACK) is True

    def test_is_valid_move_full_column(self):
        """満杯の列への着手は不正"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # 列0を満杯にする
        for y in range(board.height):
            board.set_stone(0, y, Stone.BLACK if y % 2 == 0 else Stone.WHITE)

        assert rule.is_valid_move(board, 0, 0, Stone.BLACK) is False

    def test_is_valid_move_out_of_bounds(self):
        """盤面外への着手は不正"""
        rule = GravityGomokuRule()
        board = rule.create_board()
        assert rule.is_valid_move(board, -1, 0, Stone.BLACK) is False
        assert rule.is_valid_move(board, 7, 0, Stone.BLACK) is False

    def test_apply_move_effects_stone_falls(self):
        """石が下に落ちる"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # y=0に石を置く（上の方）
        board.set_stone(3, 0, Stone.BLACK)

        # 副作用適用
        affected = rule.apply_move_effects(board, 3, 0, Stone.BLACK)

        # 石は一番下（y=5）に移動
        assert len(affected) == 1
        assert affected[0] == Position(3, 5)
        assert board.get_stone(3, 0) == Stone.EMPTY
        assert board.get_stone(3, 5) == Stone.BLACK

    def test_apply_move_effects_stacks(self):
        """石が既存の石の上に積み重なる"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # まず1つ目の石を一番下に置く
        board.set_stone(3, 5, Stone.BLACK)

        # 2つ目の石を上に置く
        board.set_stone(3, 0, Stone.WHITE)
        affected = rule.apply_move_effects(board, 3, 0, Stone.WHITE)

        # 白はy=4に落ちる
        assert len(affected) == 1
        assert affected[0] == Position(3, 4)
        assert board.get_stone(3, 4) == Stone.WHITE
        assert board.get_stone(3, 5) == Stone.BLACK

    def test_apply_move_effects_no_movement(self):
        """移動がない場合は空リスト"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # 一番下に置く
        board.set_stone(3, 5, Stone.BLACK)
        affected = rule.apply_move_effects(board, 3, 5, Stone.BLACK)

        assert len(affected) == 0
        assert board.get_stone(3, 5) == Stone.BLACK

    def test_check_winner_horizontal(self):
        """横4連で勝利"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # 一番下に横4連
        for x in range(4):
            board.set_stone(x, 5, Stone.BLACK)

        status = rule.check_winner(board, 3, 5, Stone.BLACK)
        assert status == GameStatus.BLACK_WIN

    def test_check_winner_vertical(self):
        """縦4連で勝利"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # 縦4連
        for y in range(4):
            board.set_stone(0, 5 - y, Stone.BLACK)

        status = rule.check_winner(board, 0, 2, Stone.BLACK)
        assert status == GameStatus.BLACK_WIN

    def test_check_winner_diagonal(self):
        """斜め4連で勝利"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # 斜め4連
        for i in range(4):
            board.set_stone(i, 5 - i, Stone.BLACK)

        status = rule.check_winner(board, 3, 2, Stone.BLACK)
        assert status == GameStatus.BLACK_WIN

    def test_check_winner_draw(self):
        """盤面が埋まったら引き分け"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # 4連ができないパターンで盤面を埋める
        # 2列ごとに色を反転させる（黒黒白白黒黒白...）
        for y in range(board.height):
            for x in range(board.width):
                # 列グループ（0-1=黒、2-3=白、4-5=黒、6=白）で色を変える
                # かつ、行ごとに反転させる
                col_group = x // 2
                base_stone = Stone.BLACK if col_group % 2 == 0 else Stone.WHITE
                if y % 2 == 1:
                    base_stone = base_stone.opponent()
                board.set_stone(x, y, base_stone)

        # 最後に置いた石の位置で判定
        status = rule.check_winner(board, 6, 5, Stone.WHITE)
        assert status == GameStatus.DRAW

    def test_get_valid_moves_returns_columns(self):
        """get_valid_movesは有効な列を返す"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        moves = rule.get_valid_moves(board, Stone.BLACK)

        # 7列すべてが有効
        assert len(moves) == 7
        # 各列のy=0として返される
        for move in moves:
            assert move.y == 0

    def test_get_rule_config(self):
        """設定をシリアライズできる"""
        rule = GravityGomokuRule(width=10, height=8, win_condition=5)
        config = rule.get_rule_config()

        assert config["rule_id"] == "GravityGomokuRule"
        assert config["width"] == 10
        assert config["height"] == 8
        assert config["win_condition"] == 5

    def test_from_config(self):
        """設定からルールを復元できる"""
        config = {
            "rule_id": "GravityGomokuRule",
            "width": 10,
            "height": 8,
            "win_condition": 5,
        }
        rule = GravityGomokuRule.from_config(config)

        assert rule.board_width == 10
        assert rule.board_height == 8
        assert rule.win_condition == 5


class TestGravityGomokuWithEngine:
    """GravityGomokuRuleとGameEngineの統合テスト"""

    def test_stone_falls_via_engine(self):
        """GameEngine経由で石が落ちる"""
        rule = GravityGomokuRule()
        engine = GameEngine(rule)

        # 列3の上から置く（y=0）
        engine.play_move(3, 0)

        # 石は一番下（y=5）に落ちる
        assert engine.get_stone_at(3, 0) == Stone.EMPTY
        assert engine.get_stone_at(3, 5) == Stone.BLACK

    def test_stone_moved_event_fired(self):
        """STONE_MOVEDイベントが発火する"""
        rule = GravityGomokuRule()
        engine = GameEngine(rule)

        events = []

        def recorder(event):
            events.append(event)

        engine.add_listener(recorder)

        engine.play_move(3, 0)

        # MOVE_PLAYED と STONE_MOVED の2つのイベント
        assert len(events) == 2
        assert events[0].event_type == "MOVE_PLAYED"
        assert events[0].position == Position(3, 0)
        assert events[1].event_type == "STONE_MOVED"
        assert events[1].position == Position(3, 5)

    def test_history_records_final_position(self):
        """履歴には最終位置が記録される"""
        rule = GravityGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(3, 0)

        history = engine.move_history
        assert len(history) == 1
        assert history[0] == (Position(3, 5), Stone.BLACK)

    def test_win_detected_at_final_position(self):
        """最終位置で勝敗判定される"""
        rule = GravityGomokuRule()
        engine = GameEngine(rule)

        # 黒: 列0, 1, 2, 3を順に落とす → 横4連
        # 白: 列4, 5, 6
        engine.play_move(0, 0)  # 黒 → (0, 5)
        engine.play_move(4, 0)  # 白 → (4, 5)
        engine.play_move(1, 0)  # 黒 → (1, 5)
        engine.play_move(5, 0)  # 白 → (5, 5)
        engine.play_move(2, 0)  # 黒 → (2, 5)
        engine.play_move(6, 0)  # 白 → (6, 5)
        engine.play_move(3, 0)  # 黒 → (3, 5) 横4連完成！

        assert engine.status == GameStatus.BLACK_WIN

    def test_stacking_works(self):
        """石が正しく積み重なる"""
        rule = GravityGomokuRule()
        engine = GameEngine(rule)

        # 同じ列に3回置く
        engine.play_move(3, 0)  # 黒 → (3, 5)
        engine.play_move(3, 0)  # 白 → (3, 4)
        engine.play_move(3, 0)  # 黒 → (3, 3)

        assert engine.get_stone_at(3, 5) == Stone.BLACK
        assert engine.get_stone_at(3, 4) == Stone.WHITE
        assert engine.get_stone_at(3, 3) == Stone.BLACK

    def test_full_column_rejected(self):
        """満杯の列への着手は失敗"""
        rule = GravityGomokuRule()
        board = rule.create_board()

        # 盤面に直接石を設置して列0を満杯にする（勝敗判定を避けるため）
        # 交互に黒白を配置（縦に4連ができないようにする）
        for y in range(6):
            stone = Stone.BLACK if y % 2 == 0 else Stone.WHITE
            board.set_stone(0, y, stone)

        engine = GameEngine(rule)
        engine._board = board

        # 列0は満杯なので着手失敗
        current_turn = engine.current_turn
        result = engine.play_move(0, 0)
        assert result is False
        # 手番は変わらない
        assert engine.current_turn == current_turn


class TestRuleRegistry:
    """RuleRegistryのテスト"""

    def setup_method(self):
        """各テストの前にレジストリをクリア"""
        RuleRegistry.clear()

    def teardown_method(self):
        """各テストの後にデフォルトルールを再登録"""
        RuleRegistry.clear()
        RuleRegistry.register(StandardGomokuRule)
        RuleRegistry.register(GravityGomokuRule)

    def test_register_and_get(self):
        """ルールを登録して取得できる"""
        RuleRegistry.register(StandardGomokuRule)

        rule_class = RuleRegistry.get("StandardGomokuRule")
        assert rule_class == StandardGomokuRule

    def test_create_rule(self):
        """ルールインスタンスを作成できる"""
        RuleRegistry.register(GravityGomokuRule)

        rule = RuleRegistry.create("GravityGomokuRule", width=10, height=8)

        assert isinstance(rule, GravityGomokuRule)
        assert rule.board_width == 10
        assert rule.board_height == 8

    def test_create_from_config(self):
        """設定からルールを復元できる"""
        RuleRegistry.register(GravityGomokuRule)

        config = {
            "rule_id": "GravityGomokuRule",
            "width": 10,
            "height": 8,
            "win_condition": 5,
        }
        rule = RuleRegistry.create_from_config(config)

        assert isinstance(rule, GravityGomokuRule)
        assert rule.board_width == 10
        assert rule.win_condition == 5

    def test_list_available(self):
        """登録済みルール一覧を取得できる"""
        RuleRegistry.register(StandardGomokuRule)
        RuleRegistry.register(GravityGomokuRule)

        available = RuleRegistry.list_available()

        assert "StandardGomokuRule" in available
        assert "GravityGomokuRule" in available

    def test_is_registered(self):
        """登録状態を確認できる"""
        RuleRegistry.register(StandardGomokuRule)

        assert RuleRegistry.is_registered("StandardGomokuRule") is True
        assert RuleRegistry.is_registered("UnknownRule") is False

    def test_unregister(self):
        """ルールの登録を解除できる"""
        RuleRegistry.register(StandardGomokuRule)
        assert RuleRegistry.is_registered("StandardGomokuRule") is True

        result = RuleRegistry.unregister("StandardGomokuRule")

        assert result is True
        assert RuleRegistry.is_registered("StandardGomokuRule") is False

    def test_duplicate_registration_raises(self):
        """同じルールの重複登録はエラー"""
        RuleRegistry.register(StandardGomokuRule)

        with pytest.raises(ValueError, match="already registered"):
            RuleRegistry.register(StandardGomokuRule)

    def test_get_unregistered_raises(self):
        """未登録のルール取得はエラー"""
        with pytest.raises(KeyError, match="not registered"):
            RuleRegistry.get("UnknownRule")

    def test_create_from_config_without_rule_id_raises(self):
        """rule_idがないconfigはエラー"""
        with pytest.raises(KeyError, match="rule_id"):
            RuleRegistry.create_from_config({})

    def test_register_non_game_rule_raises(self):
        """GameRuleでないクラスの登録はエラー"""
        with pytest.raises(TypeError, match="not a subclass"):
            RuleRegistry.register(str)


class TestStandardGomokuRuleBackwardCompatibility:
    """StandardGomokuRuleの後方互換性テスト"""

    def test_apply_move_effects_returns_empty(self):
        """apply_move_effectsは空リストを返す"""
        rule = StandardGomokuRule()
        board = rule.create_board()
        board.set_stone(7, 7, Stone.BLACK)

        affected = rule.apply_move_effects(board, 7, 7, Stone.BLACK)

        assert affected == []

    def test_rule_id_is_class_name(self):
        """rule_idはクラス名"""
        rule = StandardGomokuRule()
        assert rule.rule_id == "StandardGomokuRule"

    def test_get_rule_config(self):
        """get_rule_configはrule_idを含む"""
        rule = StandardGomokuRule()
        config = rule.get_rule_config()

        assert config["rule_id"] == "StandardGomokuRule"

    def test_from_config(self):
        """from_configでインスタンス化できる"""
        config = {"rule_id": "StandardGomokuRule"}
        rule = StandardGomokuRule.from_config(config)

        assert isinstance(rule, StandardGomokuRule)
        assert rule.board_width == 15
        assert rule.board_height == 15

    def test_engine_still_works(self):
        """既存のGameEngineとの連携が維持される"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        events = []
        engine.add_listener(lambda e: events.append(e))

        engine.play_move(7, 7)

        # MOVE_PLAYEDのみ（STONE_MOVEDはなし）
        assert len(events) == 1
        assert events[0].event_type == "MOVE_PLAYED"

    def test_history_records_original_position(self):
        """履歴は元の位置を記録（移動なし）"""
        rule = StandardGomokuRule()
        engine = GameEngine(rule)

        engine.play_move(7, 7)

        history = engine.move_history
        assert history[0] == (Position(7, 7), Stone.BLACK)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
