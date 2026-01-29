"""
Variant Go Platform - Threat Search Module

VCF（Victory by Continuous Fours）とVCT（Victory by Continuous Threats）
の実装を提供します。

VCF: 四（Four）の連続で詰みを探索
VCT: 三（Three）も含めた連続脅威で詰みを探索

用語:
- Threat（脅威）: 次に5連を完成できる形
- Four（四）: あと1手で5連になる形
- Open Three（活三）: 両端が空いた3連（次に四を2箇所作れる）
- Three（三）: 片端が空いた3連（次に四を1箇所作れる）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Set, Tuple
import time

from game_core import Board, GameRule, Position, Stone, GameStatus
from ai_strategies import (
    AIStrategy, ThinkingProgress, ThinkingProgressCallback
)


class ThreatType(Enum):
    """脅威の種類（優先度順）"""
    FIVE = auto()           # 5連（勝ち）
    STRAIGHT_FOUR = auto()  # 両端開き四（達四）- 止められない
    FOUR = auto()           # 片端開き四
    OPEN_THREE = auto()     # 両端開き三（活三）
    THREE = auto()          # 片端開き三


@dataclass
class Threat:
    """脅威を表すデータクラス"""
    threat_type: ThreatType
    position: Position              # この脅威を作る/延長する手の位置
    line_positions: List[Position]  # 脅威を構成する石の位置
    defense_points: List[Position]  # 防御点（相手が打つべき場所）


@dataclass
class VCFResult:
    """VCF探索結果"""
    is_winning: bool                # 詰みが見つかったか
    win_sequence: List[Position]    # 詰み手順
    depth_searched: int             # 探索した深さ
    nodes_visited: int              # 訪問ノード数


class ThreatDetector:
    """
    盤面から脅威パターンを検出するクラス

    五目並べの脅威パターン（四、三など）を検出し、
    VCF/VCT探索に使用する情報を提供します。
    """

    # 4方向の探索ベクトル
    DIRECTIONS = [(1, 0), (0, 1), (1, 1), (1, -1)]

    def __init__(self, win_condition: int = 5):
        self._win_condition = win_condition

    def find_threats(
        self,
        board: Board,
        stone: Stone,
        threat_types: Optional[Set[ThreatType]] = None
    ) -> List[Threat]:
        """
        指定した石の色のすべての脅威を検出

        Args:
            board: 現在の盤面
            stone: 検出対象の石の色
            threat_types: 検出する脅威の種類（Noneの場合は全て）

        Returns:
            検出された脅威のリスト
        """
        if threat_types is None:
            threat_types = set(ThreatType)

        threats: List[Threat] = []
        checked_lines: Set[Tuple[int, int, int, int]] = set()

        for y in range(board.height):
            for x in range(board.width):
                for dx, dy in self.DIRECTIONS:
                    line_key = self._normalize_line_key(x, y, dx, dy)
                    if line_key in checked_lines:
                        continue
                    checked_lines.add(line_key)

                    line_threats = self._analyze_line(
                        board, x, y, dx, dy, stone, threat_types
                    )
                    threats.extend(line_threats)

        return threats

    def find_winning_moves(self, board: Board, stone: Stone) -> List[Position]:
        """
        即座に勝てる手を探す

        Args:
            board: 現在の盤面
            stone: 探索対象の石の色

        Returns:
            勝てる手のリスト
        """
        threats = self.find_threats(
            board, stone, {ThreatType.FIVE, ThreatType.STRAIGHT_FOUR, ThreatType.FOUR}
        )

        winning_moves = []
        for threat in threats:
            if threat.threat_type in {ThreatType.FIVE, ThreatType.STRAIGHT_FOUR, ThreatType.FOUR}:
                if threat.position not in winning_moves:
                    winning_moves.append(threat.position)

        return winning_moves

    def find_fours(self, board: Board, stone: Stone) -> List[Threat]:
        """四を探す"""
        return self.find_threats(
            board, stone, {ThreatType.FOUR, ThreatType.STRAIGHT_FOUR}
        )

    def find_open_threes(self, board: Board, stone: Stone) -> List[Threat]:
        """活三を探す"""
        return self.find_threats(board, stone, {ThreatType.OPEN_THREE})

    def _normalize_line_key(
        self, x: int, y: int, dx: int, dy: int
    ) -> Tuple[int, int, int, int]:
        """ラインの正規化キーを生成（重複チェック用）"""
        if dx < 0 or (dx == 0 and dy < 0):
            dx, dy = -dx, -dy
        return (x, y, dx, dy)

    def _analyze_line(
        self,
        board: Board,
        start_x: int,
        start_y: int,
        dx: int,
        dy: int,
        stone: Stone,
        threat_types: Set[ThreatType]
    ) -> List[Threat]:
        """
        1本のラインを解析して脅威を検出

        ウィンドウ方式で、win_condition + 1 のウィンドウを
        スライドさせながら脅威パターンをチェックします。
        """
        threats: List[Threat] = []
        window_size = self._win_condition + 1  # 5連なら6マスのウィンドウ

        # ラインの開始位置を調整（盤面外から始まらないように）
        x, y = start_x, start_y

        # 逆方向の端を見つける
        while board.is_within_bounds(x - dx, y - dy):
            x -= dx
            y -= dy

        # ウィンドウをスライドしながら解析
        while True:
            window_threats = self._analyze_window(
                board, x, y, dx, dy, window_size, stone, threat_types
            )
            threats.extend(window_threats)

            # 次のウィンドウへ
            x += dx
            y += dy

            # ウィンドウ終端が盤面外に出たら終了
            end_x = x + dx * (window_size - 1)
            end_y = y + dy * (window_size - 1)
            if not board.is_within_bounds(end_x, end_y):
                break

        return threats

    def _analyze_window(
        self,
        board: Board,
        x: int,
        y: int,
        dx: int,
        dy: int,
        window_size: int,
        stone: Stone,
        threat_types: Set[ThreatType]
    ) -> List[Threat]:
        """
        指定位置から始まるウィンドウ内の脅威を解析
        """
        threats: List[Threat] = []

        # ウィンドウ内の石を収集
        my_stones: List[Position] = []
        empty_positions: List[Position] = []
        opponent_stones: List[Position] = []

        for i in range(window_size):
            wx, wy = x + dx * i, y + dy * i
            if not board.is_within_bounds(wx, wy):
                return threats  # 盤面外

            cell = board.get_stone(wx, wy)
            pos = Position(wx, wy)

            if cell == stone:
                my_stones.append(pos)
            elif cell == Stone.EMPTY:
                empty_positions.append(pos)
            else:
                opponent_stones.append(pos)

        # 相手の石があるウィンドウは脅威にならない
        if opponent_stones:
            return threats

        my_count = len(my_stones)
        empty_count = len(empty_positions)

        # 5連チェック（勝ち）
        if my_count >= self._win_condition and ThreatType.FIVE in threat_types:
            threats.append(Threat(
                threat_type=ThreatType.FIVE,
                position=my_stones[0],  # 任意（既に完成）
                line_positions=my_stones[:self._win_condition],
                defense_points=[]
            ))

        # 4連チェック
        if my_count == self._win_condition - 1 and empty_count >= 1:
            # 両端が空いているか確認
            is_open_both = self._check_open_ends(
                board, x, y, dx, dy, window_size
            )

            for empty_pos in empty_positions:
                if is_open_both and ThreatType.STRAIGHT_FOUR in threat_types:
                    threats.append(Threat(
                        threat_type=ThreatType.STRAIGHT_FOUR,
                        position=empty_pos,
                        line_positions=my_stones.copy(),
                        defense_points=empty_positions.copy()
                    ))
                elif ThreatType.FOUR in threat_types:
                    threats.append(Threat(
                        threat_type=ThreatType.FOUR,
                        position=empty_pos,
                        line_positions=my_stones.copy(),
                        defense_points=[empty_pos]
                    ))

        # 3連チェック
        if my_count == self._win_condition - 2 and empty_count >= 2:
            is_open_both = self._check_open_ends(
                board, x, y, dx, dy, window_size
            )

            for empty_pos in empty_positions:
                if is_open_both and ThreatType.OPEN_THREE in threat_types:
                    threats.append(Threat(
                        threat_type=ThreatType.OPEN_THREE,
                        position=empty_pos,
                        line_positions=my_stones.copy(),
                        defense_points=empty_positions.copy()
                    ))
                elif ThreatType.THREE in threat_types:
                    threats.append(Threat(
                        threat_type=ThreatType.THREE,
                        position=empty_pos,
                        line_positions=my_stones.copy(),
                        defense_points=[empty_pos]
                    ))

        return threats

    def _check_open_ends(
        self,
        board: Board,
        x: int,
        y: int,
        dx: int,
        dy: int,
        window_size: int
    ) -> bool:
        """ウィンドウの両端が開いているかチェック"""
        # 前端チェック
        before_x, before_y = x - dx, y - dy
        before_open = (
            board.is_within_bounds(before_x, before_y) and
            board.get_stone(before_x, before_y) == Stone.EMPTY
        )

        # 後端チェック
        after_x = x + dx * window_size
        after_y = y + dy * window_size
        after_open = (
            board.is_within_bounds(after_x, after_y) and
            board.get_stone(after_x, after_y) == Stone.EMPTY
        )

        return before_open and after_open


class VCFSearch:
    """
    VCF（Victory by Continuous Fours）探索

    四の連続で詰みを探します。
    相手は四を止めるしかないため、応手が限定されます。

    アルゴリズム:
    1. 自分の四を見つける
    2. 相手の唯一の防御手を打つ
    3. 自分の新しい四を見つける
    4. 繰り返し、5連に到達したら詰み
    """

    def __init__(
        self,
        max_depth: int = 20,
        time_limit: Optional[float] = None
    ):
        """
        Args:
            max_depth: 最大探索深さ
            time_limit: 時間制限（秒）
        """
        self._max_depth = max_depth
        self._time_limit = time_limit
        self._detector = ThreatDetector()
        self._start_time: float = 0
        self._nodes_visited: int = 0

    def search(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone
    ) -> VCFResult:
        """
        VCF探索を実行

        Args:
            board: 現在の盤面
            rule: ゲームルール
            stone: 探索する石の色

        Returns:
            VCF探索結果
        """
        self._start_time = time.time()
        self._nodes_visited = 0

        sequence: List[Position] = []
        is_winning = self._vcf_recursive(
            board.copy(), rule, stone, 0, sequence
        )

        return VCFResult(
            is_winning=is_winning,
            win_sequence=sequence if is_winning else [],
            depth_searched=len(sequence),
            nodes_visited=self._nodes_visited
        )

    def _vcf_recursive(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        depth: int,
        sequence: List[Position]
    ) -> bool:
        """VCF再帰探索"""
        self._nodes_visited += 1

        # 深さ/時間制限チェック
        if depth >= self._max_depth:
            return False
        if self._is_timeout():
            return False

        # 即勝ち手をチェック
        winning_moves = self._detector.find_winning_moves(board, stone)
        if winning_moves:
            sequence.append(winning_moves[0])
            return True

        # 四を探す
        fours = self._detector.find_fours(board, stone)
        if not fours:
            return False

        # 各四に対して探索
        for four in fours:
            attack_pos = four.position

            # 攻撃手を打つ
            new_board = board.copy()
            new_board.set_stone(attack_pos.x, attack_pos.y, stone)

            # 5連になったかチェック
            status = rule.check_winner(new_board, attack_pos.x, attack_pos.y, stone)
            if status != GameStatus.ONGOING:
                if (status == GameStatus.BLACK_WIN and stone == Stone.BLACK) or \
                   (status == GameStatus.WHITE_WIN and stone == Stone.WHITE):
                    sequence.append(attack_pos)
                    return True

            # 相手の防御点を取得
            defense_points = four.defense_points
            if not defense_points:
                continue

            # 相手が防御できる場所が複数ある場合、達四ではない
            # 実際の四の場合、防御点は1つ
            if len(defense_points) > 1:
                # 達四の場合は防御不可能
                if four.threat_type == ThreatType.STRAIGHT_FOUR:
                    sequence.append(attack_pos)
                    return True
                continue

            defense_pos = defense_points[0]
            if defense_pos == attack_pos:
                # 防御点と攻撃点が同じ場合は飛ばす
                continue

            # 相手が防御
            new_board.set_stone(defense_pos.x, defense_pos.y, stone.opponent())

            # 再帰的に探索
            sub_sequence: List[Position] = []
            if self._vcf_recursive(new_board, rule, stone, depth + 1, sub_sequence):
                sequence.append(attack_pos)
                sequence.extend(sub_sequence)
                return True

        return False

    def _is_timeout(self) -> bool:
        """時間制限チェック"""
        if self._time_limit is None:
            return False
        return time.time() - self._start_time > self._time_limit


class VCTSearch:
    """
    VCT（Victory by Continuous Threats）探索

    三も含めた連続脅威で詰みを探します。
    VCFより強力ですが、探索空間が大きくなります。

    アルゴリズム:
    1. まずVCFを試す
    2. VCFがなければ、活三を打って脅威を作る
    3. 相手の応手後にVCFを試す
    """

    def __init__(
        self,
        max_depth: int = 10,
        time_limit: Optional[float] = None
    ):
        """
        Args:
            max_depth: 最大探索深さ
            time_limit: 時間制限（秒）
        """
        self._max_depth = max_depth
        self._time_limit = time_limit
        self._detector = ThreatDetector()
        self._vcf_search = VCFSearch(max_depth=max_depth, time_limit=None)
        self._start_time: float = 0
        self._nodes_visited: int = 0

    def search(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone
    ) -> VCFResult:
        """
        VCT探索を実行

        Args:
            board: 現在の盤面
            rule: ゲームルール
            stone: 探索する石の色

        Returns:
            探索結果
        """
        self._start_time = time.time()
        self._nodes_visited = 0

        # まずVCFを試す
        vcf_result = self._vcf_search.search(board, rule, stone)
        if vcf_result.is_winning:
            return vcf_result

        # VCTを試す
        sequence: List[Position] = []
        is_winning = self._vct_recursive(
            board.copy(), rule, stone, 0, sequence
        )

        return VCFResult(
            is_winning=is_winning,
            win_sequence=sequence if is_winning else [],
            depth_searched=len(sequence),
            nodes_visited=self._nodes_visited + vcf_result.nodes_visited
        )

    def _vct_recursive(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        depth: int,
        sequence: List[Position]
    ) -> bool:
        """VCT再帰探索"""
        self._nodes_visited += 1

        if depth >= self._max_depth:
            return False
        if self._is_timeout():
            return False

        # 活三を探す
        open_threes = self._detector.find_open_threes(board, stone)
        if not open_threes:
            return False

        # 各活三に対して探索
        for three in open_threes:
            attack_pos = three.position

            # 攻撃手を打つ
            new_board = board.copy()
            new_board.set_stone(attack_pos.x, attack_pos.y, stone)

            # 相手の防御点
            defense_points = three.defense_points
            if not defense_points:
                continue

            # 各防御点に対して
            all_defenses_lead_to_win = True
            for defense_pos in defense_points:
                if defense_pos == attack_pos:
                    continue

                defense_board = new_board.copy()
                defense_board.set_stone(defense_pos.x, defense_pos.y, stone.opponent())

                # 防御後にVCFが成立するか
                vcf_result = self._vcf_search.search(defense_board, rule, stone)
                if not vcf_result.is_winning:
                    # この防御でVCFが成立しない場合、この三は詰みにつながらない
                    all_defenses_lead_to_win = False
                    break

            if all_defenses_lead_to_win and defense_points:
                sequence.append(attack_pos)
                # 最初の防御に対するVCF手順を追加
                vcf_result = self._vcf_search.search(new_board, rule, stone)
                if vcf_result.is_winning:
                    sequence.extend(vcf_result.win_sequence)
                return True

        return False

    def _is_timeout(self) -> bool:
        """時間制限チェック"""
        if self._time_limit is None:
            return False
        return time.time() - self._start_time > self._time_limit


class VCFBasedAI(AIStrategy):
    """
    VCF探索ベースのAI

    まずVCF/VCTで詰みを探し、見つからなければ
    フォールバック戦略（ランダム）を使用します。

    主にMinimaxAIなどの補助として使用します。
    """

    def __init__(
        self,
        use_vct: bool = False,
        max_depth: int = 20,
        time_limit: Optional[float] = 1.0,
        fallback: Optional[AIStrategy] = None
    ):
        """
        Args:
            use_vct: VCT探索も使用するか（遅くなるが強力）
            max_depth: 最大探索深さ
            time_limit: 時間制限（秒）
            fallback: VCFで詰みがない場合のフォールバック戦略
        """
        super().__init__()
        self._use_vct = use_vct
        self._max_depth = max_depth
        self._time_limit = time_limit
        self._fallback = fallback

        self._vcf_search = VCFSearch(max_depth, time_limit)
        self._vct_search = VCTSearch(max_depth // 2, time_limit) if use_vct else None
        self._detector = ThreatDetector()

    @property
    def name(self) -> str:
        vct_str = "+VCT" if self._use_vct else ""
        return f"VCF{vct_str}"

    @property
    def difficulty(self) -> str:
        return "Hard"

    @property
    def supports_progress(self) -> bool:
        return True

    def select_move(self, board: Board, rule: GameRule, stone: Stone) -> Position:
        """VCF探索で最善手を選択"""
        start_time = time.time()

        # 合法手チェック
        valid_moves = rule.get_valid_moves(board, stone)
        if not valid_moves:
            raise ValueError("No valid moves available")

        # 即勝ち手をチェック
        winning_moves = self._detector.find_winning_moves(board, stone)
        for move in winning_moves:
            if move in valid_moves:
                return move

        # 相手の即勝ち手をチェック（防御が必要）
        opponent_winning = self._detector.find_winning_moves(board, stone.opponent())
        if opponent_winning:
            # 相手の勝ちを止める
            defense_pos = opponent_winning[0]
            if defense_pos in valid_moves:
                return defense_pos

        # VCF探索
        vcf_result = self._vcf_search.search(board, rule, stone)

        # 進捗通知
        self._report_progress(ThinkingProgress(
            ai_type="vcf",
            elapsed_time=time.time() - start_time,
            vcf_depth=vcf_result.depth_searched,
            is_forced_win=vcf_result.is_winning,
            win_sequence=vcf_result.win_sequence,
        ))

        if vcf_result.is_winning and vcf_result.win_sequence:
            return vcf_result.win_sequence[0]

        # VCT探索
        if self._vct_search:
            vct_result = self._vct_search.search(board, rule, stone)

            self._report_progress(ThinkingProgress(
                ai_type="vcf",
                elapsed_time=time.time() - start_time,
                vcf_depth=vct_result.depth_searched,
                is_forced_win=vct_result.is_winning,
                win_sequence=vct_result.win_sequence,
            ))

            if vct_result.is_winning and vct_result.win_sequence:
                return vct_result.win_sequence[0]

        # フォールバック
        if self._fallback:
            return self._fallback.select_move(board, rule, stone)

        # フォールバックがない場合は脅威を作る手を優先
        return self._select_best_move(board, rule, stone, valid_moves)

    def _select_best_move(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        valid_moves: List[Position]
    ) -> Position:
        """脅威作成を優先した手選択"""
        import random

        # 候補を既存の石の周囲に絞る
        candidates = self._get_nearby_moves(board, valid_moves)
        if not candidates:
            candidates = valid_moves

        # 各手のスコアを計算
        best_moves = []
        best_score = -1

        for move in candidates:
            score = self._evaluate_move(board, rule, stone, move)
            if score > best_score:
                best_score = score
                best_moves = [move]
            elif score == best_score:
                best_moves.append(move)

        return random.choice(best_moves)

    def _get_nearby_moves(
        self,
        board: Board,
        valid_moves: List[Position],
        radius: int = 2
    ) -> List[Position]:
        """既存の石の周囲の手を取得"""
        nearby = set()

        for y in range(board.height):
            for x in range(board.width):
                if board.get_stone(x, y) != Stone.EMPTY:
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if board.is_within_bounds(nx, ny) and board.is_empty(nx, ny):
                                nearby.add(Position(nx, ny))

        return [m for m in valid_moves if m in nearby]

    def _evaluate_move(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        move: Position
    ) -> int:
        """手のスコアを評価（脅威作成能力）"""
        score = 0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for dx, dy in directions:
            # この方向の連続数をカウント
            count = 1
            open_ends = 0

            # 正方向
            for i in range(1, 5):
                nx, ny = move.x + dx * i, move.y + dy * i
                if not board.is_within_bounds(nx, ny):
                    break
                cell = board.get_stone(nx, ny)
                if cell == stone:
                    count += 1
                elif cell == Stone.EMPTY:
                    open_ends += 1
                    break
                else:
                    break

            # 負方向
            for i in range(1, 5):
                nx, ny = move.x - dx * i, move.y - dy * i
                if not board.is_within_bounds(nx, ny):
                    break
                cell = board.get_stone(nx, ny)
                if cell == stone:
                    count += 1
                elif cell == Stone.EMPTY:
                    open_ends += 1
                    break
                else:
                    break

            # スコア加算
            if count >= 4:
                score += 1000  # 4連以上
            elif count == 3 and open_ends >= 1:
                score += 100   # 活三 or 三
            elif count == 2 and open_ends >= 1:
                score += 10    # 二

        return score
