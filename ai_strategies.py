"""
Variant Go Platform - AI Strategies Module (Milestone 2)

AI戦略の実装を提供します。

設計原則:
- AIStrategy は同期メソッド（計算のみ）
- 非同期化は AIPlayer 側の責務
- 各戦略は select_move() で最善手を返す

利用可能なAI:
- RandomAI: 合法手からランダム選択（Easy）
- MinimaxAI: Alpha-Beta探索（Medium）
- MCTSAI: モンテカルロ木探索（Hard）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
import random
import math
import time

from game_core import Board, GameRule, Position, Stone, GameStatus


# ----------------------
# 思考進捗通知（Phase 1）
# ----------------------

@dataclass
class ThinkingProgress:
    """
    AI思考進捗を表すデータクラス

    各AIタイプで使用するフィールドが異なります。
    UIでの表示や進捗バーの更新に使用します。
    """
    ai_type: str              # "minimax", "mcts", "vcf"
    elapsed_time: float       # 経過時間（秒）

    # Minimax用
    current_depth: int = 0        # 現在の探索深度
    max_depth: int = 0            # 最大深度
    nodes_visited: int = 0        # 訪問ノード数
    current_best_move: Optional[Position] = None  # 現時点の最善手
    current_best_score: float = 0.0    # 評価値

    # MCTS用
    simulations_completed: int = 0   # 完了シミュレーション数
    total_simulations: int = 0       # 目標シミュレーション数
    top_moves: List[Tuple[Position, float]] = field(default_factory=list)  # 上位手と勝率

    # VCF用
    vcf_depth: int = 0               # 探索深度
    is_forced_win: bool = False      # 詰みが見つかったか
    win_sequence: List[Position] = field(default_factory=list)  # 詰み手順


# 進捗コールバック型
ThinkingProgressCallback = Callable[[ThinkingProgress], None]


class AIStrategy(ABC):
    """
    AI戦略の抽象基底クラス（Strategyパターン）

    全てのAI実装はこのインターフェースを実装します。
    select_move() は同期メソッドで、純粋に計算のみを行います。

    将来の拡張:
    - 学習ベースのAI（ニューラルネットワーク）
    - 開放型のAI（外部エンジン連携）
    """

    def __init__(self):
        self._progress_callback: Optional[ThinkingProgressCallback] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """戦略名（表示用）"""
        pass

    @property
    @abstractmethod
    def difficulty(self) -> str:
        """難易度表示（Easy/Medium/Hard）"""
        pass

    @property
    def supports_progress(self) -> bool:
        """進捗通知をサポートするかどうか"""
        return False

    def set_progress_callback(self, callback: Optional[ThinkingProgressCallback]) -> None:
        """
        進捗コールバックを設定

        Args:
            callback: 進捗通知時に呼ばれるコールバック関数（Noneで解除）
        """
        self._progress_callback = callback

    def _report_progress(self, progress: ThinkingProgress) -> None:
        """進捗をコールバックに通知（内部用）"""
        if self._progress_callback:
            try:
                self._progress_callback(progress)
            except Exception:
                pass  # コールバックのエラーは無視

    @abstractmethod
    def select_move(self, board: Board, rule: GameRule, stone: Stone) -> Position:
        """
        最善手を選択

        Args:
            board: 現在の盤面（コピーなので自由に変更可能）
            rule: ゲームルール
            stone: 次に置く石の色

        Returns:
            選択した座標

        Raises:
            ValueError: 合法手がない場合
        """
        pass


class RandomAI(AIStrategy):
    """
    ランダムAI

    合法手からランダムに選択します。
    テスト用・最弱AIとして使用します。
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: 乱数シード（テスト用、省略時はランダム）
        """
        super().__init__()
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "Random"

    @property
    def difficulty(self) -> str:
        return "Easy"

    def select_move(self, board: Board, rule: GameRule, stone: Stone) -> Position:
        """合法手からランダムに選択"""
        valid_moves = rule.get_valid_moves(board, stone)

        if not valid_moves:
            raise ValueError("No valid moves available")

        return self._rng.choice(valid_moves)


# ----------------------
# Minimax AI 実装
# ----------------------

@dataclass
class EvaluationWeights:
    """評価関数の重み設定"""
    five: int = 100000       # 5連（勝ち）
    open_four: int = 10000   # 両端開き4連（止められない）
    four: int = 1000         # 片端開き4連
    open_three: int = 500    # 両端開き3連
    three: int = 100         # 片端開き3連
    open_two: int = 50       # 両端開き2連
    two: int = 10            # 片端開き2連


class MinimaxAI(AIStrategy):
    """
    Minimax探索AI（Alpha-Beta枝刈り付き）

    探索深度と評価関数でプレイの強さが決まります。

    パラメータ目安:
    - depth=2: Easy（1秒未満）
    - depth=3: Medium（数秒）
    - depth=4-5: Hard（10秒以上）
    """

    def __init__(
        self,
        depth: int = 3,
        weights: Optional[EvaluationWeights] = None,
        time_limit: Optional[float] = None,
        use_iterative_deepening: bool = False
    ):
        """
        Args:
            depth: 探索深度（3-5が推奨、深いほど強いが遅い）
            weights: 評価関数の重み設定
            time_limit: 時間制限（秒）、省略時は制限なし
            use_iterative_deepening: 反復深化を使用するか
        """
        super().__init__()
        self._depth = depth
        self._weights = weights or EvaluationWeights()
        self._time_limit = time_limit
        self._use_iterative_deepening = use_iterative_deepening
        self._start_time: float = 0
        self._root_stone: Stone = Stone.EMPTY
        self._nodes_visited: int = 0

    @property
    def name(self) -> str:
        suffix = " ID" if self._use_iterative_deepening else ""
        return f"Minimax (depth={self._depth}){suffix}"

    @property
    def difficulty(self) -> str:
        if self._depth <= 2:
            return "Easy"
        elif self._depth <= 4:
            return "Medium"
        else:
            return "Hard"

    @property
    def supports_progress(self) -> bool:
        return True

    def select_move(self, board: Board, rule: GameRule, stone: Stone) -> Position:
        """Minimax探索で最善手を選択"""
        valid_moves = rule.get_valid_moves(board, stone)

        if not valid_moves:
            raise ValueError("No valid moves available")

        # 1手目は中央付近を選択（高速化）
        if board.move_count == 0:
            center = Position(board.width // 2, board.height // 2)
            if center in valid_moves:
                return center

        self._start_time = time.time()
        self._root_stone = stone
        self._nodes_visited = 0

        if self._use_iterative_deepening:
            return self._iterative_deepening_search(board, rule, stone, valid_moves)
        else:
            return self._fixed_depth_search(board, rule, stone, valid_moves, self._depth)

    def _iterative_deepening_search(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        valid_moves: List[Position]
    ) -> Position:
        """
        反復深化探索

        深さ1から順に探索し、各深度で最善手を更新します。
        時間切れの場合は、最後に完了した深度の最善手を返します。
        """
        best_move = valid_moves[0]
        best_score = float('-inf')

        # 候補手を評価順でソート（枝刈り効率化）
        sorted_moves = self._sort_moves(board, rule, stone, valid_moves)

        for depth in range(1, self._depth + 1):
            if self._is_timeout():
                break

            depth_start_nodes = self._nodes_visited
            move, score = self._search_at_depth(board, rule, stone, sorted_moves, depth)

            # 時間切れでなければ結果を採用
            if not self._is_timeout():
                best_move = move
                best_score = score

                # 進捗通知
                self._report_progress(ThinkingProgress(
                    ai_type="minimax",
                    elapsed_time=time.time() - self._start_time,
                    current_depth=depth,
                    max_depth=self._depth,
                    nodes_visited=self._nodes_visited,
                    current_best_move=best_move,
                    current_best_score=best_score,
                ))

                # 必勝が見つかった場合は即座に返す
                if score >= self._weights.five:
                    break

        return best_move

    def _search_at_depth(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        sorted_moves: List[Position],
        depth: int
    ) -> Tuple[Position, float]:
        """指定深度で探索"""
        best_move = sorted_moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for move in sorted_moves:
            if self._is_timeout():
                break

            self._nodes_visited += 1

            # 仮に手を打つ
            new_board = board.copy()
            new_board.set_stone(move.x, move.y, stone)

            # 勝敗判定
            status = rule.check_winner(new_board, move.x, move.y, stone)
            if status != GameStatus.ONGOING:
                # 即勝ちの手
                if (status == GameStatus.BLACK_WIN and stone == Stone.BLACK) or \
                   (status == GameStatus.WHITE_WIN and stone == Stone.WHITE):
                    return move, self._weights.five * (depth + 1)

            # 相手の応手を探索
            score = self._minimax(
                new_board, rule, stone.opponent(),
                depth - 1, alpha, beta, False,
                move
            )

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

        return best_move, best_score

    def _fixed_depth_search(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        valid_moves: List[Position],
        depth: int
    ) -> Position:
        """固定深度探索（従来の動作）"""
        best_move = valid_moves[0]
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        # 候補手を評価順でソート（枝刈り効率化）
        sorted_moves = self._sort_moves(board, rule, stone, valid_moves)

        for move in sorted_moves:
            if self._is_timeout():
                break

            self._nodes_visited += 1

            # 仮に手を打つ
            new_board = board.copy()
            new_board.set_stone(move.x, move.y, stone)

            # 勝敗判定
            status = rule.check_winner(new_board, move.x, move.y, stone)
            if status != GameStatus.ONGOING:
                # 即勝ちの手
                if (status == GameStatus.BLACK_WIN and stone == Stone.BLACK) or \
                   (status == GameStatus.WHITE_WIN and stone == Stone.WHITE):
                    return move

            # 相手の応手を探索
            score = self._minimax(
                new_board, rule, stone.opponent(),
                depth - 1, alpha, beta, False,
                move
            )

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, score)

            # 進捗通知（定期的に）
            if self._nodes_visited % 100 == 0:
                self._report_progress(ThinkingProgress(
                    ai_type="minimax",
                    elapsed_time=time.time() - self._start_time,
                    current_depth=depth,
                    max_depth=depth,
                    nodes_visited=self._nodes_visited,
                    current_best_move=best_move,
                    current_best_score=best_score,
                ))

        return best_move

    def _minimax(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        depth: int,
        alpha: float,
        beta: float,
        is_maximizing: bool,
        last_move: Position
    ) -> float:
        """
        Minimax探索（Alpha-Beta枝刈り）

        Args:
            board: 現在の盤面
            rule: ゲームルール
            stone: 次に打つ石の色
            depth: 残り探索深度
            alpha: Alpha値
            beta: Beta値
            is_maximizing: 最大化プレイヤーの手番か
            last_move: 直前の手（勝敗判定用）

        Returns:
            評価スコア
        """
        self._nodes_visited += 1

        # 時間制限チェック
        if self._is_timeout():
            return 0

        # 勝敗判定（直前の手で）
        last_stone = stone.opponent()
        status = rule.check_winner(board, last_move.x, last_move.y, last_stone)

        if status != GameStatus.ONGOING:
            if status == GameStatus.DRAW:
                return 0
            # ルートプレイヤーが勝ったら正、負けたら負
            if (status == GameStatus.BLACK_WIN and self._root_stone == Stone.BLACK) or \
               (status == GameStatus.WHITE_WIN and self._root_stone == Stone.WHITE):
                return self._weights.five * (depth + 1)
            else:
                return -self._weights.five * (depth + 1)

        # 深度0
        if depth == 0:
            return self._evaluate(board, rule)

        # 候補手取得
        valid_moves = self._get_candidate_moves(board, rule, stone)

        if not valid_moves:
            return 0  # 引き分け

        if is_maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                if self._is_timeout():
                    break

                new_board = board.copy()
                new_board.set_stone(move.x, move.y, stone)

                eval_score = self._minimax(
                    new_board, rule, stone.opponent(),
                    depth - 1, alpha, beta, False, move
                )

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                if self._is_timeout():
                    break

                new_board = board.copy()
                new_board.set_stone(move.x, move.y, stone)

                eval_score = self._minimax(
                    new_board, rule, stone.opponent(),
                    depth - 1, alpha, beta, True, move
                )

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval

    def _evaluate(self, board: Board, rule: GameRule) -> float:
        """
        局面を評価

        ルートプレイヤー視点でスコアを返します。
        """
        my_score = self._count_patterns(board, rule, self._root_stone)
        opp_score = self._count_patterns(board, rule, self._root_stone.opponent())

        return my_score - opp_score * 1.1  # 防御を少し重視

    def _count_patterns(self, board: Board, rule: GameRule, stone: Stone) -> float:
        """指定した石の色のパターンをカウントしてスコア化"""
        score = 0.0
        checked: set[tuple[int, int, int, int]] = set()

        # 4方向
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for y in range(board.height):
            for x in range(board.width):
                if board.get_stone(x, y) != stone:
                    continue

                for dx, dy in directions:
                    # 同じラインを重複カウントしない
                    line_key = (x, y, dx, dy)
                    if line_key in checked:
                        continue

                    count, open_ends = self._count_line(board, x, y, dx, dy, stone)

                    # ラインをマーク
                    for i in range(count):
                        checked.add((x + i * dx, y + i * dy, dx, dy))

                    # スコア加算
                    score += self._line_score(count, open_ends)

        return score

    def _count_line(
        self,
        board: Board,
        start_x: int,
        start_y: int,
        dx: int,
        dy: int,
        stone: Stone
    ) -> tuple[int, int]:
        """
        指定方向の連続石数と開端数をカウント

        Returns:
            (連続石数, 開端数 0-2)
        """
        count = 0
        x, y = start_x, start_y

        # 後方を確認
        back_x, back_y = start_x - dx, start_y - dy
        back_open = board.is_within_bounds(back_x, back_y) and \
                    board.get_stone(back_x, back_y) == Stone.EMPTY

        # 前方にカウント
        while board.is_within_bounds(x, y) and board.get_stone(x, y) == stone:
            count += 1
            x += dx
            y += dy

        # 前方が開いているか
        front_open = board.is_within_bounds(x, y) and \
                     board.get_stone(x, y) == Stone.EMPTY

        open_ends = int(back_open) + int(front_open)

        return count, open_ends

    def _line_score(self, count: int, open_ends: int) -> float:
        """連続石数と開端数からスコアを計算"""
        if count >= 5:
            return self._weights.five
        elif count == 4:
            if open_ends == 2:
                return self._weights.open_four
            elif open_ends == 1:
                return self._weights.four
        elif count == 3:
            if open_ends == 2:
                return self._weights.open_three
            elif open_ends == 1:
                return self._weights.three
        elif count == 2:
            if open_ends == 2:
                return self._weights.open_two
            elif open_ends == 1:
                return self._weights.two

        return 0

    def _get_candidate_moves(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone
    ) -> list[Position]:
        """
        探索候補を絞り込み（既存石の周囲のみ）

        全マスを探索すると遅いため、石がある場所の周囲2マスのみを候補とします。
        """
        candidates: set[Position] = set()

        for y in range(board.height):
            for x in range(board.width):
                if board.get_stone(x, y) == Stone.EMPTY:
                    continue

                # 周囲2マスを候補に追加
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if rule.is_valid_move(board, nx, ny, stone):
                            candidates.add(Position(nx, ny))

        # 候補がない場合（最初の手など）は全合法手
        if not candidates:
            return rule.get_valid_moves(board, stone)

        return list(candidates)

    def _sort_moves(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        moves: list[Position]
    ) -> list[Position]:
        """手を評価順でソート（枝刈り効率化）"""
        scored_moves: list[tuple[float, Position]] = []

        for move in moves:
            # 簡易評価: その位置周辺の石の数
            score = 0.0
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    nx, ny = move.x + dx, move.y + dy
                    if board.is_within_bounds(nx, ny):
                        s = board.get_stone(nx, ny)
                        if s == stone:
                            score += 2
                        elif s != Stone.EMPTY:
                            score += 1
            scored_moves.append((score, move))

        scored_moves.sort(key=lambda x: x[0], reverse=True)

        return [m for _, m in scored_moves]

    def _is_timeout(self) -> bool:
        """時間制限チェック"""
        if self._time_limit is None:
            return False
        return time.time() - self._start_time > self._time_limit


# ----------------------
# MCTS AI 実装
# ----------------------

class MCTSNode:
    """MCTSの探索ノード"""

    def __init__(
        self,
        board: Board,
        stone: Stone,
        move: Optional[Position] = None,
        parent: Optional["MCTSNode"] = None
    ):
        self.board = board
        self.stone = stone  # このノードで次に打つ石の色
        self.move = move    # このノードに至った手
        self.parent = parent
        self.children: list["MCTSNode"] = []
        self.wins: float = 0
        self.visits: int = 0
        self.untried_moves: list[Position] = []

    def ucb1(self, exploration: float = 1.414) -> float:
        """UCB1値を計算"""
        if self.visits == 0:
            return float('inf')

        if self.parent is None or self.parent.visits == 0:
            return float('inf')

        exploitation = self.wins / self.visits
        exploration_term = exploration * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration_term


class MCTSAI(AIStrategy):
    """
    Monte Carlo Tree Search AI

    シミュレーションベースの探索で最善手を選択します。

    パラメータ目安:
    - simulations=500: 約1秒
    - simulations=1000: 約2秒
    - time_limit=5.0: 5秒まで探索
    """

    # 進捗通知間隔（シミュレーション数）
    PROGRESS_INTERVAL = 50

    def __init__(
        self,
        simulations: int = 1000,
        time_limit: Optional[float] = None,
        exploration: float = 1.414
    ):
        """
        Args:
            simulations: シミュレーション回数
            time_limit: 時間制限（秒）、設定時はsimulationsより優先
            exploration: UCB1の探索係数
        """
        super().__init__()
        self._simulations = simulations
        self._time_limit = time_limit
        self._exploration = exploration
        self._rng = random.Random()

    @property
    def name(self) -> str:
        if self._time_limit:
            return f"MCTS (time={self._time_limit}s)"
        return f"MCTS (sims={self._simulations})"

    @property
    def difficulty(self) -> str:
        return "Hard"

    @property
    def supports_progress(self) -> bool:
        return True

    def select_move(self, board: Board, rule: GameRule, stone: Stone) -> Position:
        """MCTS探索で最善手を選択"""
        valid_moves = rule.get_valid_moves(board, stone)

        if not valid_moves:
            raise ValueError("No valid moves available")

        if len(valid_moves) == 1:
            return valid_moves[0]

        # 1手目は中央付近
        if board.move_count == 0:
            center = Position(board.width // 2, board.height // 2)
            if center in valid_moves:
                return center

        # 候補手を絞り込む（既存の石の周囲のみ）
        candidate_moves = self._get_candidate_moves(board, valid_moves)

        # 即座の勝ち手をチェック
        winning_move = self._find_winning_move(board, rule, stone, candidate_moves)
        if winning_move:
            return winning_move

        # 相手の即座の勝ち手をブロック
        opponent = stone.opponent()
        opponent_winning = self._find_winning_move(board, rule, opponent, candidate_moves)
        if opponent_winning:
            return opponent_winning

        # 4連を作る手（次に勝てる）
        threat_move = self._find_threat_move(board, rule, stone, candidate_moves)
        if threat_move:
            return threat_move

        # 相手の4連をブロック
        opponent_threat = self._find_threat_move(board, rule, opponent, candidate_moves)
        if opponent_threat:
            return opponent_threat

        # ルートノード作成
        root = MCTSNode(board.copy(), stone)
        root.untried_moves = list(candidate_moves)

        start_time = time.time()
        simulations = 0
        total_simulations = self._simulations

        while True:
            # 終了条件チェック
            if self._time_limit:
                if time.time() - start_time > self._time_limit:
                    break
            else:
                if simulations >= self._simulations:
                    break

            # Selection & Expansion
            node = self._select(root, rule)

            # Simulation
            result = self._simulate(node, rule)

            # Backpropagation
            self._backpropagate(node, result, stone)

            simulations += 1

            # 進捗通知
            if simulations % self.PROGRESS_INTERVAL == 0:
                top_moves = self._get_top_moves(root, 5)
                self._report_progress(ThinkingProgress(
                    ai_type="mcts",
                    elapsed_time=time.time() - start_time,
                    simulations_completed=simulations,
                    total_simulations=total_simulations,
                    top_moves=top_moves,
                ))

        # 子がない場合（全て未試行のまま終了した場合）
        if not root.children:
            return self._rng.choice(valid_moves)

        # 最終進捗通知
        top_moves = self._get_top_moves(root, 5)
        self._report_progress(ThinkingProgress(
            ai_type="mcts",
            elapsed_time=time.time() - start_time,
            simulations_completed=simulations,
            total_simulations=total_simulations,
            top_moves=top_moves,
        ))

        # 最多訪問ノードを選択
        best_child = max(root.children, key=lambda c: c.visits)

        return best_child.move  # type: ignore

    def _get_candidate_moves(
        self,
        board: Board,
        valid_moves: List[Position],
        radius: int = 2
    ) -> List[Position]:
        """既存の石の周囲のみを候補とする"""
        if board.move_count == 0:
            return valid_moves

        candidates = set()

        for y in range(board.height):
            for x in range(board.width):
                if board.get_stone(x, y) != Stone.EMPTY:
                    # 周囲をチェック
                    for dy in range(-radius, radius + 1):
                        for dx in range(-radius, radius + 1):
                            nx, ny = x + dx, y + dy
                            if board.is_within_bounds(nx, ny) and board.is_empty(nx, ny):
                                candidates.add(Position(nx, ny))

        # 合法手との交差
        result = [m for m in valid_moves if m in candidates]
        return result if result else valid_moves

    def _find_winning_move(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        valid_moves: List[Position]
    ) -> Optional[Position]:
        """即座に勝てる手を探す"""
        for move in valid_moves:
            test_board = board.copy()
            test_board.set_stone(move.x, move.y, stone)
            status = rule.check_winner(test_board, move.x, move.y, stone)
            if (status == GameStatus.BLACK_WIN and stone == Stone.BLACK) or \
               (status == GameStatus.WHITE_WIN and stone == Stone.WHITE):
                return move
        return None

    def _find_threat_move(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        valid_moves: List[Position]
    ) -> Optional[Position]:
        """4連を作る手を探す（次のターンで勝てる脅威）"""
        win_condition = getattr(rule, 'win_condition', 5)
        target_count = win_condition - 1  # 4

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for move in valid_moves:
            for dx, dy in directions:
                count = 1  # この手自身

                # 正方向
                for i in range(1, win_condition):
                    nx, ny = move.x + dx * i, move.y + dy * i
                    if not board.is_within_bounds(nx, ny):
                        break
                    if board.get_stone(nx, ny) == stone:
                        count += 1
                    else:
                        break

                # 負方向
                for i in range(1, win_condition):
                    nx, ny = move.x - dx * i, move.y - dy * i
                    if not board.is_within_bounds(nx, ny):
                        break
                    if board.get_stone(nx, ny) == stone:
                        count += 1
                    else:
                        break

                if count >= target_count:
                    return move

        return None

    def _get_top_moves(
        self,
        root: MCTSNode,
        count: int
    ) -> List[Tuple[Position, float]]:
        """上位の手と勝率を取得"""
        if not root.children:
            return []

        # 訪問回数でソート
        sorted_children = sorted(
            root.children,
            key=lambda c: c.visits,
            reverse=True
        )

        top_moves = []
        for child in sorted_children[:count]:
            if child.visits > 0 and child.move:
                win_rate = child.wins / child.visits
                top_moves.append((child.move, win_rate))

        return top_moves

    def _select(self, node: MCTSNode, rule: GameRule) -> MCTSNode:
        """Selection + Expansion"""
        while True:
            # 終局判定
            if node.move:
                last_stone = node.stone.opponent()
                status = rule.check_winner(
                    node.board, node.move.x, node.move.y, last_stone
                )
                if status != GameStatus.ONGOING:
                    return node

            # 未試行の手があれば展開
            if node.untried_moves:
                return self._expand(node, rule)

            # 子がなければ終端
            if not node.children:
                return node

            # UCB1で最良の子を選択
            node = max(node.children, key=lambda c: c.ucb1(self._exploration))

    def _expand(self, node: MCTSNode, rule: GameRule) -> MCTSNode:
        """ノードを展開"""
        move = self._rng.choice(node.untried_moves)
        node.untried_moves.remove(move)

        new_board = node.board.copy()
        new_board.set_stone(move.x, move.y, node.stone)

        child = MCTSNode(
            board=new_board,
            stone=node.stone.opponent(),
            move=move,
            parent=node
        )

        # 子ノードの未試行手を設定（候補を絞り込む）
        valid_moves = rule.get_valid_moves(new_board, child.stone)
        child.untried_moves = self._get_candidate_moves(new_board, valid_moves)

        node.children.append(child)

        return child

    def _simulate(self, node: MCTSNode, rule: GameRule) -> GameStatus:
        """ランダムプレイアウト（最適化版）"""
        board = node.board.copy()
        stone = node.stone
        last_move = node.move

        # 既に終局していればその結果を返す
        if last_move:
            status = rule.check_winner(
                board, last_move.x, last_move.y, stone.opponent()
            )
            if status != GameStatus.ONGOING:
                return status

        # 空セルのリストを作成（高速化のため）
        empty_cells = []
        for y in range(board.height):
            for x in range(board.width):
                if board.is_empty(x, y):
                    empty_cells.append(Position(x, y))

        # ランダムに終局まで打つ
        max_moves = len(empty_cells)

        for _ in range(max_moves):
            if not empty_cells:
                return GameStatus.DRAW

            # ランダムに空セルを選択
            idx = self._rng.randint(0, len(empty_cells) - 1)
            move = empty_cells[idx]
            empty_cells[idx] = empty_cells[-1]
            empty_cells.pop()

            # 合法手チェック（重力ルールなどの対応）
            if not rule.is_valid_move(board, move.x, move.y, stone):
                continue

            board.set_stone(move.x, move.y, stone)

            status = rule.check_winner(board, move.x, move.y, stone)
            if status != GameStatus.ONGOING:
                return status

            stone = stone.opponent()

        return GameStatus.DRAW

    def _backpropagate(
        self,
        node: MCTSNode,
        result: GameStatus,
        root_stone: Stone
    ) -> None:
        """結果を親に伝播"""
        while node is not None:
            node.visits += 1

            # 勝敗判定
            # UCB1で子ノードを選択する際、親から見て良い手を選びたい
            # node.stone = 次に打つプレイヤー
            # node.stone.opponent() = このノードに至る手を打ったプレイヤー
            # なので、そのプレイヤーが勝ったらwinsをカウント
            if result == GameStatus.DRAW:
                node.wins += 0.5
            elif result == GameStatus.BLACK_WIN:
                # BLACKが勝った場合、BLACKが打った手（=WHITEの番のノード）にポイント
                if node.stone == Stone.WHITE:
                    node.wins += 1
            elif result == GameStatus.WHITE_WIN:
                # WHITEが勝った場合、WHITEが打った手（=BLACKの番のノード）にポイント
                if node.stone == Stone.BLACK:
                    node.wins += 1

            node = node.parent  # type: ignore


# ----------------------
# AI戦略ファクトリ
# ----------------------

class AIStrategyFactory:
    """AI戦略のファクトリ"""

    @staticmethod
    def create(name: str, **kwargs) -> AIStrategy:
        """
        名前からAI戦略を作成

        Args:
            name: 戦略名（"random", "minimax", "mcts"）
            **kwargs: 各戦略固有のパラメータ

        Returns:
            AI戦略インスタンス

        Examples:
            >>> factory = AIStrategyFactory()
            >>> ai = factory.create("minimax", depth=3)
            >>> ai = factory.create("mcts", simulations=1000)
        """
        name_lower = name.lower()

        if name_lower == "random":
            return RandomAI(seed=kwargs.get("seed"))

        elif name_lower == "minimax":
            return MinimaxAI(
                depth=kwargs.get("depth", 3),
                time_limit=kwargs.get("time_limit"),
                use_iterative_deepening=kwargs.get("use_iterative_deepening", False)
            )

        elif name_lower == "mcts":
            return MCTSAI(
                simulations=kwargs.get("simulations", 1000),
                time_limit=kwargs.get("time_limit"),
                exploration=kwargs.get("exploration", 1.414)
            )

        else:
            raise ValueError(f"Unknown AI strategy: {name}")

    @staticmethod
    def list_available() -> list[str]:
        """利用可能な戦略名一覧"""
        return ["random", "minimax", "mcts", "vcf"]
