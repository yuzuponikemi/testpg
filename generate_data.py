#!/usr/bin/env python3
"""
学習データ生成スクリプト

Minimaxプレイヤー同士、またはMinimax vs Randomの対戦データを
大量に生成し、Transformerモデルの模倣学習用データとして保存します。

使用方法:
    python generate_data.py --games 1000 --output training_data.jsonl
    python generate_data.py -n 100 -o data.jsonl --depth 3 --workers 8
"""

import argparse
import json
import random
import time
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import List, Optional, Tuple

from game_core import (
    Board, GameRule, StandardGomokuRule, GameStatus, Stone, Position
)
from ai_strategies import AIStrategy, MinimaxAI, RandomAI, EvaluationWeights
from data_augmentation import augment_moves


# =============================================================================
# Stochastic Minimax AI（ゆらぎ付きMinimax）
# =============================================================================

class StochasticMinimaxAI(AIStrategy):
    """
    確率的なMinimaxAI

    評価値上位の手からランダムに選択することで、
    毎回同じ棋譜にならないようにします。
    """

    def __init__(
        self,
        depth: int = 3,
        top_k: int = 3,
        temperature: float = 1.0,
        seed: Optional[int] = None
    ):
        """
        Args:
            depth: 探索深度
            top_k: 上位何手から選択するか
            temperature: 選択の確率分布の温度（高いほどランダム）
            seed: 乱数シード
        """
        super().__init__()
        self._depth = depth
        self._top_k = top_k
        self._temperature = temperature
        self._weights = EvaluationWeights()
        self._rng = random.Random(seed)
        self._root_stone: Stone = Stone.EMPTY

    @property
    def name(self) -> str:
        return f"StochasticMinimax(d={self._depth}, k={self._top_k})"

    @property
    def difficulty(self) -> str:
        return "Medium"

    def select_move(self, board: Board, rule: GameRule, stone: Stone) -> Position:
        """上位k手から確率的に選択"""
        valid_moves = rule.get_valid_moves(board, stone)

        if not valid_moves:
            raise ValueError("No valid moves available")

        if len(valid_moves) == 1:
            return valid_moves[0]

        # 1手目は中央付近にランダム性を持たせる
        if board.move_count == 0:
            center_x, center_y = board.width // 2, board.height // 2
            center_moves = [
                m for m in valid_moves
                if abs(m.x - center_x) <= 1 and abs(m.y - center_y) <= 1
            ]
            if center_moves:
                return self._rng.choice(center_moves)

        self._root_stone = stone

        # 全ての手を評価
        scored_moves = self._evaluate_all_moves(board, rule, stone, valid_moves)

        # 上位k手を取得
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        top_moves = scored_moves[:min(self._top_k, len(scored_moves))]

        # 温度付きソフトマックスで選択
        if self._temperature <= 0:
            return top_moves[0][1]

        # スコアを確率に変換
        scores = [s for s, _ in top_moves]
        min_score = min(scores)
        adjusted_scores = [(s - min_score) / self._temperature for s in scores]

        # オーバーフロー防止
        max_adj = max(adjusted_scores)
        exp_scores = [2 ** (s - max_adj) for s in adjusted_scores]  # expの代わりに2^xを使用
        total = sum(exp_scores)
        probs = [e / total for e in exp_scores]

        # 確率的に選択
        r = self._rng.random()
        cumsum = 0
        for prob, (_, move) in zip(probs, top_moves):
            cumsum += prob
            if r <= cumsum:
                return move

        return top_moves[-1][1]

    def _evaluate_all_moves(
        self,
        board: Board,
        rule: GameRule,
        stone: Stone,
        valid_moves: List[Position]
    ) -> List[Tuple[float, Position]]:
        """全ての手を評価してスコア付き"""
        results = []

        # 候補手を絞り込む（既存石の周囲のみ）
        candidates = self._get_candidate_moves(board, rule, stone, valid_moves)

        for move in candidates:
            # 仮に手を打つ
            new_board = board.copy()
            new_board.set_stone(move.x, move.y, stone)

            # 勝敗判定
            status = rule.check_winner(new_board, move.x, move.y, stone)
            if status != GameStatus.ONGOING:
                if (status == GameStatus.BLACK_WIN and stone == Stone.BLACK) or \
                   (status == GameStatus.WHITE_WIN and stone == Stone.WHITE):
                    # 即勝ちは最高スコア
                    results.append((self._weights.five * 10, move))
                    continue

            # Minimax探索
            score = self._minimax(
                new_board, rule, stone.opponent(),
                self._depth - 1, float('-inf'), float('inf'), False, move
            )
            results.append((score, move))

        return results

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
        """Minimax探索（Alpha-Beta枝刈り）"""
        # 勝敗判定
        last_stone = stone.opponent()
        status = rule.check_winner(board, last_move.x, last_move.y, last_stone)

        if status != GameStatus.ONGOING:
            if status == GameStatus.DRAW:
                return 0
            if (status == GameStatus.BLACK_WIN and self._root_stone == Stone.BLACK) or \
               (status == GameStatus.WHITE_WIN and self._root_stone == Stone.WHITE):
                return self._weights.five * (depth + 1)
            else:
                return -self._weights.five * (depth + 1)

        if depth == 0:
            return self._evaluate(board, rule)

        candidates = self._get_candidate_moves(board, rule, stone)
        if not candidates:
            return 0

        if is_maximizing:
            max_eval = float('-inf')
            for move in candidates:
                new_board = board.copy()
                new_board.set_stone(move.x, move.y, stone)
                eval_score = self._minimax(
                    new_board, rule, stone.opponent(),
                    depth - 1, alpha, beta, False, move
                )
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in candidates:
                new_board = board.copy()
                new_board.set_stone(move.x, move.y, stone)
                eval_score = self._minimax(
                    new_board, rule, stone.opponent(),
                    depth - 1, alpha, beta, True, move
                )
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def _evaluate(self, board: Board, rule: GameRule) -> float:
        """局面評価"""
        my_score = self._count_patterns(board, self._root_stone)
        opp_score = self._count_patterns(board, self._root_stone.opponent())
        return my_score - opp_score * 1.1

    def _count_patterns(self, board: Board, stone: Stone) -> float:
        """パターンカウント"""
        score = 0.0
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        for y in range(board.height):
            for x in range(board.width):
                if board.get_stone(x, y) != stone:
                    continue
                for dx, dy in directions:
                    count, open_ends = self._count_line(board, x, y, dx, dy, stone)
                    score += self._line_score(count, open_ends)

        return score

    def _count_line(
        self, board: Board, x: int, y: int, dx: int, dy: int, stone: Stone
    ) -> Tuple[int, int]:
        """連続石数をカウント"""
        # 後方が同じ石ならスキップ（重複カウント防止）
        bx, by = x - dx, y - dy
        if board.is_within_bounds(bx, by) and board.get_stone(bx, by) == stone:
            return 0, 0

        count = 0
        cx, cy = x, y
        while board.is_within_bounds(cx, cy) and board.get_stone(cx, cy) == stone:
            count += 1
            cx += dx
            cy += dy

        # 開端数
        back_open = board.is_within_bounds(bx, by) and board.get_stone(bx, by) == Stone.EMPTY
        front_open = board.is_within_bounds(cx, cy) and board.get_stone(cx, cy) == Stone.EMPTY
        open_ends = int(back_open) + int(front_open)

        return count, open_ends

    def _line_score(self, count: int, open_ends: int) -> float:
        """連続数からスコア計算"""
        if count >= 5:
            return self._weights.five
        elif count == 4:
            return self._weights.open_four if open_ends == 2 else self._weights.four if open_ends == 1 else 0
        elif count == 3:
            return self._weights.open_three if open_ends == 2 else self._weights.three if open_ends == 1 else 0
        elif count == 2:
            return self._weights.open_two if open_ends == 2 else self._weights.two if open_ends == 1 else 0
        return 0

    def _get_candidate_moves(
        self, board: Board, rule: GameRule, stone: Stone,
        valid_moves: Optional[List[Position]] = None
    ) -> List[Position]:
        """候補手を絞り込み"""
        candidates = set()

        for y in range(board.height):
            for x in range(board.width):
                if board.get_stone(x, y) == Stone.EMPTY:
                    continue
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        nx, ny = x + dx, y + dy
                        if rule.is_valid_move(board, nx, ny, stone):
                            candidates.add(Position(nx, ny))

        if not candidates:
            return valid_moves if valid_moves else rule.get_valid_moves(board, stone)

        return list(candidates)


# =============================================================================
# ゲーム実行と記録
# =============================================================================

@dataclass
class GameRecord:
    """1局の記録"""
    moves: List[int]  # 着手位置のシーケンス（0-224のインデックス）
    winner: int       # 1=黒勝ち, -1=白勝ち, 0=引き分け
    match_type: str   # "minimax_vs_minimax" or "minimax_vs_random"


def position_to_index(pos: Position, width: int = 15) -> int:
    """Position を 0-224 のインデックスに変換"""
    return pos.y * width + pos.x


def play_single_game(args: Tuple[int, int, int, str]) -> Optional[GameRecord]:
    """
    1局をプレイして記録を返す

    Args:
        args: (game_id, depth, seed, match_type)

    Returns:
        GameRecord or None (エラー時)
    """
    game_id, depth, seed, match_type = args

    try:
        # 乱数シードを設定（再現性のため）
        rng = random.Random(seed)

        # ルールとAIを作成
        rule = StandardGomokuRule()
        board = Board(rule.board_width, rule.board_height)

        # AIの作成（match_typeに応じて）
        if match_type == "minimax_vs_minimax":
            black_ai = StochasticMinimaxAI(depth=depth, top_k=3, temperature=0.5, seed=rng.randint(0, 2**31))
            white_ai = StochasticMinimaxAI(depth=depth, top_k=3, temperature=0.5, seed=rng.randint(0, 2**31))
        else:  # minimax_vs_random
            if rng.random() < 0.5:
                black_ai = StochasticMinimaxAI(depth=depth, top_k=3, temperature=0.5, seed=rng.randint(0, 2**31))
                white_ai = RandomAI(seed=rng.randint(0, 2**31))
            else:
                black_ai = RandomAI(seed=rng.randint(0, 2**31))
                white_ai = StochasticMinimaxAI(depth=depth, top_k=3, temperature=0.5, seed=rng.randint(0, 2**31))

        moves: List[int] = []
        current_stone = Stone.BLACK
        last_move: Optional[Position] = None

        # ゲームループ
        max_moves = rule.board_width * rule.board_height
        for _ in range(max_moves):
            # 現在のプレイヤーのAIを選択
            ai = black_ai if current_stone == Stone.BLACK else white_ai

            # 着手
            try:
                move = ai.select_move(board, rule, current_stone)
            except ValueError:
                # 合法手がない場合は引き分け
                return GameRecord(moves=moves, winner=0, match_type=match_type)

            # 着手を記録
            moves.append(position_to_index(move, rule.board_width))

            # 盤面に反映
            board.set_stone(move.x, move.y, current_stone)
            last_move = move

            # 勝敗判定
            status = rule.check_winner(board, move.x, move.y, current_stone)
            if status == GameStatus.BLACK_WIN:
                return GameRecord(moves=moves, winner=1, match_type=match_type)
            elif status == GameStatus.WHITE_WIN:
                return GameRecord(moves=moves, winner=-1, match_type=match_type)
            elif status == GameStatus.DRAW:
                return GameRecord(moves=moves, winner=0, match_type=match_type)

            # 手番交代
            current_stone = current_stone.opponent()

        # 最大手数到達（引き分け）
        return GameRecord(moves=moves, winner=0, match_type=match_type)

    except Exception as e:
        print(f"Game {game_id} error: {e}")
        return None


def generate_games(
    num_games: int,
    depth: int,
    output_path: Path,
    num_workers: int,
    minimax_ratio: float = 0.5,
    augment: bool = False
) -> None:
    """
    複数のゲームを並列生成してJSONLファイルに保存

    Args:
        num_games: 生成するゲーム数
        depth: Minimaxの探索深度
        output_path: 出力ファイルパス
        num_workers: 並列ワーカー数
        minimax_ratio: minimax_vs_minimaxの割合（残りはminimax_vs_random）
        augment: 8x対称変換でデータ拡張するか
    """
    print(f"Generating {num_games} games with {num_workers} workers...")
    print(f"  Minimax depth: {depth}")
    print(f"  Minimax vs Minimax ratio: {minimax_ratio:.0%}")
    print(f"  Data augmentation: {'8x symmetry' if augment else 'off'}")
    print(f"  Output: {output_path}")
    print()

    # タスクを準備
    tasks = []
    base_seed = int(time.time() * 1000) % (2**31)

    for i in range(num_games):
        match_type = "minimax_vs_minimax" if random.random() < minimax_ratio else "minimax_vs_random"
        seed = base_seed + i
        tasks.append((i, depth, seed, match_type))

    # 並列実行
    start_time = time.time()
    completed = 0
    records: List[GameRecord] = []

    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(play_single_game, tasks):
            completed += 1
            if result:
                records.append(result)

            # 進捗表示
            if completed % 10 == 0 or completed == num_games:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (num_games - completed) / rate if rate > 0 else 0
                print(f"\r  Progress: {completed}/{num_games} ({completed/num_games:.1%}) "
                      f"| {rate:.1f} games/s | ETA: {eta:.0f}s", end="", flush=True)

    print()

    # 結果を保存
    written_count = 0
    with open(output_path, 'w') as f:
        for record in records:
            if augment:
                # 8つの対称変換を適用
                for t_idx in range(8):
                    aug_moves = augment_moves(record.moves, t_idx)
                    line = json.dumps({
                        "input": aug_moves,
                        "winner": record.winner,
                        "match_type": record.match_type
                    })
                    f.write(line + '\n')
                    written_count += 1
            else:
                line = json.dumps({
                    "input": record.moves,
                    "winner": record.winner,
                    "match_type": record.match_type
                })
                f.write(line + '\n')
                written_count += 1

    # 統計を表示
    elapsed = time.time() - start_time
    black_wins = sum(1 for r in records if r.winner == 1)
    white_wins = sum(1 for r in records if r.winner == -1)
    draws = sum(1 for r in records if r.winner == 0)
    minimax_games = sum(1 for r in records if r.match_type == "minimax_vs_minimax")

    print()
    print("=" * 50)
    print(f"Generation complete!")
    print(f"  Total games: {len(records)}")
    if augment:
        print(f"  Written samples: {written_count} ({len(records)} x 8 symmetries)")
    else:
        print(f"  Written samples: {written_count}")
    print(f"  Time: {elapsed:.1f}s ({len(records)/elapsed:.1f} games/s)")
    print(f"  Black wins: {black_wins} ({black_wins/len(records):.1%})")
    print(f"  White wins: {white_wins} ({white_wins/len(records):.1%})")
    print(f"  Draws: {draws} ({draws/len(records):.1%})")
    print(f"  Minimax vs Minimax: {minimax_games}")
    print(f"  Minimax vs Random: {len(records) - minimax_games}")
    print(f"  Output: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Generate training data for imitation learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_data.py -n 100                    # Generate 100 games
  python generate_data.py -n 1000 -o data.jsonl    # Custom output file
  python generate_data.py -n 500 --depth 4         # Deeper search
  python generate_data.py -n 1000 --workers 16     # More parallelism
        """
    )

    parser.add_argument(
        "-n", "--games",
        type=int,
        default=100,
        help="Number of games to generate (default: 100)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="training_data.jsonl",
        help="Output file path (default: training_data.jsonl)"
    )
    parser.add_argument(
        "-d", "--depth",
        type=int,
        default=3,
        help="Minimax search depth (default: 3)"
    )
    parser.add_argument(
        "-w", "--workers",
        type=int,
        default=None,
        help=f"Number of worker processes (default: CPU count = {cpu_count()})"
    )
    parser.add_argument(
        "--minimax-ratio",
        type=float,
        default=0.5,
        help="Ratio of Minimax vs Minimax games (default: 0.5)"
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply 8x symmetry augmentation when writing JSONL"
    )

    args = parser.parse_args()

    num_workers = args.workers if args.workers else cpu_count()
    output_path = Path(args.output)

    generate_games(
        num_games=args.games,
        depth=args.depth,
        output_path=output_path,
        num_workers=num_workers,
        minimax_ratio=args.minimax_ratio,
        augment=args.augment
    )


if __name__ == "__main__":
    main()
