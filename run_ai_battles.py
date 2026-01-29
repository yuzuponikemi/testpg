"""
AI対戦バッチスクリプト

複数のAI同士を対戦させ、棋譜を記録します。

使用例:
    # デフォルト設定（Minimax vs MCTS を3回）
    python run_ai_battles.py

    # カスタム設定
    python run_ai_battles.py --black vcf --white minimax --games 10

    # 詳細表示
    python run_ai_battles.py --verbose
"""

import asyncio
import argparse
from datetime import datetime
from pathlib import Path

from game_core import RuleRegistry, GameStatus
from players import AIPlayer, GameSession, GameSessionConfig
from ai_strategies import AIStrategyFactory
from game_record import GameRecorder


def create_ai_player(ai_type: str, name: str, depth: int = 3, simulations: int = 500, use_vct: bool = False):
    """AIプレイヤーを作成"""
    if ai_type == "vcf":
        from threat_search import VCFBasedAI
        from ai_strategies import RandomAI
        strategy = VCFBasedAI(use_vct=use_vct, fallback=RandomAI())
    else:
        strategy = AIStrategyFactory.create(
            ai_type,
            depth=depth,
            simulations=simulations,
        )
    return AIPlayer(strategy, f"{name} ({strategy.name})")


async def run_single_game(
    rule,
    black_player,
    white_player,
    recorder: GameRecorder,
    game_number: int,
    verbose: bool = False
) -> GameStatus:
    """1対局を実行"""
    # セッション作成
    session_config = GameSessionConfig(delay_between_moves=0.0)
    session = GameSession(
        rule=rule,
        black_player=black_player,
        white_player=white_player,
        config=session_config,
    )

    # 記録開始
    recorder.start_game(rule, black_player, white_player)

    # イベントリスナーで手を記録
    def on_move(event):
        if event.event_type == "MOVE_PLAYED" and event.position and event.stone:
            recorder.record_move(event.position, event.stone)

    session.engine.add_listener(on_move)

    if verbose:
        print(f"  Game {game_number}: Starting...")

    # 対局実行
    result = await session.run_game()

    # 記録終了・保存
    recorder.end_game(result)
    filepath = recorder.save()

    if verbose:
        print(f"  Game {game_number}: {result.name} ({len(session.engine.move_history)} moves)")
        print(f"    Saved to: {filepath}")

    return result


async def run_battles(args):
    """複数対局を実行"""
    # 出力ディレクトリ
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ルール作成
    rule_map = {
        "standard": "StandardGomokuRule",
        "gravity": "GravityGomokuRule",
    }
    rule = RuleRegistry.create(rule_map[args.rule])

    # レコーダー作成
    recorder = GameRecorder(
        output_dir=str(output_dir),
        format=args.format,
        enabled=True,
    )

    print("=" * 60)
    print("AI Battle Script")
    print("=" * 60)
    print(f"Black: {args.black}")
    print(f"White: {args.white}")
    print(f"Rule: {args.rule}")
    print(f"Games: {args.games}")
    print(f"Output: {output_dir.absolute()}")
    print(f"Format: {args.format}")
    print("=" * 60)

    # 統計
    results = {
        GameStatus.BLACK_WIN: 0,
        GameStatus.WHITE_WIN: 0,
        GameStatus.DRAW: 0,
    }

    start_time = datetime.now()

    for i in range(1, args.games + 1):
        # プレイヤー作成（毎回新規作成してステートをリセット）
        black_player = create_ai_player(
            args.black, "Black",
            depth=args.depth,
            simulations=args.simulations,
            use_vct=args.vct,
        )
        white_player = create_ai_player(
            args.white, "White",
            depth=args.depth,
            simulations=args.simulations,
            use_vct=args.vct,
        )

        result = await run_single_game(
            rule, black_player, white_player,
            recorder, i, args.verbose
        )
        results[result] += 1

        if not args.verbose:
            # 進捗表示
            print(f"\rProgress: {i}/{args.games} games completed", end="", flush=True)

    if not args.verbose:
        print()  # 改行

    elapsed = datetime.now() - start_time

    # 結果サマリー
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Black ({args.black}) wins: {results[GameStatus.BLACK_WIN]}")
    print(f"White ({args.white}) wins: {results[GameStatus.WHITE_WIN]}")
    print(f"Draws: {results[GameStatus.DRAW]}")
    print(f"Total time: {elapsed}")
    print(f"Average time per game: {elapsed / args.games}")
    print("=" * 60)

    # 勝率計算
    total = args.games
    if total > 0:
        black_rate = results[GameStatus.BLACK_WIN] / total * 100
        white_rate = results[GameStatus.WHITE_WIN] / total * 100
        draw_rate = results[GameStatus.DRAW] / total * 100
        print(f"Win rates: Black {black_rate:.1f}% / White {white_rate:.1f}% / Draw {draw_rate:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run AI vs AI battles and record games",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_ai_battles.py                          # Default: minimax vs mcts, 3 games
  python run_ai_battles.py -b vcf -w minimax -n 10  # VCF vs Minimax, 10 games
  python run_ai_battles.py --verbose                # Show detailed progress
  python run_ai_battles.py --format jsonl           # Save as JSONL (one file)
        """
    )

    parser.add_argument(
        "--black", "-b",
        choices=["random", "minimax", "mcts", "vcf"],
        default="minimax",
        help="Black player AI type (default: minimax)"
    )
    parser.add_argument(
        "--white", "-w",
        choices=["random", "minimax", "mcts", "vcf"],
        default="mcts",
        help="White player AI type (default: mcts)"
    )
    parser.add_argument(
        "--rule", "-r",
        choices=["standard", "gravity"],
        default="standard",
        help="Game rule (default: standard)"
    )
    parser.add_argument(
        "--games", "-n",
        type=int,
        default=3,
        help="Number of games to play (default: 3)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        default=3,
        help="Minimax search depth (default: 3)"
    )
    parser.add_argument(
        "--simulations", "-s",
        type=int,
        default=500,
        help="MCTS simulation count (default: 500)"
    )
    parser.add_argument(
        "--vct",
        action="store_true",
        help="Enable VCT for VCF AI"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./game_logs",
        help="Output directory for game records (default: ./game_logs)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "jsonl", "text", "csv", "seq"],
        default="json",
        help="Record format: json, jsonl, text, csv, seq (default: json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )

    args = parser.parse_args()
    asyncio.run(run_battles(args))


if __name__ == "__main__":
    main()
