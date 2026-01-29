"""
Variant Go Platform - Main Application (Milestone 4)

Fletアプリのエントリーポイント。

実行方法:
    # GUI設定画面から開始
    python main.py

    # コマンドラインでAI対AI対戦
    python main.py --black minimax --white mcts
    python main.py --black vcf --white minimax --depth 4
    python main.py -b mcts -w mcts --simulations 2000

    # 利用可能なプレイヤータイプ:
    #   human, random, minimax, mcts, vcf

オプション:
    --black, -b     黒プレイヤーのタイプ
    --white, -w     白プレイヤーのタイプ
    --rule, -r      ルール (standard, gravity)
    --depth, -d     Minimax探索深度 (1-5)
    --simulations   MCTSシミュレーション回数
    --iterative     反復深化を有効化
    --vct           VCT探索を有効化 (VCF AIのみ)
    --no-thinking   思考パネルを非表示
    --record        棋譜を記録
    --record-dir    棋譜保存先 (default: ./game_logs)
    --record-format 保存形式: json (1ファイル1対局) or jsonl (追記)
"""

import argparse
import flet as ft

from ui.settings_view import SettingsView, GameConfig
from ui.game_view import GameView


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="Variant Go Platform - Board Game with Variant Rules",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Player Types:
  human     Human player
  random    Random AI (Easy)
  minimax   Minimax search AI (Medium)
  mcts      Monte Carlo Tree Search AI (Hard)
  vcf       VCF threat search AI (Hard)

Examples:
  python main.py                           # Start with GUI settings
  python main.py -b minimax -w mcts        # Minimax vs MCTS
  python main.py -b vcf -w vcf --vct       # VCF+VCT vs VCF+VCT
  python main.py -b human -w minimax -d 4  # Human vs Minimax(depth=4)
        """
    )

    parser.add_argument(
        "--black", "-b",
        choices=["human", "random", "minimax", "mcts", "vcf"],
        help="Black player type"
    )
    parser.add_argument(
        "--white", "-w",
        choices=["human", "random", "minimax", "mcts", "vcf"],
        help="White player type"
    )
    parser.add_argument(
        "--rule", "-r",
        choices=["standard", "gravity"],
        default="standard",
        help="Game rule (default: standard)"
    )
    parser.add_argument(
        "--depth", "-d",
        type=int,
        choices=[1, 2, 3, 4, 5],
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
        "--iterative", "-i",
        action="store_true",
        help="Enable iterative deepening (Minimax only)"
    )
    parser.add_argument(
        "--vct",
        action="store_true",
        help="Enable VCT search (VCF AI only)"
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Hide thinking panel"
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable game recording"
    )
    parser.add_argument(
        "--record-dir",
        type=str,
        default="./game_logs",
        help="Directory to save game records (default: ./game_logs)"
    )
    parser.add_argument(
        "--record-format",
        choices=["json", "jsonl", "text", "csv", "seq"],
        default="json",
        help="Record format: json, jsonl, text, csv, seq"
    )

    return parser.parse_args()


def create_config_from_args(args) -> GameConfig:
    """コマンドライン引数からGameConfigを作成"""
    rule_map = {
        "standard": "StandardGomokuRule",
        "gravity": "GravityGomokuRule",
    }

    return GameConfig(
        rule_id=rule_map[args.rule],
        black_player_type=args.black,
        white_player_type=args.white,
        ai_depth=args.depth,
        ai_simulations=args.simulations,
        use_iterative_deepening=args.iterative,
        use_vct=args.vct,
        show_thinking=not args.no_thinking,
        record_games=args.record,
        record_dir=args.record_dir,
        record_format=args.record_format,
    )


def main(page: ft.Page, initial_config: GameConfig = None) -> None:
    """Fletアプリのメイン関数"""
    # ページ設定
    page.title = "Variant Go Platform"
    page.window.width = 550
    page.window.height = 750
    page.window.min_width = 400
    page.window.min_height = 600
    page.theme_mode = ft.ThemeMode.LIGHT
    page.padding = 0

    def go_to_game(config: GameConfig) -> None:
        """ゲーム画面に遷移"""
        page.views.clear()
        game_view = GameView(page, config, go_to_settings)
        page.views.append(game_view)
        page.update()
        # ゲームを開始
        game_view.start_game()

    def go_to_settings() -> None:
        """設定画面に遷移"""
        page.views.clear()
        page.views.append(SettingsView(page, go_to_game))
        page.update()

    # 初期設定があればゲーム画面へ、なければ設定画面へ
    if initial_config:
        go_to_game(initial_config)
    else:
        go_to_settings()


# グローバル変数で初期設定を保持（ft.runのコールバックに渡すため）
_initial_config: GameConfig = None


def _main_wrapper(page: ft.Page) -> None:
    """ft.run用のラッパー"""
    main(page, _initial_config)


if __name__ == "__main__":
    args = parse_args()

    # --blackまたは--whiteが指定されていればコマンドラインモード
    if args.black or args.white:
        # 両方指定されていなければデフォルトを設定
        if not args.black:
            args.black = "human"
        if not args.white:
            args.white = "minimax"

        _initial_config = create_config_from_args(args)
        print(f"Starting game: {args.black} vs {args.white}")
        print(f"Rule: {args.rule}, Depth: {args.depth}, Simulations: {args.simulations}")
        if args.iterative:
            print("Iterative deepening: enabled")
        if args.vct:
            print("VCT search: enabled")
        if args.record:
            print(f"Recording: {args.record_format} format to {args.record_dir}")

    ft.run(_main_wrapper)
