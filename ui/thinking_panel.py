"""
Variant Go Platform - Thinking Panel (AI Visualization)

AI思考過程の可視化パネルを提供します。
"""

import flet as ft
from typing import Optional, List
from dataclasses import dataclass
import threading

from ai_strategies import ThinkingProgress
from game_core import Position


@dataclass
class ThinkingLogEntry:
    """思考ログエントリ"""
    timestamp: float
    message: str
    detail: Optional[str] = None


class ThinkingPanel(ft.Container):
    """
    AI思考過程を表示するパネル

    表示要素:
    - 進捗バー（深度/シミュレーション進捗）
    - 統計テキスト（ノード数、時間、評価値）
    - 最善手テキスト
    - 思考ログ（詳細表示時）
    """

    def __init__(
        self,
        show_detailed_log: bool = True,
        max_log_entries: int = 50
    ):
        """
        Args:
            show_detailed_log: 詳細ログを表示するか
            max_log_entries: ログの最大エントリ数
        """
        super().__init__()

        self._show_detailed_log = show_detailed_log
        self._max_log_entries = max_log_entries
        self._log_entries: List[ThinkingLogEntry] = []
        self._lock = threading.Lock()
        self._page: Optional[ft.Page] = None

        # UI要素
        self._ai_type_text: ft.Text = None  # type: ignore
        self._progress_bar: ft.ProgressBar = None  # type: ignore
        self._progress_text: ft.Text = None  # type: ignore
        self._stats_text: ft.Text = None  # type: ignore
        self._best_move_text: ft.Text = None  # type: ignore
        self._log_column: ft.Column = None  # type: ignore

        self._build_ui()

    def _build_ui(self) -> None:
        """UIを構築"""
        # AIタイプ表示
        self._ai_type_text = ft.Text(
            "AI is thinking...",
            size=14,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.BLUE_700,
        )

        # 進捗バー
        self._progress_bar = ft.ProgressBar(
            width=300,
            value=0,
            color=ft.Colors.BLUE_400,
            bgcolor=ft.Colors.GREY_300,
        )

        self._progress_text = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_600,
        )

        # 統計テキスト
        self._stats_text = ft.Text(
            "",
            size=12,
            color=ft.Colors.GREY_700,
        )

        # 最善手テキスト
        self._best_move_text = ft.Text(
            "",
            size=14,
            weight=ft.FontWeight.BOLD,
            color=ft.Colors.GREEN_700,
        )

        # 詳細ログ
        self._log_column = ft.Column(
            controls=[],
            spacing=2,
            scroll=ft.ScrollMode.AUTO,
            height=100 if self._show_detailed_log else 0,
        )

        # ログコンテナ
        log_container = ft.Container(
            content=self._log_column,
            bgcolor=ft.Colors.GREY_100,
            border_radius=5,
            padding=5,
            visible=self._show_detailed_log,
        )

        # レイアウト
        content = ft.Column(
            controls=[
                self._ai_type_text,
                ft.Container(height=5),
                self._progress_bar,
                self._progress_text,
                ft.Container(height=5),
                self._stats_text,
                self._best_move_text,
                ft.Container(height=5),
                log_container,
            ],
            spacing=3,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.content = content
        self.padding = 10
        self.bgcolor = ft.Colors.BLUE_50
        self.border_radius = 10
        self.border = ft.border.all(1, ft.Colors.BLUE_200)

    def set_page(self, page: ft.Page) -> None:
        """Pageを設定（スレッドセーフな更新用）"""
        self._page = page

    def update_progress(self, progress: ThinkingProgress) -> None:
        """
        進捗を更新（スレッドセーフ）

        Args:
            progress: 思考進捗データ
        """
        with self._lock:
            self._update_ui(progress)

    def _update_ui(self, progress: ThinkingProgress) -> None:
        """UI要素を更新（内部用）"""
        # AIタイプ
        ai_type_names = {
            "minimax": "Minimax Search",
            "mcts": "Monte Carlo Tree Search",
            "vcf": "VCF Threat Search",
        }
        self._ai_type_text.value = ai_type_names.get(
            progress.ai_type,
            f"AI ({progress.ai_type})"
        )

        # 進捗バーと時間
        elapsed_str = f"{progress.elapsed_time:.1f}s"

        if progress.ai_type == "minimax":
            # Minimax: 深度ベース
            if progress.max_depth > 0:
                bar_value = progress.current_depth / progress.max_depth
            else:
                bar_value = 0

            self._progress_bar.value = bar_value
            self._progress_text.value = (
                f"Depth: {progress.current_depth}/{progress.max_depth} | "
                f"Nodes: {progress.nodes_visited:,} | "
                f"Time: {elapsed_str}"
            )

            # 統計
            if progress.current_best_score != 0:
                score_str = f"{progress.current_best_score:+.0f}"
            else:
                score_str = "0"
            self._stats_text.value = f"Score: {score_str}"

            # 最善手
            if progress.current_best_move:
                pos = progress.current_best_move
                self._best_move_text.value = f"Best: ({pos.x}, {pos.y})"
            else:
                self._best_move_text.value = ""

            # ログ追加
            if progress.current_depth > 0:
                self._add_log_entry(
                    f"Depth {progress.current_depth} complete",
                    f"Score: {score_str}, Nodes: {progress.nodes_visited:,}"
                )

        elif progress.ai_type == "mcts":
            # MCTS: シミュレーションベース
            if progress.total_simulations > 0:
                bar_value = progress.simulations_completed / progress.total_simulations
            else:
                bar_value = 0

            self._progress_bar.value = bar_value
            self._progress_text.value = (
                f"Simulations: {progress.simulations_completed:,}/"
                f"{progress.total_simulations:,} | "
                f"Time: {elapsed_str}"
            )

            # 上位手
            if progress.top_moves:
                top_move = progress.top_moves[0]
                pos, win_rate = top_move
                self._stats_text.value = f"Win rate: {win_rate*100:.1f}%"
                self._best_move_text.value = f"Best: ({pos.x}, {pos.y})"

                # ログ追加
                moves_str = ", ".join(
                    f"({p.x},{p.y}):{r*100:.0f}%"
                    for p, r in progress.top_moves[:3]
                )
                self._add_log_entry(
                    f"{progress.simulations_completed} simulations",
                    f"Top: {moves_str}"
                )
            else:
                self._stats_text.value = ""
                self._best_move_text.value = ""

        elif progress.ai_type == "vcf":
            # VCF: 詰み探索
            self._progress_bar.value = None  # インデターミネート

            if progress.is_forced_win:
                self._progress_text.value = f"Winning sequence found! | Time: {elapsed_str}"
                self._stats_text.value = f"Depth: {progress.vcf_depth}"

                if progress.win_sequence:
                    first_move = progress.win_sequence[0]
                    self._best_move_text.value = f"Win: ({first_move.x}, {first_move.y})"

                    # 詰み手順をログに追加
                    seq_str = " -> ".join(
                        f"({p.x},{p.y})" for p in progress.win_sequence[:5]
                    )
                    if len(progress.win_sequence) > 5:
                        seq_str += "..."
                    self._add_log_entry("VCF found!", seq_str)
            else:
                self._progress_text.value = f"Searching... Depth: {progress.vcf_depth} | Time: {elapsed_str}"
                self._stats_text.value = ""
                self._best_move_text.value = ""

        # ページ更新
        if self._page:
            try:
                self._page.update()
            except Exception:
                pass  # ページが無効な場合は無視

    def _add_log_entry(self, message: str, detail: Optional[str] = None) -> None:
        """ログエントリを追加"""
        if not self._show_detailed_log:
            return

        # テキスト作成
        if detail:
            text = ft.Text(
                f"{message}: {detail}",
                size=10,
                color=ft.Colors.GREY_600,
            )
        else:
            text = ft.Text(
                message,
                size=10,
                color=ft.Colors.GREY_600,
            )

        # 最大エントリ数を超えたら古いものを削除
        if len(self._log_column.controls) >= self._max_log_entries:
            self._log_column.controls.pop(0)

        self._log_column.controls.append(text)

    def clear(self) -> None:
        """パネルをクリア"""
        with self._lock:
            self._ai_type_text.value = "AI is thinking..."
            self._progress_bar.value = 0
            self._progress_text.value = ""
            self._stats_text.value = ""
            self._best_move_text.value = ""
            self._log_column.controls.clear()

            if self._page:
                try:
                    self._page.update()
                except Exception:
                    pass

    def set_detailed_log_visible(self, visible: bool) -> None:
        """詳細ログの表示/非表示を切り替え"""
        self._show_detailed_log = visible
        # コンテナの可視性を更新
        if len(self.content.controls) > 8:
            log_container = self.content.controls[8]
            if isinstance(log_container, ft.Container):
                log_container.visible = visible
                self._log_column.height = 100 if visible else 0

                if self._page:
                    try:
                        self._page.update()
                    except Exception:
                        pass
