"""
Variant Go Platform - Board Component (Milestone 4)

盤面表示とタップ操作を提供するFletコンポーネント。
"""

import flet as ft
from typing import Callable, Optional

from game_core import GameEngine, Stone, GameEvent


class BoardComponent(ft.Container):
    """
    盤面表示コンポーネント

    GameEngineの状態を表示し、セルクリック時にコールバックを呼びます。
    """

    def __init__(
        self,
        engine: GameEngine,
        on_cell_click: Optional[Callable[[int, int], None]] = None,
        cell_size: int = 40,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._engine = engine
        self._on_cell_click = on_cell_click
        self._cell_size = cell_size
        self._cells: list[list[ft.Container]] = []
        self._click_enabled = True
        self._last_move: Optional[tuple[int, int]] = None

        self._width = engine.rule.board_width
        self._height = engine.rule.board_height

        self._build_board()
        engine.add_listener(self._on_game_event)

    def _build_board(self) -> None:
        """盤面UIを構築"""
        rows = []
        self._cells = []

        for y in range(self._height):
            row_cells: list[ft.Container] = []
            row_containers = []

            for x in range(self._width):
                cell = self._create_cell(x, y)
                row_cells.append(cell)
                row_containers.append(cell)

            self._cells.append(row_cells)
            rows.append(ft.Row(
                controls=row_containers,
                spacing=1,
                alignment=ft.MainAxisAlignment.CENTER
            ))

        self.content = ft.Column(
            controls=rows,
            spacing=1,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        )

        board_width = self._width * (self._cell_size + 1)
        board_height = self._height * (self._cell_size + 1)
        self.width = board_width
        self.height = board_height
        self.bgcolor = "#8B7355"
        self.border_radius = 5
        self.padding = 5

    def _create_cell(self, x: int, y: int) -> ft.Container:
        """1つのセルを作成"""
        stone = self._engine.get_stone_at(x, y)
        cell_content = self._get_stone_content(stone)

        cell = ft.Container(
            content=cell_content,
            width=self._cell_size,
            height=self._cell_size,
            bgcolor="#DEB887",
            border_radius=3,
            alignment=ft.Alignment(0, 0),
            on_click=lambda e, x=x, y=y: self._handle_click(x, y),
        )

        return cell

    def _get_stone_content(self, stone: Stone) -> Optional[ft.Container]:
        """石の表示コンテンツを取得"""
        if stone == Stone.EMPTY:
            return None

        stone_size = int(self._cell_size * 0.8)

        if stone == Stone.BLACK:
            return ft.Container(
                width=stone_size,
                height=stone_size,
                bgcolor="#1a1a1a",
                border_radius=stone_size // 2,
                shadow=ft.BoxShadow(
                    spread_radius=1,
                    blur_radius=3,
                    color=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                    offset=ft.Offset(2, 2),
                ),
            )
        else:  # WHITE
            return ft.Container(
                width=stone_size,
                height=stone_size,
                bgcolor="#f5f5f5",
                border_radius=stone_size // 2,
                border=ft.border.all(1, "#cccccc"),
                shadow=ft.BoxShadow(
                    spread_radius=1,
                    blur_radius=3,
                    color=ft.Colors.with_opacity(0.3, ft.Colors.BLACK),
                    offset=ft.Offset(2, 2),
                ),
            )

    def _handle_click(self, x: int, y: int) -> None:
        """セルクリックハンドラ"""
        if not self._click_enabled:
            return

        if self._on_cell_click:
            self._on_cell_click(x, y)

    def _on_game_event(self, event: GameEvent) -> None:
        """ゲームイベントハンドラ"""
        if event.event_type in ["MOVE_PLAYED", "STONE_MOVED", "GAME_RESET"]:
            if event.event_type == "MOVE_PLAYED" and event.position:
                self._last_move = (event.position.x, event.position.y)
            elif event.event_type == "GAME_RESET":
                self._last_move = None
            self.refresh_board()

    def refresh_board(self) -> None:
        """盤面表示を更新"""
        for y in range(self._height):
            for x in range(self._width):
                stone = self._engine.get_stone_at(x, y)
                self._cells[y][x].content = self._get_stone_content(stone)

                # 最後の手をハイライト
                if self._last_move and (x, y) == self._last_move:
                    self._cells[y][x].border = ft.border.all(2, "#ff6b6b")
                else:
                    self._cells[y][x].border = None

        if self.page:
            self.update()

    def set_click_enabled(self, enabled: bool) -> None:
        """クリックの有効/無効を設定"""
        self._click_enabled = enabled

    def set_cell_size(self, size: int) -> None:
        """セルサイズを変更"""
        self._cell_size = size
        self._build_board()
        self.refresh_board()

    @property
    def engine(self) -> GameEngine:
        """ゲームエンジン"""
        return self._engine

    def update_engine(self, engine: GameEngine) -> None:
        """エンジンを更新"""
        self._engine.remove_listener(self._on_game_event)

        self._engine = engine
        self._width = engine.rule.board_width
        self._height = engine.rule.board_height
        self._last_move = None

        engine.add_listener(self._on_game_event)

        self._build_board()
        self.refresh_board()
