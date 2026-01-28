"""
Variant Go Platform - Settings View (Milestone 4)

ゲーム設定画面を提供します。
"""

import flet as ft
from dataclasses import dataclass
from typing import Callable

from game_core import RuleRegistry


@dataclass
class GameConfig:
    """ゲーム設定"""
    rule_id: str
    black_player_type: str
    white_player_type: str
    ai_depth: int = 3
    ai_simulations: int = 500


class SettingsView(ft.View):
    """
    設定画面

    ルール選択、プレイヤー設定を行い、ゲーム開始を呼び出します。

    使用例:
        def go_to_game(config):
            # ゲーム画面に遷移
            pass

        settings = SettingsView(page, go_to_game)
    """

    def __init__(
        self,
        page: ft.Page,
        on_start_game: Callable[[GameConfig], None]
    ):
        """
        Args:
            page: Fletページ
            on_start_game: ゲーム開始時のコールバック
        """
        super().__init__(route="/settings")
        self._page = page
        self._on_start_game = on_start_game

        # 利用可能なルール
        self._available_rules = RuleRegistry.list_available()

        # プレイヤータイプ
        self._player_types = [
            ("human", "Human"),
            ("random", "AI: Random (Easy)"),
            ("minimax", "AI: Minimax (Medium)"),
            ("mcts", "AI: MCTS (Hard)"),
        ]

        # UI要素
        self._rule_dropdown: ft.Dropdown = None  # type: ignore
        self._black_dropdown: ft.Dropdown = None  # type: ignore
        self._white_dropdown: ft.Dropdown = None  # type: ignore
        self._ai_depth_slider: ft.Slider = None  # type: ignore
        self._ai_depth_text: ft.Text = None  # type: ignore

        self._build_ui()

    def _build_ui(self) -> None:
        """UIを構築"""
        # タイトル
        title = ft.Text(
            "Variant Go Platform",
            size=28,
            weight=ft.FontWeight.BOLD,
            text_align=ft.TextAlign.CENTER,
        )

        subtitle = ft.Text(
            "Game Settings",
            size=16,
            color=ft.Colors.GREY_600,
            text_align=ft.TextAlign.CENTER,
        )

        # ルール選択
        rule_options = []
        for rule_id in self._available_rules:
            # ルールインスタンスを作成して表示名を取得
            rule = RuleRegistry.create(rule_id)
            rule_options.append(
                ft.dropdown.Option(key=rule_id, text=rule.rule_name)
            )

        self._rule_dropdown = ft.Dropdown(
            label="Game Rule",
            options=rule_options,
            value=self._available_rules[0] if self._available_rules else None,
            width=350,
        )

        # 黒プレイヤー選択
        player_options = [
            ft.dropdown.Option(key=key, text=label)
            for key, label in self._player_types
        ]

        self._black_dropdown = ft.Dropdown(
            label="Black Player (First)",
            options=player_options.copy(),
            value="human",
            width=350,
        )

        # 白プレイヤー選択
        self._white_dropdown = ft.Dropdown(
            label="White Player (Second)",
            options=[
                ft.dropdown.Option(key=key, text=label)
                for key, label in self._player_types
            ],
            value="minimax",
            width=350,
        )

        # AI深度設定
        self._ai_depth_text = ft.Text("AI Depth: 3", size=14)
        self._ai_depth_slider = ft.Slider(
            min=1,
            max=5,
            divisions=4,
            value=3,
            label="{value}",
            on_change=self._on_depth_change,
            width=350,
        )

        ai_settings = ft.Column(
            controls=[
                ft.Divider(),
                ft.Text("AI Settings", size=16, weight=ft.FontWeight.BOLD),
                self._ai_depth_text,
                self._ai_depth_slider,
            ],
            spacing=10,
        )

        # ゲーム開始ボタン
        start_button = ft.ElevatedButton(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.PLAY_ARROW, color=ft.Colors.WHITE),
                    ft.Text("Start Game", color=ft.Colors.WHITE),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=8,
            ),
            on_click=self._on_start_click,
            width=200,
            height=50,
            bgcolor=ft.Colors.BLUE_600,
        )

        # レイアウト
        content = ft.Container(
            content=ft.Column(
                controls=[
                    title,
                    subtitle,
                    ft.Container(height=30),
                    self._rule_dropdown,
                    ft.Container(height=20),
                    self._black_dropdown,
                    ft.Container(height=10),
                    self._white_dropdown,
                    ai_settings,
                    ft.Container(height=30),
                    start_button,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=5,
            ),
            padding=30,
            alignment=ft.Alignment(0, 0),
            expand=True,
        )

        self.controls = [content]
        self.vertical_alignment = ft.MainAxisAlignment.CENTER
        self.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    def _on_depth_change(self, e: ft.ControlEvent) -> None:
        """深度スライダー変更ハンドラ"""
        depth = int(e.control.value)
        self._ai_depth_text.value = f"AI Depth: {depth}"
        self._page.update()

    def _on_start_click(self, e: ft.ControlEvent) -> None:
        """ゲーム開始ボタンクリックハンドラ"""
        config = GameConfig(
            rule_id=self._rule_dropdown.value or self._available_rules[0],
            black_player_type=self._black_dropdown.value or "human",
            white_player_type=self._white_dropdown.value or "minimax",
            ai_depth=int(self._ai_depth_slider.value),
            ai_simulations=500,
        )

        self._on_start_game(config)
