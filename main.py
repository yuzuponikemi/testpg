"""
Variant Go Platform - Main Application (Milestone 4)

Fletアプリのエントリーポイント。

実行方法:
    flet run main.py

または:
    python main.py
"""

import flet as ft

from ui.settings_view import SettingsView, GameConfig
from ui.game_view import GameView


def main(page: ft.Page) -> None:
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

    # 初期画面は設定画面
    go_to_settings()


if __name__ == "__main__":
    ft.run(main)
