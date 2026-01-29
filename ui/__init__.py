"""
Variant Go Platform - UI Module (Milestone 4-5)

Fletを使用したGUIコンポーネントを提供します。

コンポーネント:
- BoardComponent: 盤面表示コンポーネント
- SettingsView: 設定画面
- GameView: ゲーム画面
- ThinkingPanel: AI思考可視化パネル
"""

from ui.board_component import BoardComponent
from ui.settings_view import SettingsView
from ui.game_view import GameView
from ui.thinking_panel import ThinkingPanel

__all__ = ["BoardComponent", "SettingsView", "GameView", "ThinkingPanel"]
