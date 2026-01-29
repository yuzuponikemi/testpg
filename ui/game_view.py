"""
Variant Go Platform - Game View (Milestone 4)

ゲーム画面を提供します。
"""

import flet as ft
import asyncio
from typing import Callable, Optional

from game_core import (
    GameEngine, RuleRegistry, Stone, GameStatus, GameEvent, Position
)
from players import Player, HumanPlayer, AIPlayer, GameSession, GameSessionConfig
from ai_strategies import AIStrategyFactory, ThinkingProgress
from game_record import GameRecorder

from ui.settings_view import GameConfig
from ui.board_component import BoardComponent
from ui.thinking_panel import ThinkingPanel


class GameView(ft.View):
    """
    ゲーム画面

    盤面表示、ステータス表示、操作ボタンを提供します。
    """

    def __init__(
        self,
        page: ft.Page,
        config: GameConfig,
        on_back: Callable[[], None]
    ):
        super().__init__(route="/game")
        self._page = page
        self._config = config
        self._on_back = on_back

        # ゲームセットアップ
        self._rule = RuleRegistry.create(config.rule_id)

        # プレイヤー作成
        self._black_player = self._create_player(config.black_player_type, "Black")
        self._white_player = self._create_player(config.white_player_type, "White")

        # セッション
        self._session: Optional[GameSession] = None
        self._game_task: Optional[asyncio.Task] = None
        self._waiting_for_human = False

        # 棋譜記録
        record_enabled = getattr(config, 'record_games', False)
        record_dir = getattr(config, 'record_dir', './game_logs')
        record_format = getattr(config, 'record_format', 'json')
        self._recorder = GameRecorder(
            output_dir=record_dir,
            format=record_format,
            enabled=record_enabled,
        )

        # 初期セッションを作成
        self._create_session()

        # UI要素
        self._board_component: BoardComponent = None  # type: ignore
        self._status_text: ft.Text = None  # type: ignore
        self._turn_text: ft.Text = None  # type: ignore
        self._result_container: ft.Container = None  # type: ignore
        self._thinking_panel: ThinkingPanel = None  # type: ignore

        self._build_ui()

        # AIに進捗コールバックを設定
        self._setup_ai_callbacks()

    def _create_player(self, player_type: str, name: str) -> Player:
        """プレイヤーを作成"""
        if player_type == "human":
            return HumanPlayer(name)
        else:
            # VCF AIの場合
            if player_type == "vcf":
                from threat_search import VCFBasedAI
                from ai_strategies import RandomAI
                strategy = VCFBasedAI(
                    use_vct=getattr(self._config, 'use_vct', False),
                    fallback=RandomAI(),
                )
            else:
                strategy = AIStrategyFactory.create(
                    player_type,
                    depth=self._config.ai_depth,
                    simulations=self._config.ai_simulations,
                    use_iterative_deepening=getattr(
                        self._config, 'use_iterative_deepening', False
                    ),
                )
            return AIPlayer(strategy, f"{name} ({strategy.name})")

    def _create_session(self) -> None:
        """GameSessionを作成"""
        session_config = GameSessionConfig(
            delay_between_moves=0.3,
        )
        self._session = GameSession(
            rule=self._rule,
            black_player=self._black_player,
            white_player=self._white_player,
            config=session_config,
        )

        # 棋譜記録を開始
        self._recorder.start_game(
            self._rule,
            self._black_player,
            self._white_player,
            self._config,
        )

    def _build_ui(self) -> None:
        """UIを構築"""
        # ヘッダー
        back_button = ft.IconButton(
            icon=ft.Icons.ARROW_BACK,
            icon_color=ft.Colors.BLUE_600,
            tooltip="Back to Settings",
            on_click=self._on_back_click,
        )

        title = ft.Text(
            "Variant Go Platform",
            size=20,
            weight=ft.FontWeight.BOLD,
        )

        header = ft.Row(
            controls=[back_button, title],
            alignment=ft.MainAxisAlignment.START,
        )

        # 盤面サイズの計算
        max_board_size = 400
        board_width = self._rule.board_width
        board_height = self._rule.board_height
        cell_size = min(
            max_board_size // max(board_width, board_height),
            40
        )
        cell_size = max(cell_size, 25)

        # 盤面コンポーネント
        self._board_component = BoardComponent(
            engine=self._session.engine,
            on_cell_click=self._on_cell_click,
            cell_size=cell_size,
        )

        # ステータス表示
        self._turn_text = ft.Text(
            "Black's Turn",
            size=18,
            weight=ft.FontWeight.BOLD,
        )

        self._status_text = ft.Text(
            f"Rule: {self._rule.rule_name}",
            size=14,
            color=ft.Colors.GREY_600,
        )

        status_container = ft.Column(
            controls=[self._turn_text, self._status_text],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=5,
        )

        # 結果表示（初期は非表示）
        self._result_container = ft.Container(
            content=None,
            visible=False,
        )

        # 思考パネル（初期は非表示）
        show_thinking = getattr(self._config, 'show_thinking', True)
        self._thinking_panel = ThinkingPanel(show_detailed_log=show_thinking)
        self._thinking_panel.set_page(self._page)
        self._thinking_panel.visible = False

        # 操作ボタン
        new_game_button = ft.ElevatedButton(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.REFRESH, size=18),
                    ft.Text("New Game"),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=5,
            ),
            on_click=self._on_new_game_click,
        )

        settings_button = ft.OutlinedButton(
            content=ft.Row(
                [
                    ft.Icon(ft.Icons.SETTINGS, size=18),
                    ft.Text("Settings"),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=5,
            ),
            on_click=self._on_back_click,
        )

        button_row = ft.Row(
            controls=[new_game_button, settings_button],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=20,
        )

        # レイアウト
        content = ft.Container(
            content=ft.Column(
                controls=[
                    header,
                    ft.Divider(),
                    ft.Container(height=10),
                    self._board_component,
                    ft.Container(height=10),
                    self._thinking_panel,
                    ft.Container(height=10),
                    status_container,
                    self._result_container,
                    ft.Container(height=20),
                    button_row,
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                scroll=ft.ScrollMode.AUTO,
            ),
            padding=20,
            expand=True,
        )

        self.controls = [content]

        # イベントリスナー登録
        self._session.engine.add_listener(self._on_game_event)
        self._session.add_listener(self._on_session_event)

    def _setup_ai_callbacks(self) -> None:
        """AIプレイヤーに進捗コールバックを設定"""
        for player in [self._black_player, self._white_player]:
            if isinstance(player, AIPlayer):
                if player.strategy.supports_progress:
                    player.strategy.set_progress_callback(self._on_thinking_progress)

    def _on_thinking_progress(self, progress: ThinkingProgress) -> None:
        """AI思考進捗のコールバック（別スレッドから呼ばれる）"""
        # スレッドセーフに更新
        if self._thinking_panel and self._page:
            try:
                self._thinking_panel.update_progress(progress)
            except RuntimeError:
                pass  # セッション破棄済み

    def start_game(self) -> None:
        """ゲームを開始（外部から呼び出し用）"""
        self._start_game()

    def stop_game(self) -> None:
        """ゲームを停止"""
        if self._session:
            self._session.cancel()
        if self._game_task and not self._game_task.done():
            self._game_task.cancel()

    def _start_game(self) -> None:
        """ゲームを開始"""
        self._game_task = self._page.run_task(self._run_game_loop)

    async def _run_game_loop(self) -> None:
        """ゲームループを実行"""
        if not self._session:
            return

        try:
            # TURN_STARTイベントでターン表示が更新される
            result = await self._session.run_game()
            self._show_result(result)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Game error: {e}")
            import traceback
            traceback.print_exc()

    def _on_cell_click(self, x: int, y: int) -> None:
        """セルクリックハンドラ"""
        if not self._session or self._session.engine.is_game_over:
            return

        current_player = self._session.current_player

        # 人間プレイヤーの番の場合のみ入力を受け付ける
        if isinstance(current_player, HumanPlayer):
            # 合法手かチェック
            board = self._session.engine.board
            turn = self._session.engine.current_turn
            is_valid = self._rule.is_valid_move(board, x, y, turn)

            if is_valid:
                current_player.set_move(Position(x, y))

    def _on_game_event(self, event: GameEvent) -> None:
        """ゲームエンジンイベントハンドラ（盤面更新用）"""
        # 盤面の更新はBoardComponentが行う
        if event.event_type == "GAME_RESET":
            self._result_container.visible = False
            self._update_turn_display()
            self._safe_update()
        elif event.event_type == "MOVE_PLAYED":
            # 棋譜に記録
            if event.position and event.stone:
                self._recorder.record_move(event.position, event.stone)

    def _on_session_event(self, event) -> None:
        """セッションイベントハンドラ（ターン表示用）"""
        if event.event_type == "TURN_START":
            # ターン開始時に表示を更新
            self._update_turn_display()
        elif event.event_type == "GAME_END":
            # ゲーム終了は_run_game_loopで処理
            pass

        self._safe_update()

    def _update_turn_display(self) -> None:
        """手番表示を更新"""
        if self._session.engine.is_game_over:
            # ゲーム終了時もパネルは表示したまま（最終結果を見られる）
            return

        current = self._session.engine.current_turn
        if current == Stone.BLACK:
            player = self._black_player
            self._turn_text.value = f"Black's Turn ({player.name})"
            self._turn_text.color = ft.Colors.BLACK
        else:
            player = self._white_player
            self._turn_text.value = f"White's Turn ({player.name})"
            self._turn_text.color = ft.Colors.GREY_700

        # 人間の番なら入力待ち、AIの番なら計算中を表示
        if isinstance(player, HumanPlayer):
            self._status_text.value = "Your turn - Click on the board"
            # 思考パネルは表示したまま（前のAI思考結果を見られる）
        else:
            self._status_text.value = "AI is thinking..."
            # 思考パネルを表示してクリア（新しいAIの番の開始時のみ）
            show_thinking = getattr(self._config, 'show_thinking', True)
            self._thinking_panel.visible = show_thinking
            if show_thinking:
                self._thinking_panel.clear()

        self._safe_update()

    def _show_result(self, result: GameStatus) -> None:
        """結果を表示"""
        # 棋譜を保存
        self._recorder.end_game(result)
        saved_path = self._recorder.save()
        if saved_path:
            print(f"Game record saved: {saved_path}")

        if result == GameStatus.BLACK_WIN:
            message = "Black Wins!"
            color = ft.Colors.BLACK
        elif result == GameStatus.WHITE_WIN:
            message = "White Wins!"
            color = ft.Colors.GREY_700
        else:
            message = "Draw!"
            color = ft.Colors.BLUE_600

        self._turn_text.value = message
        self._turn_text.color = color
        self._status_text.value = "Game Over"

        # 結果パネル
        self._result_container.content = ft.Container(
            content=ft.Text(
                message,
                size=24,
                weight=ft.FontWeight.BOLD,
                color=ft.Colors.WHITE,
            ),
            bgcolor=color,
            padding=ft.padding.symmetric(horizontal=30, vertical=15),
            border_radius=10,
        )
        self._result_container.visible = True

        self._safe_update()

    def _safe_update(self) -> None:
        """ページを安全に更新（セッション破棄済みの場合は無視）"""
        if self._page:
            try:
                self._page.update()
            except RuntimeError:
                pass  # セッション破棄済み

    def _on_new_game_click(self, e: ft.ControlEvent) -> None:
        """新規ゲームボタンクリックハンドラ"""
        # 現在のゲームをキャンセル
        if self._session:
            self._session.cancel()
        if self._game_task and not self._game_task.done():
            self._game_task.cancel()

        # プレイヤーを再作成
        self._black_player = self._create_player(
            self._config.black_player_type, "Black"
        )
        self._white_player = self._create_player(
            self._config.white_player_type, "White"
        )

        # AIコールバックを設定
        self._setup_ai_callbacks()

        # 新しいセッションを作成
        self._create_session()

        # BoardComponentのエンジンを更新
        self._board_component.update_engine(self._session.engine)

        # イベントリスナーを再登録
        self._session.engine.add_listener(self._on_game_event)
        self._session.add_listener(self._on_session_event)

        # 表示をリセット
        self._result_container.visible = False
        self._thinking_panel.visible = False
        self._thinking_panel.clear()

        # 新しいゲームを開始
        self._start_game()

    def _on_back_click(self, e: ft.ControlEvent) -> None:
        """戻るボタンクリックハンドラ"""
        if self._session:
            self._session.cancel()
        if self._game_task and not self._game_task.done():
            self._game_task.cancel()

        self._on_back()
