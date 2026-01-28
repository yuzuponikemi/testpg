"""
Variant Go Platform - Core Game Logic (Milestone 1)

このモジュールは、将来の拡張（変則ルール、AI対戦、Android対応）を見越した
ボードゲームプラットフォームのコアロジックを提供します。

アーキテクチャ:
- Board: 盤面状態のみを保持する純粋なデータクラス
- GameRule (ABC): ルール定義の抽象基底クラス（Strategyパターンの基盤）
- StandardGomokuRule: 標準五目並べルールの具象実装
- GameEngine: ゲーム進行管理（Observerパターンによる状態通知）
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Optional, Protocol
import copy


class Stone(Enum):
    """盤面上の石を表す列挙型"""
    EMPTY = auto()
    BLACK = auto()
    WHITE = auto()

    def opponent(self) -> "Stone":
        """相手の石色を返す"""
        if self == Stone.BLACK:
            return Stone.WHITE
        elif self == Stone.WHITE:
            return Stone.BLACK
        return Stone.EMPTY


class GameStatus(Enum):
    """ゲームの状態を表す列挙型"""
    ONGOING = auto()      # 進行中
    BLACK_WIN = auto()    # 黒の勝利
    WHITE_WIN = auto()    # 白の勝利
    DRAW = auto()         # 引き分け（盤面が埋まった場合など）


@dataclass
class Position:
    """盤面上の座標を表す不変データクラス"""
    x: int
    y: int

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Position):
            return NotImplemented
        return self.x == other.x and self.y == other.y


@dataclass
class GameEvent:
    """
    ゲームイベントを表すデータクラス

    Observerに通知されるイベント情報を格納します。
    将来的には、イベントタイプを追加して拡張可能です。
    （例: UNDO, REDO, TIMER_UPDATE など）
    """
    event_type: str
    position: Optional[Position] = None
    stone: Optional[Stone] = None
    status: Optional[GameStatus] = None
    message: str = ""


# Observerのコールバック型定義
# 将来的にasync対応する場合は、AsyncCallable型も定義可能
GameEventCallback = Callable[[GameEvent], None]


class Board:
    """
    盤面の状態を保持する純粋なデータクラス

    責務:
    - 石の配置状態の保持
    - 座標の境界チェック
    - 石の取得・設置

    将来の拡張ポイント:
    - Milestone 3で、異なる盤面形状（六角形、トーラス型など）に対応する場合、
      このクラスを継承または別実装を作成することを想定しています。
    """

    def __init__(self, width: int, height: int) -> None:
        """
        盤面を初期化します

        Args:
            width: 盤面の横幅
            height: 盤面の縦幅
        """
        self._width = width
        self._height = height
        # 2次元リストで盤面を表現（将来的にはsparse表現も検討可能）
        self._grid: list[list[Stone]] = [
            [Stone.EMPTY for _ in range(width)] for _ in range(height)
        ]
        self._move_count = 0

    @property
    def width(self) -> int:
        """盤面の横幅"""
        return self._width

    @property
    def height(self) -> int:
        """盤面の縦幅"""
        return self._height

    @property
    def move_count(self) -> int:
        """これまでに打たれた手数"""
        return self._move_count

    def is_within_bounds(self, x: int, y: int) -> bool:
        """座標が盤面内かどうかを判定"""
        return 0 <= x < self._width and 0 <= y < self._height

    def get_stone(self, x: int, y: int) -> Stone:
        """
        指定座標の石を取得

        Args:
            x: x座標
            y: y座標

        Returns:
            その座標の石（範囲外の場合もEMPTYを返す）
        """
        if not self.is_within_bounds(x, y):
            return Stone.EMPTY
        return self._grid[y][x]

    def set_stone(self, x: int, y: int, stone: Stone) -> bool:
        """
        指定座標に石を設置（内部用）

        このメソッドはルールチェックを行いません。
        外部からはGameEngine経由で石を置いてください。

        Args:
            x: x座標
            y: y座標
            stone: 設置する石

        Returns:
            成功したらTrue、範囲外ならFalse
        """
        if not self.is_within_bounds(x, y):
            return False

        old_stone = self._grid[y][x]
        self._grid[y][x] = stone

        # 手数カウントの更新
        if old_stone == Stone.EMPTY and stone != Stone.EMPTY:
            self._move_count += 1
        elif old_stone != Stone.EMPTY and stone == Stone.EMPTY:
            self._move_count -= 1

        return True

    def is_empty(self, x: int, y: int) -> bool:
        """指定座標が空かどうかを判定"""
        return self.get_stone(x, y) == Stone.EMPTY

    def is_full(self) -> bool:
        """盤面が全て埋まっているかを判定"""
        return self._move_count >= self._width * self._height

    def copy(self) -> "Board":
        """盤面のディープコピーを作成"""
        new_board = Board(self._width, self._height)
        new_board._grid = copy.deepcopy(self._grid)
        new_board._move_count = self._move_count
        return new_board

    def clear(self) -> None:
        """盤面をクリア"""
        self._grid = [
            [Stone.EMPTY for _ in range(self._width)] for _ in range(self._height)
        ]
        self._move_count = 0


class GameRule(ABC):
    """
    ゲームルールの抽象基底クラス（Strategyパターンの基盤）

    このクラスは将来のMilestone 3で、様々な変則ルールを
    プラグインのように差し替え可能にするための基盤です。

    拡張例:
    - GravityGomokuRule: 重力付き五目並べ（石が下に落ちる）
    - ReversiGomokuRule: オセロ風五目並べ（挟んだ石を取る）
    - Connect6Rule: 六目並べ（黒1手目以降は2手ずつ）
    - ForbiddenMoveRule: 禁じ手ありルール（三三、四四、長連禁止）
    """

    @property
    @abstractmethod
    def board_width(self) -> int:
        """盤面の横幅を返す"""
        pass

    @property
    @abstractmethod
    def board_height(self) -> int:
        """盤面の縦幅を返す"""
        pass

    @property
    @abstractmethod
    def win_condition(self) -> int:
        """勝利に必要な連続数を返す"""
        pass

    @property
    @abstractmethod
    def rule_name(self) -> str:
        """ルール名を返す（UI表示用）"""
        pass

    @abstractmethod
    def is_valid_move(self, board: Board, x: int, y: int, stone: Stone) -> bool:
        """
        指定の手が合法かどうかを判定

        将来の拡張ポイント:
        - 禁じ手ルールの実装時にオーバーライド
        - 重力ルールでは、下に空きがある場合はFalseを返すなど

        Args:
            board: 現在の盤面
            x: x座標
            y: y座標
            stone: 置こうとしている石

        Returns:
            合法手ならTrue
        """
        pass

    @abstractmethod
    def check_winner(self, board: Board, last_x: int, last_y: int, last_stone: Stone) -> GameStatus:
        """
        最後に置かれた石を起点に勝敗を判定

        将来の拡張ポイント:
        - オセロ風ルールでは、石数のカウントなど異なるロジックに
        - 特殊勝利条件（盤面全体の形状など）もここでチェック

        Args:
            board: 現在の盤面
            last_x: 最後に置かれたx座標
            last_y: 最後に置かれたy座標
            last_stone: 最後に置かれた石

        Returns:
            GameStatus（ONGOING, BLACK_WIN, WHITE_WIN, DRAW）
        """
        pass

    def create_board(self) -> Board:
        """このルール用の盤面を生成"""
        return Board(self.board_width, self.board_height)

    def get_valid_moves(self, board: Board, stone: Stone) -> list[Position]:
        """
        現在の盤面で合法な全ての手を返す

        将来の拡張ポイント:
        - AI実装（Milestone 2）で使用
        - 効率化が必要な場合はサブクラスでオーバーライド

        Args:
            board: 現在の盤面
            stone: 次に置く石

        Returns:
            合法手の座標リスト
        """
        valid_moves: list[Position] = []
        for y in range(board.height):
            for x in range(board.width):
                if self.is_valid_move(board, x, y, stone):
                    valid_moves.append(Position(x, y))
        return valid_moves

    # === Milestone 3: 新規メソッド（プラグインシステム対応） ===

    @property
    def rule_id(self) -> str:
        """
        ルールの一意識別子

        RuleRegistryでの登録・取得、設定のシリアライズに使用します。
        デフォルトではクラス名を返しますが、サブクラスでオーバーライド可能です。

        Returns:
            ルール識別子文字列
        """
        return self.__class__.__name__

    def apply_move_effects(
        self,
        board: Board,
        x: int,
        y: int,
        stone: Stone
    ) -> list[Position]:
        """
        石を置いた後の副作用を適用

        このメソッドはplay_move内で石を置いた直後に呼ばれます。
        重力落下、石の捕獲などのルール固有の副作用を処理します。

        Args:
            board: 現在の盤面（石は既に置かれている）
            x: 置いた石のx座標
            y: 置いた石のy座標
            stone: 置いた石の色

        Returns:
            移動・変化した座標のリスト（イベント通知用）
            - 空リストの場合は副作用なし
            - 重力の場合: 最終位置を含むリスト
            - 捕獲の場合: 捕獲された石の位置リスト

        デフォルト実装: 何もしない（空リストを返す）
        """
        return []

    def get_rule_config(self) -> dict:
        """
        ルール設定をdict形式でシリアライズ

        保存・復元用。from_config()と対になります。
        サブクラスでカスタムパラメータがある場合はオーバーライドしてください。

        Returns:
            設定dict。最低限 {"rule_id": "..."} を含む
        """
        return {
            "rule_id": self.rule_id,
        }

    @classmethod
    def from_config(cls, config: dict) -> "GameRule":
        """
        設定dictからルールインスタンスを復元

        RuleRegistry.create_from_config()から呼ばれます。
        サブクラスでカスタムパラメータがある場合はオーバーライドしてください。

        Args:
            config: get_rule_config()で生成されたdict

        Returns:
            復元されたGameRuleインスタンス

        デフォルト実装: 引数なしでインスタンス化
        """
        return cls()


class StandardGomokuRule(GameRule):
    """
    標準的な五目並べルール

    - 盤面: 15x15
    - 勝利条件: 縦・横・斜めに5つ連続で並べる
    - 禁じ手: なし（6つ以上並んでも勝ち）
    """

    @property
    def board_width(self) -> int:
        return 15

    @property
    def board_height(self) -> int:
        return 15

    @property
    def win_condition(self) -> int:
        return 5

    @property
    def rule_name(self) -> str:
        return "Standard Gomoku (15x15)"

    def is_valid_move(self, board: Board, x: int, y: int, stone: Stone) -> bool:
        """
        合法手判定（標準ルール）

        標準ルールでは以下をチェック:
        1. 座標が盤面内である
        2. その座標が空である

        将来の禁じ手ルール実装時は、このメソッドをオーバーライドして
        三三、四四、長連のチェックを追加します。
        """
        if not board.is_within_bounds(x, y):
            return False
        if not board.is_empty(x, y):
            return False
        return True

    def check_winner(self, board: Board, last_x: int, last_y: int, last_stone: Stone) -> GameStatus:
        """
        勝敗判定（標準ルール）

        最後に置かれた石を起点に、8方向をチェックして
        5つ以上連続しているかを確認します。
        """
        if last_stone == Stone.EMPTY:
            return GameStatus.ONGOING

        # 4つの軸（横、縦、右斜め、左斜め）をチェック
        directions = [
            [(1, 0), (-1, 0)],   # 横
            [(0, 1), (0, -1)],   # 縦
            [(1, 1), (-1, -1)],  # 右斜め下
            [(1, -1), (-1, 1)],  # 右斜め上
        ]

        for axis in directions:
            count = 1  # 起点の石をカウント

            for dx, dy in axis:
                nx, ny = last_x + dx, last_y + dy
                while board.get_stone(nx, ny) == last_stone:
                    count += 1
                    nx += dx
                    ny += dy

            if count >= self.win_condition:
                if last_stone == Stone.BLACK:
                    return GameStatus.BLACK_WIN
                else:
                    return GameStatus.WHITE_WIN

        # 盤面が埋まったら引き分け
        if board.is_full():
            return GameStatus.DRAW

        return GameStatus.ONGOING


class GravityGomokuRule(GameRule):
    """
    重力付き五目並べ（Connect Four風）

    石が下に落ちるルール。
    - デフォルト盤面: 7x6
    - 勝利条件: 縦・横・斜めに4つ連続で並べる

    プレイヤーは列を選択するだけで、石は自動的にその列の一番下の空きマスに落ちます。
    """

    def __init__(
        self,
        width: int = 7,
        height: int = 6,
        win_condition: int = 4
    ) -> None:
        """
        重力ルールを初期化

        Args:
            width: 盤面の横幅（列数）
            height: 盤面の縦幅（行数）
            win_condition: 勝利に必要な連続数
        """
        self._width = width
        self._height = height
        self._win_condition = win_condition

    @property
    def board_width(self) -> int:
        return self._width

    @property
    def board_height(self) -> int:
        return self._height

    @property
    def win_condition(self) -> int:
        return self._win_condition

    @property
    def rule_name(self) -> str:
        return f"Gravity Gomoku ({self._width}x{self._height}, {self._win_condition}-in-a-row)"

    def _find_landing_y(self, board: Board, x: int) -> Optional[int]:
        """
        指定した列で石が落ちる位置を探す

        Args:
            board: 現在の盤面
            x: 列番号

        Returns:
            着地するy座標。列が満杯ならNone
        """
        if not (0 <= x < board.width):
            return None

        # 下から上に探索して、最初の空きマスを見つける
        for y in range(board.height - 1, -1, -1):
            if board.is_empty(x, y):
                return y

        return None  # 列が満杯

    def is_valid_move(self, board: Board, x: int, y: int, stone: Stone) -> bool:
        """
        合法手判定（重力ルール）

        重力ルールでは、y座標は無視され、列が満杯でなければ有効。
        実際の着地位置は apply_move_effects で決定されます。

        Args:
            board: 現在の盤面
            x: 列番号
            y: 行番号（重力ルールでは無視される）
            stone: 置く石

        Returns:
            その列に石を落とせるならTrue
        """
        # 盤面の幅チェック
        if not (0 <= x < board.width):
            return False

        # 列が満杯でないことを確認
        return self._find_landing_y(board, x) is not None

    def apply_move_effects(
        self,
        board: Board,
        x: int,
        y: int,
        stone: Stone
    ) -> list[Position]:
        """
        石を下に落とす

        このメソッドは石が置かれた直後に呼ばれます。
        指定座標の石を、その列の最下段の空きマスに移動させます。

        Args:
            board: 現在の盤面
            x: 置いた石のx座標
            y: 置いた石のy座標（元の位置）
            stone: 置いた石

        Returns:
            移動した場合は最終位置を含むリスト、移動しなかった場合は空リスト
        """
        # 現在の位置から、その列で石が落ちるべき位置を探す
        # 注: 石は既に(x, y)に置かれているので、(x, y)自体も空ではない
        # 現在の石を一時的に除いて探索

        # まず現在位置の石を取得（確認用）
        current = board.get_stone(x, y)
        if current == Stone.EMPTY:
            # 石がない（おかしい状態）
            return []

        # 石を一時的に除去
        board.set_stone(x, y, Stone.EMPTY)

        # 落下先を探す
        landing_y = self._find_landing_y(board, x)

        if landing_y is None:
            # あり得ないはずだが、念のため元に戻す
            board.set_stone(x, y, stone)
            return []

        # 落下先に石を置く
        board.set_stone(x, landing_y, stone)

        # 移動があった場合のみ位置を返す
        if landing_y != y:
            return [Position(x, landing_y)]

        return []

    def check_winner(
        self,
        board: Board,
        last_x: int,
        last_y: int,
        last_stone: Stone
    ) -> GameStatus:
        """
        勝敗判定（重力ルール）

        最後に置かれた石を起点に、4方向をチェックして
        指定数以上連続しているかを確認します。

        StandardGomokuRuleと同じロジックですが、win_conditionが異なります。
        """
        if last_stone == Stone.EMPTY:
            return GameStatus.ONGOING

        # 4つの軸（横、縦、右斜め下、右斜め上）をチェック
        directions = [
            [(1, 0), (-1, 0)],   # 横
            [(0, 1), (0, -1)],   # 縦
            [(1, 1), (-1, -1)],  # 右斜め下
            [(1, -1), (-1, 1)],  # 右斜め上
        ]

        for axis in directions:
            count = 1  # 起点の石をカウント

            for dx, dy in axis:
                nx, ny = last_x + dx, last_y + dy
                while board.get_stone(nx, ny) == last_stone:
                    count += 1
                    nx += dx
                    ny += dy

            if count >= self._win_condition:
                if last_stone == Stone.BLACK:
                    return GameStatus.BLACK_WIN
                else:
                    return GameStatus.WHITE_WIN

        # 盤面が埋まったら引き分け
        if board.is_full():
            return GameStatus.DRAW

        return GameStatus.ONGOING

    def get_valid_moves(self, board: Board, stone: Stone) -> list[Position]:
        """
        合法手一覧（重力ルール用に最適化）

        重力ルールでは、各列の一番上の行（y=0）のみをチェックすれば良い。
        返される座標のy=0は「その列に落とす」という意味になります。

        Args:
            board: 現在の盤面
            stone: 次に置く石

        Returns:
            有効な列の座標リスト（y=0として返す）
        """
        valid_moves: list[Position] = []
        for x in range(board.width):
            if self._find_landing_y(board, x) is not None:
                # 列が空いていれば、その列は有効
                # y=0として返す（実際の着地はapply_move_effectsで決まる）
                valid_moves.append(Position(x, 0))
        return valid_moves

    def get_rule_config(self) -> dict:
        """ルール設定をシリアライズ"""
        return {
            "rule_id": self.rule_id,
            "width": self._width,
            "height": self._height,
            "win_condition": self._win_condition,
        }

    @classmethod
    def from_config(cls, config: dict) -> "GravityGomokuRule":
        """設定からルールを復元"""
        return cls(
            width=config.get("width", 7),
            height=config.get("height", 6),
            win_condition=config.get("win_condition", 4),
        )


class RuleRegistry:
    """
    ゲームルールのレジストリ

    ルールをプラグインのように登録・取得できます。
    rule_idをキーとして管理します。

    使用例:
        # ルールの登録
        RuleRegistry.register(StandardGomokuRule)
        RuleRegistry.register(GravityGomokuRule)

        # ルールの取得
        rule_class = RuleRegistry.get("StandardGomokuRule")
        rule = rule_class()

        # 設定からの復元
        config = {"rule_id": "GravityGomokuRule", "width": 10, "height": 8}
        rule = RuleRegistry.create_from_config(config)

        # 利用可能なルール一覧
        available = RuleRegistry.list_available()
    """

    _registry: dict[str, type[GameRule]] = {}

    @classmethod
    def register(cls, rule_class: type[GameRule]) -> None:
        """
        ルールクラスを登録

        Args:
            rule_class: 登録するGameRuleのサブクラス

        Raises:
            ValueError: 同じrule_idが既に登録されている場合
            TypeError: GameRuleのサブクラスでない場合
        """
        if not isinstance(rule_class, type) or not issubclass(rule_class, GameRule):
            raise TypeError(f"{rule_class} is not a subclass of GameRule")

        # インスタンスを作ってrule_idを取得
        instance = rule_class()
        rule_id = instance.rule_id

        if rule_id in cls._registry:
            raise ValueError(f"Rule '{rule_id}' is already registered")

        cls._registry[rule_id] = rule_class

    @classmethod
    def get(cls, rule_id: str) -> type[GameRule]:
        """
        rule_idからルールクラスを取得

        Args:
            rule_id: ルールの識別子

        Returns:
            ルールクラス

        Raises:
            KeyError: 登録されていないrule_idの場合
        """
        if rule_id not in cls._registry:
            raise KeyError(f"Rule '{rule_id}' is not registered")

        return cls._registry[rule_id]

    @classmethod
    def create(cls, rule_id: str, **kwargs) -> GameRule:
        """
        rule_idからルールインスタンスを作成

        Args:
            rule_id: ルールの識別子
            **kwargs: ルールのコンストラクタ引数

        Returns:
            ルールインスタンス
        """
        rule_class = cls.get(rule_id)
        return rule_class(**kwargs)

    @classmethod
    def create_from_config(cls, config: dict) -> GameRule:
        """
        設定dictからルールインスタンスを復元

        Args:
            config: get_rule_config()で生成されたdict

        Returns:
            復元されたルールインスタンス

        Raises:
            KeyError: rule_idが登録されていない場合
            KeyError: configにrule_idがない場合
        """
        if "rule_id" not in config:
            raise KeyError("config must contain 'rule_id'")

        rule_id = config["rule_id"]
        rule_class = cls.get(rule_id)

        return rule_class.from_config(config)

    @classmethod
    def list_available(cls) -> list[str]:
        """
        登録されているrule_idの一覧を返す

        Returns:
            rule_idのリスト（アルファベット順）
        """
        return sorted(cls._registry.keys())

    @classmethod
    def is_registered(cls, rule_id: str) -> bool:
        """rule_idが登録されているかチェック"""
        return rule_id in cls._registry

    @classmethod
    def unregister(cls, rule_id: str) -> bool:
        """
        ルールの登録を解除（主にテスト用）

        Returns:
            解除できたらTrue、存在しなければFalse
        """
        if rule_id in cls._registry:
            del cls._registry[rule_id]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """全ての登録をクリア（主にテスト用）"""
        cls._registry.clear()


class GameEngine:
    """
    ゲーム進行を管理するコントローラ/ファサードクラス

    責務:
    - 手番管理
    - ルールと盤面の連携
    - Observerパターンによる状態変化の通知

    外部（GUIやAI）からは、このクラスのメソッドのみを通じて
    ゲームを操作します。

    将来の拡張ポイント:
    - Milestone 2: AIプレイヤーの統合
    - Milestone 3: Undo/Redo機能の追加
    - Milestone 4: 非同期対応（async/await）でUI応答性向上
    """

    def __init__(self, rule: GameRule) -> None:
        """
        ゲームエンジンを初期化

        Args:
            rule: 使用するゲームルール
        """
        self._rule = rule
        self._board = rule.create_board()
        self._current_turn = Stone.BLACK  # 黒先手
        self._status = GameStatus.ONGOING
        self._listeners: list[GameEventCallback] = []
        self._move_history: list[tuple[Position, Stone]] = []  # Undo用の履歴

    @property
    def board(self) -> Board:
        """現在の盤面（読み取り専用のコピーを返す）"""
        return self._board.copy()

    @property
    def current_turn(self) -> Stone:
        """現在の手番"""
        return self._current_turn

    @property
    def status(self) -> GameStatus:
        """ゲームの状態"""
        return self._status

    @property
    def rule(self) -> GameRule:
        """使用中のルール"""
        return self._rule

    @property
    def move_history(self) -> list[tuple[Position, Stone]]:
        """着手履歴のコピー"""
        return list(self._move_history)

    @property
    def is_game_over(self) -> bool:
        """ゲームが終了しているか"""
        return self._status != GameStatus.ONGOING

    def add_listener(self, callback: GameEventCallback) -> None:
        """
        イベントリスナーを登録（Observerパターン）

        登録されたコールバックは、以下のイベント発生時に呼ばれます:
        - MOVE_PLAYED: 石が置かれた時
        - GAME_OVER: ゲームが終了した時
        - GAME_RESET: ゲームがリセットされた時

        将来の拡張ポイント:
        - イベントタイプでフィルタリングする機能
        - 優先度付きリスナー
        - 非同期コールバック対応

        Args:
            callback: イベント発生時に呼ばれるコールバック関数
        """
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_listener(self, callback: GameEventCallback) -> bool:
        """
        イベントリスナーを解除

        Args:
            callback: 解除するコールバック関数

        Returns:
            解除できたらTrue、存在しなければFalse
        """
        if callback in self._listeners:
            self._listeners.remove(callback)
            return True
        return False

    def _notify_listeners(self, event: GameEvent) -> None:
        """全リスナーにイベントを通知"""
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as e:
                # リスナーの例外がゲームロジックに影響しないようにする
                # 将来的にはロギング機構を追加
                print(f"Listener error: {e}")

    def play_move(self, x: int, y: int) -> bool:
        """
        現在の手番のプレイヤーが指定座標に石を置く

        この関数がGUIやAIから呼ばれる主要なインターフェースです。

        Args:
            x: x座標
            y: y座標

        Returns:
            成功したらTrue、失敗（不正な手、ゲーム終了済み）ならFalse
        """
        # ゲーム終了後は打てない
        if self.is_game_over:
            return False

        # ルールによる合法性チェック
        if not self._rule.is_valid_move(self._board, x, y, self._current_turn):
            return False

        # 石を置く
        stone = self._current_turn
        self._board.set_stone(x, y, stone)

        # MOVE_PLAYEDイベントを通知（元の位置）
        move_event = GameEvent(
            event_type="MOVE_PLAYED",
            position=Position(x, y),
            stone=stone,
            status=self._status,
            message=f"{stone.name} played at ({x}, {y})"
        )
        self._notify_listeners(move_event)

        # ルール固有の副作用を適用（重力落下、捕獲など）
        affected = self._rule.apply_move_effects(self._board, x, y, stone)

        # 最終位置を決定（副作用で移動した場合は最終位置を使用）
        if affected:
            final_x, final_y = affected[-1].x, affected[-1].y
        else:
            final_x, final_y = x, y

        # 履歴には最終位置を記録
        self._move_history.append((Position(final_x, final_y), stone))

        # STONE_MOVEDイベントを通知（移動があった場合）
        for pos in affected:
            moved_event = GameEvent(
                event_type="STONE_MOVED",
                position=pos,
                stone=stone,
                status=self._status,
                message=f"{stone.name} moved to ({pos.x}, {pos.y})"
            )
            self._notify_listeners(moved_event)

        # 勝敗判定（最終位置で判定）
        self._status = self._rule.check_winner(self._board, final_x, final_y, stone)

        if self.is_game_over:
            # GAME_OVERイベントを通知
            game_over_event = GameEvent(
                event_type="GAME_OVER",
                position=Position(final_x, final_y),
                stone=stone,
                status=self._status,
                message=f"Game over: {self._status.name}"
            )
            self._notify_listeners(game_over_event)
        else:
            # 手番交代
            self._current_turn = self._current_turn.opponent()

        return True

    def reset(self) -> None:
        """
        ゲームをリセットして初期状態に戻す
        """
        self._board.clear()
        self._current_turn = Stone.BLACK
        self._status = GameStatus.ONGOING
        self._move_history.clear()

        # GAME_RESETイベントを通知
        reset_event = GameEvent(
            event_type="GAME_RESET",
            status=GameStatus.ONGOING,
            message="Game has been reset"
        )
        self._notify_listeners(reset_event)

    def get_stone_at(self, x: int, y: int) -> Stone:
        """
        指定座標の石を取得（読み取り専用アクセス）

        Args:
            x: x座標
            y: y座標

        Returns:
            その座標の石
        """
        return self._board.get_stone(x, y)

    def get_valid_moves(self) -> list[Position]:
        """
        現在の手番で打てる全ての合法手を取得

        将来的にAI実装（Milestone 2）で使用します。

        Returns:
            合法手の座標リスト
        """
        return self._rule.get_valid_moves(self._board, self._current_turn)


# === デフォルトルールの登録 ===
# ルールクラスをRuleRegistryに自動登録
RuleRegistry.register(StandardGomokuRule)
RuleRegistry.register(GravityGomokuRule)
