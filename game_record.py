"""
Variant Go Platform - Game Record Module

棋譜（対局記録）の保存・読み込みを提供します。

用途:
- 機械学習の訓練データ
- GPT/LLMの訓練データ
- 対局のリプレイ・検証

フォーマット:
- JSON形式（1ファイル1対局）
- JSONL形式（1行1対局、大量データ向け）
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from game_core import Position, Stone, GameStatus, GameEvent


@dataclass
class MoveRecord:
    """1手の記録"""
    move_number: int          # 手数（1から開始）
    x: int                    # x座標
    y: int                    # y座標
    stone: str                # "BLACK" or "WHITE"
    timestamp: float          # ゲーム開始からの経過時間（秒）

    # AI思考情報（オプション）
    thinking_time: Optional[float] = None  # 思考時間（秒）
    eval_score: Optional[float] = None     # 評価値
    depth: Optional[int] = None            # 探索深度
    nodes: Optional[int] = None            # 探索ノード数
    pv: Optional[List[Dict[str, int]]] = None  # 読み筋（Principal Variation）


@dataclass
class PlayerRecord:
    """プレイヤー情報の記録"""
    stone: str                # "BLACK" or "WHITE"
    player_type: str          # "human", "minimax", "mcts", "vcf", etc.
    name: str                 # 表示名

    # AI設定（オプション）
    ai_depth: Optional[int] = None
    ai_simulations: Optional[int] = None
    use_iterative_deepening: bool = False
    use_vct: bool = False


@dataclass
class GameRecord:
    """1対局の記録"""
    # メタデータ
    game_id: str              # ユニークID（タイムスタンプベース）
    timestamp: str            # 対局日時（ISO 8601形式）

    # ルール情報
    rule_id: str              # "StandardGomokuRule", "GravityGomokuRule"
    board_width: int
    board_height: int
    win_condition: int        # 勝利に必要な連続数

    # プレイヤー情報
    black_player: PlayerRecord
    white_player: PlayerRecord

    # 対局結果
    result: str               # "BLACK_WIN", "WHITE_WIN", "DRAW", "ONGOING"
    total_moves: int          # 総手数
    duration: float           # 対局時間（秒）

    # 手順
    moves: List[MoveRecord] = field(default_factory=list)

    # 追加情報
    version: str = "1.0"      # レコードフォーマットバージョン
    platform: str = "Variant Go Platform"

    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """JSON文字列に変換"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @staticmethod
    def _pos_to_notation(x: int, y: int) -> str:
        """座標を棋譜表記に変換（例: 7,7 -> h8）"""
        # x: a-o (0-14), y: 1-15 (0-14)
        col = chr(ord('a') + x)
        row = str(y + 1)
        return f"{col}{row}"

    def to_simple_text(self) -> str:
        """
        シンプルなテキスト形式に変換（人間可読 + LLM訓練向け）

        例:
        # Variant Go: StandardGomokuRule 15x15
        # Black: Minimax vs White: MCTS
        # Result: BLACK_WIN (9 moves)
        1. h8
        2. c12
        3. g7
        ...
        """
        lines = []

        # ヘッダー（コメント形式）
        lines.append(f"# Variant Go: {self.rule_id} {self.board_width}x{self.board_height}")
        lines.append(f"# Black: {self.black_player.name} vs White: {self.white_player.name}")
        lines.append(f"# Result: {self.result} ({self.total_moves} moves)")
        lines.append("")

        # 手順
        for move in self.moves:
            notation = self._pos_to_notation(move.x, move.y)
            lines.append(f"{move.move_number}. {notation}")

        return "\n".join(lines)

    def to_csv(self) -> str:
        """
        CSV形式に変換（データ分析向け）

        例:
        move,x,y,col,row,stone
        1,7,7,h,8,BLACK
        2,2,11,c,12,WHITE
        """
        lines = ["move,x,y,col,row,stone"]

        for move in self.moves:
            col = chr(ord('a') + move.x)
            row = move.y + 1
            lines.append(f"{move.move_number},{move.x},{move.y},{col},{row},{move.stone}")

        return "\n".join(lines)

    def to_sequence(self) -> str:
        """
        シーケンス形式に変換（Transformer訓練向け）

        純粋なトークン列。手番は交互なので色情報は不要。
        例: h8 c12 g7 i14 f6 h6 e5 k4 d4

        先頭に結果トークンを付与することで、結果予測タスクにも使用可能:
        例: [BLACK_WIN] h8 c12 g7 ...
        """
        tokens = []

        # 結果トークン（オプション：訓練時に使い分け可能）
        tokens.append(f"[{self.result}]")

        # 手順トークン
        for move in self.moves:
            notation = self._pos_to_notation(move.x, move.y)
            tokens.append(notation)

        return " ".join(tokens)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameRecord":
        """辞書から復元"""
        # PlayerRecordを復元
        black_player = PlayerRecord(**data.pop("black_player"))
        white_player = PlayerRecord(**data.pop("white_player"))

        # MoveRecordを復元
        moves = [MoveRecord(**m) for m in data.pop("moves")]

        return cls(
            black_player=black_player,
            white_player=white_player,
            moves=moves,
            **data
        )

    @classmethod
    def from_json(cls, json_str: str) -> "GameRecord":
        """JSON文字列から復元"""
        return cls.from_dict(json.loads(json_str))


class GameRecorder:
    """
    対局を記録するクラス

    使用例:
        recorder = GameRecorder(output_dir="./game_logs")
        recorder.start_game(rule, black_player, white_player, config)

        # 各手の後に呼ぶ
        recorder.record_move(position, stone, thinking_info)

        # 対局終了時
        recorder.end_game(result)
        recorder.save()
    """

    def __init__(
        self,
        output_dir: str = "./game_logs",
        format: str = "json",  # "json" or "jsonl"
        enabled: bool = True
    ):
        """
        Args:
            output_dir: 保存先ディレクトリ
            format: 保存形式（"json": 1ファイル1対局, "jsonl": 追記形式）
            enabled: 記録を有効にするか
        """
        self._output_dir = Path(output_dir)
        self._format = format
        self._enabled = enabled

        self._current_record: Optional[GameRecord] = None
        self._game_start_time: float = 0
        self._last_move_time: float = 0

        # 出力ディレクトリを作成
        if self._enabled:
            self._output_dir.mkdir(parents=True, exist_ok=True)

    def start_game(
        self,
        rule,  # GameRule
        black_player,  # Player
        white_player,  # Player
        config = None  # GameConfig
    ) -> None:
        """対局開始を記録"""
        if not self._enabled:
            return

        import time
        self._game_start_time = time.time()
        self._last_move_time = self._game_start_time

        # ゲームIDを生成
        game_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        # プレイヤー情報を作成
        black_record = self._create_player_record(
            Stone.BLACK, black_player, config
        )
        white_record = self._create_player_record(
            Stone.WHITE, white_player, config
        )

        # レコードを作成
        self._current_record = GameRecord(
            game_id=game_id,
            timestamp=datetime.now().isoformat(),
            rule_id=rule.rule_id,
            board_width=rule.board_width,
            board_height=rule.board_height,
            win_condition=rule.win_condition,
            black_player=black_record,
            white_player=white_record,
            result="ONGOING",
            total_moves=0,
            duration=0,
            moves=[],
        )

    def _create_player_record(
        self,
        stone: Stone,
        player,  # Player
        config  # GameConfig
    ) -> PlayerRecord:
        """プレイヤー記録を作成"""
        from players import AIPlayer

        player_type = "human"
        ai_depth = None
        ai_simulations = None
        use_iterative_deepening = False
        use_vct = False

        if isinstance(player, AIPlayer):
            strategy = player.strategy
            player_type = strategy.__class__.__name__.replace("AI", "").lower()

            # MinimaxAI
            if hasattr(strategy, "_depth"):
                ai_depth = strategy._depth
            if hasattr(strategy, "_use_iterative_deepening"):
                use_iterative_deepening = strategy._use_iterative_deepening

            # MCTSAI
            if hasattr(strategy, "_simulations"):
                ai_simulations = strategy._simulations

            # VCFBasedAI
            if hasattr(strategy, "_use_vct"):
                use_vct = strategy._use_vct

        # configから取得（fallback）
        if config:
            if ai_depth is None:
                ai_depth = getattr(config, "ai_depth", None)
            if ai_simulations is None:
                ai_simulations = getattr(config, "ai_simulations", None)

        return PlayerRecord(
            stone=stone.name,
            player_type=player_type,
            name=player.name,
            ai_depth=ai_depth,
            ai_simulations=ai_simulations,
            use_iterative_deepening=use_iterative_deepening,
            use_vct=use_vct,
        )

    def record_move(
        self,
        position: Position,
        stone: Stone,
        thinking_time: Optional[float] = None,
        eval_score: Optional[float] = None,
        depth: Optional[int] = None,
        nodes: Optional[int] = None,
        pv: Optional[List[Position]] = None
    ) -> None:
        """1手を記録"""
        if not self._enabled or not self._current_record:
            return

        import time
        current_time = time.time()

        move_number = len(self._current_record.moves) + 1
        timestamp = current_time - self._game_start_time

        # 思考時間（前の手からの経過時間）
        if thinking_time is None:
            thinking_time = current_time - self._last_move_time

        self._last_move_time = current_time

        # PVを変換
        pv_list = None
        if pv:
            pv_list = [{"x": p.x, "y": p.y} for p in pv]

        move = MoveRecord(
            move_number=move_number,
            x=position.x,
            y=position.y,
            stone=stone.name,
            timestamp=timestamp,
            thinking_time=thinking_time,
            eval_score=eval_score,
            depth=depth,
            nodes=nodes,
            pv=pv_list,
        )

        self._current_record.moves.append(move)
        self._current_record.total_moves = move_number

    def end_game(self, result: GameStatus) -> None:
        """対局終了を記録"""
        if not self._enabled or not self._current_record:
            return

        import time
        self._current_record.result = result.name
        self._current_record.duration = time.time() - self._game_start_time

    def save(self) -> Optional[str]:
        """記録を保存"""
        if not self._enabled or not self._current_record:
            return None

        if self._format == "json":
            return self._save_json()
        elif self._format == "jsonl":
            return self._save_jsonl()
        elif self._format == "text":
            return self._save_text()
        elif self._format == "csv":
            return self._save_csv()
        elif self._format == "seq":
            return self._save_sequence()
        else:
            return self._save_json()

    def _save_json(self) -> str:
        """JSON形式で保存（1ファイル1対局）"""
        filename = f"game_{self._current_record.game_id}.json"
        filepath = self._output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._current_record.to_json())

        return str(filepath)

    def _save_jsonl(self) -> str:
        """JSONL形式で保存（追記）"""
        # 日付ごとにファイルを分ける
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"games_{date_str}.jsonl"
        filepath = self._output_dir / filename

        with open(filepath, "a", encoding="utf-8") as f:
            # 1行で書き出し
            json_line = json.dumps(
                self._current_record.to_dict(),
                ensure_ascii=False
            )
            f.write(json_line + "\n")

        return str(filepath)

    def _save_text(self) -> str:
        """シンプルテキスト形式で保存（1ファイル1対局）"""
        filename = f"game_{self._current_record.game_id}.txt"
        filepath = self._output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._current_record.to_simple_text())

        return str(filepath)

    def _save_csv(self) -> str:
        """CSV形式で保存（1ファイル1対局）"""
        filename = f"game_{self._current_record.game_id}.csv"
        filepath = self._output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self._current_record.to_csv())

        return str(filepath)

    def _save_sequence(self) -> str:
        """シーケンス形式で保存（追記、Transformer訓練向け）"""
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"games_{date_str}.seq"
        filepath = self._output_dir / filename

        with open(filepath, "a", encoding="utf-8") as f:
            f.write(self._current_record.to_sequence() + "\n")

        return str(filepath)

    def get_current_record(self) -> Optional[GameRecord]:
        """現在の記録を取得"""
        return self._current_record


def load_game_record(filepath: str) -> GameRecord:
    """ファイルから棋譜を読み込む"""
    with open(filepath, "r", encoding="utf-8") as f:
        return GameRecord.from_json(f.read())


def load_game_records_jsonl(filepath: str) -> List[GameRecord]:
    """JSONLファイルから複数の棋譜を読み込む"""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(GameRecord.from_json(line))
    return records
