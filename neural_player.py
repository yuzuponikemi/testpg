"""
NeuralPlayer - 学習済みTransformerモデルを使用するAIプレイヤー

GomokuGPTモデルを読み込み、次の一手を予測してプレイします。
合法手のみを選択するようにフィルタリングを行います。

使用例:
    from neural_player import NeuralPlayer

    player = NeuralPlayer("checkpoints/gomoku_gpt.pth")
    move = player.select_move(board, rule, Stone.BLACK)
"""

import time
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn.functional as F

from game_core import Board, GameRule, Position, Stone, GameStatus
from ai_strategies import AIStrategy, ThinkingProgress


class NeuralPlayer(AIStrategy):
    """
    学習済みTransformerモデルを使用するAIプレイヤー

    GomokuGPTモデルから次の一手の確率分布を取得し、
    合法手の中から着手を選択します。
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.5,
        top_k: Optional[int] = 10,
        device: Optional[str] = None
    ):
        """
        Args:
            model_path: 学習済みモデルのパス (.pth ファイル)
            temperature: サンプリング温度
                - 0.0: 常に最高確率の手を選択（決定的）
                - 1.0: 確率分布に従ってサンプリング（多様性あり）
                - 0.5: その中間（推奨）
            top_k: 上位k個の候補からサンプリング（Noneなら全体から）
            device: 使用デバイス（"cuda", "mps", "cpu"、Noneなら自動選択）
        """
        super().__init__()
        self._model_path = model_path
        self._temperature = temperature
        self._top_k = top_k

        # デバイス設定
        if device:
            self._device = torch.device(device)
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # モデル読み込み
        self._model = None
        self._config = None
        self._load_model()

        # 棋譜履歴（ゲームごとにリセット）
        self._move_history: List[int] = []

    def _load_model(self) -> None:
        """モデルを読み込む"""
        from model import GomokuGPT, GomokuGPTConfig

        path = Path(self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {self._model_path}")

        print(f"Loading NeuralPlayer model from {self._model_path}...")

        checkpoint = torch.load(self._model_path, map_location=self._device, weights_only=True)

        # 設定を取得（辞書から復元）
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            self._config = GomokuGPTConfig.from_dict(checkpoint['config'])
        else:
            self._config = GomokuGPTConfig()

        # モデル作成
        self._model = GomokuGPT(self._config)

        # 重み読み込み
        if 'model_state_dict' in checkpoint:
            self._model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self._model.load_state_dict(checkpoint)

        self._model.to(self._device)
        self._model.eval()

        print(f"  Device: {self._device}")
        print(f"  Temperature: {self._temperature}")
        print(f"  Top-k: {self._top_k}")

    @property
    def name(self) -> str:
        temp_str = f"t={self._temperature}" if self._temperature > 0 else "greedy"
        return f"NeuralAI ({temp_str})"

    @property
    def difficulty(self) -> str:
        return "Hard"

    @property
    def supports_progress(self) -> bool:
        return True

    def reset_history(self) -> None:
        """棋譜履歴をリセット（新しいゲーム開始時に呼び出す）"""
        self._move_history = []

    def select_move(self, board: Board, rule: GameRule, stone: Stone) -> Position:
        """
        次の一手を選択

        Args:
            board: 現在の盤面
            rule: ゲームルール
            stone: 次に置く石の色

        Returns:
            選択した着手位置
        """
        start_time = time.time()

        # 合法手を取得
        valid_moves = rule.get_valid_moves(board, stone)
        if not valid_moves:
            raise ValueError("No valid moves available")

        # 合法手が1つだけならそれを返す
        if len(valid_moves) == 1:
            return valid_moves[0]

        # 盤面から棋譜を復元（move_historyがない場合）
        if not self._move_history:
            self._reconstruct_history(board)

        # モデルで予測
        with torch.no_grad():
            probs = self._get_move_probabilities(valid_moves, board.width)

        # 進捗通知
        elapsed = time.time() - start_time
        top_moves = self._get_top_moves(probs, valid_moves, 5)
        self._report_progress(ThinkingProgress(
            ai_type="neural",
            elapsed_time=elapsed,
            top_moves=top_moves,
        ))

        # 着手を選択
        move = self._sample_move(probs, valid_moves)

        # 履歴に追加
        move_idx = move.y * board.width + move.x
        self._move_history.append(move_idx)

        return move

    def _reconstruct_history(self, board: Board) -> None:
        """
        盤面から棋譜履歴を復元

        注意: 正確な手順は分からないため、石の位置だけを記録。
        モデルは位置の順序から学習しているため、精度が落ちる可能性あり。
        """
        # 盤面上の石を収集（順序は不明なのでスキャン順）
        for y in range(board.height):
            for x in range(board.width):
                stone = board.get_stone(x, y)
                if stone != Stone.EMPTY:
                    self._move_history.append(y * board.width + x)

    def _get_move_probabilities(
        self,
        valid_moves: List[Position],
        board_width: int
    ) -> torch.Tensor:
        """
        モデルから着手確率を取得し、合法手でフィルタリング

        Returns:
            合法手でマスクされた確率分布 (225,)
        """
        # 入力シーケンス: [START, move1, move2, ...]
        # max_seq_len を超えないよう末尾を使用
        max_history = self._config.max_seq_len - 1  # STARTトークン分を引く
        history = self._move_history[-max_history:]
        input_ids = [self._config.start_token] + history
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)

        # 順伝播
        logits, _ = self._model(input_tensor)
        logits = logits[0, -1, :]  # 最後の位置の出力 (225,)

        # 合法手マスク作成
        valid_indices = [m.y * board_width + m.x for m in valid_moves]
        mask = torch.full_like(logits, float('-inf'))
        mask[valid_indices] = 0

        # マスク適用
        masked_logits = logits + mask

        # 温度でスケーリング
        if self._temperature > 0:
            masked_logits = masked_logits / self._temperature

        # Top-k フィルタリング
        if self._top_k is not None and self._top_k < len(valid_moves):
            v, _ = torch.topk(masked_logits, min(self._top_k, len(valid_indices)))
            masked_logits[masked_logits < v[-1]] = float('-inf')

        # 確率に変換
        probs = F.softmax(masked_logits, dim=-1)

        return probs

    def _sample_move(
        self,
        probs: torch.Tensor,
        valid_moves: List[Position]
    ) -> Position:
        """
        確率分布から着手をサンプリング

        Args:
            probs: 確率分布 (225,)
            valid_moves: 合法手リスト

        Returns:
            選択された着手
        """
        if self._temperature == 0:
            # Greedy: 最高確率の手
            idx = probs.argmax().item()
        else:
            # サンプリング
            idx = torch.multinomial(probs, num_samples=1).item()

        # インデックスからPositionに変換
        board_width = 15  # 標準盤面
        x = idx % board_width
        y = idx // board_width

        return Position(x, y)

    def _get_top_moves(
        self,
        probs: torch.Tensor,
        valid_moves: List[Position],
        k: int
    ) -> List[tuple[Position, float]]:
        """上位k個の手と確率を取得"""
        board_width = 15

        # 上位k個を取得
        top_probs, top_indices = probs.topk(min(k, len(valid_moves)))

        result = []
        for prob, idx in zip(top_probs.tolist(), top_indices.tolist()):
            x = idx % board_width
            y = idx // board_width
            result.append((Position(x, y), prob))

        return result

    def notify_opponent_move(self, move: Position, board_width: int = 15) -> None:
        """
        相手の着手を通知（履歴に追加）

        Args:
            move: 相手の着手位置
            board_width: 盤面の幅
        """
        move_idx = move.y * board_width + move.x
        self._move_history.append(move_idx)


class NeuralPlayerFactory:
    """NeuralPlayerのファクトリクラス"""

    _instance: Optional[NeuralPlayer] = None
    _model_path: Optional[str] = None

    @classmethod
    def create(
        cls,
        model_path: str = "checkpoints/gomoku_gpt_best.pth",
        temperature: float = 0.5,
        top_k: Optional[int] = 10,
        force_reload: bool = False
    ) -> NeuralPlayer:
        """
        NeuralPlayerを作成（シングルトン的にモデルを再利用）

        Args:
            model_path: モデルのパス
            temperature: サンプリング温度
            top_k: Top-k サンプリング
            force_reload: 強制的にモデルを再読み込み

        Returns:
            NeuralPlayer インスタンス
        """
        # 同じモデルなら再利用
        if cls._instance is not None and cls._model_path == model_path and not force_reload:
            # 設定だけ更新
            cls._instance._temperature = temperature
            cls._instance._top_k = top_k
            cls._instance.reset_history()
            return cls._instance

        # 新規作成
        cls._instance = NeuralPlayer(
            model_path=model_path,
            temperature=temperature,
            top_k=top_k
        )
        cls._model_path = model_path

        return cls._instance

    @classmethod
    def is_model_available(cls, model_path: str = "checkpoints/gomoku_gpt_best.pth") -> bool:
        """モデルファイルが存在するかチェック"""
        return Path(model_path).exists()


# =============================================================================
# テスト
# =============================================================================

def test_neural_player():
    """NeuralPlayerの動作テスト"""
    from game_core import StandardGomokuRule, GameEngine

    print("=" * 60)
    print("NeuralPlayer Test")
    print("=" * 60)

    model_path = "checkpoints/gomoku_gpt_best.pth"

    if not Path(model_path).exists():
        print(f"\n[SKIP] Model not found: {model_path}")
        print("Please train a model first:")
        print("  python generate_data.py -n 100 --depth 2 -o train.jsonl")
        print("  python train.py -d train.jsonl -e 10")
        return False

    # プレイヤー作成
    print("\n1. Creating NeuralPlayer...")
    player = NeuralPlayer(model_path, temperature=0.5)
    print(f"   Name: {player.name}")
    print(f"   Difficulty: {player.difficulty}")

    # ゲームセットアップ
    print("\n2. Setting up game...")
    rule = StandardGomokuRule()
    engine = GameEngine(rule)
    print(f"   Board: {rule.board_width}x{rule.board_height}")

    # 数手プレイ
    print("\n3. Playing test moves...")
    current_stone = Stone.BLACK
    test_moves = 10

    for i in range(test_moves):
        try:
            move = player.select_move(engine.board, rule, current_stone)
            print(f"   Move {i+1}: {current_stone.name} -> ({move.x}, {move.y})")

            # 合法手チェック
            if not rule.is_valid_move(engine.board, move.x, move.y, current_stone):
                print(f"   [ERROR] Illegal move!")
                return False

            # 着手
            engine.play_move(move.x, move.y)

            # 相手の着手を通知
            player.notify_opponent_move(move)

            # 勝敗チェック
            status = rule.check_winner(engine.board, move.x, move.y, current_stone)
            if status != GameStatus.ONGOING:
                print(f"   Game ended: {status.name}")
                break

            current_stone = current_stone.opponent()

        except Exception as e:
            print(f"   [ERROR] {e}")
            import traceback
            traceback.print_exc()
            return False

    print("\n4. All tests passed!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    test_neural_player()
