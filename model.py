"""
GomokuGPT - 五目並べ専用 Transformer モデル

Decoder-only Transformer (GPT style) で次の一手を予測します。
NanoGPT スタイルの軽量実装です。

アーキテクチャ:
- Token Embedding (225マス + 特殊トークン)
- Learnable Positional Embedding
- N層の Transformer Block (Multi-head Attention + FFN)
- 出力層 (225クラス分類)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, asdict, fields
from typing import Optional


# =============================================================================
# 設定
# =============================================================================

@dataclass
class GomokuGPTConfig:
    """モデル設定"""
    # 盤面サイズ
    board_size: int = 15
    vocab_size: int = 228  # 225マス + PAD(225) + START(226) + END(227)

    # 特殊トークン
    pad_token: int = 225
    start_token: int = 226
    end_token: int = 227

    # モデルサイズ
    n_layer: int = 4          # Transformer層の数
    n_head: int = 4           # Attention head数
    n_embd: int = 128         # 埋め込み次元
    dropout: float = 0.1      # Dropout率

    # シーケンス長（五目並べは最大225手だが、通常は50手以内で終わる）
    max_seq_len: int = 128

    @property
    def n_positions(self) -> int:
        """出力位置数（盤面のマス数）"""
        return self.board_size * self.board_size  # 225

    def to_dict(self) -> dict:
        """辞書に変換（torch.save 用）"""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GomokuGPTConfig":
        """辞書から復元"""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)


# =============================================================================
# Transformer コンポーネント
# =============================================================================

class CausalSelfAttention(nn.Module):
    """
    因果的自己注意機構 (Causal Self-Attention)

    未来のトークンを参照できないようにマスクします。
    Multi-head Attention を効率的に実装しています。
    """

    def __init__(self, config: GomokuGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout

        # Query, Key, Value を一度に計算（効率化）
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        # 出力投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # 因果マスク（下三角行列）
        # register_buffer でモデルと一緒に保存・移動される
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, n_embd)

        Returns:
            (batch_size, seq_len, n_embd)
        """
        B, T, C = x.size()  # batch, sequence length, embedding dim

        # Q, K, V を計算
        qkv = self.c_attn(x)  # (B, T, 3 * n_embd)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # Multi-head に分割: (B, T, n_head, head_dim) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # att = (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))

        # 因果マスク適用（未来を見えなくする）
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # Softmax で正規化
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Attention 重みを Value に適用
        y = att @ v  # (B, n_head, T, head_dim)

        # Head を結合: (B, n_head, T, head_dim) -> (B, T, n_embd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 出力投影
        y = self.resid_dropout(self.c_proj(y))

        return y


class MLP(nn.Module):
    """
    Feed-Forward Network (FFN)

    2層の全結合層で、中間層は4倍に拡張します。
    GELU活性化関数を使用（GPT-2スタイル）。
    """

    def __init__(self, config: GomokuGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block

    Pre-norm アーキテクチャ（LayerNorm を Attention/FFN の前に適用）を採用。
    これは学習の安定性が高く、GPT-2/3 で使用されています。
    """

    def __init__(self, config: GomokuGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm + Residual connection
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


# =============================================================================
# GomokuGPT メインモデル
# =============================================================================

class GomokuGPT(nn.Module):
    """
    五目並べ専用 GPT モデル

    着手シーケンスから次の一手を予測します。

    入力: [START, move1, move2, ..., move_t]
    出力: 各位置での次の手の確率分布

    使用例:
        config = GomokuGPTConfig()
        model = GomokuGPT(config)

        # 入力: (batch_size, seq_len) のトークン列
        input_ids = torch.tensor([[226, 112, 127, 113]])  # START, moves...
        logits = model(input_ids)  # (batch_size, seq_len, 225)
    """

    def __init__(self, config: GomokuGPTConfig):
        super().__init__()
        self.config = config

        # トークン埋め込み
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)

        # 位置埋め込み（学習可能）
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)

        # Dropout
        self.drop = nn.Dropout(config.dropout)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # 最終層の正規化
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 出力層（次の手の予測: 225クラス）
        self.head = nn.Linear(config.n_embd, config.n_positions, bias=False)

        # 重みの初期化
        self.apply(self._init_weights)

        # パラメータ数を表示
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GomokuGPT initialized with {n_params:,} parameters")

    def _init_weights(self, module: nn.Module) -> None:
        """重みの初期化（GPT-2スタイル）"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        順伝播

        Args:
            input_ids: (batch_size, seq_len) トークンID列
            targets: (batch_size, seq_len) 正解ラベル（学習時のみ）
            loss_mask: (batch_size, seq_len) 損失マスク（0.0/1.0）
                       Noneの場合は全トークンで損失計算（PADは除外）

        Returns:
            logits: (batch_size, seq_len, n_positions) 各位置での予測確率
            loss: CrossEntropyLoss（targetsが指定された場合）
        """
        device = input_ids.device
        B, T = input_ids.size()

        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        # 位置インデックスを作成
        pos = torch.arange(0, T, dtype=torch.long, device=device)  # (T,)

        # 埋め込み
        tok_emb = self.tok_emb(input_ids)  # (B, T, n_embd)
        pos_emb = self.pos_emb(pos)         # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x)

        # 最終正規化
        x = self.ln_f(x)

        # 出力層（次の手の予測）
        logits = self.head(x)  # (B, T, 225)

        # 損失計算（学習時）
        loss = None
        if targets is not None:
            if loss_mask is not None:
                # Winner-only学習: 要素ごとの損失を計算してマスク適用
                per_token_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.config.pad_token,
                    reduction='none'
                )
                mask_flat = loss_mask.view(-1)
                # PADトークンも除外するためマスクを掛け合わせる
                pad_mask = (targets.view(-1) != self.config.pad_token).float()
                combined_mask = mask_flat * pad_mask
                denom = combined_mask.sum()
                if denom > 0:
                    loss = (per_token_loss * combined_mask).sum() / denom
                else:
                    # マスクが全てゼロの場合はフォールバック
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=self.config.pad_token
                    )
            else:
                # 通常の損失計算
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    targets.view(-1),
                    ignore_index=self.config.pad_token
                )

        return logits, loss

    @torch.no_grad()
    def predict_next_move(
        self,
        moves: list[int],
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> tuple[int, torch.Tensor]:
        """
        次の一手を予測

        Args:
            moves: これまでの着手リスト（インデックス 0-224）
            temperature: サンプリング温度（高いほどランダム）
            top_k: 上位k個からサンプリング（Noneなら全体から）

        Returns:
            next_move: 予測された次の手（0-224）
            probs: 全位置の確率分布
        """
        self.eval()
        device = next(self.parameters()).device

        # 入力シーケンスを作成: [START, move1, move2, ...]
        input_ids = [self.config.start_token] + moves
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # 順伝播
        logits, _ = self(input_tensor)

        # 最後の位置の出力を取得
        logits = logits[0, -1, :]  # (225,)

        # 温度でスケーリング
        if temperature != 1.0:
            logits = logits / temperature

        # Top-k フィルタリング
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[-1]] = float('-inf')

        # 確率に変換
        probs = F.softmax(logits, dim=-1)

        # サンプリング
        next_move = torch.multinomial(probs, num_samples=1).item()

        return next_move, probs

    @torch.no_grad()
    def get_move_probabilities(
        self,
        moves: list[int],
        valid_moves: Optional[list[int]] = None
    ) -> torch.Tensor:
        """
        合法手の確率分布を取得

        Args:
            moves: これまでの着手リスト
            valid_moves: 合法手のリスト（Noneなら全マス）

        Returns:
            確率分布（合法手のみ正規化、またはマスク適用）
        """
        self.eval()
        device = next(self.parameters()).device

        # 入力シーケンスを作成
        input_ids = [self.config.start_token] + moves
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # 順伝播
        logits, _ = self(input_tensor)
        logits = logits[0, -1, :]  # (225,)

        # 合法手マスク
        if valid_moves is not None:
            mask = torch.full_like(logits, float('-inf'))
            mask[valid_moves] = 0
            logits = logits + mask

        # 確率に変換
        probs = F.softmax(logits, dim=-1)

        return probs


# =============================================================================
# ユーティリティ
# =============================================================================

def count_parameters(model: nn.Module) -> int:
    """モデルのパラメータ数をカウント"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(path: str, config: Optional[GomokuGPTConfig] = None) -> GomokuGPT:
    """
    保存されたモデルを読み込む

    Args:
        path: チェックポイントのパス
        config: モデル設定（Noneならデフォルト）

    Returns:
        読み込まれたモデル
    """
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)

    # チェックポイントから設定を復元
    if config is None:
        if 'config' in checkpoint and isinstance(checkpoint['config'], dict):
            config = GomokuGPTConfig.from_dict(checkpoint['config'])
        else:
            config = GomokuGPTConfig()

    model = GomokuGPT(config)

    # state_dict のみの場合と、チェックポイント全体の場合に対応
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    return model


if __name__ == "__main__":
    # テスト実行
    print("=" * 50)
    print("GomokuGPT Model Test")
    print("=" * 50)

    config = GomokuGPTConfig()
    model = GomokuGPT(config)

    print(f"\nConfig:")
    print(f"  Board size: {config.board_size}x{config.board_size}")
    print(f"  Vocab size: {config.vocab_size}")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Max sequence length: {config.max_seq_len}")

    # テスト入力
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 225, (batch_size, seq_len))
    targets = torch.randint(0, 225, (batch_size, seq_len))

    print(f"\nTest forward pass:")
    print(f"  Input shape: {input_ids.shape}")

    logits, loss = model(input_ids, targets)
    print(f"  Output shape: {logits.shape}")
    print(f"  Loss: {loss.item():.4f}")

    # 推論テスト
    print(f"\nTest inference:")
    moves = [112, 127, 113]  # 中央付近の3手
    next_move, probs = model.predict_next_move(moves)
    print(f"  Input moves: {moves}")
    print(f"  Predicted next move: {next_move}")
    print(f"  Top 5 probabilities: {probs.topk(5)}")
