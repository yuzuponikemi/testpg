#!/usr/bin/env python3
"""
GomokuGPT 学習スクリプト

training_data.jsonl から学習データを読み込み、
次の一手を予測するTransformerモデルを学習します。

使用方法:
    python train.py --data training_data.jsonl --epochs 50
    python train.py -d data.jsonl -e 100 --lr 0.001 --batch-size 64
"""

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import GomokuGPT, GomokuGPTConfig
from data_augmentation import augment_moves


# =============================================================================
# データセット
# =============================================================================

class GomokuDataset(Dataset):
    """
    五目並べの棋譜データセット

    各サンプルは (input_sequence, target_sequence, loss_mask) の3-tuple:
    - input:     [START, move1, move2, ..., move_{t-1}]
    - target:    [move1, move2, ..., move_t]
    - loss_mask: 勝者の手のみ1.0、それ以外は0.0（winner_only時）

    Transformerは各位置で「次の手」を予測するので、
    入力をずらしたものが正解ラベルになります。

    augment=True にすると、8つの盤面対称変換でデータを8倍に拡張します。
    winner_only=True にすると、引き分けを除外し勝者の手のみで損失を計算します。
    """

    def __init__(
        self,
        data_path: str,
        config: GomokuGPTConfig,
        max_samples: Optional[int] = None,
        augment: bool = False,
        winner_only: bool = False
    ):
        """
        Args:
            data_path: JSONLファイルのパス
            config: モデル設定
            max_samples: 最大サンプル数（デバッグ用）
            augment: データ拡張を有効にするか（8倍）
            winner_only: 勝者の手のみで学習するか
        """
        self.config = config
        self.samples: list[tuple[list[int], int]] = []  # (moves, winner)
        self._augment = augment
        self._winner_only = winner_only

        print(f"Loading data from {data_path}...")

        raw_games: list[tuple[list[int], int]] = []
        skipped_draws = 0
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break

                record = json.loads(line)
                moves = record['input']
                winner = record.get('winner', 0)

                # 短すぎるゲームは除外（最低3手）
                if len(moves) < 3:
                    continue

                # winner_onlyモードでは引き分けを除外
                if winner_only and winner == 0:
                    skipped_draws += 1
                    continue

                # 最大長でクリップ（STARTトークン分を引く）
                if len(moves) > config.max_seq_len - 1:
                    moves = moves[:config.max_seq_len - 1]

                raw_games.append((moves, winner))

        # データ拡張
        if augment:
            for moves, winner in raw_games:
                for t_idx in range(8):
                    self.samples.append((augment_moves(moves, t_idx), winner))
            print(f"Loaded {len(raw_games)} games x 8 symmetries = {len(self.samples)} samples")
        else:
            self.samples = raw_games
            print(f"Loaded {len(self.samples)} games")

        if winner_only and skipped_draws > 0:
            print(f"  Skipped {skipped_draws} draws (winner_only mode)")

        # 統計情報
        lengths = [len(s[0]) for s in self.samples]
        print(f"  Average game length: {sum(lengths)/len(lengths):.1f} moves")
        print(f"  Min/Max length: {min(lengths)}/{max(lengths)} moves")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        1サンプルを取得

        Returns:
            input_ids: [START, move1, move2, ..., move_{n-1}]
            targets:   [move1, move2, ..., move_n]
            loss_mask: 勝者の手のみ1.0（winner_only時）、通常は全て1.0
        """
        moves, winner = self.samples[idx]
        seq_len = len(moves)

        # 入力: [START, move1, ..., move_{n-1}]
        input_ids = [self.config.start_token] + moves[:-1]

        # 正解: [move1, move2, ..., move_n]
        targets = moves

        # Loss mask生成
        if self._winner_only and winner != 0:
            # winner=1(黒勝ち): 黒の手=偶数位置(0,2,4,...)のみ学習
            # winner=-1(白勝ち): 白の手=奇数位置(1,3,5,...)のみ学習
            mask = [0.0] * seq_len
            for i in range(seq_len):
                if winner == 1 and i % 2 == 0:  # 黒の手(偶数位置)
                    mask[i] = 1.0
                elif winner == -1 and i % 2 == 1:  # 白の手(奇数位置)
                    mask[i] = 1.0
        else:
            mask = [1.0] * seq_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(mask, dtype=torch.float)
        )


def collate_fn(
    batch: list,
    pad_token: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    バッチ内のシーケンスをパディングして揃える

    Args:
        batch: [(input_ids, targets, loss_mask), ...] のリスト
        pad_token: パディングトークン

    Returns:
        padded_inputs: (batch_size, max_len)
        padded_targets: (batch_size, max_len)
        padded_masks: (batch_size, max_len)
    """
    inputs, targets, masks = zip(*batch)

    # 最大長を取得
    max_len = max(len(x) for x in inputs)

    # パディング
    padded_inputs = []
    padded_targets = []
    padded_masks = []

    for inp, tgt, msk in zip(inputs, targets, masks):
        pad_len = max_len - len(inp)
        padded_inputs.append(
            torch.cat([inp, torch.full((pad_len,), pad_token, dtype=torch.long)])
        )
        padded_targets.append(
            torch.cat([tgt, torch.full((pad_len,), pad_token, dtype=torch.long)])
        )
        padded_masks.append(
            torch.cat([msk, torch.zeros(pad_len, dtype=torch.float)])
        )

    return torch.stack(padded_inputs), torch.stack(padded_targets), torch.stack(padded_masks)


# =============================================================================
# 学習ループ
# =============================================================================

def train_epoch(
    model: GomokuGPT,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """
    1エポックの学習

    Returns:
        平均損失
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for batch_idx, (inputs, targets, masks) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        # 順伝播
        optimizer.zero_grad()
        logits, loss = model(inputs, targets, masks)

        # 逆伝播
        loss.backward()

        # 勾配クリッピング（学習の安定化）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # 進捗表示
        if (batch_idx + 1) % 10 == 0 or batch_idx == len(dataloader) - 1:
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_batches
            print(f"\r  Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s", end="")

    print()  # 改行
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: GomokuGPT,
    dataloader: DataLoader,
    device: torch.device
) -> tuple[float, float]:
    """
    検証データでの評価

    Returns:
        (平均損失, 精度)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    num_batches = 0

    for inputs, targets, masks in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        logits, loss = model(inputs, targets, masks)

        total_loss += loss.item()
        num_batches += 1

        # 精度計算（PADトークンを除外）
        preds = logits.argmax(dim=-1)
        mask = targets != model.config.pad_token
        total_correct += ((preds == targets) & mask).sum().item()
        total_tokens += mask.sum().item()

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0

    return avg_loss, accuracy


def train(
    data_path: str,
    output_dir: str,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    val_split: float = 0.1,
    save_every: int = 10,
    config: Optional[GomokuGPTConfig] = None,
    augment: bool = False,
    winner_only: bool = False
) -> None:
    """
    メインの学習関数

    Args:
        data_path: 学習データのパス
        output_dir: 出力ディレクトリ
        epochs: エポック数
        batch_size: バッチサイズ
        learning_rate: 学習率
        val_split: 検証データの割合
        save_every: 何エポックごとに保存するか
        config: モデル設定
        augment: データ拡張（8x対称変換）を有効にするか
        winner_only: 勝者の手のみで学習するか
    """
    # デバイス設定
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 出力ディレクトリ作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 設定
    if config is None:
        config = GomokuGPTConfig()

    # データセット読み込み
    full_dataset = GomokuDataset(data_path, config, augment=augment, winner_only=winner_only)

    # 学習/検証分割
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    print(f"\nDataset split: {train_size} train, {val_size} validation")

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, config.pad_token),
        num_workers=0,  # マルチプロセスは使わない（小規模データ向け）
        pin_memory=True if device.type == "cuda" else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, config.pad_token),
        num_workers=0,
        pin_memory=True if device.type == "cuda" else False
    )

    # モデル
    model = GomokuGPT(config).to(device)

    # オプティマイザ
    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # 学習率スケジューラ（Cosine Annealing）
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=learning_rate * 0.1)

    # 学習ループ
    print("\n" + "=" * 60)
    print("Starting training")
    print("=" * 60)

    best_val_loss = float('inf')
    train_start = time.time()

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)

        # 学習
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)

        # 検証
        val_loss, val_acc = evaluate(model, val_loader, device)

        # 学習率更新
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Accuracy: {val_acc:.2%}")
        print(f"  LR: {current_lr:.2e}")

        # ベストモデル保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output_path / "gomoku_gpt_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.to_dict()
            }, best_path)
            print(f"  [*] Best model saved: {best_path}")

        # 定期保存
        if epoch % save_every == 0:
            checkpoint_path = output_path / f"gomoku_gpt_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.to_dict()
            }, checkpoint_path)
            print(f"  [*] Checkpoint saved: {checkpoint_path}")

    # 最終モデル保存
    final_path = output_path / "gomoku_gpt.pth"
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': config.to_dict()
    }, final_path)

    total_time = time.time() - train_start
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Final model: {final_path}")
    print(f"  Best model: {output_path / 'gomoku_gpt_best.pth'}")


def main():
    parser = argparse.ArgumentParser(
        description="Train GomokuGPT model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py -d training_data.jsonl                    # Basic training
  python train.py -d data.jsonl -e 100 --lr 0.001          # More epochs
  python train.py -d data.jsonl --batch-size 64 --n-layer 6  # Larger model
        """
    )

    # データ
    parser.add_argument(
        "-d", "--data",
        type=str,
        required=True,
        help="Path to training data (JSONL format)"
    )

    # 出力
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints (default: ./checkpoints)"
    )

    # 学習設定
    parser.add_argument(
        "-e", "--epochs",
        type=int,
        default=50,
        help="Number of epochs (default: 50)"
    )

    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)"
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate (default: 3e-4)"
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1)"
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)"
    )

    # モデル設定
    parser.add_argument(
        "--n-layer",
        type=int,
        default=4,
        help="Number of transformer layers (default: 4)"
    )

    parser.add_argument(
        "--n-head",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)"
    )

    parser.add_argument(
        "--n-embd",
        type=int,
        default=128,
        help="Embedding dimension (default: 128)"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate (default: 0.1)"
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable 8x data augmentation via board symmetry"
    )

    parser.add_argument(
        "--winner-only",
        action="store_true",
        help="Only train on winner's moves (skip draws, mask loser's moves)"
    )

    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=128,
        help="Maximum sequence length (default: 128)"
    )

    args = parser.parse_args()

    # モデル設定
    config = GomokuGPTConfig(
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len
    )

    print("=" * 60)
    print("GomokuGPT Training")
    print("=" * 60)
    print(f"\nModel config:")
    print(f"  Layers: {config.n_layer}")
    print(f"  Heads: {config.n_head}")
    print(f"  Embedding dim: {config.n_embd}")
    print(f"  Dropout: {config.dropout}")
    print(f"\nTraining config:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Validation split: {args.val_split}")
    print(f"  Data augmentation: {'8x symmetry' if args.augment else 'off'}")
    print(f"  Winner-only: {'on' if args.winner_only else 'off'}")
    print(f"  Max seq len: {args.max_seq_len}")
    print()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_split=args.val_split,
        save_every=args.save_every,
        config=config,
        augment=args.augment,
        winner_only=args.winner_only
    )


if __name__ == "__main__":
    main()
