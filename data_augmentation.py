"""
盤面対称変換ユーティリティ（データ拡張用）

15x15の五目並べ盤面に対する8つの対称変換（恒等、90/180/270度回転、
左右/上下反転、主対角線/副対角線反転）を提供します。

train.py と generate_data.py の両方から使用されます。
"""

from typing import Callable


def _idx_to_xy(idx: int, width: int = 15) -> tuple[int, int]:
    """インデックスを(x, y)座標に変換"""
    return idx % width, idx // width


def _xy_to_idx(x: int, y: int, width: int = 15) -> int:
    """(x, y)座標をインデックスに変換"""
    return y * width + x


def _make_transform_table(transform_fn: Callable, size: int = 15) -> list[int]:
    """変換関数から0-224のルックアップテーブルを作成"""
    table = [0] * (size * size)
    for idx in range(size * size):
        x, y = _idx_to_xy(idx, size)
        nx, ny = transform_fn(x, y, size)
        table[idx] = _xy_to_idx(nx, ny, size)
    return table


# 8つの対称変換（恒等変換を含む）
_SYMMETRY_TRANSFORMS = [
    lambda x, y, s: (x, y),                      # 恒等
    lambda x, y, s: (s - 1 - y, x),              # 90度回転
    lambda x, y, s: (s - 1 - x, s - 1 - y),      # 180度回転
    lambda x, y, s: (y, s - 1 - x),              # 270度回転
    lambda x, y, s: (s - 1 - x, y),              # 左右反転
    lambda x, y, s: (x, s - 1 - y),              # 上下反転
    lambda x, y, s: (y, x),                      # 主対角線反転
    lambda x, y, s: (s - 1 - y, s - 1 - x),      # 副対角線反転
]

# ルックアップテーブルを事前計算
_SYMMETRY_TABLES: list[list[int]] = [
    _make_transform_table(fn) for fn in _SYMMETRY_TRANSFORMS
]


def augment_moves(moves: list[int], transform_idx: int) -> list[int]:
    """棋譜の全手に対称変換を適用"""
    table = _SYMMETRY_TABLES[transform_idx]
    return [table[m] for m in moves]


def augment_all(moves: list[int]) -> list[list[int]]:
    """8つの対称変換すべてを適用して返す"""
    return [augment_moves(moves, i) for i in range(8)]
