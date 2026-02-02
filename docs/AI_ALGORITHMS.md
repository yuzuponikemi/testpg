# AI アルゴリズム解説

本ドキュメントでは、Variant Go Platform に実装されている4種類のAIアルゴリズムについて、実装の詳細と数理的背景を解説します。

---

## 目次

1. [RandomAI - ランダム戦略](#1-randomai---ランダム戦略)
2. [MinimaxAI - ミニマックス探索](#2-minimaxi---ミニマックス探索)
3. [MCTSAI - モンテカルロ木探索](#3-mctsai---モンテカルロ木探索)
4. [VCFBasedAI - 詰め探索](#4-vcfbasedai---詰め探索)
5. [アルゴリズム比較](#5-アルゴリズム比較)

---

## 1. RandomAI - ランダム戦略

### 概要

最も単純なAI。合法手の中からランダムに1手を選択します。

### アルゴリズム

```
入力: 盤面 B, ルール R, 手番の石 S
出力: 着手位置

1. valid_moves ← R.get_valid_moves(B, S)
2. return random.choice(valid_moves)
```

### 計算量

- **時間計算量**: O(n) （n = 盤面サイズ、合法手列挙のため）
- **空間計算量**: O(n)

### 用途

- テスト用ベースライン
- 他のAIの強さを測る基準
- MCTSのプレイアウト（ランダムシミュレーション）

---

## 2. MinimaxAI - ミニマックス探索

### 概要

ゲーム理論に基づく古典的な探索アルゴリズム。自分の利益を最大化し、相手の利益を最小化する手を選択します。Alpha-Beta枝刈りにより探索効率を大幅に向上させています。

### 数理的背景

#### ミニマックス定理

二人零和有限確定完全情報ゲームにおいて、両プレイヤーが最善を尽くしたときのゲームの価値は一意に定まります（フォン・ノイマン、1928年）。

$$
V(s) = \begin{cases}
\text{evaluate}(s) & \text{if } s \text{ is terminal} \\
\max_{a \in A(s)} V(\text{result}(s, a)) & \text{if player is MAX} \\
\min_{a \in A(s)} V(\text{result}(s, a)) & \text{if player is MIN}
\end{cases}
$$

#### Alpha-Beta枝刈り

探索木の不要な枝を刈り取ることで、計算量を削減します。

- **α (alpha)**: MAX側の下限値（少なくともこの値は保証される）
- **β (beta)**: MIN側の上限値（これ以上は許容しない）

**枝刈り条件**:
- β ≤ α のとき、これ以上探索しても結果は変わらない

**最良ケースの計算量**: O(b^(d/2))（b=分岐数, d=深度）
**最悪ケースの計算量**: O(b^d)

### 実装詳細

#### 評価関数

局面の有利不利を数値化します。本実装では連続石パターンに基づく評価を行います。

| パターン | スコア | 説明 |
|----------|--------|------|
| 五連 | 100,000 | 勝利確定 |
| 両端開き四連 | 10,000 | 止められない（次に必ず五連） |
| 片端開き四連 | 1,000 | 防御必須 |
| 両端開き三連 | 500 | 次に四連を2箇所作れる |
| 片端開き三連 | 100 | 次に四連を1箇所作れる |
| 両端開き二連 | 50 | 発展の余地あり |
| 片端開き二連 | 10 | 軽微な価値 |

```python
評価値 = 自分のスコア - 相手のスコア × 1.1
```

相手のスコアに1.1を掛けることで、防御を若干重視しています。

#### 候補手絞り込み

全マス探索は非効率なため、既存の石の周囲2マスのみを候補とします。

```
候補 = {(x±2, y±2) | (x,y)に石がある} ∩ 空きマス
```

#### 反復深化 (Iterative Deepening)

深さ1から順に探索し、時間制限内で可能な限り深く読みます。

```
for depth in 1..max_depth:
    if timeout: break
    best_move = search(depth)
```

**利点**:
- 時間制限に柔軟に対応
- 浅い探索で得た手の順序を深い探索に活用（Move Ordering）
- 任意のタイミングで中断可能

### アルゴリズム（擬似コード）

```
function minimax(node, depth, α, β, maximizing):
    if depth = 0 or node is terminal:
        return evaluate(node)

    if maximizing:
        value = -∞
        for each child of node:
            value = max(value, minimax(child, depth-1, α, β, FALSE))
            α = max(α, value)
            if β ≤ α:
                break  # Beta cutoff
        return value
    else:
        value = +∞
        for each child of node:
            value = min(value, minimax(child, depth-1, α, β, TRUE))
            β = min(β, value)
            if β ≤ α:
                break  # Alpha cutoff
        return value
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|------------|------------|------|
| depth | 3 | 探索深度（大きいほど強いが遅い） |
| time_limit | None | 時間制限（秒） |
| use_iterative_deepening | False | 反復深化の有効化 |

### 強さの目安

| 深度 | 難易度 | 思考時間目安 |
|------|--------|--------------|
| 1-2 | Easy | < 1秒 |
| 3-4 | Medium | 数秒 |
| 5+ | Hard | 10秒以上 |

---

## 3. MCTSAI - モンテカルロ木探索

### 概要

確率的シミュレーションに基づく探索アルゴリズム。評価関数を必要とせず、ランダムプレイアウトの勝率から手の良さを推定します。囲碁AIで革命的な成功を収めた手法です。

### 数理的背景

#### 多腕バンディット問題

MCTSの探索/活用のトレードオフは、多腕バンディット問題として定式化できます。

**問題設定**: 複数のスロットマシン（腕）があり、各腕の報酬分布は未知。総報酬を最大化するにはどの腕を引くべきか？

#### UCB1 (Upper Confidence Bound)

最適に近い選択戦略として知られるアルゴリズム（Auer et al., 2002）。

$$
UCB1_i = \bar{X}_i + C \sqrt{\frac{\ln N}{n_i}}
$$

- $\bar{X}_i$: 腕 i の平均報酬（活用項）
- $N$: 全体の試行回数
- $n_i$: 腕 i の試行回数
- $C$: 探索係数（通常 $\sqrt{2} \approx 1.414$）

**定理**: UCB1は対数的な後悔上限を達成する（ほぼ最適）。

### MCTSの4フェーズ

```
      Selection → Expansion → Simulation → Backpropagation
         ↓           ↓            ↓              ↓
      UCB1で      新ノード    ランダム        結果を
      子選択      を追加      プレイアウト     親に伝播
```

#### 1. Selection（選択）

ルートから葉まで、UCB1値が最大の子ノードを選択して降りていきます。

```python
while node.children and not node.untried_moves:
    node = argmax(child.ucb1() for child in node.children)
```

#### 2. Expansion（展開）

未試行の手がある場合、1つ選んで新しい子ノードを作成します。

```python
move = random.choice(node.untried_moves)
child = MCTSNode(board.apply(move))
node.children.append(child)
```

#### 3. Simulation（シミュレーション）

展開したノードからランダムに終局までプレイアウトします。

```python
while not game_over:
    move = random.choice(valid_moves)
    board.apply(move)
return winner
```

#### 4. Backpropagation（逆伝播）

シミュレーション結果を祖先ノードに伝播します。

```python
while node:
    node.visits += 1
    if winner == node.player:
        node.wins += 1
    node = node.parent
```

### 本実装の最適化

#### 候補手絞り込み

既存の石の周囲2マスのみを探索対象とします。

#### 即時勝利/脅威チェック

シミュレーション前に以下をチェック:
1. 即座に勝てる手（五連完成）
2. 相手の勝利を防ぐ手
3. 四連を作る手（次に勝てる脅威）
4. 相手の四連を防ぐ手

### アルゴリズム（擬似コード）

```
function MCTS(root_state, iterations):
    root = MCTSNode(root_state)

    for i in 1..iterations:
        node = root
        state = root_state.copy()

        # Selection
        while node.untried_moves is empty and node.children is not empty:
            node = best_child(node, C)
            state.apply(node.move)

        # Expansion
        if node.untried_moves is not empty:
            move = random.choice(node.untried_moves)
            state.apply(move)
            node = node.add_child(move, state)

        # Simulation
        while state is not terminal:
            state.apply(random.choice(state.valid_moves))

        # Backpropagation
        while node is not None:
            node.visits += 1
            node.wins += reward(node.player, state.winner)
            node = node.parent

    return most_visited_child(root).move
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|------------|------------|------|
| simulations | 1000 | シミュレーション回数 |
| time_limit | None | 時間制限（秒、設定時はsimulationsより優先） |
| exploration | 1.414 | UCB1の探索係数 C |

### 特徴

**利点**:
- 評価関数が不要（ドメイン知識なしで動作）
- 非対称な探索木に強い
- 並列化が容易
- anytime algorithm（いつでも結果を返せる）

**欠点**:
- 戦術的な読みが苦手（深い詰みを見逃しやすい）
- 五目並べのような戦術ゲームでは単体では弱い

---

## 4. VCFBasedAI - 詰め探索

### 概要

五目並べ特有の「詰め」を探索するアルゴリズム。相手の応手が限定される脅威（四や三）を連続して打ち、強制的に勝利に導きます。

### 用語定義

| 用語 | 英語 | 説明 |
|------|------|------|
| 五連 | Five | 5つ連続した石（勝利） |
| 四 | Four | あと1手で五連になる形 |
| 達四 | Straight Four | 両端が空いた四（止められない） |
| 活三 | Open Three | 両端が空いた三（次に四を2箇所作れる） |
| 三 | Three | 片端が空いた三 |
| 脅威 | Threat | 次に五連を完成できる形 |

### VCF (Victory by Continuous Fours)

「四の連続による勝利」。四を打つと相手は必ず止めなければならないため、応手が1通りに限定されます。

#### 数理的背景

VCFは **AND-OR木** として表現できます。

- **OR節点（攻撃側）**: いずれかの子で勝てばよい
- **AND節点（防御側）**: すべての子で防げなければ負け

四を打った場合、防御点は1つしかないため、AND節点の分岐が1に限定されます。これにより探索空間が劇的に削減されます。

```
探索木の構造:
    攻撃（四を打つ）→ 防御（止める）→ 攻撃（次の四）→ ...
         OR              AND(分岐1)         OR
```

#### アルゴリズム

```
function VCF(board, stone, depth):
    if depth > max_depth: return False

    # 即勝ちチェック
    if has_winning_move(board, stone):
        return True

    # 全ての四を試す
    for four in find_fours(board, stone):
        new_board = board.apply(four.position)

        # 達四なら即勝ち
        if four.type == STRAIGHT_FOUR:
            return True

        # 防御点が1つの場合のみ探索
        if len(four.defense_points) == 1:
            defense = four.defense_points[0]
            new_board = new_board.apply(defense)

            if VCF(new_board, stone, depth + 1):
                return True

    return False
```

### VCT (Victory by Continuous Threats)

「脅威の連続による勝利」。VCFに加えて活三も使用します。VCFより強力ですが、探索空間が大きくなります。

#### 数理的背景

活三を打った場合、防御点は複数存在します。そのため、AND節点の分岐が増加します。

```
探索木の構造:
    攻撃（活三）→ 防御1 → VCF探索
         OR         防御2 → VCF探索
                    ...

    すべての防御に対してVCFが成立すれば勝ち
```

#### アルゴリズム

```
function VCT(board, stone, depth):
    # まずVCFを試す
    if VCF(board, stone): return True

    if depth > max_depth: return False

    # 全ての活三を試す
    for three in find_open_threes(board, stone):
        new_board = board.apply(three.position)

        # 全ての防御点に対してVCFが成立するか
        all_wins = True
        for defense in three.defense_points:
            defense_board = new_board.apply(defense)
            if not VCF(defense_board, stone):
                all_wins = False
                break

        if all_wins:
            return True

    return False
```

### 本実装の構成

```
VCFBasedAI
    │
    ├── ThreatDetector: 脅威パターン検出
    │      └── find_threats(): 盤面から四/三を検出
    │
    ├── VCFSearch: 四の連続探索
    │      └── search(): VCF詰みを探索
    │
    └── VCTSearch: 脅威の連続探索（オプション）
           └── search(): VCT詰みを探索
```

### ThreatDetector の実装

ウィンドウスライド方式で脅威を検出します。

```
盤面を4方向（横、縦、斜め×2）にスキャン
各方向で win_condition + 1 サイズのウィンドウをスライド

ウィンドウ内の構成:
  - 自石 n 個
  - 空き m 個
  - 相手石 0 個  ← 相手石があれば脅威にならない

パターン判定:
  - n ≥ 5      → FIVE（勝利）
  - n = 4, m ≥ 1 → FOUR or STRAIGHT_FOUR
  - n = 3, m ≥ 2 → THREE or OPEN_THREE
```

### パラメータ

| パラメータ | デフォルト | 説明 |
|------------|------------|------|
| use_vct | False | VCT探索も使用（遅くなるが強力） |
| max_depth | 20 | 最大探索深さ |
| time_limit | 1.0 | 時間制限（秒） |
| fallback | None | 詰みがない場合の代替戦略 |

### 特徴

**利点**:
- 人間が見落としがちな詰みを確実に発見
- 探索空間が限定されるため高速
- 五目並べでは非常に強力

**欠点**:
- 詰みがない局面では無力（フォールバック必須）
- 序中盤の形勢判断ができない

---

## 5. アルゴリズム比較

### 性能特性

| AI | 序盤 | 中盤 | 終盤 | 詰み | 計算量 |
|----|------|------|------|------|--------|
| Random | × | × | × | × | O(n) |
| Minimax | ○ | ○ | ○ | △ | O(b^d) |
| MCTS | ○ | ○ | △ | × | O(iterations) |
| VCF | × | × | ◎ | ◎ | 状況依存 |

### 推奨使用法

1. **初心者向け**: RandomAI, MinimaxAI(depth=2)
2. **中級者向け**: MinimaxAI(depth=3-4), MCTS(sims=500)
3. **上級者向け**: MinimaxAI(depth=4-5) + VCF補助
4. **最強構成**: MinimaxAI(iterative_deepening=True) + VCT

### 組み合わせ例

```python
# VCFをフォールバックに持つMinimax
vcf_ai = VCFBasedAI(use_vct=True)
minimax_ai = MinimaxAI(depth=4, use_iterative_deepening=True)

# 詰みがあればVCFで即発見、なければMinimaxで形勢判断
```

---

## 参考文献

1. Von Neumann, J. (1928). "Zur Theorie der Gesellschaftsspiele"
2. Knuth, D. E., & Moore, R. W. (1975). "An analysis of alpha-beta pruning"
3. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem"
4. Kocsis, L., & Szepesvári, C. (2006). "Bandit based Monte-Carlo Planning"
5. Allis, L. V. (1994). "Searching for Solutions in Games and Artificial Intelligence"
