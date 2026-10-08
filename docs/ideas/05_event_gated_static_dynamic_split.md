# Idea 5：事件门控的静/动高斯分解

## 1. 动机与核心思想

**一句话**：用事件**有无**做场景分割——有事件的像素是动态区，没事件是静态区；
静态区的 Gaussian 直接冻结（用静态重建，不学形变），动态区才挂形变场。

ERF-GS 让**所有**高斯都过形变场（`gaussian_renderer/__init__.py:97-100` fine 阶段对全部高斯调 `deformation`），
静态区也被形变场扰动，可能劣化已收敛的静态质量。
本方案显式分流：静态区高斯保留静态重建原样、零形变、零退化；
形变场容量全部集中到动态区，更易学快运动。
这是对"已有静态高质量重建"这一前提**最直接的兑现**。

## 2. 静态重建的角色

- 静态区高斯**原样保留**——颜色/SH/scale/opacity/rotation 全部冻结，质量不降。
- 这是本方案相对 ERF-GS 的核心价值：ERF-GS 从稀疏点冷启动，静态区也要学；
  本方案静态区已收敛，只需保护它。
- 动态区高斯从静态重建"分裂"或"继承"初始属性，挂形变场学运动。

## 3. 事件信息如何进入模型

### 3.1 事件活性→静/动掩码

事件相机对运动极度敏感（静止物体不触发事件）。
对每路相机每个时间窗，累计事件图 `E = |pos − neg|` 的和（`train.py:213` 已有类似 `event_mask`），
阈值化得**动态掩码** `M_dyn`：
```
M_dyn(cam, t) = (avg_pool3x3(Σ_events |event_images|) > 0.2)
```
（直接复用 `train.py:213` 的 `event_mask` 逻辑）。

### 3.2 高斯静/动标记

3D 高斯投影到各相机，看它落在动态掩码还是静态掩码：
- 若某高斯在**所有相机所有时刻**都落在静态掩码 → 标记为**静态高斯**（冻结）。
- 若在任意相机任意时刻落在动态掩码 → 标记为**动态高斯**（挂形变场）。
- 标记随训练更新（高斯移动后可能跨区），每隔若干迭代重算。

### 3.3 分流渲染与损失

- **静态高斯**：渲染时直接用原始属性，不调形变场（`pts = x`，`dx = 0`）。
- **动态高斯**：调形变场 `pts = x + dx`。
- 形变场 `forward`（`deformation.py:142`）接收一个 per-Gaussian `is_dynamic` 掩码，
  静态高斯的 `dx/ds/dr` 强制置零。

## 4. 相对 ERF-GS 的新意

| ERF-GS 现状 | 本方案 |
|---|---|
| 所有人高斯过形变场，静态区被扰动可能劣化（`gaussian_renderer/__init__.py:97-100`） | 静态区冻结，零退化 |
| 形变场容量分散到全场景 | 容量集中到动态区，更易学快运动 |
| 动态掩码仅在 `metrics.py` 评估用（`metrics.py:100-127`），训练未用 | 训练时用事件活性驱动分解 |
| 形变场搜索空间大 | 静态区不优化，搜索空间骤降 |

## 5. 在 ERF-GS 仓库的具体实现方案

### 5.1 挂载点（已核实）
- 形变场应用：`models/deformation/deformation.py:142-194` `forward`（`pts = x*mask + dx`，`:164`）
- 渲染分支：`gaussian_renderer/__init__.py:97-100`（fine 调形变）
- 动态掩码已有字段：`data/data_structures.py` `CameraInfo.dynamic_mask`、
  `data/multiview.py` `with_dynamic_mask`/`dynamic_masks`（当前从 RGB 掩码图读，可改为事件生成）
- 事件掩码逻辑：`train.py:213` `event_mask`
- 高斯属性管理：`models/gaussians_4dgs.py`（新增 per-Gaussian 静/动标记张量）

### 5.2 改动清单

**新增** `models/gaussians_4dgs.py`：
- 高斯模型新增 `self._is_dynamic = torch.zeros(N, 1, dtype=bool)` 标记张量。
- 新增 `update_dynamic_label(cameras, event_masks)`：投影高斯到各相机，
  与动态掩码比对，更新 `_is_dynamic`（每隔 K 迭代调用）。
  可复用 `event_densify`（`:376-617`）里的投影逻辑（`gaussians_4dgs.py:475-507`）。
- `_is_dynamic` 不入优化器（非可训练参数），随标记更新。

**修改** `models/deformation/deformation.py:142-194` `forward`：
- 接收 `is_dynamic` 掩码，静态高斯强制 `dx=ds=dr=0`：
  ```python
  def forward(self, point, ..., times_sel, is_dynamic=None):
      ...
      dx = self.pos_deform(hidden)
      if is_dynamic is not None:
          dx = dx * is_dynamic          # 静态高斯 dx=0
      pts = point_emb[:, :3] + dx       # 简化 mask=ones 情况
      # scales/rotations 同理乘 is_dynamic
  ```

**修改** `gaussian_renderer/__init__.py:97-100`：
- fine 渲染时把 `gaussians._is_dynamic` 传入 `deformation`。

**修改** `train.py`：
- 在 `densify`/`event_densify` 后调用 `gaussians.update_dynamic_label(...)`（每 K 迭代）。
- 事件掩码可复用 `:213` 的 `event_mask`，多视角聚合得 per-Gaussian 标记。

**修改** `data/multiview.py`：
- `with_dynamic_mask=True` 时，若用事件生成掩码而非读掩码图，加一个 `event_dynamic_mask` 路径。
  （评估侧 `metrics.py:100-127` 仍用 RGB 掩码图，二者解耦。）

### 5.3 伪代码

```python
# deformation.py forward
def forward(self, point, ..., times_sel, is_dynamic=None):
    hidden = self.query_time(point_emb, times_sel)
    dx = self.pos_deform(hidden)
    if is_dynamic is not None:
        dx = dx * is_dynamic.float()        # 静态高斯 dx 置零
    pts = point_emb[:, :3] + dx

# gaussians_4dgs.py 标记更新
def update_dynamic_label(self, cameras, event_masks):
    # 投影每个高斯到各相机，查 event_mask
    for cam, emask in zip(cameras, event_masks):
        proj = project(self._xyz, cam)
        hit_dyn = emask[proj.y, proj.x] > 0
        self._is_dynamic |= hit_dyn         # 任一相机命中即动态
    # 静态高斯（从未命中）保持 False
```

## 6. 可行性评估

**高**。
- 事件对运动极度敏感，静/动掩码天然准确（比 RGB 动态掩码更准）；
- 仓库已有 `dynamic_mask` 字段与评估逻辑（`metrics.py:100-127`），概念成熟；
- 改动集中在形变场加一个门控掩码 + 标记更新，不触碰损失/渲染核心；
- 静态区冻结直接兑现"静态高质量重建"价值，是所有 idea 里风险最低、收益最确定的。

## 7. 风险与边界

- **静/动边界高斯**：跨在边界的高斯被标动态，但大部分在静态区，形变场可能给它非零 `dx` 扰动静态区。
  缓解：用软门控（`is_dynamic` 为 [0,1] 连续值，按掩码覆盖率）而非硬二值。
- **标记滞后**：高斯移动后跨区，标记需定期更新。更新频率（每 200~500 迭代）权衡成本与准确性。
- **新分裂高斯**：`event_densify`（`:376-617`）补的新高斯默认标动态（它本就是为运动补的）。
- **静态区仍有轻微运动**：如相机轻微抖动。纯冻结可能丢失。可给静态高斯一个极小的全局刚性变换头（6-DoF）而非完全冻结。
- **多视角不一致**：某高斯在 A 视角静态、B 视角动态（遮挡边界）。用"任一动态即动态"的保守策略。
- **冷启动标记**：训练初期事件掩码可能不准（渲染未收敛）。可前若干迭代全标动态，待事件损失稳定后再分解。

## 8. 与其它 idea 的叠加建议

- **与 Idea 1/2/3 都强可叠加**：本方案是"在哪施加监督"的分流，不规定监督形式。
  - `+ Idea 1`：只对动态区积分速度场，静态区 `v=0`，大幅降本。
  - `+ Idea 3`：只在动态区采样点过程事件，静态区无事件天然不约束。
  - `+ Idea 2`：光流为零处即静态区，与本方案掩码一致，天然兼容。
- **推荐作为基础层**：Idea 5 风险最低、最直接兑现静态重建价值，建议作为所有方案的**基础**，
  在其上叠加 Idea 1 或 3 作时间监督主干。
- **+ Idea 4（去模糊）**：静态区无运动无模糊，模糊约束只在动态区生效，与分解一致。
