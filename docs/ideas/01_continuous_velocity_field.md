# Idea 1：连续时间神经速度场 + 逐事件时间戳渲染

## 1. 动机与核心思想

**一句话**：用一个连续时间速度场 `v(x, t)` 描述高斯运动，每个高斯的轨迹由 ODE 积分得到；
事件相机**每个事件的时间戳**都对应一次精确时刻的渲染约束，彻底利用事件的微秒级时间分辨率。

ERF-GS 用 HexPlane 把时间离散成 150 个 bin（`hexplane.py:121-187`，`resolution[3]=150`），
形变靠网格插值——时间分辨率被 bin 钉死，无法对齐单个事件的时间戳。
本方案把"位置形变 `dx`"换成"速度场 `v`"，时间连续可微，可对任意 `t` 求位置。

## 2. 静态重建的角色

- 静态重建给出每个高斯在 `t=0` 的初始位置 `x(0)`（即 `_xyz`）。
- 速度场**只学运动增量** `v(x, t)`，轨迹 `x(t) = x(0) + ∫₀ᵗ v(x(τ), τ) dτ`。
- 因为只学小量运动（静态重建已解决外观/几何），积分不会漂移太远，ODE 数值稳定。
- 外观（SH、scale、opacity）沿用静态重建，可冻结或慢更新。

## 3. 事件信息如何进入模型

### 3.1 连续时间速度场

替换 `models/deformation/deformation.py` 的位移头：
- 原：`dx = pos_deform(hidden)`，`pts = x + dx`（`deformation.py:162-164`）
- 新：`v = velocity_field(hidden)`，`pts = x + ∫₀ᵗ v dτ`

`velocity_field` 仍由 HexPlane 特征 + MLP 输出，但输出语义从"位移"变"速度"。
积分用可微 ODE 求解器（torchdiffeq `odeint` 或自定义少量 Euler/RK 步）。

### 3.2 逐事件时间戳渲染

事件相机每个事件 `(x_img, y_img, t_e, polarity)`。对一批采样事件：
1. 把场景**渲染到精确时刻 `t_e`**（用积分得到 `x(t_e)`）；
2. 取该像素的 `log-luma`：`L(t_e) = linlog(render(t_e)[y, x])`（`utils/event_utils.py:111`）；
3. 与上一事件时刻 `t_prev` 的 `L(t_prev)` 比：约束 `L(t_e) − L(t_prev) ≈ polarity · 0.2`（事件阈值）。

**不做窗口聚合**——每个事件独立约束，保留微秒级时间分辨率。

### 3.3 损失形式

```
L_event = (1/N) Σ_e | (L(t_e) − L(t_prev)) / 0.2 − polarity |
```
对"该有事件却没触发"的像素，可加对比项（见 Idea 3 的点过程思路）。

## 4. 相对 ERF-GS 的新意

| ERF-GS 现状 | 本方案 |
|---|---|
| 时间离散 150 bin，事件被聚合（`event_utils.py:69-83`）→ 丢 99% 时间点 | 连续时间，逐事件时间戳，保留每个事件 |
| 形变靠 HexPlane 插值，bin 间靠线性假设 | ODE 积分，自然处理任意快/不均匀运动 |
| 直线运动正则假设匀速（`gaussian_renderer/__init__.py:118`），对加速错 | 速度场由数据学任意运动模式，无需匀速假设 |

## 5. 在 ERF-GS 仓库的具体实现方案

### 5.1 挂载点（已核实）
- 形变场位置头：`models/deformation/deformation.py:58-60` `pos_deform`、`:162-164` 位移应用
- HexPlane 查询：`deformation.py:120` `self.grid(...)`、`hexplane.py:164-187` `get_density/forward`
- 渲染时间参数：`gaussian_renderer/__init__.py:55-76`（`viewpoint_camera.time` / `event_times`）
- 事件损失调用：`train.py:205-222`

### 5.2 改动清单

**新增** `models/deformation/velocity_field.py`：
- `class VelocityField(nn.Module)`：输入 `(x, t)` → HexPlane 特征 → MLP → 输出 3D 速度 `v`。
- 复用现有 `HexPlaneField`（`hexplane.py`）做时空特征，仅改输出头语义。

**修改** `models/deformation/deformation.py`：
- `forward`（`:142`）：把 `pos_deform` 输出当速度，加 ODE 积分：
  ```python
  def forward(...):
      ...
      v = self.velocity_field(hidden)  # 速度，非位移
      # 从 t_first 积分到 times_sel
      pts = odeint(lambda tau: self.velocity_field(query(x, tau)),
                   x, torch.linspace(t_first, times_sel, num_steps))
      pts = pts[-1]  # 取终点
  ```
- 为效率：少量固定 Euler 步（如 4 步）而非自适应 odeint，保证 batch 可控。
- `forward_pos`（`:127`）同样改为积分版。

**修改** `gaussian_renderer/__init__.py`：
- `:55-76`：支持传入**任意事件时间戳** `t_e`（而非只 `event_times[event_idx+1]`），渲染到 `t_e`。
- 逐事件渲染：对采样事件子集，分别渲染 `t_prev` 和 `t_e`，取像素 log-luma 差。

**修改** `train.py:205-222`：
- 新增逐事件损失分支：从 `event_images` 反查原始事件流时间戳（需数据侧保留或加载 `.h5`），
  采样 N 个事件，按 §3.3 计算损失。

**修改** `utils/loss_utils.py`：新增 `per_event_loss(L_prev, L_e, polarity)`。

### 5.3 伪代码

```python
# velocity_field.py
class VelocityField(nn.Module):
    def __init__(self, grid_config):
        self.grid = HexPlaneField(...)      # 复用
        self.mlp = nn.Sequential(Linear(feat_dim, W), ReLU, Linear(W, 3))
    def forward(self, x, t):
        feat = self.grid(x, t)              # hexplane.py:183
        return self.mlp(feat)               # 速度 v(x,t)

# deformation.py forward（位置部分）
v = self.velocity_field(x, t_grid)          # 在若干中间时刻采样速度
pts = x + integrate_euler(v, x, t_grid, t_target)  # 积分得 x(t_target)

# train.py 逐事件损失
for event in sampled_events:                 # (x_img, y_img, t_e, pol)
    L_e = linlog(render(t_e)[y, x])
    L_prev = linlog(render(t_prev)[y, x])
    loss += abs((L_e - L_prev)/0.2 - pol)
```

## 6. 可行性评估

**中**。
- 速度场+ODE 是成熟范式（NeRF 动态场、Dynamic Gaussians 均有先例）；
- 静态 bootstrap 使积分稳定（只学小量）；
- 主要成本：逐事件渲染。每事件需两次光栅化（`t_prev`、`t_e`）。可采样事件子集（如每帧 256~1024 个）控制成本；
- ODE 积分用固定 Euler 步即可，无需 torchdiffeq 依赖。

## 7. 风险与边界

- **逐事件渲染成本**：单帧成千上万事件无法全渲染，必须采样；采样策略（均匀/按极性/按运动区）影响收敛。
- **ODE 积分梯度**：Euler 步可微，但步数少则轨迹近似差、步数多则显存/时间涨。需调步数（建议 4~8 步）。
- **t=0 恒等**：速度场初始需零输出（`velocity_field` 末层置零），否则 `x(t)` 从第 1 步就漂移。配合零初始化形变头。
- **事件时间戳来源**：`event_images` 是聚合体素图（丢了时间戳），逐事件需从原始 `.h5` 读时间戳（`data/multiview.py:214-235` 已有 h5 读取，但预转存时丢弃了 ts）。需保留时间戳或在线读 h5。
- **快速运动的大位移**：速度场需表达大速度，可能需更大 MLP 或残差结构。

## 8. 与其它 idea 的叠加建议

- **+ Idea 5（静动分解）**：强推荐。只对动态区高斯积分速度场，静态区 `v=0` 冻结，大幅降本且静态零退化。
- **+ Idea 4（去模糊）**：若 RGB 模糊，速度场积分的子轨迹平均 = 模糊帧，作额外观测。
- **与 Idea 3 互斥**：两者都改时间监督主干（连续速度 vs 点过程），二选一。
- **与 Idea 2 互补但会重复建模运动**：光流是 2D 监督、速度场是 3D 监督，可叠加但需权衡复杂度。
