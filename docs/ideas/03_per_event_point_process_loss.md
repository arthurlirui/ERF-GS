# Idea 3：逐事件点过程损失（无聚合）

## 1. 动机与核心思想

**一句话**：把事件流当作**点过程**而非图像——每个事件是一条物理约束
"在 `(x_img, y_img, t_e)` 处，log-luma 跨越了阈值 ±0.2"，不 voxel 化、不窗口求和。

ERF-GS 把事件聚合成 2 通道体素图（`utils/event_utils.py:69-83` `events_to_voxel`），
窗口内所有事件压成一张图——丢掉了 99% 的时间点。
更严重的是，GT 计的是窗口内**总活动量**（来回振荡的亮度会不断触发脉冲），
而预测只看**端点净变化**（`event_from_image` 的 log 差），对往复运动系统性失配。
本方案逐事件约束，每次亮度跳变都被独立记录，往复振荡不再是问题。

## 2. 静态重建的角色

- 静态重建给出 `t=0` 的**精确 log-luma 基准** `L(0)`（渲染静态高斯即得）。
- 每个事件都相对这个基准做增量检查：`L(t_e) − L(0) ≈ 累计阈值跨越`。
- 静态结果是整个时间序列的"锚"，事件只验证"相对锚变了多少"——只学运动增量。

## 3. 事件信息如何进入模型

### 3.1 事件作为点过程

原始事件流（`data/multiview.py:214-223` 从 h5 读 `ts, xs, ys, ps`）：
每个事件 `(x_img, y_img, t_e, p)`，`p∈{+1,−1}`。
物理含义：在 `t_e` 时刻，像素 `(x_img, y_img)` 的 `log-luma` 跨越了 `±0.2` 阈值，方向 `p`。

### 3.2 逐事件约束

对一批采样事件，渲染精确时刻 `t_e`，取像素 log-luma `L(t_e) = linlog(render(t_e)[y, x])`。
- **正事件**（`p=+1`）：约束 `L(t_e) − L(t_last_pos) ≥ +0.2`（自上一个同极性事件以来增亮一档）。
- **负事件**：约束 `L(t_e) − L(t_last_neg) ≤ −0.2`。
- **未触发约束**：对"该有事件却没触发"的像素/时刻，施加软惩罚（对比损失）——
  即 `|L(t_e) − L(t_prev)|` 应小（没跨阈）。

### 3.3 损失形式

```
# 触发事件：log-luma 变化应匹配极性
L_trig = (1/N) Σ_e ReLU(0.2 − p·(L(t_e) − L(t_prev)))

# 未触发（采样无事件处）：变化应小于阈值
L_notrig = (1/M) Σ_m ReLU(|L(t_m) − L(t_prev_m)| − 0.2)

L_pp = L_trig + λ_notrig · L_notrig
```
`L(t)` 用 `linlog`（`utils/event_utils.py:111`）保留与事件相机一致的对数响应。

## 4. 相对 ERF-GS 的新意

| ERF-GS 现状 | 本方案 |
|---|---|
| 事件聚合为体素图（`event_utils.py:69-83`），丢时间分辨率 | 逐事件点过程，保留每个事件时间戳 |
| GT=总活动量 vs 预测=端点净变化，往复运动失配 | 每次跳变独立约束，往复振荡成立 |
| `linlog`+STE 量化 L1，有量化地板 | 物理阈值约束（ReLU margin），无量化地板 |
| 时间被困在 150 bin | 任意 `t_e`，时间连续 |

## 5. 在 ERF-GS 仓库的具体实现方案

### 5.1 挂载点（已核实）
- 事件原始数据：`data/multiview.py:214-223`（h5 读取 `ts, xs, ys, ps`，预转存时 `ts` 被丢弃，需保留）
- 事件损失调用：`train.py:205-222`
- 渲染时间参数：`gaussian_renderer/__init__.py:55-76`（需支持任意 `t_e`）
- linlog：`utils/event_utils.py:111`

### 5.2 改动清单

**修改** `data/multiview.py:214-249`：
- 预转存时**保留事件时间戳**（当前只存聚合体素图 `.bin`，丢 `ts`）。
- 方案：每个时间窗额外存 `events_ts.bin`（时间戳数组）+ `events_xy_pol.bin`（像素+极性），
  或在线从 `.h5` 读（牺牲 IO 速度）。
- `CameraInfo`（`data_structures.py`）新增 `event_timestamps`/`event_coords` 字段。

**修改** `gaussian_renderer/__init__.py:55-76`：
- `render` 支持传入**任意标量时间** `t_e`（而非只 `event_times[event_idx+1]`），
  渲染到该时刻（形变场 forward 用 `t_e`）。

**修改** `train.py:205-222`：
- 新增点过程损失分支：
  - 从 `event_timestamps` 采样 N 个事件 + M 个无事件采样点；
  - 对每个，渲染 `t_e`（和 `t_prev`），取像素 log-luma；
  - 按 §3.3 计算 `L_pp`。
- 权重 `lambda_pp` 配置化。

**新增** `utils/loss_utils.py` `point_process_loss(L_e, L_prev, polarity, thr=0.2)`。

### 5.3 伪代码

```python
# 采样事件（触发）和无事件点
triggered = sample_events(cam.event_timestamps, cam.event_coords, N)  # (x,y,t,p)
not_triggered = sample_no_event_positions(cam, M)

L_prev = linlog(render(triggered.t_prev)[y, x])
L_e = linlog(render(triggered.t_e)[y, x])
L_trig = relu(0.2 - triggered.p * (L_e - L_prev)).mean()

L_m = linlog(render(not_triggered.t_m)[y, x])
L_notrig = relu(abs(L_m - L_prev) - 0.2).mean()

loss_pp = L_trig + lambda_notrig * L_notrig
```

## 6. 可行性评估

**中**。
- 点过程损失是事件相机最原生、最物理的监督形式，学术新意最高；
- 静态 bootstrap 提供 `L(0)` 锚，只需学运动增量；
- 主要成本：逐事件渲染。需采样事件子集（如每帧 512~2048 个）+ 无事件采样点；
- 渲染单像素 log-luma 可用增量近似（只重算受影响高斯），降低成本。

## 7. 风险与边界

- **事件时间戳保留**：当前预转存丢弃 `ts`（`data/multiview.py:223` 后只用 `xs, ys, ps`）。
  需改预转存保留时间戳，或在线读 h5（IO 慢）。这是最大工程改动。
- **逐事件渲染成本**：单帧上万事件无法全渲染，必须采样。采样策略（按时间均匀/按极性平衡/按运动区密集）影响收敛。
- **`t_prev` 定义**：是上一个同极性事件、上一个任意事件、还是 `t=0`？影响约束语义。建议用"上一个同像素事件"最贴近物理，但需维护每像素事件历史。
- **未触发采样**：选哪些"无事件点"是开放问题；可在事件稀疏区均匀采样。
- **linlog 暗区近似**：仍用 `linlog`，暗区误差还在（可叠加 Idea 6 可微模拟器缓解）。
- **遮挡与视点**：某像素在 `t_e` 被新物体遮挡，其 log-luma 变化来自遮挡而非运动，会误导。需多视角交叉或深度门控。

## 8. 与其它 idea 的叠加建议

- **+ Idea 5（静动分解）**：强推荐。只在动态区（事件活跃）采样点过程，静态区无事件天然不约束，降本。
- **+ Idea 1 互斥**：两者都改时间监督主干。Idea 1 用连续速度场+积分，Idea 3 用逐事件阈值约束——哲学不同，二选一。
  - 选 Idea 1：若看重轨迹连续性、ODE 可解释。
  - 选 Idea 3：若看重事件物理保真、往复运动。
- **+ Idea 2（光流）**：点过程（3D 强度）+ 光流（2D 运动）互补，可叠加但复杂度高。
- **+ Idea 4（去模糊）**：点过程约束子轨迹强度，模糊帧约束子轨迹平均，天然兼容。
