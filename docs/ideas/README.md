# 创新方案集：基于事件相机的 4DGS 高速动态重建

> 前提：已有一个**静态高质量高斯泼溅（3DGS）重建结果**（外观/几何已基本收敛），
> 现需把**动态事件信息**融入 4DGS 做动态重建。
> 本目录的每份文档是一个独立设计，针对 ERF-GS 仓库给出挂载点与改动清单，供日后参考与选型。

## 统一前提：静态重建如何 bootstrap

- 静态重建给出 `t=0` 参考时刻的锐利外观与几何（xyz、SH、scale、opacity、rotation 全属性）。
- 4DGS 的形变场（HexPlane + MLP，`models/deformation/deformation.py`）**只需学"运动增量"**，
  而非从稀疏 COLMAP 冷启动学外观+运动——这是本方案集相对 ERF-GS 原作的最大优势。
- bootstrap 落地需要：全属性 PLY 读取器（仓库目前仅有稀疏 COLMAP ply 读取 `data/multiview.py:fetch_ply`）、
  形变场零初始化（使 `dx=0` 严格恒等，消除 startup perturb）。
  这些是所有 idea 共用的基础设施，详见各文档"实现方案"。

## 与 ERF-GS 现状的对标

| 维度 | ERF-GS 现状（局限） | 本方案集的改进方向 |
|---|---|---|
| 事件时间分辨率 | 体素化为 150 个时间 bin，窗口聚合（`utils/event_utils.py:69-83`）→ 丢掉微秒级分辨率 | Idea 1/3：连续时间 / 逐事件，保留每个事件的时间戳 |
| 事件监督形式 | `linlog` 强度差 + STE 量化 L1（`utils/loss_utils.py:137-142`），暗区近似、往复运动失配 | Idea 2/3：转光流 / 点过程，绕开 linlog 近似与量化地板 |
| 运动先验 | 帧内匀速直线正则（`gaussian_renderer/__init__.py:118-142`），对加速/旋转错 | Idea 1：连续速度场，由数据学任意运动模式 |
| 静态重建利用 | 从稀疏 COLMAP 点云冷启动，静态区也被形变场扰动可能劣化 | Idea 5：事件门控静动分解，静态区冻结零退化 |
| 运动模糊 RGB | 仅作 L1 监督，浪费了模糊里的运动信息 | （叠加项 Idea 4，未单列文档）模糊帧=曝光内运动积分约束 |

## 四个方案概览

| 文档 | 核心 | 静态重建角色 | 事件进入方式 | 可行性 | 学术新意 |
|---|---|---|---|---|---|
| [01](01_continuous_velocity_field.md) | 连续时间神经速度场 + 逐事件时间戳渲染 | 提供 `x(0)` 初始位置，速度场只学运动增量 | 逐事件时间戳渲染、log-luma 跨阈约束 | 中 | 高 |
| [02](02_event_to_flow_supervision.md) | 事件→2D 光流→监督高斯屏幕运动 | 提供外观，冻结/慢更新，事件只管"往哪动" | 事件光流匹配高斯屏幕位移 | 高 | 中 |
| [03](03_per_event_point_process_loss.md) | 逐事件点过程损失，不聚合 | 提供 `t=0` log-luma 锚，事件做增量检查 | 每个事件一条物理约束 | 中 | 高 |
| [05](05_event_gated_static_dynamic_split.md) | 事件活性门控静/动高斯分解 | 静态区高斯原样冻结，动态区挂形变 | 事件累计图生成静/动掩码 | 高 | 中 |

## 互斥 / 可叠加关系

```
              ┌── Idea 1 (连续速度场)  ──┐
事件时间利用 ──┤                          ├── 任选其一作"时间监督主干"
              └── Idea 3 (点过程)      ──┘
                                         ×  互斥：Idea 1/3 都改形变场时间建模，二选一
Idea 2 (光流) —— 独立的 2D 监督路线，与 1/3 的时间监督互补可叠加，但会重复建模运动
Idea 5 (静动分解) —— 与 1/2/3 任一都强可叠加：分解后只对动态区施加 1/2/3 的监督
```

**推荐组合**：`Idea 5（兑现静态重建价值，静态区零退化）+ Idea 1 或 3（兑现事件时间分辨率）`。
若 RGB 帧带运动模糊，叠加 Idea 4（模糊帧作曝光内积分约束）。

## 本目录文档不包含

- 实际训练/模型/渲染代码改动（文档给出挂载点与改动清单，是设计而非实现）
- 配置文件、实验脚本、实验结果
- Idea 4（去模糊）与 Idea 6（可微事件模拟器）的独立文档——作为增强项在各文档"叠加建议"中提及

## 仓库挂载点速查（所有文档共用，已逐行核实）

- 形变场：`models/deformation/deformation.py:142` `forward`、`:127` `forward_pos`、`:116` `query_time`（HexPlane 查询）、`:58-69` 各形变头 MLP
- HexPlane 网格：`models/deformation/hexplane.py:121-187`，时间分辨率 `resolution[3]`（默认 150）
- 渲染分支：`gaussian_renderer/__init__.py:89-100`（coarse 不调形变 / fine 调形变）、`:55-76`（时间参数来源）
- 事件损失：`train.py:205-222`、`utils/loss_utils.py:137-142` `eloss`、`utils/event_utils.py:111-129` `linlog/event_from_image`
- 事件数据：`data/data_structures.py` `CameraInfo.event_images [2,T,h,w]` / `event_times`；体素化 `utils/event_utils.py:69-83`
- 动态掩码：`data/data_structures.py` `CameraInfo.dynamic_mask`、`data/multiview.py` `with_dynamic_mask`/`dynamic_masks`
- 事件稠密化：`models/gaussians_4dgs.py:376-617` `event_densify`
- 高斯属性加载/保存：`models/gaussians_4dgs.py:57-101` `create_from_pcd`、`:198-210` `construct_list_of_attributes`、`:154-176` `load`、`:212-230` `save`
