# 事件相机 × 高斯泼溅（Event × Gaussian Splatting）研究综述与创新 Idea

> 整理时间：2026-08-31。覆盖 2024–2026 年"事件相机 + 3D/4D Gaussian Splatting"方向的核心论文，
> 按场景分类总结创新点，附 BibTeX，并基于现有工作的空白给出 5 个新的创新 idea。

---

## 一、论文清单与创新点总结

按"解决的子问题"分类。同一论文可能跨类。

### A. 静态场景 Novel View Synthesis（事件驱动 3DGS 重建）

| # | 论文 | 发表 | 核心创新 |
|---|---|---|---|
| A1 | **EventSplat**: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering | CVPR 2025 | 首个用移动事件相机做 3DGS NVS 的方法。用 event-to-video 模型（E2VID）初始化点云，用样条插值在事件相机轨迹上取高质量位姿。比 event-NeRF 快一个数量级渲染。**场景静态、相机快速运动**。 |
| A2 | **EF-3DGS**: Event-Aided Free-Trajectory 3D Gaussian Splatting | NeurIPS 2025 (Spotlight) | 首次把事件相机引入"自由轨迹"3DGS 重建（无精确位姿）。三组件：(1) Event Generation Model 融合事件与帧做帧间连续监督；(2) Contrast Maximization(CMax) 校准位姿 + 梯度域约束 3DGS；(3) Photometric BA + Fixed-GS 分离结构与颜色优化。高速场景 PSNR +3dB、ATE -40%。 |
| A3 | **EvTrajGS**: Accurate and Efficient 3D Gaussian Splatting from Unposed Event Streams | arXiv 2026-08 | 连续时间轨迹参数化（CTPF）把离散位姿拟合成连续函数，再用时间耦合位姿集成（TCPI）聚合相邻位姿，做无位姿事件 3DGS 联合优化。避免 SLAM 式增量跟踪的高开销，比 IncEventGS 快 7×。 |
| A4 | **GS-EVT**: Cross-Modal Event Camera Tracking Based on Gaussian Splatting | ICRA 2025 | 跨模态：用 RGB 帧建 3DGS 地图，用事件相机做跟踪。参考位姿 + 一阶动力学的局部差分图像渲染，与积分事件图比对，粗到细优化。 |

### B. 运动模糊去模糊（事件辅助 3DGS 从模糊图像重建）

| # | 论文 | 发表 | 核心创新 |
|---|---|---|---|
| B1 | **EvaGaussians**: Event Stream Assisted Gaussian Splatting from Blurry Images | ICCV 2025 | 事件流融入 3DGS 的初始化与优化，从运动模糊图像学高质量 3DGS。贡献两个新数据集（RGB+事件+位姿）。 |
| B2 | **E2GS**: Event Enhanced Gaussian Splatting | arXiv 2024 | 首个把事件数据纳入 Gaussian Splatting 的工作。结合模糊图像 + 事件做去模糊 NVS，140 FPS 渲染，比 E²NeRF 训练快 60×、渲染快 3500×。用 EDI 模型把曝光分成 N 个 event bin 估计中间清晰帧。 |
| B3 | **EBAD-Gaussian**: Event-driven Bundle Adjusted Deblur Gaussian Splatting | arXiv 2025 | 事件驱动的 bundle adjust 去模糊 3DGS。合成曝光内多清晰帧构造模糊 loss，事件监督任意时刻亮度变化，EDI 先验约束中间帧。联合学高斯参数 + 恢复曝光内相机轨迹。 |
| B4 | **EvFlow-GS**: Event Enhanced Motion Deblurring with Optical Flow for 3DGS | ICME 2026 | 联合优化可学双积分(LDI)+位姿+3DGS。用光流 warp 事件对齐运动轨迹得到锐利监督；event-residual prior 替代 log-intensity 监督（明确批评 linlog 监督四大缺陷：噪声放大、阈值恒定、低强度区退化、匀速假设）。Bézier 9 控制点 SE(3) 轨迹建模相机运动。 |
| B5 | **JADE-GS**: Joint Allocation of Deblurring Evidence for Event-Assisted 3DGS | arXiv 2026-07 | 把"EDI 解析反演"与"学习式帧+事件复原"两类先验视为两种证据，用轻量 Spatial Prior Router 做像素级分配融合成额外监督目标。Router 无需清晰参考帧训练，优化后移除。 |
| B6 | **E2D-GS**: Event Enhanced 高斯泼溅去模糊 | Expert Systems with Applications 2025 | 事件流对运动模糊做物理建模，差分一致性模块去噪 + 正则化，联合优化高斯参数与相机轨迹。 |

### C. 低光照 / HDR（事件辅助 3DGS 在极端光照下）

| # | 论文 | 发表 | 核心创新 |
|---|---|---|---|
| C1 | **Dark-EvGS**: Event Camera as an Eye for Radiance Field in the Dark | arXiv 2025 (TPAMI 投稿) | 首个事件引导的暗光 3DGS 亮帧合成。Triplet-level supervision（整体结构+细粒度细节+锐利渲染三层）；Color Tone Matching Block 保证颜色一致性；首个真实暗光事件-辐射场数据集。 |
| C2 | **EvHDR-GS**: Event-guided HDR Video Reconstruction with 3DGS | AAAI 2026 | 把 HDR 视频重建转成 HDR 场景重建，用 3DGS 的 3D 一致性保证时间亮度一致。HDR 3D Gaussians 同时表示 HDR/LDR 颜色；可学 HDR-to-LDR 变换由事件流自监督学习，消除数据偏差。 |

### D. 动态场景 / 4D 重建（事件驱动形变高斯）

| # | 论文 | 发表 | 核心创新 |
|---|---|---|---|
| D1 | **FastEventDGS**: Deformable Gaussian Splatting for Fast Dynamic Scenes from a Single Event Camera | CVPR 2026 | **首个仅用单目事件相机训练 Deformable 3DGS 做动态场景**。连续 B 样条轨迹参数化实现任意时刻事件监督；双事件生成模型（长区间积分 ESI + 线性流模型）；local patch motion loss 把高斯轨迹钉到事件边缘抑制过拟合；VGGT 专家深度细化。PSNR 从 ~16dB 提到 22–24dB。 |
| D2 | **ERF-GS**: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints | CVMJ (in production) | 多目 disjoint 事件-RGB 视点重建快速运动。linlog+STE 事件损失；直线运动/恒色正则；事件驱动稠密化（事件像素多目反投影补新高斯）。本仓库基线。 |
| D3 | **PEGS**: Physics-Event Enhanced Large Spatiotemporal Motion Reconstruction via 3DGS | arXiv 2025-11 | 物理先验 + 事件增强做大时空尺度刚体运动重建。三层监督：加速度一致性约束（牛顿动力学）、事件流高时间分辨率引导、Kalman 正则化融合多源观测。运动感知模拟退火策略。首个自然快速刚体运动的 RGB-Event 配对数据集。 |

### E. SLAM / 跟踪（事件 + 3DGS 定位建图）

| # | 论文 | 发表 | 核心创新 |
|---|---|---|---|
| E1 | **MotionGS-SLAM**: Event-Modulated Gaussian Splatting for Motion-Blur Robust SLAM | arXiv 2026-08 | 把运动模糊当作"正向问题"而非"逆向去模糊"——在渲染器内物理建模模糊形成。事件调制高斯核：空间调制把各向同性点变成运动方向对齐的各向异性椭圆笔触；时间调制按局部速度调整曝光积分采样密度。4D 时空 hash grid 关联事件与高斯。 |
| E2 | (见 A4 GS-EVT) | ICRA 2025 | 跨模态事件相机跟踪。 |

### F. 数据集 / 仿真（用 3DGS 生成事件数据）

| # | 论文 | 发表 | 核心创新 |
|---|---|---|---|
| F1 | **GS2E**: Gaussian Splatting is an Effective Data Generator for Event Stream Generation | NeurIPS 2025 | 用 3DGS 重建真实稀疏多视角 RGB 场景，再用物理信息事件仿真管线（自适应轨迹插值 + 物理一致对比阈值建模）生成大规模多视角事件数据集（1150+ 场景）。解决事件数据集稀缺与几何不一致问题。 |

---

## 二、按"技术维度"的横向归纳

为便于找空白，把上述论文用到的关键技术抽出来：

| 技术要素 | 采用的论文 | 现状 |
|---|---|---|
| **事件→图像监督（EDI/ESI 积分）** | E2GS, EBAD-Gaussian, EvaGaussians, FastEventDGS, ERF-GS | 成熟，但 EvFlow-GS 已批评其缺陷 |
| **事件→光流/运动监督** | EvFlow-GS, EF-3DGS(CMax), FastEventDGS(event flow loss) | 兴起，避开强度域监督的噪声问题 |
| **连续相机轨迹（B样条/Bézier）** | EvTrajGS, FastEventDGS, EvFlow-GS, EBAD-Gaussian | 已成标准做法，用于任意时刻事件监督 |
| **形变场 / Deformable 3DGS** | FastEventDGS, ERF-GS | 事件驱动形变场刚起步（仅 2 篇） |
| **物理约束（加速度/动力学）** | PEGS | 仅 PEGS 一篇，且限刚体 SE3 |
| **运动模糊正向建模** | MotionGS-SLAM, EvFlow-GS, EBAD-Gaussian | 模糊帧当观测而非噪声 |
| **HDR / 暗光** | Dark-EvGS, EvHDR-GS | 两个方向各一篇 |
| **遮挡处理** | （无） | **空白**——所有论文都不显式处理遮挡 |
| **静态先验 bootstrap** | （无） | **空白**——所有论文都从稀疏点/事件冷启动 |
| **静/动分解** | （事件驱动方向无） | DeGauss/DSD-GS 做了但用 RGB/光流，非事件驱动 |

---

## 三、BibTeX

```bibtex
% ===== A. 静态 NVS =====
@inproceedings{yura2025eventsplat,
  title={EventSplat: 3D Gaussian Splatting from Moving Event Cameras for Real-time Rendering},
  author={Yura, Toshiya and Mirzaei, Ashkan and Gilitschenski, Igor},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={26876--26886},
  year={2025}
}

@inproceedings{liao2025ef3dgs,
  title={EF-3DGS: Event-Aided Free-Trajectory 3D Gaussian Splatting},
  author={Liao, Bohao and Zhai, Wei and Wan, Zengyu and Cheng, Zhixin and Yang, Wenfei and Cao, Yang and Zhang, Tianzhu and Zha, Zheng-Jun},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025},
  note={Spotlight}
}

@article{chen2026evtrajgs,
  title={EvTrajGS: Accurate and Efficient 3D Gaussian Splatting from Unposed Event Streams},
  author={Chen, Zixuan and Zhang, Jiakai and Dong, Junhao and Wang, Guangcong and Lai, Jianhuang and Ong, Yew-Soon and Xie, Xiaohua},
  journal={arXiv preprint arXiv:2608.08585},
  year={2026}
}

@inproceedings{liu2025gsevt,
  title={GS-EVT: Cross-Modal Event Camera Tracking Based on Gaussian Splatting},
  author={Liu, Tao and Yuan, Runze and Ju, Yi'ang and Xu, Xun and Yang, Jiaqi and Meng, Xiangting and Lagorce, Xavier and Kneip, Laurent},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2025}
}

% ===== B. 去模糊 =====
@inproceedings{yu2025evagaussians,
  title={EvaGaussians: Event Stream Assisted Gaussian Splatting from Blurry Images},
  author={Yu, Wangbo and Feng, Chaoran and Li, Jianing and Tang, Jiye and Yang, Jiashu and Tang, Zhenyu and Cao, Meng and Jia, Xu and Yang, Yuchao and Yuan, Li and Tian, Yonghong},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={24780--24790},
  year={2025}
}

@article{deguchi2024e2gs,
  title={E2GS: Event Enhanced Gaussian Splatting},
  author={Deguchi, Hiroyuki and Masuda, Mana and Nakabayashi, Takuya and Saito, Hideo},
  journal={arXiv preprint arXiv:2406.14978},
  year={2024}
}

@article{deng2025ebad,
  title={EBAD-Gaussian: Event-driven Bundle Adjusted Deblur Gaussian Splatting},
  author={Deng, Yufei and Wang, Yuanjian and Xiao, Rong and Tang, Chenwei and Zhou, Jizhe and Fan, Jiahao and Xiong, Deng and Lv, Jiancheng and Tang, Huajin},
  journal={arXiv preprint},
  year={2025}
}

@inproceedings{an2026evflow,
  title={EvFlow-GS: Event Enhanced Motion Deblurring with Optical Flow for 3D Gaussian Splatting},
  author={An, Feiyu and Deng, Yufei and Zhang, Zihui and Xiao, Rong},
  booktitle={IEEE International Conference on Multimedia and Expo (ICME)},
  year={2026}
}

@article{2026jadegs,
  title={JADE-GS: Joint Allocation of Deblurring Evidence for Event-Assisted 3D Gaussian Splatting},
  author={Anonymous},
  journal={arXiv preprint arXiv:2607.12362},
  year={2026},
  note={arXiv 2026-07}
}

@article{2025e2dgs,
  title={E2D-GS: Event-Enhanced Gaussian Splatting Deblurring},
  author={Anonymous},
  journal={Expert Systems with Applications},
  year={2025}
}

% ===== C. 暗光 / HDR =====
@article{wu2025darkevgs,
  title={Dark-EvGS: Event Camera as an Eye for Radiance Field in the Dark},
  author={Wu, Jingqian and Duan, Peiqi and Wang, Zongqiang and Wang, Changwei and Shi, Boxin and Lam, Edmund Y.},
  journal={arXiv preprint arXiv:2507.11931},
  year={2025}
}

@inproceedings{chen2026evhdrgs,
  title={EvHDR-GS: Event-guided HDR Video Reconstruction with 3D Gaussian Splatting},
  author={Chen, Zehao and Lu, Zhan and Ma, De and Tang, Huajin and Jiang, Xudong and Zheng, Qian and Pan, Gang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}

% ===== D. 动态 / 4D =====
@inproceedings{2026fasteventdgs,
  title={FastEventDGS: Deformable Gaussian Splatting for Fast Dynamic Scenes from a Single Event Camera},
  author={Anonymous},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}

@article{bai2026erfgs,
  title={ERF-GS: Reconstructing Fast Motion from Disjoint Event-RGB Viewpoints},
  author={Bai, Xiaoyang and Li, Zhenyang and Xu, Weiwei and Lam, Edmund Y. and Peng, Yifan},
  journal={Computational Visual Media (CVMJ)},
  year={2026},
  note={in production}
}

@article{xu2025pegs,
  title={PEGS: Physics-Event Enhanced Large Spatiotemporal Motion Reconstruction via 3D Gaussian Splatting},
  author={Xu, Yijun and Zhang, Jingrui and Liu, Hongyi and Chen, Yuhan and Wang, Yuanyang and Guo, Qingyao and Wang, Dingwen and Yu, Lei and He, Chu},
  journal={arXiv preprint arXiv:2511.17116},
  year={2025}
}

% ===== E. SLAM =====
@article{hu2026motiongs,
  title={MotionGS-SLAM: Event-Modulated Gaussian Splatting for Motion-Blur Robust SLAM},
  author={Hu, Zhiqiang and Huang, Shouren and Ishikawa, Masatoshi},
  journal={arXiv preprint arXiv:2608.15024},
  year={2026}
}

% ===== F. 数据集 =====
@inproceedings{li2025gs2e,
  title={GS2E: Gaussian Splatting is an Effective Data Generator for Event Stream Generation},
  author={Li, Yuchen and Feng, Chaoran and Tang, Zhenyu and Deng, Kaiyuan and Yu, Wangbo and Tian, Yonghong and Yuan, Li},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2025}
}
```

---

## 四、5 个有创新性的 Idea

> 基于上述综述识别的空白（遮挡处理、静态先验 bootstrap、物理约束 × 事件、静动分解 × 事件、可微事件模拟）。
> 每个 idea 指明：填补的空白、核心思路、与现有工作的差异、可行性。

### Idea ① 事件驱动的遮挡感知 4DGS：用事件"出现/消失"边界检测遮挡并冻结被遮挡高斯

**填补空白**：现有所有 Event×GS 论文（A–F 类）都不显式处理遮挡。ERF-GS/FastEventDGS 假设高斯始终可见，遮挡时被遮挡高斯会因缺少监督而漂移或被错误剪枝。

**核心思路**：事件相机对运动敏感——物体被遮挡时该像素**事件停止触发**，物体重新出现时**事件重新爆发**。用事件活跃度的"出现/消失"边界在线检测遮挡区间；遮挡期间冻结被遮挡高斯（不学形变、不参与稠密化），解除遮挡后恢复优化。这把"遮挡"从麻烦变成事件相机独有的观测信号。

**与现有差异**：没有任何 Event×GS 工作做遮挡。OASIS（ACM MM 2026）做遮挡感知但是手部 avatar 重建、用 RGB 可见性，非事件、非动态场景。本 idea 是事件相机**独有的能力**（RGB 无法区分"物体消失"和"被遮挡"，事件可以——静止被遮挡物不触发事件）。

**可行性**：中高。事件活跃度阈值化检测遮挡边界是成熟信号；冻结/解冻高斯是工程实现。难点是"被遮挡"vs"静止"的区分（两者都不触发事件），需用多视角交叉或运动先验。

---

### Idea ② 静态高质量高斯 bootstrap 的 4DGS：从已收敛静态 3DGS 出发只学运动增量

**填补空白**：现有所有 Event×GS 动态工作（FastEventDGS、ERF-GS、PEGS）都从稀疏 COLMAP 点云冷启动，外观+运动+位姿同时学，静态区也被形变场扰动可能劣化。**没有工作利用"已有静态高质量高斯重建"这一先验**。

**核心思路**：用全属性 3DGS `.ply`（xyz+SH+scale+opacity+rotation）初始化 `GaussianModel`，形变场零初始化（`dx=0` 严格恒等），fine 阶段才启用形变+事件损失。事件只学"运动增量"，外观/几何由静态重建锁定。配合事件门控静动分解（静态区完全冻结）。

**与现有差异**：所有动态 Event×GS 都冷启动。SPIN-4DGS（非事件）揭示了冷启动下快速运动物体"消失"的失败模式——静态 bootstrap 恰好解决这个（静态区已收敛不参与竞争，动态区有更多优化空间）。

**可行性**：高。本仓库已确认 `GaussianModel` 无全属性 PLY 加载器（需新增），形变场零初始化是单点改动。风险最低、最直接兑现"已知静态高质量重建"前提。

---

### Idea ③ 逐事件点过程损失：把事件流当作 marked temporal point process 直接监督可微渲染

**填补空白**：现有所有 Event×GS 把事件**聚合**成 voxel 图/事件帧（ERF-GS 的 `events_to_voxel`、FastEventDGS 的 ESI 积分），丢掉微秒级时间分辨率。更严重的是 GT 计"窗口总活动量"而预测只看"端点净变化"，对往复振荡运动系统性失配。

**核心思路**：事件流是 marked temporal point process——每个事件 `(x,y,t,p)` 是一条物理约束"log-luma 跨越 ±0.2 阈值"。不聚合，逐事件约束：渲染精确时刻 `t_e` 取像素 log-luma，要求相对上一事件时刻的变化匹配极性。对"该有事件却没触发"的像素加对比项。点过程似然 + 可微 splatting 的组合在重建领域是空白（控制领域有 arXiv:2603.05011 做系统辨识但非 3DGS）。

**与现有差异**：ERF-GS 用 linlog+STE 聚合监督（EvFlow-GS 已批评其四大缺陷）；FastEventDGS 用 ESI 积分仍是聚合。本 idea 是事件相机最原生、最物理的监督形式，学术新意最高。

**可行性**：中。需保留事件时间戳（现有预转存丢弃了 `ts`）、逐事件采样、可微 point-process likelihood。工程量大但思路清晰。

---

### Idea ④ 物理约束的事件驱动 4DGS：加速度一致性 + 局部刚性正则化形变场

**填补空白**：物理约束 × 事件 × 高斯形变场三者组合无先例。PEGS 有物理（加速度一致性）但限刚体 SE3、不形变高斯；LaGSplat 有 Lagrangian 动力学但无事件、单目；ERF-GS/FastEventDGS 有事件+形变但无物理约束，运动场欠约束易漂移。

**核心思路**：在事件驱动的形变场上加物理正则：(1) 加速度一致性——同一高斯轨迹的 `d²x/dt²` 跨多视图应一致（PEGS 范式，轻量）；(2) 局部刚性——动态物体内部相邻高斯相对位移受限（动态物体不是流体）；(3) 事件提供高时间分辨率的位置约束，物理提供时间二阶平滑约束，二者互补。

**与现有差异**：PEGS 是刚体 SE3 + 无遮挡 + 非形变；本 idea 处理可形变动态物体（高斯形变场），用事件而非 RGB 做位置约束。LaGSplat 学完整 Lagrangian 太重；本 idea 只用二阶运动一致性做正则，工程轻。

**可行性**：中高。加速度一致性只需形变场对时间求二阶导（形变场本就连续可微）。局部刚性用 KNN 邻域相对位移 L2 正则。难点是物理约束与事件损失的权重平衡。

---

### Idea ⑤ 可微事件相机物理模拟器：学每像素对比阈值与噪声，端到端可微

**填补空白**：现有所有 Event×GS 用 `linlog + 固定阈值 0.2 + STE` 近似事件生成（ERF-GS），或用外部仿真器（GS2E 用 ESIM）。EvFlow-GS 明确指出"阈值恒定假设"是缺陷之一——真实事件相机的对比阈值随像素、时间、场景变化。**没有工作把事件相机物理建模成可微模块与 3DGS 联合训练**。

**核心思路**：把事件生成物理（对数响应、对比阈值、噪声）做成可微模块：阈值跨越用 sigmoid 软近似（可微），每像素阈值/偏置/噪声参数可学。3DGS 渲染 → 可微事件 sim → 预测事件 → 与真实事件比。梯度流过真实事件生成物理，而非直通估计。

**与现有差异**：linlog+STE 是硬量化+直通，梯度近似差；本 idea 是连续可微物理模型，可学每像素参数适配真实相机。GS2E 用 ESIM 仿真但不可微、不联合训练。学术上把"事件相机建模"和"GS 渲染"统一到一个可微管线。

**可行性**：中低。可微阈值跨越的 sigmoid 软近似有现成做法；工程量大但思路清晰。适合作为"让事件损失更准"的通用增强，可与 idea ①②③④任一叠加。

---

## 五、Idea 组合建议

```
Idea ② 静态 bootstrap（基础层，兑现静态先验价值）
     +
Idea ① 事件遮挡感知（主 contribution，事件独有能力，空白最明确）
     +
Idea ④ 物理约束（辅助，给形变场加时间二阶平滑）
     ──→ 面向 CVPR 的"事件驱动遮挡感知物理约束快速动态重建"故事线

Idea ③ 点过程损失（学术新意最高，但工程门槛高）—— 可作为独立主线或与 ① 叠加
Idea ⑤ 可微事件模拟器（通用增强，可叠加任意上述）
```

**最推荐组合（CVPR 适配度最高）**：`② + ① + ④`——静态 bootstrap 解决 SPIN-4DGS 的失败模式、事件遮挡感知是 modality-specific 空白、物理约束给形变场正则。三者各有清晰差异点，且工程上可落地。

**最激进组合（学术新意最高）**：`② + ③ + ⑤`——静态先验 + 逐事件点过程 + 可微事件物理。完全绕开 linlog/STE/聚合三套近似，建立"事件原生物理监督"新范式。风险是工程量大。
