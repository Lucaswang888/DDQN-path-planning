
# 动态洋流场景下的深度强化学习（Deep Reinforcement Learning, DRL） 的自主导航系统 🌊:


# OceanMaze: 基于 DDQN 的复杂流场动态路径规划

**OceanMaze** 是一个基于深度强化学习（Deep Reinforcement Learning, DRL）的自主水面无人艇（ASV）路径规划仿真系统。本项目重点模拟了在**动态洋流（Lamb Vortex Field）**和**密集漂移障碍物**环境下的能量感知导航问题。

该项目是相关论文算法的复现与增强版本，采用了 **Double DQN (DDQN)** 结合 **优先经验回放 (PER)** 和 **N-Step Learning**，实现了智能体在复杂时变环境中的高效避障与借力导航。

---

## 📖 项目背景

在真实的海洋环境中，洋流和漂浮物对航行有着巨大影响。传统的静态路径规划无法有效应对环境的动态变化。本项目构建了一个包含以下特征的复杂环境：

* **流场干扰**：引入兰姆涡（Lamb Vortex）模拟非均匀洋流场。
* **动态障碍**：模拟大量随洋流实时漂移的浮冰或漂浮物（`drift: True`）。
* **复杂地形**：包含巨型环礁、峡谷和狭窄通道等非凸地形。

智能体通过强化学习，不仅学会了避障，还学会了**利用洋流（顺流而行）**来节省能量并快速到达终点。

## 🚀 核心特性

### 1. 高级 DRL 算法架构
本项目集成了多种 State-of-the-Art (SOTA) 技巧以提升训练效率：
* **Double DQN (DDQN)**: 分离选择网络与评估网络，消除 Q 值过高估计偏差。
* **Prioritized Experience Replay (PER)**: 基于 `SumTree` 实现，优先回放 TD-error 大的样本，大幅提升数据利用率。
* **N-Step Learning (N=3)**: 利用多步回报，加快奖励信号传播，适应长序列决策。
* **Curriculum Learning (课程学习)**: 训练难度系数 `difficulty` 从 0.2 随 Episode 逐渐增加至 1.0，引导智能体从简单环境过渡到复杂环境。

### 2. 物理与运动学模型 (Kinematics & Energy)
系统并未采用简单的网格移动，而是基于连续空间的运动学合成：
* **速度矢量合成**: 机器人的实际位移由推进速度与环境流速合成：
    $$\vec{P}_{new} = (\vec{V}_{prop} + \vec{V}_{current}) \cdot \Delta t$$
* **流场感知**: 智能体能够感知当前坐标下的流速矢量 $(u, v)$。
* **能量隐喻**: 模拟物理耗能 $E \propto c \cdot v^3 \cdot t$。由于推进功率恒定，智能体通过学习最小化**时间步数**，从而隐式地学会寻找顺流路径（利用流场加速）并避开逆流区域。

### 3. 混合状态感知 (Sensor)
智能体通过 12 维向量感知世界：
* **[0-7] 模拟雷达**: 8 个方向的射线探测，获取静态/动态障碍物距离。
* **[8-9] 目标导向**: 归一化目标距离 + 相对角度。
* **[10-11] 自身感知**: 当前位置的洋流速度分量 $(u, v)$。

## 🛠️ 环境依赖

本项目基于 Python 3 开发。请确保安装以下依赖库：

```bash
pip install numpy torch matplotlib
````

*注意：本项目依赖外部物理模块 `flow_dynamics.py`（需包含 `LambVortexField` 类和 `kinematic_update` 函数）。*

## 💻 使用说明

所有配置项均位于 `main.py` 顶部的配置区域。

### 1\. 训练模式 (Training)

将配置修改为训练模式：

```python
# main.py
IS_TRAINING = True  # 开启训练
```

运行脚本：

```bash
python main.py
```

  * **过程**: 控制台会实时打印 Episode 奖励、步数及当前难度系数。
  * **保存**: 模型会自动保存至 `checkpoints/ddqn_final_state.pth`。
  * **可视化**: 每当打破最佳分数记录时，会自动保存静态轨迹图 `train_best_ep{EP}.png`。

### 2\. 测试与可视化模式 (Testing)

加载训练好的模型并生成 GIF 动图：

```python
# main.py
IS_TRAINING = False # 关闭训练，进入测试模式
```

运行脚本：

```bash
python main.py
```

  * **过程**: 智能体将尝试 10 次高难度导航任务。
  * **输出**: 仅当导航**成功 (Success)** 时，程序会自动录制并生成 GIF 动图保存至 `test_results_gif/` 文件夹。
  * **效果**: 您将看到机器人在密集的绿色漂浮雷区中穿梭，并利用蓝色箭头指示的洋流进行导航。

## 📂 文件结构

```text
.
├── main.py              # 主程序：包含 DRL 算法、环境类、训练/测试循环
├── flow_dynamics.py     # 物理流场模块 (需确保存在)
├── checkpoints/         # 自动创建：存放训练好的模型权重 (.pth)
└── test_results_gif/    # 自动创建：存放测试生成的 GIF 结果
```

## 📊 关键参数配置

您可以在 `main.py` 中调整以下超参数以改变训练行为：

| 参数名 | 默认值 | 描述 |
| :--- | :--- | :--- |
| `PROPULSION_SPEED` | 5.0 | 机器人恒定推进速度 |
| `TIME_STEP` | 0.5 | 物理仿真步长 (dt) |
| `BATCH_SIZE` | 128 | 经验回放 Batch 大小 |
| `LR` | 1e-4 | 学习率 |
| `MEMORY_CAPACITY` | 50000 | PER 经验池容量 |
| `EPISODES` | 3000 | 总训练轮数 |

