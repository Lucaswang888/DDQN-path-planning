:blush:
动态洋流场景下的深度强化学习（Deep Reinforcement Learning, DRL） 的自主导航系统 🌊:

基础网络：DDQN
优化策略：
优先经验回放 (PER, Prioritized Experience Replay)
多步学习 N-Step Learning

研究问题：
我们假设 ASV 的推进速度是恒定的。
顺洋流，速度快，用时短，耗能少
逆洋流，速度慢，用时长，耗能多

环境感知

OceanMaze: 基于 DDQN 的复杂流场动态路径规划OceanMaze 是一个基于深度强化学习（Deep Reinforcement Learning, DRL）的自主水面无人艇（ASV）路径规划仿真系统。本项目重点模拟了在动态洋流（Lamb Vortex Field）和密集漂移障碍物环境下的能量感知导航问题。该项目是相关论文算法的复现与增强版本，采用了 Double DQN (DDQN) 结合 优先经验回放 (PER) 和 N-Step Learning，实现了智能体在复杂时变环境中的高效避障与借力导航。📖 项目简介在真实的海洋环境中，洋流和漂浮物对航行有着巨大影响。传统的静态路径规划无法有效应对环境的动态变化。本项目构建了一个包含以下特征的复杂环境：流场干扰：引入兰姆涡（Lamb Vortex）模拟非均匀洋流。动态障碍：大量随洋流实时漂移的障碍物（模拟浮冰或漂浮物）。复杂地形：包含巨型环礁、峡谷和狭窄通道。智能体通过强化学习，不仅学会了避障，还学会了**利用洋流（顺流而行）**来节省能量并快速到达终点。🚀 核心特性1. 高级 DRL 算法架构本项目并未使用基础 DQN，而是集成了多种 State-of-the-Art (SOTA) 技巧以提升收敛性和稳定性：Double DQN (DDQN): 消除 Q 值过高估计偏差，训练更稳定。Prioritized Experience Replay (PER): 基于 SumTree 实现，优先学习高误差样本（"难"的样本），大幅提升数据利用率。N-Step Learning (N=3): 能够更快地传播奖励信号，适应长序列决策任务。Curriculum Learning (课程学习): 训练难度从 0.2 随 Episode 逐渐增加至 1.0，引导智能体逐步掌握复杂环境。2. 物理与运动学模型系统并未采用简单的网格移动，而是基于连续空间的运动学合成：速度矢量合成: 机器人的实际位移 $\vec{P}_{new} = (\vec{V}_{propulsion} + \vec{V}_{current}) \cdot \Delta t$。流场感知: 智能体能够感知当前坐标下的流速矢量 $(u, v)$。能量模型: 模拟真实物理耗能 $E \propto c \cdot v^3 \cdot t$。由于推进速度恒定，智能体通过最小化时间步数来隐式地优化能量（即学会寻找顺流路径，避免逆流）。3. 混合状态感知 (Sensor)智能体通过 12 维向量感知世界：[0-7] 模拟雷达: 8 个方向的射线探测，获取障碍物距离。[8-9] 目标导向: 归一化距离 + 相对角度。[10-11] 自身感知: 当前位置的洋流速度分量 $(u, v)$。🛠️ 安装与依赖本项目基于 Python 3 开发，依赖以下库：Bashpip install numpy torch matplotlib
注意：本项目包含两个主要文件 main.py 和 flow_dynamics.py（需确保物理模块存在）。💻 使用说明项目的所有配置均位于 main.py 顶部的配置区域。1. 训练模式 (Training)将配置修改为训练模式：Python# main.py
IS_TRAINING = True  # 开启训练
运行脚本：Bashpython main.py
输出: 训练过程中会实时打印 Episode 奖励、步数及难度系数。保存: 模型会自动保存至 checkpoints/ddqn_final_state.pth。可视化: 训练中每当打破最佳记录时，会生成静态轨迹图 train_best_ep{EP}.png。2. 测试与可视化模式 (Testing)加载训练好的模型并生成 GIF 动图：Python# main.py
IS_TRAINING = False # 关闭训练，进入测试模式
运行脚本：Bashpython main.py
功能: 智能体将尝试 10 次导航任务。输出: 仅当导航成功 (Success) 时，程序会自动录制并生成 GIF 动图保存至 test_results_gif/ 文件夹。效果: 你将看到机器人在密集的绿色漂浮雷区中穿梭，并巧妙利用蓝色流场箭头指示的洋流。📂 文件结构Plaintext.
├── main.py              # 主程序：包含 DRL 算法、环境类、训练/测试循环
├── flow_dynamics.py     # (需自行确保) 物理流场模块，提供 LambVortexField
├── checkpoints/         # 存放训练好的模型权重 (.pth)
└── test_results_gif/    # 存放测试生成的 GIF 结果
📊 关键参数配置你可以在 main.py 中调整以下超参数：参数名默认值描述PROPULSION_SPEED5.0机器人恒定推进速度TIME_STEP0.5物理仿真步长 (dt)BATCH_SIZE128经验回放 Batch 大小LR1e-4学习率MEMORY_CAPACITY50000经验池容量IS_TRAININGTrue/False模式切换开关🤝 致谢本项目参考了 ASV 路径规划领域的相关文献，特别是关于能量感知（Energy-Aware）导航和流场利用的研究。代码实现了一个完整的“论文复现级”仿真环境。
