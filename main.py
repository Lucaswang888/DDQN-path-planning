# -*- coding: utf-8 -*-
# main.py
# 论文复现版 (巨型障碍物 + 密集漂移雷区 + GIF可视化)

import os
import sys
import math
import time
import random
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import collections
from matplotlib.animation import FuncAnimation

# === 引入物理模块 ===
from flow_dynamics import LambVortexField, kinematic_update

# ================= 配置区域 =================
IS_TRAINING = False  # True: 训练模式; False: 测试模式(生成GIF)

MODEL_PATH = os.path.join("checkpoints", "ddqn_final_state.pth")

# 物理参数
PROPULSION_SPEED = 5.0
DRAG_C = 0.5
TIME_STEP = 0.5

# 训练参数
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
MAX_STEPS = 2000
EPISODES = 3000
N_STEP = 3

# PER 参数
MEMORY_CAPACITY = 50000
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_BETA_INC = 0.0005
ABS_ERROR_UPPER = 1.0

# === 奖励权重 ===
PROGRESS_WEIGHT = 2.0
DIR_WEIGHT = 1.5
TIME_PENALTY = 0.02
GOAL_REWARD = 1000.0
COLLISION_REWARD = -50.0

LAMBDA_SCORE = 0.0005

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= 日志工具 =================
class Prog:
    COLORS = dict(INIT="\033[95m", LEARN="\033[94m", TEST="\033[96m", BEST="\033[92m", END="\033[0m")

    @staticmethod
    def log(kind, msg):
        c = Prog.COLORS.get(kind, "")
        e = Prog.COLORS["END"]
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} {c}[{kind}]{e} {msg}")
        sys.stdout.flush()


# ================= SumTree & PER =================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write = 0
        self.count = 0

    def add(self, p, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity: self.write = 0
        if self.count < self.capacity: self.count += 1

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, v):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left];
                parent = right
        d_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[d_idx]

    @property
    def total_p(self):
        return self.tree[0]


class PrioritizedMemory:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.epsilon = 0.01
        self.alpha = PER_ALPHA
        self.beta = PER_BETA
        self.beta_inc = PER_BETA_INC
        self.n_step_buffer = collections.deque(maxlen=N_STEP)

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch, idxs, priorities = [], [], []
        segment = self.tree.total_p / n
        self.beta = np.min([1., self.beta + self.beta_inc])
        for i in range(n):
            a, b = segment * i, segment * (i + 1)
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            if data is 0:
                idx = random.randint(self.tree.capacity - 1, self.tree.capacity + self.tree.count - 2)
                data = self.tree.data[idx - self.tree.capacity + 1]
                p = self.tree.tree[idx]
            batch.append(data)
            idxs.append(idx)
            priorities.append(p)
        probs = np.array(priorities) / self.tree.total_p
        is_weight = np.power(self.tree.count * probs, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


# ================= 环境类 (大幅增强版) =================
class OceanMaze:
    def __init__(self):
        self.width, self.height = 1000, 1000
        self.vf = LambVortexField(self.width, self.height)

        self.obstacles = []

        # ==========================================
        # 1. 巨型固定复合岛屿 (Fixed Mega Islands)
        # ==========================================

        # [中心巨型环礁] - 由5个大圆组成的巨型障碍
        # 半径从原来的 40-50 增加到 70-80
        center = np.array([500., 500.])
        self.obstacles.append({'c': center, 'r': 80, 'drift': False})
        self.obstacles.append({'c': center + [60, 0], 'r': 60, 'drift': False})
        self.obstacles.append({'c': center - [60, 0], 'r': 60, 'drift': False})
        self.obstacles.append({'c': center + [0, 60], 'r': 60, 'drift': False})
        self.obstacles.append({'c': center - [0, 60], 'r': 60, 'drift': False})

        # [左下角-连绵山脉] - 增加长度和厚度
        start_x, start_y = 100, 200
        for i in range(10):  # 数量增加到10个
            self.obstacles.append({
                'c': np.array([start_x + i * 35, start_y + i * 20], dtype=np.float64),
                'r': 40,  # 半径从 25 增加到 40
                'drift': False
            })

        # [顶部-宽阔峡谷壁] - 范围扩大
        for i in range(12):  # 数量增加
            angle = math.radians(i * 15 + 180)
            cx = 500 + 200 * math.cos(angle)  # 弧度半径增大
            cy = 920 + 80 * math.sin(angle)
            self.obstacles.append({'c': np.array([cx, cy], dtype=np.float64), 'r': 35, 'drift': False})

        # [角落堡垒]
        self.obstacles.append({'c': np.array([100., 800.]), 'r': 60, 'drift': False})
        self.obstacles.append({'c': np.array([800., 150.]), 'r': 60, 'drift': False})

        # ==========================================
        # 2. 密集漂移雷区 (Moving Minefield) - 绿色
        # ==========================================

        # [横穿地图的浮冰带] - 数量和尺寸翻倍
        for i in range(10):
            self.obstacles.append({
                'c': np.array([200. + i * 60, 400. + random.uniform(-80, 80)], dtype=np.float64),
                'r': 30,  # 半径 30 (以前是15-20)
                'drift': True
            })

        # [右侧纵向流]
        for i in range(8):
            self.obstacles.append({
                'c': np.array([800. + random.uniform(-40, 40), 300. + i * 70], dtype=np.float64),
                'r': 30,
                'drift': True
            })

        # [随机散布的巨型漂流物]
        for _ in range(15):  # 随机增加 15 个大障碍
            rx = random.uniform(100, 900)
            ry = random.uniform(100, 900)
            rr = random.uniform(25, 45)  # 尺寸随机大
            self.obstacles.append({'c': np.array([rx, ry]), 'r': rr, 'drift': True})

        self.start_pos = [100, 100]
        self.goal_pos = [900, 900]
        self.goal_radius = 30
        self.reset()

    def update_obstacles(self, dt):
        for o in self.obstacles:
            if o['drift']:
                flow_vel = self.vf.get_velocity(o['c'][0], o['c'][1], self.time)
                o['c'] += flow_vel * dt
                # 边界检查
                if o['c'][0] < o['r']:
                    o['c'][0] = o['r']
                elif o['c'][0] > self.width - o['r']:
                    o['c'][0] = self.width - o['r']
                if o['c'][1] < o['r']:
                    o['c'][1] = o['r']
                elif o['c'][1] > self.height - o['r']:
                    o['c'][1] = self.height - o['r']

    def _check_valid_pos(self, pos):
        if not (0 < pos[0] < self.width and 0 < pos[1] < self.height):
            return False
        for o in self.obstacles:
            # 障碍物变大了，这里安全距离稍微调小一点(15)，否则太难找到空位
            if np.linalg.norm(np.array(pos) - o['c']) <= o['r'] + 15:
                return False
        return True

    def reset(self, difficulty=1.0):
        # 增加尝试次数，防止因地图太拥挤而卡死
        retry_count = 0
        while True:
            retry_count += 1
            self.start_pos = [random.uniform(50, self.width - 50), random.uniform(50, self.height - 50)]
            if not self._check_valid_pos(self.start_pos): continue

            min_dist = 200
            max_dist = 300 + (1000 * difficulty)
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(min_dist, max_dist)

            self.goal_pos = [
                self.start_pos[0] + dist * math.cos(angle),
                self.start_pos[1] + dist * math.sin(angle)
            ]
            if self._check_valid_pos(self.goal_pos):
                break

            # 如果尝试超过1000次没找到位置，强制缩小随机范围（极端情况保护）
            if retry_count > 1000:
                print("Warning: Map too crowded, retrying...")
                retry_count = 0

        self.robot = {'pos': list(self.start_pos), 'ori': random.uniform(-3.14, 3.14)}
        self.time = 0.0

    def is_collision(self, x, y):
        if not (0 <= x <= self.width and 0 <= y <= self.height): return True
        p = np.array([x, y])
        for o in self.obstacles:
            if np.linalg.norm(p - o['c']) <= o['r'] + 5: return True
        return False

    def is_goal(self, x, y):
        return math.hypot(x - self.goal_pos[0], y - self.goal_pos[1]) <= self.goal_radius

    def draw(self, ax):
        x = np.linspace(0, self.width, 30)
        y = np.linspace(0, self.height, 30)
        X, Y = np.meshgrid(x, y)
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(30):
            for j in range(30):
                vec = self.vf.get_velocity(X[i, j], Y[i, j], self.time)
                U[i, j], V[i, j] = vec[0], vec[1]
        ax.quiver(X, Y, U, V, color='blue', alpha=0.15)

        for o in self.obstacles:
            if o['drift']:
                ax.add_patch(plt.Circle(o['c'], o['r'], color='green', alpha=0.8))
            else:
                ax.add_patch(plt.Circle(o['c'], o['r'], color='#444444', alpha=0.9))

        ax.add_patch(plt.Circle(self.start_pos, 15, color='blue', label='Start'))
        ax.add_patch(plt.Circle(self.goal_pos, self.goal_radius, color='red', alpha=0.5, label='Goal'))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)


# ================= Sensor & Executor =================
class Sensor:
    def __init__(self, maze):
        self.maze = maze

    def get_state(self):
        pos = self.maze.robot['pos']
        ori = self.maze.robot['ori']
        t = self.maze.time
        feats = []
        for i in range(8):
            ang = ori + i * (math.pi / 4)
            d = 300.0
            p = np.array(pos)
            v = np.array([math.cos(ang), math.sin(ang)])
            for o in self.maze.obstacles:
                oc = p - o['c']
                b = 2 * np.dot(v, oc)
                c = np.dot(oc, oc) - o['r'] ** 2
                delta = b ** 2 - 4 * c
                if delta >= 0:
                    dist = (-b - math.sqrt(delta)) / 2
                    if 0 < dist < d: d = dist
            if v[0] != 0:
                dx = (self.maze.width - p[0]) / v[0] if v[0] > 0 else -p[0] / v[0]
                if 0 < dx < d: d = dx
            if v[1] != 0:
                dy = (self.maze.height - p[1]) / v[1] if v[1] > 0 else -p[1] / v[1]
                if 0 < dy < d: d = dy
            feats.append(d / 300.0)

        dx, dy = self.maze.goal_pos[0] - pos[0], self.maze.goal_pos[1] - pos[1]
        dist = math.hypot(dx, dy)
        ang_goal = math.atan2(dy, dx) - ori
        ang_goal = (ang_goal + math.pi) % (2 * math.pi) - math.pi
        flow = self.maze.vf.get_velocity(pos[0], pos[1], t)
        return np.array(feats + [dist / 1414.0, ang_goal / math.pi, flow[0], flow[1]], dtype=np.float32)


class Executor:
    def __init__(self, maze):
        self.maze = maze
        self.propulsion_speed = PROPULSION_SPEED
        self.dt = TIME_STEP
        self.drag_c = DRAG_C

    def step(self, action):
        self.maze.update_obstacles(self.dt)

        d_theta = 0.0
        if action == 1: d_theta = 0.2
        if action == 2: d_theta = -0.2
        self.maze.robot['ori'] += d_theta

        current_ori = self.maze.robot['ori']
        pos = np.array(self.maze.robot['pos'])
        t = self.maze.time

        va_vec = np.array([
            self.propulsion_speed * math.cos(current_ori),
            self.propulsion_speed * math.sin(current_ori)
        ])
        vo_vec = self.maze.vf.get_velocity(pos[0], pos[1], t)
        new_pos, vs_vec = kinematic_update(pos, va_vec, vo_vec, self.dt)

        self.maze.robot['pos'] = list(new_pos)
        self.maze.time += self.dt

        step_energy = self.drag_c * (self.propulsion_speed ** 3) * self.dt

        goal = np.array(self.maze.goal_pos)
        d_old = np.linalg.norm(pos - goal)
        d_new = np.linalg.norm(new_pos - goal)

        reward = -TIME_PENALTY
        reward += (d_old - d_new) * PROGRESS_WEIGHT

        vec_to_goal = goal - new_pos
        norm_goal = np.linalg.norm(vec_to_goal)
        vs_norm = np.linalg.norm(vs_vec)

        if norm_goal > 1e-3 and vs_norm > 1e-3:
            cosine = np.dot(vs_vec, vec_to_goal) / (vs_norm * norm_goal)
            reward += DIR_WEIGHT * cosine

        done = False
        if self.maze.is_collision(new_pos[0], new_pos[1]):
            reward += COLLISION_REWARD
            self.maze.robot['pos'] = list(pos)

        if self.maze.is_goal(new_pos[0], new_pos[1]):
            reward += GOAL_REWARD
            done = True

        return reward, done, step_energy


# ================= 智能体定义 =================
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, a_dim)
        )

    def forward(self, x): return self.fc(x)


class Agent:
    def __init__(self, s_dim, a_dim):
        self.eval_net = Net(s_dim, a_dim).to(DEVICE)
        self.target_net = Net(s_dim, a_dim).to(DEVICE)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.opt = optim.Adam(self.eval_net.parameters(), lr=LR)
        self.memory = PrioritizedMemory(MEMORY_CAPACITY)
        self.steps = 0
        self.eps = 1.0 if IS_TRAINING else 0.05

    def act(self, s):
        if random.random() < self.eps: return random.randint(0, 2)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): return self.eval_net(s_t).argmax().item()

    def store_transition(self, s, a, r, ns, done):
        transition = (s, a, r, ns, done)
        self.memory.n_step_buffer.append(transition)
        if len(self.memory.n_step_buffer) < N_STEP and not done: return
        R, gamma = 0, 1
        for (_, _, r_i, _, _) in self.memory.n_step_buffer:
            R += r_i * gamma;
            gamma *= GAMMA
        s0, a0 = self.memory.n_step_buffer[0][:2]
        nsn, donen = self.memory.n_step_buffer[-1][3:]
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        if max_p == 0: max_p = ABS_ERROR_UPPER
        self.memory.add(max_p, (s0, a0, R, nsn, donen))
        if done: self.memory.n_step_buffer.clear()

    def learn(self):
        if self.memory.tree.count < BATCH_SIZE: return
        batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
        s, a, r, ns, d = zip(*batch)
        s_t = torch.FloatTensor(np.array(s)).to(DEVICE)
        a_t = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
        r_t = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)
        ns_t = torch.FloatTensor(np.array(ns)).to(DEVICE)
        d_t = torch.FloatTensor(d).unsqueeze(1).to(DEVICE)
        w_t = torch.FloatTensor(is_weights).unsqueeze(1).to(DEVICE)
        q_eval = self.eval_net(s_t).gather(1, a_t)
        with torch.no_grad():
            a_next = self.eval_net(ns_t).argmax(dim=1, keepdim=True)
            q_next = self.target_net(ns_t).gather(1, a_next)
            q_target = r_t + (GAMMA ** N_STEP) * q_next * (1 - d_t)
        td_errors = (q_target - q_eval).detach().cpu().numpy().flatten()
        loss = (w_t * (q_target - q_eval).pow(2)).mean()
        self.opt.zero_grad();
        loss.backward();
        self.opt.step()
        for i in range(BATCH_SIZE): self.memory.update(idxs[i], td_errors[i])
        self.steps += 1
        if self.steps % 200 == 0: self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.eps > 0.05: self.eps *= 0.99995

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.eval_net.state_dict(), path)
        Prog.log("BEST", f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.eval_net.load_state_dict(torch.load(path, map_location=DEVICE))
            self.target_net.load_state_dict(self.eval_net.state_dict())
            Prog.log("TEST", f"Model loaded: {path}")
        else:
            Prog.log("WARN", "Model not found")


def calc_len(traj):
    if len(traj) < 2: return 0
    return sum(math.hypot(traj[i + 1][0] - traj[i][0], traj[i + 1][1] - traj[i][1]) for i in range(len(traj) - 1))


def save_plot(maze, traj, filename, title_text=None):
    """保存静态图片 (训练时用)"""
    fig, ax = plt.subplots(figsize=(10, 10))
    maze.draw(ax)
    t_np = np.array(traj)
    ax.plot(t_np[:, 0], t_np[:, 1], 'r-', lw=2, label="Trajectory")
    ax.legend()
    ax.set_title(title_text if title_text else "Trajectory")
    plt.savefig(filename)
    plt.close(fig)


def save_gif(maze, robot_hist, obs_hist, filename, title_text=None):
    """生成动态GIF (测试时用)"""
    fig, ax = plt.subplots(figsize=(8, 8))

    # 抽样帧数
    skip = 2
    frames = range(0, len(robot_hist), skip)

    def update(frame_idx):
        ax.clear()
        maze.robot['pos'] = list(robot_hist[frame_idx])

        current_obs_data = obs_hist[frame_idx]
        for i, o in enumerate(maze.obstacles):
            o['c'] = current_obs_data[i]

        maze.draw(ax)

        hist_np = np.array(robot_hist[:frame_idx + 1])
        if len(hist_np) > 1:
            ax.plot(hist_np[:, 0], hist_np[:, 1], 'r-', lw=2, label="Trajectory")

        ax.set_title(f"{title_text} | Step {frame_idx}")

    print(f"Generating GIF: {filename} ...")
    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    ani.save(filename, writer='pillow', fps=15)
    plt.close(fig)


if __name__ == "__main__":
    maze = OceanMaze()
    sensor = Sensor(maze)
    executor = Executor(maze)
    agent = Agent(s_dim=12, a_dim=3)

    if IS_TRAINING:
        Prog.log("INIT", "START TRAINING (Curriculum: Easy -> Hard)")
        best_score = float('inf')
        for ep in range(EPISODES):
            if ep < 800:
                difficulty = 0.2 + (0.3 * ep / 800)
            elif ep < 1500:
                difficulty = 0.5 + (0.3 * (ep - 800) / 700)
            else:
                difficulty = 1.0

            maze.reset(difficulty=difficulty)
            traj = [tuple(maze.robot['pos'])]
            ep_reward = 0
            ep_energy = 0

            for t in range(MAX_STEPS):
                s = sensor.get_state()
                a = agent.act(s)
                r, done, eng = executor.step(a)
                ns = sensor.get_state()
                agent.store_transition(s, a, r, ns, done)
                agent.learn()

                ep_reward += r
                ep_energy += eng
                traj.append(tuple(maze.robot['pos']))

                if done:
                    plen = calc_len(traj)
                    score = plen + LAMBDA_SCORE * ep_energy
                    Prog.log("LEARN", f"Ep {ep} (Diff {difficulty:.1f}) | Len:{plen:.0f} | Score:{score:.1f}")
                    if difficulty > 0.5 and score < best_score:
                        best_score = score
                        agent.save(MODEL_PATH)
                        title = f"Ep {ep} Best Score: {best_score:.1f} (Diff {difficulty:.1f})"
                        save_plot(maze, traj, f"train_best_ep{ep}.png", title)
                        Prog.log("BEST", "New Record!")
                    break

            if ep % 20 == 0:
                Prog.log("LEARN", f"Ep {ep} running... Eps: {agent.eps:.3f}")

        final_path = os.path.join("checkpoints", "ddqn_final_state.pth")
        agent.save(final_path)
        Prog.log("INIT", f"Training Finished! Final model saved to {final_path}")

    else:
        # === 测试模式：只保存成功的GIF ===
        Prog.log("TEST", "START TESTING PHASE (Saving Successes Only)")
        agent.load(MODEL_PATH)

        save_dir = "test_results_gif"
        os.makedirs(save_dir, exist_ok=True)
        print(f"GIFs will be saved to: {os.path.abspath(save_dir)}")

        success_count = 0

        # 增加循环次数到10次，防止成功率低时生成不了文件
        for i in range(10):
            maze.reset(difficulty=1.0)

            traj = [tuple(maze.robot['pos'])]
            obs_log = [[o['c'].copy() for o in maze.obstacles]]

            ep_eng = 0
            done = False

            for step_i in range(MAX_STEPS):
                s = sensor.get_state()
                a = agent.act(s)
                _, done, eng = executor.step(a)
                ep_eng += eng

                traj.append(tuple(maze.robot['pos']))
                obs_log.append([o['c'].copy() for o in maze.obstacles])

                if done:
                    success_count += 1
                    print(f"Test {i + 1:02d}: Success! Len {calc_len(traj):.0f}")
                    break
            else:
                print(f"Test {i + 1:02d}: Failed.")

            # === 修改处：只在 done (成功) 时生成 GIF ===
            if done:
                title = f"Test {i + 1} | Success"
                file_name = f"test_{i + 1:02d}_success.gif"
                full_path = os.path.join(save_dir, file_name)
                save_gif(maze, traj, obs_log, full_path, title)
            else:
                print(f"Test {i + 1:02d}: Fail (Skipping GIF save)")

        print("-" * 30)
        print(f"Final Success Rate: {success_count / 10 * 100:.1f}%")