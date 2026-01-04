# -*- coding: utf-8 -*-
# main_v7.py
# 碰撞不死版 (Immortal Mode): 撞墙提示 Collision 但不结束，阻挡前进

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

# === 引入外部物理模块 ===
from flow_dynamics import LambVortexField, kinematic_update

# ================= 配置区域 =================
IS_TRAINING = False
MODEL_PATH = os.path.join("checkpoints", "ddqn_v7_immortal.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 物理参数 ---
PROPULSION_SPEED = 6.0
OBSTACLE_SPEED = 0.5
TIME_STEP = 0.5

# 训练参数
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
MAX_STEPS = 800  # 既然撞不死，步数限制就很重要了
EPISODES = 5000
N_STEP = 3

# PER 参数
MEMORY_CAPACITY = 50000
PER_ALPHA = 0.6
PER_BETA = 0.4
PER_BETA_INC = 0.0005
ABS_ERROR_UPPER = 1.0

# === 奖励权重 (调整适应新逻辑) ===
PROGRESS_WEIGHT = 2.5
DIR_WEIGHT = 0.5
TIME_PENALTY = 0.05
STAGNATION_PENALTY = 0.5
GOAL_REWARD = 1000.0
# [重要] 碰撞惩罚不要太大，因为现在一回合可能撞很多次
# 如果太大，Accumulated Reward 可能会变成负无穷，导致梯度爆炸
COLLISION_REWARD = -10.0

# ================= 动作空间 (9动作) =================
ACTION_MAP = {
    0: (1.0, 0),  # 全速直行
    1: (1.0, 10),  # 全速左微调
    2: (1.0, -10),  # 全速右微调
    3: (0.6, 0),  # 慢速巡航
    4: (0.6, 20),  # 慢速左转
    5: (0.6, -20),  # 慢速右转
    6: (0.6, 30),  # 左侧切
    7: (0.6, -30),  # 右侧切
    8: (0.0, 0),  # 漂移
}
A_DIM = len(ACTION_MAP)


# ================= 日志工具 =================
class Prog:
    COLORS = dict(INIT="\033[95m", LEARN="\033[94m", TEST="\033[96m", BEST="\033[92m", WARN="\033[93m", END="\033[0m")

    @staticmethod
    def log(kind, msg):
        c = Prog.COLORS.get(kind, "")
        e = Prog.COLORS["END"]
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} {c}[{kind}]{e} {msg}")
        sys.stdout.flush()


# ================= SumTree & PER (保持不变) =================
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
            left = 2 * parent + 1;
            right = left + 1
            if left >= len(self.tree): leaf = parent; break
            if v <= self.tree[left]:
                parent = left
            else:
                v -= self.tree[left]; parent = right
        d_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[d_idx]

    @property
    def total_p(self):
        return self.tree[0]


class PrioritizedMemory:
    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.epsilon = 0.01;
        self.alpha = PER_ALPHA
        self.beta = PER_BETA;
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
                data = self.tree.data[idx - self.capacity + 1];
                p = self.tree.tree[idx]
            batch.append(data);
            idxs.append(idx);
            priorities.append(p)
        probs = np.array(priorities) / self.tree.total_p
        is_weight = np.power(self.tree.count * probs, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)


# ================= 环境类 =================
class OceanMaze:
    def __init__(self):
        self.width, self.height = 1000, 1000
        self.vf = LambVortexField(self.width, self.height)

        self.static_obstacles = []
        self.dynamic_obstacles = []

        center = np.array([500., 500.])
        self.static_obstacles.append({'c': center, 'r': 90})
        self.static_obstacles.append({'c': center + [80, 0], 'r': 60})
        self.static_obstacles.append({'c': center - [80, 0], 'r': 60})
        self.static_obstacles.append({'c': center + [0, 80], 'r': 60})
        self.static_obstacles.append({'c': center + [280, 280], 'r': 60})
        self.static_obstacles.append({'c': center + [-280, -280], 'r': 60})

        for i in range(8):
            self.dynamic_obstacles.append({
                'c': np.array([random.uniform(100, 900), random.uniform(100, 900)]),
                'r': random.uniform(25, 40),
            })

        self.start_pos = [100, 100]
        self.goal_pos = [900, 900]
        self.goal_radius = 40
        self.reset()

    def update_obstacles(self, dt):
        t = self.time
        for dyn in self.dynamic_obstacles:
            flow_vel = self.vf.get_velocity(dyn['c'][0], dyn['c'][1], t)
            flow_norm = np.linalg.norm(flow_vel)
            if flow_norm > 0:
                move_dir = flow_vel / flow_norm
            else:
                move_dir = np.array([1.0, 0.0])
            step = (move_dir * OBSTACLE_SPEED) * dt
            dyn['c'] += step

            if dyn['c'][0] < dyn['r']:
                dyn['c'][0] = dyn['r'] + 2;
            elif dyn['c'][0] > self.width - dyn['r']:
                dyn['c'][0] = self.width - dyn['r'] - 2
            if dyn['c'][1] < dyn['r']:
                dyn['c'][1] = dyn['r'] + 2
            elif dyn['c'][1] > self.height - dyn['r']:
                dyn['c'][1] = self.height - dyn['r'] - 2

            for stat in self.static_obstacles:
                dist_vec = dyn['c'] - stat['c']
                dist = np.linalg.norm(dist_vec)
                min_dist = dyn['r'] + stat['r']
                if dist < min_dist:
                    normal = dist_vec / (dist + 1e-6)
                    overlap = min_dist - dist
                    dyn['c'] += normal * (overlap + 2.0)

    def reset(self, difficulty=1.0):
        while True:
            self.start_pos = [random.uniform(50, self.width - 50), random.uniform(50, self.height - 50)]
            if not self._check_valid(self.start_pos): continue

            dist = random.uniform(400, 800)
            angle = random.uniform(0, 2 * math.pi)
            self.goal_pos = [
                self.start_pos[0] + dist * math.cos(angle),
                self.start_pos[1] + dist * math.sin(angle)
            ]
            if self._check_valid(self.goal_pos): break

        self.robot = {
            'pos': np.array(self.start_pos, dtype=np.float64),
            'ori': random.uniform(-3.14, 3.14),
            'path_history': collections.deque(maxlen=15)
        }
        self.time = 0.0

        for dyn in self.dynamic_obstacles:
            while True:
                dyn['c'] = np.array([random.uniform(100, 900), random.uniform(100, 900)])
                if self._check_valid(dyn['c'], radius=dyn['r'] + 20): break

    def _check_valid(self, pos, radius=10):
        if not (0 < pos[0] < self.width and 0 < pos[1] < self.height): return False
        for o in self.static_obstacles:
            if np.linalg.norm(np.array(pos) - o['c']) <= o['r'] + radius: return False
        return True

    def is_collision(self, pos):
        if not (0 <= pos[0] <= self.width and 0 <= pos[1] <= self.height): return True
        for o in self.static_obstacles:
            if np.linalg.norm(pos - o['c']) <= o['r'] + 5: return True
        for o in self.dynamic_obstacles:
            if np.linalg.norm(pos - o['c']) <= o['r'] + 5: return True
        return False

    def is_goal(self, pos):
        return np.linalg.norm(pos - np.array(self.goal_pos)) <= self.goal_radius

    def draw(self, ax):
        x = np.linspace(0, self.width, 25)
        y = np.linspace(0, self.height, 25)
        X, Y = np.meshgrid(x, y)
        U, V = np.zeros_like(X), np.zeros_like(Y)
        for i in range(25):
            for j in range(25):
                vec = self.vf.get_velocity(X[i, j], Y[i, j], self.time)
                U[i, j], V[i, j] = vec[0], vec[1]
        ax.quiver(X, Y, U, V, color='#1f77b4', alpha=0.3, width=0.003, scale=50)

        for o in self.static_obstacles:
            ax.add_patch(plt.Circle(o['c'], o['r'], color='#555555', alpha=0.9))
        for o in self.dynamic_obstacles:
            ax.add_patch(plt.Circle(o['c'], o['r'], color='#2ca02c', alpha=0.8))

        ax.add_patch(plt.Circle(self.start_pos, 15, color='blue', label='Start'))
        ax.add_patch(plt.Circle(self.goal_pos, self.goal_radius, color='red', alpha=0.5, label='Goal'))
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)


# ================= 传感器 =================
class Sensor:
    def __init__(self, maze):
        self.maze = maze

    def get_state(self):
        pos = self.maze.robot['pos']
        ori = self.maze.robot['ori']
        feats = []
        all_obs = self.maze.static_obstacles + self.maze.dynamic_obstacles
        for i in range(8):
            ang = ori + i * (math.pi / 4);
            d = 400.0
            v_ray = np.array([math.cos(ang), math.sin(ang)])
            for o in all_obs:
                oc = pos - o['c'];
                b = 2 * np.dot(v_ray, oc);
                c = np.dot(oc, oc) - o['r'] ** 2
                delta = b ** 2 - 4 * c
                if delta >= 0:
                    dist = (-b - math.sqrt(delta)) / 2
                    if 0 < dist < d: d = dist
            if v_ray[0] != 0:
                dx = (self.maze.width - pos[0]) / v_ray[0] if v_ray[0] > 0 else -pos[0] / v_ray[0]
                if 0 < dx < d: d = dx
            if v_ray[1] != 0:
                dy = (self.maze.height - pos[1]) / v_ray[1] if v_ray[1] > 0 else -pos[1] / v_ray[1]
                if 0 < dy < d: d = dy
            feats.append(d / 400.0)

        delta_goal = np.array(self.maze.goal_pos) - pos
        dist_goal = np.linalg.norm(delta_goal)
        ang_goal = math.atan2(delta_goal[1], delta_goal[0]) - ori
        ang_goal = (ang_goal + math.pi) % (2 * math.pi) - math.pi
        flow_vel = self.maze.vf.get_velocity(pos[0], pos[1], self.maze.time)
        state = np.concatenate([feats, [dist_goal / 1414.0, ang_goal / math.pi], flow_vel / 5.0])
        return state.astype(np.float32)


# ================= 执行器 (修改版：碰撞不结束) =================
class Executor:
    def __init__(self, maze):
        self.maze = maze
        self.speed = PROPULSION_SPEED
        self.dt = TIME_STEP

    def step(self, action_idx):
        self.maze.update_obstacles(self.dt)

        robot = self.maze.robot
        pos = robot['pos']
        robot['path_history'].append(pos.copy())

        speed_ratio, d_deg = ACTION_MAP[action_idx]
        d_theta = math.radians(d_deg)
        robot['ori'] = (robot['ori'] + d_theta + math.pi) % (2 * math.pi) - math.pi

        va_mag = self.speed * speed_ratio
        va_vec = np.array([va_mag * math.cos(robot['ori']), va_mag * math.sin(robot['ori'])])
        vo_vec = self.maze.vf.get_velocity(pos[0], pos[1], self.maze.time)

        new_pos, vs_vec = kinematic_update(pos, va_vec, vo_vec, self.dt)

        # [核心逻辑修改]
        is_collided = False
        if self.maze.is_collision(new_pos):
            # 发生碰撞：
            # 1. 奖励惩罚
            # 2. 位置【不更新】 (被挡住)
            # 3. 游戏【不结束】 (done = False)
            reward = COLLISION_REWARD
            is_collided = True
            # robot['pos'] 保持不变
        else:
            # 未碰撞，正常移动
            robot['pos'] = new_pos
            reward = -TIME_PENALTY  # 正常的时间消耗
            is_collided = False

        self.maze.time += self.dt

        # 目标引导奖励
        dist_old = np.linalg.norm(pos - np.array(self.maze.goal_pos))
        dist_new = np.linalg.norm(robot['pos'] - np.array(self.maze.goal_pos))
        reward += (dist_old - dist_new) * PROGRESS_WEIGHT

        # 滞留检测
        if len(robot['path_history']) >= 10:
            hist_arr = np.array(robot['path_history'])
            mean_pos = np.mean(hist_arr, axis=0)
            if np.linalg.norm(robot['pos'] - mean_pos) < 15.0:
                reward -= STAGNATION_PENALTY

        done = False
        if self.maze.is_goal(robot['pos']):
            reward += GOAL_REWARD
            done = True

        return reward, done, is_collided


# ================= Agent =================
class Net(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(s_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
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
        if random.random() < self.eps: return random.randint(0, A_DIM - 1)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        with torch.no_grad(): return self.eval_net(s_t).argmax().item()

    def store_transition(self, s, a, r, ns, done):
        self.memory.n_step_buffer.append((s, a, r, ns, done))
        if len(self.memory.n_step_buffer) < N_STEP and not done: return
        R, gamma = 0, 1
        for (_, _, r_i, _, _) in self.memory.n_step_buffer: R += r_i * gamma; gamma *= GAMMA
        s0, a0 = self.memory.n_step_buffer[0][:2]
        nsn, donen = self.memory.n_step_buffer[-1][3:]
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:]) or ABS_ERROR_UPPER
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
            q_next = self.target_net(ns_t).max(1, keepdim=True)[0]
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


# ================= 绘图辅助 =================
def save_gif(maze, traj, obs_hist, filename, title_text):
    fig, ax = plt.subplots(figsize=(8, 8))
    frames = range(0, len(traj), 2)

    def update(i):
        ax.clear()
        maze.robot['pos'] = np.array(traj[i])
        current_obs = obs_hist[i]
        for idx, dyn_o in enumerate(maze.dynamic_obstacles):
            dyn_o['c'] = current_obs[idx]
        maze.draw(ax)
        t_np = np.array(traj[:i + 1])
        if len(t_np) > 1: ax.plot(t_np[:, 0], t_np[:, 1], 'r-', lw=2)
        ax.set_title(f"{title_text} | Step {i}")

    ani = FuncAnimation(fig, update, frames=frames, interval=80)
    ani.save(filename, writer='pillow', fps=20)
    plt.close(fig)


# ================= 主程序 =================
if __name__ == "__main__":
    maze = OceanMaze()
    sensor = Sensor(maze)
    executor = Executor(maze)
    agent = Agent(s_dim=12, a_dim=A_DIM)

    if IS_TRAINING:
        Prog.log("INIT", f"START TRAINING (Immortal Collision Mode)")
        best_score = float('inf')

        for ep in range(EPISODES):
            maze.reset()
            traj = [maze.robot['pos'].copy()]
            ep_reward = 0
            coll_cnt = 0  # 统计单回合碰撞次数

            for t in range(MAX_STEPS):
                s = sensor.get_state()
                a = agent.act(s)

                # 获取 is_collided
                r, done, is_collided = executor.step(a)
                if is_collided: coll_cnt += 1

                ns = sensor.get_state()

                agent.store_transition(s, a, r, ns, done)
                agent.learn()

                ep_reward += r
                traj.append(maze.robot['pos'].copy())

                if done:
                    score = len(traj)
                    Prog.log("LEARN", f"Ep {ep} | Steps:{len(traj)} | Collisions:{coll_cnt} | R:{ep_reward:.1f}")
                    if ep > 300 and score < best_score:
                        best_score = score
                        agent.save(MODEL_PATH)
                    break
            else:
                # 超时也要打印
                Prog.log("WARN", f"Ep {ep} TIMEOUT | Collisions:{coll_cnt} | R:{ep_reward:.1f}")

            if ep % 50 == 0:
                Prog.log("INFO", f"Epsilon: {agent.eps:.3f}")

    else:
        Prog.log("TEST", "START TESTING")
        agent.load(MODEL_PATH)
        os.makedirs("test_results_v7", exist_ok=True)

        for i in range(5):
            maze.reset()
            traj = [maze.robot['pos'].copy()]
            obs_log = []
            obs_log.append([o['c'].copy() for o in maze.dynamic_obstacles])

            coll_cnt = 0
            final_status = "TIMEOUT"

            for step_i in range(MAX_STEPS):
                s = sensor.get_state()
                a = agent.act(s)

                _, done, is_collided = executor.step(a)
                if is_collided:
                    coll_cnt += 1
                    # 测试时可以打印碰撞提示，但不需要每次都打印，否则刷屏
                    # print("Collision detected!")

                traj.append(maze.robot['pos'].copy())
                obs_log.append([o['c'].copy() for o in maze.dynamic_obstacles])

                if done:
                    if maze.is_goal(maze.robot['pos']):
                        final_status = "SUCCESS"
                        print(f"Test {i}: \033[92mSUCCESS\033[0m (Steps: {step_i}, Collisions: {coll_cnt})")
                        save_gif(maze, traj, obs_log, f"test_results_v7/test_{i}_success.gif",
                                 f"Success | Col:{coll_cnt}")
                    break

            if final_status == "TIMEOUT":
                print(f"Test {i}: \033[93mTIMEOUT\033[0m (Collisions: {coll_cnt})")
                save_gif(maze, traj, obs_log, f"test_results_v7/test_{i}_timeout.gif", f"Timeout | Col:{coll_cnt}")