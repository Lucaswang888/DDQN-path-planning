# -*- coding: utf-8 -*-
# main.py
# 论文复现版 (支持 argparse + 函数封装)

import os
import sys
import math
import time
import random
import datetime
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import collections
from matplotlib.animation import FuncAnimation

# === 引入物理模块 ===
from flow_dynamics import LambVortexField, kinematic_update

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
                leaf = parent; break
            if v <= self.tree[left]: parent = left
            else: v -= self.tree[left]; parent = right
        d_idx = leaf - self.capacity + 1
        return leaf, self.tree[leaf], self.data[d_idx]

    @property
    def total_p(self): return self.tree[0]

class PrioritizedMemory:
    def __init__(self, args):
        self.args = args
        self.tree = SumTree(args.memory_capacity)
        self.epsilon = 0.01
        self.alpha = args.per_alpha
        self.beta = args.per_beta
        self.beta_inc = args.per_beta_inc
        self.n_step_buffer = collections.deque(maxlen=args.n_step)

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
            batch.append(data); idxs.append(idx); priorities.append(p)
        probs = np.array(priorities) / self.tree.total_p
        is_weight = np.power(self.tree.count * probs, -self.beta)
        is_weight /= is_weight.max()
        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

# ================= 环境类 =================
class OceanMaze:
    def __init__(self, args):
        self.args = args
        self.width, self.height = 1000, 1000
        self.vf = LambVortexField(self.width, self.height)
        self.obstacles = []
        self._init_obstacles()
        self.start_pos = [100, 100]
        self.goal_pos = [900, 900]
        self.goal_radius = args.goal_radius
        self.reset()

    def _init_obstacles(self):
        # [中心巨型环礁]
        center = np.array([500., 500.])
        self.obstacles.append({'c': center, 'r': 80, 'drift': False})
        self.obstacles.append({'c': center + [60, 0], 'r': 60, 'drift': False})
        self.obstacles.append({'c': center - [60, 0], 'r': 60, 'drift': False})
        self.obstacles.append({'c': center + [0, 60], 'r': 60, 'drift': False})
        self.obstacles.append({'c': center - [0, 60], 'r': 60, 'drift': False})

        # [左下角-连绵山脉]
        start_x, start_y = 100, 200
        for i in range(10):
            self.obstacles.append({'c': np.array([start_x + i * 35, start_y + i * 20], dtype=np.float64), 'r': 40, 'drift': False})

        # [顶部-宽阔峡谷壁]
        for i in range(12):
            angle = math.radians(i * 15 + 180)
            cx = 500 + 200 * math.cos(angle)
            cy = 920 + 80 * math.sin(angle)
            self.obstacles.append({'c': np.array([cx, cy], dtype=np.float64), 'r': 35, 'drift': False})

        self.obstacles.append({'c': np.array([100., 800.]), 'r': 60, 'drift': False})
        self.obstacles.append({'c': np.array([800., 150.]), 'r': 60, 'drift': False})

        # [漂移障碍]
        for i in range(10):
            self.obstacles.append({'c': np.array([200. + i * 60, 400. + random.uniform(-80, 80)], dtype=np.float64), 'r': 30, 'drift': True})
        for i in range(8):
            self.obstacles.append({'c': np.array([800. + random.uniform(-40, 40), 300. + i * 70], dtype=np.float64), 'r': 30, 'drift': True})
        for _ in range(15):
            rx, ry = random.uniform(100, 900), random.uniform(100, 900)
            self.obstacles.append({'c': np.array([rx, ry]), 'r': random.uniform(25, 45), 'drift': True})

    def update_obstacles(self, dt):
        for o in self.obstacles:
            if o['drift']:
                flow_vel = self.vf.get_velocity(o['c'][0], o['c'][1], self.time)
                o['c'] += flow_vel * dt
                o['c'][0] = np.clip(o['c'][0], o['r'], self.width - o['r'])
                o['c'][1] = np.clip(o['c'][1], o['r'], self.height - o['r'])

    def _check_valid_pos(self, pos):
        if not (0 < pos[0] < self.width and 0 < pos[1] < self.height): return False
        for o in self.obstacles:
            if np.linalg.norm(np.array(pos) - o['c']) <= o['r'] + 15: return False
        return True

    def reset(self, difficulty=1.0):
        retry_count = 0
        while True:
            retry_count += 1
            self.start_pos = [random.uniform(50, self.width - 50), random.uniform(50, self.height - 50)]
            if not self._check_valid_pos(self.start_pos): continue
            
            min_dist, max_dist = 200, 300 + (1000 * difficulty)
            angle = random.uniform(0, 2 * math.pi)
            dist = random.uniform(min_dist, max_dist)
            self.goal_pos = [self.start_pos[0] + dist * math.cos(angle), self.start_pos[1] + dist * math.sin(angle)]
            
            if self._check_valid_pos(self.goal_pos): break
            if retry_count > 1000: break

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
                vec = self.vf.get_velocity(X[i,j], Y[i,j], self.time)
                U[i,j], V[i,j] = vec[0], vec[1]
        ax.quiver(X, Y, U, V, color='blue', alpha=0.15)
        for o in self.obstacles:
            color = 'green' if o['drift'] else '#444444'
            ax.add_patch(plt.Circle(o['c'], o['r'], color=color, alpha=0.8))
        ax.add_patch(plt.Circle(self.start_pos, 15, color='blue', label='Start'))
        ax.add_patch(plt.Circle(self.goal_pos, self.goal_radius, color='red', alpha=0.5, label='Goal'))
        ax.set_xlim(0, self.width); ax.set_ylim(0, self.height)

# ================= Sensor & Executor =================
class Sensor:
    def __init__(self, maze, args):
        self.maze = maze
        self.args = args

    def get_state(self):
        pos, ori, t = self.maze.robot['pos'], self.maze.robot['ori'], self.maze.time
        feats = []
        for i in range(8):
            ang = ori + i * (math.pi / 4)
            d = 300.0
            p, v = np.array(pos), np.array([math.cos(ang), math.sin(ang)])
            for o in self.maze.obstacles:
                oc = p - o['c']
                b, c = 2 * np.dot(v, oc), np.dot(oc, oc) - o['r'] ** 2
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
    def __init__(self, maze, args):
        self.maze = maze
        self.args = args
        self.propulsion_speed = args.propulsion_speed
        self.dt = args.time_step
        self.drag_c = args.drag_c

    def step(self, action):
        self.maze.update_obstacles(self.dt)
        d_theta = 0.0
        if action == 1: d_theta = 0.2
        if action == 2: d_theta = -0.2
        self.maze.robot['ori'] += d_theta

        current_ori, pos, t = self.maze.robot['ori'], np.array(self.maze.robot['pos']), self.maze.time
        va_vec = np.array([self.propulsion_speed * math.cos(current_ori), self.propulsion_speed * math.sin(current_ori)])
        vo_vec = self.maze.vf.get_velocity(pos[0], pos[1], t)
        new_pos, vs_vec = kinematic_update(pos, va_vec, vo_vec, self.dt)
        self.maze.robot['pos'] = list(new_pos)
        self.maze.time += self.dt

        step_energy = self.drag_c * (self.propulsion_speed ** 3) * self.dt
        goal = np.array(self.maze.goal_pos)
        d_old, d_new = np.linalg.norm(pos - goal), np.linalg.norm(new_pos - goal)

        reward = -self.args.time_penalty
        reward += (d_old - d_new) * self.args.progress_weight

        vec_to_goal = goal - new_pos
        norm_goal, vs_norm = np.linalg.norm(vec_to_goal), np.linalg.norm(vs_vec)
        if norm_goal > 1e-3 and vs_norm > 1e-3:
            reward += self.args.dir_weight * (np.dot(vs_vec, vec_to_goal) / (vs_norm * norm_goal))

        done = False
        if self.maze.is_collision(new_pos[0], new_pos[1]):
            reward += self.args.collision_reward
            self.maze.robot['pos'] = list(pos)
        if self.maze.is_goal(new_pos[0], new_pos[1]):
            reward += self.args.goal_reward
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
    def __init__(self, s_dim, a_dim, args, device):
        self.args = args
        self.device = device
        self.eval_net = Net(s_dim, a_dim).to(device)
        self.target_net = Net(s_dim, a_dim).to(device)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.opt = optim.Adam(self.eval_net.parameters(), lr=args.base_lr)
        self.memory = PrioritizedMemory(args)
        self.steps = 0
        self.eps = 1.0 if args.phase == 'train' else 0.05

    def act(self, s):
        if random.random() < self.eps: return random.randint(0, 2)
        s_t = torch.FloatTensor(s).unsqueeze(0).to(self.device)
        with torch.no_grad(): return self.eval_net(s_t).argmax().item()

    def store_transition(self, s, a, r, ns, done):
        transition = (s, a, r, ns, done)
        self.memory.n_step_buffer.append(transition)
        if len(self.memory.n_step_buffer) < self.args.n_step and not done: return
        R, gamma = 0, 1
        for (_, _, r_i, _, _) in self.memory.n_step_buffer:
            R += r_i * gamma; gamma *= self.args.gamma
        s0, a0 = self.memory.n_step_buffer[0][:2]
        nsn, donen = self.memory.n_step_buffer[-1][3:]
        max_p = np.max(self.memory.tree.tree[-self.memory.tree.capacity:])
        if max_p == 0: max_p = self.args.abs_error_upper
        self.memory.add(max_p, (s0, a0, R, nsn, donen))
        if done: self.memory.n_step_buffer.clear()

    def learn(self):
        if self.memory.tree.count < self.args.batch_size: return
        batch, idxs, is_weights = self.memory.sample(self.args.batch_size)
        s, a, r, ns, d = zip(*batch)
        s_t = torch.FloatTensor(np.array(s)).to(self.device)
        a_t = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r_t = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns_t = torch.FloatTensor(np.array(ns)).to(self.device)
        d_t = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        w_t = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        q_eval = self.eval_net(s_t).gather(1, a_t)
        with torch.no_grad():
            a_next = self.eval_net(ns_t).argmax(dim=1, keepdim=True)
            q_next = self.target_net(ns_t).gather(1, a_next)
            q_target = r_t + (self.args.gamma ** self.args.n_step) * q_next * (1 - d_t)

        td_errors = (q_target - q_eval).detach().cpu().numpy().flatten()
        loss = (w_t * (q_target - q_eval).pow(2)).mean()
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        for i in range(self.args.batch_size): self.memory.update(idxs[i], td_errors[i])
        self.steps += 1
        if self.steps % 200 == 0: self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.eps > 0.05: self.eps *= 0.99995

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.eval_net.state_dict(), path)
        Prog.log("BEST", f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.eval_net.load_state_dict(torch.load(path, map_location=self.device))
            self.target_net.load_state_dict(self.eval_net.state_dict())
            Prog.log("TEST", f"Model loaded: {path}")
        else: Prog.log("WARN", f"Model not found: {path}")

# ================= 辅助函数 =================
def calc_len(traj):
    if len(traj) < 2: return 0
    return sum(math.hypot(traj[i + 1][0] - traj[i][0], traj[i + 1][1] - traj[i][1]) for i in range(len(traj) - 1))

def save_plot(maze, traj, filename, title_text=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    maze.draw(ax)
    t_np = np.array(traj)
    ax.plot(t_np[:, 0], t_np[:, 1], 'r-', lw=2, label="Trajectory")
    ax.legend(); ax.set_title(title_text)
    plt.savefig(filename); plt.close(fig)

def save_gif(maze, robot_hist, obs_hist, filename, title_text=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    skip = 2; frames = range(0, len(robot_hist), skip)
    def update(frame_idx):
        ax.clear(); maze.robot['pos'] = list(robot_hist[frame_idx])
        current_obs_data = obs_hist[frame_idx]
        for i, o in enumerate(maze.obstacles): o['c'] = current_obs_data[i]
        maze.draw(ax)
        hist_np = np.array(robot_hist[:frame_idx + 1])
        if len(hist_np) > 1: ax.plot(hist_np[:, 0], hist_np[:, 1], 'r-', lw=2)
        ax.set_title(f"{title_text} | Step {frame_idx}")
    ani = FuncAnimation(fig, update, frames=frames, interval=100)
    ani.save(filename, writer='pillow', fps=15); plt.close(fig)

# ================= 训练函数 =================
def train_eval(args, device):
    # 初始化环境和智能体
    maze = OceanMaze(args)
    sensor = Sensor(maze, args)
    executor = Executor(maze, args)
    agent = Agent(s_dim=12, a_dim=3, args=args, device=device)
    
    model_path = os.path.join(args.output_dir, args.model_name)
    final_model_path = os.path.join(args.output_dir, "ddqn_final_state.pth")

    if args.resume:
        agent.load(model_path) # Resume from best model if exists

    Prog.log("INIT", "START TRAINING (Curriculum: Easy -> Hard)")
    best_score = float('inf')
    
    for ep in range(args.episodes):
        # Curriculum Learning
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

        for t in range(args.max_steps):
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
                score = plen + args.lambda_score * ep_energy
                Prog.log("LEARN", f"Ep {ep} (Diff {difficulty:.1f}) | Len:{plen:.0f} | Score:{score:.1f}")
                
                if difficulty > 0.5 and score < best_score:
                    best_score = score
                    agent.save(model_path)
                    title = f"Ep {ep} Best Score: {best_score:.1f} (Diff {difficulty:.1f})"
                    save_plot(maze, traj, f"train_best_ep{ep}.png", title)
                    Prog.log("BEST", "New Record!")
                break

        if ep % 20 == 0:
            Prog.log("LEARN", f"Ep {ep} running... Eps: {agent.eps:.3f}")

    # Save final model
    agent.save(final_model_path)
    Prog.log("INIT", f"Training Finished! Final model saved to {final_model_path}")

# ================= 测试函数 =================
def test_eval(args, device):
    # 初始化环境和智能体
    maze = OceanMaze(args)
    sensor = Sensor(maze, args)
    executor = Executor(maze, args)
    agent = Agent(s_dim=12, a_dim=3, args=args, device=device)
    
    model_path = os.path.join(args.output_dir, args.model_name)
    agent.load(model_path)
    
    os.makedirs(args.result_dir, exist_ok=True)
    print(f"GIFs will be saved to: {os.path.abspath(args.result_dir)}")

    success_count = 0
    # Test 10 episodes
    for i in range(10):
        maze.reset(difficulty=1.0)
        traj = [tuple(maze.robot['pos'])]
        obs_log = [[o['c'].copy() for o in maze.obstacles]]
        ep_eng = 0
        done = False

        # Testing usually allows more steps to ensure completion
        test_max_steps = args.max_steps * 2 
        
        for step_i in range(test_max_steps):
            s = sensor.get_state()
            a = agent.act(s) # eps is handled inside Agent based on phase
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

        if done:
            title = f"Test {i + 1} | Success"
            file_name = f"test_{i + 1:02d}_success.gif"
            full_path = os.path.join(args.result_dir, file_name)
            save_gif(maze, traj, obs_log, full_path, title)
        else:
            print(f"Test {i + 1:02d}: Fail (Skipping GIF)")

    print("-" * 30)
    print(f"Final Success Rate: {success_count / 10 * 100:.1f}%")

# ================= 主程序入口 =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDQN Path Planning in Ocean Currents")

    # 1. Path & Mode
    parser.add_argument('--phase', type=str, default='train', choices=['train', 'test'], help='Phase: train or test.')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='Directory to save checkpoints.')
    parser.add_argument('--model_name', type=str, default='ddqn_final_state.pth', help='Model filename to load/save.')
    parser.add_argument('--result_dir', type=str, default='test_results_gif', help='Directory for test results.')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint.')

    # 2. Physics (论文参数)
    parser.add_argument('--propulsion_speed', type=float, default=5.0, help='Robot propulsion speed (Va).')
    parser.add_argument('--drag_c', type=float, default=0.5, help='Drag coefficient.')
    parser.add_argument('--time_step', type=float, default=0.5, help='Simulation time step (dt).')
    parser.add_argument('--start_radius', type=float, default=15, help='Start point radius.')
    parser.add_argument('--goal_radius', type=float, default=30, help='Goal point radius.')

    # 3. Training Hyperparams
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor.')
    parser.add_argument('--base_lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--max_steps', type=int, default=500, help='Max steps per episode.') # 训练用小一点，测试时可大
    parser.add_argument('--episodes', type=int, default=3000, help='Total training episodes.')
    parser.add_argument('--n_step', type=int, default=3, help='N-step return.')

    # 4. PER (Prioritized Experience Replay)
    parser.add_argument('--memory_capacity', type=int, default=50000, help='Replay buffer capacity.')
    parser.add_argument('--per_alpha', type=float, default=0.6, help='PER Alpha.')
    parser.add_argument('--per_beta', type=float, default=0.4, help='PER Beta initial.')
    parser.add_argument('--per_beta_inc', type=float, default=0.0005, help='PER Beta increment.')
    parser.add_argument('--abs_error_upper', type=float, default=1.0, help='PER absolute error upper bound.')

    # 5. Rewards
    parser.add_argument('--progress_weight', type=float, default=2.0, help='Reward weight for progress.')
    parser.add_argument('--dir_weight', type=float, default=1.5, help='Reward weight for direction.')
    parser.add_argument('--time_penalty', type=float, default=0.5, help='Step penalty.')
    parser.add_argument('--goal_reward', type=float, default=5000.0, help='Reward for reaching goal.')
    parser.add_argument('--collision_reward', type=float, default=-50.0, help='Penalty for collision.')
    parser.add_argument('--lambda_score', type=float, default=0.0005, help='Score balance coefficient.')

    # 6. Device
    parser.add_argument('--gpus', type=str, default="0", help="GPU IDs (e.g. '0').")

    args = parser.parse_args()

    # === Setup Device ===
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[System] Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[System] Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("[System] Using CPU")

    # === Execution Phase ===
    if args.phase == 'train':
        train_eval(args, device)
    elif args.phase == 'test':
        test_eval(args, device)
