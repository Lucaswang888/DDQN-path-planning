# -*- coding: utf-8 -*-
# main.py
# 固定起点终点与洋流地图的单任务训练/测试脚本

import copy
import sys
import time
import datetime
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import collections
import math
import os

# 引入你的动力学模块
from flow_dynamics import integrate_step

# ================= 配置区域 =================
# 【核心开关】 True = 训练模式; False = 测试模式 (加载模型并演示)
IS_TRAINING = True 

# 模型保存路径
MODEL_PATH = "checkpoints/ddqn_fixed_task.pth"

# 固定任务配置
START_POS = [100, 100]  # 起点
GOAL_POS  = [900, 900]  # 终点
# ===========================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_DDQN = True

# 奖励参数 (针对固定任务微调)
PROGRESS_WEIGHT = 2.0      # 加大进度奖励，鼓励快速靠近
ENERGY_WEIGHT   = 0.0005
TIME_PENALTY    = 0.05     # 加大时间惩罚，逼迫它走更短路径
GOAL_REWARD     = 1000.0   # 终点大奖
COLLISION_REWARD = -10.0   # 稍微给点碰撞惩罚，让它长记性

# 绘图设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class Prog:
    COLORS = dict(
        INIT="\033[95m", LEARN="\033[94m", EVAL="\033[96m",
        BEST="\033[92m", WARN="\033[93m", STOP="\033[91m", END="\033[0m"
    )
    @staticmethod
    def log(kind, msg):
        c = Prog.COLORS.get(kind, "")
        e = Prog.COLORS["END"]
        print(f"{datetime.datetime.now().strftime('%H:%M:%S')} {c}[{kind}]{e} {msg}")
        sys.stdout.flush()

def calculate_path_length(trajectory):
    if len(trajectory) < 2: return 0.0
    return sum(math.sqrt((trajectory[i+1][0]-trajectory[i][0])**2 + 
                         (trajectory[i+1][1]-trajectory[i][1])**2) 
               for i in range(len(trajectory)-1))

# ===== 1. 简化的固定向量流场 =====
class VectorField:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.vortices = []
        self.uniform_flows = []
        self._create_fixed_field() # 不再随机，使用固定场
        
        # 预计算网格用于绘图
        self.grid_resolution = 40
        self._precompute_grid()

    def _create_fixed_field(self):
        """定义一个固定的复杂洋流场"""
        # 一个向右上方的背景流
        self.uniform_flows.append((0.8, 0.8))
        
        # 添加几个固定的涡旋 (x, y, strength, radius, direction)
        # 阻碍直接冲向终点的大涡旋
        self.vortices.append((500, 500, 4.0, 250, 1))   # 中央逆时针
        self.vortices.append((200, 200, 3.0, 150, -1))  # 左下顺时针
        self.vortices.append((800, 800, 3.0, 150, -1))  # 右上顺时针

    def get_flow_vector(self, x, y):
        vx, vy = 0.0, 0.0
        for ux, uy in self.uniform_flows:
            vx += ux
            vy += uy
            
        for cx, cy, strength, radius, direction in self.vortices:
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx*dx + dy*dy) + 1e-6
            decay = math.exp(-(dist**2)/(2*radius**2))
            tangent = strength * decay / dist
            if direction > 0: # 逆时针
                vx += -dy * tangent
                vy += dx * tangent
            else:
                vx += dy * tangent
                vy += -dx * tangent
        return np.array([vx, vy], dtype=np.float32)

    def _precompute_grid(self):
        x = np.linspace(0, self.width, self.grid_resolution)
        y = np.linspace(0, self.height, self.grid_resolution)
        self.grid_X, self.grid_Y = np.meshgrid(x, y)
        self.grid_U = np.zeros_like(self.grid_X)
        self.grid_V = np.zeros_like(self.grid_Y)
        self.grid_M = np.zeros_like(self.grid_X)
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                v = self.get_flow_vector(self.grid_X[i,j], self.grid_Y[i,j])
                self.grid_U[i,j], self.grid_V[i,j] = v[0], v[1]
                self.grid_M[i,j] = np.linalg.norm(v)

    def draw(self, ax):
        ax.contourf(self.grid_X, self.grid_Y, self.grid_M, levels=15, cmap='Blues', alpha=0.3)
        ax.quiver(self.grid_X, self.grid_Y, self.grid_U, self.grid_V, color='blue', alpha=0.5)

# ===== 2. 简化的固定障碍物与地图 =====
class CircleObstacle:
    def __init__(self, x, y, r): self.c = np.array([x,y]); self.r = r
    def contains(self, x, y): return np.linalg.norm(np.array([x,y])-self.c) <= self.r
    def ray_cast(self, start, angle, max_d):
        d = np.array([math.cos(angle), math.sin(angle)])
        oc = start - self.c
        a, b = np.dot(d,d), 2*np.dot(oc,d)
        c = np.dot(oc,oc) - self.r**2
        delta = b*b - 4*a*c
        if delta < 0: return max_d
        t = (-b - math.sqrt(delta))/(2*a)
        return t if 0 < t < max_d else max_d
    def draw(self, ax): ax.add_patch(plt.Circle(self.c, self.r, color='gray', alpha=0.7))

class OceanMaze:
    def __init__(self):
        self.width = 1000
        self.height = 1000
        self.vector_field = VectorField(self.width, self.height)
        self.obstacles = [
            CircleObstacle(300, 600, 80),
            CircleObstacle(700, 400, 80),
            CircleObstacle(500, 200, 60),
            CircleObstacle(500, 800, 60),
        ]
        
        # === 核心修改：固定起点和终点 ===
        self.start_pos = list(START_POS)
        self.destination = list(GOAL_POS)
        self.dest_radius = 30
        
        self.robot = {'pos': self.start_pos.copy(), 'ori': 0, 'rad': 15}

    def reset(self):
        """重置机器人到固定起点"""
        self.robot['pos'] = self.start_pos.copy()
        self.robot['ori'] = 0 # 每次都脸朝右开始

    def is_collision(self, x, y):
        if not (0 <= x <= self.width and 0 <= y <= self.height): return True
        for o in self.obstacles:
            if o.contains(x, y): return True
        return False
        
    def is_goal(self, x, y):
        return math.hypot(x - self.destination[0], y - self.destination[1]) <= self.dest_radius

    def draw(self, ax):
        self.vector_field.draw(ax)
        for o in self.obstacles: o.draw(ax)
        ax.add_patch(plt.Circle(self.start_pos, 15, color='green', label='Start'))
        ax.add_patch(plt.Circle(self.destination, self.dest_radius, color='red', alpha=0.5, label='Goal'))
        ax.set_xlim(0, self.width); ax.set_ylim(0, self.height)

# ===== 3. 传感器与执行器 =====
class Sensor:
    def __init__(self, maze): self.maze = maze
    def get_state(self):
        pos, ori = self.maze.robot['pos'], self.maze.robot['ori']
        feats = []
        # 8个方向的雷达
        for i in range(8):
            ang = ori + i * (2*math.pi/8)
            d = 200 # max range
            # 简化：只检测墙壁距离
            for obs in self.maze.obstacles:
                d = min(d, obs.ray_cast(np.array(pos), ang, 200))
            feats.append(d / 200.0)
        
        # 目标相对信息
        dx, dy = self.maze.destination[0]-pos[0], self.maze.destination[1]-pos[1]
        dist = math.sqrt(dx*dx+dy*dy)
        angle = math.atan2(dy, dx) - ori
        angle = (angle + math.pi) % (2*math.pi) - math.pi
        
        # 流场信息
        flow = self.maze.vector_field.get_flow_vector(pos[0], pos[1])
        
        return np.array(feats + [dist/1414.0, angle/math.pi, flow[0], flow[1]], dtype=np.float32)

class Executor:
    def __init__(self, maze):
        self.maze = maze
        self.dt = 1.0
        self.vel = np.zeros(2)

    def step(self, action):
        # Action: 0=Keep, 1=Left, 2=Right
        if action == 1: self.maze.robot['ori'] += 0.5
        if action == 2: self.maze.robot['ori'] -= 0.5
        
        # 主动推力
        thrust = 15.0 # 力的大小
        ax_act = thrust * math.cos(self.maze.robot['ori'])
        ay_act = thrust * math.sin(self.maze.robot['ori'])
        
        # 动力学积分
        pos_np = np.array(self.maze.robot['pos'])
        
        def acc_func(p, t):
            f = self.maze.vector_field.get_flow_vector(p[0], p[1])
            return np.array([ax_act + f[0], ay_act + f[1]]) # 推力 + 洋流
            
        new_pos, new_vel = integrate_step(acc_func, pos_np, self.vel, 0, self.dt)
        self.vel = new_vel
        self.maze.robot['pos'] = list(new_pos)
        
        # 奖励计算
        reward = -TIME_PENALTY
        d_old = math.hypot(pos_np[0]-self.maze.destination[0], pos_np[1]-self.maze.destination[1])
        d_new = math.hypot(new_pos[0]-self.maze.destination[0], new_pos[1]-self.maze.destination[1])
        reward += (d_old - d_new) * PROGRESS_WEIGHT
        
        # 能量消耗 (逆流惩罚)
        flow = self.maze.vector_field.get_flow_vector(pos_np[0], pos_np[1])
        dot = ax_act*flow[0] + ay_act*flow[1]
        if dot < 0: reward -= ENERGY_WEIGHT * abs(dot) # 逆流更费电
        
        done = False
        if self.maze.is_collision(new_pos[0], new_pos[1]):
            reward += COLLISION_REWARD
            # 撞墙稍微弹回来一点，防止卡死，但不结束
            self.maze.robot['pos'] = list(pos_np) 
            self.vel = np.zeros(2)
        
        if self.maze.is_goal(new_pos[0], new_pos[1]):
            reward += GOAL_REWARD
            done = True
            
        return reward, done

# ===== 4. DQN Agent =====
class DQN(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, a_dim)
        )
    def forward(self, x): return self.net(x)

class Agent:
    def __init__(self, state_dim, action_dim):
        self.eval_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net = DQN(state_dim, action_dim).to(DEVICE)
        self.target_net.load_state_dict(self.eval_net.state_dict())
        self.opt = torch.optim.Adam(self.eval_net.parameters(), lr=3e-4)
        self.mem = collections.deque(maxlen=20000)
        self.batch = 128
        self.eps = 1.0 if IS_TRAINING else 0.05
        self.steps = 0

    def act(self, s):
        if random.random() < self.eps: return random.randint(0, 2)
        state = torch.FloatTensor(s).unsqueeze(0).to(DEVICE)
        return self.eval_net(state).argmax().item()

    def learn(self):
        if len(self.mem) < self.batch: return
        batch = random.sample(self.mem, self.batch)
        s, a, r, ns, d = zip(*batch)
        
        s_t = torch.FloatTensor(np.array(s)).to(DEVICE)
        a_t = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
        r_t = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)
        ns_t = torch.FloatTensor(np.array(ns)).to(DEVICE)
        d_t = torch.FloatTensor(d).unsqueeze(1).to(DEVICE)

        q_curr = self.eval_net(s_t).gather(1, a_t)
        q_next = self.target_net(ns_t).max(1)[0].unsqueeze(1)
        q_targ = r_t + 0.99 * q_next * (1 - d_t)
        
        loss = nn.MSELoss()(q_curr, q_targ)
        self.opt.zero_grad(); loss.backward(); self.opt.step()
        
        self.steps += 1
        if self.steps % 200 == 0: self.target_net.load_state_dict(self.eval_net.state_dict())
        if self.eps > 0.05: self.eps *= 0.9999

    def save(self, path):
        torch.save(self.eval_net.state_dict(), path)
        Prog.log("BEST", f"Model saved to {path}")

    def load(self, path):
        if os.path.exists(path):
            self.eval_net.load_state_dict(torch.load(path, map_location=DEVICE))
            self.target_net.load_state_dict(self.eval_net.state_dict())
            Prog.log("INIT", f"Loaded model from {path}")
        else:
            Prog.log("WARN", "Model file not found!")

# ===== 主程序逻辑 =====
if __name__ == "__main__":
    maze = OceanMaze()
    sensor = Sensor(maze)
    executor = Executor(maze)
    agent = Agent(12, 3) # State dim ~ 12, Action dim = 3

    if IS_TRAINING:
        Prog.log("INIT", f"START TRAINING (Fixed Task: {START_POS} -> {GOAL_POS})")
        episodes = 1500
        best_len = float('inf')
        
        for ep in range(episodes):
            maze.reset()
            executor.vel = np.zeros(2) # 重置速度
            traj = [tuple(maze.robot['pos'])]
            ep_reward = 0
            
            for t in range(600): # Max steps
                s = sensor.get_state()
                a = agent.act(s)
                r, done = executor.step(a)
                ns = sensor.get_state()
                agent.mem.append((s, a, r, ns, done))
                agent.learn()
                
                ep_reward += r
                traj.append(tuple(maze.robot['pos']))
                
                if done:
                    plen = calculate_path_length(traj)
                    Prog.log("LEARN", f"Ep {ep} | Reward {ep_reward:.1f} | Len {plen:.1f}")
                    if plen < best_len:
                        best_len = plen
                        agent.save(MODEL_PATH) # 只有破纪录才保存
                    break
            
            # 没到终点也可以衰减一下epsilon
            if ep % 10 == 0: 
                Prog.log("LEARN", f"Ep {ep} running... Eps: {agent.eps:.3f}")

    else:
        # === 测试模式 ===
        Prog.log("INIT", "START TESTING PHASE")
        agent.load(MODEL_PATH)
        maze.reset()
        executor.vel = np.zeros(2)
        traj = [tuple(maze.robot['pos'])]
        
        done = False
        steps = 0
        while not done and steps < 1000:
            s = sensor.get_state()
            a = agent.act(s) # 此时 eps 很小，基本是贪心策略
            r, done = executor.step(a)
            traj.append(tuple(maze.robot['pos']))
            steps += 1
            if steps % 50 == 0: print(f"Step {steps}...")
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10,10))
        maze.draw(ax)
        t_np = np.array(traj)
        ax.plot(t_np[:,0], t_np[:,1], 'r-', linewidth=2, label='Trajectory')
        ax.set_title(f"Test Run: {len(traj)} steps")
        plt.legend()
        plt.show()
