# -*- coding: utf-8 -*-
"""
flow_dynamics.py
----------------
包含论文 "Adaptive Energy-Aware Navigation..." 中的核心物理模型：
1. Lamb-Oseen 涡旋流场模型 (Eq. 2.1)
2. 运动学合成模型 (Eq. 3.6)
"""

import numpy as np
import math


class LambVortexField:
    """
    基于论文 Eq (2.1) 实现的 Lamb 涡旋流场
    """

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.vortices = []
        self.uniform_flow = np.array([0.0, 0.0])  # 可选背景流
        self._init_paper_vortices()

    def _init_paper_vortices(self):
        """
        大幅降低 D 值，确保最大流速在机器人可控范围内 (例如 2.0 m/s 左右)
        """
        self.vortices = [
            # 原来是 50000 -> 改为 3000
            # 原来是 60000 -> 改为 4000
            {'x': 300, 'y': 500, 'D': 3000, 'R': 150, 'vx': 0.1, 'vy': 0.05},
            {'x': 700, 'y': 500, 'D': -4000, 'R': 150, 'vx': -0.1, 'vy': -0.05},
            {'x': 500, 'y': 200, 'D': 2000, 'R': 120, 'vx': 0.0, 'vy': 0.1},
            {'x': 500, 'y': 800, 'D': -2000, 'R': 120, 'vx': 0.0, 'vy': -0.1},
            {'x': 500, 'y': 500, 'D': 1500, 'R': 80, 'vx': 0.0, 'vy': 0.0}
        ]

    def get_velocity(self, x, y, t=0.0):
        """
        根据论文 Eq (2.1) 计算位置 (x,y) 在时间 t 的洋流速度 Vo
        """
        v_total = self.uniform_flow.copy()

        for v in self.vortices:
            # 涡旋中心随时间漂移
            cx = v['x'] + v['vx'] * t
            cy = v['y'] + v['vy'] * t

            dx = x - cx
            dy = y - cy
            r2 = dx ** 2 + dy ** 2 + 1e-6  # 避免除零

            # --- 论文 Eq (2.1) 核心公式 ---
            # 系数部分: D / (2 * pi * r^2) * (1 - exp(-r^2/R^2))
            # 这里的 r^2 在分母，论文公式是 ||c-c0||^2
            factor = (v['D'] / (2 * np.pi * r2)) * (1 - np.exp(-r2 / (v['R'] ** 2)))

            # 线速度转化为分量：
            # vx = -dy * factor
            # vy =  dx * factor
            v_total[0] += -dy * factor
            v_total[1] += dx * factor

        return v_total


def kinematic_update(pos, propulsion_vec, flow_vec, dt):
    """
    基于论文 Eq (3.6) 的运动学更新
    Vs (对地速度) = Va (推进速度) + Vo (洋流速度)

    参数:
    - pos: 当前位置 [x, y]
    - propulsion_vec: 机器人推进矢量 Va [vx, vy]
    - flow_vec: 洋流矢量 Vo [vx, vy]
    - dt: 时间步长

    返回:
    - new_pos: 新位置
    - ground_vel: 对地速度 Vs
    """
    # 向量合成
    ground_vel = propulsion_vec + flow_vec

    # 位置更新: P_new = P_old + Vs * dt
    new_pos = pos + ground_vel * dt

    return new_pos, ground_vel