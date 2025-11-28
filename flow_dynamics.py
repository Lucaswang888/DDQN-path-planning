# -*- coding: utf-8 -*-
"""
flow_dynamics.py
----------------
一个简单的二维力学积分模块：

给定：
- 加速度向量场 a(x, t)，这是“力场 / 加速度场”
- 粒子当前位置 x(t) 和当前速度 v(t)
- 时间步长 dt

返回：
- 下一时刻的位置 x(t + dt)
- 下一时刻的速度 v(t + dt)

默认使用半隐式欧拉（symplectic Euler）积分：
    v_{t+dt} = v_t + a(x_t, t) * dt
    x_{t+dt} = x_t + v_{t+dt} * dt
"""

from __future__ import annotations

from typing import Callable, Tuple
import numpy as np

# 加速度场类型：接受位置和时间，返回加速度向量
AccelerationField = Callable[[np.ndarray, float], np.ndarray]


def integrate_step(
    acc_field: AccelerationField,
    position: np.ndarray,
    velocity: np.ndarray,
    t: float,
    dt: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对单个粒子做一步时间积分。

    参数
    ----
    acc_field : callable
        加速度向量场函数 a(x, t)，返回形如 [ax, ay] 的 numpy 数组。
    position : np.ndarray
        当前粒子位置向量 x(t)，形状应为 (2,) 或 (n_dim,)。
    velocity : np.ndarray
        当前粒子速度向量 v(t)，形状与 position 相同。
    t : float
        当前时间 t（秒）。
    dt : float
        时间步长 Δt（秒），即“时间精度”。

    返回
    ----
    next_position : np.ndarray
        下一时刻位置 x(t + dt)。
    next_velocity : np.ndarray
        下一时刻速度 v(t + dt)。
    """
    pos = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float)

    # 计算当前时刻的加速度 a(x(t), t)
    acc = np.asarray(acc_field(pos, t), dtype=float)

    # 半隐式欧拉：先更新速度，再用新速度更新位置
    next_velocity = vel + acc * dt
    next_position = pos + next_velocity * dt

    return next_position, next_velocity
