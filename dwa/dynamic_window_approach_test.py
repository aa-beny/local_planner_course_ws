"""
Mobile robot motion planning sample with Dynamic Window Approach

原始版本：https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/DynamicWindowApproach/dynamic_window_approach.py
教學整理：保留原本 PythonRobotics 架構，補上中文註解，並顯示所有候選軌跡。

畫面說明：
- 灰色細線：這一輪所有候選軌跡
- 綠色粗線：DWA 最後選中的最佳軌跡
- 黑色點：障礙物
- 藍色 X：目標點
- 紅色 X：機器人目前位置
"""

import math
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

# 是否顯示動畫。若只想跑結果、不顯示畫面，可以改成 False。
show_animation = True

# 是否畫出所有候選軌跡。教學時建議 True，才看得出 DWA「先試很多條，再選最好的一條」。
show_all_candidate_trajectories = True


class RobotType(Enum):
    """機器人外型，用於碰撞檢查與畫圖。"""

    circle = 0
    rectangle = 1


class Config:
    """
    DWA 模擬參數。

    對應投影片：
    - 速度限制 / 加速度限制
    - 速度解析度
    - 預測時間
    - 評價函數權重
    """

    def __init__(self):
        # ===== 速度與加速度限制 =====
        self.max_speed = 1.0  # [m/s] 最大線速度 v
        self.min_speed = -0.5  # [m/s] 最小線速度 v；負值代表允許倒退
        self.max_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s] 最大角速度 ω
        self.max_accel = 0.2  # [m/s^2] 最大線加速度
        self.max_delta_yaw_rate = 40.0 * math.pi / 180.0  # [rad/s^2] 最大角加速度

        # ===== 速度採樣解析度 =====
        self.v_resolution = 0.01  # [m/s] 線速度採樣間隔，越小候選速度越多
        self.yaw_rate_resolution = 0.1 * math.pi / 180.0  # [rad/s] 角速度採樣間隔

        # ===== 軌跡預測設定 =====
        self.dt = 0.1  # [s] 每次模擬往前推進的時間間隔
        self.predict_time = 3.0  # [s] 每組速度要預測多久的未來軌跡

        # ===== 評價函數權重 =====
        # 這份程式採用 cost 最小化：cost 越小，軌跡越好。
        # 對應投影片的 heading / velocity / dist，只是寫法從「分數最大化」改成「成本最小化」。
        self.to_goal_cost_gain = 1.0  # 朝向目標的 cost 權重
        self.speed_cost_gain = 1.0  # 速度 cost 權重
        self.obstacle_cost_gain = 1.0  # 障礙物 cost 權重

        # 若機器人在障礙物前剛好選到 v=0、ω=0，可能會卡住；這個值用來避免卡死。
        self.robot_stuck_flag_cons = 0.001

        # 預設機器人外型。main() 裡可以改成 circle 或 rectangle。
        self.robot_type = RobotType.circle

        # ===== 圓形機器人參數 =====
        # robot_radius 也用於判斷是否到達目標。
        # self.robot_radius = 1.0  # [m] 圓形機器人半徑
        self.robot_radius = 0.5  # [m] 圓形機器人半徑

        # ===== 矩形機器人參數 =====
        self.robot_width = 0.5  # [m] 矩形機器人寬度
        self.robot_length = 1.2  # [m] 矩形機器人長度

        # ===== 障礙物座標 =====
        # 每一列是一個障礙物點：[x, y]
        self.ob = np.array([
    # ===== 外圍牆 =====
    [-1.0, -1.0], [-0.5, -1.0], [0.0, -1.0], [0.5, -1.0],
    [1.0, -1.0], [1.5, -1.0], [2.0, -1.0], [2.5, -1.0],
    [3.0, -1.0], [3.5, -1.0], [4.0, -1.0], [4.5, -1.0],
    [5.0, -1.0], [5.5, -1.0], [6.0, -1.0], [6.5, -1.0],
    [7.0, -1.0], [7.5, -1.0], [8.0, -1.0], [8.5, -1.0],
    [9.0, -1.0], [9.5, -1.0], [10.0, -1.0], [10.5, -1.0],
    [11.0, -1.0], [11.5, -1.0], [12.0, -1.0],

    [-1.0, 12.0], [-0.5, 12.0], [0.0, 12.0], [0.5, 12.0],
    [1.0, 12.0], [1.5, 12.0], [2.0, 12.0], [2.5, 12.0],
    [3.0, 12.0], [3.5, 12.0], [4.0, 12.0], [4.5, 12.0],
    [5.0, 12.0], [5.5, 12.0], [6.0, 12.0], [6.5, 12.0],
    [7.0, 12.0], [7.5, 12.0], [8.0, 12.0], [8.5, 12.0],
    [9.0, 12.0], [9.5, 12.0], [10.0, 12.0], [10.5, 12.0],
    [11.0, 12.0], [11.5, 12.0], [12.0, 12.0],

    [-1.0, -0.5], [-1.0, 0.0], [-1.0, 0.5], [-1.0, 1.0],
    [-1.0, 1.5], [-1.0, 2.0], [-1.0, 2.5], [-1.0, 3.0],
    [-1.0, 3.5], [-1.0, 4.0], [-1.0, 4.5], [-1.0, 5.0],
    [-1.0, 5.5], [-1.0, 6.0], [-1.0, 6.5], [-1.0, 7.0],
    [-1.0, 7.5], [-1.0, 8.0], [-1.0, 8.5], [-1.0, 9.0],
    [-1.0, 9.5], [-1.0, 10.0], [-1.0, 10.5], [-1.0, 11.0],
    [-1.0, 11.5],

    [12.0, -0.5], [12.0, 0.0], [12.0, 0.5], [12.0, 1.0],
    [12.0, 1.5], [12.0, 2.0], [12.0, 2.5], [12.0, 3.0],
    [12.0, 3.5], [12.0, 4.0], [12.0, 4.5], [12.0, 5.0],
    [12.0, 5.5], [12.0, 6.0], [12.0, 6.5], [12.0, 7.0],
    [12.0, 7.5], [12.0, 8.0], [12.0, 8.5], [12.0, 9.0],
    [12.0, 9.5], [12.0, 10.0], [12.0, 10.5], [12.0, 11.0],
    [12.0, 11.5],

    # ===== 第一面內牆：x = 3，通道在 y = 4.0 ~ 5.5 =====
    [3.0, -0.5], [3.0, 0.0], [3.0, 0.5], [3.0, 1.0],
    [3.0, 1.5], [3.0, 2.0], [3.0, 2.5], [3.0, 3.0],
    [3.0, 3.5],
    [3.0, 6.0], [3.0, 6.5], [3.0, 7.0], [3.0, 7.5],
    [3.0, 8.0], [3.0, 8.5], [3.0, 9.0], [3.0, 9.5],
    [3.0, 10.0], [3.0, 10.5], [3.0, 11.0], [3.0, 11.5],

    # ===== 第二面內牆：x = 6，通道在 y = 5.5 ~ 7.0 =====
    [6.0, -0.5], [6.0, 0.0], [6.0, 0.5], [6.0, 1.0],
    [6.0, 1.5], [6.0, 2.0], [6.0, 2.5], [6.0, 3.0],
    [6.0, 3.5], [6.0, 4.0], [6.0, 4.5], [6.0, 5.0],
    [6.0, 7.5], [6.0, 8.0], [6.0, 8.5], [6.0, 9.0],
    [6.0, 9.5], [6.0, 10.0], [6.0, 10.5], [6.0, 11.0],
    [6.0, 11.5],

    # ===== 第三面內牆：x = 8.5，通道在 y = 7.5 ~ 9.0 =====
    [8.5, -0.5], [8.5, 0.0], [8.5, 0.5], [8.5, 1.0],
    [8.5, 1.5], [8.5, 2.0], [8.5, 2.5], [8.5, 3.0],
    [8.5, 3.5], [8.5, 4.0], [8.5, 4.5], [8.5, 5.0],
    [8.5, 5.5], [8.5, 6.0], [8.5, 6.5], [8.5, 7.0],
    [8.5, 9.5], [8.5, 10.0], [8.5, 10.5], [8.5, 11.0],
    [8.5, 11.5],
])

    @property
    def robot_type(self):
        return self._robot_type

    @robot_type.setter
    def robot_type(self, value):
        """限制 robot_type 一定要是 RobotType，避免傳錯型別。"""
        if not isinstance(value, RobotType):
            raise TypeError("robot_type must be an instance of RobotType")
        self._robot_type = value


# 全域設定物件，原始程式也是用這個 config 給 main() 和各函式使用。
config = Config()


def dwa_control(x, config, goal, ob):
    """
    DWA 控制主函式。

    對應投影片：DWA 基本流程。
    這裡做兩件事：
    1. 計算動態窗口，也就是目前可採樣的速度範圍。
    2. 在速度範圍內採樣、預測、評分，選出最佳速度。
    """

    # dw = [v_min, v_max, omega_min, omega_max]
    dw = calc_dynamic_window(x, config)

    # u 是最佳控制輸入 [v, omega]
    # trajectory 是最佳速度對應的預測軌跡
    # all_trajectories 是這一輪所有候選軌跡，主要用來畫給學生看
    u, trajectory, all_trajectories = calc_control_and_trajectory(
        x, dw, config, goal, ob
    )

    return u, trajectory, all_trajectories


def motion(x, u, dt):
    """
    差速輪運動模型。

    對應投影片：差速輪運動模型：速度如何推算下一個狀態。

    x = [位置x, 位置y, 朝向yaw, 線速度v, 角速度omega]
    u = [線速度v, 角速度omega]
    """

    # 先用角速度更新朝向：theta(t+1) = theta(t) + omega * dt
    x[2] += u[1] * dt

    # 再用新的朝向更新位置：x(t+1), y(t+1)
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt

    # 記錄目前這一步使用的線速度與角速度
    x[3] = u[0]
    x[4] = u[1]

    return x


def calc_dynamic_window(x, config):
    """
    計算動態窗口 Dynamic Window。

    對應投影片：動態窗口：限制目前可行的速度範圍。

    這份程式主要取 Vs 和 Vd 的交集：
    - Vs：機器人本身速度限制
    - Vd：根據目前速度與加速度限制，下一個控制週期內可達的速度範圍

    注意：投影片中的 Va（煞車安全速度集合）在這份程式沒有獨立算成一個集合，
    障礙物安全性放在 calc_obstacle_cost() 裡處理。
    """

    # Vs：由機器人規格決定的速度範圍
    # [最小線速度, 最大線速度, 最小角速度, 最大角速度]
    Vs = [config.min_speed, config.max_speed,
          -config.max_yaw_rate, config.max_yaw_rate]

    # Vd：從目前速度出發，下一個 dt 內受加速度限制後能到達的範圍
    # x[3] 是目前線速度 v，x[4] 是目前角速度 omega
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_delta_yaw_rate * config.dt,
          x[4] + config.max_delta_yaw_rate * config.dt]

    # dw：真正拿來採樣的速度範圍，取 Vs 和 Vd 的交集
    # [v_min, v_max, omega_min, omega_max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    return dw


def predict_trajectory(x_init, v, y, config):
    """
    用一組速度 v、omega 預測未來軌跡。

    對應投影片：多組速度會產生多條候選軌跡。

    原始程式用參數名稱 y 代表 yaw_rate，也就是角速度 omega。
    這裡保留原本名稱，避免大幅更動原始架構。
    """

    # 複製目前狀態，避免直接改到外面的 x
    x = np.array(x_init)

    # trajectory 用來存這組速度模擬出來的整條軌跡
    trajectory = np.array(x)

    # 從現在開始，一路模擬到 predict_time 秒後
    time = 0
    while time <= config.predict_time:
        # 假設這段時間都使用同一組速度 [v, omega]
        x = motion(x, [v, y], config.dt)

        # 把每一個模擬狀態接到 trajectory 後面（存點）
        trajectory = np.vstack((trajectory, x))
        time += config.dt

    return trajectory


def calc_control_and_trajectory(x, dw, config, goal, ob):
    """
    在動態窗口內採樣速度、模擬軌跡、計算 cost，最後選出最佳控制輸入。

    對應投影片：
    - 速度離散化
    - 評價函數
    - DWA 程式流程
    """

    # 保留目前狀態，所有候選軌跡都從同一個狀態開始模擬
    x_init = x[:]

    # 目前找到的最小 cost，先設成無限大
    min_cost = float("inf")

    # best_u = [最佳線速度 v, 最佳角速度 omega]
    best_u = [0.0, 0.0]

    # best_trajectory 存最後被選中的最佳軌跡
    best_trajectory = np.array([x])

    # 教學用：存下所有候選軌跡，方便在畫面中用灰色線顯示
    all_trajectories = []

    # 在動態窗口 dw 內，依照速度解析度離散化採樣
    for v in np.arange(dw[0], dw[1], config.v_resolution):
        for y in np.arange(dw[2], dw[3], config.yaw_rate_resolution):

            # 1. 用這組 v、omega 預測一條候選軌跡
            trajectory = predict_trajectory(x_init, v, y, config)

            # 把候選軌跡存起來，等等 main() 會畫成灰色線
            all_trajectories.append(trajectory)

            # 2. 計算這條軌跡的三種 cost
            # 目標 cost：軌跡最後的朝向是否對準目標
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(trajectory, goal)

            # 速度 cost：越慢 cost 越大，鼓勵機器人保持速度
            speed_cost = config.speed_cost_gain * (config.max_speed - trajectory[-1, 3])

            # 障礙物 cost：越靠近障礙物 cost 越大；碰撞則回傳無限大
            ob_cost = config.obstacle_cost_gain * calc_obstacle_cost(trajectory, ob, config)

            # 3. 總 cost：這份程式採用 cost 最小化
            final_cost = to_goal_cost + speed_cost + ob_cost

            # 4. 如果這條軌跡目前最好，就更新最佳速度與最佳軌跡
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [v, y]
                best_trajectory = trajectory

                # 避免機器人在障礙物前選到 v=0 且 omega=0 後卡住
                if abs(best_u[0]) < config.robot_stuck_flag_cons \
                        and abs(x[3]) < config.robot_stuck_flag_cons:
                    best_u[1] = -config.max_delta_yaw_rate

    return best_u, best_trajectory, all_trajectories


def calc_obstacle_cost(trajectory, ob, config):
    """
    計算障礙物 cost。

    對應投影片：評價函數中的 dist / obstacle cost。

    - 如果軌跡會碰撞障礙物，cost = 無限大，代表這條不能選。
    - 如果沒有碰撞，距離障礙物越近，cost 越大。
    """

    # 障礙物的 x、y 座標
    ox = ob[:, 0]
    oy = ob[:, 1]

    # 計算軌跡上每個點與每個障礙物的 x、y 差值
    dx = trajectory[:, 0] - ox[:, None]
    dy = trajectory[:, 1] - oy[:, None]

    # r 是軌跡點到障礙物的距離矩陣
    r = np.hypot(dx, dy)

    if config.robot_type == RobotType.rectangle:
        # 矩形機器人：把障礙物轉到機器人座標系中，檢查是否落在矩形車體內
        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)],
                        [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])

        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])

        upper_check = local_ob[:, 0] <= config.robot_length / 2
        right_check = local_ob[:, 1] <= config.robot_width / 2
        bottom_check = local_ob[:, 0] >= -config.robot_length / 2
        left_check = local_ob[:, 1] >= -config.robot_width / 2

        # 只要有任一障礙物落在矩形車體內，就視為碰撞
        if (np.logical_and(np.logical_and(upper_check, right_check),
                           np.logical_and(bottom_check, left_check))).any():
            return float("Inf")

    elif config.robot_type == RobotType.circle:
        # 圓形機器人：距離小於半徑就視為碰撞
        if np.array(r <= config.robot_radius).any():
            return float("Inf")

    # 沒有碰撞時，用最近障礙物距離的倒數當 cost
    # 距離越小，1 / 距離越大，代表越危險
    min_r = np.min(r)
    return 1.0 / min_r


def calc_to_goal_cost(trajectory, goal):
    """
    計算朝向目標的 cost。

    對應投影片：評價函數中的 heading。

    這裡不是直接看離目標多遠，而是看軌跡最後的朝向，
    是否對準「軌跡終點到目標點」的方向。
    """

    # 從預測軌跡最後一點指向目標點的向量
    dx = goal[0] - trajectory[-1, 0]
    dy = goal[1] - trajectory[-1, 1]

    # 目標方向角
    error_angle = math.atan2(dy, dx)

    # 目標方向角與機器人最後朝向的差
    cost_angle = error_angle - trajectory[-1, 2]

    # 把角度差限制在 [-pi, pi]，再取絕對值
    cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))

    return cost


def plot_arrow(x, y, yaw, length=0.5, width=0.1):  # pragma: no cover
    """畫出機器人目前朝向。"""
    plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
              head_length=width, head_width=width)
    plt.plot(x, y)


def plot_robot(x, y, yaw, config):  # pragma: no cover
    """依照 robot_type 畫出機器人外型。"""

    if config.robot_type == RobotType.rectangle:
        # 矩形機器人的四個角點，先在機器人自身座標系中定義
        outline = np.array([[-config.robot_length / 2, config.robot_length / 2,
                             config.robot_length / 2, -config.robot_length / 2,
                             -config.robot_length / 2],
                            [config.robot_width / 2, config.robot_width / 2,
                             -config.robot_width / 2, -config.robot_width / 2,
                             config.robot_width / 2]])

        # 根據 yaw 旋轉到世界座標系
        Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                         [-math.sin(yaw), math.cos(yaw)]])
        outline = (outline.T.dot(Rot1)).T

        # 平移到機器人目前位置
        outline[0, :] += x
        outline[1, :] += y

        plt.plot(np.array(outline[0, :]).flatten(),
                 np.array(outline[1, :]).flatten(), "-k")

    elif config.robot_type == RobotType.circle:
        # 圓形機器人用圓形表示
        circle = plt.Circle((x, y), config.robot_radius, color="b", fill=False)
        plt.gcf().gca().add_artist(circle)

        # 畫一條從圓心往外的線表示朝向
        out_x, out_y = (np.array([x, y]) +
                        np.array([np.cos(yaw), np.sin(yaw)]) * config.robot_radius)
        plt.plot([x, out_x], [y, out_y], "-k")


def main(gx=10.0, gy=10.0, robot_type=RobotType.circle):
    """
    主程式：初始化狀態，重複執行 DWA，直到到達目標。

    對應投影片：DWA 程式流程。
    """

    print(__file__ + " start!!")

    # 初始狀態 x = [位置x, 位置y, 朝向yaw, 線速度v, 角速度omega]
    x = np.array([0.0, 0.0, math.pi / 8.0, 0.0, 0.0])

    # 目標位置 [x, y]
    goal = np.array([gx, gy])

    # 設定機器人外型
    config.robot_type = robot_type

    # trajectory 存實際走過的路徑，用於最後畫紅線
    trajectory = np.array(x)

    # 障礙物座標
    ob = config.ob

    while True:
        # 1. DWA 根據目前狀態 x，算出最佳速度 u 與預測軌跡
        u, predicted_trajectory, all_trajectories = dwa_control(x, config, goal, ob)

        # 2. 用最佳速度推進機器人狀態，模擬機器人真的走一步
        x = motion(x, u, config.dt)

        # 3. 記錄機器人實際走過的狀態
        trajectory = np.vstack((trajectory, x))

        if show_animation:
            plt.cla()

            # 按 Esc 可以停止模擬
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            # 灰色很多條：所有候選軌跡
            if show_all_candidate_trajectories:
                for traj in all_trajectories:
                    plt.plot(traj[:, 0], traj[:, 1], color="0.80", linewidth=0.5)

            # 綠色粗線：DWA 最後選中的最佳軌跡
            plt.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1],
                     color="green", linewidth=2.5)

            # 黑色點：障礙物
            plt.plot(ob[:, 0], ob[:, 1], "ok")

            # 藍色 X：目標
            plt.plot(goal[0], goal[1], "xb", markersize=8)

            # 機器人外型與朝向
            plot_robot(x[0], x[1], x[2], config)
            plot_arrow(x[0], x[1], x[2])

            # 紅色 X：機器人目前位置。放最後畫，避免被車體外框蓋住。
            plt.plot(x[0], x[1], "xr", markersize=8)

            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.0001)

        # 判斷是否抵達目標
        dist_to_goal = math.hypot(x[0] - goal[0], x[1] - goal[1])
        if dist_to_goal <= config.robot_radius:
            print("Goal!!")
            break

    print("Done")

    if show_animation:
        # 紅色線：機器人實際走過的完整路徑
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.pause(0.0001)
        plt.show()


if __name__ == '__main__':
    # 預設用矩形機器人，方便看出車體方向。
    main(robot_type=RobotType.rectangle)

    # 若想用圓形機器人，可以改用這行：
    # main(robot_type=RobotType.circle)
