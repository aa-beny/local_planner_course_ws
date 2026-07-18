"""
Force-based Artificial Potential Field (APF) path planner

這份程式用「力向量」方式示範 APF，方便對照講義：
- F_att：目標吸引力
- F_rep：單一障礙物斥力
- F_obs：所有障礙物斥力合成
- F_sum：最後總合力

畫面說明：
- black star：起點
- magenta star：目標點
- black dot：障礙物
- gray dashed circle：障礙物影響範圍 d0
- red line：機器人走過的路徑
- red dot：機器人目前位置
- blue arrow：F_att
- orange arrow：F_obs
- green arrow：F_sum
"""

import matplotlib.pyplot as plt
import numpy as np


# =========================
# APF 參數設定
# =========================

# 是否顯示動畫。
# True：會一邊跑一邊畫圖。
# False：只計算路徑，不顯示動畫。
show_animation = True

# eta：引力增益參數。
# 對應講義：|F_att| = eta × d
# 數值越大，目標點對機器人的拉力越強。
ATTRACTIVE_GAIN_ETA = 10.0

# k：斥力增益參數。
# 對應講義：F_rep = k(1/d - 1/d0) × 1/d^2
# 數值越大，障礙物附近的推力越強。
REPULSIVE_GAIN_K = 200.0

# d0：障礙物影響範圍 / 安全距離。
# 當機器人到障礙物的距離 d > d0 時，該障礙物不產生斥力。
# 當 d <= d0 時，障礙物開始產生斥力。
SAFE_DISTANCE_D0 = 1.0

# STEP_GAIN：把總合力 F_sum 轉成位置位移的比例。
#     move_step = STEP_GAIN * F_sum
#
# 所以：
# - F_sum 的方向會決定機器人往哪裡走。
# - F_sum 的大小會影響這一步走多遠。
# - STEP_GAIN 越大，機器人每一步移動越明顯。
STEP_GAIN = 0.001

# 判斷到達目標的距離門檻。
# 機器人離目標小於這個距離，就視為到達。
GOAL_TOLERANCE = 0.5

# 最大步數，避免參數設錯時程式一直跑下去。
MAX_STEPS = 2000

# 如果機器人離障礙物太近，就視為避障失敗。
MIN_SAFE_DISTANCE_TO_OBSTACLE = 1.0

# 如果距離障礙物太近，公式中的 1/d 會變得非常大。
# 所以把最小距離限制在這個值，避免數值爆掉。
MIN_OBSTACLE_DISTANCE = 0.1

# 畫力箭頭時使用的固定長度。
# 真正計算移動時使用的是原本的 F_sum；這個參數只影響畫面上箭頭長度。
FORCE_ARROW_LENGTH = 1.2

# 動畫更新速度。數值越大，畫面越慢。
PAUSE_TIME = 0.01


# =========================
# 可嘗試的參數組合
# =========================

# 1. 正常成功：通常可以繞過障礙物並到達目標。
# ATTRACTIVE_GAIN_ETA = 10.0
# REPULSIVE_GAIN_K = 200.0
# SAFE_DISTANCE_D0 = 4.0


# 2. 引力過大、斥力太弱：機器人太想往目標走，容易太靠近障礙物。
# ATTRACTIVE_GAIN_ETA = 30.0
# REPULSIVE_GAIN_K = 10.0
# SAFE_DISTANCE_D0 = 4.0


# 3. d0 太小：障礙物影響範圍太小，機器人太晚開始避障。
# ATTRACTIVE_GAIN_ETA = 10.0
# REPULSIVE_GAIN_K = 200.0
# SAFE_DISTANCE_D0 = 1.0


# 4. 斥力太強、d0 太大：機器人容易被障礙物推得太遠，可能繞很大或走不到目標。
# ATTRACTIVE_GAIN_ETA = 1.0
# REPULSIVE_GAIN_K = 500.0
# SAFE_DISTANCE_D0 = 10.0


# 5. 引力太弱：目標拉力不足，機器人可能移動很慢，跑完最大步數仍到不了目標。
# ATTRACTIVE_GAIN_ETA = 1.0
# REPULSIVE_GAIN_K = 200.0
# SAFE_DISTANCE_D0 = 4.0



# =========================
# 基礎工具函式
# =========================

def distance_between(point_a, point_b):
    """
    計算兩個 2D 點之間的直線距離。

    point_a = [x1, y1]
    point_b = [x2, y2]
    距離 = sqrt((x1 - x2)^2 + (y1 - y2)^2)
    """

    return np.linalg.norm(point_a - point_b)


def normalize_vector(vector):
    """
    將向量轉成單位向量。

    單位向量只保留方向，長度會變成 1。
    這裡主要用在：
    1. 計算斥力方向。
    2. 畫箭頭時固定箭頭長度。
    3. 快到目標時避免最後一步走過頭。
    """

    vector_length = np.linalg.norm(vector)

    if vector_length < 1e-9:
        return vector

    return vector / vector_length


# =========================
# APF 力的計算
# =========================

def calc_attractive_force(robot_position, goal_position):
    """
    計算目標點產生的吸引力 F_att。

    講義概念：
        |F_att| = eta × d

    程式向量寫法：
        F_att = eta × (goal_position - robot_position)

    這裡沒有另外把距離 d 算出來，
    因為 goal_position - robot_position 這個向量本身就已經包含：
    1. 方向：從機器人指向目標
    2. 大小：距離目標越遠，向量的數值就越大

    --------------------------------------------------
    範例：
    --------------------------------------------------

    假設：
        robot_position = [0, 10]
        goal_position  = [30, 30]
        eta = 5

    先算：
        vector_to_goal = goal_position - robot_position
                       = [30, 30] - [0, 10]
                       = [30, 20]

    [30, 20] 代表：
        目標在機器人右邊 30
        目標在機器人上方 20

    所以它的方向就是「往右上方，也就是往目標走」。

    接著乘上 eta：
        F_att = eta × vector_to_goal
              = 5 × [30, 20]
              = [150, 100]

    這代表：
        目標給機器人的吸引力方向是往右上方。
        x 方向拉力是 150。
        y 方向拉力是 100。

    如果機器人離目標更遠，例如：
        robot_position = [-10, 0]

    則：
        vector_to_goal = [30, 30] - [-10, 0]
                       = [40, 30]

        F_att = 5 × [40, 30]
              = [200, 150]

    可以看到：
        距離目標越遠，vector_to_goal 的數值越大，
        F_att 也會越大。

    所以這段程式雖然沒有另外算 d，
    但效果仍然符合講義：
        距離目標越遠，吸引力越大。
    """

    # 從機器人位置指向目標點。
    # 例如 [30, 30] - [0, 10] = [30, 20]，
    # 代表目標在機器人的右上方。
    vector_to_goal = goal_position - robot_position

    # 將指向目標的向量乘上 eta，得到吸引力向量 F_att。
    # eta 越大，目標拉力越強。
    F_att = ATTRACTIVE_GAIN_ETA * vector_to_goal

    return F_att


def calc_repulsive_force(robot_position, obstacle_position):
    """
    計算單一障礙物產生的斥力 F_rep。

    講義概念：
        當 d <= d0：
            F_rep = k(1/d - 1/d0) × 1/d^2

        當 d > d0：
            F_rep = 0

    注意：
        講義公式主要是在算「斥力大小」。
        但是程式要讓機器人知道往哪裡被推，
        所以還要補上「遠離障礙物的方向」。

    --------------------------------------------------
    範例 1：障礙物太遠，不產生斥力
    --------------------------------------------------

    假設：
        robot_position    = [0, 10]
        obstacle_position = [15, 25]
        d0 = 4

    先算機器人到障礙物的方向：
        robot_position - obstacle_position
        = [0, 10] - [15, 25]
        = [-15, -15]

    距離：
        d = sqrt((-15)^2 + (-15)^2)
          ≈ 21.21

    因為：
        d = 21.21 > d0 = 4

    所以這個障礙物太遠，不影響機器人：
        F_rep = [0, 0]


    --------------------------------------------------
    範例 2：機器人進入 d0 範圍，開始受到斥力
    --------------------------------------------------

    假設：
        robot_position    = [13, 23]
        obstacle_position = [15, 25]
        d0 = 4
        k = 200

    先算「遠離障礙物」的方向向量：
        robot_position - obstacle_position
        = [13, 23] - [15, 25]
        = [-2, -2]

    這代表：
        從障礙物看機器人，
        機器人在障礙物的左下方，
        所以斥力應該把機器人往左下方推。

    距離：
        d = sqrt((-2)^2 + (-2)^2)
          ≈ 2.83

    因為：
        d = 2.83 <= d0 = 4

    所以障礙物會產生斥力。

    斥力方向：
    除距離變成單位方向向量
    只保留「遠離障礙物」的方向，讓向量長度變成 1。
        [-2, -2] / 2.83
        ≈ [-0.707, -0.707]

    斥力大小：
        k(1/d - 1/d0) × 1/d^2
        = 200 × (1/2.83 - 1/4) × 1/(2.83^2)
        ≈ 2.59

    最後：
        F_rep = 斥力大小 × 斥力方向
              ≈ 2.59 × [-0.707, -0.707]
              ≈ [-1.83, -1.83]

    代表：
        這個障礙物會把機器人往左下方推開。
    """

    # 從障礙物指向機器人的向量。
    # 這個方向就是「遠離障礙物」的方向。
    away_from_obstacle = robot_position - obstacle_position

    # 計算機器人到障礙物的距離 d。
    d = np.linalg.norm(away_from_obstacle)

    # 如果 d > d0，代表障礙物太遠，不會影響機器人。
    # 所以斥力為 [0, 0]。
    if d > SAFE_DISTANCE_D0:
        return np.array([0.0, 0.0])

    # 如果機器人離障礙物太近，1/d 會變得非常大。
    # 這裡限制最小距離，避免程式數值爆掉。
    d = max(d, MIN_OBSTACLE_DISTANCE)

    # 將 away_from_obstacle 轉成單位向量。
    # 這樣只保留方向，長度變成 1。
    repulsive_direction = away_from_obstacle / d

    # 依照講義公式計算斥力大小。
    repulsive_magnitude = (
        REPULSIVE_GAIN_K
        * (1.0 / d - 1.0 / SAFE_DISTANCE_D0)
        * (1.0 / (d ** 2))
    )

    # 斥力向量 = 斥力大小 × 遠離障礙物的方向。
    F_rep = repulsive_magnitude * repulsive_direction

    return F_rep


def calc_total_force(robot_position, goal_position, obstacle_positions):
    """
    計算機器人目前受到的總合力 F_sum。

    對應講義：
        F_sum = F_att + ΣF_rep

    也就是：
        總合力 = 目標吸引力 + 所有障礙物斥力

    --------------------------------------------------
    範例：
    --------------------------------------------------

    假設目前算出來：

        F_att = [150, 100]

    代表目標把機器人往右上方拉。

    假設有兩個障礙物：

        障礙物 1 產生：
        F_rep1 = [-1.83, -1.83]

        障礙物 2 太遠，所以：
        F_rep2 = [0, 0]

    那所有障礙物斥力合成為：

        F_obs = F_rep1 + F_rep2
              = [-1.83, -1.83] + [0, 0]
              = [-1.83, -1.83]

    最後總合力：

        F_sum = F_att + F_obs
              = [150, 100] + [-1.83, -1.83]
              = [148.17, 98.17]

    代表：
        機器人主要還是往目標方向前進，
        但因為受到障礙物斥力影響，
        方向會稍微被推開一點。

    回傳：
        F_att：目標吸引力
        F_obs：所有障礙物斥力合成
        F_sum：最後總合力
    """

    # 1. 計算目標點對機器人的吸引力。
    F_att = calc_attractive_force(robot_position, goal_position)

    # 2. 建立一個零向量，用來累加所有障礙物的斥力。
    F_obs = np.array([0.0, 0.0])

    # 3. 逐一計算每個障礙物的斥力。
    for obstacle_position in obstacle_positions:
        F_rep = calc_repulsive_force(robot_position, obstacle_position)

        # 把每個障礙物的斥力加到 F_obs 裡。
        # 最後 F_obs 就會是所有障礙物斥力的合成結果。
        F_obs += F_rep

    # 4. 總合力 = 目標吸引力 + 障礙物斥力合成。
    F_sum = F_att + F_obs

    return F_att, F_obs, F_sum


# =========================
# APF 主流程
# =========================

def apf_force_planning(sx, sy, gx, gy, ox, oy):
    """
    使用力向量版本的 APF 規劃路徑。

    流程：
    1. 取得目前位置。
    2. 計算 F_att、F_obs、F_sum。
    3. 用 move_step = STEP_GAIN * F_sum 計算下一步位移。
    4. 更新機器人位置。
    5. 重複直到到達目標，或出現失敗情況。
    """

    robot_position = np.array([sx, sy], dtype=float)
    goal_position = np.array([gx, gy], dtype=float)
    obstacle_positions = np.column_stack((ox, oy)).astype(float)

    rx = [robot_position[0]]
    ry = [robot_position[1]]

    for step_count in range(MAX_STEPS):
        # 目前位置到目標的距離。
        distance_to_goal = distance_between(robot_position, goal_position)

        # 到達目標。
        if distance_to_goal <= GOAL_TOLERANCE:
            print("成功：機器人到達目標。")
            break

        # 目前位置到最近障礙物的距離。
        distances_to_obstacles = [
            distance_between(robot_position, obstacle)
            for obstacle in obstacle_positions
        ]
        nearest_obstacle_distance = min(distances_to_obstacles)

        # 太靠近障礙物，視為避障失敗。
        if nearest_obstacle_distance <= MIN_SAFE_DISTANCE_TO_OBSTACLE:
            print("失敗：機器人太靠近障礙物，視為避障失敗。")
            print("可能原因：引力太強、斥力太弱，或 d0 太小導致太晚避障。")
            break

        # 計算三種力。
        F_att, F_obs, F_sum = calc_total_force(
            robot_position,
            goal_position,
            obstacle_positions,
        )

        # 總合力太小，代表方向不明顯，可能卡在局部最小值。
        if np.linalg.norm(F_sum) < 1e-9:
            print("失敗：F_sum 幾乎為 0，機器人沒有明確方向。")
            print("可能原因：目標吸引力與障礙物斥力互相抵消，這是 APF 常見的局部最小值問題。")
            break

        # 用總合力更新位置。
        # F_sum 的方向決定走向，F_sum 的大小影響這一步走多遠。
        move_step = STEP_GAIN * F_sum

        # 如果最後一步會超過目標，就只走到目標附近，避免走過頭。
        if np.linalg.norm(move_step) > distance_to_goal:
            move_step = distance_to_goal * normalize_vector(move_step)

        # 最後再加座標
        robot_position = robot_position + move_step

        rx.append(robot_position[0])
        ry.append(robot_position[1])

        if show_animation:
            draw_simulation(
                robot_position,
                goal_position,
                obstacle_positions,
                rx,
                ry,
                F_att,
                F_obs,
                F_sum,
            )

    else:
        print("失敗：超過最大步數仍未到達目標。")
        print("可能原因：引力太弱、斥力太強，或參數組合讓機器人前進效率太差。")

    return rx, ry


# =========================
# 畫圖相關函式
# =========================

def draw_force_arrow(start_position, force_vector, color, label):
    """
    畫出力的方向箭頭。

    畫圖時固定箭頭長度，只強調方向。
    真正移動時，程式使用的是原本的 F_sum 大小。
    """

    if np.linalg.norm(force_vector) < 1e-9:
        return

    direction = normalize_vector(force_vector)

    x = start_position[0]
    y = start_position[1]
    dx = direction[0] * FORCE_ARROW_LENGTH
    dy = direction[1] * FORCE_ARROW_LENGTH

    plt.arrow(
        x,
        y,
        dx,
        dy,
        color=color,
        width=0.04,
        head_width=0.3,
        length_includes_head=True,
    )

    plt.text(x + dx, y + dy, label, color=color, fontsize=10)


def draw_simulation(robot_position, goal_position, obstacle_positions,
                    rx, ry, F_att, F_obs, F_sum):
    """
    畫出目前 APF 規劃狀態。
    """

    plt.cla()

    # 按 Esc 可以中止程式。
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None]
    )

    # 畫障礙物影響範圍 d0。
    for obstacle in obstacle_positions:
        safe_circle = plt.Circle(
            obstacle,
            SAFE_DISTANCE_D0,
            color="gray",
            fill=False,
            linestyle="--",
            linewidth=0.8,
        )
        plt.gca().add_artist(safe_circle)

    # 畫障礙物、目標、起點、路徑、機器人。
    plt.plot(obstacle_positions[:, 0], obstacle_positions[:, 1], "ok", label="obstacle")
    plt.plot(goal_position[0], goal_position[1], "*m", markersize=12, label="goal")
    plt.plot(rx[0], ry[0], "*k", markersize=12, label="start")
    plt.plot(rx, ry, "-r", linewidth=2, label="path")
    plt.plot(robot_position[0], robot_position[1], "or", markersize=6, label="robot")

    # 畫力箭頭。
    draw_force_arrow(robot_position, F_att, "blue", "F_att")
    draw_force_arrow(robot_position, F_obs, "orange", "F_obs")
    draw_force_arrow(robot_position, F_sum, "green", "F_sum")

    # 不設定 title，避免中文顯示成亂碼，也避免畫面太亂。
    plt.axis("equal")
    plt.grid(True)
    plt.legend(loc="upper left")
    plt.pause(PAUSE_TIME)


# =========================
# 主程式
# =========================

def main():
    """
    設定起點、目標點與障礙物，並執行 APF。
    """

    print(__file__ + " start!!")

    # 起點座標。
    sx = 0.0
    sy = 10.0

    # 目標座標。
    gx = 30.0
    gy = 30.0

    # 障礙物座標。
    # ox 和 oy 是一一對應的：
    # 第 1 個障礙物在 (ox[0], oy[0])，第 2 個障礙物在 (ox[1], oy[1])，以此類推。
    ox = [15.0, 5.0, 20.0, 25.0]
    oy = [25.0, 15.0, 26.0, 25.0]

    rx, ry = apf_force_planning(sx, sy, gx, gy, ox, oy)

    if show_animation:
        plt.plot(rx, ry, "-r", linewidth=2)
        plt.show()


if __name__ == '__main__':
    main()