import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Thông số robot
radius = 0.5
R = 3
speed = 0.05
lim = 5

# Tạo figure và trục
fig, ax = plt.subplots(figsize=(6, 6))

# Robot: hình tròn và hướng
robot_circle = plt.Circle((0, 0), radius, fill=False, color='blue', linewidth=2, zorder=3)
heading_line, = ax.plot([], [], color='red', linewidth=2, zorder=4)

# Thêm hình tròn vào trục
ax.add_patch(robot_circle)

# Tạo text gắn với robot (tọa độ Oxy, không dùng transform)
text_coord = ax.text(0, 0, '', fontsize=10, color='black',
                     verticalalignment='bottom', horizontalalignment='left',
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=1),
                     zorder=5)

# Thiết lập trục
ax.set_aspect('equal')
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks(np.arange(-lim, lim + 1, 1))
ax.set_yticks(np.arange(-lim, lim + 1, 1))
ax.axhline(0, color='black', zorder=1)
ax.axvline(0, color='black', zorder=1)
ax.grid(True, zorder=0)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Robot 2D di chuyển – tọa độ đi cùng")

# Khởi tạo
def init():
    heading_line.set_data([], [])
    text_coord.set_text('')
    return robot_circle, heading_line, text_coord

# Cập nhật mỗi frame
def update(frame):
    angle = frame * speed
    x = R * np.cos(angle)
    y = R * np.sin(angle)
    theta = angle + np.pi / 2

    # Cập nhật hình tròn và hướng
    robot_circle.center = (x, y)
    x_end = x + radius * np.cos(theta)
    y_end = y + radius * np.sin(theta)
    heading_line.set_data([x, x_end], [y, y_end])

    # Cập nhật text ngay cạnh robot
    text_coord.set_position((x + 0.6, y + 0.4))  # chỉnh lệch để không đè lên robot
    text_coord.set_text(f'({x:.2f}, {y:.2f})')

    return robot_circle, heading_line, text_coord

FPS = 24

# Animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 1000), init_func=init, interval=1000/FPS, blit=True)

plt.show()
