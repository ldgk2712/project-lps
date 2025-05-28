import numpy as np
import matplotlib.pyplot as plt

# Tạo lưới tọa độ
x1 = np.linspace(-8, 2, 400)
x2 = np.linspace(-5, 2, 400)
X1, X2 = np.meshgrid(x1, x2)

# Các ràng buộc
ineq1 = (-X1 - 2*X2 <= 6)   # Bất đẳng thức (1)
ineq2 = (X1 - 2*X2 <= 4)    # Bất đẳng thức (2)
ineq3 = (-X1 + X2 <= 1)     # Bất đẳng thức (3)
ineq4 = (X1 <= 0)           # x1 <= 0
ineq5 = (X2 <= 0)           # x2 <= 0

# Miền thỏa tất cả điều kiện
feasible_region = ineq1 & ineq2 & ineq3 & ineq4 & ineq5

# Vẽ miền nghiệm
plt.figure(figsize=(8, 8))
plt.contourf(X1, X2, feasible_region, levels=[0.5, 1], colors=['#cccccc'], alpha=0.8)

# Vẽ các đường biên (boundary)
x = np.linspace(-8, 2, 400)
plt.plot(x, (-x - 6)/2, label='(1): $-x_1 - 2x_2 \\leq 6$')   # Biên (1)
plt.plot(x, (x - 4)/2, label='(2): $x_1 - 2x_2 \\leq 4$')     # Biên (2)
plt.plot(x, (-x - 1)/(-1), label='(3): $-x_1 + x_2 \\leq 1$')         # Biên (3)
plt.axvline(0, color='k', label='(4): $x_1 \\leq 0$')  # Biên (4)
plt.axhline(0, color='k', label='(5): $x_2 \\leq 0$')   # Biên (5)

# Giao điểm vùng nghiệm
plt.xlim(-7, 2)
plt.ylim(-4, 2)
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.grid(True)
plt.legend()
plt.title("Miền thỏa của hệ bất phương trình")
# Mũi tên chỉ hướng miền nghiệm của từng bất đẳng thức
# Tính điểm giữa mỗi đường để vẽ mũi tên pháp tuyến

def draw_constraint_arrow(func, a, b, c, x_range, label, color='blue', scale=3, eps=0.1):
    """
    Vẽ mũi tên pháp tuyến theo hướng MIỀN THỎA của bất phương trình ax1 + bx2 <= c

    func: hàm tính x2 = f(x1) (biểu diễn đường biên)
    a, b, c: hệ số trong bất phương trình ax1 + bx2 <= c
    x_range: khoảng x1 để chọn điểm vẽ
    """
    x_arrow = np.mean(x_range)
    y_arrow = func(x_arrow)

    # vector pháp tuyến (a, b)
    norm = np.sqrt(a**2 + b**2)
    dx, dy = a / norm, b / norm

    # kiểm tra điểm test gần đường theo hướng pháp tuyến
    x_test = x_arrow + eps * dx
    y_test = y_arrow + eps * dy
    lhs = a * x_test + b * y_test

    if lhs > c:  # nếu không thỏa, đảo hướng
        dx, dy = -dx, -dy

    plt.quiver(x_arrow, y_arrow, dx, dy, angles='xy', scale_units='xy', scale=scale, color=color)
    plt.text(x_arrow + 0.2, y_arrow + 0.2, label, color=color)


# (1) -x1 - 2x2 <= 6 → x2 = (-x1 - 6)/2
draw_constraint_arrow(lambda x: (-x - 6) / 2, a=-1, b=-2, c=6, x_range=[-6, -2], label='(1)')

# (2) x1 - 2x2 <= 4 → x2 = (x - 4)/2
draw_constraint_arrow(lambda x: (x - 4) / 2, a=1, b=-2, c=4, x_range=[-3, 1], label='(2)')

# (3) -x1 + x2 <= 1 → x2 = x + 1
draw_constraint_arrow(lambda x: x + 1, a=-1, b=1, c=1, x_range=[-4, 0], label='(3)')

# (4) x1 <= 0
draw_constraint_arrow(lambda x: 0 * x, a=1, b=0, c=0, x_range=[0, 0], label='(4)')

# (5) x2 <= 0
draw_constraint_arrow(lambda x: 0, a=0, b=1, c=0, x_range=[-4, -4], label='(5)')

plt.show()
