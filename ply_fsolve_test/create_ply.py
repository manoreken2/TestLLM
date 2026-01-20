# pip install scipy

from scipy.optimize import fsolve

# x^2 + y^2 - r^2 = 0
def circle_surface(x, y, r):
    return x * x + y * y - r * r

# find x value satisfy circle equation: x^2 + 0.5^2 = 1 , starts from initial pos (0.5, 0.5)
x_init = 0.5
y = 0.5
r = 1.0
solution = fsolve(circle_surface, x_init, args=(y, r) )

x = solution[0]

print(f"Solution (x, y) = ({x}, {y})")
