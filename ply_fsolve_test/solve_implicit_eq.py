# pip install scipy

from scipy.optimize import fsolve

def eq1(x, y, r=1):
    return x * x + y * y - r * r

solution = fsolve(eq1, 0.5, args=(0.5) )

print(solution)
