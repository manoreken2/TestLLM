# pip install scipy

from scipy.optimize import fsolve
import numpy as np

# 球の方程式 x^2 + y^2 + z^2 - r^2 = 0
def eq_1(x, y, z, r):
    return x * x + y * y + z * z - r * r

def solve_eq(x_init, args):
    # 既知(固定)のy値、z値、r値を与え、xの初期値を与え、方程式eqを満たすx値を求める。
    solution = fsolve(eq_1, x_init, args = args)
    return solution[0]

def find_point_moveX(p_a, r):
    x, y, z = p_a

    # xを動かして球の表面上の点の座標pを求める。
    x = solve_eq(x, (y, z, r))

    # 球の表面上の点の座標 p
    p = np.array([x, y, z])
    return p

def run():
    # 球の半径。
    r = 1

    # 最初の位置P_a。(球の表面上の点ではない。)
    p_a = np.array([0.5, 0.5, 0.0])

    # 初期移動ベクトル
    d_xyz = np.array([-1, 1, 0])

    # ステップ移動量
    d_length = 0.01

    # k: 1周回ったことを判定するための係数。
    k = 0.8

    p0 = find_point_moveX(p_a, r)
    print(f"P_0 = ({p0})")

    # 次の点p1は、d_xyzの方向に、d_lengthだけ移動したあたりにある。
    p_a = p0 + d_length * ( d_xyz / np.linalg.norm( d_xyz ) )
    p1 = find_point_moveX(p_a, r)
    print(f"P_1 = ({p1})")

    # 次の点p2は、p1-p0の方向に、d_lengthだけ移動したあたりにある。
    p_a = p1 + d_length * ( ( p1 - p0 ) / np.linalg.norm( p1 - p0 ) )
    p2 = find_point_moveX(p_a, r)
    print(f"P_2 = ({p2})")

if __name__ == '__main__':
    run()

