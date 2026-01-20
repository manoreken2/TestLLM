# pip install scipy

from scipy.optimize import fsolve
import numpy as np
from enum import Enum

class Axis(Enum):
    X = 0
    Y = 1
    Z = 2

class gather_points:
    def __init__(self):
        pass

    def eq_x(self, x, y, z): # 既知(固定)のy値、z値、coeffs値を与え、xの初期値を与え、方程式eqを満たすx値を求める。
        return self.eq_eval(x, y, z, self.coeffs)
    def eq_y(self, y, x, z): # 既知(固定)のx値、z値、coeffs値を与え、yの初期値を与え、方程式eqを満たすy値を求める。
        return self.eq_eval(x, y, z, self.coeffs)
    def eq_z(self, z, x, y): # 既知(固定)のx値、y値、coeffs値を与え、zの初期値を与え、方程式eqを満たすz値を求める。
        return self.eq_eval(x, y, z, self.coeffs)

    def solve_eq(self, axis, p_a):
        # 開始地点p_aからaxisの軸の方向に移動して球の表面上の点の座標pを求める。
        x, y, z = p_a

        match axis:
            case Axis.X:
                s = fsolve(self.eq_x, x, args = (y, z) )
                p = np.array([ s[0], y, z ])
            case Axis.Y:
                s = fsolve(self.eq_y, y, args = (x, z) )
                p = np.array([ x, s[0], z ])
            case Axis.Z:
                s = fsolve(self.eq_z, z, args = (x, y) )
                p = np.array([ x, y, s[0] ])
        return p

    def get_best_axis(self, d):
        # 移動ベクトルの成分x,yを見て、値が小さい軸を戻す。
        if abs(d[0]) < abs(d[1]):
            return Axis.X
        return Axis.Y

    def points_too_close(self, p0, p1):
        return np.linalg.norm( p0 - p1 ) < self.stop_thr
    
    def validate_and_store(self, p_candidate):
        x, y, z = p_candidate
        v = self.eq_eval(x, y, z, self.coeffs)
        if self.valid_thr < v:
            return False
        self.p.append(p_candidate)
        return True

    def loop_xy(self, p_init, d_init):
        # 最初の点を確定します。      
        p_0 = self.solve_eq(Axis.X, p_init)
        p_0 = self.solve_eq(Axis.Y, p_0)

        # 次の点p1は、p0からd_initの方向に、長さd_lengthだけ移動した付近にある。
        p_a = p_0 + self.d_length * ( d_init / np.linalg.norm( d_init ) )
        
        p_1 = self.solve_eq(self.get_best_axis(d_init), p_a)

        if self.points_too_close(p_0, p_1):
            print(f"There is no point found near {p_init}")
            return False

        if not self.validate_and_store( p_0 ):
            print(f"Point found is not on the surface. {p_0}")
            return False

        if not self.validate_and_store( p_1 ):
            print(f"Point found is not on the surface. {p_1}")
            return False

        p_prev = p_0
        p_cur  = p_1

        while True:
            # 次の点p_nextは、p_curからp_cur - p_prevの方向に、d_lengthだけ移動したあたりにある。
            d_xyz = p_cur - p_prev
            p_a = p_cur + self.d_length * ( d_xyz / np.linalg.norm( d_xyz ) )
            p_next = self.solve_eq(self.get_best_axis(d_xyz), p_a)

            if self.points_too_close( p_next, p_cur ):
                print(f"Advancement stopped. {p_next} {p_cur}")
                return True
            if self.points_too_close( p_next, p_0 ):
                print(f"Loop formed. {p_next} {p_0}")
                return True
            if not self.validate_and_store( p_next ):
                print(f"Point found is not on the surface. {p_next}")
                return False

            print(f"{len(self.p)}: P_next = ({p_next})")

            p_prev = p_cur
            p_cur = p_next


    def run(self, eq_eval, coeffs, p_init, d1_dir, d2_dir, d_length, k, valid_thr):
        self.eq_eval = eq_eval
        self.coeffs = coeffs
        self.d_length = d_length
        self.k = k
        self.valid_thr = valid_thr

        self.p = []
        self.stop_thr = k * d_length
        
        while self.loop_xy(p_init, d1_dir):
            # d2_dir(典型的にはz上方向)に移動。
            p_init = p_init + d_length * ( d2_dir / np.linalg.norm( d2_dir ) )

        print("Algorithm finished.")

# 球の方程式 x^2 + y^2 + z^2 - r^2 = 0
def eq_eval(x, y, z, coeffs):
    r = coeffs
    return x * x + y * y + z * z - r * r

if __name__ == '__main__':
    p_init = np.array( [ 0, 1, 0 ] ) # 初期位置。
    d1_dir = np.array( [ 1, 0, 0 ] ) # 初期位置から次の点を探すときに移動するxy平面上の方向。
    d2_dir = np.array( [ 0, 0, 1 ] ) # 初期位置から次の点を探すときに移動する、面上でd1と直行する方向。

    coeffs = 1        # 方程式の係数リスト。1個の場合は、スカラー値を入れる。
    d_length = 0.03   # ステップ移動量。
    k = 0.8           # 2つの点が近いと判定する距離 k * d_length 
    valid_thr = 0.001 # 新たに見つかった点が面上にあるかどうかを判定する閾値。

    gp = gather_points()
    gp.run(eq_eval, coeffs, p_init, d1_dir, d2_dir, d_length, k, valid_thr)

