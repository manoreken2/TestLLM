# pip install scipy

from scipy.optimize import fsolve
import numpy as np
from enum import Enum


def write_ply(filename, plist):
    with open(filename, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(plist)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in plist:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")


class Axis(Enum):
    X = 0
    Y = 1
    Z = 2


def get_longest_axis(v):
    dx = abs(v[0])
    dy = abs(v[1])
    dz = abs(v[2])
    if dx < dy:
        if dy < dz:
            return Axis.Z
        else:
            return Axis.Y
    elif dx < dz:
        return Axis.Z
    else:
        return Axis.X


def get_shortest_axis(d):
    x = abs(d[0])
    y = abs(d[1])
    z = abs(d[2])
    if x > y:
        if y > z:
            return Axis.Z
        else:
            return Axis.Y
    elif x > z:
        return Axis.Z
    else:
        return Axis.X


def shortest_axis_xy(d):
    x = abs(d[0])
    y = abs(d[1])
    if y < x:
        return Axis.Y
    else:
        return Axis.X


def another_axis_xy(axis):
    if axis == Axis.Z:
        raise ValueError("axis")

    if axis == Axis.X:
        return Axis.Y
    else:
        return Axis.X


def longest_axis_xy(d):
    x = abs(d[0])
    y = abs(d[1])
    if y < x:
        return Axis.X
    else:
        return Axis.Y


class create_surface_ply:
    def __init__(self, eq_eval, coeffs, stop_k, point_regist_k, valid_thr):
        self.eq_eval = eq_eval
        self.coeffs = coeffs
        self.stop_k = stop_k
        self.point_regist_k = point_regist_k
        self.valid_thr = valid_thr

        self.plist = []

    def eq_x(
        self, x, y, z
    ):  # 既知(固定)のy値、z値、coeffs値を与え、xの初期値を与え、方程式eqを満たすx値を求める。
        return self.eq_eval(x, y, z, self.coeffs)

    def eq_y(
        self, y, x, z
    ):  # 既知(固定)のx値、z値、coeffs値を与え、yの初期値を与え、方程式eqを満たすy値を求める。
        return self.eq_eval(x, y, z, self.coeffs)

    def eq_z(
        self, z, x, y
    ):  # 既知(固定)のx値、y値、coeffs値を与え、zの初期値を与え、方程式eqを満たすz値を求める。
        return self.eq_eval(x, y, z, self.coeffs)

    def solve_eq(self, axis, pa):
        # 開始地点paからaxisの軸の方向に移動して球の表面上の点の座標pを求める。
        x, y, z = pa

        match axis:
            case Axis.X:
                s = fsolve(self.eq_x, x, args=(y, z))
                p = np.array([s[0], y, z])
            case Axis.Y:
                s = fsolve(self.eq_y, y, args=(x, z))
                p = np.array([x, s[0], z])
            case Axis.Z:
                s = fsolve(self.eq_z, z, args=(x, y))
                p = np.array([x, y, s[0]])
        return p

    def points_too_close(self, p0, p1, thr):
        return np.linalg.norm(p0 - p1) < thr

    def point_on_the_surface(self, p):
        x, y, z = p
        v = self.eq_eval(x, y, z, self.coeffs)
        if self.valid_thr < v:
            return False
        return True

    def validate_and_store(self, pcandidate):
        if not self.point_on_the_surface(pcandidate):
            return False
        self.append_new_point(pcandidate)
        return True

    # 近くの表面上の点を探します。
    def find_surface_point_xy(self, p):
        p = self.solve_eq(Axis.X, p)
        p = self.solve_eq(Axis.Y, p)
        return p

    def find_surface_point_yx(self, p):
        p = self.solve_eq(Axis.Y, p)
        p = self.solve_eq(Axis.X, p)
        return p

    def find_surface_point_zx(self, p):
        p = self.solve_eq(Axis.Z, p)
        p = self.solve_eq(Axis.X, p)
        return p

    def find_surface_point_zy(self, p):
        p = self.solve_eq(Axis.Z, p)
        p = self.solve_eq(Axis.Y, p)
        return p

    def find_surface_point(self, p, axis1, axis2):
        match axis1:
            case Axis.X:
                return self.find_surface_point_xy(p)
            case Axis.Y:
                return self.find_surface_point_yx(p)
            case Axis.Z:
                match axis2:
                    case Axis.X:
                        return self.find_surface_point_zx(p)
                    case Axis.Y:
                        return self.find_surface_point_zy(p)

    # 既に存在するどの点とも離れている場合のみpnewを追加。
    def append_new_point(self, pnew):
        for p in self.plist:
            if self.points_too_close(p, pnew, self.point_regist_thr):
                return
        self.plist.append(pnew)

    def determine_surface_point_xy(self, p_init, d_init):
        ax1 = shortest_axis_xy(d_init)
        ax2 = another_axis_xy(ax1)
        p0 = self.find_surface_point(p_init, ax1, ax2)
        return p0

    def next_surface_point(self, p, d, d_length, adj_axis):
        ratio = 1.0
        while 0.2 < ratio:
            pa = p + d_length * ratio * (d / np.linalg.norm(d))
            p1 = self.solve_eq(adj_axis, pa)
            if self.point_on_the_surface(p1):
                return p1
            # 半分の移動量にしてリトライする
            ratio = ratio / 2.0
        return p1

    def next_surface_point_dxy(self, p, d, d_length):
        adj_axis = shortest_axis_xy(d)
        return self.next_surface_point(p, d, d_length, adj_axis)

    def next_surface_point_dz(self, p, d, d_length):
        adj_axis = longest_axis_xy(d)
        return self.next_surface_point(p, d, d_length, adj_axis)

    # xy方向に移動していって、1周するまで点を追加。
    def gather_points_xy(self, p_init, d_init, dxy_length, z_stop):
        if z_stop <= p_init[2]:
            print(f"Z reached z_stop.")
            return False

        # 最初の点p0を確定します。(zを動かさず、xy値を微調整し曲面上の点p0を得る)
        p0 = self.determine_surface_point_xy(p_init, d_init)
        if not self.validate_and_store(p0):
            print(f"Point found is not on the surface. {p0}")
            return False

        # 次の点p1は、p0からd_initの方向に移動して見つける。
        p1 = self.next_surface_point_dxy(p0, d_init, dxy_length)
        if not self.validate_and_store(p1):
            print(f"Point found is not on the surface. p1={p1}")
            return False

        if self.points_too_close(p0, p1, self.valid_thr):
            print(f"There is no adjacent point found near {p_init}")
            return False

        print(f"gather_points_xy p0={p0}, p1={p1} d_xy={p1-p0}")

        pprev = p0
        pcur = p1

        while True:
            # 次の点pnextは、pcurからpcur - pprevの方向に、d_lengthだけ移動したあたりにある。
            d_xy = pcur - pprev
            pnext = self.next_surface_point_dxy(pcur, d_xy, dxy_length)
            if not self.validate_and_store(pnext):
                print(f"Point found is not on the surface. pnext={pnext}")
                return False

            if self.points_too_close(pnext, pcur, self.stopthr):
                print(f"Advancement stopped. {pnext} {pcur}")
                return True

            if self.points_too_close(pnext, p0, self.stopthr):
                print(f"Loop formed. {pnext} {p0}")
                return True

            # print(f"{len(self.plist)}: pnext={pnext} d_xy={d_xy}")

            pprev = pcur
            pcur = pnext

    # dxy_dirは、pinit地点から次の点を探すときに移動するx方向。面に沿っているのが良い。
    # dz_dirは、pinit地点から次の点を探すときに移動するz方向。面に沿っているのが良い。
    def gather_points(self, pinit, dxy_dir, dz_dir, dxy_length, dz_length, z_stop):

        self.stopthr = self.stop_k * dxy_length
        self.point_regist_thr = self.point_regist_k * dxy_length

        p0 = self.determine_surface_point_xy(pinit, dxy_dir)
        p1z = self.next_surface_point_dz(p0, dz_dir, dz_length)

        dz = p1z - p0

        while self.gather_points_xy(p0, dxy_dir, dxy_length, z_stop):
            # dz方向に移動後、dzの最長軸方向に微調整する。
            p1z = self.next_surface_point_dz(p0, dz, dz_length)
            print(f"Move z direction: p0={p0} dz={dz} p1z={p1z}")

            if self.points_too_close(p0, p1z, self.stopthr):
                print(
                    f"Move z direction failed. dz={dz} p0={p0} p1z={p1z} stopthr={self.stopthr}"
                )
                break
            dz = p1z - p0
            p0 = p1z
        return True

    def write_ply(self, filename):
        write_ply(filename, self.plist)
        print(f"write_ply {filename}")
