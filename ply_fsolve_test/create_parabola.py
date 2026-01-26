# pip install scipy
from create_surface_ply import create_surface_ply
import numpy as np


# パラボラの方程式: x^2 + y^2 = h - z
def eq_eval(x, y, z, coeffs):
    h = coeffs

    return x**2 + y**2 + z - h


if __name__ == "__main__":
    r = 1.0  # 球の半径。

    coeffs = r  # 方程式の係数リスト。

    dxy_length = 1.0 / 32  # xy平面移動時ステップ移動量。
    dz_length = 1.0 / 32  # z上方向移動時ステップ移動量。

    stop_k = 0.5  # 2つの点が近いと判定する距離(1周判定用) k * d_length
    point_regist_k = 0.7  # 新しい点の登録時、近くに点があったら登録しない dxy_length * point_regist_k
    valid_thr = 0.001  # 新たに見つかった点が面上にあるかどうかを判定する閾値。
    z_stop = 1.99

    a = create_surface_ply(eq_eval, coeffs, stop_k, point_regist_k, valid_thr)

    pinit = np.array([1.41, 0, 0])  # 初期位置。面上の点。
    dxy_dir = np.array(
        [0, 1, 0]
    )  # 初期位置から次の点に移動するxy平面上の方向ベクトル。
    dz_dir = np.array([-0.1, 0, 0.9])  # 初期位置から次の点に移動するz方向。
    a.gather_points(pinit, dxy_dir, dz_dir, dxy_length, dz_length, z_stop)

    a.write_ply("parabola32.ply")
