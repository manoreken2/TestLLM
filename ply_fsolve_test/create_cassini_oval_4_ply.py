from create_surface_ply import create_surface_ply
import numpy as np


# cassini oval 4の方程式
def eq_eval(x, y, z, coeffs):
    a, b, c = coeffs
    return (
        b
        - c * z
        - ((x - a) ** 2 + (y - a) ** 2)
        * ((x + a) ** 2 + (y - a) ** 2)
        * ((x - a) ** 2 + (y + a) ** 2)
        * ((x + a) ** 2 + (y + a) ** 2)
    )


if __name__ == "__main__":

    coeffs = (
        1,
        20,
        20.0,
    )  # 方程式の係数リスト。

    dxy_length = 1.0 / 32  # xy平面移動時ステップ移動量。
    dz_length = 1.0 / 32  # z上方向移動時ステップ移動量。

    stop_k = 0.5  # 2つの点が近いと判定する距離(1周判定用) k * dxy_length
    point_regist_k = 0.7  # 新しい点の登録時、近くに点があったら登録しない dxy_length * point_regist_k
    valid_thr = 0.001  # 新たに見つかった点が面上にあるかどうかを判定する閾値。
    z_stop = 1  # 高さzがz_stopに達したら処理終了。

    a = create_surface_ply(eq_eval, coeffs, stop_k, point_regist_k, valid_thr)

    pinit = np.array([1.08, 1.08, 0])  # 初期位置。面上の点が望ましい。
    dxy_dir = np.array([1, -1, 0])  # 初期位置から次の点に移動するxy平面上の方向。
    dz_dir = np.array([0, 0, 1])  # 初期位置から次の点に移動するz方向。
    a.gather_points(pinit, dxy_dir, dz_dir, dxy_length, dz_length, z_stop)

    pinit = np.array([-1.08, 1.08, 0])  # 初期位置。面上の点が望ましい。
    dxy_dir = np.array([-1, -1, 0])  # 初期位置から次の点に移動するxy平面上の方向。
    dz_dir = np.array([0, 0, 1])  # 初期位置から次の点に移動するz方向。
    a.gather_points(pinit, dxy_dir, dz_dir, dxy_length, dz_length, z_stop)

    pinit = np.array([1.08, -1.08, 0])  # 初期位置。面上の点が望ましい。
    dxy_dir = np.array([1, 1, 0])  # 初期位置から次の点に移動するxy平面上の方向。
    dz_dir = np.array([0, 0, 1])  # 初期位置から次の点に移動するz方向。
    a.gather_points(pinit, dxy_dir, dz_dir, dxy_length, dz_length, z_stop)

    pinit = np.array([-1.08, -1.08, 0])  # 初期位置。面上の点が望ましい。
    dxy_dir = np.array([-1, 1, 0])  # 初期位置から次の点に移動するxy平面上の方向。
    dz_dir = np.array([0, 0, 1])  # 初期位置から次の点に移動するz方向。
    a.gather_points(pinit, dxy_dir, dz_dir, dxy_length, dz_length, z_stop)

    a.write_ply("cassini_oval_4.ply")
