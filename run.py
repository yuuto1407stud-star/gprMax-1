import subprocess
import h5py
import numpy as np

# ===== 設定 =====
input_file = "my_model/test.in"
base_name = "my_model/test"
n_steps = 541   # 必要に応じて変更

# ===== ① シミュレーション =====

subprocess.run(f"python -m gprMax {input_file} -n {n_steps} -gpu", shell=True, check=True)
subprocess.run(f"python -m tools.outputfiles_merge {base_name}", shell=True, check=True)

# ===== ③ HDF5 → Ez抽出 =====
with h5py.File(base_name + "_merged.out", "r") as f:
    ez = f["/rxs/rx1/Ez"][:]

# ===== ④ dat保存 =====
np.savetxt("1.dat", ez)

# ===== ⑤ gnuplot変換 =====
subprocess.run("python read_fdtdout_to_gnuplot.py 1", shell=True)

print("✅ 完全自動で画像生成まで完了")


import matplotlib.pyplot as plt
import numpy as np
import math

# 1. データの読み込みとバックグラウンド除去
data = np.loadtxt("1.dat")
mean_trace = np.mean(data, axis=1, keepdims=True)
data_processed = data - mean_trace

# コントラスト調整（薄ければ0.1、濃すぎれば0.3などに調整してください）
v_limit = np.max(np.abs(data_processed)) * 0.2

# ==========================================
# 2. 縦軸（深さ）の計算
# ==========================================
er = 6.2
c = 3e8  # 光速
v = c / math.sqrt(er)
t_max = 6e-9  # 観測時間
max_depth = (v * t_max) / 2.0 

# ==========================================
# 3. 横軸（位置・距離）の計算
# ==========================================
start_x = 0.05      # アンテナの測定開始位置 (m)
step_size = 0.0025   # 1ステップあたりの移動距離 (m)
n_traces = data.shape[1] # 測定回数（Trace数）

# 終了位置の計算
end_x = start_x + step_size * (n_traces - 1)

# ==========================================
# 4. 描画 (extentで縦横のスケールを実際の距離にマッピング)
# ==========================================
# extent = [左端のX座標, 右端のX座標, 一番下の深さ, 一番上の深さ]
plt.imshow(data_processed, aspect='auto', cmap='gray', 
           vmin=-v_limit, vmax=v_limit,
           extent=[start_x, end_x, max_depth, 0])

# 軸ラベルを両方とも距離(m)に変更
plt.xlabel("Position X (m)")
plt.ylabel("Depth (m)")
plt.colorbar(label="Ez")

plt.savefig("result_distance.png", dpi=300)
plt.show()