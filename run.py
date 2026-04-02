import subprocess
import h5py
import numpy as np
import os
import shutil
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import math

# ===== ① 設定 =====
input_file = "my_model/test.in"
base_name = "my_model/test"
n_steps = 541
er = 6.2           # 比誘電率
t_max = 6e-9       # タイムウィンドウ
start_x = 0.05     # 測定開始位置 (m)
step_size = 0.0025 # 掃引ピッチ (m)

# ===== ② 実行ごとのフォルダ作成 =====
timestamp = datetime.now().strftime("%m%d_%H%M")
run_dir = f"run_{timestamp}"

vti_dir = os.path.join(run_dir, "vti_files")
out_dir = os.path.join(run_dir, "out_files")

os.makedirs(vti_dir, exist_ok=True)
os.makedirs(out_dir, exist_ok=True)

# ===== ③ シミュレーション実行 (GPU使用) =====
# -gpu オプションを追加して高速化
print("🚀 シミュレーション開始 (GPU使用)...")
subprocess.run(f"python -m gprMax {input_file} -n {n_steps} -gpu", shell=True)

# ===== ④ データの結合と抽出 =====
print("🔗 データを結合中...")
subprocess.run(f"python -m tools.outputfiles_merge {base_name}", shell=True)

merged_file = base_name + "_merged.out"
with h5py.File(merged_file, "r") as f:
    ez = f["/rxs/rx1/Ez"][:]

# 中間データ保存
np.savetxt("1.dat", ez)

# gnuplot用スクリプト実行 (必要であれば)
subprocess.run("python read_fdtdout_to_gnuplot.py 1", shell=True)

# ===== ⑤ 画像生成 (B-scan) =====
print("📊 画像を生成中...")
data = np.loadtxt("1.dat")
mean_trace = np.mean(data, axis=1, keepdims=True)
data_processed = data - mean_trace

v_limit = np.max(np.abs(data_processed)) * 0.2

# 深度計算
c = 3e8
v = c / math.sqrt(er)
max_depth = (v * t_max) / 2.0 

n_traces = data.shape[1]
end_x = start_x + step_size * (n_traces - 1)

plt.figure(figsize=(10, 6))
plt.imshow(data_processed, aspect='auto', cmap='gray',
           vmin=-v_limit, vmax=v_limit,
           extent=[start_x, end_x, max_depth, 0])

plt.xlabel("Position X (m)")
plt.ylabel("Depth (m)")
plt.title(f"GPR B-scan: {run_dir}")
plt.colorbar(label="Ez")

output_img = "result_distance.png"
plt.savefig(output_img, dpi=300)
plt.close()

# ===== ⑥ ファイル整理 (ここが重要！) =====
print("📁 ファイルを整理中...")

# my_model フォルダの中にある .out と .vti を探して移動
model_dir = os.path.dirname(input_file)

# .out ファイルの移動
for file_path in glob.glob(os.path.join(model_dir, "*.out")):
    file_name = os.path.basename(file_path)
    shutil.move(file_path, os.path.join(out_dir, file_name))

# .vti ファイルの移動
for file_path in glob.glob(os.path.join(model_dir, "*.vti")):
    file_name = os.path.basename(file_path)
    shutil.move(file_path, os.path.join(vti_dir, file_name))

# 成果物 (dat, png) の移動
for file in ["1.dat", output_img]:
    if os.path.exists(file):
        shutil.move(file, os.path.join(run_dir, file))

# 設定ファイル (test.in) のコピー保存
if os.path.exists(input_file):
    shutil.copy(input_file, os.path.join(run_dir, f"copied_{os.path.basename(input_file)}"))

print(f"✅ すべての完了！ 結果は {run_dir} に保存されました。")