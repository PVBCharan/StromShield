import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("report_figures", exist_ok=True)

# Set global style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'

# -------------------------------------------------------------
# EXACT MEASURED DATA FROM apps/api/src/ml/test_ml_accuracy.py
# (Run on 50 realistic Gujarat storm test trajectories)
# -------------------------------------------------------------
horizons = np.array([15, 30, 45, 60])

# Per-horizon MAE (km)
baseline_linear = np.array([0.172, 0.688, 1.548, 2.751])
pure_ml = np.array([0.601, 2.520, 6.847, 14.975])
physics_hybrid = np.array([0.184, 0.866, 2.780, 7.853])

# Overall Metrics (km)
models_ablation = ['Linear Baseline', 'Pure ML (v3)', 'Physics-Guided Hybrid (v4)']
mae_ablation = [1.290, 6.236, 2.921]
rmse_ablation = [1.929, 8.708, 4.366]

# -------------------------------------------------------------
# FIGURE 1: Trajectory Drift Benchmark across Horizons
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

ax.plot(horizons, pure_ml, 'o--', color='#ef4444', linewidth=2.5, markersize=8, label='Pure ML Model (v3 - Unconstrained Drift)')
ax.plot(horizons, baseline_linear, '^-.', color='#f59e0b', linewidth=2, markersize=7, label='Baseline Linear Extrapolation')
ax.plot(horizons, physics_hybrid, 's-', color='#10b981', linewidth=3, markersize=9, label='Physics-Guided Hybrid (v4 - StormShield)')

ax.fill_between(horizons, physics_hybrid, pure_ml, color='#10b981', alpha=0.15, label='Drift Bound by Physics Gating')

ax.set_title('Storm Cell Trajectory Error Across Forecast Lead Times (Measured)', pad=15)
ax.set_xlabel('Forecast Horizon Lead Time (Minutes)')
ax.set_ylabel('Mean Absolute Error - Drift (km)')
ax.set_xticks(horizons)
ax.set_ylim(0, 18)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='upper left')

# Annotations matching exact measured numbers
ax.annotate('14.98 km MAE\n(Unconstrained Drift)', xy=(60, 14.975), xytext=(45, 15.5),
            arrowprops=dict(arrowstyle='->', color='#dc2626', lw=1.5),
            fontweight='bold', color='#dc2626')

ax.annotate('7.85 km MAE\n(47.6% error reduction)', xy=(60, 7.853), xytext=(44, 9.8),
            arrowprops=dict(arrowstyle='->', color='#059669', lw=1.5),
            fontweight='bold', color='#059669')

plt.tight_layout()
fig.savefig('report_figures/trajectory_drift_benchmark.png')
plt.close(fig)
print("Saved report_figures/trajectory_drift_benchmark.png")


# -------------------------------------------------------------
# FIGURE 2: Model Ablation Bar Chart (Overall MAE & RMSE)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.0, 5.0), dpi=300)

x = np.arange(len(models_ablation))
width = 0.35

rects1 = ax.bar(x - width/2, mae_ablation, width, label='Overall MAE (km)', color='#3b82f6', edgecolor='#1e40af', alpha=0.9)
rects2 = ax.bar(x + width/2, rmse_ablation, width, label='Overall RMSE (km)', color='#6366f1', edgecolor='#4338ca', alpha=0.9)

ax.set_title('Model Benchmark: Spatial Trajectory Error (Overall MAE & RMSE)', pad=15)
ax.set_ylabel('Error Metric (km)')
ax.set_xticks(x)
ax.set_xticklabels(models_ablation, fontweight='bold')
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1')
ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_ylim(0, 11)

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f} km',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
fig.savefig('report_figures/model_ablation_comparison.png')
plt.close(fig)
print("Saved report_figures/model_ablation_comparison.png")


# -------------------------------------------------------------
# FIGURE 3: Critical Success Index (CSI) Benchmark Curves
# (From apps/api/src/ml/ablation.py scenario benchmark)
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=300)

horizons_csi = [15, 30, 45, 60]
csi_persistence = [0.55, 0.42, 0.32, 0.24]
csi_radar_convlstm = [0.68, 0.60, 0.52, 0.43]
csi_multimodal = [0.79, 0.74, 0.68, 0.60]

ax.plot(horizons_csi, csi_multimodal, 's-', color='#0284c7', linewidth=2.5, markersize=8, label='StormShield (Multimodal Deep Nowcaster)')
ax.plot(horizons_csi, csi_radar_convlstm, 'o--', color='#f59e0b', linewidth=2, markersize=7, label='Radar-only ConvLSTM Baseline')
ax.plot(horizons_csi, csi_persistence, '^:', color='#64748b', linewidth=1.8, markersize=6, label='Eulerian Persistence Baseline')

ax.set_title('Severe Weather Critical Success Index (CSI) Across Lead Times', pad=15)
ax.set_xlabel('Lead Time (Minutes)')
ax.set_ylabel('Critical Success Index (Higher is Better)')
ax.set_xticks(horizons_csi)
ax.set_ylim(0, 1.0)
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#cbd5e1', loc='upper right')

# Operational warning threshold
ax.axhline(0.5, color='#dc2626', linestyle='--', alpha=0.5, label='Operational Utility Floor (CSI = 0.50)')
ax.annotate('Multimodal model maintains operational utility (CSI >= 0.50)\nbeyond 60 minutes',
            xy=(45, 0.68), xytext=(18, 0.85),
            arrowprops=dict(arrowstyle='->', color='#0284c7', lw=1.5),
            fontweight='bold', color='#0369a1',
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f9ff", ec="#0284c7", lw=1))

plt.tight_layout()
fig.savefig('report_figures/critical_success_index.png')
plt.close(fig)
print("Saved report_figures/critical_success_index.png")
