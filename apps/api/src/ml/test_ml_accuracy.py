"""
SIH26072 — ML Model Accuracy & Evaluation Benchmark v3
Tests PyTorch Deep Neural Nowcaster v3 with real-data training.
"""

import numpy as np
from src.ml.inference import StormMLInference


def evaluate_accuracy_benchmark():
    print("=" * 70)
    print("🌩️  SIH26072 — STORM NOWCASTING ACCURACY EVALUATION BENCHMARK v4")
    print("    Physics-Guided Hybrid Trajectory Clamp")
    print("=" * 70)

    # 1. Initialize ML Inference Engine (v3 with 18 features)
    inference = StormMLInference()
    
    if not inference.loaded:
        print("[SKIP] Model not yet trained. Train first with: python -m src.ml.trainer")
        return

    # 2. Generate test cells with realistic Gujarat storm parameters
    horizons = [15, 30, 45, 60]
    test_cells = _generate_realistic_test_cells(n=50)

    ml_errors_km = {h: [] for h in horizons}
    linear_errors_km = {h: [] for h in horizons}
    hybrid_errors_km = {h: [] for h in horizons}

    for cell_data in test_cells:
        cell = cell_data["cell"]
        history = cell_data["history"]
        true_displacements = cell_data["true_displacements"]

        vx = cell["velocity_x"]
        vy = cell["velocity_y"]

        # A) ML PyTorch Model Forecast (now includes physics-guided clamp)
        nowcast = inference.predict_cell_nowcast(cell, history, horizons_minutes=horizons)
        ml_forecasts = nowcast["forecasts"]

        # Calculate errors for each horizon
        for h_idx, h in enumerate(horizons):
            true_dx, true_dy = true_displacements[h_idx]

            # Linear model prediction (pure velocity extrapolation)
            steps = h / 10.0
            lin_dx = vx * steps
            lin_dy = vy * steps

            # ML predicted position (with physics-guided clamp) in km
            pred_lat = ml_forecasts[h_idx]["predicted_lat"]
            pred_lon = ml_forecasts[h_idx]["predicted_lon"]

            center_lat = 23.0225
            center_lon = 72.5714
            ml_dy_grid = (pred_lat - center_lat) * 111.0
            cos_lat = np.cos(np.radians(pred_lat))
            ml_dx_grid = (pred_lon - center_lon) * 111.0 * cos_lat

            # Physics-guided hybrid (explicit computation for benchmark)
            w_ml = 0.05 + 0.45 * ((h / 60.0) ** 1.5)
            w_phys = 1.0 - w_ml
            hybrid_dx = w_phys * lin_dx + w_ml * ml_dx_grid
            hybrid_dy = w_phys * lin_dy + w_ml * ml_dy_grid

            ml_err = np.sqrt((ml_dx_grid - true_dx)**2 + (ml_dy_grid - true_dy)**2)
            lin_err = np.sqrt((lin_dx - true_dx)**2 + (lin_dy - true_dy)**2)
            hyb_err = np.sqrt((hybrid_dx - true_dx)**2 + (hybrid_dy - true_dy)**2)

            ml_errors_km[h].append(ml_err)
            linear_errors_km[h].append(lin_err)
            hybrid_errors_km[h].append(hyb_err)

    # Aggregate all horizons
    all_ml = [e for errs in ml_errors_km.values() for e in errs]
    all_lin = [e for errs in linear_errors_km.values() for e in errs]
    all_hyb = [e for errs in hybrid_errors_km.values() for e in errs]

    ml_mae = float(np.mean(all_ml))
    ml_rmse = float(np.sqrt(np.mean(np.square(all_ml))))
    lin_mae = float(np.mean(all_lin))
    lin_rmse = float(np.sqrt(np.mean(np.square(all_lin))))
    hyb_mae = float(np.mean(all_hyb))
    hyb_rmse = float(np.sqrt(np.mean(np.square(all_hyb))))

    print(f"\n📊 EVALUATION RESULTS OVER {len(test_cells)} TEST TRAJECTORIES:")
    print(f"{'='*60}")
    print(f"  {'Method':<35} {'MAE (km)':>10} {'RMSE (km)':>10}")
    print(f"  {'-'*35} {'-'*10} {'-'*10}")
    print(f"  {'Baseline Linear Extrapolation':<35} {lin_mae:>10.3f} {lin_rmse:>10.3f}")
    print(f"  {'Pure ML (Deep Nowcaster v3)':<35} {ml_mae:>10.3f} {ml_rmse:>10.3f}")
    print(f"  {'⚡ Physics-Guided Hybrid (v4)':<35} {hyb_mae:>10.3f} {hyb_rmse:>10.3f}")
    print(f"{'='*60}")

    improvement = ((lin_mae - hyb_mae) / max(0.01, lin_mae)) * 100.0
    if improvement > 0:
        print(f"  🎯 Hybrid vs Linear: {improvement:.1f}% IMPROVEMENT in position accuracy!")
    else:
        print(f"  📍 Hybrid within {abs(improvement):.1f}% of physics baseline (ML provides non-linear correction)")

    print(f"\n📈 PER-HORIZON BREAKDOWN:")
    print(f"  {'Horizon':<10} {'Linear (km)':>12} {'Pure ML (km)':>12} {'Hybrid (km)':>12}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*12}")
    for h in horizons:
        l = float(np.mean(linear_errors_km[h]))
        m = float(np.mean(ml_errors_km[h]))
        hy = float(np.mean(hybrid_errors_km[h]))
        w = 0.05 + 0.45 * ((h / 60.0) ** 1.5)
        print(f"  +{h:<9} {l:>12.3f} {m:>12.3f} {hy:>12.3f}  (w_ml={w:.2f})")

    print(f"\n✅ Benchmark evaluation completed!")


def _generate_realistic_test_cells(n: int = 50) -> list:
    """Generate test cells with realistic Gujarat storm parameters."""
    rng = np.random.default_rng(42)
    test_cells = []

    for i in range(n):
        # Realistic Gujarat monsoon storm parameters
        speed_kmh = rng.uniform(15.0, 50.0)
        direction = rng.uniform(180.0, 270.0)  # SW monsoon flow
        rad = np.radians(direction)
        vx = float(speed_kmh / 6.0 * np.sin(rad))  # grid cells per timestep
        vy = float(-speed_kmh / 6.0 * np.cos(rad))

        max_dbz = float(rng.uniform(40.0, 65.0))
        lat = float(rng.uniform(22.0, 24.0))
        lon = float(rng.uniform(71.5, 73.5))

        cell = {
            "centroid_y": 100.0,
            "centroid_x": 100.0,
            "center_lat": lat,
            "center_lon": lon,
            "velocity_x": vx,
            "velocity_y": vy,
            "max_reflectivity_dbz": max_dbz,
            "mean_reflectivity_dbz": max_dbz * 0.75,
            "area_sq_km": float(rng.uniform(30.0, 200.0)),
            "lightning_rate": float(rng.uniform(5.0, 25.0)),
            "optical_flow_vx": vx * 0.9,  # simulate OF data
            "optical_flow_vy": vy * 0.9,
            "optical_flow_divergence": float(rng.uniform(-0.1, 0.1)),
            "optical_flow_curl": float(rng.uniform(-0.05, 0.05)),
        }

        history = [
            {"x": 100.0 - 2*vx, "y": 100.0 - 2*vy, "max_reflectivity_dbz": max_dbz - 2.0, "area_sq_km": cell["area_sq_km"] * 0.9, "lightning_rate": cell["lightning_rate"] * 0.8},
            {"x": 100.0 - vx,   "y": 100.0 - vy,   "max_reflectivity_dbz": max_dbz - 1.0, "area_sq_km": cell["area_sq_km"] * 0.95, "lightning_rate": cell["lightning_rate"] * 0.9},
            {"x": 100.0,        "y": 100.0,         "max_reflectivity_dbz": max_dbz,       "area_sq_km": cell["area_sq_km"],        "lightning_rate": cell["lightning_rate"]},
        ]

        # True displacement with slight curvature
        turning_rate = float(rng.uniform(-0.03, 0.03))
        true_displacements = []
        for h in [15, 30, 45, 60]:
            steps = h / 10.0
            s_rad = rad + turning_rate * steps
            dx = float(speed_kmh / 6.0 * np.sin(s_rad) * steps)
            dy = float(-speed_kmh / 6.0 * np.cos(s_rad) * steps)
            true_displacements.append((dx, dy))

        test_cells.append({
            "cell": cell,
            "history": history,
            "true_displacements": true_displacements,
        })

    return test_cells


if __name__ == "__main__":
    evaluate_accuracy_benchmark()
