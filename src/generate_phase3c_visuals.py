"""
src/generate_phase3c_visuals.py
-------------------------------
Generates visual comparison figures for Phase 3C synthetic sensor degradation:
1. Camera optical degradation (L0 -> L3)
2. LiDAR point cloud BEV degradation (L0 -> L3)
3. Radar corruption (L0 -> L3)
4. Multimodal condition comparison (C0, C1, C2, C3, C4, C5)
5. Real fog diagnostic comparison (Synthetic C4 vs fog_6_0)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from radiate_fusion import RadiateCalib
from src.dataset import RadiateIndexer, RadiateMultimodalDataset
from src.degradation import degrade_sample, compare_synthetic_vs_real_fog

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
OUT_DIR = os.path.join(BASE_DIR, "outputs", "phase3c")
os.makedirs(OUT_DIR, exist_ok=True)


def generate_camera_visuals(sample):
    print("Generating camera degradation visuals...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle("Phase 3C — Camera Optical Degradation (Koschmieder Atmospheric Scattering)", fontsize=14, fontweight="bold")

    levels = [0, 1, 2, 3]
    for idx, lvl in enumerate(levels):
        ax = axes[idx // 2, idx % 2]
        cond = f"C1_L{lvl}" if lvl > 0 else "C0"
        deg = degrade_sample(sample, cond, seed=42)
        img_np = deg["camera_image"].detach().cpu().numpy().transpose(1, 2, 0)
        img_np = np.clip(img_np, 0.0, 1.0)

        meta = deg["degradation_metadata"]["sensor_statistics"].get("camera", {})
        beta = meta.get("beta", 0.0)
        sigma = meta.get("blur_sigma", 0.0)
        trans = meta.get("mean_transmission", 1.0)
        title = f"Level {lvl} ({deg['degradation_metadata']['sensor_statistics'].get('camera', {}).get('level_name', 'Clean')})\n"
        title += f"beta={beta:.2f} m^-1 | blur_sigma={sigma:.1f} | mean_trans={trans:.2f}"

        ax.imshow(img_np)
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "camera_degradation_levels.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_lidar_visuals(sample):
    print("Generating LiDAR degradation visuals...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Phase 3C — LiDAR Point Cloud BEV Degradation (Beer-Lambert Extinction & Range Cutoff)", fontsize=14, fontweight="bold")

    levels = [0, 1, 2, 3]
    for idx, lvl in enumerate(levels):
        ax = axes[idx]
        cond = f"C2_L{lvl}" if lvl > 0 else "C0"
        deg = degrade_sample(sample, cond, seed=42)
        # Channel 1: Point density
        bev_density = deg["lidar_bev"][1].detach().cpu().numpy()

        meta = deg["degradation_metadata"]["sensor_statistics"].get("lidar", {})
        pts = meta.get("retained_point_count", 45000)
        pct = meta.get("retention_pct", 100.0)
        max_r = meta.get("max_retained_range", 85.0)

        ax.imshow(bev_density, cmap="viridis", origin="upper")
        ax.set_title(f"Level {lvl}\nPoints: {pts} ({pct:.1f}%)\nMax Range: {max_r:.1f}m", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "lidar_degradation_levels.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_radar_visuals(sample):
    print("Generating radar corruption visuals...")
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Phase 3C — Controlled Radar Corruption (Power Loss, Speckle, Clutter)", fontsize=14, fontweight="bold")

    levels = [0, 1, 2, 3]
    for idx, lvl in enumerate(levels):
        ax = axes[idx]
        cond = f"C3_L{lvl}" if lvl > 0 else "C0"
        deg = degrade_sample(sample, cond, seed=42)
        radar_np = deg["radar_bev"][0].detach().cpu().numpy()

        meta = deg["degradation_metadata"]["sensor_statistics"].get("radar", {})
        loss = meta.get("power_loss_db", 0.0)
        mean = meta.get("mean_degraded", float(radar_np.mean()))

        ax.imshow(radar_np, cmap="inferno", origin="upper")
        ax.set_title(f"Level {lvl}\nLoss: {loss:.1f} dB\nMean Signal: {mean:.3f}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "radar_corruption_levels.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_condition_comparison(sample):
    print("Generating multimodal condition comparison...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 11))
    fig.suptitle("Phase 3C — Multimodal Condition Matrix (C0 Clean vs C1, C2, C3, C4, C5 at Level 3)", fontsize=14, fontweight="bold")

    conditions = [
        ("C0", "C0: Clean Baseline"),
        ("C1_L3", "C1: Camera Dense Fog (L3)"),
        ("C2_L3", "C2: LiDAR Severe Fog (L3)"),
        ("C3_L3", "C3: Radar Severe Clutter (L3)"),
        ("C4_L3", "C4: Fog-Matched (Cam+LiDAR L3)"),
        ("C5_L3", "C5: Catastrophic All-Degraded (L3)"),
    ]

    for idx, (cond, label) in enumerate(conditions):
        ax = axes[idx // 3, idx % 3]
        deg = degrade_sample(sample, cond, seed=42)

        # Create fused composite BEV overlay:
        # Red = Radar, Green = LiDAR Density, Blue = Camera BEV Intensity
        r = deg["radar_bev"][0].detach().cpu().numpy()
        g = deg["lidar_bev"][1].detach().cpu().numpy()
        b = deg["camera_bev"].mean(dim=0).detach().cpu().numpy()

        composite = np.stack([r, g, b], axis=-1)
        composite = np.clip(composite * 1.5, 0.0, 1.0)

        ax.imshow(composite, origin="upper")
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "multimodal_conditions_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def generate_real_fog_diagnostic(sample_city, sample_fog):
    print("Generating real fog diagnostic report and visual...")
    # Degrade city_3_0 under C4 at levels 1, 2, 3
    deg_c4 = {
        lvl: degrade_sample(sample_city, f"C4_L{lvl}", seed=42)
        for lvl in [1, 2, 3]
    }

    report = compare_synthetic_vs_real_fog(sample_city, deg_c4, sample_fog)

    # Plot comparative bar charts
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 3C — Non-Learning Diagnostic: Synthetic Fog (C4) vs Real Fog (fog_6_0)", fontsize=13, fontweight="bold")

    # 1. Camera Sharpness (Laplacian variance)
    ax1 = axes[0]
    cam_labels = ["Clean (city)", "C4 L1", "C4 L2", "C4 L3", "Real Fog (fog_6_0)"]
    cam_sharpness = [
        report["camera_comparison"]["clean_city3_0"]["laplacian_variance"],
        report["camera_comparison"]["synthetic_c4_lvl1"]["laplacian_variance"],
        report["camera_comparison"]["synthetic_c4_lvl2"]["laplacian_variance"],
        report["camera_comparison"]["synthetic_c4_lvl3"]["laplacian_variance"],
        report["camera_comparison"]["real_fog6_0"]["laplacian_variance"],
    ]
    colors = ["#2ecc71", "#f39c12", "#e67e22", "#d35400", "#34495e"]
    bars1 = ax1.bar(cam_labels, cam_sharpness, color=colors)
    ax1.set_title("Camera Image Sharpness (Laplacian Variance)", fontsize=11)
    ax1.set_ylabel("Variance of Laplacian")
    ax1.tick_params(axis="x", rotation=20)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}", ha="center", va="bottom", fontsize=9)

    # 2. LiDAR Point Count
    ax2 = axes[1]
    lidar_labels = ["Clean (city)", "C4 L1", "C4 L2", "C4 L3", "Real Fog (fog_6_0)"]
    lidar_counts = [
        report["lidar_comparison"]["clean_city3_0"]["point_count"],
        report["lidar_comparison"]["synthetic_c4_lvl1"]["point_count"],
        report["lidar_comparison"]["synthetic_c4_lvl2"]["point_count"],
        report["lidar_comparison"]["synthetic_c4_lvl3"]["point_count"],
        report["lidar_comparison"]["real_fog6_0"]["point_count"],
    ]
    bars2 = ax2.bar(lidar_labels, lidar_counts, color=colors)
    ax2.set_title("LiDAR Point Cloud Size (Point Count)", fontsize=11)
    ax2.set_ylabel("Total Points")
    ax2.tick_params(axis="x", rotation=20)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 500, f"{int(yval)}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "real_fog_diagnostic_comparison.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    calib = RadiateCalib("config/default-calib.yaml")
    indexer_city = RadiateIndexer("city_3_0", split="train")
    ds_city = RadiateMultimodalDataset(indexer=indexer_city, calib=calib)
    sample_city = ds_city[0]

    indexer_fog = RadiateIndexer("fog_6_0")
    ds_fog = RadiateMultimodalDataset(indexer=indexer_fog, calib=calib)
    sample_fog = ds_fog[10]

    generate_camera_visuals(sample_city)
    generate_lidar_visuals(sample_city)
    generate_radar_visuals(sample_city)
    generate_condition_comparison(sample_city)
    generate_real_fog_diagnostic(sample_city, sample_fog)
    print("All Phase 3C visualizations generated successfully!")


if __name__ == "__main__":
    main()
