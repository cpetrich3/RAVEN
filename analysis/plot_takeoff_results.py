import os
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Scan <input_dir> for all 'vehicle_local_position.csv' files,\n"
            "compute key metrics, produce report‐ready plots (with consistent colors),\n"
            "and save summary CSV to <output_dir>."
        )
    )
    p.add_argument(
        "input_dir",
        type=str,
        help="Root directory to search recursively for 'vehicle_local_position.csv' files."
    )
    p.add_argument(
        "output_dir",
        type=str,
        help="Directory where PNG plots and 'summary_stats.csv' will be saved."
    )
    return p.parse_args()

def find_hover_stable_time(df, target=5.0, tol=0.05, window_s=0.1):
    """
    Return the first t_sec ≥ target where altitude_m remains within [target-tol, target+tol]
    for at least window_s seconds.
    """
    dt = np.median(np.diff(df["t_sec"].values))
    window_n = max(1, int(window_s / dt))
    alt = df["altitude_m"].values
    t = df["t_sec"].values

    # find first index where altitude_m ≥ target
    reach_idxs = np.where(alt >= target)[0]
    if len(reach_idxs) == 0:
        return np.nan
    start = reach_idxs[0]
    # slide window to ensure stability
    for i in range(start, len(alt) - window_n):
        if np.all(np.abs(alt[i : i + window_n] - target) <= tol):
            return t[i]
    return np.nan

def main():
    args = parse_args()
    input_dir = os.path.abspath(os.path.expanduser(args.input_dir))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    # 1. Find all vehicle_local_position.csv under input_dir (recursive)
    pattern = os.path.join(input_dir, "**", "vehicle_local_position.csv")
    all_paths = glob.glob(pattern, recursive=True)
    if not all_paths:
        raise FileNotFoundError(f"No 'vehicle_local_position.csv' found under {input_dir} (recursive).")

    # 2. Load each CSV, compute metrics, and collect run IDs
    all_runs = []
    summary_rows = []
    run_ids = []

    for csv_path in sorted(all_paths):
        run_folder = Path(csv_path).parent.name  
        run_id = run_folder.replace("_csv", "")    
        run_ids.append(run_id)

        df = pd.read_csv(csv_path)
        df["t_sec"] = (df["timestamp"] - df["timestamp"].iloc[0]) / 1e6
        df["altitude_m"] = -df["z"]   # PX4’s z is negative above ground
        df["speed_horiz"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
        df["speed_total"] = np.sqrt(df["vx"]**2 + df["vy"]**2 + df["vz"]**2)
        df["lat_error"] = np.sqrt(df["x"]**2 + df["y"]**2)
        df["run_id"] = run_id
        all_runs.append(df)

        # Compute numeric metrics for summary:
        landing_error = df["lat_error"].iloc[-1]
        peak_speed_horiz = df["speed_horiz"].max()
        peak_speed_total = df["speed_total"].max()
        peak_alt = df["altitude_m"].max()
        peak_overshoot = max(0.0, peak_alt - 5.0)

        reach_idxs = np.where(df["altitude_m"] >= 5.0)[0]
        t_reach_5m = float(df["t_sec"].iloc[reach_idxs[0]]) if len(reach_idxs) > 0 else np.nan
        t_hover_stable = find_hover_stable_time(df, target=5.0, tol=0.05, window_s=0.1)
        t_touchdown = df["t_sec"].iloc[-1]

        mask_below_02 = df.index[df["altitude_m"] <= 0.2]
        if len(mask_below_02) > 0:
            idx_02 = mask_below_02[0]
            vvert_at_02 = df["vz"].iloc[idx_02]
        else:
            vvert_at_02 = np.nan

        summary_rows.append({
            "run_id": run_id,
            "t_liftoff_s": 0.0,
            "t_reach_5m_s": t_reach_5m,
            "t_hover_stable_s": t_hover_stable,
            "t_touchdown_s": t_touchdown,
            "landing_error_m": landing_error,
            "peak_horiz_speed_mps": peak_speed_horiz,
            "peak_total_speed_mps": peak_speed_total,
            "peak_overshoot_m": peak_overshoot,
            "vvert_at_0.2m_mps": vvert_at_02
        })

    # Deduplicate run_ids and sort them consistently
    run_ids = sorted(set(run_ids))
    # Ask Matplotlib for its default color cycle, then assign one color per run:
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    # If we have more runs than default cycle, it will wrap—but assuming ≤10 runs, default is fine.
    run_colors = {run_id: color_cycle[i] for i, run_id in enumerate(run_ids)}

    # Create a single DataFrame combining all runs
    combined = pd.concat(all_runs, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows)

    # Make sure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # PLOT 1: Trajectory with Inset
    fig, ax_main = plt.subplots(figsize=(6, 6))
    for run_id in run_ids:
        grp = combined[combined["run_id"] == run_id]
        c = run_colors[run_id]
        # Full trace in light gray
        ax_main.plot(grp["x"], grp["y"], color="lightgray", linewidth=1, alpha=0.5)
        # First 2 seconds in the run’s color
        mask_f2 = grp["t_sec"] <= 2.0
        ax_main.plot(grp.loc[mask_f2, "x"], grp.loc[mask_f2, "y"], color=c, linewidth=2)
        # Last 2 seconds in the same color
        t_end = grp["t_sec"].iloc[-1]
        mask_l2 = grp["t_sec"] >= (t_end - 2.0)
        ax_main.plot(grp.loc[mask_l2, "x"], grp.loc[mask_l2, "y"], color=c, linewidth=2)
        # Mark start (green) and end (run-color X)
        ax_main.scatter(grp["x"].iloc[0], grp["y"].iloc[0], color="green", marker="o", s=50)
        ax_main.scatter(grp["x"].iloc[-1], grp["y"].iloc[-1], color=c, marker="X", s=60)

    # Mark the global origin
    ax_main.scatter(0, 0, color="black", marker="*", s=100, label="Origin")

    ax_main.set_title("Top-Down Trajectory (X = North, Y = East)")
    ax_main.set_xlabel("North (m)")
    ax_main.set_ylabel("East (m)")
    ax_main.grid(True)

    # Legend: show one example color for each run
    legend_handles = []
    for run_id in run_ids:
        legend_handles.append(plt.Line2D([0], [0], color=run_colors[run_id], lw=2, label=run_id))
    legend_handles.append(plt.Line2D([0], [0], marker="*", color="black", linestyle="None", markersize=10, label="Origin"))
    ax_main.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")

    # Inset (enlarged) around origin ±0.2 m
    ax_ins = inset_axes(
        ax_main,
        width="45%", height="45%",
        loc="lower left",
        bbox_to_anchor=(0.05, 0.05, 0.45, 0.45),
        bbox_transform=ax_main.transAxes
    )
    for run_id in run_ids:
        grp = combined[combined["run_id"] == run_id]
        c = run_colors[run_id]
        # Faded full trace
        ax_ins.plot(grp["x"], grp["y"], color="lightgray", linewidth=1, alpha=0.5)
        # Last 2 s segment in run color
        t_end = grp["t_sec"].iloc[-1]
        mask_l2 = grp["t_sec"] >= (t_end - 2.0)
        ax_ins.plot(grp.loc[mask_l2, "x"], grp.loc[mask_l2, "y"], color=c, linewidth=2)
        # Landing point
        ax_ins.scatter(grp["x"].iloc[-1], grp["y"].iloc[-1], color=c, marker="X", s=60)
    ax_ins.scatter(0, 0, color="black", marker="*", s=60)

    ax_ins.set_xlim(-0.2, 0.2)
    ax_ins.set_ylim(-0.2, 0.2)
    ax_ins.set_xticks([-0.2, 0.0, 0.2])
    ax_ins.set_yticks([-0.2, 0.0, 0.2])
    ax_ins.set_title("Landing Zoom (±0.2 m)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "trajectory_consistent_colors.png"), dpi=300)
    plt.close(fig)

    # PLOT 2: Altitude Profile (Climb & Descent)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=False)

    # Panel 1: Climb & Hover
    for run_id in run_ids:
        grp = combined[combined["run_id"] == run_id]
        c = run_colors[run_id]
        ax1.plot(grp["t_sec"], grp["altitude_m"], color=c, label=run_id, alpha=0.7)

    # Annotate only the first run’s events (to avoid overlap)
    first = summary_rows[0]
    ax1.annotate("Liftoff", xy=(0.0, 0.0), xytext=(1.0, 2.0),
                 arrowprops=dict(arrowstyle="->", lw=1))
    if not np.isnan(first["t_reach_5m_s"]):
        ax1.annotate("Reached 5 m", xy=(first["t_reach_5m_s"], 5.0),
                     xytext=(first["t_reach_5m_s"] + 0.5, 6.0),
                     arrowprops=dict(arrowstyle="->", lw=1))
    if not np.isnan(first["t_hover_stable_s"]):
        ax1.annotate("Hover Stable", xy=(first["t_hover_stable_s"], 5.0),
                     xytext=(first["t_hover_stable_s"] + 0.5, 5.5),
                     arrowprops=dict(arrowstyle="->", lw=1))

    ax1.set_xlim(0, first["t_hover_stable_s"] + 2.0)
    ax1.set_ylim(0, 6)
    ax1.set_ylabel("Altitude (m)")
    ax1.set_title("Climb & Hover (0–6 m)")
    ax1.grid(True)
    # no legend here (we’ll put in the second panel)

    # Panel 2: Descent & Landing
    for run_id in run_ids:
        grp = combined[combined["run_id"] == run_id]
        c = run_colors[run_id]
        ax2.plot(grp["t_sec"], grp["altitude_m"], color=c, label=run_id, alpha=0.7)

    ax2.annotate("Touchdown", xy=(first["t_touchdown_s"], 0.0),
                 xytext=(first["t_touchdown_s"] - 2.0, 1.0),
                 arrowprops=dict(arrowstyle="->", lw=1))

    ax2.set_xlim(first["t_hover_stable_s"] - 2.0, first["t_touchdown_s"] + 2.0)
    ax2.set_ylim(-0.5, 6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Altitude (m)")
    ax2.set_title("Descent & Landing")
    ax2.grid(True)

    # Shared legend on the right side
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.02, 0.5), loc="center left", fontsize="small")

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    fig.savefig(os.path.join(output_dir, "altitude_consistent_colors.png"), dpi=300)
    plt.close(fig)

    # PLOT 3: Horizontal Speed vs Altitude (Last 2 m)
    plt.figure(figsize=(6, 4))
    for run_id in run_ids:
        grp = combined[combined["run_id"] == run_id]
        c = run_colors[run_id]
        mask_last2m = grp["altitude_m"] <= 2.0
        plt.scatter(grp.loc[mask_last2m, "speed_horiz"],
                    grp.loc[mask_last2m, "altitude_m"],
                    color=c, label=run_id, s=20, alpha=0.7)

    plt.xlabel("Horizontal Speed (m/s)")
    plt.ylabel("Altitude (m)")
    plt.title("Horizontal Speed vs. Altitude (Last 2 m of Descent)")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "horiz_speed_vs_altitude_consistent_colors.png"), dpi=300)
    plt.close()

    # PLOT 4: Landing Dispersion with 0.2 m Circle
    touchdowns = combined.groupby("run_id").apply(lambda g: g.loc[g["t_sec"].idxmax()])
    plt.figure(figsize=(5, 5))
    for run_id in run_ids:
        row = touchdowns.loc[run_id]
        c = run_colors[run_id]
        plt.scatter(row["x"], row["y"], color=c, s=80, label=run_id, alpha=0.7)

    # Plot origin
    plt.scatter(0, 0, color="black", marker="*", s=100, label="Origin")

    # Draw 0.2 m dashed circle
    circle = plt.Circle((0, 0), 0.2, color="blue", fill=False, linestyle="--", linewidth=1.5)
    plt.gca().add_patch(circle)

    # Label each point
    for run_id in run_ids:
        row = touchdowns.loc[run_id]
        plt.text(row["x"] + 0.005, row["y"] + 0.005, run_id, fontsize=8)

    plt.title("Landing Dispersion (±0.2 m Pass Boundary)")
    plt.xlabel("North (m)")
    plt.ylabel("East (m)")
    plt.axis("equal")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "landing_dispersion_consistent_colors.png"), dpi=300)
    plt.close()

    # PLOT 5: Lateral Error vs Time (Last 10 s Before Touchdown)
    t_touch_dict = {row["run_id"]: row["t_touchdown_s"] for row in summary_rows}
    plt.figure(figsize=(8, 4))
    for run_id in run_ids:
        grp = combined[combined["run_id"] == run_id]
        c = run_colors[run_id]
        t_td = t_touch_dict[run_id]
        mask_final10 = (grp["t_sec"] >= (t_td - 10.0)) & (grp["t_sec"] <= t_td)
        plt.plot(grp.loc[mask_final10, "t_sec"] - t_td,
                 grp.loc[mask_final10, "lat_error"],
                 color=c, label=run_id, linewidth=1.5)

    plt.axhline(0.2, color="orange", linestyle="--", label="Pass Threshold (0.2 m)")
    plt.xlabel("Time to Touchdown (s) [0 = touchdown]")
    plt.ylabel("Lateral Error (m)")
    plt.title("Lateral Error During Final 10 s of Descent")
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_time_consistent_colors.png"), dpi=300)
    plt.close()

    # PLOT 6a: Scatter – Landing Error vs Overshoot (cm)

    landing_cm = (summary_df["landing_error_m"].values * 100).tolist()
    overshoot_cm = (summary_df["peak_overshoot_m"].values * 100).tolist()

    plt.figure(figsize=(5, 4))
    for i, run_id in enumerate(run_ids):
        x = overshoot_cm[i]
        y = landing_cm[i]
        c = run_colors[run_id]
        plt.scatter(x, y, color=c, s=80)
        plt.text(x + 0.5, y + 0.5, run_id, fontsize=8)

    plt.xlabel("Peak Overshoot Above 5 m (cm)")
    plt.ylabel("Final Landing Error (cm)")
    plt.title("Landing Error vs. Climb Overshoot")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_overshoot_consistent_colors.png"), dpi=300)
    plt.close()

    # PLOT 6b: Scatter – Landing Error vs Descent Rate @ 0.2 m 
    landing_cm = (summary_df["landing_error_m"].values * 100).tolist()
    vvert_02_ms = summary_df["vvert_at_0.2m_mps"].tolist()

    plt.figure(figsize=(5, 4))
    for i, run_id in enumerate(run_ids):
        x = vvert_02_ms[i]
        y = landing_cm[i]
        c = run_colors[run_id]
        plt.scatter(x, y, color=c, s=80)
        plt.text(x + 0.005, y + 0.5, run_id, fontsize=8)

    plt.xlabel("Vertical Speed at 0.2 m (m/s)")
    plt.ylabel("Final Landing Error (cm)")
    plt.title("Landing Error vs. Descent Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_descent_rate_consistent_colors.png"), dpi=300)
    plt.close()

    # PLOT 6c: Scatter – Landing Error vs Horizontal Dist @ 1 m
    dist_at_1m_cm = {}
    for df in all_runs:
        run_id = df["run_id"].iloc[0]
        mask_desc = df["vz"] < 0
        mask_alt1 = df["altitude_m"] <= 1.0
        idx_candidates = df.index[mask_desc & mask_alt1]
        if len(idx_candidates) > 0:
            idx_1m = idx_candidates[0]
            dist_at_1m_cm[run_id] = np.sqrt(df["x"].iloc[idx_1m]**2 + df["y"].iloc[idx_1m]**2) * 100
        else:
            dist_at_1m_cm[run_id] = np.nan

    landing_cm_ser = pd.Series({run_id: summary_df.set_index("run_id")["landing_error_m"][run_id] * 100
                                for run_id in run_ids})
    dist1p_cm_ser = pd.Series(dist_at_1m_cm)

    plt.figure(figsize=(5, 4))
    for run_id in run_ids:
        x = dist1p_cm_ser[run_id]
        y = landing_cm_ser[run_id]
        c = run_colors[run_id]
        plt.scatter(x, y, color=c, s=80)
        plt.text(x + 0.5, y + 0.5, run_id, fontsize=8)

    plt.xlabel("Distance from Origin at 1 m Altitude (cm)")
    plt.ylabel("Final Landing Error (cm)")
    plt.title("Landing Error vs. Dist at 1 m Altitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_vs_dist_at_1m_consistent_colors.png"), dpi=300)
    plt.close()

    summary_csv_path = os.path.join(output_dir, "summary_stats_consistent_colors.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print("All consistent‐color plots and summary CSV saved to:", output_dir)
