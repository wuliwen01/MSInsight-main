import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from obspy.imaging.beachball import beach


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "result" / "manuscript_figures"

FIGURE_NAMES = {
    "fig1": "Fig. 1. Geometric source-receiver and source-side takeoff directions in a layered velocity model",
    "fig2": "Fig. 2. Station geometries of Dataset A and Dataset B",
    "fig3": "Fig. 3. Distributions of jSSA focal mechanism errors for Dataset A and Dataset B",
    "fig4": "Fig. 4. Example jSSA beachball solutions from Dataset A and Dataset B",
    "fig5": "Fig. 5. Paired station-holdout validation for the Ningxia field data",
    "fig6": "Fig. 6. Polar distributions of jSSA focal mechanism parameters for the Ningxia field data",
}

REAL_HOLDOUT_CSV = ROOT / "result" / "real_source_takeoff_jssa_holdout_50e3s" / "real_jssa_geometry_vs_source_takeoff_events.csv"

ANGLES = ["strike", "dip", "rake"]
ANGLE_LABELS = ["Strike", "Dip", "Rake"]
METRICS = ["mean", "median"]
COLORS = {
    "geom": "#4C78A8",
    "source": "#F58518",
    "jssa": "#6B7280",
    "bfnet": "#2F855A",
    "dual": "#805AD5",
}


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_csv_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def savefig(fig, name):
    ensure_dir(OUT_DIR)
    png = OUT_DIR / f"{name}.png"
    pdf = OUT_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png


def load_metrics():
    paths = {
        "A_geom": ROOT / "result" / "dataset_a_final" / "bfnet_geometric_source_receiver" / "bfnet_ml_ray_5000_geom_s2_100plus50_aux005_metrics.json",
        "A_source": ROOT / "result" / "dataset_a_final" / "bfnet_source_side_takeoff" / "bfnet_ml_ray_5000_source_s2_100plus50_aux005_metrics.json",
        "B_geom": ROOT / "result" / "cross_station_generalization_b500" / "bfnet_geometric_source_receiver_metrics.json",
        "B_source": ROOT / "result" / "cross_station_generalization_b500" / "bfnet_source_side_takeoff_metrics.json",
        "A_jssa_pair": ROOT / "result" / "dataset_a_final" / "jssa_geometric_vs_source_takeoff" / "jssa_ml_ray_5000_geom_vs_source_metrics.json",
        "A_bfnet_pair": ROOT / "result" / "dataset_a_final" / "bfnet_geometric_vs_source_takeoff_paired_stats.json",
    }
    data = {}
    for key, path in paths.items():
        if path.exists():
            data[key] = read_json(path)
    return data


def get_values(metric_payload, method, stat):
    return [metric_payload[method][f"{angle}_{stat}"] for angle in ANGLES]


def method_values(metrics, dataset_key, stat):
    payload = metrics[dataset_key]
    return {
        "jSSA": get_values(payload, "jssa", stat),
        "BFNet": get_values(payload, "bfnet", stat),
    }


def station_file(path):
    rows = []
    if not path.exists():
        return np.empty((0, 2))
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                rows.append((float(parts[1]) / 1000.0, float(parts[2]) / 1000.0))
    return np.asarray(rows, dtype=float)


def load_prediction_errors(path):
    rows = read_csv_rows(path)
    errors = {angle: [] for angle in ANGLES}
    sums = []
    true = {angle: [] for angle in ANGLES}
    pred = {angle: [] for angle in ANGLES}
    jssa = {angle: [] for angle in ANGLES}
    for row in rows:
        total = 0.0
        for angle in ANGLES:
            err = float(row[f"bfnet_err_{angle}"])
            errors[angle].append(err)
            total += err
            true[angle].append(float(row[f"true_{angle}"]))
            pred[angle].append(float(row[f"bfnet_{angle}"]))
            if f"jssa_err_{angle}" in row and row[f"jssa_err_{angle}"] != "":
                jssa[angle].append(float(row[f"jssa_err_{angle}"]))
        sums.append(total)
    return {
        "rows": rows,
        "errors": {k: np.asarray(v, dtype=float) for k, v in errors.items()},
        "sum": np.asarray(sums, dtype=float),
        "true": {k: np.asarray(v, dtype=float) for k, v in true.items()},
        "pred": {k: np.asarray(v, dtype=float) for k, v in pred.items()},
        "jssa": {k: np.asarray(v, dtype=float) for k, v in jssa.items()},
    }


def read_train_log(path):
    if not path.exists():
        return []
    rows = read_csv_rows(path)
    out = []
    for row in rows:
        out.append(
            {
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "val_loss": float(row["val_loss"]),
                "train_mt_loss": float(row.get("train_mt_loss") or "nan"),
                "val_mt_loss": float(row.get("val_mt_loss") or "nan"),
                "train_aux_loss": float(row.get("train_aux_loss") or "nan"),
                "val_aux_loss": float(row.get("val_aux_loss") or "nan"),
                "lr": float(row["lr"]),
            }
        )
    return out


def select_last_complete_epoch_run(rows, expected_epochs):
    if not rows:
        return rows
    runs = []
    current = []
    for row in rows:
        if current and row["epoch"] <= current[-1]["epoch"]:
            runs.append(current)
            current = []
        current.append(row)
    if current:
        runs.append(current)

    for run in reversed(runs):
        by_epoch = {row["epoch"]: row for row in run}
        if all(epoch in by_epoch for epoch in range(1, expected_epochs + 1)):
            return [by_epoch[epoch] for epoch in range(1, expected_epochs + 1)]

    # Fallback for partially retained logs: keep the last complete-looking block.
    return rows[-expected_epochs:] if len(rows) >= expected_epochs else rows


def combine_logs(prefix):
    branch = "bfnet_geometric_source_receiver" if prefix == "geom" else "bfnet_source_side_takeoff"
    base = ROOT / "result" / "dataset_a_final" / branch
    if prefix == "geom":
        files = [
            "bfnet_ml_ray_5000_geom_s1_100_best_train_log.csv",
            "bfnet_ml_ray_5000_geom_s1best_s2_100_aux005_train_log.csv",
            "bfnet_ml_ray_5000_geom_s2_100plus50_aux005_train_log.csv",
        ]
    else:
        files = [
            "bfnet_ml_ray_5000_source_s1_100_best_train_log.csv",
            "bfnet_ml_ray_5000_source_s1best_s2_100_aux005_train_log.csv",
            "bfnet_ml_ray_5000_source_s2_100plus50_aux005_train_log.csv",
        ]
    logs = []
    offset = 0
    stage_names = ["stage1", "stage2", "finetune"]
    expected_epochs = [100, 100, 50]
    for stage_name, fname, n_expected in zip(stage_names, files, expected_epochs):
        rows = select_last_complete_epoch_run(read_train_log(base / fname), n_expected)
        for row in rows:
            row = dict(row)
            row["global_epoch"] = offset + row["epoch"]
            row["stage_name"] = stage_name
            logs.append(row)
        offset += len(rows)
    return logs


def make_ray_direction_figure():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_aspect("equal", adjustable="box")

    def trim_polyline(points, start_trim=0.055, end_trim=0.055):
        pts = np.asarray(points, dtype=float).copy()
        start_vec = pts[1] - pts[0]
        start_len = np.linalg.norm(start_vec)
        if start_len > 0:
            pts[0] = pts[0] + start_vec / start_len * start_trim
        end_vec = pts[-2] - pts[-1]
        end_len = np.linalg.norm(end_vec)
        if end_len > 0:
            pts[-1] = pts[-1] + end_vec / end_len * end_trim
        return pts

    layers = [
        (0.12, -0.25, "Near-surface layer"),
        (-0.25, -0.60, "Velocity layer 1"),
        (-0.60, -0.95, "Velocity layer 2"),
        (-0.95, -1.28, "Velocity layer 3"),
    ]
    for top, bottom, label in layers:
        ax.add_patch(patches.Rectangle((-1.1, bottom), 2.2, top - bottom, fc="#FFFFFF", ec="none"))
        ax.text(-1.04, (top + bottom) / 2, label, fontsize=9, va="center")
    for y in [-0.25, -0.60, -0.95]:
        ax.axhline(y, color="#4A5568", lw=1)
    source = np.array([-0.58, -1.10])
    station = np.array([0.78, -0.08])
    bent = np.array([source, [-0.18, -0.95], [0.24, -0.60], [0.48, -0.25], station])
    bent_plot = trim_polyline(bent)
    geom_plot = trim_polyline(np.array([source, station]), start_trim=0.055, end_trim=0.018)

    ax.annotate(
        "",
        xy=geom_plot[-1],
        xytext=geom_plot[0],
        arrowprops=dict(
            arrowstyle="->",
            lw=2.0,
            color=COLORS["geom"],
            linestyle=(0, (5, 4)),
            mutation_scale=13,
            shrinkA=0,
            shrinkB=4,
        ),
    )
    ax.plot(bent_plot[:, 0], bent_plot[:, 1], "-", lw=2.4, color=COLORS["source"], label="Source-takeoff ray")
    ax.annotate("", xy=bent_plot[1], xytext=bent_plot[0], arrowprops=dict(arrowstyle="->", lw=2.4, color=COLORS["source"]))
    ax.scatter(*source, s=120, marker="*", color="#D53F8C", label="Source", zorder=5)
    ax.scatter(*station, s=80, marker="^", color="#111111", label="Station", zorder=5)
    ax.set_xlabel("")
    ax.set_ylabel("")
    geom_handle = plt.Line2D(
        [0, 1],
        [0, 0],
        color=COLORS["geom"],
        lw=2.0,
        linestyle=(0, (4, 3)),
        dash_capstyle="butt",
        label="Geometric source-receiver",
    )
    source_handle = plt.Line2D([0], [0], color=COLORS["source"], lw=2.4, label="Source-side takeoff")
    station_handle = plt.Line2D([0], [0], marker="^", color="none", markerfacecolor="#111111", markeredgecolor="#111111", markersize=7, label="Station")
    source_marker_handle = plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#D53F8C", markeredgecolor="#D53F8C", markersize=9, label="Source")
    ax.legend(
        handles=[source_marker_handle, station_handle, geom_handle, source_handle],
        loc="lower right",
        bbox_to_anchor=(0.97, 0.02),
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        fontsize=7.5,
        handlelength=2.8,
        borderpad=0.35,
        labelspacing=0.3,
    )
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.28, 0.12)
    ax.set_xticks([])
    ax.set_yticks([])
    savefig(fig, FIGURE_NAMES["fig1"])


def make_station_layouts():
    sta_a = station_file(ROOT / "conf" / "station_synth_dataset_a.xyz")
    sta_b = station_file(ROOT / "conf" / "station_synth_dataset_b.xyz")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, sta, title, color in [
        (axes[0], sta_a, "Dataset A: radial stations", COLORS["geom"]),
        (axes[1], sta_b, "Dataset B: 8x8 square stations", COLORS["source"]),
    ]:
        ax.scatter(sta[:, 0], sta[:, 1], s=42, color=color, edgecolor="white", linewidth=0.6)
        ax.scatter([0], [0], marker="*", s=120, color="#D53F8C", label="Reference source")
        ax.axhline(0, color="#A0AEC0", lw=0.7)
        ax.axvline(0, color="#A0AEC0", lw=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("X (km)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Y (km)")
    savefig(fig, FIGURE_NAMES["fig2"])


def sdr_from_row(row, prefix):
    return [
        float(row[f"{prefix}_strike"]),
        float(row[f"{prefix}_dip"]),
        float(row[f"{prefix}_rake"]),
    ]


def error_sum_from_row(row, prefix):
    return sum(float(row[f"{prefix}_err_{angle}"]) for angle in ANGLES)


def draw_beachball(ax, sdr, title, subtitle="", color="#4C78A8"):
    collection = beach(
        sdr,
        xy=(0.0, 0.0),
        width=1.55,
        linewidth=0.8,
        facecolor=color,
        bgcolor="white",
        edgecolor="#111827",
        alpha=0.95,
    )
    ax.add_collection(collection)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, pad=2)
    if subtitle:
        ax.text(0.5, -0.08, subtitle, transform=ax.transAxes, ha="center", va="top", fontsize=8)


def beachball_mechanism_figures():
    geom_path = ROOT / "result" / "dataset_a_final" / "bfnet_geometric_source_receiver" / "bfnet_ml_ray_5000_geom_s2_100plus50_aux005_predictions.csv"
    source_path = ROOT / "result" / "dataset_a_final" / "bfnet_source_side_takeoff" / "bfnet_ml_ray_5000_source_s2_100plus50_aux005_predictions.csv"
    geom_b_path = ROOT / "result" / "cross_station_generalization_b500" / "bfnet_geometric_source_receiver_predictions.csv"
    source_b_path = ROOT / "result" / "cross_station_generalization_b500" / "bfnet_source_side_takeoff_predictions.csv"
    if not geom_path.exists() or not source_path.exists():
        return
    geom_rows = {row["event_id"]: row for row in read_csv_rows(geom_path)}
    source_rows = {row["event_id"]: row for row in read_csv_rows(source_path)}
    common = sorted(set(geom_rows) & set(source_rows))
    if not common:
        return

    if not geom_b_path.exists() or not source_b_path.exists():
        return
    geom_b_rows = {row["event_id"]: row for row in read_csv_rows(geom_b_path)}
    source_b_rows = {row["event_id"]: row for row in read_csv_rows(source_b_path)}

    # Visually representative jSSA cases where source-side takeoff is closer
    # to the true mechanism than the geometric source-receiver direction.
    fig26_cases = [
        ("A1", geom_rows, source_rows, "ev_004565"),
        ("A2", geom_rows, source_rows, "ev_003846"),
        ("A3", geom_rows, source_rows, "ev_000724"),
        ("B1", geom_b_rows, source_b_rows, "ev_000034"),
        ("B2", geom_b_rows, source_b_rows, "ev_000141"),
        ("B3", geom_b_rows, source_b_rows, "ev_000408"),
    ]
    selected_jssa = [
        (label, g_rows, s_rows, event_id)
        for label, g_rows, s_rows, event_id in fig26_cases
        if event_id in g_rows and event_id in s_rows
    ]
    if not selected_jssa:
        return

    fig, axes = plt.subplots(3, len(selected_jssa), figsize=(2.1 * len(selected_jssa), 5.8))
    if len(selected_jssa) == 1:
        axes = np.asarray(axes).reshape(3, 1)
    for i, (event_label, g_rows, s_rows, event_id) in enumerate(selected_jssa):
        g = g_rows[event_id]
        s = s_rows[event_id]
        draw_beachball(axes[0, i], sdr_from_row(s, "true"), event_label, color="#BDBDBD")
        draw_beachball(axes[1, i], sdr_from_row(g, "jssa"), "", color="#4E79A7")
        draw_beachball(axes[2, i], sdr_from_row(s, "jssa"), "", color="#F28E2B")

    if len(selected_jssa) >= 6:
        axes[0, 1].text(0.5, 1.24, "Dataset A", transform=axes[0, 1].transAxes, ha="center", va="bottom", fontsize=11)
        axes[0, 4].text(0.5, 1.24, "Dataset B", transform=axes[0, 4].transAxes, ha="center", va="bottom", fontsize=11)

    row_labels = ["True", "Geometric source-receiver", "source-side takeoff"]
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].text(
            -0.28,
            0.5,
            label,
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )
    fig.subplots_adjust(wspace=0.22, hspace=0.18, top=0.95)
    savefig(fig, FIGURE_NAMES["fig4"])


def jssa_error_distribution_figure():
    datasets = [
        (
            "Dataset A",
            ROOT / "result" / "dataset_a_final" / "bfnet_geometric_source_receiver" / "bfnet_ml_ray_5000_geom_s2_100plus50_aux005_predictions.csv",
            ROOT / "result" / "dataset_a_final" / "bfnet_source_side_takeoff" / "bfnet_ml_ray_5000_source_s2_100plus50_aux005_predictions.csv",
        ),
        (
            "Dataset B",
            ROOT / "result" / "cross_station_generalization_b500" / "bfnet_geometric_source_receiver_predictions.csv",
            ROOT / "result" / "cross_station_generalization_b500" / "bfnet_source_side_takeoff_predictions.csv",
        ),
    ]
    if any(not geom_path.exists() or not source_path.exists() for _, geom_path, source_path in datasets):
        return

    bins = np.arange(0, 65, 5)
    xticks = [0, 15, 30, 45, 60]
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.2), sharex=True)
    colors = {
        "Geometric source-receiver": "#4E79A7",
        "source-side takeoff": "#F28E2B",
    }

    for row_idx, (dataset_name, geom_path, source_path) in enumerate(datasets):
        geom_rows = read_csv_rows(geom_path)
        source_rows = read_csv_rows(source_path)
        for col_idx, angle in enumerate(ANGLES):
            ax = axes[row_idx, col_idx]
            for label, rows in [
                ("Geometric source-receiver", geom_rows),
                ("source-side takeoff", source_rows),
            ]:
                values = np.asarray([float(row[f"jssa_err_{angle}"]) for row in rows])
                values = values[(values >= bins[0]) & (values < bins[-1])]
                weights = np.full(values.shape, 100.0 / len(values))
                ax.hist(
                    values,
                    bins=bins,
                    weights=weights,
                    color=colors[label],
                    alpha=0.16,
                    edgecolor="none",
                    label=label,
                )
                ax.hist(
                    values,
                    bins=bins,
                    weights=weights,
                    color=colors[label],
                    histtype="step",
                    linewidth=1.8,
                )
            if row_idx == 0:
                ax.set_title(f"{ANGLE_LABELS[col_idx]} error", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{dataset_name}\nEvent percentage (%)", fontsize=10)
            if row_idx == 1:
                ax.set_xlabel("Error (deg)", fontsize=10)
            ax.set_xlim(0, 60)
            ax.set_xticks(xticks)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, alpha=0.22, linewidth=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=9)
    fig.subplots_adjust(top=0.88, hspace=0.34, wspace=0.25)
    savefig(fig, FIGURE_NAMES["fig3"])


def read_real_rows(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def real_value(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def real_key_full(row):
    return (row["file"], row["event_id"])


def real_key_holdout(row):
    return (row["file"], row["event_id"], row["split_id"])


def pair_real_rows(rows, key_func, split_kind=None):
    buckets = defaultdict(dict)
    for row in rows:
        if split_kind and row.get("split_kind") != split_kind:
            continue
        buckets[key_func(row)][row["direction"]] = row
    pairs = []
    for key, group in buckets.items():
        if "geometry" in group and "source-takeoff" in group:
            pairs.append((key, group["geometry"], group["source-takeoff"]))
    return pairs


def real_station_holdout_scatter():
    rows = read_real_rows(REAL_HOLDOUT_CSV)
    pairs = pair_real_rows(rows, real_key_holdout, "holdout")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    specs = [
        ("holdout_norm_residual", "Holdout residual", "lower"),
        ("holdout_amp_corr", "Amplitude correlation", "higher"),
        ("mechanism_entropy", "Mechanism entropy", "lower"),
    ]
    for ax, (field, title, better_side) in zip(axes, specs):
        gx = np.array([real_value(g, field) for _, g, _ in pairs])
        sy = np.array([real_value(s, field) for _, _, s in pairs])
        mask = np.isfinite(gx) & np.isfinite(sy)
        gx, sy = gx[mask], sy[mask]
        if better_side == "lower":
            source_better = int(np.sum(sy < gx))
        else:
            source_better = int(np.sum(sy > gx))

        ax.scatter(gx, sy, s=22, color="#4C78A8", alpha=0.72, edgecolor="white", linewidth=0.35)
        mn = min(float(np.min(gx)), float(np.min(sy)))
        mx = max(float(np.max(gx)), float(np.max(sy)))
        pad = (mx - mn) * 0.08 if mx > mn else 0.01
        ax.plot([mn - pad, mx + pad], [mn - pad, mx + pad], color="0.35", lw=1.0, ls="--")
        ax.set_xlim(mn - pad, mx + pad)
        ax.set_ylim(mn - pad, mx + pad)
        ax.set_xlabel("Geometric source-receiver")
        ax.set_ylabel("Source-side takeoff")
        ax.set_title(title)
        ax.text(
            0.04,
            0.96,
            f"Source-side better: {source_better}/{len(gx)}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="0.75", boxstyle="round,pad=0.25"),
        )
        ax.grid(color="0.92")
    savefig(fig, FIGURE_NAMES["fig5"])


def draw_real_polar_distribution_compare(ax, geom_values, source_values, title, angle_mode):
    geom_values = np.asarray(geom_values, dtype=float)
    source_values = np.asarray(source_values, dtype=float)
    mask = np.isfinite(geom_values) & np.isfinite(source_values)
    geom_values = geom_values[mask]
    source_values = source_values[mask]

    if angle_mode == "dip":
        bins_deg = np.linspace(0.0, 90.0, 7)
        theta_ticks = [0, 30, 60, 90]
        theta_labels = ["0°", "30°", "60°", "90°"]
        ax.set_thetamin(0)
        ax.set_thetamax(90)
        geom_plot = geom_values
        source_plot = source_values
    elif angle_mode == "rake":
        bins_deg = np.linspace(0.0, 360.0, 13)
        theta_ticks = [0, 60, 120, 180, 240, 300]
        theta_labels = ["0°", "60°", "120°", "180°", "-120°", "-60°"]
        geom_plot = (geom_values + 360.0) % 360.0
        source_plot = (source_values + 360.0) % 360.0
    else:
        bins_deg = np.linspace(0.0, 360.0, 13)
        theta_ticks = [0, 60, 120, 180, 240, 300]
        theta_labels = ["0°", "60°", "120°", "180°", "240°", "300°"]
        geom_plot = geom_values % 360.0
        source_plot = source_values % 360.0

    centers_deg = (bins_deg[:-1] + bins_deg[1:]) / 2.0
    geom_counts, _ = np.histogram(geom_plot, bins=bins_deg)
    source_counts, _ = np.histogram(source_plot, bins=bins_deg)
    theta = np.deg2rad(centers_deg)
    width = np.deg2rad(np.diff(bins_deg).mean())
    offset = width * 0.16

    ax.bar(
        theta - offset,
        geom_counts,
        width=width * 0.36,
        color="#4C78A8",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.6,
        label="Geometric source-receiver",
    )
    ax.bar(
        theta + offset,
        source_counts,
        width=width * 0.36,
        color="#F58518",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.6,
        label="Source-side takeoff",
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_thetagrids(theta_ticks, labels=theta_labels)
    ax.set_title(title, y=1.12, pad=0)
    ax.grid(color="0.86", linewidth=0.7)


def real_mechanism_distribution_figure():
    rows = read_real_rows(REAL_HOLDOUT_CSV)
    pairs = pair_real_rows(rows, real_key_holdout, "full")
    if not pairs:
        pairs = pair_real_rows(rows, real_key_full, "full")

    geom = {
        "strike": np.array([real_value(g, "strike") for _, g, _ in pairs], dtype=float),
        "dip": np.array([real_value(g, "dip") for _, g, _ in pairs], dtype=float),
        "rake": np.array([real_value(g, "rake") for _, g, _ in pairs], dtype=float),
    }
    source = {
        "strike": np.array([real_value(s, "strike") for _, _, s in pairs], dtype=float),
        "dip": np.array([real_value(s, "dip") for _, _, s in pairs], dtype=float),
        "rake": np.array([real_value(s, "rake") for _, _, s in pairs], dtype=float),
    }

    fig = plt.figure(figsize=(11.5, 4.2))
    axes = [
        fig.add_axes([0.055, 0.22, 0.27, 0.66], projection="polar"),
        fig.add_axes([0.365, 0.22, 0.27, 0.66], projection="polar"),
        fig.add_axes([0.675, 0.22, 0.27, 0.66], projection="polar"),
    ]
    draw_real_polar_distribution_compare(axes[0], geom["strike"], source["strike"], "(a) Strike", "strike")
    draw_real_polar_distribution_compare(axes[1], geom["dip"], source["dip"], "(b) Dip", "dip")
    draw_real_polar_distribution_compare(axes[2], geom["rake"], source["rake"], "(c) Rake", "rake")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    savefig(fig, FIGURE_NAMES["fig6"])


def write_manifest():
    manifest = OUT_DIR / "figure_captions.md"
    lines = [
        f"{FIGURE_NAMES['fig1']}.",
        "",
        f"{FIGURE_NAMES['fig2']}.",
        "",
        f"{FIGURE_NAMES['fig3']}.",
        "",
        f"{FIGURE_NAMES['fig4']}.",
        "",
        f"{FIGURE_NAMES['fig5']}.",
        "",
        f"{FIGURE_NAMES['fig6']}.",
    ]
    manifest.write_text("\n".join(lines), encoding="utf-8")


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    ensure_dir(OUT_DIR)
    metrics = load_metrics()

    make_ray_direction_figure()
    make_station_layouts()
    if "A_geom" in metrics and "A_source" in metrics:
        beachball_mechanism_figures()
        jssa_error_distribution_figure()
    real_station_holdout_scatter()
    real_mechanism_distribution_figure()
    write_manifest()
    print(f"Generated figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
