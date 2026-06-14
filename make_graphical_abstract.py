from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "result" / "graphical_abstract"


def add_box(ax, xy, width, height, text, fc="#FFFFFF", ec="#111827", fontsize=9.5):
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.05,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        linespacing=1.15,
    )
    return box


def add_arrow(ax, start, end, color="#111827", lw=1.8, style="-", rad=0.0, mutation_scale=14):
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        linestyle=style,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)


def add_arrow_head_only(ax, start, end, color="#111827", mutation_scale=12):
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=0.0,
        color=color,
    )
    ax.add_patch(arrow)


def draw_layered_model(ax):
    x0, y0, w, h = 0.035, 0.200, 0.285, 0.58
    layer_colors = ["#F8FAFC", "#F7FBFF", "#EFF6FF", "#DBEAFE"]
    layer_labels = ["near-surface layer", "velocity layer 1", "velocity layer 2", "velocity layer 3"]
    for i in range(4):
        yy = y0 + i * h / 4
        ax.add_patch(
            patches.Rectangle(
                (x0, yy),
                w,
                h / 4,
                facecolor=layer_colors[i],
                edgecolor="#CBD5E1",
                linewidth=0.75,
            )
        )
        short_labels = ["layer 3", "layer 2", "layer 1", "near surface"]
        label_y = yy + h / 8
        if short_labels[i] == "layer 3":
            label_y += 0.036
        ax.text(
            x0 + 0.012,
            label_y,
            short_labels[i],
            fontsize=5.9,
            va="center",
            color="#64748B",
        )
    source = (x0 + 0.020, y0 + 0.062)
    station = (x0 + w - 0.018, y0 + h - 0.070)
    ax.scatter(*source, s=58, marker="*", color="#DC2626", zorder=5)
    ax.scatter(*station, s=50, marker="^", color="#16A34A", edgecolor="#064E3B", linewidth=0.5, zorder=5)
    ax.text(source[0], source[1] - 0.040, "source", ha="center", fontsize=6.8, color="#7F1D1D")
    ax.text(station[0], station[1] + 0.030, "station", ha="center", fontsize=6.8, color="#064E3B")

    # Geometric direction.
    ax.plot(
        [source[0] + 0.012, station[0] - 0.012],
        [source[1] + 0.012, station[1] - 0.015],
        color="#2563EB",
        linestyle=(0, (5, 3)),
        linewidth=1.9,
        label="geometric source-receiver",
    )

    # Refracted source-side takeoff path shown as a piecewise ray path.
    ray_x = [
        source[0] + 0.020,
        x0 + 0.105,
        x0 + 0.180,
        x0 + 0.205,
        station[0] - 0.006,
    ]
    ray_y = [
        source[1] + 0.022,
        y0 + h / 4,
        y0 + h / 2,
        y0 + 3 * h / 4,
        station[1] - 0.010,
    ]
    ax.plot(ray_x, ray_y, color="#F97316", linewidth=2.3)
    # No arrowhead on the ray path: the color contrast alone marks the
    # source-side takeoff path and avoids adding an extra triangular symbol.

    legend_x = x0 + 0.130
    legend_y = y0 + 0.046
    ax.plot([legend_x, legend_x + 0.030], [legend_y, legend_y], color="#2563EB", linestyle=(0, (5, 3)), linewidth=1.7)
    ax.text(legend_x + 0.037, legend_y, "geometric source-receiver", fontsize=5.8, va="center", color="#475569")
    ax.plot([legend_x, legend_x + 0.030], [legend_y - 0.030, legend_y - 0.030], color="#F97316", linewidth=2.0)
    ax.text(legend_x + 0.037, legend_y - 0.030, "source-side takeoff", fontsize=5.8, va="center", color="#475569")

    ax.text(x0 + 0.025, 0.835, "Problem", fontsize=10.5, weight="bold", color="#111827")
    ax.text(
        x0 + w / 2,
        y0 - 0.048,
        "Straight direction differs from refracted ray takeoff direction",
        fontsize=7.4,
        color="#334155",
        va="top",
        ha="center",
    )

def draw_correction(ax):
    add_box(
        ax,
        (0.370, 0.615),
        0.205,
        0.155,
        "multilayer ray tracing\nreceiver incidence + azimuth",
        fc="#ECFEFF",
        ec="#0891B2",
        fontsize=8.2,
    )
    add_box(
        ax,
        (0.390, 0.395),
        0.165,
        0.135,
        "ray-parameter conservation",
        fc="#FFF7ED",
        ec="#EA580C",
        fontsize=8.2,
    )
    add_box(
        ax,
        (0.370, 0.175),
        0.205,
        0.155,
        "source-side takeoff\nradiation direction",
        fc="#F0FDF4",
        ec="#16A34A",
        fontsize=8.2,
    )
    add_arrow(ax, (0.473, 0.615), (0.473, 0.530), color="#64748B", lw=1.5)
    add_arrow(ax, (0.473, 0.395), (0.473, 0.330), color="#64748B", lw=1.5)
    ax.text(0.372, 0.835, "Correction", fontsize=10.5, weight="bold", color="#111827")
    ax.text(
        0.360,
        0.115,
        "replace geometric source-receiver direction\nin radiation-intensity calculation",
        fontsize=8.2,
        color="#111827",
        va="center",
    )


def draw_outcome(ax):
    x0, y0 = 0.645, 0.185
    ax.text(x0, 0.835, "Outcome", fontsize=10.5, weight="bold", color="#111827")

    add_box(
        ax,
        (x0, 0.645),
        0.255,
        0.125,
        "jSSA brightness-field construction",
        fc="#F5F3FF",
        ec="#7C3AED",
        fontsize=8.5,
    )
    add_box(
        ax,
        (x0, 0.445),
        0.255,
        0.120,
        "focal mechanism inversion\nstrike / dip / rake",
        fc="#F8FAFC",
        ec="#334155",
        fontsize=8.5,
    )
    add_arrow(ax, (x0 + 0.127, 0.645), (x0 + 0.127, 0.565), color="#64748B", lw=1.5)

    ax.scatter([x0 + 0.000], [0.395], s=26, marker="s", color="#93B5D8")
    ax.text(x0 + 0.016, 0.395, "geometric source-receiver", fontsize=6.8, va="center", color="#475569")
    ax.scatter([x0 + 0.182], [0.395], s=26, marker="s", color="#F97316")
    ax.text(x0 + 0.198, 0.395, "source-side takeoff", fontsize=6.8, va="center", color="#475569")
    labels = ["strike error", "dip error", "rake error"]
    geom_values = [0.88, 0.72, 0.92]
    source_values = [0.68, 0.60, 0.62]
    bar_x = x0 + 0.078
    for i, (lab, geom_val, source_val) in enumerate(zip(labels, geom_values, source_values)):
        yy = y0 + (2 - i) * 0.074
        ax.text(x0 - 0.006, yy + 0.012, lab, fontsize=7.0, color="#334155", va="center")
        ax.add_patch(patches.Rectangle((bar_x, yy + 0.016), 0.165 * geom_val, 0.014, facecolor="#93B5D8", edgecolor="none"))
        ax.add_patch(patches.Rectangle((bar_x, yy - 0.006), 0.165 * source_val, 0.014, facecolor="#F97316", edgecolor="none"))
        ax.text(x0 + 0.248, yy + 0.012, "decrease", fontsize=7.0, color="#16A34A", va="center")
    ax.text(x0, 0.095, "synthetic tests: lower mechanism errors\nreal field: improved holdout trends", fontsize=7.4, color="#334155")


def make_graphical_abstract():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "mathtext.fontset": "dejavusans",
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(13.28, 5.31), dpi=200)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="none"))
    ax.text(
        0.03,
        0.935,
        "Source-side takeoff radiation correction for layered microseismic focal mechanism inversion",
        fontsize=13.5,
        weight="bold",
        color="#111827",
    )
    ax.text(
        0.03,
        0.895,
        "Ray-traced takeoff directions replace straight source-receiver directions in theoretical radiation-intensity calculation.",
        fontsize=8.2,
        color="#475569",
    )

    draw_layered_model(ax)
    add_arrow(ax, (0.325, 0.50), (0.360, 0.50), color="#111827", lw=1.8, mutation_scale=14)
    draw_correction(ax)
    add_arrow(ax, (0.580, 0.50), (0.630, 0.50), color="#111827", lw=1.8, mutation_scale=14)
    draw_outcome(ax)

    # Visual separators.
    for x in [0.345, 0.615]:
        ax.plot([x, x], [0.13, 0.83], color="#E2E8F0", linewidth=1.0)

    fig.savefig(OUT_DIR / "graphical_abstract_source_takeoff.png", dpi=200, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT_DIR / "graphical_abstract_source_takeoff.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    make_graphical_abstract()
