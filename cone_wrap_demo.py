"""
A straight line on an unrolled cone wraps around the cone.

A cone is a flat surface. Cut it along one seam and it unrolls into a
circular sector. Lay copies of that sector side by side and you are
looking at the cone's surface, repeated. A straight line drawn across
those copies never turns. Roll the paper back up and the same line
winds around the cone: it is a geodesic, straight along the surface
and curved only because the surface itself has been rolled.

Run:
    python cone_wrap_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Wedge


# Three copies fill a half-plane. One straight line can cross a half-plane
# at most, so three is the most loops a single straight cut can show.
SECTOR_COPIES = 3
SLANT_LENGTH = 1.0
# Closest the line comes to the apex, measured along the paper.
# Far enough from the tip that a full turn is a visible loop on the cone.
LINE_OFFSET = 0.62
SAMPLES = 480

PAPER = "#f6f1e7"
INK = "#241c14"
SECTOR_COLORS = ("#f6d7a8", "#f3e4bc", "#e7d3a1")
LAP_COLORS = ("#c4452e", "#1f7a45", "#2458a6")
CONE_COLOR = "#e4c396"


def cone_angles(copies):
    """Sector angle and the cone's half-angle, for `copies` sectors in a half-plane."""
    beta = np.pi / copies
    # Arc of one sector equals the cone's base circumference: beta = 2 pi sin(alpha).
    alpha = np.arcsin(beta / (2 * np.pi))
    return alpha, beta


def straight_line_on_paper(length, offset, samples):
    """Horizontal chord at distance `offset` from the apex, clipped to the sector radius."""
    half = np.sqrt(length**2 - offset**2)
    # Travel from the right-hand rim, in toward the apex, and out to the left rim.
    abscissa = np.linspace(half, -half, samples)
    x = abscissa
    y = np.full_like(abscissa, offset)
    return x, y


def map_to_cone(x, y, alpha, beta):
    """Roll paper coordinates back onto the cone. Apex sits at the top."""
    slant = np.hypot(x, y)
    unrolled = np.arctan2(y, x)
    lap = np.floor(unrolled / beta).astype(int)
    local = unrolled - lap * beta
    # Each sector occupies the full turn around the cone.
    phi = local * (2 * np.pi / beta)
    radius = slant * np.sin(alpha)
    height = SLANT_LENGTH * np.cos(alpha)
    # Nudge along the outward normal so the ink sits on the surface.
    lift = 0.01
    cone_x = radius * np.cos(phi) + lift * np.cos(alpha) * np.cos(phi)
    cone_y = radius * np.sin(phi) + lift * np.cos(alpha) * np.sin(phi)
    cone_z = height - slant * np.cos(alpha) + lift * np.sin(alpha)
    return cone_x, cone_y, cone_z, slant, unrolled, phi, lap


def _style_3d(ax):
    ax.set_proj_type("persp")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("#efe6d6")
    ax.grid(False)
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-62)


def _draw_cone(ax, alpha):
    height = SLANT_LENGTH * np.cos(alpha)
    slant = np.linspace(0, SLANT_LENGTH, 50)
    phi = np.linspace(0, 2 * np.pi, 90)
    slant_grid, phi_grid = np.meshgrid(slant, phi)
    radius = slant_grid * np.sin(alpha)
    xs = radius * np.cos(phi_grid)
    ys = radius * np.sin(phi_grid)
    zs = height - slant_grid * np.cos(alpha)
    ax.plot_surface(
        xs, ys, zs,
        color=CONE_COLOR,
        alpha=0.55,
        linewidth=0,
        antialiased=True,
        shade=True,
        zorder=1,
    )

    # Latitude rings make the slope readable on a slender cone.
    ring_phi = np.linspace(0, 2 * np.pi, 180)
    for level in (0.45, 0.7, 1.0):
        ring_r = level * np.sin(alpha)
        ax.plot(
            ring_r * np.cos(ring_phi),
            ring_r * np.sin(ring_phi),
            np.full_like(ring_phi, height - level * np.cos(alpha)),
            color="#8c6840",
            lw=0.7,
            alpha=0.7,
            zorder=2,
        )

    # The seam that was cut. Every radial edge on the unrolled fan is this line.
    seam_s = np.linspace(0, SLANT_LENGTH, 40)
    ax.plot(
        seam_s * np.sin(alpha),
        np.zeros_like(seam_s),
        height - seam_s * np.cos(alpha),
        color=INK,
        lw=1.6,
        ls=(0, (2, 2)),
        zorder=3,
    )
    ax.scatter([0], [0], [height], color=INK, s=18, zorder=4)

    limit = SLANT_LENGTH * np.sin(alpha) * 1.35
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(0, height * 1.08)
    ax.set_box_aspect((2 * limit, 2 * limit, height * 1.08))


def _screen_angle(unrolled):
    """Mirror the fan so the walk starts on the left and proceeds to the right."""
    return np.pi - unrolled


def _draw_fan(ax, beta, length, laps):
    for lap in range(laps):
        # Screen angle runs backwards, so the first loop lands on the left.
        start = _screen_angle((lap + 1) * beta)
        stop = _screen_angle(lap * beta)
        wedge = Wedge(
            (0, 0),
            length,
            np.rad2deg(start),
            np.rad2deg(stop),
            facecolor=SECTOR_COLORS[lap],
            edgecolor="#6b542c",
            linewidth=1.1,
            alpha=0.95,
            zorder=1,
        )
        ax.add_patch(wedge)
        mid = _screen_angle((lap + 0.5) * beta)
        # Low in each sector, clear of the chord and of the moving dot.
        ax.text(
            0.38 * length * np.cos(mid),
            0.38 * length * np.sin(mid),
            f"loop {lap + 1}",
            ha="center",
            va="center",
            color=LAP_COLORS[lap],
            fontsize=11,
            fontweight="medium",
            zorder=2,
        )

    ax.plot(0, 0, "o", color=INK, ms=5, zorder=4)
    ax.annotate(
        "apex",
        (0, 0),
        textcoords="offset points",
        xytext=(8, -14),
        color=INK,
        fontsize=10,
    )
    ax.set_aspect("equal")
    ax.set_xlim(-1.18 * length, 1.18 * length)
    ax.set_ylim(-0.28 * length, 1.18 * length)
    ax.axis("off")


def _split_laps(lap_index):
    """Index ranges for each contiguous run of one sector copy."""
    if len(lap_index) == 0:
        return []
    cuts = np.where(np.diff(lap_index) != 0)[0] + 1
    ranges = []
    start = 0
    for cut in list(cuts) + [len(lap_index)]:
        # Keep the shared endpoint so colored pieces meet.
        stop = min(len(lap_index), cut + (0 if cut == len(lap_index) else 1))
        ranges.append((start, stop, int(lap_index[start])))
        start = cut
    return ranges


def build_demo():
    alpha, beta = cone_angles(SECTOR_COPIES)
    paper_x, paper_y = straight_line_on_paper(SLANT_LENGTH, LINE_OFFSET, SAMPLES)
    cone_x, cone_y, cone_z, slant, unrolled, phi, lap = map_to_cone(
        paper_x, paper_y, alpha, beta
    )

    fig = plt.figure(figsize=(13.4, 7.3), facecolor=PAPER)
    fig.suptitle(
        "A straight line on the unrolled cone wraps around the cone",
        fontsize=16,
        color=INK,
        y=0.97,
    )
    grid = fig.add_gridspec(
        1, 2, left=0.03, right=0.985, top=0.86, bottom=0.13, wspace=0.06
    )
    ax_paper = fig.add_subplot(grid[0, 0])
    ax_cone = fig.add_subplot(grid[0, 1], projection="3d")
    ax_paper.set_facecolor(PAPER)
    ax_cone.set_facecolor(PAPER)
    _style_3d(ax_cone)

    ax_paper.set_title(
        "Unrolled  ·  the path is one straight line",
        color=INK,
        fontsize=12,
        pad=8,
    )
    ax_cone.set_title(
        "Rolled back up  ·  the same path winds around",
        color=INK,
        fontsize=12,
        pad=8,
    )

    _draw_fan(ax_paper, beta, SLANT_LENGTH, SECTOR_COPIES)
    _draw_cone(ax_cone, alpha)

    for start, stop, lap_id in _split_laps(lap):
        color = LAP_COLORS[lap_id % len(LAP_COLORS)]
        ax_paper.plot(
            -paper_x[start:stop],
            paper_y[start:stop],
            color=color,
            lw=3.2,
            solid_capstyle="round",
            zorder=5,
        )
        ax_cone.plot(
            cone_x[start:stop],
            cone_y[start:stop],
            cone_z[start:stop],
            color=color,
            lw=2.8,
            zorder=5,
        )

    follower_paper, = ax_paper.plot(
        [-paper_x[0]], [paper_y[0]],
        "o",
        ms=11,
        color=LAP_COLORS[int(lap[0])],
        markeredgecolor="white",
        markeredgewidth=1.6,
        zorder=6,
    )
    follower_cone, = ax_cone.plot(
        [cone_x[0]], [cone_y[0]], [cone_z[0]],
        "o",
        ms=9,
        color=LAP_COLORS[int(lap[0])],
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=6,
    )

    status = fig.text(
        0.5, 0.055,
        "",
        ha="center",
        va="center",
        color=INK,
        fontsize=11,
    )
    fig.text(
        0.5, 0.018,
        "Cut the cone along the dashed seam and lay the copies flat. "
        "The ink never bends. Each color is one more trip around the seam.",
        ha="center",
        va="center",
        color="#5c5146",
        fontsize=10,
    )

    def update(frame):
        color = LAP_COLORS[int(lap[frame]) % len(LAP_COLORS)]
        follower_paper.set_data([-paper_x[frame]], [paper_y[frame]])
        follower_paper.set_color(color)
        follower_cone.set_data([cone_x[frame]], [cone_y[frame]])
        follower_cone.set_3d_properties([cone_z[frame]])
        follower_cone.set_color(color)
        ax_cone.view_init(elev=24, azim=-62 + 18 * np.sin(2 * np.pi * frame / SAMPLES))
        wound = np.rad2deg((unrolled[frame] - unrolled[0]) * (2 * np.pi / beta))
        status.set_text(
            f"following the line    "
            f"distance from apex  {slant[frame]:.2f}    "
            f"loop {int(lap[frame]) + 1} of {SECTOR_COPIES}    "
            f"wound {wound:.0f}° around the cone"
        )
        return follower_paper, follower_cone, status

    update(0)
    animation = FuncAnimation(
        fig,
        update,
        frames=SAMPLES,
        interval=25,
        blit=False,
        repeat=True,
    )
    # Keep the animation alive for as long as the figure is.
    fig._cone_animation = animation

    geometry = {
        "alpha_deg": float(np.rad2deg(alpha)),
        "sector_deg": float(np.rad2deg(beta)),
        "laps": lap,
        "paper": (paper_x, paper_y),
        "cone": (cone_x, cone_y, cone_z),
        "phi": phi,
    }
    return fig, update, geometry


def main():
    fig, _update, _geometry = build_demo()
    try:
        fig.canvas.manager.set_window_title("Unrolled cone")
    except AttributeError:
        pass
    plt.show()


if __name__ == "__main__":
    main()
