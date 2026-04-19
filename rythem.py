# Polished Colab-safe Rhythm Attention Visualizer
# - better colors
# - cleaner layout
# - output tile = dominant attended beat type
# - normalized attention bars
# - dark demo-friendly theme

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch
from IPython.display import HTML, display

# =========================
# User settings
# =========================
CSV_DIR = "."
FPS = 2
SAVE_PATH = None   # e.g. None, "demo.gif", "demo.mp4"

# =========================
# Theme
# =========================
BG          = "#0b1020"
PANEL       = "#121a2b"
GRID        = "#24314d"
TEXT        = "#eef2ff"
SUBTEXT     = "#a8b3cf"
ACCENT      = "#ffe600"

# Beat-type colors
COLOR_SILENCE = "#2b2f3a"   # dark gray
COLOR_WEAK    = "#33d1ff"   # cyan
COLOR_MEDIUM  = "#4f6fff"   # blue
COLOR_STRONG  = "#d04dff"   # purple

# Heatmaps
RAW_CMAP   = "viridis"
ATTN_CMAP  = "magma"

# =========================
# Helpers
# =========================
def find_csv(csv_dir: Path, name: str) -> Path:
    p = csv_dir / name
    if not p.exists():
        raise FileNotFoundError(f"Missing required file: {p}")
    return p

def load_tokens(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    for col in df.columns:
        c = str(col).strip().lower()
        if c in ("beat_type", "token", "tokens", "beat", "value"):
            vals = pd.to_numeric(df[col], errors="coerce").dropna().astype(int).to_numpy()
            if len(vals) > 0:
                return vals
    df_num = df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df_num.iloc[:, -1].astype(int).to_numpy()

def load_matrix_noheader(path: Path) -> np.ndarray:
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return df.to_numpy()

def load_perf(path: Path):
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
        if df.shape[0] >= 1:
            return {str(k): v for k, v in df.iloc[0].to_dict().items()}
    except Exception:
        pass
    return {}

def maybe_strip_index_col(mat: np.ndarray, expected_rows=None, expected_cols=None) -> np.ndarray:
    m = np.asarray(mat)
    if m.ndim != 2:
        return m
    if expected_rows is not None and m.shape[0] != expected_rows:
        return m
    if expected_cols is not None and m.shape[1] == expected_cols + 1:
        first_col = m[:, 0]
        if np.array_equal(first_col.astype(int), np.arange(len(first_col))):
            return m[:, 1:]
    return m

def normalize_for_display(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    xmin = np.min(x)
    xmax = np.max(x)
    if math.isclose(xmax, xmin):
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

def token_color(token: int):
    if token == 0:
        return COLOR_SILENCE
    if token == 1:
        return COLOR_WEAK
    if token == 2:
        return COLOR_MEDIUM
    if token == 3:
        return COLOR_STRONG
    return "#888888"

def dominant_attention_token(q: int, attention_probs: np.ndarray, tokens: np.ndarray) -> int:
    row = attention_probs[q].astype(float)
    if np.all(row == 0):
        return 0
    k = int(np.argmax(row))
    return int(tokens[k])

def row_attention_colors(q: int, attention_probs: np.ndarray, tokens: np.ndarray):
    row = attention_probs[q].astype(float)
    row_disp = row.copy()
    if np.max(row_disp) > 0:
        row_disp = row_disp / np.max(row_disp)
    colors = []
    for i in range(len(row_disp)):
        if i > q:
            colors.append("#1b2130")
        else:
            base = token_color(int(tokens[i]))
            colors.append(base)
    return row_disp, colors

# =========================
# Load data
# =========================
csv_dir = Path(CSV_DIR)

tokens = load_tokens(find_csv(csv_dir, "tokens.csv"))
num_tokens = len(tokens)

raw_scores = load_matrix_noheader(find_csv(csv_dir, "raw_scores.csv"))
masked_scores = load_matrix_noheader(find_csv(csv_dir, "masked_scores.csv"))
attention_probs = load_matrix_noheader(find_csv(csv_dir, "attention_probs.csv"))
output_mat = load_matrix_noheader(find_csv(csv_dir, "output.csv"))
perf = load_perf(csv_dir / "perf.csv")

raw_scores = maybe_strip_index_col(raw_scores, expected_rows=num_tokens, expected_cols=num_tokens)
masked_scores = maybe_strip_index_col(masked_scores, expected_rows=num_tokens, expected_cols=num_tokens)
attention_probs = maybe_strip_index_col(attention_probs, expected_rows=num_tokens, expected_cols=num_tokens)
output_mat = maybe_strip_index_col(output_mat, expected_rows=num_tokens)

assert raw_scores.shape == (num_tokens, num_tokens)
assert masked_scores.shape == (num_tokens, num_tokens)
assert attention_probs.shape == (num_tokens, num_tokens)
assert output_mat.shape[0] == num_tokens

output_dim = output_mat.shape[1]

raw_disp = normalize_for_display(raw_scores)
prob_disp = attention_probs.astype(float)
if np.max(prob_disp) > 0:
    prob_disp = prob_disp / np.max(prob_disp)

# =========================
# Matplotlib style
# =========================
plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "axes.edgecolor": SUBTEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": SUBTEXT,
    "ytick.color": SUBTEXT,
    "text.color": TEXT,
    "axes.titlecolor": TEXT,
    "font.size": 12,
})

# =========================
# Layout
# =========================
plt.close("all")
fig = plt.figure(figsize=(18, 10), facecolor=BG)
gs = fig.add_gridspec(
    3, 4,
    height_ratios=[0.95, 3.25, 2.2],
    width_ratios=[1.35, 1.35, 1.05, 0.92],
    hspace=0.32,
    wspace=0.30
)

ax_tokens = fig.add_subplot(gs[0, :3])
ax_raw    = fig.add_subplot(gs[1, 0])
ax_mask   = fig.add_subplot(gs[1, 1])
ax_bars   = fig.add_subplot(gs[1, 2])
ax_out    = fig.add_subplot(gs[2, :3])
ax_info   = fig.add_subplot(gs[:, 3])

for ax in [ax_tokens, ax_raw, ax_mask, ax_bars, ax_out, ax_info]:
    ax.set_facecolor(PANEL)

fig.suptitle("Rhythm Attention Visualizer", fontsize=34, fontweight="bold", y=0.972)

# =========================
# Top token strip
# =========================
ax_tokens.set_xlim(-0.8, num_tokens - 0.2)
ax_tokens.set_ylim(-1.1, 1.35)
ax_tokens.set_xticks(range(num_tokens))
ax_tokens.set_xticklabels(range(num_tokens), fontsize=10, color=SUBTEXT)
ax_tokens.set_yticks([])
for spine in ax_tokens.spines.values():
    spine.set_visible(False)

token_scatter = ax_tokens.scatter(
    np.arange(num_tokens),
    np.zeros(num_tokens),
    s=430,
    c=[token_color(int(t)) for t in tokens],
    edgecolors="#cfd8ea",
    linewidths=1.5,
    zorder=3
)

query_marker = ax_tokens.scatter(
    [0], [0],
    s=950,
    facecolors="none",
    edgecolors=ACCENT,
    linewidths=3.0,
    zorder=4
)

ax_tokens.text(
    0.01, 0.93,
    "0 = silence    1 = weak    2 = medium    3 = strong",
    transform=ax_tokens.transAxes,
    fontsize=14,
    color=SUBTEXT,
    va="top"
)

token_title = ax_tokens.text(
    0.5, 1.12,
    "Input Rhythm Tokens   |   Current Beat = 0",
    transform=ax_tokens.transAxes,
    ha="center",
    va="bottom",
    fontsize=24,
    color=TEXT
)

# =========================
# Raw score matrix
# =========================
im_raw = ax_raw.imshow(raw_disp, aspect="equal", cmap=RAW_CMAP, origin="upper", vmin=0, vmax=1)
ax_raw.set_title("Raw Score Matrix", fontsize=22, pad=12)
ax_raw.set_xlabel("Key / Compared Beat", fontsize=14)
ax_raw.set_ylabel("Query / Current Beat", fontsize=14)
raw_row_line = ax_raw.axhline(0, color="#e5edf9", lw=2.2, ls="--", alpha=0.95)

# =========================
# Causal attention
# =========================
im_mask = ax_mask.imshow(prob_disp, aspect="equal", cmap=ATTN_CMAP, origin="upper", vmin=0, vmax=1)
ax_mask.set_title("Causal Attention Weights", fontsize=22, pad=12)
ax_mask.set_xlabel("Key / Past Beats", fontsize=14)
ax_mask.set_ylabel("Query / Current Beat", fontsize=14)
mask_row_line = ax_mask.axhline(0, color="#00eaff", lw=2.2, ls="--", alpha=0.95)
diag_line, = ax_mask.plot([], [], color="#d7deec", lw=1.6, ls=":", alpha=0.9)

cbar_raw = fig.colorbar(im_raw, ax=ax_raw, fraction=0.046, pad=0.04)
cbar_raw.set_label("Relative Score", fontsize=12)
cbar_mask = fig.colorbar(im_mask, ax=ax_mask, fraction=0.046, pad=0.04)
cbar_mask.set_label("Attention Weight", fontsize=12)

# =========================
# Current row bars
# =========================
row0, row0_colors = row_attention_colors(0, attention_probs, tokens)
bars = ax_bars.bar(np.arange(num_tokens), row0, width=0.82, color=row0_colors)
ax_bars.set_title("Current Row Attention", fontsize=22, pad=12)
ax_bars.set_xlabel("Past Beat Index", fontsize=14)
ax_bars.set_ylabel("Normalized Weight", fontsize=14)
ax_bars.set_xlim(-0.6, num_tokens - 0.4)
ax_bars.set_ylim(0, 1.08)
ax_bars.set_xticks(np.arange(0, num_tokens, 2))
ax_bars.tick_params(axis="x", labelsize=9)
ax_bars.grid(axis="y", color=GRID, alpha=0.8, linestyle="--", linewidth=0.8)

# =========================
# Output vector + tile
# =========================
ax_out.set_title("Output Vector + Output Tile", fontsize=24, pad=10)
out_ylim = max(1.0, float(np.max(output_mat)) * 1.18)
ax_out.set_xlim(-1.0, output_dim + 7.2)
ax_out.set_ylim(0, out_ylim)
ax_out.set_xlabel("Output Dimension", fontsize=14)
ax_out.set_ylabel("Value", fontsize=14)
ax_out.grid(axis="y", color=GRID, alpha=0.8, linestyle="--", linewidth=0.8)

output_bars = ax_out.bar(np.arange(output_dim), output_mat[0], width=0.82)
for i, b in enumerate(output_bars):
    b.set_color(plt.cm.winter(i / max(1, output_dim - 1)))

tile_x0 = output_dim + 1.5
tile_y0 = out_ylim * 0.18
tile_w  = 4.5
tile_h  = out_ylim * 0.66

tile_box = FancyBboxPatch(
    (tile_x0, tile_y0), tile_w, tile_h,
    boxstyle="round,pad=0.03,rounding_size=0.10",
    linewidth=2.5,
    edgecolor="#d7deec",
    facecolor=token_color(dominant_attention_token(0, attention_probs, tokens))
)
ax_out.add_patch(tile_box)

tile_title = ax_out.text(
    tile_x0 + tile_w / 2,
    tile_y0 + tile_h + out_ylim * 0.04,
    "Dominant Attention Type",
    ha="center", va="bottom",
    fontsize=16, fontweight="bold"
)

query_text = ax_out.text(
    0.012, 0.94,
    "Query q = 0",
    transform=ax_out.transAxes,
    fontsize=24, fontweight="bold",
    va="top"
)

# =========================
# Right info panel
# =========================
ax_info.axis("off")
info_text = ax_info.text(
    0.03, 0.95,
    "",
    va="top", ha="left",
    fontsize=14, linespacing=1.4, color=TEXT
)

# =========================
# Update
# =========================
def update(frame: int):
    q = int(frame)

    # top strip
    query_marker.set_offsets(np.array([[q, 0.0]]))
    token_title.set_text(f"Input Rhythm Tokens   |   Current Beat = {q}")

    # heatmap highlights
    raw_row_line.set_ydata([q, q])
    mask_row_line.set_ydata([q, q])
    diag_line.set_data([-0.5, num_tokens - 0.5], [-0.5, num_tokens - 0.5])

    # current row attention
    row_disp, row_colors = row_attention_colors(q, attention_probs, tokens)
    for i, b in enumerate(bars):
        b.set_height(float(row_disp[i]))
        b.set_color(row_colors[i])
        b.set_alpha(0.25 if i > q else 1.0)

    # output vector
    out_row = output_mat[q]
    for i, b in enumerate(output_bars):
        b.set_height(float(out_row[i]))
        b.set_color(plt.cm.winter(i / max(1, output_dim - 1)))

    # output tile = dominant attended beat type
    dom_type = dominant_attention_token(q, attention_probs, tokens)
    tile_box.set_facecolor(token_color(dom_type))

    type_name = {
        0: "Silence",
        1: "Weak",
        2: "Medium",
        3: "Strong"
    }.get(dom_type, "Unknown")

    query_text.set_text(f"Query q = {q}")

    perf_lines = []
    if perf:
        perf_lines = [
            "",
            "Performance",
            f"• cycles:  {perf.get('cycle_count', '-')}",
            f"• score:   {perf.get('score_cycles', '-')}",
            f"• softmax: {perf.get('softmax_cycles', '-')}",
            f"• wsum:    {perf.get('wsum_cycles', '-')}",
            f"• loads:   {perf.get('load_events', '-')}",
        ]

    info_lines = [
        "How it works",
        "",
        "1. Each beat is one token",
        "2. FPGA compares current beat",
        "   against all legal past beats",
        "3. Future beats are blocked",
        "4. Softmax creates attention weights",
        "5. Weighted sum produces the output",
        "",
        "Current frame",
        f"• query beat: {q}",
        f"• allowed past range: 0..{q}",
        f"• dominant attended type: {type_name}",
        "",
        "Panels",
        "• top: rhythm tokens",
        "• left: raw score matrix",
        "• middle: causal attention",
        "• right: current-row weights",
        "• bottom: output vector + dominant type",
    ] + perf_lines

    info_text.set_text("\n".join(info_lines))

    return [query_marker, raw_row_line, mask_row_line, diag_line, token_title, query_text, tile_box, info_text] + list(bars) + list(output_bars)

ani = animation.FuncAnimation(
    fig,
    update,
    frames=num_tokens,
    interval=1000 / FPS,
    blit=False,
    repeat=True
)

# =========================
# Save / display
# =========================
if SAVE_PATH is not None:
    save_path = Path(SAVE_PATH)
    suffix = save_path.suffix.lower()
    if suffix == ".gif":
        ani.save(save_path, writer=animation.PillowWriter(fps=FPS))
    elif suffix == ".mp4":
        ani.save(save_path, writer=animation.FFMpegWriter(fps=FPS))
    else:
        raise ValueError("SAVE_PATH must end with .gif or .mp4")
    print(f"Saved animation to: {save_path.resolve()}")

plt.close(fig)
display(HTML(ani.to_jshtml()))
