import os
import math
import string
import secrets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import librosa
import librosa.display

# --- GLOBAL CONFIGURATION ---
DOCUMENT_TEXT_WIDTH_PT = 469.75499
DOCUMENT_FONT_SIZE = 12

PLOT_CFG = {
    'primary': "#1f77b4",
    'secondary': "#ff7f0e",
    'tertiary': "#cccccc",
    'highlight': "#9467bd",
    'gray_bar': "#999999",
    'scatter_size': 40,
    'grid_alpha': 0.18
}

def random_suffix(length=8):
    alphabet = string.ascii_lowercase + string.digits
    return ''.join(secrets.choice(alphabet) for _ in range(length))

def set_report_style(width_pt, font_size=10, aspect_ratio=0.618, dpi=200):
    inches_per_pt = 1 / 72.27
    fig_width_in = width_pt * inches_per_pt
    fig_height_in = fig_width_in * aspect_ratio

    plt.rcParams.update({
        "text.usetex": True,
        "figure.figsize": (fig_width_in, fig_height_in),
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 2,
        "ytick.labelsize": font_size - 2,
        "legend.fontsize": font_size - 2,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
    })


def save_figure(fig, prefix, ext="pdf"):
    os.makedirs("./Figures", exist_ok=True)
    filepath = f"./Figures/{prefix}_{random_suffix()}.{ext}"
    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.05)
    print(f"Saved {filepath}")
    plt.close(fig)


# --- MATH & SIGNAL UTILS ---
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)


def draw_vertical_interval(ax, x, y0, y1, label, text_dx=-10, text_dy=0, color="#4d4d4d", lw=1.0, zorder=6):
    ax.annotate("", xy=(x, y0), xytext=(x, y1),
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=8),
                zorder=zorder)
    ax.annotate(label, xy=(x, 0.5 * (y0 + y1)), xytext=(text_dx, text_dy), textcoords="offset points", ha="right",
                va="center", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.15),
                zorder=zorder + 1)


def draw_horizontal_interval(ax, x0, x1, y, label, text_dy=5, color="#4d4d4d", lw=1.0, zorder=6):
    ax.annotate("", xy=(x0, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle="<->", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=8),
                zorder=zorder)
    ax.annotate(label, xy=(0.5 * (x0 + x1), y), xytext=(0, text_dy), textcoords="offset points", ha="center",
                va="bottom", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.15),
                zorder=zorder + 1)


def add_zero_order_hold(ax, ts, fs, x, duration, label_xn=False):
    dt = 1 / fs
    end_time = max(ts[-1] + dt, duration)
    ts_step = np.append(ts, end_time)
    x_step = np.append(x, x[-1])
    step_label = r"$x[n]$" if label_xn else None
    ax.step(ts_step, x_step, where="post", lw=2.5, color=PLOT_CFG['secondary'], zorder=2, label=step_label)


# --- FIGURE GENERATORS ---

def generate_figure_1_sampling():
    set_report_style(DOCUMENT_TEXT_WIDTH_PT, font_size=DOCUMENT_FONT_SIZE, aspect_ratio=0.4)

    fs = 10
    duration = 1.5
    N = int(duration * fs)
    phi1, phi2, phi3 = np.deg2rad(40), np.deg2rad(-70), np.deg2rad(110)

    # Discrete
    n = np.arange(N)
    ts = n / fs
    env_n = 0.72 + 0.22 * np.cos(2 * np.pi * 0.55 * ts - 0.4)
    x = env_n * np.cos(2 * np.pi * 7.2 * ts + phi1) + 0.20 * np.cos(2 * np.pi * (2.4 * ts + 3.0 * ts ** 2) + phi2)

    # Continuous
    t = np.linspace(0, duration, 200000)
    env_t = 0.72 + 0.22 * np.cos(2 * np.pi * 0.55 * t - 0.4)
    xc = env_t * np.cos(2 * np.pi * 7.2 * t + phi1) + 0.20 * np.cos(2 * np.pi * (2.4 * t + 3.0 * t ** 2) + phi2)

    # Scale
    scale = max(np.max(np.abs(x)), np.max(np.abs(xc)))
    amp = 0.7
    x = (0.92 * x / scale) * amp
    xc = (0.92 * xc / scale) * amp

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, xc, lw=1.8, color=PLOT_CFG['primary'], alpha=1, label=r"$x(t)$", zorder=1)
    ax.scatter(ts, x, s=22, color=PLOT_CFG['secondary'], edgecolor="white", label=r"$x[n]$", zorder=3, linewidth=0.6)
    ax.vlines(ts, -1, 1, colors=PLOT_CFG['tertiary'], linestyles="dashed", alpha=0.7, label="Sample times", zorder=-1)

    ax.set_xlim(0.0, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Sampling a Continuous-Time Signal $x(t)$")
    ax.grid(False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))

    # Annotation
    k = 3
    x1, x2 = ts[k], ts[k + 1]
    yl = ax.get_ylim()
    y_bracket = yl[1] - 0.92 * (yl[1] - yl[0])
    ax.annotate("", xy=(x1, y_bracket), xytext=(x2, y_bracket),
                arrowprops=dict(arrowstyle="<->", lw=0.7, color="black"))
    tick_h = 0.04 * (yl[1] - yl[0])
    ax.vlines([x1, x2], y_bracket - tick_h / 2, y_bracket + tick_h / 2, colors="gray", lw=1)
    ax.axhline(0, color="black", linewidth=0.6, alpha=0.2, zorder=0)
    ax.text((x1 + x2) / 2, y_bracket + 0.03 * (yl[1] - yl[0]), r"$t_s$", ha="center", va="bottom", fontsize=10,
            color="black")

    plt.tight_layout()
    save_figure(fig, "sampling_figure")


def generate_figure_2_aliasing():
    set_report_style(DOCUMENT_TEXT_WIDTH_PT, font_size=DOCUMENT_FONT_SIZE, aspect_ratio=0.78)

    fs = 5.0
    duration = 2.0
    f_base = 1.0
    frequencies = [1, 6, 11, 16]

    ts = np.arange(0, duration + 1 / fs, 1 / fs)
    xn = np.cos(2 * np.pi * f_base * ts)
    t = np.linspace(0, duration, 8000)

    fig, axes = plt.subplots(len(frequencies), 1, sharex=True)
    fig.suptitle("Aliasing: Different Continuous Signals Producing Identical Samples \n", y=0.965, fontsize=12)

    for i, (ax, f) in enumerate(zip(axes, frequencies)):
        xt = np.cos(2 * np.pi * f * t)
        ax.plot(t, xt, lw=2.1, color=PLOT_CFG['primary'], alpha=0.8, zorder=1)
        ax.scatter(ts, xn, s=PLOT_CFG['scatter_size'], color=PLOT_CFG['secondary'], edgecolor="white", zorder=3,
                   linewidth=1.2)
        add_zero_order_hold(ax, ts, fs, xn, duration, label_xn=(i == 0))

        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.tick_params(axis='both', which='both', length=0)
        ax.grid(True, alpha=PLOT_CFG['grid_alpha'], color=PLOT_CFG['tertiary'])
        ax.set_ylim(-1.08, 1.08)
        ax.set_yticks([-1, 0, 1])
        ax.set_title(rf"$q={f}\,\mathrm{{Hz}}$", loc="left", pad=2, fontsize=11)

    custom_lines = [
        Line2D([0], [0], color=PLOT_CFG['primary'], lw=2.2, alpha=0.8),
        Line2D([0], [0], color=PLOT_CFG['secondary'], lw=2.2, alpha=0.8),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=PLOT_CFG['secondary'], markeredgecolor='white',
               markersize=8, markeredgewidth=1.4)
    ]
    fig.legend(custom_lines, [r'$x(t)$', r'$x[n]$'], loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=2,
               framealpha=0.9)
    axes[-1].set_xlabel("Time (seconds)")
    plt.xlim(0, duration)
    fig.supylabel('Amplitude', x=0.04)
    plt.subplots_adjust(top=0.9, hspace=0.22)
    save_figure(fig, "aliasing_fig")


def generate_figure_3_stft_partitioning():
    set_report_style(DOCUMENT_TEXT_WIDTH_PT, font_size=DOCUMENT_FONT_SIZE, aspect_ratio=0.6)

    fs = 15.0
    duration = 2.0
    N_total = int(duration * fs) + 1
    ts = np.linspace(0, duration, N_total)
    xn = np.sin(2 * np.pi * 1.2 * ts) + 0.4 * np.cos(2 * np.pi * 3.5 * ts)

    t = np.linspace(0, duration, 1000)
    xt = np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.cos(2 * np.pi * 3.5 * t)

    L, H = 12, 4
    M = 1 + (N_total - L) // H
    m_hl = 1
    start_idx, end_idx = m_hl * H, m_hl * H + L

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [2.5, 1.5]})
    fig.subplots_adjust(top=0.78, hspace=0.1)

    # Top Plot
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.grid(True, alpha=0.3, color=PLOT_CFG['tertiary'], linestyle='--')
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    ax1.plot(t, xt, lw=2, color=PLOT_CFG['primary'], alpha=0.7, label=r"$x(t)$", zorder=1)

    ax1.scatter(ts, xn, s=25, linewidth=0.8, color=PLOT_CFG['secondary'], edgecolor="white", zorder=3, label=r"$x[n]$")

    ax1.scatter(ts[start_idx:end_idx], xn[start_idx:end_idx], s=45, linewidth=1,
                color=PLOT_CFG['highlight'], edgecolor="white", zorder=4, label=fr"Segment $m={m_hl}$")

    ax1.axvspan(ts[start_idx], ts[end_idx - 1], color=PLOT_CFG['highlight'], alpha=0.1, zorder=0)
    ax1.set_ylabel("Amplitude", fontsize=12)

    fig.suptitle("STFT Signal Partitioning", y=1.02, fontsize=15)
    # ax1.set_title(fr"$N={N_total}$ total samples $\quad \vert \quad$ $L={L}$ samples per segment $\quad \vert \quad$ $H={H}$ hop size", pad=40, fontsize=12)

    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=11)

    # Bottom Plot
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.grid(True, alpha=0.3, color=PLOT_CFG['tertiary'], axis='x', linestyle='--')

    num_frames = min(M, 6)
    for m in range(num_frames):
        n_start, n_end = m * H, m * H + L - 1
        t_start, t_end = ts[n_start], ts[n_end]
        color = PLOT_CFG['highlight'] if m == m_hl else PLOT_CFG['gray_bar']
        alpha = 0.9 if m == m_hl else 0.5

        ax2.hlines(y=-m, xmin=t_start, xmax=t_end, color=color, alpha=alpha, linewidth=6, zorder=2, capstyle='butt')
        if m == m_hl:
            ax2.text(t_end + 0.03, -m, r"$\leftarrow x[n + mH]$", va='center', ha='left',
                     color=color, fontsize=11)

    # Dimension lines using the '|-|' arrowstyle
    def draw_dim_line(ax, x1, x2, y, text, text_offset=0.25):
        ax.annotate("", xy=(x1, y), xytext=(x2, y),
                    arrowprops=dict(arrowstyle="|-|", color="#444", lw=1.2, mutation_scale=5))

        va = 'bottom' if text_offset > 0 else 'top'
        ax.text((x1 + x2) / 2, y + text_offset, text, ha='center', va=va, fontsize=11, color="#111")

    # L is drawn above the first frame
    length_correction = 0.009
    draw_dim_line(ax2, ts[0] - length_correction, ts[L - 1] + length_correction, y=0.8, text=fr"$L = {L}$", text_offset=0.15)

    # H is drawn in the gap between the start of m=0 and m=1
    draw_dim_line(ax2, ts[0] - length_correction, ts[H] + length_correction, y=-0.8, text=fr"$H = {H}$", text_offset=-0.20)

    # Configure the left axis ticks
    ax2.set_yticks([-m for m in range(num_frames)])
    ax2.set_yticklabels([f"{m}" for m in range(num_frames)], fontsize=10)
    ax2.set_ylabel("Segment Index ($m$)", fontsize=11)
    ax2.set_ylim(-5.5, 1.5)
    ax2.set_xlabel("Time (seconds)")

    plt.xlim(-0.15, duration + 0.05)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, hspace=0.05)
    save_figure(fig, "stft_partitioning", ext="pdf")


def generate_spectrogram_figure(y, sr, title, filename_prefix):
    set_report_style(DOCUMENT_TEXT_WIDTH_PT, font_size=DOCUMENT_FONT_SIZE, aspect_ratio=0.5, dpi=300)

    hop_len, n_fft, win_len, window_type = 64, 8192, 2048, "hann"
    stft_matrix = librosa.stft(y, n_fft=n_fft, hop_length=hop_len, window=window_type, win_length=win_len)
    spectrogram_db = librosa.amplitude_to_db(np.abs(stft_matrix), ref=np.max)

    fig, ax = plt.subplots(1, 1)
    img = librosa.display.specshow(
        spectrogram_db, sr=sr, hop_length=hop_len, x_axis='time', y_axis='log',
        cmap="inferno", vmin=-70, vmax=0, rasterized=True, ax=ax
    )

    fig.colorbar(img, ax=ax, format="%+2.0f dB", label='Level (dB)')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(title)
    plt.tight_layout()
    save_figure(fig, filename_prefix)


def generate_figure_6_mel_mapping():
    set_report_style(DOCUMENT_TEXT_WIDTH_PT, font_size=DOCUMENT_FONT_SIZE, aspect_ratio=0.6, dpi=250)
    plt.style.use("seaborn-v0_8-whitegrid")

    sample_rate = 44100
    L, B = 2048, 20
    q_min, q_max = 0.0, sample_rate / 2.0

    mel_min, mel_max = hz_to_mel(q_min), hz_to_mel(q_max)
    mel_points = np.linspace(mel_min, mel_max, B + 2)
    hz_points = mel_to_hz(mel_points)

    fig, ax = plt.subplots()
    freq_axis_fine = np.linspace(q_min, q_max, 2000)
    mel_axis_fine = hz_to_mel(freq_axis_fine)

    i_mid = np.ceil(B // 1.5).astype(int)
    idx_triplet = [i_mid - 1, i_mid, i_mid + 1]

    for j in range(len(mel_points)):
        ax.hlines(y=mel_points[j], xmin=q_min, xmax=hz_points[j], color="#d9d9d9", linestyle="--", linewidth=0.8,
                  alpha=0.65, zorder=1)
        ax.vlines(x=hz_points[j], ymin=mel_min, ymax=mel_points[j], color="#d9d9d9", linestyle="--", linewidth=0.8,
                  alpha=0.65, zorder=1)

    ax.plot(freq_axis_fine, mel_axis_fine, color=PLOT_CFG['primary'], linewidth=2.2, label="Mel scale function",
            zorder=3)
    ax.scatter(hz_points, mel_points, s=26, color=PLOT_CFG['secondary'], edgecolor="white", linewidth=0.7, zorder=4)

    x_dm = hz_points[idx_triplet[0]] - 2500
    for i in [-2, -3, 4, 5]:
        offset = i_mid + i
        ax.hlines(y=mel_points[offset], xmin=q_min, xmax=hz_points[offset], color="#9e9e9e", linestyle="--",
                  linewidth=1.0, alpha=0.85, zorder=2)
        ax.vlines(x=hz_points[offset], ymin=mel_min, ymax=mel_points[offset], color="#9e9e9e", linestyle="--",
                  linewidth=1.0, alpha=0.85, zorder=2)
        ax.scatter(hz_points[offset], mel_points[offset], s=42, color=PLOT_CFG['highlight'], edgecolor="white",
                   linewidth=0.9, zorder=5)

    draw_vertical_interval(ax, x=x_dm, y0=mel_points[10], y1=mel_points[11], label=r"$\Delta \lambda_a$")
    draw_vertical_interval(ax, x=x_dm, y0=mel_points[17], y1=mel_points[18], label=r"$\Delta \lambda_b$")

    y_dq = mel_min + 220
    draw_horizontal_interval(ax, x0=hz_points[10], x1=hz_points[11], y=y_dq, label=r"$\Delta q_a$")
    draw_horizontal_interval(ax, x0=hz_points[17], x1=hz_points[18], y=y_dq, label=r"$\Delta q_b$")

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Mel Scale")
    ax.set_title("Mel Scale to Frequency Mapping")
    ax.set_ylim([mel_min, mel_max])
    ax.set_xlim([q_min, q_max])
    ax.grid(False)

    legend_handles = [
        Line2D([0], [0], color=PLOT_CFG['primary'], lw=2.2, label=r"$\lambda(q)$"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=PLOT_CFG['secondary'], markeredgecolor="white",
               markersize=6, label=r"$(q_i, \lambda_i)$")
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.02, 1), framealpha=0.95, borderaxespad=0.2)

    plt.tight_layout()
    save_figure(fig, "mel_scale_mapping")


def generate_figure_7_mel_filterbank():
    set_report_style(DOCUMENT_TEXT_WIDTH_PT, font_size=DOCUMENT_FONT_SIZE, aspect_ratio=0.6, dpi=250)
    plt.style.use("seaborn-v0_8-whitegrid")

    sample_rate = 44100
    L, B = 2048, 10
    q_min, q_max = 0.0, sample_rate / 2.0

    mel_min, mel_max = hz_to_mel(q_min), hz_to_mel(q_max)
    mel_points = np.linspace(mel_min, mel_max, B + 2)
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.floor((L + 1) * hz_points / sample_rate).astype(int)

    filters = np.zeros((B, int(L / 2 + 1)))
    for i in range(B):
        left, center, right = fft_bins[i], fft_bins[i + 1], fft_bins[i + 2]
        if center - left != 0:
            filters[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right - center != 0:
            filters[i, center:right] = (right - np.arange(center, right)) / (right - center)

    fig, ax = plt.subplots()
    fft_freqs = np.linspace(q_min, q_max, int(L / 2 + 1))

    highlight_i = math.ceil(B // 1.5)
    highlight_i_plus_2 = highlight_i + 2

    for i in range(B):
        color = PLOT_CFG['primary'] if i == highlight_i else (
            PLOT_CFG['highlight'] if i == highlight_i_plus_2 else "#c7c7c7")
        lw, alpha = (2.8, 1.0) if i in [highlight_i, highlight_i_plus_2] else (1.1, 0.35)
        zord = 4 if i in [highlight_i, highlight_i_plus_2] else 1
        line, = ax.plot(fft_freqs, filters[i, :], linewidth=lw, color=color, alpha=alpha, zorder=zord)
        if i == highlight_i:
            highlight_line = line

    for j in range(B + 2):
        ax.axvline(hz_points[j], color="#d9d9d9", linestyle="--", linewidth=0.8, alpha=0.22, zorder=0)

    x_left, x_center, x_right = hz_points[highlight_i], hz_points[highlight_i + 1], hz_points[highlight_i + 2]

    # Red interval showing base width
    y_dq = -0.005
    draw_horizontal_interval(ax, x_center, x_right, y_dq, "")
    ax.text(0.56 * (x_left + x_right), y_dq + 0.04, r"$\Delta q_i$", color="black", fontsize=11, ha="center",
            va="bottom", zorder=8, bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.075))

    # Fix: Replaced undefined arrays with explicit declaration corresponding to the left, center, and right vertices of the highlighted triangle
    vertex_x = [x_left, x_center, x_right]
    vertex_y = [0, 1, 0]
    vertex_labels = [r"$c_{i-1}$", r"$c_i$", r"$c_{i+1}$"]
    vertex_offsets = [(-10, 10), (0, 15), (10, 10)]
    vertex_has = ["right", "center", "left"]
    vertex_vas = ["bottom", "bottom", "bottom"]

    for x, y, lab, off, ha, va in zip(vertex_x, vertex_y, vertex_labels, vertex_offsets, vertex_has, vertex_vas):
        ax.scatter(x, y, s=38, color=PLOT_CFG['secondary'], edgecolor="white", linewidth=0.8, zorder=5)
        ax.annotate(lab, xy=(x, y), xytext=off, textcoords="offset points", fontsize=9, ha=ha, va=va,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=0.12),
                    arrowprops=dict(arrowstyle="->", color="#555555", lw=0.8, shrinkA=2, shrinkB=3, alpha=0), zorder=6)

    ax.annotate(rf"$B={B}$ triangular filters", xy=(0.985, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=0.15), zorder=7)

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Filter Weight")
    ax.set_title("Mel Filterbank")
    ax.set_xlim([q_min, q_max])
    ax.set_ylim([-0.02, 1.05])
    ax.grid(False)
    ax.legend([highlight_line], [r"$\Lambda_i (k)$"], loc="upper left", framealpha=0.9)

    # Secondary Axis
    def hz_to_bin(q):
        return (L + 1) * q / sample_rate

    def bin_to_hz(c):
        return c * sample_rate / (L + 1)

    base_center_idx = highlight_i + 1
    valid_offsets, tick_bins = [], []
    for off in [-2, -1, 0, 1, 2, 3]:
        if 1 <= base_center_idx + off <= B:
            valid_offsets.append(off)
            tick_bins.append(fft_bins[base_center_idx + off])

    tick_labels = [r"$c_i$" if off == 0 else (rf"$c_{{i+{off}}}$" if off > 0 else rf"$c_{{i{off}}}$") for off in
                   valid_offsets]

    secax = ax.secondary_xaxis("top", functions=(hz_to_bin, bin_to_hz))
    secax.set_xlabel("Frequency Bin ($k$)")
    secax.set_xticks(tick_bins, labels=tick_labels)
    secax.tick_params(axis="x", which="major", length=0, pad=6)

    plt.tight_layout()
    save_figure(fig, "mel_filterbank")


def generate_figure_8_gibbs():
    set_report_style(DOCUMENT_TEXT_WIDTH_PT, font_size=DOCUMENT_FONT_SIZE, aspect_ratio=0.5)

    # 1. Generate a segment where the start and end values do not match
    N = 1000
    t_seg = np.linspace(0, 1, N, endpoint=False)

    # cos(pi * t) starts at 1 and ends at -1.
    # This guarantees a maximum jump discontinuity at the segment boundaries upon periodic extension.
    ideal_seg = np.cos(np.pi * t_seg)

    # 2. Compute the DFT of the segment
    X = np.fft.fft(ideal_seg)

    # 3. Reconstruct the continuous signal over an extended range to show periodic boundary ringing
    t_ext = np.linspace(-0.25, 1.25, 1500)
    K_high = 30
    recon_high = np.zeros_like(t_ext, dtype=complex)

    for k in range(-K_high, K_high + 1):
        # Handle standard FFT coefficient packing
        X_k = X[k] if k >= 0 else X[N + k]
        recon_high += (X_k / N) * np.exp(1j * 2 * np.pi * k * t_ext)

    recon_high = recon_high.real

    # 4. Plotting
    fig, ax = plt.subplots(1, 1)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=PLOT_CFG['grid_alpha'], color=PLOT_CFG['tertiary'], linestyle='--')

    # Plot the periodic extensions (dashed) to visually explain what the DFT "sees"
    t_left = np.linspace(-0.25, 0, 250, endpoint=False)
    t_right = np.linspace(1, 1.25, 250, endpoint=False)
    ax.plot(t_left, np.cos(np.pi * (t_left % 1.0)), color=PLOT_CFG['primary'], lw=1.5, linestyle='--', alpha=0.4,
            zorder=1)
    ax.plot(t_right, np.cos(np.pi * (t_right % 1.0)), color=PLOT_CFG['primary'], lw=1.5, linestyle='--', alpha=0.4,
            zorder=1, label="Periodic Extension")

    # Plot the original analyzed segment (solid)
    ax.plot(t_seg, ideal_seg, color=PLOT_CFG['primary'], lw=2.0, label="Original Segment", zorder=3)

    # Vertical dotted lines to emphasize the jump exactly at the boundaries
    ax.vlines([0, 1], -1, 1, color='black', linestyle=':', lw=1.5, alpha=0.5, zorder=2)

    # Plot the Fourier approximation showing the Gibbs ringing at the boundary
    ax.plot(t_ext, recon_high, color=PLOT_CFG['secondary'], lw=1.5, alpha=0.9,
            label=rf"Fourier Approximation ($K={K_high}$)", zorder=4)

    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-1.35, 1.35)
    ax.set_xlabel("Time (normalized segment)")
    ax.set_ylabel("Amplitude")
    ax.set_title("The Gibbs Phenomenon at Segment Boundaries", pad=10)
    ax.legend(loc="upper right", ncol=1, framealpha=0.95, edgecolor=PLOT_CFG['tertiary'])

    plt.tight_layout()
    save_figure(fig, "gibbs_phenomenon")


if __name__ == "__main__":
    generate_figure_3_stft_partitioning()


# if __name__ == "__main__":
#     generate_figure_1_sampling()
#     generate_figure_2_aliasing()
#     generate_figure_3_stft_partitioning()
#
#     # Executing the unified Spectrogram function for Figures 4 and 5
#     sample_rate_noise = 44100
#     white_noise = np.random.randn(int(sample_rate_noise * 10.0))
#     generate_spectrogram_figure(white_noise, sample_rate_noise, 'Spectrogram of White Noise', 'white_noise_spectrogram')
#
#     # Provide your loaded audio array below to reproduce Figure 5
#     # generate_spectrogram_figure(awol_food_song, sample_rate_noise, 'Spectrogram of a 10-second Audio Excerpt', 'awol_food_spectrogram')
#
#     generate_figure_6_mel_mapping()
#     generate_figure_7_mel_filterbank()

