"""WABAD reference plots from the supplied spectrogram notebook, cell 1.
Input: audio + reference boxes with optional labels.
Processing: STFT and reference rectangles, coloured by label.
Output: PNG spectrogram and a label legend.
"""

from pathlib import Path
import numpy as np
import librosa, librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SR = None
HOP = 512
N_FFT = 2048
FMAX_HZ = 10000
ZOOM_END = None
SHOW = False


def plot_spectrogram_with_rois(wav_path: Path, rois, out_png: Path, site_name: str):
    """Plots and saves spectrogram with ROI rectangles and species legend."""
    y, sr = librosa.load(str(wav_path), sr=SR, mono=True)

    S_full = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    S_db = librosa.amplitude_to_db(np.abs(S_full), ref=np.max)

    species_list = sorted(set([sp for _, _, _, _, sp in rois]))

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(species_list))))
    color_map = {sp: colors[i % len(colors)] for i, sp in enumerate(species_list)}

    plt.figure(figsize=(15, 6))

    librosa.display.specshow(S_db, sr=sr, hop_length=HOP, x_axis="time", y_axis="hz")

    plt.ylim(0, FMAX_HZ)

    if ZOOM_END is not None:
        plt.xlim(0, ZOOM_END)

    ax = plt.gca()

    for s, e, lf, hf, sp in rois:
        if ZOOM_END is not None and s > ZOOM_END:
            continue

        e_vis = min(e, ZOOM_END) if ZOOM_END is not None else e

        if e_vis <= s:
            continue

        rect = mpatches.Rectangle(
            (s, lf),
            e_vis - s,
            max(0.0, hf - lf),
            facecolor=color_map.get(sp, (1, 0, 0, 0.3)),
            alpha=0.3,
            edgecolor=color_map.get(sp, (1, 0, 0, 1.0)),
            lw=1.0,
        )

        ax.add_patch(rect)

    if species_list:
        legend_patches = [
            mpatches.Patch(color=color_map[sp], label=sp) for sp in species_list
        ]

        plt.legend(
            handles=legend_patches,
            loc="center left",
            bbox_to_anchor=(1, 0.5),
            fontsize=8,
            frameon=True,
        )

    plt.title(f"{site_name} | Spectrogram + Ground Truth ROIs\n{wav_path.name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")

    if SHOW:
        plt.show()

    plt.close()
