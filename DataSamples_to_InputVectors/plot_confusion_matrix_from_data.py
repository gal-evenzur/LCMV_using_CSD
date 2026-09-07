#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Presentation-quality confusion matrix plot for CSD performance evaluation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import matplotlib.patheffects as pe
from pandas import DataFrame


def plot_confusion_matrix_from_data(
    y_test,
    predictions,
    num_classes,
    columns=None,
    annot=True,
    cmap="Blues",
    fmt='.2f',
    fz=13,
    lw=1.5,
    cbar=False,
    figsize=[8, 7],
    show_null_values=1,
    pred_val_axis='lin',
    name='confusion_matrix.png',
    plot_folder=None
    , subtitle=None
):
    """
    Plot a clean, presentation-quality confusion matrix for CSD performance.

    Parameters
    ----------
    y_test          : array-like, true class labels (integer-encoded)
    predictions     : array-like, predicted class labels (integer-encoded)
    num_classes     : int, total number of classes
    columns         : list of str, class label names (optional)
    annot           : bool, whether to annotate cells
    cmap            : str, colormap name (ignored; a custom palette is used)
    fmt             : str, number format string (legacy, kept for API compatibility)
    fz              : int, base font size
    lw              : float, cell border linewidth
    cbar            : bool, show colorbar (legacy, kept for API compatibility)
    figsize         : [w, h], figure size in inches
    show_null_values: int (legacy, kept for API compatibility)
    pred_val_axis   : str, 'lin'/'y' = predicted on y-axis; 'col'/'x' = predicted on x-axis
    name            : str, output filename
    plot_folder     : str, directory to save the plot (created if absent)
    """

    # ── output directory ──────────────────────────────────────────────────────
    if not plot_folder:
        plot_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
    os.makedirs(plot_folder, exist_ok=True)

    y_test      = np.asarray(y_test,      dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    # ── class labels ──────────────────────────────────────────────────────────
    if columns is None:
        from string import ascii_uppercase
        columns = ['Class %s' % c for c in list(ascii_uppercase)[:num_classes]]

    # ── build confusion matrix ────────────────────────────────────────────────
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_test, predictions):
        cm[int(t), int(p)] += 1

    # axis orientation: 'lin'/'y' → predicted on y (rows); 'col'/'x' → predicted on x (cols)
    if pred_val_axis in ('col', 'x'):
        actual_axis, pred_axis = 'x', 'y'        # columns = predicted
    else:
        actual_axis, pred_axis = 'y', 'x'        # rows    = predicted
        cm = cm.T

    # ── metrics ───────────────────────────────────────────────────────────────
    overall_accuracy = np.trace(cm) / cm.sum() * 100
    col_sums = cm.sum(axis=0, keepdims=True)      # avoid /0
    cm_norm  = np.where(col_sums > 0, cm / col_sums, 0.0)   # column-normalised

    # ── colour palette ────────────────────────────────────────────────────────
    BG      = '#F7F9FC'
    PANEL   = '#FFFFFF'
    ACCENT  = '#1A6DB5'          # strong blue
    DIAG    = '#1A6DB5'
    OFFDIAG = '#E8F0FB'          # very light blue tint
    TEXT_LT = '#FFFFFF'
    TEXT_DK = '#1C2B3A'
    BORDER  = '#D0D8E4'
    GRAY_SM = '#6B7C93'

    cmap_custom = LinearSegmentedColormap.from_list(
        'csd_cm', ['#EBF2FC', '#A8C8EE', '#4E99D4', '#1A6DB5', '#0D3F6B'], N=256
    )

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor=BG, dpi=150)
    fig.subplots_adjust(left=0.12, right=0.96, top=0.88, bottom=0.14)
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)

    # ── draw heatmap manually ──────────────────────────────────────────────────
    n = num_classes
    for row in range(n):
        for col in range(n):
            val      = cm[row, col]
            val_norm = cm_norm[row, col]

            if row == col:                          # diagonal
                face = plt.cm.colors if False else None
                color_face = DIAG
                # shade diagonal by intensity
                rgba = cmap_custom(0.35 + 0.65 * val_norm)
                color_face = rgba
                text_color = TEXT_LT
            else:
                # off-diagonal: light tint proportional to value
                rgba = cmap_custom(val_norm * 0.6)
                color_face = rgba
                text_color = TEXT_DK if val_norm < 0.4 else TEXT_LT

            rect = mpatches.FancyBboxPatch(
                (col - 0.48, row - 0.48), 0.96, 0.96,
                boxstyle='round,pad=0.02',
                linewidth=0,
                facecolor=color_face,
                zorder=2
            )
            ax.add_patch(rect)

            if annot and val > 0:
                pct = val_norm * 100
                ax.text(
                    col, row - 0.08,
                    f'{val:,}',
                    ha='center', va='center',
                    fontsize=fz,
                    fontweight='bold',
                    color=text_color,
                    zorder=3
                )
                ax.text(
                    col, row + 0.22,
                    f'{pct:.1f}%',
                    ha='center', va='center',
                    fontsize=fz - 3,
                    color=text_color,
                    alpha=0.85,
                    zorder=3
                )
            elif annot and val == 0:
                ax.text(
                    col, row,
                    '—',
                    ha='center', va='center',
                    fontsize=fz - 2,
                    color=GRAY_SM,
                    alpha=0.5,
                    zorder=3
                )

    # ── grid lines ────────────────────────────────────────────────────────────
    for i in range(n + 1):
        ax.axhline(i - 0.5, color=BORDER, linewidth=lw, zorder=1)
        ax.axvline(i - 0.5, color=BORDER, linewidth=lw, zorder=1)

    # ── axes ticks & labels ───────────────────────────────────────────────────
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)            # invert y so row 0 is at top
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(columns, fontsize=fz - 1, color=TEXT_DK, fontweight='medium')
    ax.set_yticklabels(columns, fontsize=fz - 1, color=TEXT_DK, fontweight='medium')
    ax.tick_params(length=0, pad=8)

    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
        spine.set_linewidth(1.2)

    # ── axis labels ───────────────────────────────────────────────────────────
    if pred_val_axis in ('col', 'x'):
        xlabel, ylabel = 'Predicted label', 'True label'
    else:
        xlabel, ylabel = 'True label', 'Predicted label'

    ax.set_xlabel(xlabel, fontsize=fz, color=TEXT_DK, labelpad=10, fontweight='semibold')
    ax.set_ylabel(ylabel, fontsize=fz, color=TEXT_DK, labelpad=10, fontweight='semibold')

    # ── title block ───────────────────────────────────────────────────────────
    fig.text(
        0.54, 0.955,
        'CSD Confusion Matrix',
        ha='center', va='top',
        fontsize=fz + 3,
        fontweight='bold',
        color=TEXT_DK
    )
    fig.text(
        0.54, 0.925,
        f'Overall Accuracy  {overall_accuracy:.2f}%',
        ha='center', va='top',
        fontsize=fz,
        color=ACCENT,
        fontweight='semibold'
    )

    # Optional subtitle (e.g. SNR and T60 extracted from folder name)
    if subtitle:
        # Place subtitle at the top-left corner (inside left margin)
        fig.text(
            0.12, 0.945,
            str(subtitle),
            ha='left', va='top',
            fontsize=fz - 1,
            color=TEXT_DK,
            fontweight='medium'
        )

    # ── per-class accuracy strip (right side) ─────────────────────────────────
    diag_vals = np.diag(cm_norm)
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(n))
    ax2.set_yticklabels(
        [f'{v*100:.1f}%' for v in diag_vals],
        fontsize=fz - 3,
        color=ACCENT,
        fontweight='bold'
    )
    ax2.tick_params(length=0, pad=6)
    for spine in ax2.spines.values():
        spine.set_visible(False)

    # small right-side header
    fig.text(
        0.965, 0.88 + (0.88 - 0.14) * 0.5 / figsize[1],
        '',
        ha='center', va='center',
        fontsize=fz - 3,
        color=GRAY_SM,
        rotation=90
    )

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    out_path = os.path.join(plot_folder, name)
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
    plt.close(fig)
    print(f'Saved → {out_path}')
    return out_path