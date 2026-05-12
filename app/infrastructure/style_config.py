"""Centralized visualization and coding style configuration.

This module owns:
  PLOT_RCPARAMS   — matplotlib rcParams applied globally
  COLOR_PALETTE   — engineering-friendly, colorblind-safe color system
  ANALYSIS_COLORS — semantic color map (good/warn/bad/neutral)
  STYLE_PREAMBLE  — Python code string injected before every code execution

The preamble is imported by code_runner.py so every plot automatically
follows the project's visual standard without the agent needing to
configure it manually.
"""
from __future__ import annotations

# ──────────────────────────────────────────────────────────────────── #
# matplotlib rcParams standard                                          #
# ──────────────────────────────────────────────────────────────────── #

PLOT_RCPARAMS: dict[str, object] = {
    # Figure
    "figure.figsize": (12, 7),
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "figure.autolayout": False,         # we call tight_layout() explicitly
    # Axes
    "axes.facecolor": "#F8F9FA",
    "axes.grid": True,
    "axes.grid.alpha": 0.55,
    "axes.grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "axes.labelsize": 12,
    "axes.labelpad": 8,
    "axes.axisbelow": True,             # grid behind data
    # Ticks
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.direction": "out",
    "ytick.direction": "out",
    # Legend
    "legend.fontsize": 10,
    "legend.framealpha": 0.85,
    "legend.edgecolor": "#CCCCCC",
    "legend.loc": "best",
    # Lines
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
    "lines.antialiased": True,
    # Patches
    "patch.linewidth": 0.5,
    # Font
    "font.family": "sans-serif",
    "font.size": 11,
    # Saving
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "savefig.dpi": 150,
}

# ──────────────────────────────────────────────────────────────────── #
# Color system                                                          #
# ──────────────────────────────────────────────────────────────────── #

# Primary sequential palette — works on screen and in print, colorblind-safe
COLOR_PALETTE: list[str] = [
    "#2196F3",   # blue       — primary series
    "#FF5722",   # deep-orange — secondary series
    "#4CAF50",   # green      — tertiary / positive
    "#9C27B0",   # purple     — quaternary
    "#009688",   # teal       — quinary
    "#FFC107",   # amber      — warning / highlight
    "#795548",   # brown      — background series
    "#607D8B",   # blue-grey  — reference / baseline
]

# Semantic colors for engineering dashboards
ANALYSIS_COLORS: dict[str, str] = {
    "good":      "#4CAF50",   # green   — within spec
    "warn":      "#FFC107",   # amber   — approaching limit
    "bad":       "#F44336",   # red     — out of spec / violation
    "neutral":   "#9E9E9E",   # grey    — reference / target
    "highlight": "#2196F3",   # blue    — selected / focus
}

# ──────────────────────────────────────────────────────────────────── #
# Style preamble — injected before every execute_python_code call      #
# ──────────────────────────────────────────────────────────────────── #

# Written as a raw string to avoid f-string conflicts; the single
# placeholder {datasets_dir!r} is filled in by code_runner.py at runtime.
STYLE_PREAMBLE: str = r'''
# ── Auto-injected style configuration ────────────────────────────────
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
import seaborn as sns
import json, base64, io, sys, os
from pathlib import Path

# ── Global matplotlib style ───────────────────────────────────────────
plt.rcParams.update({
    'figure.figsize':       (12, 7),
    'figure.dpi':           150,
    'figure.facecolor':     'white',
    'axes.facecolor':       '#F8F9FA',
    'axes.grid':            True,
    'axes.grid.alpha':      0.55,
    'axes.grid.linestyle':  '--',
    'axes.spines.top':      False,
    'axes.spines.right':    False,
    'axes.titlesize':       14,
    'axes.titleweight':     'bold',
    'axes.titlepad':        14,
    'axes.labelsize':       12,
    'axes.labelpad':        8,
    'axes.axisbelow':       True,
    'xtick.labelsize':      10,
    'ytick.labelsize':      10,
    'legend.fontsize':      10,
    'legend.framealpha':    0.85,
    'legend.edgecolor':     '#CCCCCC',
    'lines.linewidth':      2.2,
    'lines.markersize':     6,
    'font.family':          'sans-serif',
    'font.size':            11,
    'savefig.bbox':         'tight',
    'savefig.facecolor':    'white',
    'savefig.dpi':          150,
})

# ── Color palette (engineering-safe, colorblind-friendly) ────────────
COLORS = {
    'blue':        '#2196F3',
    'orange':      '#FF5722',
    'green':       '#4CAF50',
    'purple':      '#9C27B0',
    'teal':        '#009688',
    'amber':       '#FFC107',
    'brown':       '#795548',
    'bluegrey':    '#607D8B',
}
PALETTE = list(COLORS.values())

# Semantic colors
C_GOOD    = '#4CAF50'   # within spec
C_WARN    = '#FFC107'   # approaching limit
C_BAD     = '#F44336'   # out of spec / violation
C_NEUTRAL = '#9E9E9E'   # reference / target
C_PRIMARY = '#2196F3'   # highlight / focus

sns.set_theme(style='whitegrid', palette=PALETTE, font_scale=1.05)

# ── Figure helpers ────────────────────────────────────────────────────

def label_bars(ax, fmt='{:.1f}', fontsize=9, pad=3):
    """Add value labels on top of each bar in a bar chart."""
    for p in ax.patches:
        h = p.get_height()
        if h == h:   # not NaN
            ax.annotate(
                fmt.format(h),
                (p.get_x() + p.get_width() / 2., h),
                ha='center', va='bottom',
                fontsize=fontsize,
                xytext=(0, pad), textcoords='offset points',
            )

def add_reference_line(ax, value, label='', color=C_NEUTRAL, linestyle='--'):
    """Draw a horizontal reference/limit line with label."""
    ax.axhline(value, color=color, linestyle=linestyle, linewidth=1.5, zorder=2)
    if label:
        ax.text(
            ax.get_xlim()[1], value, f'  {label}',
            va='center', ha='left', fontsize=9, color=color,
        )

def format_axis_units(ax, xlabel='', ylabel='', title=''):
    """Set axis labels (with units in brackets) and title, then tight_layout."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return ax

def engineering_plot(nrows=1, ncols=1, figsize=None, title=''):
    """Create a pre-styled figure. Returns (fig, ax) or (fig, axes_array)."""
    if figsize is None:
        figsize = (12, 7) if ncols == 1 else (6 * ncols, 5 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor='white')
    if title:
        fig.suptitle(title, fontsize=15, fontweight='bold', y=1.01)
    plt.tight_layout()
    return fig, axes

# ── Figure capture state ──────────────────────────────────────────────
_FIGURES = {}
_FIGURE_COUNTER = [0]
_DATASETS_DIR = {datasets_dir!r}

_original_show = plt.show

def _capture_show(*args, **kwargs):
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    fid = f"fig_{_FIGURE_COUNTER[0]:03d}"
    _FIGURES[fid] = b64
    _FIGURE_COUNTER[0] += 1
    plt.close('all')
    print(f"[Figure captured: {fid}]")

plt.show = _capture_show
# ─────────────────────────────────────────────────────────────────────
'''
