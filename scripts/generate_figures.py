"""
Generate competition result visualization for MALTO hackathon (2nd place).
Run: python generate_figures.py
Output: figures/competition_results.png
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import os

os.makedirs("figures", exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
SCORE_HISTORY = [
    ("TF-IDF + LinearSVC\nbaseline",           0.84123, "#c0c0c0"),
    ("DeBERTa 5-fold\n(transformer only)",     0.91648, "#c0c0c0"),
    ("Multi-model\nweighted vote",              0.92170, "#c0c0c0"),
    ("ModernBERT + LDAM\n+ DRW (3-fold)",      0.94120, "#c0c0c0"),
    ("ModernBERT + SVC\nensemble (5-fold)",    0.95341, "#2563EB"),
    ("Final submission\n(2nd place)",           0.95919, "#16A34A"),
]

PER_CLASS = {
    "Human":    (1.00, 1.00, 1.00),
    "ChatGPT":  (1.00, 1.00, 1.00),
    "Gemini":   (0.99, 1.00, 0.99),
    "Claude":   (1.00, 1.00, 1.00),
    "Grok":     (0.92, 0.92, 0.92),
    "DeepSeek": (0.85, 0.82, 0.84),
}

LEADERBOARD = [
    ("1st place", 0.96423, "#F59E0B"),
    ("2nd place\n(ours)", 0.95919, "#16A34A"),
    ("3rd place", 0.94500, "#c0c0c0"),
    ("4th place", 0.93100, "#c0c0c0"),
    ("5th place", 0.91800, "#c0c0c0"),
]

# ── Figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10), facecolor="#0F172A")
fig.patch.set_facecolor("#0F172A")

gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                       left=0.07, right=0.96, top=0.88, bottom=0.10)

ax_prog  = fig.add_subplot(gs[0, :])   # full width — score progression
ax_class = fig.add_subplot(gs[1, 0])   # per-class F1
ax_lb    = fig.add_subplot(gs[1, 1])   # leaderboard snapshot

PANEL_BG = "#1E293B"
GRID_CLR  = "#334155"
TEXT_CLR  = "#E2E8F0"
ACCENT    = "#2563EB"
GREEN     = "#16A34A"
GOLD      = "#F59E0B"

def style_ax(ax, title):
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.tick_params(colors=TEXT_CLR, labelsize=9)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.set_title(title, color=TEXT_CLR, fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", color=GRID_CLR, linewidth=0.6, alpha=0.6)

# ── 1. Score Progression ──────────────────────────────────────────────────────
labels  = [s[0] for s in SCORE_HISTORY]
scores  = [s[1] for s in SCORE_HISTORY]
colors  = [s[2] for s in SCORE_HISTORY]
x       = np.arange(len(labels))

bars = ax_prog.bar(x, scores, color=colors, width=0.6, zorder=3, edgecolor="#0F172A", linewidth=0.5)
# score labels on bars
for bar, score in zip(bars, scores):
    ax_prog.text(bar.get_x() + bar.get_width() / 2, score + 0.002,
                 f"{score:.5f}", ha="center", va="bottom",
                 color=TEXT_CLR, fontsize=8, fontweight="bold")

# 1st place reference line
ax_prog.axhline(0.96423, color=GOLD, linestyle="--", linewidth=1.2, zorder=4, alpha=0.85)
ax_prog.text(len(x) - 0.45, 0.9650, "1st place  0.96423",
             color=GOLD, fontsize=8, ha="right", va="bottom")

ax_prog.set_xticks(x)
ax_prog.set_xticklabels(labels, fontsize=8.5, color=TEXT_CLR)
ax_prog.set_ylim(0.80, 0.98)
ax_prog.set_ylabel("Public LB — Macro F1", fontsize=9)
style_ax(ax_prog, "Score Progression")

legend_patches = [
    mpatches.Patch(color="#c0c0c0", label="Intermediate attempts"),
    mpatches.Patch(color=ACCENT,    label="Best model (ModernBERT + SVC)"),
    mpatches.Patch(color=GREEN,     label="Final 2nd-place submission"),
]
ax_prog.legend(handles=legend_patches, loc="lower right",
               facecolor=PANEL_BG, edgecolor=GRID_CLR,
               labelcolor=TEXT_CLR, fontsize=8)

# ── 2. Per-Class F1 ───────────────────────────────────────────────────────────
classes = list(PER_CLASS.keys())
f1s     = [PER_CLASS[c][2] for c in classes]
cls_colors = [GREEN if f1 >= 0.99 else ACCENT if f1 >= 0.90 else "#EF4444" for f1 in f1s]

ypos = np.arange(len(classes))
ax_class.barh(ypos, f1s, color=cls_colors, height=0.55,
              edgecolor="#0F172A", linewidth=0.5, zorder=3)
for i, (f1, c) in enumerate(zip(f1s, cls_colors)):
    ax_class.text(f1 + 0.002, i, f"{f1:.2f}", va="center",
                  color=TEXT_CLR, fontsize=9, fontweight="bold")

ax_class.set_yticks(ypos)
ax_class.set_yticklabels(classes, fontsize=9, color=TEXT_CLR)
ax_class.set_xlim(0.75, 1.05)
ax_class.set_xlabel("F1 Score (OOF)", fontsize=9)
style_ax(ax_class, "Per-Class OOF F1")
ax_class.axvline(1.0, color=GRID_CLR, linewidth=0.8, linestyle=":")

# ── 3. Leaderboard ─────────────────────────────────────────────────────────────
lb_labels = [l[0] for l in LEADERBOARD]
lb_scores = [l[1] for l in LEADERBOARD]
lb_colors = [l[2] for l in LEADERBOARD]

lb_x = np.arange(len(lb_labels))
lb_bars = ax_lb.bar(lb_x, lb_scores, color=lb_colors, width=0.55,
                    edgecolor="#0F172A", linewidth=0.5, zorder=3)
for bar, score in zip(lb_bars, lb_scores):
    ax_lb.text(bar.get_x() + bar.get_width() / 2, score + 0.001,
               f"{score:.5f}", ha="center", va="bottom",
               color=TEXT_CLR, fontsize=8, fontweight="bold")

ax_lb.set_xticks(lb_x)
ax_lb.set_xticklabels(lb_labels, fontsize=8.5, color=TEXT_CLR)
ax_lb.set_ylim(0.88, 0.98)
ax_lb.set_ylabel("Public LB — Macro F1", fontsize=9)
style_ax(ax_lb, "Final Leaderboard (Top 5)")

# ── Title ─────────────────────────────────────────────────────────────────────
fig.text(0.5, 0.95,
         "MALTO Recruitment Hackathon — 2nd Place  |  AI Text Source Detection",
         ha="center", va="top", fontsize=14, fontweight="bold", color=TEXT_CLR)
fig.text(0.5, 0.92,
         "6-class classification: Human · DeepSeek · Grok · Claude · Gemini · ChatGPT  |  Macro F1",
         ha="center", va="top", fontsize=9, color="#94A3B8")

out = "figures/competition_results.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close(fig)
print(f"Saved → {out}")
