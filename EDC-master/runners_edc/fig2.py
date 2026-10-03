import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "stix",
                     "font.serif": ["STIXGeneral", "DejaVu Serif"]})

C = dict(grey=("#EDEDED", "#707070"), green=("#E3F1DC", "#5E9A48"),
         purple=("#E9E0F7", "#8266B8"), red=("#FBE1E1", "#C0504D"),
         yellow=("#FFF3D6", "#C49A2C"), white=("#FFFFFF", "#707070"))

fig, ax = plt.subplots(figsize=(7.16, 3.9))
ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")

def box(x, y, w, h, text, col, fs=9, bold=False):
    fc, ec = C[col]
    ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                 boxstyle="round,pad=0.25,rounding_size=0.9",
                 fc=fc, ec=ec, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.35)

def arrow(x, y0, y1):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#444444",
                                mutation_scale=9))

def panel(cx, title, weights, mid, fused, out, label):
    box(cx, 52.5, 44, 4.6, title, "purple", fs=9, bold=True)
    xs = [cx - 13, cx, cx + 13]
    for i, (x, wtxt) in enumerate(zip(xs, weights)):
        ax.text(x, 48.0, "$p_%d$" % (i + 1), ha="center", va="center", fontsize=10.5)
        box(x, 44.2, 9.5, 3.8, wtxt, "grey" if cx < 50 else "green", fs=10)
        arrow(x, 42.0, 38.6)
    box(cx, 31.8, 44, 12.6, mid, "white", fs=8.4)
    arrow(cx, 25.2, 23.4)
    box(cx, 20.8, 36, 4.2, fused, "red" if cx > 50 else "purple", fs=9.5)
    arrow(cx, 18.3, 15.9)
    box(cx, 13.3, 30, 3.8, out, "yellow", fs=9.5)
    ax.text(cx, 6.5, label, ha="center", va="center", fontsize=10, fontweight="bold")

panel(24, "EDC: fixed equal-weight fusion",
      [r"$1/3$", r"$1/3$", r"$1/3$"],
      "All scales contribute equally,\nindependent of how difficult\neach scale is to reconstruct",
      r"$p_{\mathrm{EDC}}=\frac{1}{3}\,(p_1+p_2+p_3)$",
      r"$S_{\mathrm{EDC}}=M(p_{\mathrm{EDC}})$",
      "(a) EDC baseline")

panel(76, "RQASW: adaptive scale weighting (ours)",
      [r"$w_1$", r"$w_2$", r"$w_3$"],
      r"EMA of per-scale loss (training only):" "\n"
      r"$\tilde{\ell}_k \leftarrow m\,\tilde{\ell}_k+(1-m)\,\ell_k^{(b)}$" "\n"
      r"$w_k=\tilde{\ell}_k\,/\,(\tilde{\ell}_1+\tilde{\ell}_2+\tilde{\ell}_3)$" "\n"
      "harder scale $\\rightarrow$ larger weight",
      r"$p_{\mathrm{RQASW}}=w_1p_1+w_2p_2+w_3p_3$",
      r"$S_{\mathrm{RQASW}}=M(p_{\mathrm{RQASW}})$",
      "(b) Proposed RQASW")

ax.plot([50, 50], [4, 55], ls=(0, (4, 3)), lw=0.8, color="#999999")
fig.savefig("RQASW_diagram.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("RQASW_diagram_preview.png", dpi=200, bbox_inches="tight", pad_inches=0.03)