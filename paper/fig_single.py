"""Figure 1 of the letter: a single chiral Hexbug (stage 1).

(a) direction correlation with the Eq.-(4) fit, (b) MSD with assisting lines.
Uses the notebook's own functions (stage1_unified_v2.ipynb) so the numbers are the ones of the stage-1 analysis.
Needs paper/cache/<video>.npz from track_stage1.py.     Usage: python paper/fig_single.py [video-stem]
"""
import ast, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = sys.argv[1] if len(sys.argv) > 1 else "2026-11-01_11-54-10"

# ---- reuse the notebook's functions: execute only imports, constants and function definitions
nb = json.load(open(os.path.join(ROOT, "stage1_code", "stage1_unified_v2.ipynb")))
ns = {"__name__": "nb"}
for i in (1, 10, 15, 18):
    src = "\n".join(l for l in "".join(nb["cells"][i]["source"]).splitlines() if not l.lstrip().startswith("%"))
    tree = ast.parse(src)
    os.chdir(ROOT)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign)):
            try:
                exec(compile(ast.Module([node], []), "<nb>", "exec"), ns)
            except NameError:            # assignment that needs notebook state (e.g. loaded data) - not needed here
                pass

z = np.load(os.path.join(ROOT, "paper", "cache", name + ".npz"))
t, xy = z["t1"], z["xy1"]
lag, msd = ns["calculate_msd"](t, xy)
clag, C = ns["calculate_direction_correlation"](t, xy)

# power-law guides (same ranges as the stage-1 analysis)
R1, R3 = (0.2, 1.5), (9.0, 40.0)   # fit ranges of the selected stage-1 plots
a1, b1, s1 = ns["fit_power_law"](lag, msd, R1)
a3, b3, s3 = ns["fit_power_law"](lag, msd, R3)
# damped cosine for the direction correlation (notebook fit window 0.4-20 s)
p, perr, r2, sel = ns["fit_direction_correlation"](clag, C)
a0, A1, Om, Gam = p
# Eq. (4) of the letter: C(t) = exp(-Dr t) cos(Omega t), fitted over 0.4-20 s
eq4 = lambda x, Dr, W: np.exp(-Dr * x) * np.cos(W * x)
mw = (clag >= 0.4) & (clag <= 20)
bestc = None
for w0 in (0.4, 0.7, 1.0, 1.4):
    try:
        qc, qcc = curve_fit(eq4, clag[mw], C[mw], p0=[0.05, w0])
    except (RuntimeError, ValueError):
        continue
    rr = np.sum((C[mw] - eq4(clag[mw], *qc)) ** 2)
    if bestc is None or rr < bestc[0]:
        bestc = (rr, qc, qcc)
Dr4, Om4 = bestc[1][0], abs(bestc[1][1])
R2_4 = 1 - bestc[0] / np.sum((C[mw] - C[mw].mean()) ** 2)
print(f"Eq.(4) fit 0.4-20 s: Omega = {Om4:.3f} rad/s, T = {2*np.pi/Om4:.2f} s, Dr = {Dr4:.4f} 1/s, 1/Dr = {1/Dr4:.1f} s, R2 = {R2_4:.3f}; errors {np.sqrt(np.diag(bestc[2]))}")
# chiral ABP + constant drift of the orbit centre, fitted to the whole MSD
MSD_FIT = (0.13, 52.0)  # model fit is reported in the text only; not drawn
m = (lag >= MSD_FIT[0]) & (lag <= MSD_FIT[1])
model = lambda tt, lv2, om, g, V, c: ns["chiral_msd"](tt, np.exp(lv2), om, g, c) + V ** 2 * tt ** 2
best = None
for om0 in (0.55, Om, 0.7):
    for g0 in (0.005, 0.05, 0.3):
        for V0 in (10, 20, 30):
            try:
                q, cov = curve_fit(model, lag[m], msd[m], p0=[np.log(70 ** 2), om0, g0, V0, 0.0], sigma=msd[m],
                                   bounds=([-np.inf, 0, 1e-6, 0, 0], [np.inf] * 5), maxfev=60000)
            except (RuntimeError, ValueError):
                continue
            sse = np.sum((np.log(model(lag[m], *q)) - np.log(msd[m])) ** 2)
            if best is None or sse < best[0]:
                best = (sse, q, cov)
sse, q, qcov = best
qerr = np.sqrt(np.diag(qcov))
v0, Om_m, Gam_m, V_m, c_m = np.exp(q[0] / 2), q[1], q[2], q[3], q[4]
# independent drift estimate: orbit centre = position averaged over one orbit period
dtf = np.median(np.diff(t)); kk = int(round(2 * np.pi / Om / dtf)); okk = np.isfinite(xy).all(axis=1)
xs = np.convolve(np.where(okk, xy[:, 0], np.nan), np.ones(kk) / kk, mode="valid")
ys = np.convolve(np.where(okk, xy[:, 1], np.nan), np.ones(kk) / kk, mode="valid")
tc = t[kk // 2:kk // 2 + len(xs)]; gg = np.isfinite(xs) & np.isfinite(ys)
V_traj = np.hypot(np.polyfit(tc[gg], xs[gg], 1)[0], np.polyfit(tc[gg], ys[gg], 1)[0])
print(f"video {name}: {len(t)} frames, {t[-1]:.1f} s")
print(f"MSD exponent {R1[0]}-{R1[1]} s: alpha = {a1:.2f};  {R3[0]}-{R3[1]} s: alpha = {a3:.2f}")
print(f"direction correlation fit (0.4-20 s): Omega = {Om:.3f} +- {perr[2]:.3f} 1/s (period {2*np.pi/Om:.1f} s), "
      f"Gamma = {Gam:.4f} +- {perr[3]:.4f} 1/s, amplitude {A1:.2f}, offset {a0:.3f}, R2 = {r2:.3f}")
print(f"chiral ABP + drift fitted to MSD {MSD_FIT} s: v0 = {v0:.1f} px/s, Omega = {Om_m:.3f} +- {qerr[1]:.3f} 1/s, "
      f"Gamma = {Gam_m:.3f} +- {qerr[2]:.3f} 1/s, V = {V_m:.1f} +- {qerr[3]:.1f} px/s, c = {c_m:.0f} px^2; "
      f"orbit radius v0/Omega = {v0/Om_m:.0f} px;  drift from period-averaged trajectory: {V_traj:.1f} px/s")

# ------------------------------------------------------------------ figure
BLUE, ORANGE, INK, MUTED = "#0072B2", "#D55E00", "#222222", "#666666"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False})
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.45), gridspec_kw=dict(wspace=0.4))

ax = axs[0]
ax.plot(clag, C, color=BLUE, lw=0.8)
xs = np.linspace(0.4, 100, 1000)
ax.plot(xs, eq4(xs, Dr4, Om4), color=ORANGE, lw=1.0, ls=":")
xf = np.linspace(0.4, 20, 300)
ax.plot(xf, eq4(xf, Dr4, Om4), color=ORANGE, lw=1.4)
ax.axhline(0, color=MUTED, lw=0.6)
ax.set_xlim(0, 100); ax.set_ylim(-1, 1)
ax.set_xlabel("lag time $t$ (s)"); ax.set_ylabel(r"direction correlation $C(t)$")
ax.text(0.97, 0.06, r"$e^{-D_r t}\cos(\Omega t)$", color=ORANGE, transform=ax.transAxes, ha="right", fontsize=7)
ax.text(-0.34, 1.02, "(a)", transform=ax.transAxes, fontweight="bold")

ax = axs[1]
ax.loglog(lag, msd, "o", ms=2.5, color=BLUE, mec="none")
ax.axvspan(2.0, 9.0, color="#9467bd", alpha=0.08, lw=0)
ax.axvline(5.5, color="#9467bd", lw=0.8, ls=":")
for (alpha, b, s_, off, lab) in ((a1, b1, s1, 0.35, f"$\\propto t^{{{a1:.2f}}}$"), (a3, b3, s3, 0.35, f"$\\propto t^{{{a3:.2f}}}$")):
    xx = np.logspace(np.log10(lag[s_][0]), np.log10(lag[s_][-1]), 20)
    ax.loglog(xx, off * 10 ** b * xx ** alpha, "--", color=INK, lw=0.9)
    ax.text(np.sqrt(xx[0] * xx[-1]), off * 10 ** b * np.sqrt(xx[0] * xx[-1]) ** alpha * 0.45, lab, ha="center", va="top", fontsize=7)
ax.set_xlabel("lag time $t$ (s)"); ax.set_ylabel(r"MSD (px$^2$)")
ax.text(-0.34, 1.02, "(b)", transform=ax.transAxes, fontweight="bold")

os.makedirs(os.path.join(ROOT, "paper", "figures"), exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(ROOT, "paper", "figures", f"fig1_single.{ext}"), dpi=300, bbox_inches="tight")
print("saved paper/figures/fig1_single.pdf")
