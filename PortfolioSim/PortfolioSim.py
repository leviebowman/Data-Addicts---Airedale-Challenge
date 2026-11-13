"""
Targeted portfolio simulation with a Monte Carlo robustness test.

Each program is deployed only where the data says its problem lives
(e.g. Poka-Yoke on Night shift + Controls Rev A builds). Programs keep their
catalog cost: targeting changes where a program runs, not what it costs.
Targeting keeps 100% of each program's savings versus deploying it everywhere.

Model:
  * Baseline = one year of warranty cost (Data_Full.xlsx), repeating each year.
  * Spend at month 0; benefit switches on after implementation time.
  * Programs hitting the same unit stack as 1 - prod(1 - b).
  * 3-year horizon, 10% discount. Every portfolio within $750K is scored.
  * Monte Carlo: bootstrap units (cost uncertainty), benefit realization
    ~ Triangular and schedule slip ~ Uniform, both set by risk level.

Run:  python PortfolioSim\\PortfolioSim.py   ->  outputs in PortfolioSim\\outputs\\
"""
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
DATA = HERE.parent / "WarrantyCostModel" / "Data_Full.xlsx"

BUDGET, H, RATE = 750_000, 36, 0.10          # budget, horizon (months), discount
N_DRAWS, SEED = 5000, 42
REALIZATION = {"Low": (0.6, 1.0, 1.2), "Medium": (0.3, 0.9, 1.3), "High": (0.1, 0.7, 1.3)}
MAX_SLIP = {"Low": 1, "Medium": 3, "High": 6}  # months

# ---------------------------------------------------------------- data
df = pd.read_excel(DATA)
cost = df["Warranty Cost"].to_numpy(float)
mode = df["Failure Mode"]
leak = df["Failure Category"].eq("Leak").to_numpy()
loose = mode.eq("Loose Terminal").to_numpy()
miswire = mode.eq("Sensor Miswire").to_numpy()
fan = mode.eq("Fan Rotation Error").to_numpy()
supB = df["Coil Supplier"].eq("Supplier B").to_numpy()
fac2 = df["Facility"].eq("Facility 2").to_numpy()
night = df["Shift"].eq("Night").to_numpy()
revA = df["Controls Revision"].eq("Rev A").to_numpy()
legacy = df["Product Family"].eq("LegacyX").to_numpy()

# ---------------------------------------------------------------- programs
# (name, catalog cost, months, risk, benefit, problem it hits, where it's deployed or None = everywhere, where, why)
PROGRAMS = [
    ("Braze Training", 50_000, 2, "Low", 0.05, leak, None, "All plants",
     "Every leak is logged as a braze defect, at all plants. Narrowing to Facility 2 would save "
     "~$18K of cost but give up ~$130K/yr of savings"),
    ("Smart Torque", 100_000, 3, "Low", 0.15, loose, None, "All lines",
     "Loose terminals are spread evenly across shifts, plants and products"),
    ("Wiring Poka-Yoke", 125_000, 4, "Low", 0.40, miswire | fan, night | revA, "Night shift + Controls Rev A",
     "100% of fan rotation errors are Night shift; 100% of sensor miswires are Rev A"),
    ("Design Validation", 200_000, 6, "Medium", 0.20, miswire, revA, "Controls Rev A",
     "Every controls failure is on Rev A"),
    ("Vision Inspection", 275_000, 6, "Medium", 0.35, loose | fan, None, "All lines",
     "Wiring failures are spread evenly"),
    ("Leak Detection", 350_000, 8, "Low", 0.40, leak, fac2 | supB, "Facility 2 + Supplier B coils",
     "Every leak is at Facility 2 or on a Supplier B coil"),
    ("Supplier QA", 175_000, 4, "Medium", 0.25, leak & supB, supB, "Supplier B coils",
     "Supplier B is the only supplier with leaks outside Facility 2"),
    ("LegacyX Refresh", 500_000, 12, "High", 0.35, legacy, None, "LegacyX units",
     "Program is LegacyX-specific by design"),
]
# Catalog programs not modelled: their benefit ("quality escapes") can't be priced from warranty data.
EXCLUDED = [("Predictive Quality Dashboard", 150_000), ("Additional EOLT Station", 450_000)]
NAMES = [p[0] for p in PROGRAMS]
BENEFIT = np.array([p[4] for p in PROGRAMS])
MONTHS = np.array([p[2] for p in PROGRAMS])
RISKS = [p[3] for p in PROGRAMS]
N = len(PROGRAMS)
SUBSETS = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1).astype(float)
DISC = (1 + RATE) ** (-np.arange(1, H + 1) / 12)


def setup():
    """Targeted masks, catalog prices and coverage for each program."""
    masks, prices, coverage = [], [], []
    for _, price, _, _, _, target, deploy, *_ in PROGRAMS:
        masks.append(target if deploy is None else target & deploy)
        prices.append(price)
        coverage.append(1.0 if deploy is None else deploy.mean())  # share of units covered
    return masks, np.array(prices, float), np.array(coverage)


# ---------------------------------------------------------------- model
def cells_of(masks):
    patterns, cells = np.unique(np.column_stack(masks), axis=0, return_inverse=True)
    return patterns.astype(float), cells.ravel()


def savings_table(patterns, cell_cost, realization):
    """Annual savings for every subset of programs."""
    log_keep = np.log1p(-np.clip(BENEFIT * realization, 0, 0.95)[None, :] * patterns)
    return (1 - np.exp(SUBSETS @ log_keep.T)) @ cell_cost


def simulate(masks):
    """Monte Carlo: annual-savings table (draws x subsets) and benefit start months (draws x N)."""
    rng = np.random.default_rng(SEED)
    patterns, cells = cells_of(masks)
    real = np.column_stack([rng.triangular(*REALIZATION[r], N_DRAWS) for r in RISKS])
    starts = np.column_stack([MONTHS[j] + rng.integers(0, MAX_SLIP[r] + 1, N_DRAWS) for j, r in enumerate(RISKS)])
    table = np.empty((N_DRAWS, 2 ** N))
    for d in range(N_DRAWS):
        idx = rng.integers(0, len(cost), len(cost))
        table[d] = savings_table(patterns, np.bincount(cells[idx], weights=cost[idx], minlength=len(patterns)), real[d])
    return table, starts


def monthly_savings(portfolio, table, starts):
    """Undiscounted warranty savings per month (draws x H)."""
    bits = 1 << np.array(portfolio)
    s, rows = starts[:, portfolio], np.arange(len(table))
    out = np.empty((len(table), H))
    for m in range(1, H + 1):
        out[:, m - 1] = table[rows, ((s < m) * bits).sum(axis=1)] / 12
    return out


def run():
    masks, prices, coverage = setup()
    table, starts = simulate(masks)
    patterns, cells = cells_of(masks)
    det_table = savings_table(patterns, np.bincount(cells, weights=cost, minlength=len(patterns)), np.ones(N))[None, :]
    portfolios = [list(c) for r in range(1, N + 1) for c in combinations(range(N), r)
                  if prices[list(c)].sum() <= BUDGET]
    rows, npvs = [], []
    for p in portfolios:
        spend = prices[p].sum()
        npv = monthly_savings(p, table, starts) @ DISC - spend
        det = (monthly_savings(p, det_table, MONTHS[None, :]) @ DISC - spend)[0]
        npvs.append(npv)
        rows.append({"Portfolio": " + ".join(NAMES[i] for i in p), "Cost": spend,
                     "Yearly savings (all live)": det_table[0, sum(1 << i for i in p)], "NPV (catalog)": det,
                     "Mean NPV": npv.mean(), "P5 NPV": np.percentile(npv, 5), "P95 NPV": np.percentile(npv, 95),
                     "ROI": npv.mean() / spend, "P(NPV<0)": (npv < 0).mean()})
    res = pd.DataFrame(rows)
    npvs = np.column_stack(npvs)
    res["P(best)"] = np.bincount(npvs.argmax(axis=1), minlength=len(portfolios)) / N_DRAWS
    order = res["Mean NPV"].sort_values(ascending=False).index
    return dict(res=res.loc[order].reset_index(drop=True), npvs=npvs[:, order],
                portfolios=[portfolios[i] for i in order], table=table, starts=starts,
                masks=masks, prices=prices, coverage=coverage)


# ---------------------------------------------------------------- charts
SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f3f2ef"
DARK, RED, LIGHT, GREEN = "#1d3f8f", "#d64541", "#9ec5f4", "#8fd19e"
TOP_N = 20  # "top portfolios" used for the program ranking
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})
SHORT = {"Braze Training": "Braze", "Smart Torque": "Torque", "Wiring Poka-Yoke": "Poka-Yoke",
         "Design Validation": "Design Val", "Vision Inspection": "Vision", "Leak Detection": "Leak Detect",
         "Supplier QA": "Supplier QA", "LegacyX Refresh": "LegacyX"}


def short(portfolio):
    for k, v in SHORT.items():
        portfolio = portfolio.replace(k, v)
    return portfolio


def money(x, _=None):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x / 1e3:.0f}K" if x else "$0"


def header(fig, title, sub):
    fig.text(0.012, 0.975, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.925, sub, fontsize=10, color=INK2, va="top")


def save(fig, path):
    fig.savefig(path, dpi=160)
    plt.close(fig)


SLICES = [
    ("Leaks: Supplier B coil,\nnot Facility 2", leak & supB & ~fac2),
    ("Leaks: Facility 2,\nnot Supplier B", leak & fac2 & ~supB),
    ("Leaks: Supplier B coil\nat Facility 2", leak & supB & fac2),
    ("Loose terminals", loose),
    ("Sensor miswires", miswire),
    ("Fan rotation errors", fan),
]


def fig_cost_map(sim, path):
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.6], hspace=0.08, left=0.15, right=0.98, top=0.86, bottom=0.07)
    top, grid = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    k = len(SLICES)
    vals = [cost[m].sum() for _, m in SLICES]
    top.bar(range(k), vals, color=LIGHT, width=0.86)
    for i, v in enumerate(vals):
        top.text(i, v + 6e4, f"{money(v)}  ({v / cost.sum():.0%})", ha="center", fontsize=9.5)
    top.set_xlim(-0.5, k - 0.5), top.set_ylim(0, max(vals) * 1.25)
    top.set_xticks([]), top.yaxis.set_major_formatter(money)
    top.set_ylabel("Yearly warranty cost")
    grid.set_xlim(-0.5, k - 0.5), grid.set_ylim(N - 0.5, -0.5)
    grid.grid(False)
    for s in grid.spines.values():
        s.set_visible(False)
    for r in range(N):
        for c, (_, sl) in enumerate(SLICES):
            t = cost[sim["masks"][r] & sl].sum()
            grid.add_patch(plt.Rectangle((c - 0.43, r - 0.42), 0.86, 0.84, color=DARK if t else SURFACE, ec=GRID, lw=0.8))
            if t:
                grid.text(c, r, money(t * BENEFIT[r]) + "/yr", ha="center", va="center", color="white",
                          fontsize=9, fontweight="bold")
    grid.set_xticks(range(k), [s for s, _ in SLICES], fontsize=9)
    grid.set_yticks(range(N), NAMES, fontsize=10.5, fontweight="bold", color=INK)
    grid.tick_params(length=0)
    header(fig, "Total Warranty Costs by Failure Types & Program Reach",
           "Cells represent yearly savings within failure types per program.")
    save(fig, path)


def program_ranking(sim):
    top = sim["portfolios"][:TOP_N]
    winners = sim["npvs"].argmax(axis=1)
    in_win = np.array([[i in sim["portfolios"][w] for i in range(N)] for w in np.unique(winners)])
    win_count = np.bincount(winners)[np.unique(winners)]
    rows = []
    for i, p in enumerate(PROGRAMS):
        rows.append({"Program": p[0], "Applied scope": p[7], "Cost": p[1],
                     f"In top {TOP_N}": sum(i in q for q in top),
                     "Share of runs in winning portfolio": (in_win[:, i] * win_count).sum() / N_DRAWS,
                     "In best portfolio": i in sim["portfolios"][0], "Modelled": True})
    rank = pd.DataFrame(rows).sort_values([f"In top {TOP_N}", "Share of runs in winning portfolio"], ascending=False)
    rank["Rank"] = range(1, N + 1)
    excluded = pd.DataFrame([{"Program": n, "Applied scope": "Not modelled", "Cost": c,
                              f"In top {TOP_N}": 0, "Share of runs in winning portfolio": 0.0,
                              "In best portfolio": False, "Modelled": False, "Rank": None} for n, c in EXCLUDED])
    return pd.concat([rank, excluded], ignore_index=True)


def fig_summary(rank, path):
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 1), ax.set_ylim(len(rank) + 0.6, -0.9)
    ax.axis("off")
    cols = {"Rank": 0.01, "Program": 0.06, "Applied scope": 0.26, "Cost": 0.47, "Bar": 0.62, "Runs": 0.885}
    heads = {"Rank": "Rank", "Program": "Program", "Applied scope": "Applied scope",
             "Cost": "Cost", "Bar": f"Appears in top {TOP_N} portfolios",
             "Runs": "% of runs won"}
    for k, x in cols.items():
        ax.text(x, -0.35, heads[k], fontsize=9.5, color=INK2, fontweight="bold", va="center")
    ax.plot([0, 1], [0.0, 0.0], color=GRID, lw=1)
    bar_w = 0.22
    for r, (_, row) in enumerate(rank.iterrows(), start=1):
        y = r - 0.5
        if r % 2:
            ax.add_patch(plt.Rectangle((0, y - 0.5), 1, 1, color=BAND, lw=0))
        muted = not row["Modelled"]
        color = INK2 if muted else INK
        ax.text(cols["Rank"] + 0.012, y, "—" if muted else f"{int(row['Rank'])}", fontsize=12, fontweight="bold",
                color=color, va="center", ha="center")
        ax.text(cols["Program"], y, row["Program"], fontsize=11, fontweight="bold", color=color, va="center")
        ax.text(cols["Applied scope"], y, "Not modelled: benefit can't be\npriced from warranty data" if muted
                else row["Applied scope"], fontsize=9 if muted else 10, color=color, va="center")
        ax.text(cols["Cost"], y, money(row["Cost"]), fontsize=10, color=color, va="center")
        if not muted:
            n = row[f"In top {TOP_N}"]
            ax.add_patch(plt.Rectangle((cols["Bar"], y - 0.22), bar_w, 0.44, color=GRID, lw=0))
            ax.add_patch(plt.Rectangle((cols["Bar"], y - 0.22), bar_w * n / TOP_N, 0.44,
                                       color=DARK if row["In best portfolio"] else LIGHT, lw=0))
            ax.text(cols["Bar"] + bar_w + 0.008, y, f"{n}/{TOP_N}", fontsize=10, va="center")
            ax.text(cols["Runs"] + 0.045, y, f"{row['Share of runs in winning portfolio']:.0%}", fontsize=10,
                    va="center", ha="center")
    handles = [plt.Rectangle((0, 0), 1, 1, color=DARK), plt.Rectangle((0, 0), 1, 1, color=LIGHT)]
    ax.legend(handles, ["In the recommended portfolio", "Not in the recommended portfolio"],
              loc="lower right", bbox_to_anchor=(1, -0.06), ncol=2, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04)
    header(fig, "Program Summary: Top Performing Catalog Programs",
           "% of runs won: share of 5,000 simulations in which the winning portfolio includes the program.")
    save(fig, path)


def fig_frontier(sim, path):
    res = sim["res"]
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.scatter(res["Cost"], res["Mean NPV"], s=38, color=LIGHT, edgecolor=SURFACE, lw=1, zorder=2,
               label=f"All {len(res)} portfolios within budget")
    front = res.sort_values("Cost")
    front = front[front["Mean NPV"] >= front["Mean NPV"].cummax()]
    ax.plot(front["Cost"], front["Mean NPV"], color=DARK, lw=1.4, zorder=3, label="Best net savings for each spend level")
    b = res.iloc[0]
    ax.scatter([b["Cost"]], [b["Mean NPV"]], s=130, color=DARK, edgecolor=SURFACE, lw=2, zorder=4,
               label="Recommended portfolio")
    ax.annotate(f"Recommended: {short(b['Portfolio'])}\n{money(b['Cost'])} spend · {money(b['Mean NPV'])} net savings"
                .replace("$", r"\$"), (b["Cost"], b["Mean NPV"]), xytext=(0.9, 0.6), textcoords="axes fraction",
                ha="right", va="top", fontsize=9.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.axvline(BUDGET, color=RED, lw=1.6, ls="--")
    ax.text(BUDGET + 6e3, 0.3, "$750K budget", transform=ax.get_xaxis_transform(), color=RED, ha="left",
            fontsize=9.5, rotation=90)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.xaxis.set_major_formatter(money), ax.yaxis.set_major_formatter(money)
    ax.set_xlabel("Upfront spend"), ax.set_ylabel("3-year net savings (average of 5,000 runs)")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.86, bottom=0.1)
    header(fig, "Frontier Chart: Savings vs Cost", "Each dot is one combination of programs that fits the $750K budget.")
    save(fig, path)


def fig_matrix(sim, rank, path):
    order = [NAMES.index(n) for n in rank[rank["Modelled"]]["Program"]]
    top = sim["res"].head(TOP_N)
    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(-0.5, N + 2.2), ax.set_ylim(TOP_N + 0.9, -1.5)
    ax.axis("off")
    for c, i in enumerate(order):
        ax.text(c, -0.6, NAMES[i].replace(" ", "\n", 1), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(N - 0.2, -0.6, "Spend", ha="left", va="bottom", fontsize=9, fontweight="bold")
    ax.text(N + 0.9, -0.6, "Net savings", ha="left", va="bottom", fontsize=9, fontweight="bold")
    for r in range(TOP_N):
        port = sim["portfolios"][r]
        ax.text(-0.75, r, f"#{r + 1}", ha="right", va="center", fontsize=9.5, color=INK2)
        for c, i in enumerate(order):
            on = i in port
            ax.add_patch(plt.Rectangle((c - 0.44, r - 0.4), 0.88, 0.8, lw=0.8, ec=GRID,
                                       color=(DARK if r == 0 else LIGHT) if on else SURFACE))
        ax.text(N - 0.2, r, money(top.loc[r, "Cost"]), va="center", fontsize=9.5)
        ax.text(N + 0.9, r, money(top.loc[r, "Mean NPV"]), va="center", fontsize=9.5)
    counts = [sum(i in p for p in sim["portfolios"][:TOP_N]) for i in order]
    for c, n in enumerate(counts):
        ax.text(c, TOP_N + 0.35, f"{n}/{TOP_N}", ha="center", va="center", fontsize=10, fontweight="bold", color=DARK)
    ax.text(-0.75, TOP_N + 0.35, "Total", ha="right", va="center", fontsize=9.5, color=INK2)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.03)
    header(fig, f"The Top {TOP_N} Portfolios",
           "Rows ranked by 3-year net savings; a filled cell means the program is in that portfolio. "
           "Columns follow the program ranking.")
    save(fig, path)


def fig_program_reach(sim, path):
    rows = []
    for i, p in enumerate(PROGRAMS):
        targeted = cost[sim["masks"][i]].sum()
        saves = targeted * BENEFIT[i]
        rows.append((p[0], p[7], int(sim["masks"][i].sum()), targeted, BENEFIT[i], saves, sim["prices"][i],
                     MONTHS[i] + sim["prices"][i] / (saves / 12)))
    t = pd.DataFrame(rows, columns=["name", "scope", "units", "targeted", "benefit", "saves", "price", "payback"])
    t = t.sort_values("saves")
    fig, ax = plt.subplots(figsize=(13, 6.8))
    y = np.arange(N)
    ax.barh(y, t["targeted"], color=LIGHT, height=0.62, label="Yearly warranty cost in the area it targets")
    ax.barh(y, t["saves"], color=DARK, height=0.62, label="Yearly savings it recovers (catalog benefit)")
    xmax = t["targeted"].max()
    ax.set_yticks(y, [""] * N)
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(-xmax * 0.02, yi + 0.1, r["name"], ha="right", va="bottom", fontsize=10.5, fontweight="bold")
        ax.text(-xmax * 0.02, yi + 0.08, f"{r['scope']} · {r['units']:,} units", ha="right", va="top",
                fontsize=8.5, color=INK2)
        ax.text(r["targeted"] + xmax * 0.015, yi,
                (f"saves {money(r['saves'])}/yr  ({r['benefit']:.0%} of {money(r['targeted'])})   ·   "
                 f"costs {money(r['price'])}   ·   pays back ~{r['payback']:.0f} mo").replace("$", r"\$"),
                va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, xmax * 1.85)
    ax.xaxis.set_major_formatter(money)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.86, bottom=0.08)
    header(fig, "Program Reach: Yearly Savings vs Targeted Warranty Cost",
           "Each program at its applied scope, on its own. Payback includes implementation time.")
    save(fig, path)


def fig_robustness(sim, path):
    top = sim["res"].head(8)
    fig, ax = plt.subplots(figsize=(13, 6.4))
    data = [sim["npvs"][:, i] for i in range(len(top))][::-1]
    bp = ax.boxplot(data, vert=False, widths=0.55, whis=(5, 95), showfliers=False, patch_artist=True,
                    medianprops=dict(color=SURFACE, lw=2), whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for j, box in enumerate(bp["boxes"]):
        box.set(facecolor=DARK if j == len(data) - 1 else LIGHT, edgecolor="none")
    ax.set_yticks(range(1, len(top) + 1),
                  [f"{short(p)}  ({money(c)})" for p, c in zip(top["Portfolio"], top["Cost"])][::-1], fontsize=9)
    ax.grid(axis="y", visible=False)
    for k, (_, r) in enumerate(top[::-1].iterrows(), start=1):
        pb = r["P(best)"]
        txt = "never best" if pb == 0 else ("best in <1%" if pb < 0.005 else f"best in {pb:.0%}")
        ax.text(1.01, k, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=INK2)
    ax.xaxis.set_major_formatter(money)
    ax.set_xlabel("3-year net savings (box = middle 50%, whiskers = 5th–95th percentile)")
    fig.subplots_adjust(left=0.36, right=0.9, top=0.86, bottom=0.1)
    header(fig, "Robustness Test",
           "Top 8 portfolios across 5,000 simulations varying warranty costs, how much benefit each program delivers, and rollout delays.")
    save(fig, path)


# ---------------------------------------------------------------- main
def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*"):
        old.unlink()
    print(f"Baseline yearly warranty cost: ${cost.sum():,.0f}  ({len(df):,} units)")
    sim = run()
    sim["res"].to_csv(OUT / "portfolio_results.csv", index=False)
    rank = program_ranking(sim)
    rank.drop(columns="Modelled").to_csv(OUT / "program_ranking.csv", index=False)

    pd.DataFrame({
        "Program": NAMES, "Applied scope": [p[7] for p in PROGRAMS], "Why": [p[8] for p in PROGRAMS],
        "Units covered": sim["coverage"].round(3), "Cost": sim["prices"],
        "Yearly savings (on its own)": [cost[m].sum() * b for m, b in zip(sim["masks"], BENEFIT)],
    }).to_csv(OUT / "program_scopes.csv", index=False)

    fig_cost_map(sim, OUT / "fig1_cost_map.png")
    fig_summary(rank, OUT / "fig2_program_summary.png")
    fig_frontier(sim, OUT / "fig3_portfolio_frontier.png")
    fig_matrix(sim, rank, OUT / "fig4_top20_portfolios.png")
    fig_program_reach(sim, OUT / "fig5_program_reach.png")
    fig_robustness(sim, OUT / "fig6_robustness.png")

    cols = ["Portfolio", "Cost", "Yearly savings (all live)", "NPV (catalog)", "Mean NPV", "P5 NPV", "P(best)"]
    fmt = {c: "${:,.0f}".format for c in cols[1:6]} | {"P(best)": "{:.0%}".format}
    print(f"\n{len(sim['res'])} portfolios within budget. Top 3:")
    print(sim["res"].head(3)[cols].to_string(index=False, formatters=fmt))
    print("\nProgram ranking:")
    print(rank[["Rank", "Program", "Applied scope", "Cost", f"In top {TOP_N}",
                "Share of runs in winning portfolio"]].to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
