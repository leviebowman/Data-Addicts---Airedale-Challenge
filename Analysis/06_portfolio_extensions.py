"""
06 - Portfolio extensions: claim lag, break-even, and sensitivity of the recommended portfolio.

Builds directly on PortfolioSim/PortfolioSim.py (imported read-only, so the programs, scopes,
benefits, stacking rule, Monte Carlo draws and discounting are exactly the teammates' model).
It closes "Next steps" items from PortfolioSim/NOTES.md:

  1. Claim lag.  PortfolioSim switches savings on the month a program goes live.  But a fix only
     protects units BUILT after go-live, and those units fail 10-240 days later (Days To Failure is
     uniform on that range, mean 125 days).  So in month m the warranty claims come from units built
     in month m - DTF.  We model this exactly: savings(m) = average over the observed Days-To-Failure
     distribution of savings(program set active at m - DTF).  Applied to both the catalog-benefit
     case and all 5,000 Monte Carlo draws.
  2. Break-even: how far below Airedale's catalog estimates can every benefit fall before the
     recommended portfolio stops paying back within 3 years?
  3. Tornado: one program at a time at 50% / 150% of its catalog benefit.
  4. Re-rank all 95 within-budget portfolios with lag: does the recommendation change?

Run:  .venv\\Scripts\\python.exe Analysis\\06_portfolio_extensions.py
"""
import sys

import numpy as np
import pandas as pd

from common import DARK, GRID, INK, INK2, LIGHT, MUTED, OUT, RED, ROOT, header, money, plt, save

sys.dont_write_bytecode = True  # don't leave a __pycache__ inside the teammates' PortfolioSim folder
sys.path.insert(0, str(ROOT / "PortfolioSim"))
import PortfolioSim as ps  # noqa: E402  (reads Data_Full.xlsx on import; writes nothing)

N_LAG = 24  # quantiles of the Days-To-Failure distribution used for the lag average
LAG_MONTHS = np.quantile(ps.df["Days To Failure"].to_numpy(float) / 30.44, (np.arange(N_LAG) + 0.5) / N_LAG)


def monthly(portfolio, table, starts, lag=True):
    """Monthly savings (draws x H). With lag: average over DTF quantiles of start-shifted activation."""
    if not lag:
        return ps.monthly_savings(portfolio, table, starts)
    return np.mean([ps.monthly_savings(portfolio, table, starts + q) for q in LAG_MONTHS], axis=0)


def npv(portfolio, table, starts, lag=True):
    return monthly(portfolio, table, starts, lag) @ ps.DISC - ps.PRICES[portfolio].sum()


def det_table(realization):
    return ps.savings_table(PATTERNS, CELL_COST, realization)[None, :]


def payback_month(portfolio, table, starts, lag=True):
    cum = np.cumsum(monthly(portfolio, table, starts, lag)[0] * ps.DISC) - ps.PRICES[portfolio].sum()
    hit = np.nonzero(cum >= 0)[0]
    return int(hit[0]) + 1 if len(hit) else None


def main():
    global PATTERNS, CELL_COST
    masks, prices, _ = ps.setup()
    ps.PRICES = prices
    PATTERNS, cells = ps.cells_of(masks)
    CELL_COST = np.bincount(cells, weights=ps.cost, minlength=len(PATTERNS))
    base = det_table(np.ones(ps.N))
    s0 = ps.MONTHS[None, :].astype(float)

    portfolios = [list(c) for r in range(1, ps.N + 1) for c in ps.combinations(range(ps.N), r)
                  if prices[list(c)].sum() <= ps.BUDGET]
    print(f"Days-To-Failure lag quantiles (months): {LAG_MONTHS.min():.1f} – {LAG_MONTHS.max():.1f}, "
          f"mean {LAG_MONTHS.mean():.1f}")

    # ---------------------------------------------------------------- 1 + 4: lag, re-rank (MC)
    print("Running the 5,000-draw Monte Carlo from PortfolioSim (same seed)...")
    table, starts = ps.simulate(masks)
    rows, npv_lag = [], []
    for p in portfolios:
        nl = npv(p, table, starts.astype(float), lag=True)
        npv_lag.append(nl)
        rows.append({"Portfolio": " + ".join(ps.NAMES[i] for i in p), "Cost": prices[p].sum(),
                     "NPV catalog, no lag": npv(p, base, s0, lag=False)[0],
                     "NPV catalog, with lag": npv(p, base, s0, lag=True)[0],
                     "Mean NPV no lag (MC, = PortfolioSim)": npv(p, table, starts.astype(float), lag=False).mean(),
                     "Mean NPV with lag (MC)": nl.mean(), "P5 NPV with lag (MC)": np.percentile(nl, 5),
                     "P(NPV<0) with lag": (nl < 0).mean(),
                     "Payback month, no lag": payback_month(p, base, s0, lag=False),
                     "Payback month, with lag": payback_month(p, base, s0, lag=True)})
    res = pd.DataFrame(rows)
    npv_lag = np.column_stack(npv_lag)
    res["P(best) with lag"] = np.bincount(npv_lag.argmax(1), minlength=len(portfolios)) / len(npv_lag)
    res = res.sort_values("Mean NPV with lag (MC)", ascending=False).reset_index(drop=True)
    res["ROI with lag (mean NPV / cost)"] = res["Mean NPV with lag (MC)"] / res["Cost"]
    res.to_csv(OUT / "A06_portfolios_with_lag.csv", index=False)
    pd.set_option("display.width", 250), pd.set_option("display.max_colwidth", 70)
    print(res.head(5).round(3).to_string(index=False))

    rec_name = "Braze Training + Wiring Poka-Yoke + Leak Detection + Supplier QA"
    rec = [ps.NAMES.index(n) for n in rec_name.split(" + ")]
    rec_row = res[res["Portfolio"] == rec_name].iloc[0]
    print(f"\nRecommended portfolio still ranks #{res.index[res['Portfolio'] == rec_name][0] + 1} with lag; "
          f"best in {rec_row['P(best) with lag']:.0%} of runs.")

    # ---------------------------------------------------------------- 2: break-even scale
    def npv_scaled(scale, lag=True):
        return npv(rec, det_table(np.full(ps.N, scale)), s0, lag)[0]

    lo, hi = 0.0, 1.0
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if npv_scaled(mid) < 0 else (lo, mid)
    breakeven = hi
    print(f"Break-even: the portfolio still pays back in 3 years if every program delivers "
          f"≥ {breakeven:.0%} of its catalog benefit (with claim lag).")

    # ---------------------------------------------------------------- 3: tornado
    base_npv = npv_scaled(1.0)
    torn = []
    for j in rec:
        r_lo, r_hi = np.ones(ps.N), np.ones(ps.N)
        r_lo[j], r_hi[j] = 0.5, 1.5
        torn.append({"Program": ps.NAMES[j], "Catalog benefit": ps.BENEFIT[j],
                     "NPV at 50% benefit": npv(rec, det_table(r_lo), s0)[0],
                     "NPV at 150% benefit": npv(rec, det_table(r_hi), s0)[0]})
    for j in rec:  # implementation slip of 3 extra months
        s = s0.copy()
        s[0, j] += 3
        torn.append({"Program": f"{ps.NAMES[j]} (+3 mo delay)", "Catalog benefit": ps.BENEFIT[j],
                     "NPV at 50% benefit": npv(rec, base, s)[0], "NPV at 150% benefit": base_npv})
    torn = pd.DataFrame(torn)
    torn["Swing"] = torn["NPV at 150% benefit"] - torn["NPV at 50% benefit"]
    torn = torn.sort_values("Swing", ascending=False)
    torn.to_csv(OUT / "A06_tornado.csv", index=False)
    print(torn.round(0).to_string(index=False))

    # ---------------------------------------------------------------- 5: failures avoided (D7)
    # Same stacking rule as PortfolioSim: a unit hit by programs j keeps prod(1 - b_j) of its failure
    # risk and cost.  Catalog benefits are applied to failures as well as cost (stated assumption).
    keep = np.ones(len(ps.df))
    by_prog = []
    for j in rec:
        hit = masks[j]
        before = keep.copy()
        keep = keep * np.where(hit, 1 - ps.BENEFIT[j], 1.0)
        by_prog.append({"Program": ps.NAMES[j], "Scope": ps.PROGRAMS[j][7],
                        "Failures avoided / yr (incremental, in this order)": float((before - keep).sum()),
                        "Warranty $ avoided / yr (incremental)": float(((before - keep) * ps.cost).sum())})
    avoided = 1 - keep
    ft = ps.df["Failure Description"].where(ps.df["Failure Category"] != "Leak", "Leak (all types)")
    by_type = pd.DataFrame({"Failure type": ft, "avoided": avoided, "$": avoided * ps.cost}) \
        .groupby("Failure type").agg(**{"Failures / yr": ("avoided", "size"), "Avoided / yr": ("avoided", "sum"),
                                        "Warranty $ avoided / yr": ("$", "sum")})
    by_type["Share of that failure avoided"] = by_type["Avoided / yr"] / by_type["Failures / yr"]
    pd.DataFrame(by_prog).to_csv(OUT / "A06_failures_avoided_by_program.csv", index=False)
    by_type.reset_index().to_csv(OUT / "A06_failures_avoided_by_type.csv", index=False)
    print(f"\nFailures avoided per year once all programs are live: {avoided.sum():.0f} of {len(avoided):,} "
          f"({avoided.mean():.1%}); warranty $ avoided {money((avoided * ps.cost).sum())}")
    print(by_type.round(2).to_string())
    print(pd.DataFrame(by_prog).round(0).to_string(index=False))

    summary = pd.DataFrame([{
        "Failures avoided per year (all live)": avoided.sum(),
        "Portfolio": rec_name, "Cost": prices[rec].sum(),
        "Yearly savings, all live": base[0, sum(1 << i for i in rec)],
        "3-yr NPV catalog, no lag (PortfolioSim)": rec_row["NPV catalog, no lag"],
        "3-yr NPV catalog, with lag": rec_row["NPV catalog, with lag"],
        "3-yr mean NPV with lag (MC)": rec_row["Mean NPV with lag (MC)"],
        "3-yr P5 NPV with lag (MC)": rec_row["P5 NPV with lag (MC)"],
        "P(best) with lag": rec_row["P(best) with lag"],
        "Payback month no lag": rec_row["Payback month, no lag"],
        "Payback month with lag": rec_row["Payback month, with lag"],
        "Break-even benefit scale": breakeven}])
    summary.to_csv(OUT / "A06_recommended_summary.csv", index=False)
    print(summary.T.to_string(header=False))

    fig_lag(res, rec_name)
    fig_tornado(torn, base_npv, breakeven)
    fig_cumulative(rec, base, s0)


def short(p):
    for k, v in ps.SHORT.items():
        p = p.replace(k, v)
    return p


def fig_lag(res, rec_name):
    top = res.head(6).iloc[::-1]
    fig, ax = plt.subplots(figsize=(13, 5.6))
    y = np.arange(len(top))
    nolag = "Mean NPV no lag (MC, = PortfolioSim)"
    ax.barh(y + 0.2, top[nolag], height=0.38, color=LIGHT,
            label="PortfolioSim: savings start at go-live (mean of 5,000 runs)")
    ax.barh(y - 0.2, top["Mean NPV with lag (MC)"], height=0.38,
            color=[DARK if p == rec_name else MUTED for p in top["Portfolio"]],
            label="With claim lag (mean of the same 5,000 runs)")
    for yi, (_, r) in zip(y, top.iterrows()):
        ax.text(r[nolag] + 5e4, yi + 0.2, money(r[nolag]).replace("$", r"\$"),
                va="center", fontsize=8.5, color=INK2)
        ax.text(r["Mean NPV with lag (MC)"] + 5e4, yi - 0.2,
                f"{money(r['Mean NPV with lag (MC)'])} · best in {r['P(best) with lag']:.0%} of runs".replace("$", r"\$"),
                va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [f"{short(p)}  ({money(c)})" for p, c in zip(top["Portfolio"], top["Cost"])], fontsize=9)
    ax.xaxis.set_major_formatter(money)
    ax.set_xlim(0, top[nolag].max() * 1.45)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, frameon=False, fontsize=9)
    ax.set_xlabel("3-year net savings (discounted 10%/yr, minus program cost)")
    fig.subplots_adjust(left=0.36, right=0.98, top=0.84, bottom=0.2)
    header(fig, "Claim Lag Lowers Net Savings but Does Not Change the Pick",
           "Savings ramp in as post-fix units reach the field (10–240 days to failure). Top 6 portfolios; "
           "dark = recommended.")
    save(fig, "A06_lag_reranking.png")


def fig_tornado(torn, base_npv, breakeven):
    fig, ax = plt.subplots(figsize=(13, 5.4))
    t = torn.iloc[::-1]
    y = np.arange(len(t))
    for yi, (_, r) in zip(y, t.iterrows()):
        lo, hi = r["NPV at 50% benefit"], r["NPV at 150% benefit"]
        ax.barh(yi, lo - base_npv, left=base_npv, color=RED, height=0.55)
        ax.barh(yi, hi - base_npv, left=base_npv, color=DARK, height=0.55)
        ax.text(min(lo, hi) - 4e4, yi, money(lo).replace("$", r"\$"), ha="right", va="center", fontsize=8.5, color=INK2)
        if hi > base_npv + 1:
            ax.text(hi + 4e4, yi, money(hi).replace("$", r"\$"), ha="left", va="center", fontsize=8.5, color=INK2)
    ax.axvline(base_npv, color=INK, lw=1)
    ax.set_yticks(y, [f"{r['Program']}" + ("" if "delay" in r["Program"] else f"  (catalog {r['Catalog benefit']:.0%})")
                      for _, r in t.iterrows()], fontsize=9)
    ax.xaxis.set_major_formatter(money)
    lo_all = torn["NPV at 50% benefit"].min()
    ax.set_xlim(lo_all - 7e5, torn["NPV at 150% benefit"].max() + 7e5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"3-year net savings of the recommended portfolio (with lag). Centre line = catalog case {money(base_npv)}")
    handles = [plt.Rectangle((0, 0), 1, 1, color=RED), plt.Rectangle((0, 0), 1, 1, color=DARK)]
    ax.legend(handles, ["Program delivers 50% of catalog benefit (or slips 3 months)", "Delivers 150%"],
              loc="lower right", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.3, right=0.97, top=0.84, bottom=0.12)
    header(fig, "Which Assumption Matters Most? Leak Detection's 40% Benefit",
           f"One program at a time. If every program delivers only {breakeven:.0%} of its catalog benefit, "
           "the portfolio still breaks even in 3 years.")
    save(fig, "A06_tornado.png")


def fig_cumulative(rec, base, s0):
    fig, ax = plt.subplots(figsize=(12, 5))
    months = np.arange(1, ps.H + 1)
    for lag, color, lab, off in [(False, LIGHT, "PortfolioSim (no lag)", (-150, 14)),
                                 (True, DARK, "With claim lag", (10, -18))]:
        cum = np.cumsum(monthly(rec, base, s0, lag)[0] * ps.DISC) - ps.PRICES[rec].sum()
        ax.plot(months, cum, color=color, lw=2.2, label=lab)
        be = np.nonzero(cum >= 0)[0]
        if len(be):
            ax.scatter([months[be[0]]], [cum[be[0]]], s=60, color=color, zorder=3)
            ax.annotate(f"pays back month {months[be[0]]}", (months[be[0]], cum[be[0]]), xytext=off,
                        textcoords="offset points", fontsize=9, color=INK2)
    ax.axhline(0, color=INK2, lw=0.8)
    live = {}
    for j in rec:
        live.setdefault(int(ps.MONTHS[j]), []).append(ps.SHORT[ps.NAMES[j]])
    top = ax.get_ylim()[1]
    for m, names in live.items():
        ax.axvline(m, color=GRID, lw=1)
        ax.text(m + 0.2, top * 0.97, " + ".join(names) + " live", va="top", ha="left", fontsize=8, color=INK2,
                rotation=90)
    ax.yaxis.set_major_formatter(money)
    ax.set_xlabel("Months after funding"), ax.set_ylabel("Cumulative discounted net savings")
    ax.legend(loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.12)
    header(fig, "Recommended Portfolio: Cumulative Net Savings",
           "Braze + Poka-Yoke + Leak Detection + Supplier QA ($700K), catalog benefits. Grey lines = each program goes live.")
    save(fig, "A06_cumulative.png")


if __name__ == "__main__":
    main()
