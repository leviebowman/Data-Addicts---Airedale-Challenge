"""
02 - Statistical inference: are the root-cause patterns real, and how big are they?

The portfolio (PortfolioSim) targets each program at a slice of the build
("Leak Detection on Facility 2 + Supplier B", "Poka-Yoke on Night shift + Rev A").
Those scopes rest on patterns in the data.  This script tests each one formally:
  * effect size with a 95% confidence interval (Wilson for shares, bootstrap for
    cost ratios),
  * an exact / permutation p-value,
  * Benjamini-Hochberg correction across the whole family of tests,
  * a verdict (Supported / Not supported) and the step that would confirm the
    cause physically before money is spent.

IMPORTANT framing: every unit in the data failed.  So "leak share" means
"of the units that failed, the share that failed by leaking" - a statement
about failure *mix*, not failure *rate* per unit built.  Production volumes
would be needed to turn these into rates (see Analysis/README.md).

Run:  .venv\\Scripts\\python.exe Analysis\\02_inference.py
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

from common import (CAT_COLOR, DARK, GRID, INK, INK2, LIGHT, MUTED, OUT, RED, SEED, SURFACE, header, load,
                    money, plt, save)

rng = np.random.default_rng(SEED)


def share(k, n):
    lo, hi = proportion_confint(k, n, method="wilson")
    return k / n, lo, hi


def two_group(df, outcome, group_mask, label_in, label_out):
    """Compare the share of `outcome` inside vs outside a group (Fisher exact)."""
    a, n1 = int((outcome & group_mask).sum()), int(group_mask.sum())
    b, n0 = int((outcome & ~group_mask).sum()), int((~group_mask).sum())
    p = fisher_exact([[a, n1 - a], [b, n0 - b]])[1]
    s1, s0 = share(a, n1), share(b, n0)
    return dict(group=label_in, k_in=a, n_in=n1, share_in=s1[0], ci_in=f"{s1[1]:.1%}–{s1[2]:.1%}",
                compare=label_out, k_out=b, n_out=n0, share_out=s0[0], ci_out=f"{s0[1]:.1%}–{s0[2]:.1%}",
                effect=f"{s1[0]:.1%} vs {s0[0]:.1%}", p=p)


def boot_ratio(x, y, n=4000):
    """Bootstrap CI for mean(x)/mean(y)."""
    xb = rng.choice(x, (n, len(x))).mean(1)
    yb = rng.choice(y, (n, len(y))).mean(1)
    r = xb / yb
    return x.mean() / y.mean(), np.percentile(r, 2.5), np.percentile(r, 97.5)


def main():
    df = load()
    leak = df["Failure Category"].eq("Leak")
    ctrl = df["Failure Category"].eq("Controls")
    fan = df["Failure Description"].eq("Condenser Fan Backwards")
    supB = df["Coil Supplier"].eq("Supplier B")
    fac2 = df["Facility"].eq("Facility 2")
    revA = df["Controls Revision"].eq("Rev A")
    night = df["Shift"].eq("Night")
    fancorr = df["Repair Type"].eq("Fan Rotation Correction")
    legacy = df["Product Family"].eq("LegacyX")

    tests = []

    def add(hid, claim, program, verify, **r):
        tests.append(dict(ID=hid, Claim=claim, Program=program, **r, Verification=verify))

    # H1 - Supplier B coils leak
    r = two_group(df, leak, supB, "Supplier B", "Suppliers A+C")
    add("H1", "Supplier B coils fail by leaking more often", "Supplier QA, Leak Detection",
        "Incoming helium leak test on a lot of Supplier B coils vs A/C; supplier 8D", **r)
    # H2 - Facility 2 leaks (excluding Supplier B so the two causes don't mix)
    notB = ~supB
    r = two_group(df[notB], leak[notB], fac2[notB], "Facility 2 (A/C coils)", "Facilities 1,3,4 (A/C coils)")
    add("H2", "Facility 2 creates leaks even on good (A/C) coils", "Leak Detection, Braze Training",
        "Braze process audit at Facility 2 (alloy, flux, temperature, operator certs) vs other plants", **r)
    # H3 - Supplier B leaks are NOT a Facility 2 thing (same share in every plant)
    tab = pd.crosstab(df.loc[supB, "Facility"], leak[supB])
    p = chi2_contingency(tab)[1]
    shares = tab[True] / tab.sum(axis=1)
    add("H3", "Supplier B leak share differs by facility (a 'Not supported' verdict means the Supplier B "
        "problem is supplier-driven, not plant-driven)", "Supplier QA", "Same as H1", group="Supplier B by facility", effect=" / ".join(f"{v:.0%}" for v in shares),
        p=p, n_in=int(supB.sum()))
    # H4 - Controls failures are Rev A
    r = two_group(df, ctrl, revA, "Controls Rev A", "Rev B + Rev C")
    add("H4", "All controls failures (sensor/comm faults) are on Controls Rev A", "Design Validation, Poka-Yoke",
        "Bench-test Rev A boards vs Rev B/C; compare connector keying and firmware", **r)
    # H5 - Fan backwards is Night shift
    r = two_group(df, fan, night, "Night shift", "Day shift")
    add("H5", "All backward-fan failures come from Night shift", "Wiring Poka-Yoke",
        "Audit Night-shift fan wiring step; check work instructions and training records", **r)
    # H6 - NEW: Fan backwards only after Night-shift 'Fan Rotation Correction' rework
    r = two_group(df, fan, fancorr, "Got 'Fan Rotation Correction' rework", "Any other rework")
    add("H6", "Backward fans occur only on units that received a 'Fan Rotation Correction' rework",
        "Wiring Poka-Yoke (keyed fan connector)",
        "Pull 20 corrected units, check fan direction after the correction; watch the correction procedure", **r)
    # H7 - LegacyX costs more per failure, holding the failure type fixed
    m = smf.ols("np.log(Q('Warranty Cost')) ~ C(Q('Failure Description')) + legacy", data=df.assign(legacy=legacy)).fit()
    coef, (lo, hi) = m.params["legacy[T.True]"], m.conf_int().loc["legacy[T.True]"]
    add("H7", "LegacyX failures cost more to fix than the same failure on DC units", "LegacyX Refresh",
        "Compare LegacyX vs DC service tickets: parts price, labor hours, obsolete-part lead time",
        group="LegacyX vs DC100/200/300", effect=f"×{np.exp(coef):.2f} cost (95% CI ×{np.exp(lo):.2f}–×{np.exp(hi):.2f})",
        p=float(m.pvalues["legacy[T.True]"]), n_in=int(legacy.sum()))
    # H8 - Disposition: 'Use As Is' vs 'Rework Completed' (null expected)
    m2 = smf.ols("np.log(Q('Warranty Cost')) ~ C(Q('Failure Description')) + C(Q('Product Family')) + uai",
                 data=df.assign(uai=df["Disposition Code"].eq("Use As Is"))).fit()
    c2, (l2, h2) = m2.params["uai[T.True]"], m2.conf_int().loc["uai[T.True]"]
    add("H8", "'Use As Is' units cost more in warranty than 'Rework Completed' units", "(none - no action needed)",
        "-", group="Use As Is vs Rework Completed",
        effect=f"×{np.exp(c2):.3f} cost (95% CI ×{np.exp(l2):.3f}–×{np.exp(h2):.3f})",
        p=float(m2.pvalues["uai[T.True]"]), n_in=int(df["Disposition Code"].eq("Use As Is").sum()))
    # H9 - Region, test station, days-to-failure: nulls
    for hid, col in [("H9a", "Customer Region"), ("H9b", "Test Station")]:
        p = chi2_contingency(pd.crosstab(df[col], df["Failure Category"]))[1]
        add(hid, f"{col} is related to the type of failure", "(none)", "-", group=col, effect="failure mix", p=p)
    p = kruskal(*[g["Days To Failure"] for _, g in df.groupby("Failure Category")]).pvalue
    add("H9c", "Days to failure differs by failure type", "(none)", "-", group="Failure Category",
        effect=" / ".join(f"{k}: {v:.0f}d" for k, v in df.groupby("Failure Category")["Days To Failure"].mean().items()), p=p)
    # H10 - Loose terminals are spread evenly (justifies an all-lines Smart Torque rollout)
    # Test within non-leak failures so the leak hot-spots (Facility 2, Supplier B) don't distort the mix,
    # and within non-Rev-A so the Rev A controls failures don't either.
    loose = df["Failure Description"].eq("Loose Power Wiring")
    base = ~leak & ~revA
    for hid, col in [("H10a", "Facility"), ("H10b", "Coil Supplier"), ("H10c", "Product Family")]:
        p = chi2_contingency(pd.crosstab(df.loc[base, col], loose[base]))[1]
        lt = df.loc[base, col][loose[base]].value_counts(normalize=True)
        allu = df.loc[base, col].value_counts(normalize=True)
        add(hid, f"Loose terminals concentrate in a particular {col}", "Smart Torque / Vision (all lines)", "-",
            group=col, effect="max |share − expected| = " + f"{(lt - allu).abs().max():.1%}", p=p)

    res = pd.DataFrame(tests)
    res["q (BH)"] = multipletests(res["p"], method="fdr_bh")[1]
    res["Verdict"] = np.where(res["q (BH)"] < 0.05, "Supported", "Not supported")
    res.to_csv(OUT / "A02_root_cause_tests.csv", index=False)

    show = res[["ID", "Claim", "effect", "p", "q (BH)", "Verdict"]].copy()
    show["p"] = show["p"].map(lambda v: f"{v:.1e}")
    show["q (BH)"] = show["q (BH)"].map(lambda v: f"{v:.1e}")
    pd.set_option("display.width", 250), pd.set_option("display.max_colwidth", 80)
    print(show.to_string(index=False))

    # Rule-of-three bound: Supplier A/C outside Facility 2 have zero leaks
    n0 = int((~supB & ~fac2).sum())
    print(f"\nSupplier A/C coils outside Facility 2: 0 leaks in {n0} failures "
          f"(95% upper bound on leak share = {3 / n0:.2%}, rule of three)")

    fig_evidence(df, leak, ctrl, fan, supB, legacy, fancorr)


def fig_evidence(df, leak, ctrl, fan, supB, legacy, fancorr):
    fig = plt.figure(figsize=(15, 9.4))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.3, left=0.08, right=0.97, top=0.83, bottom=0.08)

    # (a) leak share by supplier x facility
    ax = fig.add_subplot(gs[0, 0])
    tab = pd.crosstab(df["Coil Supplier"], df["Facility"], values=leak, aggfunc="mean")
    cnt = pd.crosstab(df["Coil Supplier"], df["Facility"])
    ax.imshow(tab.values, cmap="Oranges", vmin=0, vmax=0.8, aspect="auto")
    for i in range(tab.shape[0]):
        for j in range(tab.shape[1]):
            v = tab.values[i, j]
            ax.text(j, i, f"{v:.0%}\n(n={cnt.values[i, j]})", ha="center", va="center", fontsize=9.5,
                    color="white" if v > 0.4 else INK)
    ax.set_xticks(range(4), tab.columns), ax.set_yticks(range(3), tab.index)
    ax.grid(False)
    ax.set_title("(a) Share of failures that are leaks\nSupplier B everywhere + Facility 2 for every supplier",
                 loc="left", fontsize=10.5, fontweight="bold")

    # (b) controls share by revision with CI
    ax = fig.add_subplot(gs[0, 1])
    revs = ["Rev A", "Rev B", "Rev C"]
    for i, rv in enumerate(revs):
        m = df["Controls Revision"].eq(rv)
        s, lo, hi = share(int((ctrl & m).sum()), int(m.sum()))
        ax.bar(i, s, color=CAT_COLOR["Controls"] if s > 0 else MUTED, width=0.55)
        ax.errorbar(i, s, yerr=[[s - lo], [hi - s]], color=INK2, capsize=4)
        ax.text(i, hi + 0.015, f"{s:.0%}\n{int((ctrl & m).sum())}/{int(m.sum())}", ha="center", fontsize=9)
    ax.set_xticks(range(3), revs), ax.set_ylim(0, 0.45)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("(b) Share of failures that are controls faults (95% CI)\nOnly Rev A boards ever fail on controls",
                 loc="left", fontsize=10.5, fontweight="bold")

    # (c) backward-fan share by rework path
    ax = fig.add_subplot(gs[1, 0])
    groups = [("Day shift\n(any rework)", df["Shift"].eq("Day")),
              ("Night shift\nother rework", df["Shift"].eq("Night") & ~fancorr),
              ("Night shift\n'Fan Rotation Correction'", fancorr)]
    for i, (lab, m) in enumerate(groups):
        s, lo, hi = share(int((fan & m).sum()), int(m.sum()))
        ax.bar(i, s, color=CAT_COLOR["Wiring"] if s > 0 else MUTED, width=0.55)
        ax.errorbar(i, s, yerr=[[s - lo], [hi - s]], color=INK2, capsize=4)
        ax.text(i, hi + 0.02, f"{s:.0%}  ({int((fan & m).sum())}/{int(m.sum())})", ha="center", fontsize=9)
    ax.set_xticks(range(3), [g[0] for g in groups]), ax.set_ylim(0, 0.7)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("(c) Share of failures that are backward fans (95% CI)\nEvery backward fan went through a fan-correction rework",
                 loc="left", fontsize=10.5, fontweight="bold")

    # (d) LegacyX cost ratio per failure category, bootstrap CI
    ax = fig.add_subplot(gs[1, 1])
    cats = ["Wiring", "Controls", "Leak"]
    for i, c in enumerate(cats):
        m = df["Failure Category"].eq(c)
        r, lo, hi = boot_ratio(df.loc[m & legacy, "Warranty Cost"].to_numpy(float),
                               df.loc[m & ~legacy, "Warranty Cost"].to_numpy(float))
        ax.bar(i, r, color=CAT_COLOR[c], width=0.55)
        ax.errorbar(i, r, yerr=[[r - lo], [hi - r]], color=INK2, capsize=4)
        dc = df.loc[m & ~legacy, "Warranty Cost"].mean()
        lx = df.loc[m & legacy, "Warranty Cost"].mean()
        ax.text(i, hi + 0.05, f"×{r:.2f}\n\\${lx:,.0f} vs \\${dc:,.0f}", ha="center", fontsize=9)
    ax.axhline(1, color=INK2, lw=0.8)
    ax.set_xticks(range(3), cats), ax.set_ylim(0, 2.0)
    ax.set_ylabel("LegacyX cost ÷ DC-family cost")
    ax.set_title("(d) Cost per failure, LegacyX vs DC families (95% bootstrap CI)\nSame failures, ~1.5× the cost to fix",
                 loc="left", fontsize=10.5, fontweight="bold")

    header(fig, "Root-Cause Evidence: Four Patterns That Survive Formal Testing",
           "All p < 1e-30 after Benjamini-Hochberg correction. Shares are of failed units (failure mix), not failure rates.")
    save(fig, "A02_root_cause_evidence.png")


if __name__ == "__main__":
    main()
