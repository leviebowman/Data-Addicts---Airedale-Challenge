"""
05 - Unsupervised segmentation: which build configurations behave alike in the field?

Why a new clustering?
  The existing KMeans (WarrantyCostModel/UnsupervisedModel.py) clusters units on columns that
  include Defect Category, Repair Type and Failure Mode.  Script 01 shows those columns ARE the
  field failure label, so its two clusters reproduce "Wiring vs not Wiring" (ARI 0.997).  That is
  a re-labelling, not a discovery.

What this script does instead
  Unit of analysis  A BUILD CELL = one combination of Facility x Coil Supplier x Controls Revision
                    x Shift (4 x 3 x 3 x 2 = 72 cells, about 42 failed units each).  Product Family
                    is left out: script 02 shows it changes cost per failure, not the failure mix.
  Features          The cell's field-failure mix: share of Loose Power Wiring, Condenser Fan
                    Backwards, Leak (any), Controls (any).  Shares, not counts, so cells of
                    different size compare fairly.
  Method            Agglomerative (Ward) clustering; k chosen by silhouette over k = 2..8.
  Validation        (1) Bootstrap stability: resample units 200x, rebuild cell profiles, recluster,
                    measure Adjusted Rand Index vs the original segmentation.
                    (2) Control: KMeans on the build features alone (one-hot), which tests
                    whether builds have natural groups before looking at outcomes.
  Interpretation    Each segment is named by the build traits its cells share and mapped to the
                    catalog programs that act on it.  Segments describe where failures come from;
                    they are not causal estimates.

Run:  .venv\\Scripts\\python.exe Analysis\\05_segmentation.py
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import OneHotEncoder

from common import (CAT_COLOR, DARK, GRID, INK, INK2, LIGHT, MUTED, OUT, SEED, header, load, money, plt, save)

CELL = ["Facility", "Coil Supplier", "Controls Revision", "Shift"]
MIX = {"Loose wiring": "Loose Power Wiring", "Fan backwards": "Condenser Fan Backwards"}
MIX_COLS = ["Loose wiring", "Fan backwards", "Leak", "Controls"]
MIX_COLOR = {"Loose wiring": CAT_COLOR["Wiring"], "Fan backwards": "#6250d6", "Leak": CAT_COLOR["Leak"],
             "Controls": CAT_COLOR["Controls"]}
K_RANGE = range(2, 9)
rng = np.random.default_rng(SEED)


def failure_type(df):
    return np.select([df["Failure Description"].eq("Loose Power Wiring"),
                      df["Failure Description"].eq("Condenser Fan Backwards"),
                      df["Failure Category"].eq("Leak")],
                     ["Loose wiring", "Fan backwards", "Leak"], "Controls")


def profiles(df):
    t = df.assign(ftype=failure_type(df))
    mix = pd.crosstab([t[c] for c in CELL], t["ftype"], normalize="index").reindex(columns=MIX_COLS, fill_value=0)
    stats = t.groupby(CELL).agg(Units=("Serial Number", "size"), Cost=("Warranty Cost", "sum"))
    return mix.join(stats)


def ward(X, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)


def name_segment(cells):
    """Describe a segment by the build traits shared by (nearly) all its cells."""
    idx = cells.index.to_frame(index=False)
    traits = []
    rule = []
    fac2 = idx["Facility"].eq("Facility 2").mean()
    supB = idx["Coil Supplier"].eq("Supplier B").mean()
    revA = idx["Controls Revision"].eq("Rev A").mean()
    night = idx["Shift"].eq("Night").mean()
    leaky = (idx["Facility"].eq("Facility 2") | idx["Coil Supplier"].eq("Supplier B")).mean()
    traits.append("Facility 2 or Supplier B" if leaky == 1 else ("not Facility 2 / Supplier B" if leaky == 0 else
                                                                    f"{leaky:.0%} F2-or-SupB"))
    traits.append("Rev A" if revA == 1 else ("Rev B/C" if revA == 0 else f"{revA:.0%} Rev A"))
    traits.append("Night" if night == 1 else ("Day" if night == 0 else "Day + Night"))
    return " · ".join(traits)


def main():
    df = load()
    prof = profiles(df)
    X = prof[MIX_COLS].to_numpy()
    print(f"{len(prof)} build cells, {prof['Units'].min()}–{prof['Units'].max()} failed units each")

    # ---------------------------------------------------------------- choose k
    sil = {k: silhouette_score(X, ward(X, k)) for k in K_RANGE}
    k = max(sil, key=sil.get)
    labels = ward(X, k)
    print("Silhouette by k (cell profiles):", {kk: round(v, 3) for kk, v in sil.items()}, "-> k =", k)

    # Control: do BUILD features alone have natural clusters? (unit level, one-hot)
    Xb = OneHotEncoder().fit_transform(df[["Product Family"] + CELL]).toarray()
    sil_build = {kk: silhouette_score(Xb, KMeans(kk, n_init=10, random_state=SEED).fit_predict(Xb),
                                      sample_size=1500, random_state=SEED) for kk in K_RANGE}
    print("Silhouette by k (KMeans on build features only):", {kk: round(v, 3) for kk, v in sil_build.items()})

    # ---------------------------------------------------------------- bootstrap stability
    aris = []
    for b in range(200):
        boot = df.iloc[rng.integers(0, len(df), len(df))]
        pb = profiles(boot).reindex(prof.index).fillna(0)
        aris.append(adjusted_rand_score(labels, ward(pb[MIX_COLS].to_numpy(), k)))
    aris = np.array(aris)
    print(f"Bootstrap stability (200 resamples): ARI median {np.median(aris):.3f}, "
          f"5th pct {np.percentile(aris, 5):.3f}")

    # ---------------------------------------------------------------- profile segments
    prof["Segment"] = labels
    seg_rows = []
    for s, cells in prof.groupby("Segment"):
        w = cells["Units"]
        mix = (cells[MIX_COLS].mul(w, axis=0).sum() / w.sum())
        seg_rows.append({"Segment": s, "Build traits": name_segment(cells), "Cells": len(cells),
                         "Units": int(w.sum()), "Warranty cost": cells["Cost"].sum(),
                         "Cost per failed unit": cells["Cost"].sum() / w.sum(), **mix.to_dict()})
    seg = pd.DataFrame(seg_rows).sort_values("Cost per failed unit", ascending=False).reset_index(drop=True)
    seg["Share of total cost"] = seg["Warranty cost"] / seg["Warranty cost"].sum()
    seg["Programs that act here"] = seg.apply(programs_for, axis=1)
    seg.insert(0, "Segment name", [f"S{i + 1}" for i in range(len(seg))])
    remap = dict(zip(seg["Segment"], seg["Segment name"]))
    prof["Segment name"] = prof["Segment"].map(remap)

    seg.drop(columns="Segment").to_csv(OUT / "A05_segments.csv", index=False)
    prof.reset_index().drop(columns="Segment").to_csv(OUT / "A05_cell_profiles.csv", index=False)
    pd.DataFrame({"k": list(K_RANGE), "Silhouette (cell failure profiles, Ward)": [sil[kk] for kk in K_RANGE],
                  "Silhouette (build features only, KMeans)": [sil_build[kk] for kk in K_RANGE]}) \
        .to_csv(OUT / "A05_silhouette.csv", index=False)
    pd.DataFrame({"Bootstrap ARI": aris}).describe().to_csv(OUT / "A05_stability.csv")

    pd.set_option("display.width", 250), pd.set_option("display.max_colwidth", 60)
    print(seg.drop(columns="Segment").round(3).to_string(index=False))

    fig_segments(seg, sil, sil_build, k, aris)


def programs_for(r):
    p = []
    if r["Leak"] > 0.05:
        p.append("Leak Detection, Braze Training" + (", Supplier QA" if "Supplier B" in r["Build traits"] else ""))
    if r["Controls"] > 0.05:
        p.append("Design Validation, Poka-Yoke (Rev A)")
    if r["Fan backwards"] > 0.05:
        p.append("Poka-Yoke (fan) + fix fan-correction rework")
    p.append("Smart Torque (loose wiring)")
    return "; ".join(p)


def fig_segments(seg, sil, sil_build, k, aris):
    fig = plt.figure(figsize=(15, 8.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.6], wspace=0.12, left=0.06, right=0.98, top=0.82, bottom=0.1)

    ax = fig.add_subplot(gs[0])
    ks = list(K_RANGE)
    ax.plot(ks, [sil[kk] for kk in ks], color=DARK, lw=2, marker="o", ms=6, label="Build cells by failure mix (Ward)")
    ax.plot(ks, [sil_build[kk] for kk in ks], color=MUTED, lw=2, marker="o", ms=6, label="Units by build features only (KMeans)")
    ax.scatter([k], [sil[k]], s=140, facecolor="none", edgecolor=DARK, lw=2, zorder=3)
    ax.annotate(f"k = {k}", (k, sil[k]), xytext=(8, 8), textcoords="offset points", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1), ax.set_xlabel("Number of segments (k)"), ax.set_ylabel("Silhouette score (higher = cleaner)")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
    ax.set_title(f"Choosing k\nBootstrap stability: median ARI {np.median(aris):.2f}", loc="left",
                 fontsize=10.5, fontweight="bold")

    ax = fig.add_subplot(gs[1])
    n = len(seg)
    y = np.arange(n)[::-1]
    left = np.zeros(n)
    for col in MIX_COLS:
        ax.barh(y, seg[col], left=left, color=MIX_COLOR[col], height=0.5, label=col, edgecolor="white", lw=1.5)
        for yi, l, v in zip(y, left, seg[col]):
            if v > 0.07:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", color="white", fontsize=9,
                        fontweight="bold")
        left += seg[col].to_numpy()
    ax.set_xlim(0, 1.95), ax.set_ylim(-0.6, n - 0.2)
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    for yi, (_, r) in zip(y, seg.iterrows()):
        ax.text(0, yi + 0.36, f"{r['Segment name']}   {r['Build traits']}", va="bottom", fontsize=10,
                fontweight="bold", color=INK)
        ax.text(1.03, yi + 0.12, f"{r['Units']:,} units · {r['Cells']} cells · \\${r['Cost per failed unit']:,.0f}/failure · "
                f"{r['Share of total cost']:.0%} of cost", va="bottom", fontsize=9, color=INK)
        ax.text(1.03, yi - 0.05, r["Programs that act here"], va="top", fontsize=8, color=INK2, wrap=True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.26, -0.13), ncol=4, frameon=False, fontsize=9)
    ax.set_title("Segments: failure mix of each segment (bars) and who should act on it", loc="left",
                 fontsize=10.5, fontweight="bold")
    header(fig, "Unsupervised Segmentation of the 72 Build Configurations",
           "Builds have no natural groups on their own (grey line); grouped by how they fail, clear segments appear "
           "that line up with the catalog programs.")
    save(fig, "A05_segments.png")


if __name__ == "__main__":
    main()
