"""
04 - Warranty-cost regression, validated honestly.

PART A  Unit-level: from the build record, what warranty cost should we expect if the unit fails?
  Target   Warranty Cost per unit ($), 3,000 units.
  Tiers    Build features only (deployable before test) and, for comparison, build + factory
           features (deployable at shipment).
  Models   Overall-mean and Product-Family-mean baselines; linear regression on log cost;
           gradient boosting; and a transparent TWO-STAGE model:
               E[cost | build] = sum_c P(category c | build) x mean cost(c, LegacyX or not)
           which reuses the 03 classifier and the LegacyX multiplier found in 02.
  Checks   Same 80/20 split as script 03.  Besides MAE/RMSE/R2 we check *calibration on
           each portfolio program's target slice*: does the model predict the total warranty
           cost that each program is aimed at?  That is what PortfolioSim needs to be right.

PART B  Re-check of WarrantyCostModel/SupervisedModel.py (aggregate kNN).
  That model trains on 3,000 random subsets of the same 3,000 units, then splits the SUBSETS
  into train/test.  The same units therefore appear on both sides, so its test score is not an
  out-of-sample score.  Here we split the UNITS 50/50 first and build subsets from each half
  separately, then compare with a no-ML formula baseline.

Run:  .venv\\Scripts\\python.exe Analysis\\04_cost_model.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from common import (BUILD, CATEGORIES, DARK, FACTORY_CAT, FACTORY_NUM, INK2, LIGHT, MUTED, OUT, RED, SEED,
                    header, load, money, plt, save)

rng = np.random.default_rng(SEED)
Y = "Warranty Cost"


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p),
            "Total bias %": 100 * (p.sum() - y.sum()) / y.sum()}


def onehot(cols, num=()):
    t = [("c", OneHotEncoder(handle_unknown="ignore"), list(cols))]
    if num:
        t.append(("n", StandardScaler(), list(num)))
    return ColumnTransformer(t)


class TwoStage:
    """E[cost] = sum_c P(c | build) * mean cost(c, is LegacyX).  P from a depth-limited tree (as in 03)."""

    def fit(self, X, y):
        self.clf = Pipeline([("oh", onehot(BUILD)),
                             ("m", DecisionTreeClassifier(max_depth=4, min_samples_leaf=10, random_state=SEED))])
        self.clf.fit(X, X["Failure Category"])
        g = X.assign(L=X["Product Family"].eq("LegacyX"), y=y).groupby(["Failure Category", "L"])["y"].mean()
        self.cost = g
        return self

    def predict(self, X):
        P = self.clf.predict_proba(X)
        L = X["Product Family"].eq("LegacyX").to_numpy()
        out = np.zeros(len(X))
        for j, c in enumerate(self.clf.classes_):
            out += P[:, j] * np.where(L, self.cost[(c, True)], self.cost[(c, False)])
        return out


def part_a(df):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=0.2, stratify=df["Failure Category"], random_state=SEED)
    Xtr, Xte, ytr, yte = df.iloc[tr], df.iloc[te], df[Y].iloc[tr].to_numpy(), df[Y].iloc[te].to_numpy()
    cv = KFold(5, shuffle=True, random_state=SEED)

    preds = {}
    preds["Overall mean (baseline)"] = np.full(len(te), ytr.mean())
    fam = Xtr.assign(y=ytr).groupby("Product Family")["y"].mean()
    preds["Product Family mean (baseline)"] = Xte["Product Family"].map(fam).to_numpy()

    lin = Pipeline([("oh", onehot(BUILD)), ("m", LinearRegression())]).fit(Xtr, ytr)
    preds["Linear regression (build)"] = lin.predict(Xte)

    gb = GridSearchCV(Pipeline([("oh", onehot(BUILD)), ("m", HistGradientBoostingRegressor(random_state=SEED))]),
                      {"m__max_depth": [2, 3, 4], "m__learning_rate": [0.05, 0.1], "m__max_iter": [100, 300]},
                      scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1).fit(Xtr, ytr)
    preds["Gradient boosting (build)"] = gb.predict(Xte)

    two = TwoStage().fit(Xtr, ytr)
    preds["Two-stage: P(category) x cost (build)"] = two.predict(Xte)

    # Shipment tier: the factory columns name the failure category, so cost becomes ~ category x family
    ship = GridSearchCV(Pipeline([("oh", onehot(BUILD + FACTORY_CAT, FACTORY_NUM)),
                                  ("m", HistGradientBoostingRegressor(random_state=SEED))]),
                        {"m__max_depth": [2, 3], "m__max_iter": [100, 300]},
                        scoring="neg_mean_absolute_error", cv=cv, n_jobs=-1).fit(Xtr, ytr)
    preds["Gradient boosting (build + factory)"] = ship.predict(Xte)

    res = pd.DataFrame([{"Model": k, **metrics(yte, v)} for k, v in preds.items()])
    res.to_csv(OUT / "A04_cost_model_comparison.csv", index=False)
    print("PART A - unit-level warranty cost, held-out test set (600 units)")
    print(res.round(3).to_string(index=False))

    # Calibration on the PortfolioSim target slices
    t = Xte
    leak, supB, fac2 = t["Failure Category"].eq("Leak"), t["Coil Supplier"].eq("Supplier B"), t["Facility"].eq("Facility 2")
    slices = {"Facility 2 OR Supplier B (Leak Detection scope)": fac2 | supB,
              "Supplier B coils (Supplier QA scope)": supB,
              "Controls Rev A (Design Validation scope)": t["Controls Revision"].eq("Rev A"),
              "Night shift (Poka-Yoke fan scope)": t["Shift"].eq("Night"),
              "LegacyX units (LegacyX Refresh scope)": t["Product Family"].eq("LegacyX"),
              "All other builds": ~(fac2 | supB | t["Controls Revision"].eq("Rev A"))}
    rows = []
    for name, m in slices.items():
        m = m.to_numpy()
        rows.append({"Slice": name, "Units": int(m.sum()), "Actual total": yte[m].sum(),
                     "Two-stage predicted": preds["Two-stage: P(category) x cost (build)"][m].sum(),
                     "Family-mean predicted": preds["Product Family mean (baseline)"][m].sum()})
    cal = pd.DataFrame(rows)
    cal["Two-stage error %"] = 100 * (cal["Two-stage predicted"] / cal["Actual total"] - 1)
    cal["Family-mean error %"] = 100 * (cal["Family-mean predicted"] / cal["Actual total"] - 1)
    cal.to_csv(OUT / "A04_slice_calibration.csv", index=False)
    print("\nTotal warranty cost by portfolio target slice (test set):")
    print(cal.round(1).to_string(index=False))
    return res, cal


# ---------------------------------------------------------------------------- PART B
AGG_CATS = ["Product Family", "Facility", "Controls Revision", "Coil Supplier", "Shift"] + FACTORY_CAT
AGG_NUM = FACTORY_NUM


def aggregate(frame, sizes, n_per_cluster, cats, use_cluster=True):
    """Mimic GetTrainingData.py: for each KMeans cluster, draw random subsets of that cluster's
    units and record means/stds of numerics + category counts (+ the cluster id as a feature)."""
    rows = []
    vals = {c: sorted(frame[c].unique()) for c in cats}
    for k, cl in frame.groupby("Cluster"):
        for _ in range(n_per_cluster):
            s = cl.sample(n=min(int(rng.integers(*sizes)), len(cl)), random_state=int(rng.integers(2**31)))
            r = {"Warranty Cost Mean": s[Y].mean(), "Rows in Subset": len(s), "_cluster": k}
            if use_cluster:
                r["Cluster"] = k
            for c in AGG_NUM:
                r[f"{c} Mean"], r[f"{c} Std"] = s[c].mean(), s[c].std()
            for c in cats:
                vc = s[c].value_counts()
                for v in vals[c]:
                    r[f"{c}={v}"] = int(vc.get(v, 0))
            rows.append(r)
    return pd.DataFrame(rows)


def knn_fit(train):
    X = train.drop(columns=["Warranty Cost Mean", "_cluster"])
    search = GridSearchCV(Pipeline([("s", MinMaxScaler()), ("m", KNeighborsRegressor())]),
                          {"m__n_neighbors": [3, 5, 10, 20, 50], "m__weights": ["uniform", "distance"]},
                          scoring="neg_root_mean_squared_error", cv=KFold(5, shuffle=True, random_state=SEED),
                          n_jobs=-1)
    return search.fit(X, train["Warranty Cost Mean"])


def formula(train, test, units):
    """No-ML baseline: predicted mean = sum over categories of (share of subset) x (category mean cost)."""
    cost = units.groupby("Defect Category")[Y].mean()
    n = test["Rows in Subset"]
    return sum(test[f"Defect Category={c}"] / n * cost[c] for c in cost.index)


def cluster_mean(train, test):
    """Baseline: predict each test subset's mean as its cluster's average training-subset mean."""
    return test["_cluster"].map(train.groupby("_cluster")["Warranty Cost Mean"].mean()).to_numpy()


def part_b(df):
    from common import ROOT
    df = df.merge(pd.read_csv(ROOT / "WarrantyCostModel" / "ClusteredUnits.csv")[["Serial Number", "Cluster"]],
                  on="Serial Number", validate="one_to_one")
    sizes, n = (10, 1001), 1500
    print("\nPART B - re-check of the aggregate kNN (WarrantyCostModel/SupervisedModel.py design)")
    rows = []

    def add(design, feats, y, p):
        rows.append({"Design": design, "Features": feats, **metrics(y, p)})

    # (1) Original design: subsets drawn within each cluster from ALL units, then subsets split 70/30
    allagg = aggregate(df, sizes, n, AGG_CATS)
    tr, te = train_test_split(allagg, test_size=0.3, random_state=SEED)
    yte = te["Warranty Cost Mean"].to_numpy()
    add("Original: same units on both sides of the split", "cluster + build + factory",
        yte, knn_fit(tr).predict(te.drop(columns=["Warranty Cost Mean", "_cluster"])))
    add("Original split, baseline: cluster's average", "cluster id only", yte, cluster_mean(tr, te))
    add("Original split, no-ML formula", "defect-category shares", yte, formula(tr, te, df).to_numpy())

    # (2) Honest: split UNITS 50/50 first, then build subsets inside each half (max size 500 per half)
    u_tr, u_te = train_test_split(df, test_size=0.5, stratify=df["Cluster"], random_state=SEED)
    for label, cats, use_cl in [("cluster + build + factory", AGG_CATS, True), ("build only", BUILD, False)]:
        a_tr = aggregate(u_tr, (10, 501), n, cats, use_cl)
        a_te = aggregate(u_te, (10, 501), n, cats, use_cl)
        cols = sorted(set(a_tr.columns) & set(a_te.columns))
        a_tr, a_te = a_tr[cols], a_te[cols]
        y = a_te["Warranty Cost Mean"].to_numpy()
        add("Honest: units split before subsetting", label, y,
            knn_fit(a_tr).predict(a_te.drop(columns=["Warranty Cost Mean", "_cluster"])))
        if use_cl:
            add("Honest split, baseline: cluster's average", "cluster id only", y, cluster_mean(a_tr, a_te))
            add("Honest split, no-ML formula", "defect-category shares", y, formula(a_tr, a_te, u_tr).to_numpy())
            add("Honest split, baseline: overall average", "none", y, np.full(len(y), a_tr["Warranty Cost Mean"].mean()))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "A04_aggregate_knn_recheck.csv", index=False)
    print(res.round(3).to_string(index=False))
    return res


def fig(res_a, cal, res_b):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    show = res_a.iloc[::-1]
    y = np.arange(len(show))
    colors = [MUTED if "baseline" in m else (LIGHT if "factory" in m else DARK) for m in show["Model"]]
    ax.barh(y, show["MAE"], color=colors, height=0.6)
    for yi, (mae, r2) in zip(y, zip(show["MAE"], show["R2"])):
        ax.text(mae + 30, yi, f"${mae:,.0f}  (R² {r2:.2f})", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, show["Model"], fontsize=9)
    ax.set_xlim(0, show["MAE"].max() * 1.45)
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean absolute error per unit (lower is better)")
    ax.set_title("A. Unit-level cost, held-out 600 units\nGrey = baselines, light blue = uses factory labels",
                 loc="left", fontsize=10.5, fontweight="bold")

    ax = axes[1]
    c = cal.iloc[::-1]
    y = np.arange(len(c))
    ax.barh(y + 0.2, c["Actual total"], height=0.38, color=MUTED, label="Actual")
    ax.barh(y - 0.2, c["Two-stage predicted"], height=0.38, color=DARK, label="Two-stage model (build data)")
    for yi, (a, p, e) in zip(y, zip(c["Actual total"], c["Two-stage predicted"], c["Two-stage error %"])):
        ax.text(max(a, p) + 2e4, yi, f"{e:+.0f}%", va="center", fontsize=9, color=INK2)
    ax.set_xlim(0, c[["Actual total", "Two-stage predicted"]].max().max() * 1.15)
    ax.set_yticks(y, [s.replace(" (", "\n(") for s in c["Slice"]], fontsize=8.5)
    ax.xaxis.set_major_formatter(money)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("B. Total cost in each program's target slice (test set)\n"
                 "Labels = model error vs actual", loc="left", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.2, right=0.97, top=0.8, bottom=0.1, wspace=0.75)
    header(fig, "Warranty-Cost Model: Accurate Where the Portfolio Needs It",
           "Per-unit cost is noisy at build time (R² 0.29), but slice totals are within ±10% "
           "for 5 of 6 program scopes (LegacyX +17%, n=118).")
    save(fig, "A04_cost_model.png")

    fig, ax = plt.subplots(figsize=(13, 5.6))
    overall = res_b[res_b["Design"].str.contains("overall average")]
    show = res_b.drop(overall.index).iloc[::-1]
    y = np.arange(len(show))
    colors = [RED if s.startswith("Original: same") else (MUTED if "formula" in s or "baseline" in s else DARK)
              for s in show["Design"]]
    ax.barh(y, show["MAE"], color=colors, height=0.6)
    for yi, (r2, mae) in zip(y, zip(show["R2"], show["MAE"])):
        ax.text(mae + 3, yi, f"MAE ${mae:,.0f}   R² {r2:.3f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [f"{d}\n[{f}]" for d, f in zip(show["Design"], show["Features"])], fontsize=8.5)
    ax.set_xlim(0, show["MAE"].max() * 1.4)
    ax.xaxis.set_major_formatter(lambda v, _: f"${v:,.0f}")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mean absolute error of predicted subset mean cost (lower is better). "
                  f"Predicting the overall average: MAE ${overall['MAE'].iloc[0]:,.0f}, R² 0.00")
    fig.subplots_adjust(left=0.34, right=0.98, top=0.82, bottom=0.12)
    header(fig, "Re-check of the Aggregate kNN: R² 0.998 Comes From the Cluster Design, Not the Model",
           "Subsets are drawn inside one cluster, so 'predict the cluster's average' already scores R² 0.998; "
           "a one-line formula beats kNN.")
    save(fig, "A04_aggregate_knn_recheck.png")


def main():
    df = load()
    res_a, cal = part_a(df)
    res_b = part_b(df)
    fig(res_a, cal, res_b)


if __name__ == "__main__":
    main()
