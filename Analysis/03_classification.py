"""
03 - Classification: from the build record alone, which failure will a unit have?

Question   If a unit fails, which failure category (Wiring / Leak / Controls) will it be?
Unit       One unit (serial number). 3,000 units, one row each.
Features   BUILD tier only: Product Family, Facility, Controls Revision, Coil Supplier, Shift.
           (Factory columns are excluded - script 01 shows they ARE the answer.)
Models     Majority class -> class-prior -> one-split decision stump (simple baselines)
           Multinomial logistic regression, decision tree, random forest, gradient boosting.
Validation 20% stratified hold-out never touched until the end.  On the other 80%:
           5-fold stratified CV for tuning and model choice.  Random splits are right here:
           each unit appears once, there are no repeated measurements to leak across folds,
           and the dates are informational only (organizer Q&A #1).
Decision   Turn P(Leak) into a rule: "send the top-k% riskiest builds to extra leak testing",
           and measure the share of leak warranty cost those units account for.

Run:  .venv\\Scripts\\python.exe Analysis\\03_classification.py
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score, log_loss,
                             roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, export_text

from common import (BUILD, CAT_COLOR, CATEGORIES, DARK, FACTORY_CAT, GRID, INK, INK2, LIGHT, MUTED, OUT, RED,
                    SEED, header, load, plt, save)

TARGET = "Failure Category"


def pipe(model, cols=BUILD):
    return Pipeline([("onehot", ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cols)])),
                     ("model", model)])


CANDIDATES = {
    "Majority class (always 'Wiring')": (pipe(DummyClassifier(strategy="most_frequent")), {}),
    "Class prior (base rates)": (pipe(DummyClassifier(strategy="prior")), {}),
    "Decision stump (1 split)": (pipe(DecisionTreeClassifier(max_depth=1, random_state=SEED)), {}),
    "Logistic regression": (pipe(LogisticRegression(max_iter=2000)),
                            {"model__C": [0.01, 0.1, 1, 10, 100]}),
    "Decision tree": (pipe(DecisionTreeClassifier(random_state=SEED)),
                      {"model__max_depth": [2, 3, 4, 5, 6, 8], "model__min_samples_leaf": [1, 10, 30]}),
    "Random forest": (pipe(RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
                      {"model__max_depth": [3, 5, 8, None], "model__min_samples_leaf": [1, 10, 30]}),
    "Gradient boosting": (pipe(HistGradientBoostingClassifier(random_state=SEED)),
                          {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.05, 0.1],
                           "model__max_iter": [100, 300]}),
}


def scores(y, proba, classes):
    pred = classes[proba.argmax(1)]
    leak_col = list(classes).index("Leak")
    return {"Accuracy": (pred == y).mean(), "Balanced accuracy": balanced_accuracy_score(y, pred),
            "Macro F1": f1_score(y, pred, average="macro"),
            "Log loss": log_loss(y, proba, labels=classes),
            "Leak AUC": roc_auc_score(y == "Leak", proba[:, leak_col])}


def main():
    df = load()
    X, y = df, df[TARGET].to_numpy()
    tr, te = train_test_split(np.arange(len(df)), test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    print(f"Train {len(tr)} / test {len(te)} units. Features: {BUILD}")

    rows, fitted = [], {}
    for name, (est, grid) in CANDIDATES.items():
        search = GridSearchCV(est, grid, scoring="neg_log_loss", cv=cv, n_jobs=-1)
        search.fit(X.iloc[tr], y[tr])
        best = search.best_estimator_
        cv_proba = cross_val_predict(best, X.iloc[tr], y[tr], cv=cv, method="predict_proba")
        classes = best.classes_
        cv_s = scores(y[tr], cv_proba, classes)
        test_s = scores(y[te], best.predict_proba(X.iloc[te]), classes)
        fitted[name] = best
        rows.append({"Model": name, "Tuned": {k.replace("model__", ""): v for k, v in search.best_params_.items()} or "-",
                     **{f"CV {k}": v for k, v in cv_s.items()}, **{f"Test {k}": v for k, v in test_s.items()}})
    res = pd.DataFrame(rows)

    # Ceiling: the best any build-time model could do = the true cell frequencies.
    # (Estimated in-sample on all 3,000 units, so optimistic; reported for context only.)
    cell = df.groupby(BUILD)[TARGET].value_counts(normalize=True).unstack(fill_value=0)[sorted(CATEGORIES)]
    ceil_proba = cell.loc[pd.MultiIndex.from_frame(df.iloc[te][BUILD])].to_numpy()
    ceil_proba = np.clip(ceil_proba, 1e-6, 1)
    ceil = scores(y[te], ceil_proba / ceil_proba.sum(1, keepdims=True), np.array(sorted(CATEGORIES)))
    res = pd.concat([res, pd.DataFrame([{"Model": "Ceiling (true cell rates, in-sample)",
                                          "Tuned": "-", **{f"Test {k}": v for k, v in ceil.items()}}])])
    res.to_csv(OUT / "A03_classifier_comparison.csv", index=False)
    cols = ["Model", "CV Log loss", "CV Macro F1", "Test Log loss", "Test Accuracy", "Test Balanced accuracy",
            "Test Macro F1", "Test Leak AUC"]
    pd.set_option("display.width", 250)
    print(res[cols].round(3).to_string(index=False))

    # Chosen model = lowest CV log loss among real models (selection never looks at the test set)
    real = res[~res["Model"].str.startswith(("Majority", "Class prior", "Ceiling"))]
    chosen = real.sort_values("CV Log loss").iloc[0]["Model"]
    model = fitted[chosen]
    prior_ll = res.loc[res["Model"].str.startswith("Class prior"), "Test Log loss"].iloc[0]
    chosen_ll = res.loc[res["Model"] == chosen, "Test Log loss"].iloc[0]
    achieved = (prior_ll - chosen_ll) / (prior_ll - ceil["Log loss"])
    print(f"\nChosen on CV log loss: {chosen}. Test log loss {chosen_ll:.3f} vs prior {prior_ll:.3f} vs ceiling "
          f"{ceil['Log loss']:.3f} -> {achieved:.0%} of the achievable improvement.")

    proba = model.predict_proba(X.iloc[te])
    classes = model.classes_
    pred = classes[proba.argmax(1)]
    cm = confusion_matrix(y[te], pred, labels=CATEGORIES)
    pd.DataFrame(cm, index=[f"true {c}" for c in CATEGORIES], columns=[f"pred {c}" for c in CATEGORIES]) \
        .to_csv(OUT / "A03_confusion_matrix.csv")
    print(pd.DataFrame(cm, index=CATEGORIES, columns=CATEGORIES))

    # Interpretable rules: depth-3 tree on training data
    tree = pipe(DecisionTreeClassifier(max_depth=3, min_samples_leaf=30, random_state=SEED)).fit(X.iloc[tr], y[tr])
    names = tree.named_steps["onehot"].get_feature_names_out()
    names = [n.replace("c__", "") for n in names]
    rules = export_text(tree.named_steps["model"], feature_names=names, show_weights=True)
    (OUT / "A03_tree_rules.txt").write_text(rules, encoding="utf-8")

    # Permutation importance on the test set (which build fields carry the signal)
    # Rank-based score: robust to the tree's zero-probability leaves (log loss explodes on those).
    imp = permutation_importance(model, X.iloc[te][BUILD], y[te], scoring="roc_auc_ovr", n_repeats=30,
                                 random_state=SEED)
    importance = pd.DataFrame({"Feature": BUILD, "Drop in one-vs-rest AUC when shuffled": imp.importances_mean,
                               "sd": imp.importances_std}).sort_values("Drop in one-vs-rest AUC when shuffled",
                                                                       ascending=False)
    importance.to_csv(OUT / "A03_feature_importance.csv", index=False)
    print("\nPermutation importance (test set):")
    print(importance.round(4).to_string(index=False))

    # Shipment-time tier: add factory columns -> shows they are the label, not a modelling success
    ship = pipe(LogisticRegression(max_iter=2000), BUILD + FACTORY_CAT).fit(X.iloc[tr], y[tr])
    ship_acc = (ship.predict(X.iloc[te]) == y[te]).mean()
    print(f"\nWith factory columns added (shipment time): test accuracy = {ship_acc:.1%}  "
          "-> the factory record already names the field failure.")

    # ---------------------------------------------------------------- robustness: chronological split
    # Organizers say dates are informational, and there is no build date.  As a robustness check we
    # still train on the earliest 80% of warranty claims and test on the latest 20%, using the same
    # tuned model, to show the result is not an artefact of random splitting.
    order = np.argsort(pd.to_datetime(df["Claim Date"]).to_numpy(), kind="stable")
    c_tr, c_te = order[: int(0.8 * len(df))], order[int(0.8 * len(df)):]
    from sklearn.base import clone
    chrono = clone(model).fit(X.iloc[c_tr], y[c_tr])
    prior = pipe(DummyClassifier(strategy="prior")).fit(X.iloc[c_tr], y[c_tr])
    cs = scores(y[c_te], chrono.predict_proba(X.iloc[c_te]), chrono.classes_)
    ps_ = scores(y[c_te], prior.predict_proba(X.iloc[c_te]), prior.classes_)
    chrono_df = pd.DataFrame([{"Split": "Chronological (train on earliest 80% of claims)", "Model": chosen,
                               **{f"Test {k}": v for k, v in cs.items()}},
                              {"Split": "Chronological", "Model": "Class prior baseline",
                               **{f"Test {k}": v for k, v in ps_.items()}}])
    chrono_df.to_csv(OUT / "A03_chronological_check.csv", index=False)
    print(f"\nChronological split: {chosen} log loss {cs['Log loss']:.3f} vs prior {ps_['Log loss']:.3f}; "
          f"leak AUC {cs['Leak AUC']:.3f}  (random split: {chosen_ll:.3f})")

    # ---------------------------------------------------------------- decision impact
    te_df = df.iloc[te].copy()
    te_df["P(Leak)"] = proba[:, list(classes).index("Leak")]
    capture = capture_curve(te_df, "P(Leak)", te_df["Failure Category"].eq("Leak") * te_df["Warranty Cost"])
    capture.to_csv(OUT / "A03_leak_flagging_capture.csv", index=False)
    at30 = capture.loc[capture["Share of units flagged"].sub(0.3).abs().idxmin()]
    print(f"\nFlag the top 30% of builds by P(Leak): they hold {at30['Share of leak cost captured']:.0%} "
          f"of test-set leak warranty cost (random flagging: 30%).")
    # Same rule as a plain business rule: 'Facility 2 OR Supplier B'
    rule = te_df["Facility"].eq("Facility 2") | te_df["Coil Supplier"].eq("Supplier B")
    lc = te_df["Failure Category"].eq("Leak") * te_df["Warranty Cost"]
    print(f"Plain rule 'Facility 2 OR Supplier B' flags {rule.mean():.0%} of units and holds "
          f"{lc[rule].sum() / lc.sum():.0%} of leak cost.")

    pd.DataFrame([{"Chosen model": chosen, "Share of achievable log-loss improvement": achieved,"Shipment-tier accuracy": ship_acc,
                   "Top-30% leak cost captured": at30["Share of leak cost captured"],
                   "Rule F2|SupB share flagged": rule.mean(),
                   "Rule F2|SupB leak cost captured": lc[rule].sum() / lc.sum()}]) \
        .to_csv(OUT / "A03_summary.csv", index=False)

    fig_comparison(res)
    fig_confusion_and_capture(cm, capture, rule.mean(), lc[rule].sum() / lc.sum(), chosen)
    fig_calibration(y[te], proba, classes, chosen)


def fig_calibration(y_true, proba, classes, chosen):
    """Reliability: when the model says 40% leak, do ~40% of those units leak? (held-out test set)"""
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import brier_score_loss
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.plot([0, 1], [0, 1], color=MUTED, ls="--", lw=1.5, label="Perfect calibration")
    rows = []
    for c in CATEGORIES:
        j = list(classes).index(c)
        yt = (y_true == c).astype(int)
        frac, mean_p = calibration_curve(yt, proba[:, j], n_bins=6, strategy="quantile")
        brier = brier_score_loss(yt, proba[:, j])
        ax.plot(mean_p, frac, color=CAT_COLOR[c], lw=2, marker="o", ms=7, label=f"{c}  (Brier {brier:.3f})")
        rows.append({"Class": c, "Brier score": brier, "Brier of base rate": yt.mean() * (1 - yt.mean())})
    pd.DataFrame(rows).to_csv(OUT / "A03_calibration.csv", index=False)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"), ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Predicted probability (binned)"), ax.set_ylabel("Observed share in the test set")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.subplots_adjust(left=0.12, right=0.97, top=0.82, bottom=0.11)
    header(fig, "Classifier Calibration (held-out test set)",
           f"{chosen}: points sit near the diagonal; Brier scores 32–37% better than predicting base rates.")
    save(fig, "A03_calibration.png")


def capture_curve(frame, score_col, value):
    order = frame[score_col].sort_values(ascending=False, kind="stable").index
    v = value.loc[order].to_numpy()
    return pd.DataFrame({"Share of units flagged": np.arange(1, len(v) + 1) / len(v),
                         "Share of leak cost captured": v.cumsum() / v.sum()})


def fig_comparison(res):
    show = res[res["Model"] != "Class prior (base rates)"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, metric, title, better in [(axes[0], "Test Macro F1", "Macro F1 (higher is better)", "high"),
                                      (axes[1], "Test Leak AUC", "Leak AUC: ranks leak-prone builds (higher is better)", "high")]:
        y = np.arange(len(show))[::-1]
        colors = [MUTED if m.startswith(("Majority", "Decision stump")) else
                  (GRID if m.startswith("Ceiling") else DARK) for m in show["Model"]]
        ax.barh(y, show[metric], color=colors, height=0.6)
        for yi, v in zip(y, show[metric]):
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
        ax.set_yticks(y, show["Model"] if ax is axes[0] else [""] * len(show), fontsize=9)
        ax.set_xlim(0, 1.08)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold")
    fig.subplots_adjust(left=0.24, right=0.98, top=0.8, bottom=0.08, wspace=0.08)
    header(fig, "Build-Time Failure Classifier vs. Baselines (held-out test set, 600 units)",
           "Grey = simple baselines. Every model sees only Family, Facility, Controls Rev, Coil Supplier, Shift.")
    save(fig, "A03_classifier_comparison.png")


def fig_confusion_and_capture(cm, capture, rule_share, rule_capture, chosen):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{norm[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center", fontsize=10,
                    color="white" if norm[i, j] > 0.5 else INK)
    ax.set_xticks(range(3), [f"pred\n{c}" for c in CATEGORIES]), ax.set_yticks(range(3), [f"true {c}" for c in CATEGORIES])
    ax.grid(False)
    ax.set_title(f"{chosen}: confusion matrix (row %)\nControls is never the single most likely class "
                 "(max 32%, Rev A)", loc="left", fontsize=10.5, fontweight="bold")

    ax = axes[1]
    ax.plot(capture["Share of units flagged"], capture["Share of leak cost captured"], color=CAT_COLOR["Leak"], lw=2,
            label="Ranked by model P(Leak)")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1.5, ls="--", label="Random flagging")
    ax.scatter([rule_share], [rule_capture], s=70, color=DARK, zorder=3,
               label=f"Rule 'Facility 2 OR Supplier B': flag {rule_share:.0%}, catch {rule_capture:.0%}")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}"), ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Share of builds sent to extra leak testing"), ax.set_ylabel("Share of leak warranty cost in those builds")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Decision use: which builds to send to extra leak testing", loc="left", fontsize=10.5,
                 fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.8, bottom=0.13, wspace=0.25)
    header(fig, "Leaks Are Predictable at Build Time; Wiring vs. Controls Mostly Are Not",
           "Held-out test set. Half of the builds hold every leak dollar, which is the targeting behind Leak Detection.")
    save(fig, "A03_confusion_and_capture.png")


if __name__ == "__main__":
    main()
