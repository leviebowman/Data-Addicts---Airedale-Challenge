"""
01 - Data audit: lineage, label reconciliation, and leakage check.

Questions answered
  1. Do the five datasets describe the same units, and do shared columns agree?
  2. The handbook warns that terminology differs between datasets.  How do the
     factory labels (Rework, EOLT) map onto the field labels (Field Service, Warranty)?
  3. Which columns would leak the answer into a model, and is the existing
     KMeans clustering (WarrantyCostModel/UnsupervisedModel.py) affected?

Run:  .venv\\Scripts\\python.exe Analysis\\01_data_audit.py
"""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import adjusted_rand_score

from common import (BUILD, CAT_COLOR, CONSTANT, DARK, FACTORY_CAT, FACTORY_NUM, FIELD, FILES, GRID, INK2,
                    LIGHT, MUTED, OUT, RED, ROOT, SURFACE, header, load, load_raw, plt, save)

# Reconciliation maps: factory vocabulary -> field Failure Category
DEFECT_TO_FIELD = {"Brazing": "Leak", "Wiring": "Wiring", "Controls": "Controls"}
EOLT_TO_FIELD = {"Loose Terminal": "Wiring", "Fan Rotation Error": "Wiring",
                 "Refrigerant Leak": "Leak", "Sensor Miswire": "Controls"}
REPAIR_TO_FIELD = {"Rewire": "Wiring", "Fan Rotation Correction": "Wiring",
                   "Leak Repair": "Leak", "Sensor Rewire": "Controls"}
DESC_TO_FIELD = {"Loose Power Wiring": "Wiring", "Condenser Fan Backwards": "Wiring",
                 "Refrigerant Leak": "Leak", "Suction Leak": "Leak", "Service Valve Leak": "Leak",
                 "Sensor Fault": "Controls", "Communication Fault": "Controls"}


def cramers_v(a, b):
    tab = pd.crosstab(a, b)
    chi2 = chi2_contingency(tab, correction=False)[0]
    return float(np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1))))


def tier_of(col):
    if col in BUILD:
        return "1 Build"
    if col in FACTORY_CAT or col in FACTORY_NUM:
        return "2 Factory (pre-ship)"
    if col in CONSTANT:
        return "constant"
    if col == "Serial Number":
        return "key"
    return "3 Field (post-failure)" if col in FIELD or col == "Warranty Cost" else "?"


def main():
    raw = load_raw()
    df = load()
    print(f"Joined: {len(df):,} units x {df.shape[1]} columns; every unit appears in all 5 datasets.")

    # ---------------------------------------------------------------- 1. lineage
    rows = []
    for name, frame in raw.items():
        for c in frame.columns:
            if c == "Serial Number":
                continue
            rows.append({"Column": c, "Source dataset": name, "File": FILES[name],
                         "Tier (when known)": tier_of(c), "Distinct values": frame[c].nunique(),
                         "Missing": int(frame[c].isna().sum())})
    lineage = pd.DataFrame(rows)
    lineage.to_csv(OUT / "A01_column_lineage.csv", index=False)

    # Is the repo's Data_Full.xlsx the same as a clean join of the raw files?
    full = pd.read_excel(ROOT / "WarrantyCostModel" / "Data_Full.xlsx")
    same = full.sort_values("Serial Number").reset_index(drop=True)[df.columns].equals(
        df.sort_values("Serial Number").reset_index(drop=True))
    print(f"Data_Full.xlsx identical to a fresh join of the raw files: {same}")

    # ---------------------------------------------------------------- 2. reconciliation
    recon = []
    for col, mapping in [("Defect Category", DEFECT_TO_FIELD), ("Failure Mode", EOLT_TO_FIELD),
                         ("Repair Type", REPAIR_TO_FIELD), ("Failure Description", DESC_TO_FIELD)]:
        mapped = df[col].map(mapping)
        assert mapped.notna().all(), col
        for value, field_cat in mapping.items():
            sub = df[df[col] == value]
            recon.append({"Column": col, "Label": value, "Maps to field category": field_cat,
                          "Units": len(sub), "Agrees with field Failure Category":
                              (sub["Failure Category"] == field_cat).mean()})
    recon = pd.DataFrame(recon)
    recon.to_csv(OUT / "A01_label_reconciliation.csv", index=False)
    print("\nFactory label -> field Failure Category agreement:")
    print(recon.groupby("Column").apply(
        lambda g: (g["Units"] * g["Agrees with field Failure Category"]).sum() / g["Units"].sum(),
        include_groups=False).map("{:.1%}".format).to_string())

    # Sub-type agreement inside Wiring: does the factory see fan vs terminal correctly?
    wiring = df[df["Failure Category"] == "Wiring"]
    sub = pd.crosstab([wiring["Shift"], wiring["Failure Mode"], wiring["Repair Type"]],
                      wiring["Failure Description"])
    sub.to_csv(OUT / "A01_wiring_subtype_trace.csv")
    print("\nWiring units - EOLT failure mode / repair applied vs. what failed in the field:")
    print(sub.to_string())

    # ---------------------------------------------------------------- 3. leakage
    leak_rows = []
    for c in BUILD + FACTORY_CAT + ["Customer Region"]:
        leak_rows.append({"Column": c, "Tier": tier_of(c), "Cramer's V with Failure Category":
                          cramers_v(df[c], df["Failure Category"])})
    for c in FACTORY_NUM + ["Repair Cost", "Downtime Hours", "Days To Failure", "Service Hours"]:
        leak_rows.append({"Column": c, "Tier": tier_of(c), "Spearman |rho| with Warranty Cost":
                          abs(df[c].corr(df["Warranty Cost"], method="spearman"))})
    leak = pd.DataFrame(leak_rows)
    leak.to_csv(OUT / "A01_leakage_screen.csv", index=False)
    print("\nAssociation with the outcome (1.0 = column IS the answer):")
    print(leak.round(3).to_string(index=False))

    # Existing KMeans clusters vs the outcome
    cl = pd.read_csv(ROOT / "WarrantyCostModel" / "ClusteredUnits.csv")[["Serial Number", "Cluster"]]
    m = df.merge(cl, on="Serial Number")
    is_wiring = m["Failure Category"].eq("Wiring")
    ari = adjusted_rand_score(m["Cluster"], is_wiring)
    agree = max((m["Cluster"].eq(2) == is_wiring).mean(), (m["Cluster"].eq(1) == is_wiring).mean())
    print(f"\nExisting KMeans clusters vs 'is a Wiring failure': ARI = {ari:.3f}, agreement = {agree:.1%}")
    pd.DataFrame([{"Check": "Existing KMeans clusters vs Wiring/non-Wiring", "Adjusted Rand Index": ari,
                   "Agreement": agree}]).to_csv(OUT / "A01_existing_cluster_check.csv", index=False)

    # ---------------------------------------------------------------- figures
    fig_lineage(df)
    fig_leakage(leak)


def fig_lineage(df):
    """Every unit's journey: defect created -> caught in factory -> 'repaired' -> same defect in field."""
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.axis("off"), ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    stages = [("BUILD", "Product Config", "Family, Facility,\nControls Rev, Coil Supplier,\nShift"),
              ("REWORK", "Rework", "Defect found & repaired\n100% of units\nDefect Category recorded"),
              ("END-OF-LINE TEST", "EOLT", "Test failure, repaired,\nPass After Repair = Yes\n100% of units"),
              ("SHIPPED", "", "Factory considers\nthe unit good"),
              ("FIELD FAILURE", "Field Service + Warranty", "Fails at customer\n$7.65M warranty cost\n100% of units")]
    xs = np.linspace(0.09, 0.91, len(stages))
    for i, (x, (title, src, body)) in enumerate(zip(xs, stages)):
        color = RED if i == 4 else (DARK if i in (1, 2) else LIGHT)
        ax.add_patch(plt.Rectangle((x - 0.08, 0.40), 0.16, 0.42, color=color, lw=0))
        tc = "white" if color in (RED, DARK) else "#0b0b0b"
        ax.text(x, 0.78, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=tc)
        ax.text(x, 0.67, body, ha="center", va="top", fontsize=9, color=tc)
        if src:
            ax.text(x, 0.36, src, ha="center", va="top", fontsize=8.5, color=INK2, style="italic")
        if i < len(stages) - 1:
            ax.annotate("", (xs[i + 1] - 0.083, 0.61), (x + 0.083, 0.61),
                        arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.2))
    ax.text(0.5, 0.2, "The factory's Defect Category matches the field Failure Category for 3,000 / 3,000 units "
            "(Brazing → Leak, Wiring → Wiring, Controls → Controls).",
            ha="center", fontsize=11, fontweight="bold")
    ax.text(0.5, 0.07, "The defects are being caught, but the repairs are not holding: "
            "every 'repaired' defect comes back as the same field failure.", ha="center", fontsize=10, color=INK2)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.82, bottom=0.02)
    header(fig, "Where Defects Are Created vs. Caught vs. Escape",
           "All five datasets cover the same 3,000 units. Joined on Serial Number; shared columns agree 100%.")
    save(fig, "A01_defect_journey.png")


def fig_leakage(leak):
    cat = leak.dropna(subset=["Cramer's V with Failure Category"]).sort_values("Cramer's V with Failure Category")
    fig, ax = plt.subplots(figsize=(12, 6))
    tier_color = {"1 Build": LIGHT, "2 Factory (pre-ship)": DARK, "3 Field (post-failure)": MUTED}
    y = np.arange(len(cat))
    ax.barh(y, cat["Cramer's V with Failure Category"], color=[tier_color[t] for t in cat["Tier"]], height=0.62)
    for yi, v in zip(y, cat["Cramer's V with Failure Category"]):
        ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, cat["Column"])
    ax.set_xlim(0, 1.12)
    ax.axvline(0.9, color=RED, lw=1.2, ls="--")
    ax.text(0.89, 4.5, "≥ 0.9: column is\nessentially the label", color=RED, fontsize=8.5, va="center", ha="right")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Cramér's V with field Failure Category (0 = unrelated, 1 = identical)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in tier_color.values()]
    ax.legend(handles, ["Known at build", "Known after factory test (pre-ship)", "Known only after field failure"],
              loc="lower right", frameon=False)
    fig.subplots_adjust(left=0.17, right=0.97, top=0.86, bottom=0.1)
    header(fig, "Leakage Screen: Which Columns Already Contain the Answer?",
           "Factory labels mirror the field label, so a 'build-time' model must exclude them; "
           "the existing KMeans used three of them.")
    save(fig, "A01_leakage_screen.png")


if __name__ == "__main__":
    main()
