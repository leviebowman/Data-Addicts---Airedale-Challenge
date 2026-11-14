"""
Shared data loading, feature tiers, and chart styling for the Analysis/ scripts.

Every script in this folder imports from here so that:
  * the five raw workbooks are joined the same way everywhere,
  * every column is tagged with the dataset it came from and WHEN that
    information exists in a unit's life (build -> factory test -> field),
  * charts share one look with the PortfolioSim figures.
"""
import sys
from pathlib import Path

import matplotlib

sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "Data"
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 42

FILES = {
    "Field Service": "Competition_Dataset1_Field_Service_Final.xlsx",
    "Warranty": "Competition_Dataset2_Warranty_Final.xlsx",
    "Product Config": "Competition_Dataset3_Product_Config_Final.xlsx",
    "Rework": "Competition_Dataset4_Rework_Final.xlsx",
    "EOLT": "Competition_Dataset5_EOLT_Final.xlsx",
}

# ----------------------------------------------------------------------------
# Feature tiers = WHEN a column is known.  This is what keeps models honest:
# a model may only use columns that exist at the moment it would be run.
# ----------------------------------------------------------------------------
# Tier 1 - BUILD: known the moment the unit is built (bill of materials + line).
BUILD = ["Product Family", "Facility", "Controls Revision", "Coil Supplier", "Shift"]
# Tier 2 - FACTORY: known after rework + end-of-line test, before shipment.
FACTORY_CAT = ["Defect Category", "Repair Type", "Disposition Code", "Test Station", "Failure Mode"]
FACTORY_NUM = ["Labor Hours", "Material Cost", "Rework Hours"]
# Tier 3 - FIELD: only known after the unit has failed at the customer.
#          Never allowed as a predictor.  (Repair Cost is exactly Warranty Cost / 2.)
FIELD = ["Customer Region", "Failure Category", "Failure Description", "Downtime Hours",
         "Repair Cost", "Claim Date", "Days To Failure", "Service Hours", "Date"]
CONSTANT = ["Warranty Eligible", "Pass After Repair", "Retested"]  # one value for every unit

TARGET_COST = "Warranty Cost"
TARGET_CLASS = "Failure Category"
CATEGORIES = ["Wiring", "Leak", "Controls"]

# ----------------------------------------------------------------------------
# Chart style - same surface / ink / blues as PortfolioSim so the deck matches.
# Failure categories use the first three validated categorical slots.
# ----------------------------------------------------------------------------
SURFACE, INK, INK2, GRID, BAND = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#f3f2ef"
DARK, LIGHT, RED, MUTED = "#1d3f8f", "#9ec5f4", "#d64541", "#b9b8b2"
CAT_COLOR = {"Wiring": "#2a78d6", "Leak": "#eb6834", "Controls": "#1baf7a"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


def header(fig, title, sub):
    fig.text(0.012, 0.975, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.012, 0.925, sub, fontsize=10, color=INK2, va="top")


def save(fig, name):
    fig.savefig(OUT / name, dpi=160)
    plt.close(fig)
    print(f"  saved outputs/{name}")


def money(x, _=None):
    if abs(x) >= 1e6:
        return f"${x / 1e6:.2f}M"
    return f"${x / 1e3:.0f}K" if abs(x) >= 1e3 else f"${x:,.0f}"


def load_raw() -> dict[str, pd.DataFrame]:
    return {name: pd.read_excel(DATA_DIR / f) for name, f in FILES.items()}


def load() -> pd.DataFrame:
    """Join the five raw workbooks 1:1 on Serial Number and check shared columns agree.

    Built from the raw files (not WarrantyCostModel/Data_Full.xlsx) so the
    cleaning is reproducible; the result is identical to Data_Full.
    """
    raw = load_raw()
    df = None
    for name, frame in raw.items():
        if df is None:
            df = frame.copy()
            continue
        shared = [c for c in frame.columns if c in df.columns and c != "Serial Number"]
        merged = df.merge(frame, on="Serial Number", how="outer", validate="one_to_one",
                          suffixes=("", f" [{name}]"), indicator=True)
        if not merged["_merge"].eq("both").all():
            raise ValueError(f"{name}: serial numbers do not match the other datasets")
        for c in shared:  # e.g. Facility appears in Config, Rework and EOLT
            if not merged[c].equals(merged[f"{c} [{name}]"]):
                raise ValueError(f"{name}: column {c!r} disagrees with an earlier dataset")
            merged = merged.drop(columns=f"{c} [{name}]")
        df = merged.drop(columns="_merge")
    return df
