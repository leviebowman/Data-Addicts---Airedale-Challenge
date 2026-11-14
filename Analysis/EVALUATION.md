# How Well Does the Project Answer the Data Science Challenge?

An assessment against the Data Science judging criteria in the Competition Handbook (v3). "Before" is the repo as of commit `5e931b4` (`WarrantyCostModel\`, `PortfolioSim\`); "After" adds `Analysis\`.

## Scorecard against the official D1–D8 criteria

| Criterion (theme weight) | Before | After | Evidence (where to find it) |
|---|---|---|---|
| **D1 Problem formulation** (Validity 25%) | Weak: no stated questions | Strong | 5 questions, each with target, unit, method and validation (table below; METHODS §2) |
| **D2 Data prep & validity** (Validity) | Weak: factory labels used as features; no script for `Data_Full` | Strong | Reproducible join with agreement checks; label reconciliation map; build/factory/field tiers; leakage screen (README §01) |
| **D3 Modeling approach & rigor** (Rigor 15%) | Medium: kNN only, no rationale | Strong | 4 classifier families vs 3 baselines; why each method (METHODS §2); shares, not counts, for unequal group sizes; BH correction |
| **D4 Validation design** (Evaluation 10%) | Weak: aggregate kNN R² 0.998 matched by a lookup | Strong | Untouched 600-unit hold-out; CV for tuning; chronological robustness split; bootstrap ARI for clusters; unit-first split re-check |
| **D5 Predictive performance** (Predictive power 15%) | Unclear | Strong | Log loss 0.857 → 0.485 (90% of ceiling); leak AUC 0.84; Brier 37% better; cost R² 0.29 vs 0.00; slice totals +3% vs −41% |
| **D6 Interpretation & insight** (Insight 25%) | Medium: clusters were a re-labelling | Strong | "Repairs don't hold" finding; 4 interpretable segments; tree rules = the 3 root causes; predictive-vs-causal table (METHODS §5) |
| **D7 Decision impact** (Impact) | Strong: PortfolioSim | Stronger | Unit-flagging rule (48% flagged → 100% leak $); about 595 failures and $3.38M/yr avoided; claim lag, break-even, tornado; still best in 98% of runs |
| **D8 Communication & transparency** (Communication 10%) | n/a | Ready | One-slide workflow; METHODS.md rebuild spec; every claim mapped to a chart in SLIDE_PLAN.md |

## What was already strong

- **PortfolioSim** is decision-grade: targeted scopes, stacking that avoids double counting (1 − ∏(1 − b)), implementation delay, discounting, and a 5,000-run Monte Carlo over cost, benefit and delay. It answers the catalog's "test how robust a portfolio is" directly.
- The **unit-level kNN** (`WarrantyPredictorBasic.py`) had a proper train / validation / test split and two baselines.
- The **cost slicing** in NOTES.md (leaks only at F2 or on B coils, etc.) found the right root causes.

## Problems found in the existing work

1. **The KMeans clustering is leaked.** Its inputs include Defect Category, Repair Type and Failure Mode, which equal the field failure for 3,000 / 3,000 units. Its two clusters are "Wiring vs. not" (ARI 0.997). Judges reading "unsupervised model found two clusters: leak/controls and wiring" will see a re-labelling. **Replaced by** the build-cell segmentation in script 05.
2. **The aggregate kNN's R² 0.998 isn't evidence of skill.** Subsets are drawn inside one cluster, so "predict the cluster's average" also scores 0.998, and a one-line formula beats it. **Recommendation:** drop it from the main deck; keep it as a backup slide with the baselines if a judge asks.
3. **`WarrantyPredictorBasic.py` / `SupervisedModel.py` use factory columns as "pre-failure" features.** That's legal timing-wise (they exist before shipment), but they name the failure, so the model is reading the answer. Script 04 separates build-time from shipment-time models.
4. **The root README is out of date.** It mentions `WarrantyCostModel\main.py` (doesn't exist) and `TrainingData_Aggregated_Wide.csv` (the script writes `TrainingData_Updated.csv`), and it describes `SupervisedModel.py` as the unit-level model, when that's actually `WarrantyPredictorBasic.py`. Left unedited per the "new folder only" rule; fix if the repo is shared.
5. `EDA_Analysis.R` reads a `TrainingData.csv` that isn't in the repo.

## What the problem formulation looks like now (put this on a slide)

| # | Question | Target | Unit | Method | Validation |
|---|---|---|---|---|---|
| Q1 | Which build traits are associated with which failures, and how strongly? | failure type share; cost ratio | unit | Fisher exact, log-linear OLS, BH correction | CIs, multiple-testing control |
| Q2 | From the build record, which failure will a unit have? | Failure Category (3-class) | unit | multinomial classifiers vs baselines | 20% hold-out + 5-fold CV |
| Q3 | From the build record, what will its warranty cost? | Warranty Cost ($) | unit | two-stage, GBM, linear vs mean baselines | same hold-out; slice calibration |
| Q4 | Which build configurations fail alike? | none (unsupervised) | build cell (72) | Ward clustering on failure mix | silhouette, bootstrap ARI, build-only control |
| Q5 | Which investments, within $750K? | 3-yr NPV | portfolio (95) | Monte Carlo + claim lag | P(best), break-even, tornado |

## What could still be added (in priority order, if time allows)

1. **Ask for production volumes** (units built per facility / supplier / revision / shift). That's the one missing input that would turn failure *mix* into failure *rate* and allow a true failure-probability model. State it as the #1 data request on the "Next steps" slide rather than trying to model it.
2. ~~Probability calibration plot~~: done (`A03_calibration.png`; Brier 32–37% better than base rates).
3. **A Facility 2 deep-dive**: no catalog program targets a facility, and A/C-coil leaks at F2 cost $2.13M/yr. A braze process audit is a near-zero-cost action worth a line on the recommendation slide.
4. **A fan-correction work-instruction review**: all 185 backward fans followed that rework (about $0.14M/yr). Cheap to verify, cheap to fix.
5. **A run-time check of PortfolioSim's excluded programs** (Predictive Quality Dashboard, EOLT station). The 01 finding argues *against* them: the factory already catches 100% of defects, so more detection capacity (EOLT) adds little. Repair quality is the gap. That's a data-backed reason they're excluded, worth one sentence in the deck.
