# Methods: Enough Detail to Rebuild in Any Tool

Written for judging criterion **D8** ("another analyst could rebuild it in any language or tool"). Every number in `Analysis/README.md` comes from these steps. Random seed 42 throughout.

## 1. Data preparation (D2)

1. Read the five workbooks. Each has 3,000 rows and a unique `Serial Number`, and the five sets of serial numbers are identical.
2. Inner-join on `Serial Number`. Columns that appear in several files (Facility, Shift, Product Family, Failure Category) must be identical row by row; they are, so keep one copy. The result is 3,000 rows × 27 columns, identical to `WarrantyCostModel/Data_Full.xlsx`.
3. Drop three constant columns: `Warranty Eligible`, `Pass After Repair`, `Retested`.
4. **Reconcile labels:** map Rework `Defect Category` (Brazing→Leak), EOLT `Failure Mode` (Loose Terminal / Fan Rotation Error→Wiring, Refrigerant Leak→Leak, Sensor Miswire→Controls) and `Repair Type` onto the field `Failure Category`.
5. **Tag every column by when it is known.** Build: Family, Facility, Controls Rev, Coil Supplier, Shift. Factory: rework and EOLT columns. Field: everything else. **Predictive models use Build columns only.** The leakage screen (Cramér's V with the field category) is what justifies excluding the factory columns: V = 1.00 for three of them.
6. No missing values; no imputation needed.

## 2. The five questions and why each method was chosen (D1, D3)

| Q | Target · unit | Method | Why this method | Simpler alternative it must beat |
|---|---|---|---|---|
| Q1 Root causes | Share of failures of a type; cost ratio · unit | Fisher exact test + Wilson 95% CI; OLS on log(cost) with failure-description fixed effects; Kruskal-Wallis; Benjamini-Hochberg FDR across 14 tests | Exact tests suit cells with zero counts (several are 0 of 1,500+). Log-cost OLS gives a multiplicative effect *holding failure type fixed*. BH controls false discoveries across many tests. | Raw crosstab counts, which are unfair across groups of different size; we use shares |
| Q2 Failure type | `Failure Category` (3 classes) · unit | Multinomial logistic, decision tree, random forest, gradient boosting on one-hot build features; selected by 5-fold CV log loss | Log loss scores the *probabilities* we use for decisions. Trees capture "Facility 2 OR Supplier B" interactions that logistic regression has to approximate. | Majority class, class prior, 1-split stump |
| Q3 Warranty cost | `Warranty Cost` $ · unit | Two-stage: Σ_c P(c \| build) × mean cost(c, LegacyX?), with P from a depth-4 tree. Also linear regression and gradient boosting. | Cost is driven by failure type (leaks are about $7K, wiring about $500) and a LegacyX multiplier. The two-stage form mirrors that and stays explainable. | Overall mean, Product-Family mean |
| Q4 Segments | none · build cell (Facility × Supplier × Rev × Shift, 72 cells) | Ward agglomerative clustering on the cell's failure-mix shares (loose wiring, backward fan, leak, controls) | Shares make cells of 20–73 units comparable. Ward minimizes within-segment variance and needs no initialization. | KMeans on build features alone (tests whether structure exists without outcomes) |
| Q5 Investments | 3-year NPV · portfolio | Teammates' PortfolioSim: exhaustive search of 95 within-budget portfolios; benefits stack as 1 − ∏(1 − bⱼ) per unit; 5,000-draw Monte Carlo (bootstrap units, triangular benefit realization by risk, uniform slip). Plus claim lag. | Exhaustive search is cheap at 2⁸ subsets. Monte Carlo propagates all three uncertainties together. | Ranking programs one at a time (ignores overlap) |

## 3. Validation (D4)

- **Q2, Q3:** stratified (by failure category) 80/20 split. The 600 test units are scored once, at the end. Tuning uses 5-fold stratified CV on the 2,400 training units only.
  - *Why random splits fit this data:* one row per unit, no repeated measures, no build dates.
  - *Robustness:* training on the earliest 80% of claims and testing on the latest 20% gives log loss 0.492 vs 0.485 random, and leak AUC 0.85 both ways.
- **Ceiling:** the true per-cell failure rates (estimated in-sample) give the best log loss any build-only model could reach (0.443). It is reported for context and was never used for selection.
- **Q4:** silhouette for k = 2..8; 200 bootstrap resamples of units, recluster, Adjusted Rand Index vs. the original (median 0.91).
- **Aggregate kNN re-check:** units are split 50/50 *before* drawing subsets, so no unit is on both sides. It is compared against a cluster-average lookup and a no-ML formula.

## 4. Metrics and baselines (D5)

| Model | Metric | Baseline | Model | Comment |
|---|---|---|---|---|
| Failure-type classifier | Log loss | 0.857 (prior) | **0.485** | 90% of the achievable gap to the 0.443 ceiling |
| | Leak AUC | 0.50 | **0.84** | |
| | Brier (Leak) | 0.194 (base rate) | **0.122** | 37% better; calibration plot provided |
| Cost model | MAE / R² | $2,588 / 0.00 | **$1,856 / 0.29** | |
| | Slice-total error | −41% (family mean) | **+3%** (Leak Detection scope) | |
| Segmentation | Silhouette | 0.12–0.18 (build-only) | **0.70** | |

## 5. Predictive vs. causal claims (D6)

| We claim | Type | Supported by | We do **not** claim |
|---|---|---|---|
| Factory defect labels match field failures 3,000/3,000 | Descriptive fact | Crosstab | |
| Supplier B / Facility 2 / Rev A / fan-correction are *associated* with specific failures | Statistical association | Fisher tests, BH q < 1e-75 | That they *cause* them; each has a verification step (README §02) |
| Build data *predicts* failure type and cost | Predictive | Hold-out metrics vs baselines | That changing a build trait changes the outcome by the predicted amount |
| Programs cut the failures they target by the catalog % | **Assumption** (Airedale's estimate) | n/a; stress-tested | We test sensitivity: break-even at 10% of catalog benefit |
| Failure *mix*, not failure *rate* | Limitation | Only failed units are in the data | Per-unit failure probabilities; these need production volumes |

## 6. Decision impact (D7)

- **Unit flagging:** send "Facility 2 OR Supplier B" builds (48% of builds) to enhanced leak testing. They contain 100% of leak cost on the hold-out set.
- **Recommended portfolio** (Braze + Poka-Yoke + Leak Detection + Supplier QA, $700K): about **595 failures avoided per year (19.8% of failures; 52% of leaks; 40% of controls faults; 21% of backward fans)** and **$3.38M/yr** in warranty cost, once live. Over 3 years with claim lag, net of cost: **$4.85M mean, $4.01M 5th percentile**. It is the best portfolio in 98% of runs.
- Failures avoided assume each catalog % applies to failures as well as cost. That's exact for the "reduction in defects" programs and an assumption for the "reduction in warranty cost" ones.

## 7. Software

Python 3.11; pandas 3.1, scikit-learn 1.9, scipy 1.17, statsmodels, matplotlib. One command: `.venv\Scripts\python.exe Analysis\run_all.py`.
