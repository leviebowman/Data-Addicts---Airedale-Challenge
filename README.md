# Data-Addicts---Airedale-Challenge
Repository for the team Data Addicts to test and build models for the Airedale competition hosted at Virginia Tech.

## Python setup

The project uses the `.venv` virtual environment. Install the project dependencies from the repository root with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, select `.venv\Scripts\python.exe` as the Python interpreter. You can then import NumPy and Pandas in your scripts:

```python
import numpy as np
import pandas as pd
```

To join all five competition workbooks by `Serial Number` and save the result as `Data\TotalJoin.xlsx`, run:

```powershell
.\.venv\Scripts\python.exe .\Data\TotalJoin.py
```

Overlapping columns are removed only when every joined row matches; non-identical columns are retained with their source dataset in the column name.

To train the kNN regression model on `WarrantyCostModel\TrainingData_Aggregated_Wide.csv`, run:

```powershell
.\.venv\Scripts\python.exe .\WarrantyCostModel\main.py
```

## Portfolio simulation

To choose which catalog investments to fund within the $750K budget, run:

```powershell
.\.venv\Scripts\python.exe .\PortfolioSim\PortfolioSim.py
```

It reads `WarrantyCostModel\Data_Full.xlsx` and treats one year of warranty cost ($7.65M) as a recurring annual baseline. Each program is applied only where the data shows its problem occurs (for example, Wiring Poka-Yoke on Night shift + Controls Rev A builds, Leak Detection at Facility 2 + Supplier B coils), at catalog cost. Programs that hit the same units stack as 1 - (1-a)(1-b), savings start after each program's implementation time, and every portfolio within budget (95) is scored by 3-year net savings (savings discounted at 10%/yr, minus program cost). A 5,000-run Monte Carlo varies warranty costs, how much benefit each program delivers, and rollout delays by risk level to test how robust the pick is. Predictive Quality Dashboard and Additional EOLT Station are not modelled because their benefit can't be priced from warranty data.

Current recommendation: Braze Training + Wiring Poka-Yoke + Leak Detection + Supplier QA ($700K), saving $3.38M/yr with $5.71M average 3-year net savings, best in 97% of runs. Full method, scopes, rankings and caveats are in `PortfolioSim\NOTES.md`.

Outputs are written to `PortfolioSim\outputs\`:

| File | Shows |
| --- | --- |
| `fig1_cost_map.png` | Total warranty cost by failure type, and which programs reach each one |
| `fig2_program_summary.png` | All 10 catalog programs: applied scope, cost, and ranking by appearances in the top 20 portfolios |
| `fig3_portfolio_frontier.png` | Every portfolio tested, net savings vs. upfront spend, with the budget line |
| `fig4_top20_portfolios.png` | Which programs make up the 20 best portfolios |
| `fig5_program_reach.png` | Each program's yearly savings vs. the warranty cost in the area it targets, with cost and payback |
| `fig6_robustness.png` | Net-savings ranges for the top 8 portfolios across the 5,000 runs |
| `portfolio_results.csv`, `program_ranking.csv`, `program_scopes.csv` | The tables behind the charts |

Regenerate the training data with:

```powershell
.\.venv\Scripts\python.exe .\WarrantyCostModel\GetTrainingData.py
```

The model predicts `Warranty Cost Mean`. In `WarrantyCostModel\main.py`, set `FEATURE_COLUMNS` to a list of CSV column names to choose predictors; leave it as `None` to consider every column except the target. It selects features and tunes neighbor count, distance weighting, and Minkowski distance using the validation split, with a 50% training, 25% validation, and 25% test split.

`WarrantyCostModel\GetTrainingData.py` excludes field-service columns shared with the combined data, except `Serial Number`, `Date`, and `Product Family`. It also omits the standard deviation of `Warranty Cost` from aggregate predictors to avoid target leakage.


To train and evaluate the supervised warranty-cost model on pre-failure product, manufacturing, and EOLT features, run:

```powershell
.\.venv\Scripts\python.exe .\WarrantyCostModel\SupervisedModel.py
```

It compares overall-mean and Product Family-mean baselines with kNN regression, selecting neighbor count, distance weighting, and distance metric on validation data. It reports dollar errors and held-out total-cost calibration, and writes unit-level test predictions to `WarrantyCostModel\SupervisedTestPredictions.csv`. Its predictions are conditional on an observed field failure; it does not estimate fleet failure probability.
