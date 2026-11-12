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

Regenerate the training data with:

```powershell
.\.venv\Scripts\python.exe .\WarrantyCostModel\GetTrainingData.py
```

The model predicts `Warranty Cost Mean`. In `WarrantyCostModel\main.py`, set `FEATURE_COLUMNS` to a list of CSV column names to choose predictors; leave it as `None` to consider every column except the target. It selects features and tunes neighbor count, distance weighting, and Minkowski distance using the validation split, with a 50% training, 25% validation, and 25% test split.


To train and evaluate the supervised warranty-cost model on pre-failure product, manufacturing, and EOLT features, run:

```powershell
.\.venv\Scripts\python.exe .\WarrantyCostModel\SupervisedModel.py
```

It compares overall-mean and Product Family-mean baselines with kNN regression, selecting neighbor count, distance weighting, and distance metric on validation data. It reports dollar errors and held-out total-cost calibration, and writes unit-level test predictions to `WarrantyCostModel\SupervisedTestPredictions.csv`. Its predictions are conditional on an observed field failure; it does not estimate fleet failure probability.
