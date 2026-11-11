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
