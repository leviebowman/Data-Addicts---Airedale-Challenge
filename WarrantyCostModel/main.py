from pathlib import Path

import pandas as pd
import numpy as np

data_path = Path(__file__).resolve().parent / "Data_Full.xlsx"
data = pd.read_excel(data_path)

# View the first 5 rows
print(data.head())