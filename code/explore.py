import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from constants import * 

FILE = os.path.join(DATA, CSV)
df = pd.read_csv(FILE)

print(f"Data length: {len(df)}")
print(f"No of columns: {len(df.columns)} ")
print(f"Columns: {list(df.columns)}")

temp = df['T (degC)'].to_numpy()

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
axes[0].plot(range(len(temp)), temp)
axes[0].set_title("Global range")
axes[0].set_ylabel("Temperature")
axes[1].plot(range(1440), temp[:1440]) # first 10 days (24 * 60 * 10 / 10)
axes[1].set_title("Subset of 1440 samples")
axes[1].set_ylabel("Temperature")
plt.show()