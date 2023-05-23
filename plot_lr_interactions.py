import pandas as pd
import numpy as np
import glob
import os
print(os.path.abspath(os.getcwd()))
pattern = 'lr_*exp6*.csv'
files = glob.glob(pattern)
print(files)
for file in files:
  df = pd.read_csv(file)
  df['Count'] = (df < 0.001).sum(axis=1)
  df = df[['Count']]
  print(df)
