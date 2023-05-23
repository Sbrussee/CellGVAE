import pandas as pd
import numpy as np
import glob
import os
print(os.getcwd())
pattern = 'lr_*.csv'
print(os.getcwd()+pattern)
files = glob.glob(os.getcwd()+pattern)
print(files)
for file in files:
  df = pd.read_csv(pattern)
  df['Count'] = (df < 0.001).sum(axis=1)
  df = df[['Count']]
  print(df)
