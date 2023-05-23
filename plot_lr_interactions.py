import pandas as pd
import numpy as np
import glob
import os
print(os.path.abspath(os.getcwd()))
pattern = 'lr_*exp6*.csv'
files = glob.glob(pattern)
print(files)
for file in files:
  print(file)
  df = pd.read_csv(file)
  print(df)
  df = df[['count']]
