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
  df = pd.read_csv(file, skiprows=2)
  df['count'] = df[df.columns[-1]]
  df = df[['source', 'target', 'count']]
  print(df)
  df.to_csv("reduced_"+file, index=False)
