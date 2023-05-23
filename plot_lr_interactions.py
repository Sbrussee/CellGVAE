import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy as np
import glob
import os

pattern = '/lr*.csv'
files = glob.glob(os.getcwd()+pattern)

for file in files:
  df = pd.read_csv(pattern)
  df['Count'] = (df < 0.001).sum(axis=1)
  df = df[['Count']]
  print(df)
