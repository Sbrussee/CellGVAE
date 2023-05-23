import squidpy as sq
import scanpy as sc
import pandas as pd
import numpy
import glob
import os

pattern = '/ligrec*.pkl'
files = glob.glob(os.getcwd()+pattern)

for file in files:
  interactions = pd.read_pickle(file)
  name = file
  sq.pl.ligrec(interactions, pvalue_threshold=0.001, remove_empty_interactions=True,
               remove_nonsig_interactions=True, alpha=0.0001, means_range=(0.3, np.inf),
                save=f"lr_true_{name}.png")
