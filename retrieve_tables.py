import glob
import pandas as pd
import squidpy as sq
import numpy as np

#Read all ligrec pickle fles
files = glob.glob('*ligrec*.pkl')
for file in files:
    print(f"FILE: {file}")
    df = pd.read_pickle(file)
    if df != None:
        file = file.replace(".pkl", "")
        selected = df['pvalues'][(df['pvalues'] < 0.01).any(axis=1)]
        selected['count'] =  selected.lt(0.001).sum(axis=1)
        sorted = selected.sort_values(by='count', ascending=False)
        print(sorted[:10])
        sq.pl.ligrec(df, pvalue_threshold=0.001, remove_empty_interactions=True,
                     remove_nonsig_interactions=True, alpha=0.0001, means_range=(0.3, np.inf),
                      save=f"{file}.png")
        sorted[:10].to_csv(f"{file}.csv")
