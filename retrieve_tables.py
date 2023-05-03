import glob
import pandas as pd

#Read all ligrec pickle fles
files = glob.glob('*ligrec*.pkl')
for file in files:
    print(f"FILE: {file}")
    df = pd.read_pickle(file)
    if df != None:
        mask = df['pvalues'] < 0.05
        counts = mask.sum(axis=1)
        selected = df['pvalues'][(df['pvalues'] < 0.05).any(axis=1)]
        sorted = selected.sort_values(by=counts, ascending=False)
        print(sorted)
