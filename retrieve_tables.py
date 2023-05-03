import glob
import pandas as pd

#Read all ligrec pickle fles
files = glob.glob('*ligrec*.pkl')
for file in files:
    print(f"FILE: {file}")
    df = pd.read_pickle(file)
    if df != None:
        selected = df['pvalues'][(df < 0.05).any(axis=1)]
        print(selected)
