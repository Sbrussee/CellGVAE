import glob
import pandas as pd
import squidpy as sq

#Read all ligrec pickle fles
files = glob.glob('*ligrec*.pkl')
for file in files:
    print(f"FILE: {file}")
    df = pd.read_pickle(file)
    if df != None:
        mask = df['pvalues'] < 0.05
        selected = df['pvalues'][(df['pvalues'] < 0.01).any(axis=1)]
        selected['count'] =  selected.lt(0.01).sum(axis=1)
        sorted = selected.sort_values(by='count', ascending=False)
        print(sorted[:10])
        sq.pl.ligrec(df, pvalue_threshold=0.001, remove_empty_interactions=True,
                     remove_nonsig_interactions=True, alpha=0.0001, mean_range(0.3, np.inf),
                      save=f"{file}.png")
