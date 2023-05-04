import glob
import pandas as pd
import squidpy as sq
import numpy as np

#Read all ligrec pickle fles
files = glob.glob('*graph_summary*.pkl')
for file in files:
    print(f"FILE: {file}")
    df = pd.FataFrame.from_dict(pd.read_pickle(file))
    print(df)
    df.to_csv(f"{file}.csv")
