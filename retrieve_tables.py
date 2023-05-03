import glob
import pandas as pd

#Read all ligrec pickle fles
files = glob.glob('*ligrec*.pkl')
for file in files:
    df = pd.read_pickle(file)
    print(df)
