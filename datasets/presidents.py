import pandas as pd

df = pd.read_csv("presidents.csv", index_col=0)

print(df.head())