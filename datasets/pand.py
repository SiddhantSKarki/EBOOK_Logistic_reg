import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("Admission_Predict.csv", index_col=0)
df.columns = [x.lower().strip() for x in df.columns]
mask = (df["chance of admit"].gt(0.9).lt(0.6))
#print(pd.DataFrame(df.loc[mask]).dropna()["university rating"])
print(df.sort_values(by=["university rating"], ascending= False))
df["research"] = df["research"].replace({1:"Y", 0:"N"})
print(df)
# df["chance of admit"].hist()
# plt.show()

# new_df.index = [x+1 for x in range(len(new_df))]
# print(new_df)
# plt.hist(new_df["gre score"], bins=25)

#lets say we want to index the data frame by chance of admission but keep the serial number(current index)
df['serial number'] = df.index

# now we set the index to the chance of amit column
df = df.set_index('chance of admit')
print(df.head())


