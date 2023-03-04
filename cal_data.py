import os
from random import random
import tarfile
import urllib.request
import certifi
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"

#joining paths using os.path.join because it buts the delimite as necessary, OS independent and so on..
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
#Fetching the data from Git Hub 
def fetching_housingdata(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    #see Documentation for explanation
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetching_housingdata(housing_url=HOUSING_URL, housing_path= HOUSING_PATH)

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def test_train_split(data, test_ratio):
    np.random.seed(44)
    random_indices = np.random.permutation(len(data))
    split_index = (int) (len(random_indices) * test_ratio)
    test_indices = random_indices[:split_index]
    train_indices = random_indices[split_index:]
    return data.iloc[test_indices], data.iloc[train_indices]
    


housing = load_housing_data(HOUSING_PATH)
print(housing.head())

#Get the quick description of the data
#print(housing.info())

#Finding out how many categories exists
#print(housing['ocean_proximity'].value_counts())

#Summary of the numerical attributes
#print(housing.describe())

#plotting a histogram for individual attributes
#housing.hist(bins=50, figsize=(20,15))
#plt.show()
test_set, train_set = test_train_split(housing,0.2)
print(test_set, "\n")
print(train_set)

#labeling the income categories as 0-1.5 -> 1, 1.5-3 -> 2, and so on
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.,1.5,3,4.5,6.,np.inf], labels=[1,2,3,4,5])
#housing["income_cat"].hist()
#plt.show()

#using Stratified sampling from sci-kit learn to slipt training and test set
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    train = housing.loc[train_index]
    test = housing.loc[test_index]

print(test["income_cat"].value_counts() / len(test["income_cat"]))
print(train["income_cat"].value_counts() / len(train["income_cat"]))
