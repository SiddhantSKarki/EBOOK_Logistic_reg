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
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
# Usin SKLearn.model_selection train test to split the groups
test_set, train_set = train_test_split(housing,test_size=0.2,random_state=42)
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

#removing the categorized income comlum from both train and test sets

for sets_ in (train,test):
    sets_.drop("income_cat", axis=1, inplace=True)

#housing = train.copy()
#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#plt.show()

#Separating predictors and the labels 
housing = train.drop("median_house_value", axis=1)
housing_labels = train["median_house_value"].copy()

# Now using imputer to replace missing numerical values with median
imputer = SimpleImputer(strategy="median")

#dropping categorical data i.e ocean_proximity
housing_num = housing.drop("ocean_proximity", axis=1)
#fitting the imputer to the numerical data frame
imputer.fit(housing_num)
X = imputer.transform(housing_num)



#putting everything back to a pandas data frame
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

#encoding the categorical attribute
housing_cat = housing[["ocean_proximity"]]
encoder = OrdinalEncoder()
housing_encoded = encoder.fit_transform(housing_cat)
print(housing_encoded[:10])
#This method is OK but in many ML aclgorithms the algorithm matches wrong values as common (see book for exact reason)
#To avoid that we want to encode ONE category as 1 and everything else as zero, for each category,
#  so ther would be a large matrix with thousands of colums full of 1s and 0s, so save memory it returns a Sparse Matrix object
#So we use OneHotEncoder
one_hot_encoder = OneHotEncoder()
housing_one_encoded = one_hot_encoder.fit_transform(housing_cat)
pipeline_num = Pipeline([('imputer_median', SimpleImputer()),
                         ('scaler_sd', StandardScaler())])
housing_all = pipeline_num.fit_transform(housing_num)

#Transforming the columns by adding the encoded categories 
housing_num_list = list(housing_num)
housing_cat_list = ['ocean_proximity']
cat_pipeline = ColumnTransformer([('numerical', pipeline_num, housing_num_list),
                                  ('encoding_cat', OneHotEncoder(), housing_cat_list)])

housing_prepared = cat_pipeline.fit_transform(housing)
print(housing_prepared)

#Training the Linear Regressing Model

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

#Checking learned predictions
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = cat_pipeline.transform(some_data)

print("Predictions: ", lin_reg.predict(some_data_prepared))
print("Actual: ", list(some_labels))

#Using RMS for Error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)











