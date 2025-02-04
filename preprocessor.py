import os
import pickle
import joblib
import warnings
import numpy as np
import pandas as pd
# import xgboost as xgb
import sklearn
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
	OneHotEncoder,
	OrdinalEncoder,
	StandardScaler,
	MinMaxScaler,
	PowerTransformer,
	FunctionTransformer
)
from feature_engine.outliers import Winsorizer
from feature_engine.datetime import DatetimeFeatures
from feature_engine.selection import SelectBySingleFeaturePerformance
from feature_engine.encoding import (
	RareLabelEncoder,
	MeanEncoder,
	CountFrequencyEncoder
)

sklearn.set_config(transform_output="pandas") # make sure all transformer outut pandas dataframe instead of numpy arrays

############################# Data Preprocessor ############################################################################################
air_transformer = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="most_frequent")),
	("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)), #group cat with less than 10% within min 2 cat
	("encoder", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
])


feature_to_extract = ["month", "week", "day_of_week", "day_of_year"]

doj_transformer = Pipeline(steps=[
	("dt", DatetimeFeatures(features_to_extract=feature_to_extract, yearfirst=True)),
	("scaler", MinMaxScaler())
])


location_pipe1 = Pipeline(steps=[
	("grouper", RareLabelEncoder(tol=0.1, replace_with="Other", n_categories=2)), # group less than 10% values to other cat   
	("encoder", MeanEncoder()), #calculates avg value for each cat target and encode
	("scaler", PowerTransformer()) #make distribution symmetric
])


time_pipe1 = Pipeline(steps=[
	("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
	("scaler", MinMaxScaler())
])

time_pipe1 = Pipeline(steps=[
	("dt", DatetimeFeatures(features_to_extract=["hour", "minute"])),
	("scaler", MinMaxScaler())
])

def part_of_day(X, morning=4, noon=12, eve=16, night=20):
	columns = X.columns.to_list()
	X_temp = X.assign(**{
		col: pd.to_datetime(X.loc[:, col]).dt.hour
		for col in columns
	})

	return (
		X_temp
		.assign(**{
			f"{col}_part_of_day": np.select(
				[X_temp.loc[:, col].between(morning, noon, inclusive="left"),
				 X_temp.loc[:, col].between(noon, eve, inclusive="left"),
				 X_temp.loc[:, col].between(eve, night, inclusive="left")],
				["morning", "afternoon", "evening"],
				default="night"
			)
			for col in columns
		})
		.drop(columns=columns)
	)

time_pipe2 = Pipeline(steps=[
	("part", FunctionTransformer(func=part_of_day)),
	("encoder", CountFrequencyEncoder()),
	("scaler", MinMaxScaler())
])

time_transformer = FeatureUnion(transformer_list=[
	("part1", time_pipe1),
	("part2", time_pipe2)
])

# It selects numerical columns (if no specific columns are provided).
# It calculates percentile values (e.g., 25th, 50th, and 75th percentiles) for each selected column.
# It computes the similarity between each value in the dataset and these percentiles using the RBF kernel function.
# It outputs new transformed features that represent the similarity of each original feature to these percentile values.

class RBFPercentileSimilarity(BaseEstimator, TransformerMixin):
	def __init__(self, variables=None, percentiles=[0.25, 0.5, 0.75], gamma=0.1):
		self.variables = variables
		self.percentiles = percentiles
		self.gamma = gamma


	def fit(self, X, y=None):
		if not self.variables:
			self.variables = X.select_dtypes(include="number").columns.to_list()

		self.reference_values_ = {
			col: (
				X
				.loc[:, col]
				.quantile(self.percentiles)
				.values
				.reshape(-1, 1)
			)
			for col in self.variables
		}

		return self


	def transform(self, X):
		objects = []
		for col in self.variables:
			columns = [f"{col}_rbf_{int(percentile * 100)}" for percentile in self.percentiles]
			obj = pd.DataFrame(
				data=rbf_kernel(X.loc[:, [col]], Y=self.reference_values_[col], gamma=self.gamma),
				columns=columns
			)
			objects.append(obj)
		return pd.concat(objects, axis=1)


def duration_category(X, short=180, med=400):
	return (
		X
		.assign(duration_cat=np.select([X["Duration"].lt(short),
									    X["Duration"].between(short, med, inclusive="left")],
									   ["short", "medium"],
									   default="long"))
		.drop(columns="Duration")
	)


def is_over(X, value=1000):
	return (
		X
		.assign(**{
			f"duration_over_{value}": X["Duration"].ge(value).astype(int)
		})
		.drop(columns="Duration")
	)


duration_pipe1 = Pipeline(steps=[
	("rbf", RBFPercentileSimilarity()),
	("scaler", PowerTransformer())
])

duration_pipe2 = Pipeline(steps=[
	("cat", FunctionTransformer(func=duration_category)),
	("encoder", OrdinalEncoder(categories=[["short", "medium", "long"]]))
])

duration_union = FeatureUnion(transformer_list=[
	("part1", duration_pipe1),
	("part2", duration_pipe2),
	("part3", FunctionTransformer(func=is_over)),
	("part4", StandardScaler())
])

duration_transformer = Pipeline(steps=[
	("outliers", Winsorizer(capping_method="iqr", fold=1.5)),
	("imputer", SimpleImputer(strategy="median")),
	("union", duration_union)
])


def is_direct(X):
	return X.assign(is_direct_flight=X["Total_Stops"].eq(0).astype(int))


total_stops_transformer = Pipeline(steps=[
	("imputer", SimpleImputer(strategy="most_frequent")),
	("", FunctionTransformer(func=is_direct))
])


info_pipe1 = Pipeline(steps=[
    ("group", RareLabelEncoder(tol=0.1, n_categories=2, replace_with="Other")),
    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

def have_info(X):
    return X.assign(Additional_Info=X["Additional_Info"].ne("No Info").astype(int))

info_union = FeatureUnion(transformer_list=[
("part1", info_pipe1),
("part2", FunctionTransformer(func=have_info))
])

info_transformer = Pipeline(steps=[
("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
("union", info_union)
])


column_transformer = ColumnTransformer(transformers=[
	("air", air_transformer, ["Airline"]),
	("doj", doj_transformer, ["Date_of_Journey"]),
	("location", location_pipe1, ["Source", 'Destination']),
	("time", time_transformer, ["Dep_Time", "Arrival_Time"]),
	("dur", duration_transformer, ["Duration"]),
	("stops", total_stops_transformer, ["Total_Stops"]),
	("info", info_transformer, ["Additional_Info"])
], remainder="passthrough")


# applies randomforest to find the best features. The unrelevant features will get dropped.
# fits RF between each feature and the target variable. Features R2 values below 0.1 will get dropped while the others are retained

estimator = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)

selector = SelectBySingleFeaturePerformance(
	estimator=estimator,
	scoring="r2",
	threshold=0.1
) 


preprocessor = Pipeline(steps=[
	("ct", column_transformer),
	("selector", selector)
])


train = pd.read_csv("data/train.csv")
X_train = train.drop(columns="Price")
y_train = train["Price"]

# fit and save the preprocessor
preprocessor.fit(X_train, y_train)
joblib.dump(preprocessor, "preprocessor.joblib")