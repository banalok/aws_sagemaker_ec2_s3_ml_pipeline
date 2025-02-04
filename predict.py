import pickle
import joblib
import pandas as pd
import xgboost as xgb

x_test = pd.read_csv("data/test.csv")
saved_preprocessor = joblib.load("preprocessor.joblib")
x_test_pre = saved_preprocessor.transform(x_test)

with open("xgboost-model", "rb") as f:
    model = pickle.load(f)
    x_test_xgb = xgb.DMatrix(x_test_pre)
    pred = model.predict(x_test_xgb)
    print(f"The predicted price is {pred:,.0f} INR")