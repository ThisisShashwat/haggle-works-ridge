"""A scaled ridge regression on every numeric column, trained on log hours."""

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
columns = [c for c in test.columns if c != "project_id" and pd.api.types.is_numeric_dtype(test[c])]
model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
model.fit(train[columns], np.log1p(train["hours_spent"]))
guess = np.expm1(model.predict(test[columns])).clip(0)
pd.DataFrame({"project_id": test["project_id"], "hours_spent": guess}).to_csv("predictions.csv", index=False)
