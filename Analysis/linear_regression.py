import csv
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_excel(r'.\Analysis\20200728_behaviour2020_iv_564_1_drift_analysis.ods',engine='odf')

df2 = df.drop(labels=['Start time','Estimated time','Cam time'],axis=1)
df3 = df2.dropna()
df3

X = df3['Elapsed time'].values.reshape(-1,1)
Y = df3['Diff'].values.reshape(-1,1)

np.polyfit(df3['Elapsed time'].values,df3['Diff'].values,2)

linear_regressor = LinearRegression()
linear_regressor.fit(X,Y)
linear_regressor.coef_
linear_regressor.intercept_

Y_pred = linear_regressor.predict(np.arange(60*45).reshape(-1,1))