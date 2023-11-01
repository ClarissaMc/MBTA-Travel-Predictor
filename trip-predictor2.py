import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from time import strftime, localtime, time
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from mlforecast import MLForecast
from window_ops import rolling_mean

# Import data
df = pd.read_csv('data/2023-Q2_HRTravelTimes.csv')

# Add ID column
df['id'] = np.arange(df.shape[0])

# Remove end_time_sec column
df.drop('end_time_sec', axis=1, inplace=True)

# Rename columns to mlforecast standard format
df = df.rename(columns={'service_date':'ds',
						'travel_time_sec':'y'
						})

# Create unique_id column
df['unique_id'] = df['from_stop_id'].astype(str) + '_' + df['to_stop_id'].astype(str)

# Split into training and testing data
training = df.loc[df['ds'] < '2023-06-13']
testing = df.loc[df['ds'] >= '2023-06-13']

# Models to test
models = [RandomForestRegressor(random_state=0, n_estimators=100),
		  ExtraTreesRegressor(random_state=0, n_estimators=100)]

# Make model that creates and trains the above models
model = MLForecast(models=models,
				   freq='D',
				   lags=[1, 7, 14],
				   lag_transforms={
				   		1:[(rolling_mean, 7), (rolling_mean, 14), (rolling_mean, 28)],
				   		7:[(rolling_mean, 7), (rolling_mean, 14), (rolling_mean, 28)],
				   		14:[(rolling_mean, 7), (rolling_mean, 14), (rolling_mean, 28)]
				   },
				   date_features=['dayofweek'],
				   num_threads=4)

# Fit data
model.fit(training, id_col='unique_id', time_col='ds', target_col='y', static_features=['from_stop_id', 'to_stop_id', 'route_id', 'direction'])

# Test data
tested_data = model.predict(horizon=60, dynamic_dfs=[testing[['unique_id', 'ds', 'start_time_sec']]])
tested_data = tested_data.merge(testing[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')

