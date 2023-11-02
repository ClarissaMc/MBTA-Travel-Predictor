import pandas as pd
import numpy as np
import requests
import json
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
from time import strftime, localtime, time
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from window_ops.rolling import rolling_mean
from mlforecast import MLForecast

pd.set_option('display.max_columns', None)

# Import data
print("Importing data")
df = pd.read_csv('data/2023-Q2_HRTravelTimes.csv')

# Add ID column
print("Adding ID column")
df['id'] = np.arange(df.shape[0])

# Remove end_time_sec column
print("Dropping end_time_sec column")
df.drop('end_time_sec', axis=1, inplace=True)

# Rename columns to mlforecast standard format
print("Renaming columns")
df = df.rename(columns={'service_date':'ds',
						'travel_time_sec':'y'
						})

# Create unique_id column
print("Adding unique_id column")
df['unique_id'] = df['from_stop_id'].astype(str) + '_' + df['to_stop_id'].astype(str)

# Convert dates from strings to dates
print("Converting dates from strings to datetimes")
df['ds'] = pd.to_datetime(df['ds'], format='%Y-%m-%d')

# Convert route_id to numerical data
encoder = OrdinalEncoder()
print("Fitting route_id")
encoder.fit(df[['route_id']])
print("Transforming route_id")
df['route_id'] = encoder.transform(df[['route_id']])
print("Successfully transformed route_id")


df_H1 = df[(df.index < np.percentile(df.index, 25))]
print(df_H1.tail())
print(df_H1.shape)


# Split into training and testing data
training = df.loc[df['ds'] < '2023-04-20']	# 80%
testing = df.loc[df['ds'] >= '2023-04-25']	# 20%

# Models to test
models = [RandomForestRegressor(random_state=0, n_estimators=25),
		  ExtraTreesRegressor(random_state=0, n_estimators=25)]

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
print("Fitting models")
print(f"Started at: {datetime.now()}")
model.fit(training, id_col='unique_id', time_col='ds', target_col='y', static_features=['from_stop_id', 'to_stop_id', 'route_id', 'direction_id'])
print("Successfully fitted models")
print(f"Finished at: {datetime.now()}")

# Save models
joblib.dump(model.models['RandomForestRegressor'], "models/Random_Forest_Regressor.joblib")
joblib.dump(model.models['ExtraTreesRegressor'], "models/Extra_Trees_Regressor.joblib")

# Test data
print("Testing models")
print(f"Started at: {datetime.now()}")
tested_data = model.predict(horizon=60, dynamic_dfs=[testing[['unique_id', 'ds', 'start_time_sec']]])
print("Successfully tested models")
print(f"Finished at: {datetime.now()}")
print("Merging data")
print(f"Started at: {datetime.now()}")
tested_data = tested_data.merge(testing[['unique_id', 'ds', 'y']], on=['unique_id', 'ds'], how='left')
print("Successfully merged data")
print(f"Finished at: {datetime.now()}")

# Printing prediction comparison
print("Prediction Comparison:")
print(tested_data.head())


# Check for data leakage
print("Checking for data leakage")
print(f"Started at: {datetime.now()}")
df_computations = model.preprocess(training, id_col='unique_id', time_col='ds', target_col='y', static_features=['from_stop_id', 'to_stop_id', 'route_id', 'direction_id'])
print("Successfully checked for leakage")
print(f"Finished at: {datetime.now()}")

# Printing leakage check
print("Results of leakage check:")
print(df_computations.head())


# Retrieving feature importances
print("Retrieving feature importances")
print(f"Started at: {datetime.now()}")
pd.Series(model.models_['RandomForestRegressor'].feature_importances_, index=model.ts.features_order_).sort_values(ascending=False).plot.bar(
	figsize=(1280/96, 720/96), title='RandomForestRegressor Feature Importance', xlabel='Features', ylabel='Importance')
print("Successfully graphed feature importances for RandomForestRegressor")
print(f"Finished at: {datetime.now()}")
pd.Series(model.models_['ExtraTreesRegressor'].feature_importances_, index=model.ts.features_order_).sort_values(ascending=False).plot.bar(
	figsize=(1280/96, 720/96), title='ExtraTreesRegressor Feature Importance', xlabel='Features', ylabel='Importance')
print("Successfully graphed feature importances for ExtraTreesRegressor")
print(f"Finished at: {datetime.now()}")


# Measuring accuracy of models
def wmape(y_true, y_pred):
	return np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()

print(f"WMAPE RandomForestRegressor: {wmape(tested_data['y'], tested_data['RandomForestRegressor'])}")
print(f"WMAPE ExtraTreesRegressor: {wmape(tested_data['y'], tested_data['ExtraTreesRegressor'])}")


# Graphing predictions
for model in ['RandomForestRegressor', 'ExtraTreesRegressor']:
	fig, ax = plt.subplots(2, 1, figsize=(1280/96, 720/96))
	for ax_, trip in enumerate(['70105_70096', '70044_70040']):
		tested_data.loc[tested_data['unique_id'] == trip].plot(x='ds', y='y', ax=ax[ax_], label='y', title=trip, linewidth=2)
		tested_data.loc[tested_data['unique_id'] == trip].plot(x='ds', y=model, ax=ax[ax_], label=model)
		ax[ax_].set_xlabel('Date')
		ax[ax_].set_ylabel('Travel Time')
		fig.tight_layout()