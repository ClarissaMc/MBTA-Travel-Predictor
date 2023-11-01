import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
from time import strftime, localtime, time
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

pd.set_option('display.max_columns', None)

# Import data
df = pd.read_csv('data/2023-Q2_HRTravelTimes-MODIFIED.csv')
# print("Successfully imported data")

# # Clean data
# df.drop('end_time_sec', axis=1, inplace=True)	# drop unnecessary columns
# print("Successfully dropped columns")

# # Convert date to weekday
# days = []
# for idx, row in df.iterrows():
# 	date_dt = datetime.strptime(row['service_date'], '%Y-%m-%d')
# 	days.append(date_dt.weekday())
# print("Successfully converted dates to days")
# df.insert(0, 'weekday', days)
# print("Successfully inserted column")
# df.to_csv('data/2023-Q2_HRTravelTimes-MODIFIED.csv', index=False)

# df.drop('service_date', axis=1, inplace=True) # drop service_date
# df.to_csv('data/2023-Q2_HRTravelTimes-MODIFIED.csv', index=False)

# Drop rows where travel time is 0 or 1
df = df[df['travel_time_sec'] != 0]
df = df[df['travel_time_sec'] != 1]
df = df[df['travel_time_sec'] != 2]
df = df[df['travel_time_sec'] != 3]
df = df[df['travel_time_sec'] != 4]


# Convert categorical values to numerical
encoder = OrdinalEncoder()
encoder.fit(df[['route_id']])
#print("Successfully fit data")
mapping = list(encoder.categories_)
#print(mapping)
df['route_id'] = encoder.transform(df[['route_id']])
# print("Successfully transformed data")
# print(df.head())
# print(df.tail())


# Assign x and y values
x = df[['weekday', 'from_stop_id', 'to_stop_id', 'route_id', 'direction_id', 'start_time_sec']]
y = df['travel_time_sec']

# Split data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# Exploratory data analysis
print(df.sort_values(by='travel_time_sec', ascending=False))	# sort data from longest to shortest travel time
print(df['travel_time_sec'].agg('mean'))	# average travel time (~16.5 mins)

# Plotting average travel time for each line
# x_vals = ['Blue', 'Orange', 'Red']
# blue_routes = df[df['route_id'] == 0.0]
# average_blue = blue_routes['travel_time_sec'].agg('mean')
# orange_routes = df[df['route_id'] == 1.0]
# average_orange = orange_routes['travel_time_sec'].agg('mean')
# red_routes = df[df['route_id'] == 2.0]
# average_red = red_routes['travel_time_sec'].agg('mean')
# y_vals = [average_blue, average_orange, average_red]
# plt.bar(x_vals, y_vals, color=['tab:blue', 'tab:orange', 'tab:red'])
# plt.ylabel('Average Travel Time (sec)')
# plt.xlabel('MBTA Line')
# plt.title('Average Travel Time by Line')
# plt.show()


# Model Selection and Implementation

model = LinearRegression() # build linear regression model
model.fit(x_train_scaled, y_train)	# train model

y_pred = model.predict(x_test_scaled)	# predict against testing set


# Model Evaluation
mean = np.mean(y_pred, axis=0)
std = np.std(y_pred, axis=0)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


# x_train_scaled_2 = np.array(x_train).reshape(-1, 1)
# x_test_scaled_2 = np.array(x_test).reshape(-1, 1)
# y_train_scaled_2 = np.array(y_train).reshape(-1, 1)
# y_test_scaled_2 = np.array(y_test).reshape(-1, 1)

# reg = LinearRegression().fit(x_train_scaled_2, y_train_scaled_2)
# r22 = reg.score(x_test_scaled_2, y_test_scaled_2)


# Print evaluation
print("First Normalization Method")
print(f"Mean: {mean}")
print(f"Standard Deviation: {std}")
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"R2: {r2}")
# print("Second Normalization Method")
# print(f"R2: {r22}")

# df_smaller = df.drop(df.tail(5000000).index, inplace=False)
# df_smaller.to_csv('data/2023-Q2_HRTravelTimes-SMALLER.csv', index=False)
# df_smaller = pd.read_csv('data/2023-Q2_HRTravelTimes-SMALLER.csv')
# print(df_smaller.shape)