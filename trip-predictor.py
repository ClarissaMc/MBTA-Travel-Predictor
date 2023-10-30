import pandas as pd
import requests
import json
from time import strftime, localtime, time

# Predictor
class TravelPredictor:

	def __init__(self, date, start, end):
		self.start = start
		self.end = end
		# Convert epoch time to datetime
		self.dateEpoch = date
		dateString = strftime('%Y-%m-%d %H:%M:%S', localtime(date))
		self.dateDatetime = pd.to_datetime(dateString)
		#print(self.date)
		self.key = ""
		self.base_url = 'https://api-v3.mbta.com'
		self.endpoint = '/traveltimes'

	# Get weekly average for given timeframe
	# If given date is weekday, calculate 5-day weekday average
	# If given date is weekend, calculate past 6 2-day weekends
	#def getWeeklyAverage(self):
		# Weekday
		#if self.date.weekday() < 5:
			#print(self.date, "is a Weekday")
		# Weekend
		#else:
			#print(self.date, "is a Weekend")


# MAIN
today = time()
#tp = TravelPredictor(today)
#tp.getWeeklyAverage()

# Getting routes
# url = 'https://api-v3.mbta.com/stops?filter[route]=Silver2&api_key='
# response = requests.get(url)
# #print(response)
# if response.status_code == 200:
# 	data = response.json()
# 	save_file = open('stopsSilver2.json', 'w')
# 	json.dump(data, save_file, indent=6)
# 	save_file.close()
# else:
# 	print(f"Request failed with status code {response.status_code}.")
# 	print(response.json())

# Testing for correct ids
# url2 = 'https://api-v3.mbta.com/predictions?filter[stop]=place-brntn&api_key='
# response = requests.get(url2)
# if response.status_code == 200:
# 	data = response.json()
# 	save_file = open('test.json', 'w')
# 	json.dump(data, save_file, indent=6)
# 	save_file.close()
# else:
# 	print(f"Request failed with status code {response.status_code}.")
# 	print(response.json())