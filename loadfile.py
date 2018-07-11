# import os
# import glob
from scipy.io import loadmat
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
'''
Retrieve datetime from Matlab serial date
'''
def getYear(matDate):
	temp = int(matDate)
	# return (datetime.fromordinal(temp) + timedelta(days = temp % 1) \
	# 		- timedelta(days = 366)).year
	return datetime.fromordinal(max(temp - 366, 1)).year

'''
Load raw data from .mat file
'''
def loadData(db_name, path):
	data =  loadmat(path)
	
	# use temp dict to calc age later
	births = data[db_name]['dob'][0, 0][0]
	births = np.array([getYear(birth) for birth in list(births)])
	takens = data[db_name]['photo_taken'][0, 0][0]
	print(min(takens), max(takens))
	# to save result data
	result = dict()
	col_name = ['full_path', 'gender', 'face_score', 'second_face_score']
	for col in col_name:
		result[col] = data[db_name][col][0, 0][0]
	result['age'] = takens - births

	# save as pandas dataframe
	col_name.append('age')
	result = pd.DataFrame(data = result, columns = col_name)
	result['db_name'] = db_name

	# handle inf value
	result = result.replace([-np.inf, np.inf], np.nan)
	return result

# def cleanData(db_frame):
# 	print(db_frame.describe())
# 	# print('Max age: ', db_frame['age'].max())
# 	# print('Min age: ', db_frame['age'].min())
# 	cleaned = db_frame.loc[(db_frame['age'] >= 0) & (db_frame['age'] <= 100) &\
# 							(~db_frame['face_score'].isnull()) &\
# 							(db_frame['second_face_score'].isnull()) &\
# 							(~db_frame['gender'].isnull()) ,\
# 							['db_name', 'full_path', 'age', 'gender']
# 						  ]
	# print(cleaned['full_path'].iloc[0], type(cleaned['full_path'].iloc[0]))
	# print(cleaned.columns)
	# print(db_frame.shape[0])
	# print(cleaned.shape[0])
	# print('Max age: ', cleaned['age'].max())
	# print('Min age: ', cleaned['age'].min())
	# print(np.sort(cleaned['age'])[-300:])

if __name__ == '__main__':
	# print(loadData('wiki', 'wiki.mat'))
	data = loadmat('imdb.mat')
	print(data['imdb'])
	# data = cleanData(data)