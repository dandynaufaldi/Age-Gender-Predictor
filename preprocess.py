'''
1.	Load gambar dr full path
2.	Masukin ke face detect
3.	Kalau ga ada muka, resize gambar asli
	Kalau ada, masukin ke alignment
	Ukuran resize langsung sama align diatur biar sama
'''
import numpy as np
import cv2
import dlib
import os

'''
Clean the data for abnormal age (negative),	no face 
detected (face_score is NaN), more than one face 
(both face_score and second_face_score not NaN) 
'''
def cleanData(db_frame):
	cleaned = db_frame.loc[(db_frame['age'] > 0) &\
							(~db_frame['face_score'].isnull()) &\
							(db_frame['second_face_score'].isnull()) &\
							(~db_frame['gender'].isnull()) ,\
							['db_name', 'full_path', 'age', 'gender']
						  ]
	return cleaned


'''

'''