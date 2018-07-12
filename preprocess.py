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
import pandas as pd
from tqdm import tqdm
'''
Clean the data for abnormal age (negative),	no face 
detected (face_score is NaN), more than one face 
(both face_score and second_face_score not NaN)
Input : pandas DataFrame
Output : pandas DataFrame
'''
def cleanData(db_frame):
	cleaned = db_frame.loc[(db_frame['age'] >= 0) &\
							(db_frame['age'] <= 100) &\
							(~db_frame['face_score'].isnull()) &\
							(db_frame['second_face_score'].isnull()) &\
							(~db_frame['gender'].isnull()) ,\
							['db_name', 'full_path', 'age', 'gender']
						  ]
	return cleaned


'''
Load all image from dataset
Input : pandas DataFrame
Output : pandas DataFrame
'''
def loadImage(db_frame, test=False):
	images = None
	print('[PREP] Read all image...')
	if test == True:
		images = [cv2.imread(os.path.join('{}_crop'.format(db), img_path[0])) \
				for (db, img_path) in tqdm(zip(db_frame['db_name'].tolist()[:10], \
					db_frame['full_path'].tolist()[:10]), total=len(db_frame['db_name'].tolist()[:10]))]
	else :
		images = [cv2.imread(os.path.join('{}_crop'.format(db), img_path[0])) \
				for (db, img_path) in tqdm(zip(db_frame['db_name'].tolist(), \
									db_frame['full_path'].tolist()), total=len(db_frame['db_name'].tolist()))]
	db_frame['image'] = images
	return db_frame

'''
Get resized image
Return square-resized image
'''
def resizeImg(image, size=140):
	BLACK = [0,0,0]
	h = image.shape[0]
	w = image.shape[1]
	if w<h:
	    border = h-w
	    image= cv2.copyMakeBorder(image,0,0,border,0,cv2.BORDER_CONSTANT,value=BLACK)
	else:
	    border = w-h
	    image= cv2.copyMakeBorder(image,border,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
	resized = cv2.resize(image, (size,size), interpolation = cv2.INTER_CUBIC)    
	return resized


'''
Get aligned face if possible, else return whole resized image
Input : single image (numpy array)
Output : single image (numpy array)
''' 
def getAlignedFace(image, padding=0.4):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
	rects = detector(image, 1)
	result = None
	
	# if detect exactly 1 face, get aligned face
	if len(rects) == 1:
		shape = predictor(image, rects[0])
		result = dlib.get_face_chip(image, shape, padding=padding, size=140)
	# use resized full image
	else :
		result = resizeImg(image)
	return result
