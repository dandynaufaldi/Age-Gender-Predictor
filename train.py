import loadfile, preprocess
import os, cv2, pickle
import numpy as np
from tqdm import tqdm
from model import AgenderNetVGG16
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D

from keras.models import Model
def create_model():
	inputs = Input(shape=(48,48,3))
	layer = Conv2D(16, (5, 5), activation='relu' , padding='same')(inputs)
	layer = Conv2D(32, (5, 5), activation='relu', padding='valid')(layer)
	layer = Conv2D(128, (5, 5), activation='relu', padding='valid')(layer)
	layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(layer)

	layer = Conv2D(256, (5, 5), activation='relu', padding='same')(layer)
	layer = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(layer)

	layer = Conv2D(256, (5, 5), activation='relu', padding='same')(layer)

	layer = Flatten()(layer)
	layer = Dense(512, activation='relu')(layer)
	layer = Dropout(0.5)(layer)

	genderLayer = Dense(2, activation='softmax', name='gender_prediction')(layer)
	ageLayer = Dense(10, activation='softmax', name='age_prediction')(layer)
	res = Model(input=inputs, output=[genderLayer, ageLayer], name='AgenderNetVGG16')
	print(res.summary())
	return res

def prepData():
	data = loadfile.loadData('wiki', 'wiki.mat')
	data = preprocess.cleanData(data)
	data = data.iloc[:10]
	data = preprocess.loadImage(data)
	print('[PREPROC] Run face alignment...')
	X = [preprocess.getAlignedFace(img) for img in tqdm(data['image'].tolist())]
	X = np.array(X, dtype='float')
	X -= np.mean(X, axis=0)
	X /= np.std(X, axis=0)
	ageLabel = np.array(data['age'], dtype='uint8')
	genderLabel = np.array(data['gender'], dtype='uint8')
	return X, ageLabel, genderLabel

def main():
	X, ageLabel, genderLabel = prepData()	
	# print(genderLabel)
	ageLB = LabelBinarizer()
	genderLabel = np_utils.to_categorical(genderLabel, 2)
	ageLabel = ageLB.fit_transform(ageLabel)
	# print(genderLabel[0], genderLabel.shape)
	# print(ageLabel.shape)
	# print(genderLB.classes_)
	# print(ageLB.classes_)
	n_fold = 0
	
	kf = KFold(n_splits=10, shuffle=True, random_state=1)
	for train_idx, test_idx in kf.split(X):
		# model = AgenderNetVGG16().build()
		model = create_model()
		trainX = X[train_idx]
		trainAge = ageLabel[train_idx]
		trainGender = genderLabel[train_idx]
		testX = X[test_idx]
		testAge = ageLabel[test_idx]
		testGender = genderLabel[test_idx]

		losses = {
			"age_prediction": "categorical_crossentropy",
			"gender_prediction": "categorical_crossentropy",
		}
		metrics = {
			"age_prediction": "mae",
			"gender_prediction": "acc",
		}
		
		model.compile(optimizer='adam', loss=losses, 
			metrics=metrics)
		H = model.fit(trainX,
			{"age_prediction": trainAge, "gender_prediction": trainGender},
			validation_data=(testX,
				{"age_prediction": testAge, "gender_prediction": testGender}),
			epochs=30, verbose=2, batch_size=1)
		score = model.evaluate(testX,
				{"age_prediction": testAge, "gender_prediction": testGender})
		print(score)
if __name__ == '__main__':
	# data = loadfile.loadData('wiki', 'wiki.mat')
	# data = preprocess.cleanData(data)
	# data = data.iloc[:10]
	# data = preprocess.loadImage(data)
	# print('[PREPROC] Run face alignment...')
	# X = [preprocess.getAlignedFace(img) for img in tqdm(data['image'].tolist())]
	# X = np.array(X, dtype='float')
	main()
	# model = create_model()