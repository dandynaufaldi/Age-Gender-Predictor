import loadfile, preprocess
import os, cv2, pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import AgenderNetVGG16
from sklearn.preprocessing import LabelBinarizer
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD

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
	ageLB = LabelBinarizer()
	genderLabel = np_utils.to_categorical(genderLabel, 2)
	ageLabel = ageLB.fit_transform(ageLabel)
	n_fold = 0
	
	kf = KFold(n_splits=10, shuffle=True, random_state=1)
	for train_idx, test_idx in kf.split(X):
		model = AgenderNetVGG16().build()
		# model = create_model()
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
		
		#freeze conv block for 1st training phase
		print('[PHASE-1] Training ...')
		for layer in model.layers[:19]:
			layer.trainable = False
		model.compile(optimizer='adam', loss=losses, metrics=metrics)
		hist = model.fit(
			trainX,
			{"age_prediction": trainAge, 
			"gender_prediction": trainGender},
			validation_data=(
				testX,
				{"age_prediction": testAge, 
				"gender_prediction": testGender}),
			epochs=30, 
			verbose=1, 
			batch_size=100)
		pd.DataFrame(hist.history).to_hdf(os.path.join('history', 'fold{}_p1'.format(n_fold)))
		
		
		#unfreeze few last conv block
		print('[PHASE-2] Fine tuning ...')
		for layer in model.layers[11:]:
			layer.trainable = True
		sgd = SGD(lr=0.0001, momentum=0.9)
		model.compile(optimizer=sgd, loss=losses, metrics=metrics)
		hist = model.fit(
			trainX,
			{"age_prediction": trainAge, 
			"gender_prediction": trainGender},
			validation_data=(
				testX,
				{"age_prediction": testAge, 
				"gender_prediction": testGender}),
			epochs=30, 
			verbose=1, 
			batch_size=100)
		pd.DataFrame(hist.history).to_hdf(os.path.join('history', 'fold{}_p2'.format(n_fold)))
		score = model.evaluate(testX,
				{"age_prediction": testAge, "gender_prediction": testGender})
		print(score)
		model.save_weights('weight', 'fold{}_{}'.format(n_fold, '_'.join(str(num) for num in score)))
		n_fold += 1
if __name__ == '__main__':
	main()

'''
Train FC layer first, freeze VGG conv block
Unfreeze few lasy conv block, retrain
'''