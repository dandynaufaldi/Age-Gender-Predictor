import os, pickle, cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from model import AgenderNetVGG16, AgenderNetInceptionV3, AgenderNetXception
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint

def loadImage(db_frame):
	print('[PREP] Read all image...')
	images = [cv2.imread(os.path.join('{}_aligned'.format(db), img_path[0])) \
			for (db, img_path) in tqdm(zip(db_frame['db_name'].tolist(), \
				db_frame['full_path'].tolist()), total=len(db_frame['db_name'].tolist()))]
	db_frame['image'] = images
	return db_frame

def prepData():
	wiki = pd.read_csv('wiki.csv')
	wiki = loadImage(wiki)
	imdb = pd.read_csv('imdb.csv')
	imdb = loadImage(imdb)
	data = pd.concat([wiki, imdb], axis=0)
	del wiki, imdb

	X = data['image']
	X = np.array(X)
	ageLabel = np.array(data['age'], dtype='uint8')
	genderLabel = np.array(data['gender'], dtype='uint8')
	return X, ageLabel, genderLabel

def fitModel(model, trainX, trainY, testX, testY, callbacks):
	return model.fit(
			trainX,
			trainY,
			validation_data=(
				testX,
				testY),
			epochs=100, 
			verbose=2, 
			batch_size=100,
			callbacks=callbacks)

def main():
	X, ageLabel, genderLabel = prepData()
	genderLabel = np_utils.to_categorical(genderLabel, 2)
	ageLabel = np_utils.to_categorical(ageLabel, 101)
	n_fold = 0
	
	kf = KFold(n_splits=10, shuffle=True, random_state=1)
	for train_idx, test_idx in kf.split(X):
		model = AgenderNetVGG16()
		trainX = model.prepImg(X[train_idx])
		trainAge = ageLabel[train_idx]
		trainGender = genderLabel[train_idx]
		testX = model.prepImg(X[test_idx])
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
		trainY = {"age_prediction": trainAge, 
					"gender_prediction": trainGender}
		testY = {"age_prediction": testAge, 
				"gender_prediction": testGender}

		print('[PHASE-1] Training ...')
		callbacks = [
			EarlyStopping(monitor='val_loss', 
							patience=3, 
							verbose=0)]
		model.prepPhase1()
		model.compile(optimizer='adam', loss=losses, metrics=metrics)
		hist = fitModel(model, trainX, trainY, testX, testY, callbacks)
		with open(os.path.join('history', 'fold{}_p1.dict'.format(n_fold)), 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)
		
		print('[PHASE-2] Fine tuning ...')
		callbacks = [
			EarlyStopping(monitor='val_loss', 
							patience=3, 
							verbose=0),
			ModelCheckpoint("weight/loss{val_loss:.2f}-fold{n_fold:02d}.h5".format(n_fold=n_fold),
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
								 save_weights_only=True)]
		model.prepPhase2()
		sgd = SGD(lr=0.0001, momentum=0.9)
		model.compile(optimizer=sgd, loss=losses, metrics=metrics)
		hist = fitModel(model, trainX, trainY, testX, testY, callbacks)
		with open(os.path.join('history', 'fold{}_p2.dict'.format(n_fold)), 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)
		score = model.evaluate(testX,
				{"age_prediction": testAge, "gender_prediction": testGender})
		print(score)
		n_fold += 1


if __name__ == '__main__':
	main()