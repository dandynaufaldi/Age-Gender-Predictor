import os, pickle, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from model import AgenderNetVGG16, AgenderNetInceptionV3, AgenderNetXception
from generator import TrainGenerator, TestGenerator
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.training_utils import multi_gpu_model
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',
					default=1, 
					type=int,
					help='Num of GPU')
parser.add_argument('--model',
					choices=['vgg16', 'inceptionv3', 'xception'],
					default='vgg16',
					help='Model to be used')
parser.add_argument('--trial',
					action='store_true',
					help='Run training to check code')

BATCH_SIZE = 32

def trainGenerator(model, trainX, trainAge, trainGender):
	L = trainX.shape[0]
	while True:
		batch_start = 0
		batch_end = BATCH_SIZE
		while batch_start < L:
			limit = min(batch_end, L)
			X = model.prepImg(trainX[batch_start:limit])
			Y = {"age_prediction": trainAge[batch_start:limit], 
					"gender_prediction": trainGender[batch_start:limit]}

			yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

			batch_start += BATCH_SIZE   
			batch_end += BATCH_SIZE

def testGenerator(model, testX, testAge, testGender):
	L = testX.shape[0]
	while True:
		batch_start = 0
		batch_end = BATCH_SIZE
		while batch_start < L:
			limit = min(batch_end, L)
			X = model.prepImg(testX[batch_start:limit])
			Y = {"age_prediction": testAge[batch_start:limit], 
					"gender_prediction": testGender[batch_start:limit]}

			yield (X,Y) #a tuple with two numpy arrays with batch_size samples     

			batch_start += BATCH_SIZE   
			batch_end += BATCH_SIZE

def loadImage(db_frame):
	print('[PREP] Read all image...')
	images = [cv2.resize(cv2.imread(os.path.join('{}_aligned'.format(db), img_path[2:-2])), (60,60)) \
			for (db, img_path) in tqdm(zip(db_frame['db_name'].tolist(), \
				db_frame['full_path'].tolist()), total=len(db_frame['db_name'].tolist()))]
	db_frame['image'] = images
	images = np.array(images)
	return db_frame, images

def prepData(trial):
	wiki = pd.read_csv('wiki.csv')
	if trial:
		wiki = wiki.iloc[:64]
	wiki, wiki_img = loadImage(wiki)
	imdb = pd.read_csv('imdb.csv')
	if trial :
		imdb = imdb.iloc[:64]
	imdb, imdb_img = loadImage(imdb)
	data = pd.concat([wiki, imdb], axis=0)
	del wiki, imdb
	X = np.append(wiki_img, imdb_img)
	ageLabel = np.array(data['age'], dtype='uint8')
	genderLabel = np.array(data['gender'], dtype='uint8')
	return X, ageLabel, genderLabel

def fitModel(model, trainX, trainAge, trainGender, testX, testAge, testGender, callbacks, GPU):
	return model.fit_generator(
			TrainGenerator(model, trainX, trainAge, trainGender, BATCH_SIZE),
			validation_data=TestGenerator(model, testX, testAge, testGender, BATCH_SIZE),
			epochs=100, 
			verbose=2,
			steps_per_epoch=len(trainX) // (BATCH_SIZE * GPU),
			validation_steps=len(testX) // (BATCH_SIZE * GPU),
			workers=4,
			callbacks=callbacks)

def main():
	args = parser.parse_args()
	GPU = args.gpu
	MODEL = args.model
	TRIAL = args.trial
	X, ageLabel, genderLabel = prepData(TRIAL)
	
	# genderLabel = np_utils.to_categorical(genderLabel, 2)
	# ageLabel = np_utils.to_categorical(ageLabel, 101)
	n_fold = 1
	print('Data size : ', X.size/(1024*1024))
	print('[K-FOLD] Started...')
	kf = KFold(n_splits=5)
	kf_split = kf.split(X)
	for train_idx, test_idx in kf_split:
		print('[K-FOLD] Fold {}'.format(n_fold))
		model = None
		trainModel = None
		if GPU == 1:
			if MODEL == 'vgg16':
				model = AgenderNetVGG16()
			elif MODEL == 'inceptionv3':
				model = AgenderNetInceptionV3()
			else :
				model = AgenderNetXception()
			# trainModel = model
		else :
			with tf.device("/cpu:0"):
				if MODEL == 'vgg16':
					model = AgenderNetVGG16()
				elif MODEL == 'inceptionv3':
					model = AgenderNetInceptionV3()
				else :
					model = AgenderNetXception()
		print('[PREP] Prepare data for training')
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
		# trainY = {"age_prediction": trainAge, 
		# 			"gender_prediction": trainGender}
		# testY = {"age_prediction": testAge, 
		# 		"gender_prediction": testGender}

		print('[PHASE-1] Training ...')
		callbacks = [
			EarlyStopping(monitor='val_loss', 
							patience=3, 
							verbose=0)]
		model.prepPhase1()
		trainModel = model
		if GPU > 1 :
			trainModel = multi_gpu_model(model, gpus=GPU)
		trainModel.compile(optimizer='adam', loss=losses, metrics=metrics)
		hist = fitModel(trainModel, trainX, trainAge, trainGender, testX, testAge, testGender, callbacks, GPU)
		with open(os.path.join('history', 'fold{}_p1.dict'.format(n_fold)), 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)
		
		print('[PHASE-2] Fine tuning ...')
		callbacks = [
			EarlyStopping(monitor='val_loss', 
							patience=3, 
							verbose=0),
			# ModelCheckpoint("weight/loss{val_loss:.2f}-fold{n_fold:02d}.h5".format(n_fold=n_fold),
            #                      monitor="val_loss",
            #                      verbose=1,
            #                      save_best_only=True,
			# 					 save_weights_only=True)
			]
		model.prepPhase2()
		trainModel = model
		if GPU > 1 :
			trainModel = multi_gpu_model(model, gpus=GPU)
		sgd = SGD(lr=0.0001, momentum=0.9)
		trainModel.compile(optimizer=sgd, loss=losses, metrics=metrics)
		hist = fitModel(trainModel, trainX, trainAge, trainGender, testX, testAge, testGender, callbacks, GPU)
		with open(os.path.join('history', 'fold{}_p2.dict'.format(n_fold)), 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)
		
		
		score = model.evaluate(testX,
				{"age_prediction": testAge, "gender_prediction": testGender})
		weightname = '{}_{}_{}.hdf5'.format(MODEL, n_fold, '_'.join(str(s) for s in score))
		model.save_weight(os.path.join('weight', weightname))
		print(score)

		n_fold += 1
		del trainX, trainAge, trainGender, testX, testAge, testGender, model, trainModel


if __name__ == '__main__':
	main()