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
					default='inceptionv3',
					help='Model to be used')
parser.add_argument('--trial',
					action='store_true',
					help='Run training to check code')

BATCH_SIZE = 32

def loadImage(db_frame):
	print('[PREP] Read all image...')
	images = [cv2.resize(cv2.imread(os.path.join('{}_aligned'.format(db), img_path[2:-2])), (60,60)) \
			for (db, img_path) in tqdm(zip(db_frame['db_name'].tolist(), \
				db_frame['full_path'].tolist()), total=len(db_frame['db_name'].tolist()))]
	images = np.array(images)
	return db_frame, images

def prepData(trial):
	wiki = pd.read_csv('wiki.csv')
	imdb = pd.read_csv('imdb.csv')
	data = pd.concat([wiki, imdb], axis=0)
	del wiki, imdb
	db = data['db_name'].values
	paths = data['full_path'].values
	ageLabel = np.array(data['age'], dtype='uint8')
	genderLabel = np.array(data['gender'], dtype='uint8')
	return db, paths, ageLabel, genderLabel

def fitModel(model, 
			trainDb, trainPaths, trainAge, trainGender, 
			testDb, testPaths, testAge, testGender, 
			callbacks, GPU):
	return model.fit_generator(
			TrainGenerator(model, trainDb, trainPaths, trainAge, trainGender, BATCH_SIZE),
			validation_data=TestGenerator(model, testDb, testPaths, testAge, testGender, BATCH_SIZE),
			epochs=100, 
			verbose=1,
			steps_per_epoch=len(trainAge) // (BATCH_SIZE * GPU),
			validation_steps=len(testAge) // (BATCH_SIZE * GPU),
			workers=8,
			use_multiprocessing=True,
			max_queue_size=50,
			callbacks=callbacks)

def main():
	args = parser.parse_args()
	GPU = args.gpu
	MODEL = args.model
	TRIAL = args.trial
	db, paths, ageLabel, genderLabel = prepData(TRIAL)

	n_fold = 1
	print('[K-FOLD] Started...')
	kf = KFold(n_splits=10)
	kf_split = kf.split(ageLabel)
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
		trainDb = db[train_idx]
		trainPaths = paths[train_idx]
		trainAge = ageLabel[train_idx]
		trainGender = genderLabel[train_idx]
		
		testDb = db[test_idx]
		testPaths = paths[test_idx]
		testAge = ageLabel[test_idx]
		testGender = genderLabel[test_idx]

		losses = {
			"age_prediction": "categorical_crossentropy",
			"gender_prediction": "categorical_crossentropy",
		}
		metrics = {
			"age_prediction": "acc",
			"gender_prediction": "acc",
		}
		
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
		hist = fitModel(model, 
					trainDb, trainPaths, trainAge, trainGender, 
					testDb, testPaths, testAge, testGender, 
					callbacks, GPU)
		with open(os.path.join('history', 'fold{}_p1.dict'.format(n_fold)), 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)
		
		print('[PHASE-2] Fine tuning ...')
		callbacks = [
			EarlyStopping(monitor='val_loss', 
							patience=3, 
							verbose=0),
			ModelCheckpoint("weight/model.{epoch:02d}-{val_loss:.4f}-{val_acc:.4f}-{acc:.4f}.h5",
                                 monitor="val_loss",
                                 verbose=1,
                                 save_best_only=True,
								 save_weights_only=True)
			]
		model.prepPhase2()
		trainModel = model
		if GPU > 1 :
			trainModel = multi_gpu_model(model, gpus=GPU)
		sgd = SGD(lr=0.0001, momentum=0.9)
		trainModel.compile(optimizer=sgd, loss=losses, metrics=metrics)
		hist = fitModel(model, 
						trainDb, trainPaths, trainAge, trainGender, 
						testDb, testPaths, testAge, testGender, 
						callbacks, GPU)
		with open(os.path.join('history', 'fold{}_p2.dict'.format(n_fold)), 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)

		n_fold += 1
		del trainDb, trainPaths, trainAge, trainGender 
		del	testDb, testPaths, testAge, testGender
		del	callbacks, model, trainModel


if __name__ == '__main__':
	main()