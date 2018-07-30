import os, pickle, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from model import SSRNetGeneral
from keras.utils import np_utils
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
import TYY_callbacks
import argparse
from keras.utils import Sequence

def loadImage(db, paths, size):
	images = [cv2.imread(os.path.join('{}_aligned'.format(db), img_path)) 
				for (db, img_path) in zip(db,paths)]
	images = np.array(images)
	if images.shape[1] != size:
		images = [cv2.resize(image, (size,size), interpolation = cv2.INTER_CUBIC) for image in images]
	return np.array(images, dtype='uint8')

class DataGenerator(Sequence):
	def __init__(self, model, db, paths, label, batch_size, input_size):
		self.db = db 
		self.paths = paths
		self.label = label
		self.batch_size = batch_size
		self.model = model
		self.input_size = input_size

	def __len__(self):
		return int(np.ceil(len(self.db) / float(self.batch_size)))

	def __getitem__(self, idx):
		db = self.db[idx * self.batch_size:(idx + 1) * self.batch_size]
		paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_x = loadImage(db, paths, self.input_size)
		X = self.model.prepImg(batch_x)
		del db, paths, batch_x

		Y = self.label[idx * self.batch_size:(idx + 1) * self.batch_size]

		return X, Y

parser = argparse.ArgumentParser()
parser.add_argument('--label',
					choices=['age', 'gender'],
					help='Target label to be used')
parser.add_argument('--epoch',
					default=50,
					type=int,
					help='Num of training epoch')
parser.add_argument('--batch_size',
					default=64,
					type=int,
					help='Size of data batch to be used')
parser.add_argument('--num_worker',
					default=4,
					type=int,
					help='Number of worker to process data')

def prepData():
	wiki = pd.read_csv('dataset/wiki_cleaned.csv')
	imdb = pd.read_csv('dataset/imdb_cleaned.csv')
	adience = pd.read_csv('dataset/adience_u20.csv')
	data = pd.concat([wiki, imdb, adience], axis=0)
	del wiki, imdb, adience
	db = data['db_name'].values
	paths = data['full_path'].values
	ageLabel = np.array(data['age'], dtype='uint8')
	genderLabel = np.array(data['gender'], dtype='uint8')
	return db, paths, ageLabel, genderLabel

def fitModel(model, input_size,
			trainDb, trainPaths, trainLabel, 
			testDb, testPaths, testLabel,
			epoch, batch_size, num_worker,
			callbacks):
	return model.fit_generator(
			DataGenerator(model, trainDb, trainPaths, trainLabel, batch_size, input_size),
			validation_data=DataGenerator(model, testDb, testPaths, testLabel, batch_size, input_size),
			epochs=epoch, 
			verbose=2,
			#steps_per_epoch=len(trainLabel) // batch_size,
			#validation_steps=len(testLabel) // batch_size,
			workers=num_worker,
			use_multiprocessing=True,
			max_queue_size=int(batch_size * 1.5),
			callbacks=callbacks)


def main():
	#dynamicaly allocate GPU memory
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)
	K.tensorflow_backend.set_session(sess)

	args = parser.parse_args()
	LABEL = args.label
	EPOCH = args.epoch
	BATCH_SIZE = args.batch_size
	NUM_WORKER = args.num_worker
	INPUT_SIZE = 64

	db, paths, ageLabel, genderLabel = prepData()

	n_fold = 1
	print('[K-FOLD] Started...')
	kf = KFold(n_splits=10, shuffle=True, random_state=1)
	kf_split = kf.split(ageLabel)
	for train_idx, test_idx in kf_split:
		print('[K-FOLD] Fold {}'.format(n_fold))
		model = SSRNetGeneral(64, [3, 3, 3], 1.0, 1.0, LABEL)
		trainDb = db[train_idx]
		trainPaths = paths[train_idx]
		trainLabel = None
		if LABEL == 'age':
			trainLabel = ageLabel[train_idx]
		else :
			trainLabel = genderLabel[train_idx]
		
		testDb = db[test_idx]
		testPaths = paths[test_idx]
		testLabel = None
		if LABEL == 'age':
			testLabel = ageLabel[test_idx]
		else :
			testLabel = genderLabel[test_idx]
		losses = "mae"
		metrics = None
		if LABEL == 'age':
			metrics = ["mae"]
		else :
			metrics = ["binary_accuracy"]
		
		callbacks = [TYY_callbacks.DecayLearningRate([30, 60])]
		if LABEL == 'age':
			callbacks.append(ModelCheckpoint("trainweight/model.{epoch:02d}-{val_loss:.4f}-{val_mean_absolute_error:.4f}.h5",
								verbose=1,
								save_best_only=True))
		else :
			callbacks.append(ModelCheckpoint("trainweight/model.{epoch:02d}-{val_loss:.4f}-{val_binary_accuracy:.4f}.h5",
								verbose=1,
								save_best_only=True))

		model.compile(optimizer='adam', loss=losses, metrics=metrics)
		hist = fitModel(model, INPUT_SIZE,
						trainDb, trainPaths, trainLabel, 
						testDb, testPaths, testLabel, 
						EPOCH, BATCH_SIZE, NUM_WORKER, 
						callbacks)
		with open(os.path.join('history', 'fold{}_p2.dict'.format(n_fold)), 'wb') as file_hist:
			pickle.dump(hist.history, file_hist)

		n_fold += 1
		del trainDb, trainPaths, trainLabel
		del	testDb, testPaths, testLabel
		del	callbacks, model


if __name__ == '__main__':
	main()