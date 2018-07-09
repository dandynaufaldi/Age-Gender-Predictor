import loadfile, preprocess
import os, cv2, pickle
import numpy as np
from tqdm import tqdm
from model import AgenderNetVGG16
from sklearn.preprocessing import LabelBinarizer
from sklearn.cross_validation import KFold

def prepData():
	data = loadfile.loadData('wiki', 'wiki.mat')
	data = preprocess.cleanData(data)
	data = data.iloc[:10]
	data = preprocess.loadImage(data)
	print('[PREPROC] Run face alignment...')
	X = [preprocess.getAlignedFace(img) for img in tqdm(data['image'].tolist())]
	X = np.array(X, dtype='float')
	X = (X - np.mean(X)) / np.std(X)
	ageLabel = np.array(data['Age'])
	genderLabel = np.array(data['gender'])
	return X, ageLabel, genderLabel


if __name__ == '__main__':
	X, ageLabel, genderLabel = prepData()	
	genderLB = LabelBinarizer()
	ageLB = LabelBinarizer()
	genderLabel = genderLB.fit_transform(genderLabel)
	ageLabel = ageLB.fit_transform(ageLabel)

	n_fold = 0
	
	kf = KFold(n_folds=5, shuffle=True, random_state=1)
	for train_idx, test_idx in kf.split(X):
		model = AgenderNetVGG16().build()
		trainX = X[train_idx]
		trainAge = ageLabel[train_idx]
		trainGender = genderLabel[train_idx]
		testX = X[test_idx]
		testAge = ageLabel[test_idx]
		testGender = genderLabel[test_idx]
