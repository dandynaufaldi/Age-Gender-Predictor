import os, pickle, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from keras import backend as K
from keras.utils import Sequence
from SSRNET_model import SSR_net, SSR_net_general

def loadImage(db, paths, size):
    images = [cv2.imread(os.path.join('../{}_aligned'.format(db), img_path)) 
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
        X = batch_x
        del db, paths, batch_x

        batch_label = self.label[idx * self.batch_size:(idx + 1) * self.batch_size]
        Y = batch_label
        del batch_label

        return X, Y

def prepData(trial):
	wiki = pd.read_csv('../dataset/wiki_cleaned.csv')
	imdb = pd.read_csv('../dataset/imdb_cleaned.csv')
	adience = pd.read_csv('../dataset/adience_u20.csv')
	data = pd.concat([wiki, imdb, adience], axis=0)
	del wiki, imdb, adience
	db = data['db_name'].values
	paths = data['full_path'].values
	ageLabel = np.array(data['age'], dtype='uint8')
	genderLabel = np.array(data['gender'], dtype='uint8')
	return db, paths, ageLabel, genderLabel

def evaluate(model, testDb, testPaths, testLabel):
    generator = DataGenerator(model, testDb, testPaths, testLabel, 128, 64)
    scores =  model.evaluate_generator(generator, max_queue_size=1.5*128)
    del generator
    return scores

def main():
    #dynamicaly allocate GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)

    db, paths, ageLabel, genderLabel = prepData(True)
    n_fold = 1
    weight_file = "ssrnet_3_3_3_64_1.0_1.0.h5"
    weight_file_gender = "gender_ssrnet_3_3_3_64_1.0_1.0.h5"
    
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    model.compile(optimizer='adam', loss="mae", metrics=["mae"])
    model.load_weights(weight_file)
    
    model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
    model_gender.compile(optimizer='adam', loss="mae", metrics=["binary_accuracy"])
    model_gender.load_weights(weight_file_gender)
   
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(ageLabel)
    for _, test_idx in kf_split:
        print('[K-FOLD] Fold {}'.format(n_fold))      
        testDb = db[test_idx]
        testPaths = paths[test_idx]
        testAge = ageLabel[test_idx]
        testGender = genderLabel[test_idx]

        scores = evaluate(model, testDb, testPaths, testAge)
        print('Age score:', scores)

        scores = evaluate(model_gender, testDb, testPaths, testGender)
        print('Gender score:', scores)

        n_fold += 1
        del	testDb, testPaths, testAge, testGender, scores


if __name__ == '__main__':
	main()