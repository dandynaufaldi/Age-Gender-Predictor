import os, pickle, cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from keras import backend as K
from keras.utils import Sequence
from tqdm import tqdm
from SSRNET_model import SSR_net, SSR_net_general

class DataGenerator(Sequence):
    def __init__(self, model, images, label, batch_size):
        self.images = images
        self.label = label
        self.batch_size = batch_size
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.label) / float(self.batch_size)))

    def __getitem__(self, idx):
        X = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = X.astype('float16')
        Y = self.label[idx * self.batch_size:(idx + 1) * self.batch_size]
        return X, Y

def prepData(img_size):
    wiki = pd.read_csv('../dataset/wiki_cleaned.csv')
    imdb = pd.read_csv('../dataset/imdb_cleaned.csv')
    adience = pd.read_csv('../dataset/adience_u20.csv')
    data = pd.concat([wiki, imdb, adience], axis=0)
    del wiki, imdb, adience
    db = data['db_name'].values
    paths = data['full_path'].values
    images = [cv2.resize(cv2.imread(os.path.join('../{}_aligned'.format(db), img_path)), (img_size, img_size), interpolation = cv2.INTER_CUBIC) 
                for (db, img_path) in tqdm(zip(db,paths), total = len(db))]
    images = np.array(images)
    ageLabel = np.array(data['age'], dtype='uint8')
    genderLabel = np.array(data['gender'], dtype='uint8')
    return images, ageLabel, genderLabel

def evaluate(model, testImage, testLabel):
    generator = DataGenerator(model, testImage, testLabel, 128)
    scores =  model.evaluate_generator(generator, max_queue_size=1.5*128)
    del generator
    return scores

def main():
    #dynamicaly allocate GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    print('[LOAD DATA]')
    images, ageLabel, genderLabel = prepData(64)
    n_fold = 1
    
    img_size = 64
    stage_num = [3,3,3]
    lambda_local = 1
    lambda_d = 1
    imdb_model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    imdb_model.compile(optimizer='adam', loss="mae", metrics=["mae"])
    imdb_model.load_weights("imdb_age_ssrnet_3_3_3_64_1.0_1.0.h5")
    
    imdb_model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
    imdb_model_gender.compile(optimizer='adam', loss="mae", metrics=["binary_accuracy"])
    imdb_model_gender.load_weights("imdb_gender_ssrnet_3_3_3_64_1.0_1.0.h5")

    wiki_model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    wiki_model.compile(optimizer='adam', loss="mae", metrics=["mae"])
    wiki_model.load_weights("wiki_age_ssrnet_3_3_3_64_1.0_1.0.h5")
    
    wiki_model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
    wiki_model_gender.compile(optimizer='adam', loss="mae", metrics=["binary_accuracy"])
    wiki_model_gender.load_weights("wiki_gender_ssrnet_3_3_3_64_1.0_1.0.h5")

    morph_model = SSR_net(img_size,stage_num, lambda_local, lambda_d)()
    morph_model.compile(optimizer='adam', loss="mae", metrics=["mae"])
    morph_model.load_weights("morph_age_ssrnet_3_3_3_64_1.0_1.0.h5")
    
    morph_model_gender = SSR_net_general(img_size,stage_num, lambda_local, lambda_d)()
    morph_model_gender.compile(optimizer='adam', loss="mae", metrics=["binary_accuracy"])
    morph_model_gender.load_weights("morph_gender_ssrnet_3_3_3_64_1.0_1.0.h5")

   
    print('[K-FOLD] Started...')
    kf = KFold(n_splits=10, shuffle=True, random_state=1)
    kf_split = kf.split(ageLabel)
    for _, test_idx in kf_split:
        print('[K-FOLD] Fold {}'.format(n_fold))      
        testImages = images[test_idx]
        testAge = ageLabel[test_idx]
        testGender = genderLabel[test_idx]

        scores = evaluate(imdb_model, testImages, testAge)
        print('imdb Age score:', scores)
        scores = evaluate(wiki_model, testImages, testAge)
        print('wiki Age score:', scores)
        scores = evaluate(morph_model, testImages, testAge)
        print('morph Age score:', scores)

        scores = evaluate(imdb_model_gender, testImages, testGender)
        print('imdb Gender score:', scores)
        scores = evaluate(wiki_model_gender, testImages, testGender)
        print('wiki Gender score:', scores)
        scores = evaluate(morph_model_gender, testImages, testGender)
        print('morph Gender score:', scores)

        n_fold += 1
        del	testImages, testAge, testGender, scores


if __name__ == '__main__':
	main()