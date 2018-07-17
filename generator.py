from keras.utils import Sequence, np_utils
import numpy as np 
import cv2, os
def loadImage(db, paths):
    images = [cv2.imread(os.path.join('{}_aligned'.format(db), img_path[2:-2])) for (db, img_path) in zip(db,paths)]
    return np.array(images, dtype='uint8')

class TrainGenerator(Sequence):
    def __init__(self, model, db, paths, age, gender, batch_size):
        self.db = db 
        self.paths = paths
        self.age = age
        self.gender = gender
        self.batch_size = batch_size
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.db) / float(self.batch_size)))

    def __getitem__(self, idx):
        db = self.db[idx * self.batch_size:(idx + 1) * self.batch_size]
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = loadImage(db, paths)
        X = self.model.prepImg(batch_x)
        del db, paths, batch_x

        batch_age = self.age[idx * self.batch_size:(idx + 1) * self.batch_size]
        Age = np_utils.to_categorical(batch_age, 101)
        del batch_age

        batch_gender = self.gender[idx * self.batch_size:(idx + 1) * self.batch_size]
        Gender = np_utils.to_categorical(batch_gender, 2)
        del batch_gender

        Y = {'age_prediction': Age, 
			'gender_prediction': Gender}

        return X, Y

class TestGenerator(Sequence):
    def __init__(self, model, db, paths, age, gender, batch_size):
        self.db = db 
        self.paths = paths
        self.age = age
        self.gender = gender
        self.batch_size = batch_size
        self.model = model

    def __len__(self):
        return int(np.ceil(len(self.db) / float(self.batch_size)))

    def __getitem__(self, idx):
        db = self.db[idx * self.batch_size:(idx + 1) * self.batch_size]
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = loadImage(db, paths)
        X = self.model.prepImg(batch_x)
        del db, paths, batch_x

        batch_age = self.age[idx * self.batch_size:(idx + 1) * self.batch_size]
        Age = np_utils.to_categorical(batch_age, 101)
        del batch_age

        batch_gender = self.gender[idx * self.batch_size:(idx + 1) * self.batch_size]
        Gender = np_utils.to_categorical(batch_gender, 2)
        del batch_gender

        Y = {'age_prediction': Age, 
			'gender_prediction': Gender}

        return X, Y