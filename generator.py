from keras.utils import Sequence, np_utils
import numpy as np 
import cv2, os
def loadImage(db, paths, size):
    images = [cv2.imread(os.path.join('{}_aligned'.format(db), img_path)) 
                for (db, img_path) in zip(db,paths)]
    images = np.array(images)
    if images.shape[1] != size:
        images = [cv2.resize(image, (size,size), interpolation = cv2.INTER_CUBIC) for image in images]
    return np.array(images, dtype='uint8')

class DataGenerator(Sequence):
    def __init__(self, model, db, paths, age, gender, batch_size, input_size, categorical):
        self.db = db 
        self.paths = paths
        self.age = age
        self.gender = gender
        self.batch_size = batch_size
        self.model = model
        self.input_size = input_size
        self.categorical = categorical

    def __len__(self):
        return int(np.ceil(len(self.db) / float(self.batch_size)))

    def __getitem__(self, idx):
        db = self.db[idx * self.batch_size:(idx + 1) * self.batch_size]
        paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = loadImage(db, paths, self.input_size)
        X = self.model.prepImg(batch_x)
        del db, paths, batch_x

        batch_age = self.age[idx * self.batch_size:(idx + 1) * self.batch_size]
        Age = batch_age
        if self.categorical:
            Age = np_utils.to_categorical(batch_age, 101)
        del batch_age

        batch_gender = self.gender[idx * self.batch_size:(idx + 1) * self.batch_size]
        Gender = batch_gender
        if self.categorical :
            Gender = np_utils.to_categorical(batch_gender, 2)
        del batch_gender

        Y = {'age_prediction': Age, 
			'gender_prediction': Gender}

        return X, Y