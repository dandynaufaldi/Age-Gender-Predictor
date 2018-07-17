from keras.utils import Sequence, np_utils
import numpy as np 

class TrainGenerator(Sequence):
    def __init__(self, model, trainX, trainAge, trainGender, batch_size):
        self.x, self.age, self.gender = trainX, trainAge, trainGender
        self.batch_size = batch_size
        self.model = self.model

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = self.model.prepImg(batch_x)
        del batch_x

        batch_age = self.age[idx * self.batch_size:(idx + 1) * self.batch_size]
        trainAge = np_utils.to_categorical(batch_age, 101)
        del batch_age

        batch_gender = self.gender[idx * self.batch_size:(idx + 1) * self.batch_size]
        trainGender = np_utils.to_categorical(batch_gender, 2)
        del batch_gender

        Y = {'age_prediction': trainAge, 
			'gender_prediction': trainGender}

        return X, Y

class TestGenerator(Sequence):
    def __init__(self, model, testX, testAge, testGender, batch_size):
        self.x, self.age, self.gender = testX, testAge, testGender
        self.batch_size = batch_size
        self.model = self.model

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = self.model.prepImg(batch_x)
        del batch_x

        batch_age = self.age[idx * self.batch_size:(idx + 1) * self.batch_size]
        testAge = np_utils.to_categorical(batch_age, 101)
        del batch_age

        batch_gender = self.gender[idx * self.batch_size:(idx + 1) * self.batch_size]
        testGender = np_utils.to_categorical(batch_gender, 2)
        del batch_gender

        Y = {'age_prediction': testAge, 
			'gender_prediction': testGender}

        return X, Y