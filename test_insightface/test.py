import face_model
import argparse
import cv2
import sys
import numpy as np
import pandas as pd 
from tqdm import tqdm
import logging
import time
import timeit

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='', help='path to load model.')
parser.add_argument('--ga-model', default='./models/model-r34-age/model,0', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')


def UTK(model):
    data = pd.read_csv('../dataset/UTKface.csv')
    paths = data['full_path'].values
    logger.info('Read all UTKface image')
    images = [cv2.imread('../UTKface/'+path) for path in tqdm(paths)]
    logger.info('Get aligned face')
    inputs = [model.get_input(image) for image in tqdm(images)]
    del images
    img = np.array(inputs)

    pred_age = dict()
    pred_gender = dict()

    logger.info('Predict UTKface')
    start = time.time()
    pred_gender['insightface'], pred_age['insightface'] = model.get_ga(img)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))
    pred_age = pd.DataFrame.from_dict(pred_age)
    pred_gender = pd.DataFrame.from_dict(pred_gender)

    pred_age = pd.concat([data['age'], pred_age], axis=1)
    pred_gender = pd.concat([data['gender'], pred_gender], axis=1)

    pred_age.to_csv('utk_age_prediction.csv', index=False)
    pred_gender.to_csv('utk_gender_prediction.csv', index=False)

def FGNET(model):
    data = pd.read_csv('../dataset/FGNET.csv')
    paths = data['full_path'].values
    logger.info('Read all FGNET image')
    images = [cv2.imread('../FGNET/images/'+path) for path in tqdm(paths)]
    logger.info('Get aligned face')
    inputs = [model.get_input(image) for image in tqdm(images)]
    del images
    img = np.array(inputs)

    pred_age = dict()
    pred_gender = dict()

    logger.info('Predict FGNET')
    start = time.time()
    pred_gender['insightface'], pred_age['insightface'] = model.get_ga(img)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))
    pred_age = pd.DataFrame.from_dict(pred_age)

    pred_age = pd.concat([data['age'], pred_age], axis=1)

    pred_age.to_csv('fgnet_age_prediction.csv', index=False)

def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

def predictone(model, x):
    _ = model.get_ga(x, 1)

def proces_time(wrapped):
    number = 100
    elapsed = timeit.repeat(wrapped, repeat=10, number=number)
    elapsed = np.array(elapsed)
    per_pass = elapsed / number
    mean = np.mean(per_pass) * 1000
    std = np.std(per_pass) * 1000
    result = '{:6.2f} msec/pass +- {:6.2f} msec'.format(mean, std)
    return result

def check_inference_time(model):
    logger.info('Check inference time')
    image = cv2.imread('../UTKface/part1/34_1_0_20170103183147490.jpg')
    X = model.get_input(image)
    X = np.expand_dims(X, axis=0)
    wrapped = wrapper(predictone, model, X)
    logger.info(proces_time(wrapped))

def check(model):
    img = cv2.imread('Tom_Hanks_54745.png')
    img = model.get_input(img)
    a = cv2.imread('27_1_0_20170119163226214.jpg')
    a = model.get_input(a)
    img = np.array([img, a])
    gender, age = model.get_ga(img)
    print(gender)
    print(age)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    args = parser.parse_args()
    logger.info('Load model')
    model = face_model.FaceModel(args)
    UTK(model)
    FGNET(model)
    check_inference_time(model)
    # check(model)