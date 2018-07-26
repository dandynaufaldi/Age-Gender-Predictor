'''
1. Read from benchmarking dataset (use argument?)
2. detect and aligned all face
    - save original face coordinate
    saved_data = aligned_face (np.array of uint8), koordinat wajah asli
3. Predict using the model
    - run model.preprocess
    - run model.predict (use saved weight)
    - map the prediction (from np_utils.to_categorical format, use numpy argmax)
4. Save predicted result, measure 
5. Visualize the result
    - use bounding box
    - put text (age, gender)
'''
import argparse, os, glob, cv2, dlib, time
import pandas as pd 
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
from model import AgenderNetVGG16, AgenderNetInceptionV3, AgenderNetXception, SSRNet
from generator import DataGenerator
from SSRNET_model import SSR_net, SSR_net_general
import logging
def get_one_aligned_face(image,
                    padding=0.4,
                    size=140,
                    predictpath='shape_predictor_5_face_landmarks.dat'):
    """
    Get aligned face from a image using dlib

    Parameters
    ----------
    image   : numpy array -> with dtype uint8 and shape (W, H, 3)
        Image to be used in alignment
    padding     : float
        Padding to be applied around aligned face
    size        : int
        Size of aligned_face to be returned
    predictpath   : str
        Path to predictor being used to get facial landmark (5 points, 68 points, etc)
    
    Returns
    ----------
    aligned face    : numpy array -> with dtype uint8 and shape (H, W, 3)
        if detect only 1 face
            return aligned face
        else
            return resized image
    position        : dict
        Dictionary of left, top, right, and bottom position from face
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictpath)
    rects = detector(image, 1)

    aligned = None
    # position = None
    if len(rects == 1):     # detect 1 face
        shape = predictor(image, rects[0])
        aligned = dlib.get_face_chip(image, shape, padding=padding, size=size)
        # position = {'left'  : rects[0].left(),
        #             'top'   : rects[0].top(),
        #             'right' : rects[0].right(),
        #             'bottom': rects[0].bottom()}
    else :
        aligned = resize_image(image, size=size)
        # position = {'left'  : 0,
        #             'top'   : 0,
        #             'right' : image.shape[1],
        #             'bottom': image.shape[0]}

    return aligned #, position


def resize_image(image,
                size=140):
    """
    Resize image and make it square

    Parameters
    ----------
    image       : numpy array -> with dtype uint8 and shape (W, H, 3)
        Image to be resized
    size        : int
        Size of image after resizing
    
    Returns
    -------
    resized     : numpy array -> with dtype uint8 and shape (W, H, 3)
        Resized and squared image
    """
    BLACK = [0,0,0]
    h = image.shape[0]
    w = image.shape[1]
    if w < h:               # add border at right
        border = h - w
        image= cv2.copyMakeBorder(image,0,0,border,0,
                                cv2.BORDER_CONSTANT,value=BLACK)
    else:
        border = w - h      # add border at top
        image= cv2.copyMakeBorder(image,border,0,0,0,
                                cv2.BORDER_CONSTANT,value=BLACK)
    resized = cv2.resize(image, (size,size), 
                        interpolation = cv2.INTER_CUBIC)    
    return resized


def get_result(model, list_x):
    """
    Get prediction from model

    Parameters
    ----------
    model           : Keras Model instance
        Model to be used to make prediction
    list_x           : list
        List of aligned face
    
    Returns
    -------
    gender_predicted     : numpy array
        Gender prediction, encode 0=Female 1=Male
    age_predicted        : numpy array
        Age prediction in range [0, 100]
    """
    list_x = model.prepImg(list_x)
    predictions = model.predict(list_x)
    return model.decodePrediction(predictions)

def get_metrics(age_predicted, gender_predicted, age_true, gender_true):
    """
    Calculate the score for age and gender prediction

    Parameters
    ----------
    age_predicted       : numpy array
        Age prediction's result
    gender_predicted    : numpy array
        Gender prediction's result
    
    """
    gender_acc = (gender_predicted == gender_true).sum() / len(gender_predicted)
    age_mae = abs(age_predicted - age_true).sum() / len(age_predicted)
    
    return age_mae, gender_acc

def visualize(fullimage, result):
    pass
def getPosFromRect(rect):
	return (rect.left(), rect.top(), rect.right(), rect.bottom())
def temp():
    print('[LOAD MODEL]')
    model = SSRNet(64, [3, 3, 3], 1.0, 1.0)
    print('[LOAD WEIGHT]')
    model.setWeight('trainweight/agender_ssrnet/model.31-7.5452-0.8600-7.4051.h5')
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
    image = cv2.imread('faces/nodeflux.png')
    rects = detector(image, 1)
    print('Faces =', len(rects))

    print('[DETECT FACE]')
    shapes = dlib.full_object_detections()
    for rect in rects:
        shapes.append(predictor(image, rect))
    
    print('[ALIGN]')
    faces = dlib.get_face_chips(image, shapes, size=64, padding=0.4)
    faces = np.array(faces)
    
    print('[PREDICT]')
    # genders, ages = get_result(model, faces)
    faces = faces.astype('float16')
    result = model.predict(faces)
    print(result)
    genders = np.round(result[0]).astype('int')
    ages = result[1]
    genders = np.where(genders == 0, 'F', 'M')

    print('[VIZ]')
    for (i, rect) in enumerate(rects):
        (left, top, right, bottom) = getPosFromRect(rect)
        
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, "{:.0f}, {}".format(ages[i], genders[i]), (left - 10, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imwrite('result.jpg', image)

def main():
    logger.info('Load InceptionV3 model')
    inceptionv3 = AgenderNetInceptionV3()
    inceptionv3.setWeight('trainweight/inceptionv3_2/model.16-3.7887-0.9004-6.6744.h5')
    
    logger.info('Load SSRNet model')
    ssrnet = SSRNet(64, [3, 3, 3], 1.0, 1.0)
    ssrnet.setWeight('trainweight/agender_ssrnet/model.31-7.5452-0.8600-7.4051.h5')

    logger.info('Load pretrain imdb model')
    imdb_model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
    imdb_model.load_weights("tes_ssrnet/imdb_age_ssrnet_3_3_3_64_1.0_1.0.h5")
    
    imdb_model_gender = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
    imdb_model_gender.load_weights("tes_ssrnet/imdb_gender_ssrnet_3_3_3_64_1.0_1.0.h5")

    logger.info('Load pretrain wiki model')
    wiki_model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
    wiki_model.load_weights("tes_ssrnet/wiki_age_ssrnet_3_3_3_64_1.0_1.0.h5")
    
    wiki_model_gender = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
    wiki_model_gender.load_weights("tes_ssrnet/wiki_gender_ssrnet_3_3_3_64_1.0_1.0.h5")

    logger.info('Load pretrain wiki model')
    morph_model = SSR_net(64, [3, 3, 3], 1.0, 1.0)()
    morph_model.load_weights("tes_ssrnet/morph_age_ssrnet_3_3_3_64_1.0_1.0.h5")
    
    morph_model_gender = SSR_net_general(64, [3, 3, 3], 1.0, 1.0)()
    morph_model_gender.load_weights("tes_ssrnet/morph_gender_ssrnet_3_3_3_64_1.0_1.0.h5")

    data = pd.read_csv('dataset/UTKface.csv')

    paths = data['full_path'].values

    logger.info('Read all images')
    images = [cv2.imread(path) for path in tqdm(paths)]

    logger.info('Align face')
    images = [get_one_aligned_face(image) for image in tqdm(images)]
    
    X = np.array(images, dtype='float16')

    pred_age = dict()
    pred_gender = dict()

    logger.info('Predict with InceptionV3')
    start = time.time()
    pred_gender['inceptionv3'], pred_age['inceptionv3'] = get_result(inceptionv3, X)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))

    del X
    logger.info('Resize image to 64 for SSR-Net')
    images = [cv2.resize(image, (64, 64), interpolation = cv2.INTER_CUBIC) for image in tqdm(images)]
    X = np.array(images, dtype='float16')

    logger.info('Predict with SSR-Net')
    start = time.time()
    pred_gender['ssrnet'], pred_age['ssrnet'] = get_result(ssrnet, X)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))

    logger.info('Predict with IMDB_SSR-Net')
    start = time.time()
    pred_gender['ssrnet-imdb'] = imdb_model_gender.predict(X)
    pred_age['ssrnet-imdb'] = imdb_model.predict(X)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))

    logger.info('Predict with Wiki_SSR-Net')
    start = time.time()
    pred_gender['ssrnet-wiki'] = wiki_model_gender.predict(X)
    pred_age['ssrnet-wiki'] = wiki_model.predict(X)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))

    logger.info('Predict with Morph_SSR-Net')
    start = time.time()
    pred_gender['ssrnet-morph'] = morph_model_gender.predict(X)
    pred_age['ssrnet-morph'] = morph_model.predict(X)
    elapsed = time.time() - start
    logger.info('Time elapsed {:.2f} sec'.format(elapsed))

    pred_age = pd.DataFrame.from_dict(pred_age)
    pred_gender = pd.DataFrame.from_dict(pred_gender)

    pred_age = pd.concat([data['age'], pred_age], axis=1)
    pred_gender = pd.concat([data['gender'], pred_gender], axis=1)

    pred_age.to_csv('result/age_prediction.csv', index=False)
    pred_gender.to_csv('result/gender_prediction.csv', index=False)

if __name__ == '__main__':
    logging.basicConfig(level=logger.info)
    logger = logging.getLogger(__name__)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.tensorflow_backend.set_session(sess)
    start = time.time()
    main()
    stop = time.time()
    print('Time taken (sec) :', stop-start)