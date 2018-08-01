from scipy import misc
import sys
import os
import argparse
#import tensorflow as tf
import numpy as np
import mxnet as mx
import random
import cv2
import sklearn
from sklearn.decomposition import PCA
from time import sleep
from easydict import EasyDict as edict
from mtcnn_detector import MtcnnDetector
#sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'common'))
#import face_image
import face_preprocess


def do_flip(data):
  for idx in range(data.shape[0]):
    data[idx,:,:] = np.fliplr(data[idx,:,:])

def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model

'''
Get resized image
Return square-resized image
'''
def resizeImg(image, size=140):
	BLACK = [0,0,0]
	h = image.shape[0]
	w = image.shape[1]
	if w<h:
	    border = h-w
	    image= cv2.copyMakeBorder(image,0,0,border,0,cv2.BORDER_CONSTANT,value=BLACK)
	else:
	    border = w-h
	    image= cv2.copyMakeBorder(image,border,0,0,0,cv2.BORDER_CONSTANT,value=BLACK)
	resized = cv2.resize(image, (size,size), interpolation = cv2.INTER_CUBIC)    
	return resized

class FaceModel:
  def __init__(self, args):
    self.args = args
    ctx = mx.gpu(args.gpu)
    _vec = args.image_size.split(',')
    assert len(_vec)==2
    image_size = (int(_vec[0]), int(_vec[1]))
    self.model = None
    self.ga_model = None
    if len(args.model)>0:
      self.model = get_model(ctx, image_size, args.model, 'fc1')
    if len(args.ga_model)>0:
      self.ga_model = get_model(ctx, image_size, args.ga_model, 'fc1')

    self.threshold = args.threshold
    self.det_minsize = 50
    self.det_threshold = [0.6,0.7,0.8]
    #self.det_factor = 0.9
    self.image_size = image_size
    mtcnn_path = os.path.join(os.path.dirname(__file__), 'mtcnn-model')
    if args.det==0:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
    else:
      detector = MtcnnDetector(model_folder=mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
    self.detector = detector


  def get_input(self, face_img):
    ret = self.detector.detect_face(face_img, det_type = self.args.det)
    if ret is None: #return resized image
      res = resizeImg(face_img, 112)
      nimg = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      return aligned
    bbox, points = ret
    if bbox.shape[0]==0:  #return resized image
      res = resizeImg(face_img, 112)
      nimg = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
      aligned = np.transpose(nimg, (2,0,1))
      return aligned
    bbox = bbox[0,0:4]
    points = points[0,:].reshape((2,5)).T
    #print(bbox)
    #print(points)
    nimg = face_preprocess.preprocess(face_img, bbox, points, image_size='112,112')
    if nimg is None :
      nimg = resizeImg(face_img, 112)
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2,0,1))
    return aligned

  def get_feature(self, aligned):
    input_blob = np.expand_dims(aligned, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,))
    self.model.forward(db, is_train=False)
    embedding = self.model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    return embedding

  def get_ga(self, aligned, batch_size=128):
    print('Input shape', aligned.shape)
    input_blob = aligned
    data = mx.nd.array(input_blob)
    data_iter = mx.io.NDArrayIter(data, batch_size=batch_size)
    genders = []
    ages = []
    for db in data_iter :
      self.ga_model.forward(db, is_train=False)
      raw_output = self.ga_model.get_outputs()
      output = mx.nd.stack(*raw_output).asnumpy().squeeze()
      if (len(output.shape) == 1):                   #if only 1 input and 1 output
        output = np.expand_dims(output, axis=0)
      g = output[:,0:2].reshape((db.data[0].shape[0],1, 2))
      gender = np.argmax(g, axis=2).flatten()
      a = output[:,2:202].reshape( (db.data[0].shape[0],100,2) )
      a = np.argmax(a, axis=2)
      age = a.sum(axis=1)
      genders.append(gender)
      ages.append(age)
    genders = np.concatenate(genders).astype('int')[:len(aligned)]
    ages = np.concatenate(ages)[:len(aligned)]
    # print(genders.shape, ages.shape)
    return genders, ages

