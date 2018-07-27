from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D, Input, Conv2D
from keras.layers import Activation, Multiply, Lambda, AveragePooling2D, MaxPooling2D, BatchNormalization
from keras.models import Model
from keras.utils import plot_model
from keras import backend as K
import cv2
import numpy as np
class AgenderNetVGG16(Model):
	def __init__(self):
		base = VGG16(
			input_shape=(140,140,3), 
			include_top=False, 
			weights='weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
		topLayer = Flatten()(base.output)
		topLayer = Dense(4096, activation='relu')(topLayer)
		# topLayer = Dropout(0.5)(topLayer)
		topLayer = Dense(4096, activation='relu')(topLayer)
		# topLayer = Dropout(0.5)(topLayer)
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(101, activation='softmax', name='age_prediction')(topLayer)
		super().__init__(inputs=base.input, outputs=[genderLayer, ageLayer], name='AgenderNetVGG16')

	def prepPhase1(self):
		for layer in self.layers[:19]:
			layer.trainable = False

	def prepPhase2(self):
		for layer in self.layers[11:]:
			layer.trainable = True

	def setWeight(self, path):
		self.load_weights(path)

	def decodePrediction(self, prediction):
		gender_predicted = np.argmax(prediction[0], axis=1)
		age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
		return gender_predicted, age_predicted

	@staticmethod
	def prepImg(data):
		data = data.astype('float16')
		data = data[..., ::-1]
		mean = [103.939, 116.779, 123.68]
		data[..., 0] -= mean[0]
		data[..., 1] -= mean[1]
		data[..., 2] -= mean[2]
		return data

class AgenderNetInceptionV3(Model):
	def __init__(self):
		base = InceptionV3(
			input_shape=(140,140,3), 
			include_top=False, 
			weights='weight/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
		topLayer = GlobalAveragePooling2D(name='avg_pool')(base.output)
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(101, activation='softmax', name='age_prediction')(topLayer)
		super().__init__(inputs=base.input, outputs=[genderLayer, ageLayer], name='AgenderNetInceptionV3')
	
	def prepPhase1(self):
		for layer in self.layers[:311]:
			layer.trainable = False
	
	def prepPhase2(self):
		for layer in self.layers[165:]:
			layer.trainable = True

	def setWeight(self, path):
		self.load_weights(path)
	
	def decodePrediction(self, prediction):
		gender_predicted = np.argmax(prediction[0], axis=1)
		age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
		return gender_predicted, age_predicted

	@staticmethod
	def prepImg(data):
		data = data.astype('float16')
		data /= 127.5
		data -= 1.
		return data

class AgenderNetXception(Model):
	def __init__(self):
		base = Xception(
			input_shape=(140,140,3), 
			include_top=False, 
			weights='weight/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
		topLayer = GlobalAveragePooling2D(name='avg_pool')(base.output)
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(101, activation='softmax', name='age_prediction')(topLayer)
		super().__init__(inputs=base.input, outputs=[genderLayer, ageLayer], name='AgenderNetXception')

	def prepPhase1(self):
		for layer in self.layers[:132]:
			layer.trainable = False

	def prepPhase2(self):
		for layer in self.layers[76:]:
			layer.trainable = True
	
	def setWeight(self, path):
		self.load_weights(path)

	def decodePrediction(self, prediction):
		gender_predicted = np.argmax(prediction[0], axis=1)
		age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
		return gender_predicted, age_predicted

	@staticmethod
	def prepImg(data):
		data = data.astype('float16')
		data /= 127.5
		data -= 1.
		return data

class AgenderNetMobileNetV2(Model):
	def __init__(self):
		base = MobileNetV2(
			input_shape=(96,96,3), 
			include_top=False, 
			weights='weight/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5')
		topLayer = GlobalAveragePooling2D()(base.output)
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(101, activation='softmax', name='age_prediction')(topLayer)
		super().__init__(inputs=base.input, outputs=[genderLayer, ageLayer], name='AgenderNetMobileNetV2')

	
	def prepPhase1(self):
		for layer in self.layers[:132]:
			layer.trainable = False

	def prepPhase2(self):
		for layer in self.layers[78:]:
			layer.trainable = True
	
	def setWeight(self, path):
		self.load_weights(path)
	
	def decodePrediction(self, prediction):
		gender_predicted = np.argmax(prediction[0], axis=1)
		age_predicted = prediction[1].dot(np.arange(0, 101).reshape(101, 1)).flatten()
		return gender_predicted, age_predicted

	@staticmethod
	def prepImg(data):
		data = [cv2.resize(image, (96,96), interpolation = cv2.INTER_CUBIC) for image in data]
		data = np.array(data)
		data = data.astype('float16')
		data /= 128.
		data -= 1.
		return data

class SSRNet(Model):
	def __init__(self, image_size,stage_num,lambda_local,lambda_d):
		
		if K.image_dim_ordering() == "th":
			self._channel_axis = 1
			self._input_shape = (3, image_size, image_size)
		else:
			self._channel_axis = -1
			self._input_shape = (image_size, image_size, 3)

		self.stage_num = stage_num
		self.lambda_local = lambda_local
		self.lambda_d = lambda_d

		self.x_layer1 = None
		self.x_layer2 = None
		self.x_layer3 = None
		self.x = None

		self.s_layer1 = None
		self.s_layer2 = None
		self.s_layer3 = None
		self.s = None
		
		inputs = Input(shape=self._input_shape)
		self.extraction_block(inputs)

		pred_gender = self.classifier_block(1, 'gender')
		pred_age = self.classifier_block(101, 'age')

		super().__init__(inputs=inputs, outputs=[pred_gender, pred_age], name='SSR_Net')

	def extraction_block(self, inputs):
		x = Conv2D(32,(3,3))(inputs)
		x = BatchNormalization(axis=self._channel_axis)(x)
		x = Activation('relu')(x)
		self.x_layer1 = AveragePooling2D(2,2)(x)
		x = Conv2D(32,(3,3))(self.x_layer1)
		x = BatchNormalization(axis=self._channel_axis)(x)
		x = Activation('relu')(x)
		self.x_layer2 = AveragePooling2D(2,2)(x)
		x = Conv2D(32,(3,3))(self.x_layer2)
		x = BatchNormalization(axis=self._channel_axis)(x)
		x = Activation('relu')(x)
		self.x_layer3 = AveragePooling2D(2,2)(x)
		x = Conv2D(32,(3,3))(self.x_layer3)
		x = BatchNormalization(axis=self._channel_axis)(x)
		self.x = Activation('relu')(x)
		#-------------------------------------------------------------------------------------------------------------------------
		s = Conv2D(16,(3,3))(inputs)
		s = BatchNormalization(axis=self._channel_axis)(s)
		s = Activation('tanh')(s)
		self.s_layer1 = MaxPooling2D(2,2)(s)
		s = Conv2D(16,(3,3))(self.s_layer1)
		s = BatchNormalization(axis=self._channel_axis)(s)
		s = Activation('tanh')(s)
		self.s_layer2 = MaxPooling2D(2,2)(s)
		s = Conv2D(16,(3,3))(self.s_layer2)
		s = BatchNormalization(axis=self._channel_axis)(s)
		s = Activation('tanh')(s)
		self.s_layer3 = MaxPooling2D(2,2)(s)
		s = Conv2D(16,(3,3))(self.s_layer3)
		s = BatchNormalization(axis=self._channel_axis)(s)
		self.s = Activation('tanh')(s)

	def classifier_block(self, V, name):
		s_layer4 = Conv2D(10,(1,1),activation='relu')(self.s)
		s_layer4 = Flatten()(s_layer4)
		s_layer4_mix = Dropout(0.2)(s_layer4)
		s_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(s_layer4_mix)
		
		x_layer4 = Conv2D(10,(1,1),activation='relu')(self.x)
		x_layer4 = Flatten()(x_layer4)
		x_layer4_mix = Dropout(0.2)(x_layer4)
		x_layer4_mix = Dense(units=self.stage_num[0], activation="relu")(x_layer4_mix)
		
		feat_s1_pre = Multiply()([s_layer4,x_layer4])
		delta_s1 = Dense(1,activation='tanh',name=name+'_delta_s1')(feat_s1_pre)
		
		feat_s1 = Multiply()([s_layer4_mix,x_layer4_mix])
		feat_s1 = Dense(2*self.stage_num[0],activation='relu')(feat_s1)
		pred_s1 = Dense(units=self.stage_num[0], activation="relu",name=name+'_pred_stage1')(feat_s1)
		local_s1 = Dense(units=self.stage_num[0], activation='tanh', name=name+'_local_delta_stage1')(feat_s1)
		#-------------------------------------------------------------------------------------------------------------------------
		s_layer2 = Conv2D(10,(1,1),activation='relu')(self.s_layer2)
		s_layer2 = MaxPooling2D(4,4)(s_layer2)
		s_layer2 = Flatten()(s_layer2)
		s_layer2_mix = Dropout(0.2)(s_layer2)
		s_layer2_mix = Dense(self.stage_num[1],activation='relu')(s_layer2_mix)
		
		x_layer2 = Conv2D(10,(1,1),activation='relu')(self.x_layer2)
		x_layer2 = AveragePooling2D(4,4)(x_layer2)
		x_layer2 = Flatten()(x_layer2)
		x_layer2_mix = Dropout(0.2)(x_layer2)
		x_layer2_mix = Dense(self.stage_num[1],activation='relu')(x_layer2_mix)
		
		feat_s2_pre = Multiply()([s_layer2,x_layer2])
		delta_s2 = Dense(1,activation='tanh',name=name+'_delta_s2')(feat_s2_pre)
		
		feat_s2 = Multiply()([s_layer2_mix,x_layer2_mix])
		feat_s2 = Dense(2*self.stage_num[1],activation='relu')(feat_s2)
		pred_s2 = Dense(units=self.stage_num[1], activation="relu",name=name+'_pred_stage2')(feat_s2)
		local_s2 = Dense(units=self.stage_num[1], activation='tanh', name=name+'_local_delta_stage2')(feat_s2)
		#-------------------------------------------------------------------------------------------------------------------------
		s_layer1 = Conv2D(10,(1,1),activation='relu')(self.s_layer1)
		s_layer1 = MaxPooling2D(8,8)(s_layer1)
		s_layer1 = Flatten()(s_layer1)
		s_layer1_mix = Dropout(0.2)(s_layer1)
		s_layer1_mix = Dense(self.stage_num[2],activation='relu')(s_layer1_mix)
		
		x_layer1 = Conv2D(10,(1,1),activation='relu')(self.x_layer1)
		x_layer1 = AveragePooling2D(8,8)(x_layer1)
		x_layer1 = Flatten()(x_layer1)
		x_layer1_mix = Dropout(0.2)(x_layer1)
		x_layer1_mix = Dense(self.stage_num[2],activation='relu')(x_layer1_mix)

		feat_s3_pre = Multiply()([s_layer1,x_layer1])
		delta_s3 = Dense(1,activation='tanh',name=name+'_delta_s3')(feat_s3_pre)
		
		feat_s3 = Multiply()([s_layer1_mix,x_layer1_mix])
		feat_s3 = Dense(2*self.stage_num[2],activation='relu')(feat_s3)
		pred_s3 = Dense(units=self.stage_num[2], activation="relu",name=name+'_pred_stage3')(feat_s3)
		local_s3 = Dense(units=self.stage_num[2], activation='tanh', name=name+'_local_delta_stage3')(feat_s3)
		#-------------------------------------------------------------------------------------------------------------------------
		
		def SSR_module(x,s1,s2,s3,lambda_local,lambda_d, V):
			a = x[0][:,0]*0
			b = x[0][:,0]*0
			c = x[0][:,0]*0

			for i in range(0,s1):
				a = a+(i+lambda_local*x[6][:,i])*x[0][:,i]
			a = K.expand_dims(a,-1)
			a = a/(s1*(1+lambda_d*x[3]))

			for j in range(0,s2):
				b = b+(j+lambda_local*x[7][:,j])*x[1][:,j]
			b = K.expand_dims(b,-1)
			b = b/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))

			for k in range(0,s3):
				c = c+(k+lambda_local*x[8][:,k])*x[2][:,k]
			c = K.expand_dims(c,-1)
			c = c/(s1*(1+lambda_d*x[3]))/(s2*(1+lambda_d*x[4]))/(s3*(1+lambda_d*x[5]))


			out = (a+b+c)*V
			return out
		
		pred = Lambda(SSR_module, 
						arguments={'s1':self.stage_num[0],
								   's2':self.stage_num[1],
								   's3':self.stage_num[2],
								   'lambda_local':self.lambda_local,
								   'lambda_d':self.lambda_d,
								   'V':V},
								   name=name+'_prediction')([pred_s1,pred_s2,pred_s3,delta_s1,delta_s2,delta_s3, local_s1, local_s2, local_s3])
		return pred
	
	def prepPhase1(self):
		pass

	def prepPhase2(self):
		pass
	
	def setWeight(self, path):
		self.load_weights(path)

	def decodePrediction(self, prediction):
		gender_predicted = np.around(prediction[0]).astype('int').squeeze()
		age_predicted = prediction[1].squeeze()
		return gender_predicted, age_predicted

	@staticmethod
	def prepImg(data):
		data = data.astype('float16')
		return data

if __name__ == '__main__':
	stage_num = [3,3,3]
	lambda_local = 1.0
	lambda_d = 1.0
	# model = SSRNet(64, stage_num, lambda_local, lambda_d)
	model = AgenderNetMobileNetV2()
	print(model.summary())
	for (i, layer) in enumerate(model.layers):
	    print(i, layer.name, layer.trainable)
	# model.prepPhase1()
	# for (i, layer) in enumerate(model.layers):
	# 	print(i, layer.name, layer.trainable)
	# model.prepPhase2()
	# for (i, layer) in enumerate(model.layers):
	# 	print(i, layer.name, layer.trainable)

	plot_model(model, to_file='MobileNetV2.png')