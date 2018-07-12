from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.utils.vis_utils import plot_model


class AgenderNetVGG16(Model):
	def __init__(self):
		base = VGG16(
			input_shape=(140,140,3), 
			include_top=False, 
			weights='weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
		topLayer = Flatten()(base.output)
		topLayer = Dense(4096, activation='relu')(topLayer)
		topLayer = Dropout(0.5)(topLayer)
		topLayer = Dense(4096, activation='relu')(topLayer)
		topLayer = Dropout(0.5)(topLayer)
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(10, activation='softmax', name='age_prediction')(topLayer)
		super(AgenderNetVGG16, self).__init__(inputs=base.input, outputs=[genderLayer, ageLayer], name='AgenderNetVGG16')

	def prepPhase1(self):
		for layer in self.layers[:19]:
			layer.trainable = False
	
	def prepPhase2(self):
		for layer in self.layers[11:]:
			layer.trainable = True
	
	@staticmethod
	def prepImg(data):
		data = data.astype('float32')
		data /= 127.5
		data -= 1.
		return data

class AgenderNetInceptionV3(Model):
	def __init__(self):
		base = InceptionV3(
			input_shape=(140,140,3), 
			include_top=False, 
			weights='weight/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')
		topLayer = GlobalAveragePooling2D(name='avg_pool')(base.output)
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(10, activation='softmax', name='age_prediction')(topLayer)
		super(AgenderNetInceptionV3, self).__init__(inputs=base.input, outputs=[genderLayer, ageLayer], name='AgenderNetInceptionV3')
	
	def prepPhase1(self):
		for layer in self.layers[:311]:
			layer.trainable = False
	
	def prepPhase2(self):
		for layer in self.layers[280:]:
			layer.trainable = True

	@staticmethod
	def prepImg(data):
		data = data.astype('float32')
		data /= 255.
		data -= 0.5
		data *= 2.
		return data

class AgenderNetXception(Model):
	def __init__(self):
		base = Xception(
			input_shape=(140,140,3), 
			include_top=False, 
			weights='weight/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')
		topLayer = GlobalAveragePooling2D(name='avg_pool')(base.output)
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(10, activation='softmax', name='age_prediction')(topLayer)
		super(AgenderNetXception, self).__init__(inputs=base.input, outputs=[genderLayer, ageLayer], name='AgenderNetXception')

	def prepPhase1(self):
		for layer in self.layers[:132]:
			layer.trainable = False

	def prepPhase2(self):
		for layer in self.layers[:115]:
			layer.trainable = True
	
	@staticmethod
	def prepImg(data):
		data = data.astype('float32')
		data /= 255.
		data -= 0.5
		data *= 2.
		return data

if __name__ == '__main__':
	model = AgenderNetXception()
	print(model.summary())
	for (i, layer) in enumerate(model.layers):
		print(i, layer.name, layer.trainable)
	# model.prepPhase1()
	# for (i, layer) in enumerate(model.layers):
	# 	print(i, layer.name, layer.trainable)
	# model.prepPhase2()
	# for (i, layer) in enumerate(model.layers):
	# 	print(i, layer.name, layer.trainable)
	
	plot_model(model, to_file='AgenderNetXception.png')