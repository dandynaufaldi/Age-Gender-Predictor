from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense
from keras.models import Model
from keras.utils.vis_utils import plot_model

class AgenderNetVGG16:
	@staticmethod
	def build():
		vgg16 = VGG16(weights='weight/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(vgg16.layers[-2].output)
		ageLayer = Dense(101, activation='softmax', name='age_prediction')(vgg16.layers[-2].output)
		model = Model(input=vgg16.input, output=[genderLayer, ageLayer], name='AgenderNetVGG16')
		return model

if __name__ == '__main__':
	model = AgenderNetVGG16().build()
	print(model.summary())
	plot_model(model, to_file='vgg.png')