from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model

class AgenderNetVGG16:
	@staticmethod
	def build():
		vgg16 = VGG16(input_shape=(224,224,3), include_top=False, weights='weight/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
		# print(vgg16.summary())

		topLayer = Flatten(input_shape=vgg16.layers[-1].output_shape[1:])(vgg16.output)
		topLayer = Dense(512, activation='relu')(topLayer)
		topLayer = Dropout(0.2)(topLayer)
		topLayer = Dense(512, activation='relu')(topLayer)
		topLayer = Dropout(0.2)(topLayer)

		genderLayer = Dense(2, activation='softmax', name='gender_prediction')(topLayer)
		ageLayer = Dense(10, activation='softmax', name='age_prediction')(topLayer)
		model = Model(input=vgg16.input, output=[genderLayer, ageLayer], name='AgenderNetVGG16')
		# print(model.summary())
		return model

if __name__ == '__main__':
	model = AgenderNetVGG16().build()
	print(model.summary())
	for (i, layer) in enumerate(model.layers):
		print(i, layer.name, layer.trainable)
	# plot_model(model, to_file='AgenderNetVGG16.png')