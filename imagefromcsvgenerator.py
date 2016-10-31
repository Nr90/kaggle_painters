import pandas
import numpy as np
import keras.preprocessing.image as image_utils
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import np_utils


class ImageFromCSVGenerator:
	def img_from_filename(self, image_path):
		image = image_utils.load_img(image_path, target_size=(224, 224))
		image = image_utils.img_to_array(image)
		image = np.expand_dims(image, axis=0)
		image = preprocess_input(image)
		image = np.squeeze(image)
		image = image / 255
		return image

	def flow_from_csv(self, csv_path, batch_size):
		dataframe = pandas.read_csv(csv_path)
		nb_examples = dataframe.values.shape[0]
		nb_batches = nb_examples / batch_size
		while 1:
			dataframe.reindex(np.random.permutation(dataframe.index))
			artists = np_utils.to_categorical(dataframe['artist'].values)
			for batch_idx in range(0, nb_batches):
				start_idx = 0 + batch_size * batch_idx
				end_idx = batch_size + batch_size * batch_idx
				filenames = dataframe['filename'][start_idx:end_idx].values
				images = []
				for filename in filenames:
					filename = 'data/images/' + filename
					image = self.img_from_filename(filename)
					images.append(image)
				images = np.array(images)
				batch_artists = artists[start_idx:end_idx, :]
				yield (images, batch_artists)
