# Starter code for Homework 3 of Representation Learning course @ USC.
# The course is delivered to you by the Information Sciences Institute (ISI).

# Author: Yuzhong Huang
# Email: yuzhongh@usc.edu

import os
import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import pdb
import argparse
import tensorflow as tf

from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K



# Feel free to change this, so long it is an sklearn class that implements fit()
# and predict()
TASK_2A_AUTOGRADER_CLASSIFIER = sklearn.linear_model.LogisticRegression

def plot_results(vae, x_test):
	"""Plots labels and MNIST digits as a function of the 2D latent vector

	# Arguments
		models (tuple): encoder and decoder models
		data (tuple): test data and label
		batch_size (int): prediction batch size
		model_name (string): which model is using this function
	"""

	x_test = np.expand_dims(x_test, -1)
	x_re = vae.predict(x_test, batch_size=len(x_test))

	'''
	fig, axs = plt.subplots(len(x_test), 2)
	for i in range(len(x_test)):
		pdb.set_trace()
		axs[i, 0].imshow(np.clip(x_test[i] * 255, 0, 255).reshape(64, 64).astype(np.uint8))
		axs[i, 1].imshow(np.clip(x_re[i] * 255, 0, 255).reshape(64, 64).astype(np.uint8))
	plt.show()
	'''
	for i in range(len(x_test)):
		fig, axs = plt.subplots(1, 2)
		axs[0].imshow(np.clip(np.around(x_test[i] * 255), 0, 255).reshape(64, 64).astype(np.uint8), cmap='Greys_r')
		axs[1].imshow(np.clip(np.around(x_re[i] * 255), 0, 255).reshape(64, 64).astype(np.uint8), cmap='Greys_r')
		plt.show()
	pdb.set_trace()
	print('123')

class VAEdSprite(object):
	def __init__(self, args):
		self.args = args
		# VAE model = encoder + decoder

		# build encoder model
		image_size = 64
		original_dim = image_size * image_size

		inputs = Input(shape=(image_size, image_size, 1), name='encoder_input')
		x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(inputs)
		x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
		x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
		x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
		x = Conv2D(filters=256, kernel_size=4, strides=1, padding='valid', activation='relu', kernel_initializer='he_normal')(x)
		x = Reshape((256, ))(x)
		z_mean = Dense(args.latent_dim, name='z_mean', kernel_initializer='he_normal')(x)
		z_log_var = Dense(args.latent_dim, name='z_log_var', kernel_initializer='he_normal')(x)

		# use reparameterization trick to push the sampling out as input
		z = Lambda(self.sampling, name='z')([z_mean, z_log_var])

		# instantiate encoder model
		self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
		self.encoder.summary()
		plot_model(self.encoder, to_file='vae_cnn_encoder.pdf', show_shapes=True)

		# build decoder model
		latent_inputs = Input(shape=(args.latent_dim,), name='z_sampling')
		x = Dense(256, activation='relu', kernel_initializer='he_normal')(latent_inputs)
		x = Reshape((1, 1, 256))(x)
		x = Conv2DTranspose(filters=64, kernel_size=4, strides=1, padding='valid', activation='relu', kernel_initializer='he_normal')(x)
		x = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
		x = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
		x = Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
		x_re = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid', kernel_initializer='he_normal')(x)

		# instantiate decoder model
		self.decoder = Model(latent_inputs, x_re, name='decoder')
		self.decoder.summary()
		plot_model(self.decoder, to_file='vae_cnn_decoder.pdf', show_shapes=True)

		# build entire model
		outputs = self.decoder(self.encoder(inputs)[2])
		self.vae = Model(inputs, outputs, name='vae_cnn')

		inputs = Reshape((4096,))(inputs)
		outputs = Reshape((4096,))(outputs)
		reconstruction_loss = original_dim * binary_crossentropy(inputs, outputs)
		kl_loss = K.sum(0.5 * (K.square(z_mean) + K.exp(z_log_var) - z_log_var - 1), axis=-1)

		self.beta = K.variable(args.beta, name='beta', dtype=tf.float32)
		self.beta._trainable = False
		vae_loss = K.mean(reconstruction_loss + self.beta * kl_loss)
		self.vae.add_loss(vae_loss)

		adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, amsgrad=False)
		self.vae.compile(optimizer=adam)
		self.vae.summary()
		plot_model(self.vae, to_file='vae_cnn.pdf', show_shapes=True)

	# reparameterization trick
	@staticmethod
	def sampling(args):
		"""Reparameterization trick by sampling from an isotropic unit Gaussian.

		# Arguments
			args (tensor): mean and log of variance of Q(z|X)

		# Returns
			z (tensor): sampled latent vector
		"""
		z_mean, z_log_var = args
		return K.random_normal(shape=K.shape(z_mean), mean=z_mean, stddev=K.exp(0.5*z_log_var))

	def load(self):
		"""Restores the model parameters. Called once by grader before sample_*."""
		# TODO(student): Implement.

	def fit(self, train_images):
		"""Trains beta VAE.

		Args:
			train_images: numpy (uint8) array of shape [num images, 64, 64]. Only 2
				values are used: 0 and 1 (where 1 == "on" or "white"). To directly plot
				the image, you can visualize e.g. imshow(train_images[0] * 255).
		"""
		x_train = np.expand_dims(train_images, -1)
		history = self.vae.fit(x_train, epochs=self.args.epochs, batch_size=self.args.batch_size)
		self.vae.save(self.args.save_filename)

		pdb.set_trace()
		return history


	def sample_z_given_x(self, batch_x):
		"""Computes latent variables given an image batch.

		Args:
			batch_x: same type as `train_images` in `fit`. But the number of images
				for this function will be much smaller than of `fit`. Let N be number of
				images.

		Returns:
			Should return float32 numpy array with any number of dimensions (matrix,
			3D array, 4d, ...), as long as the first dimension is N. The latent
			variables for batch_x[i] should be on return[i].
		"""
		pass


	def reconstruct_x_given_z(self, batch_z):
		"""Should give a reconstruction of images given their z's

		Args:
			batch_z: numpy array with shape and type identical to output of
				sample_z_given_x

		Returns:
			uint8 3D array: one image
		"""
		pass



# TODO(student): You can make this class inherit class VAEdSprite.
class SSLatentVAE(object):

	def sample_z_given_x(self, batch_x):
		"""Same as above. Maybe can be removed, if inherits from VAEdSprite."""

	def get_partition_indices(self):
		"""Returns list of 5 pairs (start_idx, end_idx) with start_idx inclusive.

		The indices must be disjoint and can be used to slice the output of
		sample_z_given_x(). If output of sample_z_given_x is not flat, it will be
		flattened and the indices returned by this function will be taken on the
		flat version.
		"""
		# TODO(student): Implement.
		return [(0, 0)] * 5  # Violates due to overlaps; and not end_idx > start_idx

	def load(self):
		pass

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Beta-VAE')
	parser.add_argument('--dataset', default='dsprites.npz', type=str, help='dataset filename')
	parser.add_argument('--ids', default='ids.npz', type=str, help='id filename')
	parser.add_argument('--beta', default=5, type=float, help='weight for KL-divergence loss')
	parser.add_argument('--batch_size', default=256, type=int, help='batch size')
	parser.add_argument('--latent_dim', default=10, type=int, help='latent dim')
	parser.add_argument('--epochs', default=10, type=int, help='epochs')
	parser.add_argument('--save_filename', default='vae_cnn.h5', type=str, help='save filename')
	args = parser.parse_args()
	print(args)


	dataset = np.load(args.dataset)
	ids = np.load(args.ids)
	print('Files in dataset:', dataset.files)

	all_imgs = dataset['imgs']
	x_train = all_imgs[ids['train']]
	x_test = all_imgs[ids['test_reconstruct']]

	vae = VAEdSprite(args)
	vae.fit(x_train)

	plot_results(x_test, vae, batch_size=len(x_test), model_name="vae_cnn")

	pdb.set_trace()
	print('Done')
