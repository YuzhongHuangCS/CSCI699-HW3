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

from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from mpl_toolkits.axes_grid1 import ImageGrid

# Feel free to change this, so long it is an sklearn class that implements fit()
# and predict()
TASK_2A_AUTOGRADER_CLASSIFIER = sklearn.linear_model.LogisticRegression
IMAGE_SIZE = 64

class VAEdSprite(object):
	def __init__(self, args):
		super(VAEdSprite, self).__init__()
		self.args = args
		# VAE model = encoder + decoder

		# build encoder model
		inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE), name='encoder_input')
		x = Reshape((IMAGE_SIZE, IMAGE_SIZE, 1))(inputs)
		x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation='relu', kernel_initializer='he_normal')(x)
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
		x = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation='sigmoid', kernel_initializer='he_normal')(x)
		x_re = Reshape((IMAGE_SIZE, IMAGE_SIZE))(x)

		# instantiate decoder model
		self.decoder = Model(latent_inputs, x_re, name='decoder')
		self.decoder.summary()
		plot_model(self.decoder, to_file='vae_cnn_decoder.pdf', show_shapes=True)

		# build entire model
		outputs = self.decoder(self.encoder(inputs)[2])
		self.vae = Model(inputs, outputs, name='vae_cnn')

		inputs = Reshape((4096,))(inputs)
		outputs = Reshape((4096,))(outputs)
		reconstruction_loss = (IMAGE_SIZE * IMAGE_SIZE) * binary_crossentropy(inputs, outputs)
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
		self.vae.load_weights(self.args.model_filename)

	def fit(self, train_images, callbacks=None):
		"""Trains beta VAE.

		Args:
			train_images: numpy (uint8) array of shape [num images, 64, 64]. Only 2
				values are used: 0 and 1 (where 1 == "on" or "white"). To directly plot
				the image, you can visualize e.g. imshow(train_images[0] * 255).
		"""
		history = self.vae.fit(train_images, epochs=self.args.epochs, batch_size=self.args.batch_size, callbacks=callbacks)
		self.vae.save_weights(self.args.model_filename)
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
		return self.encoder.predict(batch_x)[2]


	def reconstruct_x_given_z(self, batch_z):
		"""Should give a reconstruction of images given their z's

		Args:
			batch_z: numpy array with shape and type identical to output of
				sample_z_given_x

		Returns:
			np.float32 3D array: one image
		"""
		return self.decoder.predict(batch_z)


# TODO(student): You can make this class inherit class VAEdSprite.
class SSLatentVAE(VAEdSprite):
	def __init__(self, args):
		super(SSLatentVAE, self).__init__(args)

	def get_partition_indices(self):
		"""Returns list of 5 pairs (start_idx, end_idx) with start_idx inclusive.

		The indices must be disjoint and can be used to slice the output of
		sample_z_given_x(). If output of sample_z_given_x is not flat, it will be
		flattened and the indices returned by this function will be taken on the
		flat version.
		"""
		# TODO(student): Implement.
		return [(0, 0)] * 5  # Violates due to overlaps; and not end_idx > start_idx

class PlotCallback(tf.keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		self.plot_test(self.md, self.x_test, epoch)
		self.plot_inter(self.md, self.x_inter, epoch)

	@staticmethod
	def plot_test(md, x_test, epoch=0):
		n_test = len(x_test)
		x_re = md.vae.predict(x_test)

		fig = plt.figure(1, (n_test, 2.))
		grid = ImageGrid(fig, 111, nrows_ncols=(2, n_test), axes_pad=0.05)

		for i in range(n_test):
			PlotCallback.cell_imshow(grid[i], x_test[i])
			PlotCallback.cell_imshow(grid[i+n_test], x_re[i])

		grid[0].set_ylabel('Input')
		grid[n_test].set_ylabel('Reconstruct')
		plt.savefig('reconstruct/epoch_{}.pdf'.format(epoch))
		plt.close()

	@staticmethod
	def cell_imshow(cell, img):
		cell.imshow(np.clip(np.around(img * 255), 0, 255).astype(np.uint8), cmap='Greys_r')
		cell.set_xticks([])
		cell.set_yticks([])

	@staticmethod
	def plot_inter(md, x_inter, epoch=0):
		n_inter = len(x_inter)
		z_inter = md.sample_z_given_x(x_inter.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)).reshape(n_inter, 2, -1)

		fig = plt.figure(1, (8, n_inter))
		grid = ImageGrid(fig, 111, nrows_ncols=(n_inter, 8), axes_pad=0.05)

		for i in range(n_inter):
			PlotCallback.cell_imshow(grid[i*8], x_inter[i, 0])
			for j, alpha in enumerate([1.0, 0.8, 0.6, 0.4, 0.2, 0.0]):
				z = alpha * z_inter[i, 0] + (1-alpha) * z_inter[i, 1]
				x = md.reconstruct_x_given_z(np.expand_dims(z, 0))[0]

				PlotCallback.cell_imshow(grid[i*8+j+1], x)
			PlotCallback.cell_imshow(grid[i*8+7], x_inter[i, 1])

		for i, t in enumerate(['a', '1.0', '0.8', '0.6', '0.4', '0.2', '0.0', 'b']):
			grid[i].set_title(t)

		plt.savefig('interpolate/epoch_{}.pdf'.format(epoch))
		plt.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Beta-VAE')
	parser.add_argument('--dataset', default='dsprites.npz', type=str, help='dataset filename')
	parser.add_argument('--ids', default='ids.npz', type=str, help='id filename')
	parser.add_argument('--beta', default=5, type=float, help='weight for KL-divergence loss')
	parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
	parser.add_argument('--latent_dim', default=10, type=int, help='latent dim')
	parser.add_argument('--epochs', default=100, type=int, help='epochs')
	parser.add_argument('--model_filename', default='vae_cnn.h5', type=str, help='model filename')
	args = parser.parse_args()
	print(args)


	dataset = np.load(args.dataset)
	ids = np.load(args.ids)
	print('Files in dataset:', dataset.files)
	print('Files in ids:', ids.files)

	all_imgs = dataset['imgs']
	x_train = all_imgs[ids['train']]
	x_test = all_imgs[ids['test_reconstruct']]
	x_inter = all_imgs[ids['test_interpolate']]

	cb = PlotCallback()
	cb.x_test = x_test
	cb.x_inter = x_inter

	md = VAEdSprite(args)
	cb.md = md
	md.fit(x_train, [cb])
	#PlotCallback.plot_test(x_test, vae, batch_size=len(x_test), model_name="vae_cnn")

	pdb.set_trace()
	print('Done')
