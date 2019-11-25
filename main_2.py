# Starter code for Homework 3 of Representation Learning course @ USC.
# The course is delivered to you by the Information Sciences Institute (ISI).

# Author: Yuzhong Huang
# Email: yuzhongh@usc.edu

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import sklearn.linear_model
import matplotlib.pyplot as plt
import pdb
import argparse
import tensorflow as tf

from tensorflow import keras
#import keras
from tensorflow.keras.layers import Lambda, Input, Dense, Conv2D, Conv2DTranspose, Reshape, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
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
		label_inputs = Input(shape=(6, ), dtype=tf.int32, name='label_input')

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
		self.encoder = Model([inputs, label_inputs], [z_mean, z_log_var, z], name='encoder')
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
		x_logits = Conv2DTranspose(filters=1, kernel_size=4, strides=2, padding='same', activation=None, kernel_initializer='he_normal')(x)
		x_re = Reshape((IMAGE_SIZE, IMAGE_SIZE))(Activation('sigmoid')(x_logits))

		# instantiate decoder model
		self.decoder = Model(latent_inputs, [x_logits, x_re], name='decoder')
		self.decoder.summary()
		plot_model(self.decoder, to_file='vae_cnn_decoder.pdf', show_shapes=True)

		# build entire model
		outputs = self.decoder(self.encoder([inputs, label_inputs])[2])
		self.vae = Model([inputs, label_inputs], outputs, name='vae_cnn')

		inputs = Reshape((4096,))(inputs)
		logits = Reshape((4096,))(outputs[0])
		reconstruct_loss = K.mean((IMAGE_SIZE * IMAGE_SIZE) * binary_crossentropy(inputs, logits, from_logits=True))
		kl_loss = K.mean(K.sum(0.5 * (K.square(z_mean) + K.exp(z_log_var) - z_log_var - 1), axis=-1))

		n_sup = K.cast(K.sum(label_inputs[:, 5]), tf.float32)
		shape_l = K.sum(sparse_categorical_crossentropy(label_inputs[:, 0], z[:, :3], from_logits=True) * K.cast(label_inputs[:, 5], tf.float32)) / n_sup
		scale_l = K.sum(sparse_categorical_crossentropy(label_inputs[:, 1], z[:, 3:9], from_logits=True) * K.cast(label_inputs[:, 5], tf.float32)) / n_sup
		ori_l = K.sum(sparse_categorical_crossentropy(label_inputs[:, 2], z[:, 9:49], from_logits=True) * K.cast(label_inputs[:, 5], tf.float32)) / n_sup
		posx_l = K.sum(sparse_categorical_crossentropy(label_inputs[:, 3], z[:, 49:81], from_logits=True) * K.cast(label_inputs[:, 5], tf.float32)) / n_sup
		posy_l = K.sum(sparse_categorical_crossentropy(label_inputs[:, 4], z[:, 81:], from_logits=True) * K.cast(label_inputs[:, 5], tf.float32)) / n_sup

		self.beta = K.variable(args.beta, name='beta', dtype=tf.float32)
		self.beta._trainable = False

		vae_loss = reconstruct_loss + self.beta * kl_loss + (shape_l + scale_l + ori_l + posx_l + posy_l) * 10
		self.vae.add_loss(vae_loss)
		self.vae.add_metric(reconstruct_loss, 'reconstruct_l')
		self.vae.add_metric(kl_loss, 'kl_l')
		self.vae.add_metric(shape_l, 'shape_l')
		self.vae.add_metric(scale_l, 'scale_l')
		self.vae.add_metric(ori_l, 'ori_l')
		self.vae.add_metric(posx_l, 'posx_l')
		self.vae.add_metric(posy_l, 'posy_l')

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
		'''
		m = keras.models.load_model('vae.model', compile=False)
		m.load_weights('vae_cnn_99.h5')

		i = 0
		for w in self.vae.trainable_weights:
			if w.shape == m.trainable_weights[i].shape:
				K.set_value(w, 	K.get_value(m.trainable_weights[i]))
				i += 1
				print(w.name)
		print('#Loaded layers', i)
		'''

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
		return self.decoder.predict(batch_z)[1]


# TODO(student): You can make this class inherit class VAEdSprite.
class SSLatentVAE(VAEdSprite):
	def __init__(self, args=None):
		if args is None:
			args = get_args()
		super(SSLatentVAE, self).__init__(args)

	def get_partition_indices(self):
		"""Returns list of 5 pairs (start_idx, end_idx) with start_idx inclusive.

		The indices must be disjoint and can be used to slice the output of
		sample_z_given_x(). If output of sample_z_given_x is not flat, it will be
		flattened and the indices returned by this function will be taken on the
		flat version.
		"""
		# TODO(student): Implement.
		return [(0, 3), (3, 9), (9, 49), (49, 81), (81, 113)]

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
		label_test = np.zeros((len(batch_x), 6)).astype(np.int32)
		return self.encoder.predict([batch_x, label_test])[2]

class PlotCallback(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs=None):
		if os.path.exists('stop.lock'):
			pdb.set_trace()
			print('Stop here')
		self.md.vae.save_weights('vae_cnn_{}.h5'.format(epoch))
		self.plot_test(self.md, self.x_test, epoch)
		self.plot_inter(self.md, self.x_inter, epoch)
		self.plot_mix(self.md, self.x_inter, epoch)

	@staticmethod
	def plot_test(md, x_test, epoch=0):
		n_test = len(x_test)
		label_test = np.zeros((n_test, 6)).astype(np.int32)
		x_re = md.vae.predict([x_test, label_test])[1]

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
		#label_inter = np.zeros((n_inter*2, 6)).astype(np.int32)
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

	@staticmethod
	def plot_mix(md, x_mix, epoch=0):
		n_mix = len(x_mix)
		label_inter = np.zeros((n_mix*2, 6)).astype(np.int32)
		z_inter = md.sample_z_given_x(x_mix.reshape(-1, IMAGE_SIZE, IMAGE_SIZE)).reshape(n_mix, 2, -1)

		fig = plt.figure(1, (3, n_mix))
		grid = ImageGrid(fig, 111, nrows_ncols=(n_mix, 3), axes_pad=0.05)
		s = md.get_partition_indices()
		slices = [s[0], (s[0][0], s[1][1]), s[2], (s[3][0], s[4][1]), s[4]]
		fname = ['shape', 'shape\nscale', 'orientation', 'pos x\npos y', 'pos y']

		for i in range(n_mix):
			PlotCallback.cell_imshow(grid[i*3], x_mix[i, 1])
			grid[i*3].set_ylabel(fname[i])

			z = z_inter[i, 0]
			z[slices[i][0]:slices[i][1]] = z_inter[i, 1][slices[i][0]:slices[i][1]]
			x = md.reconstruct_x_given_z(np.expand_dims(z, 0))[0]
			PlotCallback.cell_imshow(grid[i*3+1], x)
			PlotCallback.cell_imshow(grid[i*3+2], x_mix[i, 0])

		for i, t in enumerate(['Add', 'Mixed', 'Base']):
			grid[i].set_title(t)

		plt.savefig('mix/epoch_{}.pdf'.format(epoch))
		plt.close()

def get_args():
	parser = argparse.ArgumentParser(description='Beta-VAE')
	parser.add_argument('--dataset', default='dsprites.npz', type=str, help='dataset filename')
	parser.add_argument('--ids', default='ids.npz', type=str, help='id filename')
	parser.add_argument('--beta', default=2, type=float, help='weight for KL-divergence loss')
	parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
	parser.add_argument('--latent_dim', default=113, type=int, help='latent dim')
	parser.add_argument('--epochs', default=500, type=int, help='epochs')
	parser.add_argument('--model_filename', default='vae_cnn_413.h5', type=str, help='model filename')
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	args = get_args()
	print(args)


	dataset = np.load(args.dataset)
	ids = np.load(args.ids)
	print('Files in dataset:', dataset.files)
	print('Files in ids:', ids.files)

	all_imgs = dataset['imgs']
	x_train = all_imgs[ids['train']]
	x_test = all_imgs[ids['test_reconstruct']]
	x_inter = all_imgs[ids['test_interpolate']]

	unsup_train_classes = np.zeros((len(x_train), 6))
	sup_train_images = all_imgs[ids['supervised_train']]
	sup_train_classes = dataset['latents_classes'][ids['supervised_train']][:, 1:]
	sup_train_flags = np.ones((len(sup_train_images), 1))
	sup_train_classes = np.hstack([sup_train_classes, sup_train_flags])

	x_all = np.vstack([x_train, sup_train_images])
	label_all = np.vstack([unsup_train_classes, sup_train_classes]).astype(np.int32)

	cb = PlotCallback()
	cb.x_test = x_test
	cb.x_inter = x_inter

	md = SSLatentVAE(args)
	cb.md = md
	md.load()

	#np.random.seed(1234)
	#np.random.shuffle(x_train)
	#md.fit({'encoder_input': x_all, 'label_input': label_all}, [cb])
	#for i, s in enumerate(range(0, len(x_train), 10)):
	#	print(i, s)

	x_mix = x_train[(36, 37, 82, 83, 2884, 2885, 624, 625, 1344, 1345), :, :].reshape(5, 2, 64, 64)
	PlotCallback.plot_mix(md, x_mix, 0)

	#pdb.set_trace()
	print('Done')
