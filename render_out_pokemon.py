from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import tensorflow

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from numpy.random import default_rng

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import os

start_string = "Spr-3r-"
data_ex = ".npy"

pokemon_train = []
for i in range(386):
    #the + 1 is because we dont have a Spr_1b_000.png
    image_string = "./things/" + start_string + str(i + 1).zfill(3) + data_ex
    pokemon_train.append(np.load(image_string))

pokemon_train = np.array(pokemon_train)
pokemon_train = np.reshape(pokemon_train, (len(pokemon_train), 64 * 64 * 3))



# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0 
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

original_dim = 64 * 64 * 3
# network parameters
input_shape = (original_dim, )
intermediate_dim = 512 * 5
batch_size = 32
latent_dim = 2
epochs = 50000

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation='relu')(inputs)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation='relu')(latent_inputs)
outputs = Dense(original_dim, activation='sigmoid')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-r",
                        "--ran",
                        help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    reconstruction_loss = mse(inputs, outputs)

    reconstruction_loss *= original_dim
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)
    vae.load_weights('./me')

    decoded = []
    file_path = ""
    if args.ran:
        d_input = []
        for i in range(0, 386):
            mu = (random.random() * 5.0 ) - 5.0
            stdev = random.random()
            d_input.append([mu, stdev])
        decoded = decoder.predict(np.array(d_input))
        file_path = "random_images"
    else:
        decoded = vae.predict(pokemon_train)
        file_path = "verfication"

    encoded = encoder.predict(pokemon_train)
    delta_f = 0.01
    axcolor = 'lightgoldenrodyellow'


    plt.subplots_adjust(left = 0.25, bottom=0.25)
    r, c = 1, 1
    gen_imgs = decoded#0.5 * d + 0.5
    fig, axs = plt.subplots(r, c)

    mean = encoded[0][0][0]
    std = encoded[0][1][0]

    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_d = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    half_mean = mean / 2.0
    test = mean * 20.0 + half_mean * 20.0
    foo = Slider(axfreq, 'Freq', -5, 5, valinit=mean, valstep=0.001)
    print(std)
    std_slid = Slider(ax_d, 'sdf', -1, 1, valinit=std, valstep=0.00001)

    pass_forward = np.array([[mean, std]])
    output = decoder.predict(pass_forward)
    
    def update(val):
        pass_forward = np.array([[foo.val, std_slid.val]])
        output = decoder.predict(pass_forward)
        axs.imshow(output.reshape((64, 64, 3)), cmap='rainbow')
        fig.canvas.draw_idle()
    foo.on_changed(update)
    std_slid.on_changed(update)

    axs.imshow(output.reshape((64, 64, 3)), cmap='rainbow')
    axs.axis('off')
    plt.show()
