'''Example of VAE on MNIST dataset using MLP

The VAE has a modular design. The encoder, decoder and VAE
are 3 models that share weights. After training the VAE model,
the encoder can be used to  generate latent vectors.
The decoder can be used to generate MNIST digits by sampling the
latent vector from a Gaussian distribution with mean=0 and std=1.

# Reference

[1] Kingma, Diederik P., and Max Welling.
"Auto-encoding variational bayes."
https://arxiv.org/abs/1312.6114
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import layers
from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os

from cheauder_utils import devectorize_smiles, load_data_vae

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps

# dataset
#X, y, charset, chars = cheauder_utils.load_data_vae('../data/250k_rndm_zinc_drugs_clean_3tiny.csv', charset_filename='model/charset_ZINC.json',col_smiles=0, col_target=1, delimiter=',', max_len=120)
X, y, charset, chars = load_data_vae('../data/250k_rndm_zinc_drugs_clean_3tiny.csv', col_smiles=0, col_target=1, delimiter=',', max_len=120)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

compound_length = x_train.shape[1]
charset_length = x_train.shape[2]
input_shape = x_train.shape[1:]
intermediate_dim = 512
batch_size = 1512
latent_dim = 196
epochs = 70
dropout_rate_mid = 0.082832929704794792
# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
#x = Dense(intermediate_dim, activation='relu')(inputs)
x = inputs
for a,b in [[9,9], [9,9], [10,11]]:
    x = layers.Conv1D(a, b, activation='tanh')(x)
    x = layers.BatchNormalization(axis=-1)(x)
x = layers.Flatten()(x)
x = layers.Dense(latent_dim, activation='relu')(x)
x = layers.Dropout(dropout_rate_mid)(x)
x = layers.BatchNormalization(axis=-1)(x)

z_mean = Dense(latent_dim, name='z_mean')(x)
#z_log_var = Dense(latent_dim, name='z_log_var')(x)
z_log_var = x
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    #return K.in_train_phase(z_mean + K.exp(0.5 * z_log_var) * epsilon, z_mean)
    return K.in_train_phase(z_mean + K.exp(0.5 * z_log_var) * epsilon, z_mean)

z = Lambda(sampling, output_shape=(latent_dim,),name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
#encoder.summary()

# build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
#x = latent_inputs
x = layers.Dense(latent_dim, activation='relu')(latent_inputs)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.Dropout(dropout_rate_mid)(x)


x = layers.RepeatVector(compound_length)(x) ###### params['MAX_LEN']
x = layers.GRU(488, activation='tanh', return_sequences=True)(x)
x = layers.GRU(488, activation='tanh', return_sequences=True)(x)
x = layers.GRU(488, activation='tanh', return_sequences=True)(x)
x = layers.GRU(charset_length, return_sequences=True, activation='softmax')(x) #### params['NCHARS']

outputs = x

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
#decoder.summary()

# instantiate VAE model
z_decoded = decoder(z)
#encoder = Model(inputs, outputs, name='vae_mlp')

class CustomVariationalLayer(layers.Layer):
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = binary_crossentropy(K.flatten(x), K.flatten(z_decoded)) * compound_length * charset_length
        kl_loss = -.5 * K.mean(
        1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs, **kwargs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x
y = CustomVariationalLayer()([inputs, z_decoded])


vae = Model(inputs, y, name='vae_mlp')


#plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
#plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)
#plot_model(encoder, to_file='vae_mlp.png', show_shapes=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    vae.compile(optimizer='adam', loss=None)


    vae.summary()

    if args.weights:
        print("using weights")
        vae.load_weights(args.weights)

        model_json = decoder.to_json()
        with open("decoder.json", "w") as json_file:
            json_file.write(model_json)

        model_json = encoder.to_json()
        with open("encoder.json", "w") as json_file:
            json_file.write(model_json)
        exit()
    else:
        # train the autoencoder
        checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss',
            verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test,None),
                callbacks=[checkpoint])

        vae.save_weights('./weights-2p.h5')

    z_mean, _, z = encoder.predict(x_test, batch_size=batch_size)
    x_test_pred_cont = decoder.predict(z_mean, batch_size=batch_size)
    x_test_pred = np.array([(x.argmax(axis=1)[:, None] == np.arange(x.shape[1])).astype(int) for x in x_test_pred_cont])

    correct = np.sum([np.array_equal(x,y) for x, y in zip(x_test, x_test_pred)])
    print(correct)
    print(correct/len(x_test))
    print(devectorize_smiles(x_test[-10:], chars))
    print(devectorize_smiles(x_test_pred[-10:], chars))

    #plot_results(models, data, batch_size=batch_size, model_name="vae_mlp")

