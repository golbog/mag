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
from keras.layers import Lambda, Input, Dense, Dropout, BatchNormalization
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
import pickle

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse
import os

import cheauder_utils

#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps

class VAE:
    def __init__(self, X, inter_dim=512, batch_size=1024, latent_dim=196, epochs=500, dropout_rate=0.09, gru_size=488, verbose=True, **kwargs):
        self.compound_length = X.shape[1]
        self.charset_length = X.shape[2]
        self.input_shape = X.shape[1:]
        self.intermediate_dim = inter_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.epochs = epochs
        self.dropout_rate_mid = dropout_rate
        self.gru_size = gru_size
        self.verobse = verbose

        self.encoder = None
        self.decoder = None
        self.vae = None

        self.z_log_var = None
        self.z_mean = None


    def build_encoder(self):
        """
        Build an encoder for variational autoencoder that outpouts mean and log variance needed for reparametrization.

        """
        self.inputs = Input(shape=self.input_shape, name='Vhod_v_kodirnik')
        x = self.inputs
        i = 1
        for a,b in [[9,9], [9,9], [10,11]]:
            x = layers.Conv1D(a, b, activation='tanh', name="Konvolucijski_sloj_"+str(i))(x)
            x = layers.BatchNormalization(axis=-1, name='Paketna_normalizacija_'+str(i))(x)
            i += 1
        x = layers.Flatten(name='Zdruzevalni_sloj')(x)
        x = layers.Dense(self.latent_dim, activation='tanh', name='Polno_povezan_sloj_1')(x)
        x = layers.Dropout(self.dropout_rate_mid, name="Izpustveni_sloj_1")(x)
        x = layers.BatchNormalization(axis=-1, name="Paketna_normalizacija_"+str(i))(x)

        self.z_mean = Dense(self.latent_dim, name='Srednja_vrednost')(x)

        self.z_log_var = Dense(self.latent_dim, name='Varianca')(x)


    def build_variational_layer(self):
        """
        Build a variational middle layer "transforms" mean and log variance into sampled point in latent space.
        Note: requires built encoder.
        """
        def sampling(args):
            """
            Reparameterization trick by sampling from an isotropic unit Gaussian.

            # Arguments:
                args (tensor): mean and log of variance of Q(z|X)

            # Returns:
                z (tensor): sampled latent vector
            """
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0, stddev=1)
            return K.in_train_phase(z_mean + K.exp(0.5 * z_log_var) * epsilon, z_mean)

        self.z = Lambda(sampling, output_shape=(self.latent_dim,),name='Vzorcenje')([self.z_mean, self.z_log_var])


    # build decoder model
    def build_decoder(self):
        """
        Build a decoder that transforms latent representations into decoded outputs.

        """
        self.latent_inputs = Input(shape=(self.latent_dim,), name='Vhod_v_dekodirnik')
        x = layers.Dense(self.latent_dim, activation='tanh', name="Polno_povezan_sloj_2")(self.latent_inputs)

        x = layers.Dropout(self.dropout_rate_mid, name="Izpustveni_sloj_2")(x)
        x = layers.BatchNormalization(axis=-1, name="Paketna_normalizacija_5")(x)

        x = layers.RepeatVector(self.compound_length, name="Ponovi_vektor")(x)
        x = layers.GRU(self.gru_size, activation='tanh', return_sequences=True, name="GRU_sloj_1")(x)
        x = layers.GRU(self.gru_size, activation='tanh', return_sequences=True, name="GRU_sloj_2")(x)
        x = layers.GRU(self.gru_size, activation='tanh', return_sequences=True, name="GRU_sloj_3")(x)
        x = layers.GRU(self.charset_length, return_sequences=True, activation='softmax')(x)

        self.outputs = x


    def build_model(self):
        """
        Build the variational autoencoder.
        # Returns:
                self (object): self
        """
        self.build_encoder()
        self.build_decoder()
        self.build_variational_layer()

        compound_length = self.compound_length
        charset_length = self.charset_length
        z_log_var = self.z_log_var
        z_mean = self.z_mean

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

        self.encoder = Model(self.inputs, [self.z_mean, self.z_log_var, self.z], name='encoder')
        self.decoder = Model(self.latent_inputs, self.outputs, name='decoder')

        z_decoded = self.decoder(self.z)
        y = CustomVariationalLayer()([self.inputs, z_decoded])

        self.vae = Model(self.inputs, y, name='vae_mlp')

        if self.verobse:
            self.encoder.summary()
            self.decoder.summary()
            self.vae.summary()

        return self


if __name__ == '__main__':
    lr = 0.0003919
    momentum = 0.97170
    epochs = 500
    batch_size = 200
    base = 'model/'

    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)

    X, y, charset, chars = cheauder_utils.load_data_vae('../data/250k_rndm_zinc_drugs_clean_3small.csv', col_smiles=0,
                                                     col_target=1, delimiter=',', max_len=120)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    vae_base = VAE(x_train).build_model()
    args = parser.parse_args()
    models = (vae_base.encoder, vae_base.decoder)
    vae = vae_base.vae
    data = (x_test, y_test)

    vae.compile(optimizer=Adam(lr=lr, beta_1=momentum), loss=None)
    if args.weights:
        vae.load_weights(args.weights)
    else:
        earlyStop = EarlyStopping(monitor='val_loss', min_delta=0, patience = 20)
        checkpoint = ModelCheckpoint(base+'weights.{epoch:02d}-{val_loss:.2f}.h5', monitor='val_loss', 
            verbose=1, save_best_only=True, mode='min', save_weights_only=True)
        history = vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test,None),
                callbacks=[checkpoint, earlyStop])

        cheauder_utils.save_vae(vae, base)
