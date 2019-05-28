import sys
from keras.losses import mse, binary_crossentropy

import umap.umap_ as umap
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn.model_selection import LeaveOneOut
import numpy as np
import argparse
from collections import OrderedDict
from keras.models import Model, model_from_json, load_model
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from keras import layers
from keras import backend as K

import re
import csv
import json
import numpy as np

class CustomVariationalLayer(layers.Layer):
    # Placeholder - does not need to be a correct equation.
    def vae_loss(self, x, z_decoded):
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        xent_loss = binary_crossentropy(K.flatten(x), K.flatten(z_decoded))
        kl_loss = -.5 * K.mean(
        1 + z_decoded - K.square(x) - K.exp(x), axis=-1)
        return K.mean(xent_loss + kl_loss)
    def call(self, inputs, **kwargs):
        x = inputs[0]
        z_decoded = inputs[1]
        loss = self.vae_loss(x, z_decoded)
        self.add_loss(loss, inputs=inputs)
        return x

def read_data(filename, col_smiles=0, col_target=1, start_row=1, delimiter=' ', quotechar='\''):
    """
    Read the csv file with SMILES strings.
    :param filename: Name of the csv file.
    :param col_smiles: Column in which SMILES string are written in the file. Starting with 0.
    :param col_target: Column in which target class are written in the file. Starting with 0.
    :param start_row: In which row in the file is data written. Starting with 0.
    :param delimiter: Delimiter used in the file.
    :param quotechar: Quote char used in the file.
    :return X: Numpy array of SMILES strings.
    :return y: Numpy array of target classes.
    :return label: Names of the target classes.
    """
    smiles = list()
    targets = list()
    with open(filename) as file:
        reader = csv.reader(file, delimiter=delimiter, quotechar=quotechar)
        for i in range(start_row): #testiraj
            row = next(reader)
            if i == 0:
                row = np.array(row)
                label = row[col_target]
        for row in reader:
            if len(row) > 2:
                row = np.array(row)
                smile = row[col_smiles]
                smiles.append(smile.split()[0])
                targets.append(row[col_target] if col_target != -1 else 0)
    return np.array(smiles), np.array(targets), label

def load_data_vae(filename, charset_filename=None, col_smiles=0, col_target=1, start_row=1, delimiter=',', quotechar='\"', max_len=-1):
    """
    Load data intended for learning the VAE model.
    :param filename: Name of the csv file.
    :param charset_filename: (Optional) Name of the charset file.
    :param col_smiles: Column in which SMILES string are written in the file. Starting with 0.
    :param col_target: Column in which target class are written in the file. Starting with 0.
    :param start_row: In which row in the file is data written. Starting with 0.
    :param delimiter: Delimiter used in the file.
    :param quotechar: Quote char used in the file.
    :param max_len: Max length for the returned vectos.
    :return X: Numpy array of vectorized SMILES string (number of SMILES string x max len)
    :return y: Numpy array of target class.
    :return charset: Charset dictionary.
    :return chars: List of chars in same order as is used in vectors.
    """
    X, y, label = read_data(filename, col_smiles, col_target, start_row, delimiter, quotechar)
    if charset_filename is None:
        chars, charset = create_charset(X)
    else:
        chars, charset = load_charset(charset_filename)
    if max_len == -1:
        max_len = max([len(x) for x in X]) + 1
    X = vectorize_smiles(X, charset, max_len)
    return X, y, charset, chars

def load_data(filename, vae_encoder, ae_encoder, charset_filename=None, col_smiles=0, col_target=1, start_row=1, delimiter=' ', quotechar='\"', max_len=-1):
    """
    Load data intended for testing/using models.
    :param filename: Name of the csv file.
    :param vae_encoder: Pretrained encoder part of the VAE model.
    :param ae_encoder: Pretrained encoder part of the  AE model.
    :param charset_filename: (Optional) Name of the charset file.
    :param col_smiles: Column in which SMILES string are written in the file. Starting with 0.
    :param col_target: Column in which target class are written in the file. Starting with 0.
    :param start_row: In which row in the file is data written. Starting with 0.
    :param delimiter: Delimiter used in the file.
    :param quotechar: Quote char used in the file.
    :param max_len: Max length for the returned vectos.
    :return Xvae: Numpy array latent representation of input SMILES strings encoded with VAE (number of SMILES string x max len)
    :return Xae: Numpy array latent representation of input SMILES strings encoded with AE (number of SMILES string x max len)
    :return Xfinger: Numpy array of fingerprinted SMILES strings (number of SMILES string x max len)
    :return y: Numpy array of target class.
    :return label: Label of the target class.
    :return smiles: SMILES strings in the file.
    """
    smiles, y, label = read_data(filename, col_smiles=col_smiles, col_target=col_target, delimiter=delimiter, start_row=start_row, quotechar=quotechar)

    # VAE,AE
    if charset_filename is None:
        chars, charset = create_charset(smiles)
    else:
        chars, charset = load_charset(charset_filename)
    if max_len == -1:
        max_len = max([len(x) for x in smiles]) + 1
    oh_smiles = vectorize_smiles(smiles, charset, max_len)
    Xvae, _, _ = vae_encoder.predict(oh_smiles)
    Xae = ae_encoder.predict(oh_smiles)

    # fingerprints
    Xfinger, y, valid = smiles_to_fingerprints(smiles, y)

    Xvae = Xvae[smiles]
    Xae = Xae[smiles]

    return Xvae, Xae, Xfinger, y, label, smiles


def smiles_to_fingerprints(smiles, y):
    """
    Transform SMILES strings into RDKit's 4 different implementations: topolocigal, circular, substructure and AVALON.
    :param smiles: numpy array of SMILES strings.
    :param y: target class.
    :return Xfinger: Dictionary of numpy arrays of fingerprints for every fingerprint type.
    :return y: numpy array of SMILES  (only valid).
    :return valid: Indices of validly fingerprinted SMILES strings.
    """
    moles = np.array([AllChem.MolFromSmiles(x) for x in smiles])
    Xfinger = dict()

    valid = [i for i, x in enumerate(moles) if x is not None]
    moles = moles[valid]
    y = np.array(y[valid], dtype=np.float)

    # top
    fps_top = [Chem.RDKFingerprint(x) for x in moles]
    fps_top = fp_to_numpy(fps_top)
    Xfinger['topological'] = fps_top

    # circ
    n = 3
    fps_cir = [Chem.AllChem.GetMorganFingerprintAsBitVect(x, n) for x in moles]
    fps_cir = fp_to_numpy(fps_cir)
    Xfinger['circular'] = fps_cir

    # substruct
    fps_sub = [MACCSkeys.FingerprintMol(x) for x in moles]
    fps_sub = fp_to_numpy(fps_sub)
    Xfinger['substructure'] = fps_sub

    # avalon
    fps_ava = [Chem.AllChem.GetMorganFingerprintAsBitVect(x, n) for x in moles]
    fps_ava = fp_to_numpy(fps_ava)
    Xfinger['avalon'] = fps_ava

    return Xfinger, y, valid

def is_valid_smiles(a):
    """
    Returns true if the give SMILES string is valid
    """
    return AllChem.MolFromSmiles(a) is not None and a != '' and a is not None


def load_charset(charset_filename):
    """
    Read the charset file and return dictionary and list of the data.
    """
    with open(charset_filename, "r") as f:
        charset = json.load(f)
    chars = np.sort(list(charset.keys()))
    return chars, charset

def create_fingerprints(moles):
    """
    Create Morgan fingerprints (circular) for the given RDKit's moles data.
    :param moles: list of RDKit's moles (Mol).
    :return: Fingerprints for the given data.
    """
    fingerprints = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 4) for x in moles])
    X = list()
    for x in fingerprints:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(x, arr)
        X.append(arr)
    return np.array(X)

def create_charset(X):
    """
    Create charset for the given list of SMILES strings.
    :param X: list of SMILES strings.
    :return chars: List of chars.
    :return charset: Dictionary of chars (char: index).
    """
    chars = list(set(''.join(x for x in X))) # all unique chars (random sorting)
    chars.append(' ')
    chars = np.sort(chars)
    charset = {c:i for i,c in enumerate(chars)}
    return chars, charset

def pad_smiles(smiles, max_len):
    """
    Pad and cut SMILES strings to a given length (padded with whitespace).
    :param smiles: List of SMILES strings.
    :param max_len: Max length.
    :return: List of padded SMILES strings.
    """
    res = list()
    for smile in smiles:
        res.append(pad_smile(smile, max_len))
    return np.array(res)

def pad_smile(smile, max_len):
    """
    Pad a single SMILES string.
    """
    if len(smile) < max_len:
        return smile + ' ' * (max_len - len(smile))
    return smile[:max_len]

def vectorize_smiles(smiles, charset, max_len=120, canon=False):
    """
    Vectorize SMILES strings based on the given charset. Skipping
    :param smiles: List of SMILES strings.
    :param charset: Charset used to vectorize.
    :param max_len: Max length.
    :param canon: Is SMILE string canonized before vectorized.
    :return: Vectorized SMILES strings (numpy array of binary vectors).
    """
    res = list()
    for smile in smiles:
        res.append(vectorize_smile(smile, charset, max_len, canon))
    return np.array(res)

def vectorize_smile(smile, charset, max_len=120, canon=True):
    """
    Vectorize a single SMILES string.
    """
    if canon:
        smile = canon_smiles(smile)
    x = np.zeros((max_len, len(charset)))
    for i, char in enumerate(pad_smile(smile, max_len)):
        try:
            x[i,charset[char]] = 1
        except KeyError:
            continue
    return x

def fp_to_numpy(X):
    """
    Transform RDKit's fingerprint to a numpy array.
    :param X: Fingerprint.
    :return: Numpy array of the fingerprint.
    """
    Xfinger = list()
    for x in X:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(x, arr)
        Xfinger.append(arr)
    return np.array(Xfinger)

def devectorize_smiles(smiles, chars):
    """
    Transform vectorized SMILES strings back to a string format.
    :param smiles: List of vectorized SMILES strings.
    :param chars: List of used chars (have to be in the same order as used in vectorization.
    :return: List of SMILES strings.
    """
    return np.array([devectorize_smile(smile, chars) for smile in smiles])

def devectorize_smile(smile, chars):
    """
    Transform a single vectorized SMILES string back to a string format.
    """
    return ''.join([chars[char_index] for char_index in np.where(smile==1)[1]]).split()[0]

def load_coder_json(file_coder, file_weights, custom_objects):
    """
    Load encoder/decoder in json format and weights from a file.
    :param file_coder: Filename of model in json format.
    :param file_weights: Filename of weights.
    :param custom_objects: Custom objects used in the model.
    :return: Model with given weights.
    """
    with open(file_coder, "r") as json_file:
        coder_json = json_file.read()

        coder = model_from_json(coder_json,
                                  custom_objects=custom_objects)
        coder.load_weights(file_weights)
        return coder

def plot_kde(X):
    """
    Plot kernel density estimation graph for the given data.
    :param X: Numpy array (n x m).
    :return: matplotlib plot of the KDE graph.
    """
    from sklearn.neighbors import KernelDensity
    import matplotlib.pyplot as plt
    X_plot = np.linspace(X.min() - 1, X.max() + 1, 1000)
    fig, ax = plt.subplots()
    for column in range(X.shape[1]):
        kde = KernelDensity(kernel='gaussian', bandwidth=(X.max() - X.min()) / 50).fit(
            X[:, column][:, np.newaxis])
        log_dens = kde.score_samples(X_plot[:, np.newaxis])
        ax.plot(X_plot, np.exp(log_dens), '-')
    return plt

def correctly_decoded(X_decoded, X_real, chars):
    """
    Print percentage of correctly decoded characters with added noise to simulate slightly "random" output of the VAE coder.
    :param X_decoded: Data that was encoded and decoded with an autoencoder.
    :param X_real: Read data.
    :param chars: Used characters.
    """
    pred = devectorize_smiles(X_decoded, chars)
    real = devectorize_smiles(X_real, chars)

    num_correct = np.sum([np.array_equal(x, y) for x, y in zip(real, pred)])
    print("Percentage correctly predicted: {0:.2f}".format(100*num_correct/len(pred)))

    mean_error = np.sum([sum(1 for a, b in zip(x, y) if a != b) for x, y in zip(real, pred)]) / len(real)
    print("Mean error: {0:.3f}".format(mean_error))

def correctly_decoded_with_tries(X, X_real, decoder, chars, noise_norm=False, n=1000):
    """
    Print percentage of correctly decoded characters with added noise to simulate slightly "random" output of the VAE coder.
    :param X_decoded: Data that was encoded and decoded with an autoencoder.
    :param X_real: Read data.
    :param chars: Used characters.
    """
    correct = 0
    for ix, x in enumerate(X):
        for i in range(n):
            pred = decoder.predict(np.array([perturb_z(x, noise_norm=False if i == 0 else noise_norm)]))[0]
            pred = np.array((pred.argmax(axis=1)[:, None] == np.arange(pred.shape[1])).astype(int))
            pred = devectorize_smile(pred, chars)
            real = devectorize_smile(X_real[ix], chars)
            if np.array_equal(real, pred):
                correct += 1
                break
    print("Percentage correctly predicted: {0:.2f}".format(100 * correct / len(X)))

def classification_test(Xs, y, classifiers, n_splits=10, verbose=True):
    """
    Test classification accuracy for the given data.
    :param Xs: Data.
    :param y: Targer.
    :param classifiers: Dictionary of classifiers.
    :param n_splits: Number of splits
    :param verbose: Verbosity.
    :return: Dictionary with classification accuracies.
    """
    res = OrderedDict()
    cv = StratifiedKFold(n_splits=n_splits)

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                cl = c.fit(X[train], y[train].ravel()) ## Aded ravel()
                val += roc_auc_score(y[test], cl.predict(X[test])) / n_splits
            res[(Xname, cname)] = val
            if verbose:
                print("{:18s} | {:6s} : {:.3f}".format(Xname, cname, val))
    return res

def perturb_z(z, noise_norm=False):
    """
    Add noise to the point in latent space.
    """
    if noise_norm > 0.0:
        noise_vec = np.random.normal(0, 1, size=z.shape)
        noise_vec = noise_vec / np.linalg.norm(noise_vec)
        return z + (noise_norm * noise_vec)
    return z

def to_csv(filename, smiles, X, y):
    """
    Helper funtion used to save data for further use in Orange.
    """
    with open(filename, 'w') as f:
        line = '{},{},{}\n'.format('smiles', ','.join('dim{}'.format(i) for i in range(len(X[0]))), 'y')
        f.write(line)
        for i in range(len(X)):
            line = '{},{},{}\n'.format(smiles[i], ','.join('{}'.format(x) for x in X[i]), y[i])
            f.write(line)
        return True

def plot_tsne_classification(X, y, plot_title, legend_title, legend_labels, alpha=.3, markersize=4):
    y = np.array(y)

    pca = PCA(n_components=20)
    pca_res = pca.fit_transform(X)

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=500)
    res = tsne.fit_transform(pca_res)

    for yi in np.unique(y):
        plt.plot(np.array(res[:, 0])[y == yi], np.array(res[:, 1])[y == yi], 'o', label=legend_labels[yi], alpha=alpha,
                 markersize=markersize)
    plt.title(plot_title)
    plt.axis('off')
    plt.legend(title=legend_title)
    plt.show()


def do_plot(name,X, y, plt, plot_title, legend_title, legend_labels, pca_n_comp=20, alpha=.4, markersize=4, perp=30, n_iter=500):
    if name.lower()=='umap':
        u = umap.UMAP()
        res = u.fit_transform(X)
    elif name.lower()=='tsne':
        pca = PCA(n_components=pca_n_comp)
        pca_res = pca.fit_transform(X)
        tsne = TSNE(n_components=2, verbose=0, perplexity=perp, n_iter=n_iter)
        res = tsne.fit_transform(pca_res)
    elif name.lower()=='':
        res = X
    else:
        return None

    for yi in np.unique(y):
        plt.plot(np.array(res[:, 0])[y == yi], np.array(res[:, 1])[y == yi], 'o', label=legend_labels[yi], alpha=alpha,
                 markersize=markersize)
    plt.title('{} ({:.3f})'.format(plot_title, one_out_classification(res, y)))
    plt.xlabel('{} 1. komponenta'.format(name.upper()))
    plt.ylabel('{} 2. komponenta'.format(name.upper()))
    plt.legend(title=legend_title)
    return res

def canon_smiles(s):
    """
    Canonize a SMILES string.
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(s))

def load_models(model_files, weight_files, folder='./'):
    """
    Load autoencoder models from .h5 files (json can be given, but weights need to be given as well).
    :param model_files: Dictionary of files with models.
    :param weight_files: (optional) Dictionary of files with weights.
    :param folder: Location where files are saved.
    :return: Loaded models.
    """
    def binary_crossentropy_k(y_true, y_pred):
        return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

    models = dict()
    for k, v in model_files.items():
        models[k] = dict()
        for key, val in v.items():
            print(key)
            custom_objects = dict()
            if key[0] == 'V':
                custom_objects['CustomVariationalLayer'] = CustomVariationalLayer
            else:
                custom_objects['binary_crossentropy_k'] = binary_crossentropy_k
            custom_objects['latent_dim'] = int(re.findall('\d+', key)[0])

            model = load_model(folder + val, custom_objects=custom_objects)

            if key in weight_files:
                model.load_weights(folder + weight_files[key])
            if key[0] == 'V':
                models[k][key] = Model(input=model.layers[0].input, output=model.layers[13].output)
            else:
                models[k][key] = Model(input=model.layers[0].input, output=model.layers[1].get_output_at(1))

    return models

def save_vae(vae, folder_name='./'):
    """
    Save autoencoder to files.
    :param vae: Keras model.
    :param folder_name: Name of the folder.
    """
    vae.save(folder_name + 'model.h5')
    vae.save_weights(folder_name + 'weights.h5')
    vae.encoder.save_weights(folder_name + 'encoder.h5')
    vae.decoder.save_weights(folder_name + 'decoder.h5')

    model_json = vae.decoder.to_json()
    with open(folder_name + "decoder.json", "w") as json_file:
        json_file.write(model_json)

    model_json = vae.encoder.to_json()
    with open(folder_name + "encoder.json", "w") as json_file:
        json_file.write(model_json)

def one_out_classification(X, y, classifier=neighbors.KNeighborsClassifier(), scorer=roc_auc_score):
    """
    One out classification of the X based on y with used classifier and scorer.
    :return:
    """
    predictions = np.zeros(len(y))
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        predictions[test_index] = classifier.fit(X_train, y_train).predict(X_test)
    return scorer(y, predictions)

