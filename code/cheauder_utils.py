import sys

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
from keras.models import Model, model_from_json
from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from keras import layers
from keras import backend as K

import csv
import json
import numpy as np

def load_data_vae(filename, charset_filename=None, col_smiles=0, col_target=1, start_row=1, delimiter=',', quotechar='\"', max_len=-1):
    X, y, label = read_data(filename, col_smiles, col_target, start_row, delimiter, quotechar)
    if charset_filename is None:
        chars, charset = create_charset(X)
    else:
        chars, charset = load_charset(charset_filename)
    if max_len == -1:
        max_len = max([len(x) for x in X]) + 1
    X = vectorize_smiles(X, charset, max_len)
    return X, y, charset, chars

def load_data(filename, encoder, charset_filename=None, col_smiles=0, col_target=1, start_row=1, delimiter=' ', quotechar='\"', max_len=-1):
    smiles, y, label = read_data(filename, col_smiles=col_smiles, col_target=col_target, delimiter=delimiter, start_row=start_row, quotechar=quotechar)

    # VAE
    if charset_filename is None:
        chars, charset = create_charset(smiles)
    else:
        chars, charset = load_charset(charset_filename)
    if max_len == -1:
        max_len = max([len(x) for x in smiles]) + 1
    Xvae, _ , _ = encoder.predict(vectorize_smiles(smiles, charset, max_len))

    # fingerprints
    moles = np.array([AllChem.MolFromSmiles(x) for x in smiles])

    valid = [i for i, x in enumerate(moles) if x is not None]
    moles = moles[valid]
    y = np.array(y[valid], dtype=np.float)
    Xvae = Xvae[valid] # also remove from encoded

    fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x, 4) for x in moles]
    Xfinger = list()
    for x in fingerprints:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(x, arr)
        Xfinger.append(arr)
    Xfinger = np.array(Xfinger)

    return Xvae, Xfinger, y, label, smiles


def smiles_to_fingerprints(smiles, y):
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
    return AllChem.MolFromSmiles(a) is not None and a != '' and a is not None

def read_data(filename, col_smiles=0, col_target=1, start_row=1, delimiter=' ', quotechar='\''):
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
                if smile[-1] == '\n':
                    smile = smile[:-1]
                smiles.append(smile)
                targets.append(row[col_target] if col_target != -1 else 0)
    return np.array(smiles), np.array(targets), label

def load_charset(charset_filename):
    with open(charset_filename, "r") as f:
        charset = json.load(f)
    chars = np.sort(list(charset.keys()))
    return chars, charset

def create_fingerprints(moles):
    fingerprints = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 4) for x in moles])
    X = list()
    for x in fingerprints:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(x, arr)
        X.append(arr)
    return np.array(X)

def create_charset(X):
    chars = list(set(''.join(x for x in X))) # all unique chars (random sorting)
    chars.append(' ')
    chars = np.sort(chars)
    charset = {c:i for i,c in enumerate(chars)}
    #with open('charset_ZINC.json', 'w') as fp:
    #    json.dump(charset, fp)
    return chars, charset

def pad_smiles(smiles, max_len):
    res = list()
    for smile in smiles:
        res.append(pad_smile(smile, max_len))
    return np.array(res)

def pad_smile(smile, max_len):
    if len(smile) < max_len:
        return smile + ' ' * (max_len - len(smile))
    return smile[:max_len]

def vectorize_smiles(smiles, charset, max_len=120):
    res = list()
    for smile in smiles:
        res.append(vectorize_smile(smile, charset, max_len))
    return np.array(res)

def vectorize_smile(smile, charset, max_len=120):
    x = np.zeros((max_len, len(charset)))
    for i, char in enumerate(pad_smile(smile, max_len)):
        #if char == ' ':
        #    continue
        try:
            x[i,charset[char]] = 1
        except KeyError:
            continue
    return x

def fp_to_numpy(X):
    Xfinger = list()
    for x in X:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(x, arr)
        Xfinger.append(arr)
    return np.array(Xfinger)

def devectorize_smiles(smiles, chars):
    return np.array([devectorize_smile(smile, chars) for smile in smiles])

def devectorize_smile(smile, chars):
    return ''.join([chars[char_index] for char_index in np.where(smile==1)[1]])

def load_coder_json(file_coder, file_weights, custom_objects):
    with open(file_coder, "r") as json_file:
        coder_json = json_file.read()

        coder = model_from_json(coder_json,
                                  custom_objects=custom_objects)
        coder.load_weights(file_weights)
        return coder

def plot_kde(X):
    from sklearn.neighbors import KernelDensity
    import matplotlib.pyplot as plt
    X_plot = np.linspace(X.min() - 1, X.max() + 1, 1000)
    fig, ax = plt.subplots()
    for column in range(X.shape[1]):
        kde = KernelDensity(kernel='gaussian', bandwidth=(X.max() - X.min()) / 50).fit(
            X[:, column][:, np.newaxis])
        log_dens = kde.score_samples(X_plot[:, np.newaxis])
        ax.plot(X_plot, np.exp(log_dens), '-')
    plt.show()

def correctly_decoded(X_decoded, X_real, chars):
    pred = devectorize_smiles(X_decoded, chars)
    real = devectorize_smiles(X_real, chars)

    num_correct = np.sum([np.array_equal(x, y) for x, y in zip(real, pred)])
    print("Percentage correctly predicted: {0:.2f}".format(100*num_correct/len(pred)))

    mean_error = np.sum([sum(1 for a, b in zip(x, y) if a != b) for x, y in zip(real, pred)]) / len(real)
    print("Mean error: {0:.2f}".format(mean_error))

def correctly_decoded_with_tries(X, X_real, decoder, chars, noise_norm=False, n=1000):
    correct = 0
    for ix, x in enumerate(X):
        print(ix)
        for i in range(n):
            pred = decoder.predict(np.array([perturb_z(x, noise_norm=False if i == 0 else noise_norm)]))[0]
            pred = np.array((pred.argmax(axis=1)[:, None] == np.arange(pred.shape[1])).astype(int))
            pred = devectorize_smile(pred, chars)
            real = devectorize_smile(X_real[ix], chars)
            if np.array_equal(real, pred):
                print("correct")
                correct += 1
                break
    print("Percentage correctly predicted: {0:.2f}".format(100 * correct / len(X)))

def classification_test(Xs, y, classifiers, n_splits=10, verbose=True):
    res = OrderedDict()
    cv = StratifiedKFold(n_splits=n_splits)

    Xs['Joined'] = np.hstack((list(Xs.values())))

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                cl = c.fit(X[train], y[train])
                val += roc_auc_score(y[test], cl.predict(X[test])) / n_splits
            res[(Xname, cname)] = val
            if verbose:
                print("{:11s} | {:6s} : {:.3f}".format(Xname, cname, val))
    return res

def perturb_z(z, noise_norm=False):
    if noise_norm > 0.0:
        noise_vec = np.random.normal(0, 1, size=z.shape)
        noise_vec = noise_vec / np.linalg.norm(noise_vec)
        return z + (noise_norm * noise_vec)
    return z

def to_csv(filename, smiles, X, y):
    with open(filename, 'w') as f:
        line = '{},{},{}\n'.format('smiles', ','.join('dim{}'.format(i) for i in range(len(X[0]))), 'y')
        f.write(line)
        for i in range(len(X)):
            line = '{},{},{}\n'.format(smiles[i], ','.join('{}'.format(x) for x in X[i]), y[i])
            f.write(line)
        return True

def plot_tsne_classification(X, y, title, legend_title, alpha=.3, markersize=4):
    y = np.array(y)

    pca = PCA(n_components=20)
    pca_res = pca.fit_transform(X)

    tsne = TSNE(n_components=2, verbose=0, perplexity=30, n_iter=500)
    res = tsne.fit_transform(pca_res)

    for yi in np.unique(y):
        plt.plot(np.array(res[:, 0])[y == yi], np.array(res[:, 1])[y == yi], 'o', label=str(yi), alpha=alpha,
                 markersize=markersize)
    plt.title(title)
    plt.xlabel('TSNE 1. component')
    plt.ylabel('TSNE 2. component')
    plt.legend(title=legend_title)
    plt.show()

def full_classification_test(classifiers, n_splits=10):
    ## placeholder
    sys.path.insert(0, '../code')
    from vae_smiles import CustomVariationalLayer
    encoder = load_coder_json("../code/model/encoder.json",
                                             "../code/weights/encoder.h5",
                                             custom_objects={'CustomVariationalLayer': CustomVariationalLayer,
                                                             'latent_dim': 196})
    res = OrderedDict()
    cv = StratifiedKFold(n_splits=n_splits)
    Xvae, Xfinger, y, label, smiles = load_data('../data/BBBP.csv', encoder,
                                                charset_filename='../code/model/charset_ZINC.json',
                                                col_smiles=3, col_target=2, delimiter=',',
                                                max_len=120)
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}
    Xs['Joined'] = np.hstack((list(Xs.values())))

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                c.fit(X[train], y[train])
                val += roc_auc_score(y[test], c.predict(X[test])) / n_splits
            res[(Xname, cname)] = val
            #print("{:10s} | {:6s} : {:.2f}".format(Xname, cname, val / n_splits))

    Xvae, Xfinger, y, label, smiles = load_data('../data/clintox.csv', encoder,
                                                charset_filename='../code/model/charset_ZINC.json',
                                                col_smiles=0, col_target=1, delimiter=',',
                                                max_len=120)
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}
    Xs['Joined'] = np.hstack((list(Xs.values())))

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                c.fit(X[train], y[train])
                val += roc_auc_score(y[test], c.predict(X[test])) / n_splits
            res[(Xname, cname)] += val
            # print("{:10s} | {:6s} : {:.2f}".format(Xname, cname, val / n_splits))

    Xvae, Xfinger, y, label, smiles = load_data('../data/clintox.csv', encoder,
                                                charset_filename='../code/model/charset_ZINC.json',
                                                col_smiles=0, col_target=2, delimiter=',',
                                                max_len=120)
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}
    Xs['Joined'] = np.hstack((list(Xs.values())))

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                c.fit(X[train], y[train])
                val += roc_auc_score(y[test], c.predict(X[test])) / n_splits
            res[(Xname, cname)] += val
            # print("{:10s} | {:6s} : {:.2f}".format(Xname, cname, val / n_splits))

    Xvae, Xfinger, y, label, smiles = load_data('../data/sider.csv', encoder,
                                                charset_filename='../code/model/charset_ZINC.json',
                                                col_smiles=0, col_target=1, delimiter=',',
                                                max_len=120)
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}
    Xs['Joined'] = np.hstack((list(Xs.values())))

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                c.fit(X[train], y[train])
                val += roc_auc_score(y[test], c.predict(X[test])) / n_splits
            res[(Xname, cname)] += val
            # print("{:10s} | {:6s} : {:.2f}".format(Xname, cname, val / n_splits))

    Xvae, Xfinger, y, label, smiles = load_data('../data/sider.csv', encoder,
                                                charset_filename='../code/model/charset_ZINC.json',
                                                col_smiles=0, col_target=2, delimiter=',',
                                                max_len=120)
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}
    Xs['Joined'] = np.hstack((list(Xs.values())))

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                c.fit(X[train], y[train])
                val += roc_auc_score(y[test], c.predict(X[test])) / n_splits
            res[(Xname, cname)] += val
            # print("{:10s} | {:6s} : {:.2f}".format(Xname, cname, val / n_splits))

    Xvae, Xfinger, y, label, smiles = load_data('../data/sider.csv', encoder,
                                                charset_filename='../code/model/charset_ZINC.json',
                                                col_smiles=0, col_target=4, delimiter=',',
                                                max_len=120)
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}
    Xs['Joined'] = np.hstack((list(Xs.values())))

    for Xname, X in Xs.items():
        for cname, c in classifiers.items():
            val = 0
            for train, test in cv.split(X, y):
                c.fit(X[train], y[train])
                val += roc_auc_score(y[test], c.predict(X[test])) / n_splits
            res[(Xname, cname)] += val
            # print("{:10s} | {:6s} : {:.2f}".format(Xname, cname, val / n_splits))


    for key, val in res.items():
        res[key] = val / 6

    for key, val in res.items():
        print("{:11s} | {:6s} : {:.3f}".format(key[0], key[1], val))

    return res

def one_out_classification(X, y, classifier=neighbors.KNeighborsClassifier(), scorer=roc_auc_score):
    predictions = np.zeros(len(y))
    loo = LeaveOneOut()
    for train_index, test_index in loo.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        predictions[test_index] = classifier.fit(X_train, y_train).predict(X_test)
    return scorer(y, predictions)

def do_plot(name,X, y, plt, title, legend, pca_n_comp=20, alpha=.4, markersize=4, perp=30, n_iter=500):
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
        plt.plot(np.array(res[:, 0])[y == yi], np.array(res[:, 1])[y == yi], 'o', label=str(yi), alpha=alpha,
                 markersize=markersize)
    plt.title(title)
    plt.xlabel('UMAP 1. komponenta')
    plt.ylabel('UMAP 2. komponenta')
    plt.legend(title=legend)
    return res


if __name__ == '__main__':
    sys.path.insert(0, '../code')
    from vae_smiles import CustomVariationalLayer
    from sklearn.manifold import TSNE, MDS
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split, StratifiedKFold

    #smiles, target, label = read_data('../data/Lipophilicity.csv',col_smiles=2, col_target=1, delimiter=',')
    #smiles, target, label = read_data('../data/SAMPL.csv',col_smiles=1, col_target=2, delimiter=',', quotechar='\"')
    #smiles, target, label = read_data('../data/ESOL.csv',col_smiles=9, col_target=1, delimiter=',', quotechar='\"')
    smiles, target, label = read_data('../data/250k_rndm_zinc_drugs_clean_3.csv',col_smiles=0, col_target=[1,2,3], delimiter=',', quotechar='\"')
    target = target.astype(np.float)
    print('{:.2f} {:.2f}'.format(np.mean(target[:,0]), np.std(target[:,0])))
    print('{:.2f} {:.2f}'.format(np.mean(target[:,1]), np.std(target[:,1])))
    print('{:.2f} {:.2f}'.format(np.mean(target[:,2]), np.std(target[:,2])))
    exit()
    encoder = load_coder_json("../code/model/encoder.json",
                              "../code/weights/encoder.h5",
                              custom_objects={'CustomVariationalLayer': CustomVariationalLayer,
                                              'latent_dim': 196})
    """
    encoder = load_coder_json("../code/model/1encoder_196_120x36.json",
                                             "../code/weights/1encoder_weights.h5",
                                             custom_objects={'CustomVariationalLayer': CustomVariationalLayer,
                                                             'latent_dim': 196})

    decoder = load_coder_json("../code/model/decoder.json",
                                             "../code/weights/decoder.h5",
                                             custom_objects={'CustomVariationalLayer': CustomVariationalLayer,
                                                             'latent_dim': 196})

    Xsmi, _, charset, chars = load_data_vae('../data/version.smi',
                                                           charset_filename='../code/model/charset_ZINC.json',
                                                           col_smiles=0, col_target=-1, delimiter=' ',
                                                           max_len=120)

    Xsmi = Xsmi[np.random.choice(len(Xsmi), size=100, replace=False)]
    z_mean, _, _ = encoder.predict(Xsmi, batch_size=500)
    correctly_decoded_with_tries(z_mean, Xsmi, decoder, chars, noise_norm=5., n=1000)
    

    Xvae, Xfinger, y, label, smiles = load_data('../data/BBBP.csv', encoder,
                                                               charset_filename='../code/model/charset_ZINC.json',
                                                               col_smiles=3, col_target=2, delimiter=',',
                                                               max_len=120)

    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits)
    svc = SVC(kernel='linear')
    rfc = RandomForestClassifier(n_estimators=500, random_state=0)
    classifiers = {'SVM': svc, 'RFC': rfc}
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}

    classification_test(Xs, y, classifiers, n_splits=10)
    exit()
"""

    Xvae, Xfinger, y, label, smiles = load_data('../data/BBBP.csv', encoder,
                                                charset_filename='../code/model/charset_ZINC.json',
                                                col_smiles=3, col_target=2, delimiter=',',
                                                max_len=120)

    from xgboost import XGBClassifier
    cv = StratifiedKFold(n_splits=10)
    n_splits = 10
    cv = StratifiedKFold(n_splits=n_splits)
    svc = SVC(kernel='linear')
    rfc = RandomForestClassifier(n_estimators=500, random_state=0)
    classifiers = {'SVM': svc, 'RFC': rfc, 'XGBoost': XGBClassifier()}
    classifiers = {'XGBoost': XGBClassifier()}
    X_train, X_test, y_train, y_test = train_test_split(Xvae, y, test_size=0.2, random_state=322)
    Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}
    classification_test(Xs, y, classifiers, n_splits=10)