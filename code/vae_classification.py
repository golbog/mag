from keras import layers
from keras.layers import Lambda, Input, Dense
from keras.models import Model, model_from_json
from keras.callbacks import ModelCheckpoint
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

from sklearn.manifold import TSNE, MDS
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from vae_smiles import CustomVariationalLayer

import cheauder_utils


with open("model/encoder_196_120x36.json", "r") as json_file:
    vae_json = json_file.read()

vae = model_from_json(vae_json, custom_objects={'CustomVariationalLayer': CustomVariationalLayer,
                                                'latent_dim': 196})
vae.load_weights("weights/encoder_weights.h5")

X, y, charset, chars = cheauder_utils.load_data('../data/BBBP.csv',
                                                charset_filename='model/charset_ZINC.json',
                                                col_smiles=3, col_target=2, delimiter=',', max_len=120)

Xvae, _, _ = vae.predict(X)

smiles,y = cheauder_utils.read_data('../data/BBBP.csv', col_smiles=3,col_target=2, delimiter=',')
moles = np.array([AllChem.MolFromSmiles(x) for x in smiles])

valid = [i for i,x in enumerate(moles) if x is not None]
y = np.array(y[valid], dtype=np.float)
moles = moles[valid]
Xvae = Xvae[valid]

fingerprints = [AllChem.GetMorganFingerprintAsBitVect(x, 4) for x in moles]

Xfinger = list()
for x in fingerprints:
    arr = np.zeros((0,))
    DataStructs.ConvertToNumpyArray(x, arr)
    Xfinger.append(arr)
Xfinger = np.array(Xfinger)

n_splits = 10
cv = StratifiedKFold(n_splits=n_splits)
svc = SVC(kernel='linear')
rfc = RandomForestClassifier(n_estimators=500, random_state=0)
kn = KNeighborsClassifier(n_neighbors =3)
classifiers = {'SVC': svc, 'RFC':rfc, 'KN':kn}
X_train, X_test, y_train, y_test = train_test_split(Xvae, y, test_size=0.2, random_state=322)
Xs = {'VAE': Xvae, 'Fingerprint': Xfinger}

for Xname, X in Xs.items():
    for cname, c in classifiers.items():
        val = 0
        for train, test in cv.split(Xvae, y):
            c.fit(X[train], y[train])
            y_pred = c.predict(X[test])
            val += roc_auc_score(y[test], y_pred)
        print("{} | {} : {}".format(Xname, cname, val/n_splits))

