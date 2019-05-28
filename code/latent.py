import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import cheauder_utils


def combine_dict(d1, d2):
    return {key: d1.get(key, 0) + d2.get(key, 0)
            for key in d1.keys()}


model_files = {
    0: {'Variacijski avtokodirnik (196, tanh)': 'model-vaetanh196.h5',
        'Avtokodirnik (196, tanh)': 'model-aetanh196.h5'},
    1: {
        'Variacijski avtokodirnik (196, RELU)': 'model-vaerelu196.h5',
        'Avtokodirnik (196, RELU)': 'model-aerelu196.h5',
        },
    2: {
        'Variacijski avtokodirnik (2, tanh)': 'model-vaetanh2.h5',
        'Avtokodirnik (2, tanh)': 'model-aetanh2.h5',
        },
}

weight_files = {
    'Avtokodirnik (196, tanh)': 'weights-aetanh196.h5'
}

f_lat = 'lat.png'
f_kde = 'kde.png'

def plot_kde(X, ax, title):
    from sklearn.neighbors import KernelDensity
    X_plot = np.linspace(X.min() - 1, X.max() + 1, 1000)
    for column in range(X.shape[1]):
        kde = KernelDensity(kernel='gaussian', bandwidth=(X.max() - X.min()) / 50).fit(
            X[:, column][:, np.newaxis])
        log_dens = kde.score_samples(X_plot[:, np.newaxis])
        ax.plot(X_plot, np.exp(log_dens), '-')
        ax.set_title(title)


if __name__ == '__main__':
    X, y, charset, chars = cheauder_utils.load_data_vae('data/250k_rndm_zinc_drugs_clean_3.csv', col_smiles=0,
                                                        col_target=1, delimiter=',', max_len=120)
    _, x_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=142)

    models = cheauder_utils.load_models(model_files, weight_files)
    chars, charset35 = cheauder_utils.load_charset('model/charset_ZINC35.json')

    # latentni prostor
    res = dict()
    f1, plts1 = plt.subplots(3,2, figsize=(15,15))
    f2, plts2 = plt.subplots(3,2, figsize=(15,15))
    for k,v in models.items():
        res[k] = dict()
        [(name1,en1), (name2,en2)] = v.items()

        encoded1 = en1.predict(X)
        encoded2 = en2.predict(X)

        from sklearn.decomposition import PCA
        j = 0
        for name, encoded in [(name1, encoded1), (name2, encoded2)]:
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(encoded)

            markersize = .8
            alpha = .6
            plts1[k][j].plot(pca_res[:,0], pca_res[:,1], 'o', alpha=alpha, markersize=markersize)
            plts1[k][j].set_title(name)
            plts1[k][j].set_axis_off()

            kde_graph = plot_kde(encoded, plts2[k][j], name)

            j += 1

    f1.savefig(f_lat)
    f2.savefig(f_kde)