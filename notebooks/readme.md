**010-data_vizualization.ipynb** - This notebook shows basic implementation of reading data, changing SMILES strings into fingerprints and applying PCA and then t-SNE or MDS. This data is then visualized in a graph.

**020-data_classification.ipynb** - This notebook classifies fingerprints with random forest classifier and shows ROC AUC values of classification of a seperate test set.

**030-study_replication.ipnyb** - This notebook test our implementation of Gómez-Bombarelli et al. variational autoencoder. Tests include percentage of correctly decoded and mean error, classification comparison with RDKit's fingerprints and TSNE plot of latent space.