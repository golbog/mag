**010-data.ipnyb** - Predstavitev zbranih podatkov (dimenzionalnost, tip razredne spremenljivke, statistika zapisov SMILESS) in predstavitev notacije SMILES.

**020-fingerprints.ipnyb** - Kratek opis skupin prstnih odtisev molekulskih struktur in prikaz uporabe v RDkitu. Prikaz iskajne skupnih podstruktur v množici molekul.

**030-new-drugs.ipnyb** - Prikaz enega izmed načinov ustvarjanja novih molekul s pomočjo variacijske avtokodirnika.

**040-vis.ipnyb** - TSNE in UMAP vizualizacije prstnih odtisov in vizualizacija 2d latentnega prostora avtokodirnika in variacijskega avtokodirnika. Kvantitivno vrednotenje (ROC AUC) razredov v vizualizacijah: leave-one-out in kNN v 2D prostoru značilk.

**050-klasifikacija.ipnyb** - Primerjanje klasifikacijske točnosti med prstnimi odtisi, variacijskimi in nevariacijskimi vložitvami ter smiselnimi kombinacijami.

**050-vae.ipnyb** - Prikaz uporabe variacijskega avtokodirnika in delež pravilno dekodiranih ter povprečno število napak.

**000-classification_test.ipnyb** - This notebook shows and tests our implementation of Gómez-Bombarelli et al. variational autoencoder. Tests include classification comparison with RDKit's fingerprints and TSNE plot of latent spaces on BBBP dataset. Classification quality is also tested and averaged over many different databases.