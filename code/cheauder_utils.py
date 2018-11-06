from rdkit.Chem import AllChem
from rdkit import DataStructs
import csv
import numpy as np

def is_valid_smiles(a):
    return AllChem.MolFromSmiles(a) is not None and a != '' and a is not None

def read_data(filename, col_smiles=0, col_target=1, start_row=1, delimiter=' ', quotechar='\''):
    smiles = list()
    targets = list()
    with open(filename) as file:
        reader = csv.reader(file, delimiter=delimiter, quotechar=quotechar)
        for _ in range(start_row): #testiraj
            next(reader)
        for row in reader:
            if len(row) > 2:
                row = np.array(row)
                smiles.append(row[col_smiles])
                targets.append(row[col_target])
    return np.array(smiles), np.array(targets)

def create_fingerprints(moles):
    fingerprints = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 4) for x in moles])

    X = list()
    for x in fingerprints:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(x, arr)
        X.append(arr)
    return np.array(X)

def create_charset(x):
    chars = list(set(x)) # all unique chars (random sorting)
    charset = {i:c for i,c in enumerate(chars)}
    charset[-1] = ' ' # add whitespace ------------ maybe -1 is not the best choice, maybe not even needed
    return chars, charset

def pad_smile(smile, max_len):
    if len(smile) < max_len:
        return smile + ' ' * (max_len - len(smile))
    return smile

def vectorize_smile(smile, charset, max_len=125):
    x = np.zeros((max_len, len(charset)))
    for i, char in enumerate(pad_smile(smile, max_len)):
        if char == ' ':
            continue
        x[i,charset[char]] = 1
    return x
