from rdkit.Chem import AllChem
from rdkit import DataStructs
import csv
import json
import numpy as np

def load_data(filename, charset_filename=None, col_smiles=0, col_target=1, start_row=1, delimiter=' ', quotechar='\"', max_len=-1):
    X, y = read_data(filename, col_smiles, col_target, start_row, delimiter, quotechar)
    if charset_filename is None:
        chars, charset = create_charset(X)
    else:
        chars, charset = load_charset(charset_filename)
    if max_len == -1:
        max_len = max([len(x) for x in X]) + 1
    X = vectorize_smiles(X, charset, max_len)
    return X, y, charset, chars

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
                smile = row[col_smiles]
                if smile[-1] != '\n':
                    smile += '\n'
                smiles.append(smile)
                targets.append(row[col_target])
    return np.array(smiles), np.array(targets)

def load_charset(charset_filename):
    with open(charset_filename, "r") as f:
        charset = json.load(f)
    chars = np.sort(list(charset.values()))
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

def vectorize_smiles(smiles, charset, max_len=125):
    res = list()
    for smile in smiles:
        res.append(vectorize_smile(smile, charset, max_len))
    return np.array(res)

def vectorize_smile(smile, charset, max_len=125):
    x = np.zeros((max_len, len(charset)))
    for i, char in enumerate(pad_smile(smile, max_len)):
        #if char == ' ':
        #    continue
        try:
            x[i,charset[char]] = 1
        except KeyError:
            continue
    return x

def devectorize_smiles(smiles, chars):
    return np.array([devectorize_smile(smile, chars) for smile in smiles])

def devectorize_smile(smile, chars):
    return ''.join([chars[char_index] for char_index in np.where(smile==1)[1]])


if __name__ == '__main__':
    X,y,charset,chars = load_data('../data/250k_rndm_zinc_drugs_clean_3small.csv', col_smiles=0, col_target=1, delimiter=',', max_len=120)
    pass