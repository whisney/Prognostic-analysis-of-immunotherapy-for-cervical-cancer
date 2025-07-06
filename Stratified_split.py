import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import pickle

df = pd.read_excel('metadata/Response.xlsx')

ID_list = list(df['ID'])
label_list = list(df['Label'])

train_val_sfolder = StratifiedShuffleSplit(n_splits=1, test_size=0.3)

for train_index, val_index in train_val_sfolder.split(ID_list, label_list):
    train_ID = []
    val_ID = []

    for id in train_index:
        train_ID.append(ID_list[id])
    for id in val_index:
        val_ID.append(ID_list[id])

print(len(train_ID), len(val_ID))

with open(r'train_val_split.pkl', 'wb') as f:
    pickle.dump({'train': train_ID, 'val': val_ID}, f, pickle.HIGHEST_PROTOCOL)