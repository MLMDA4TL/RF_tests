import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split


def categorical_encoder(X):
    """
    Transforming categorical features into numerical using One-Hot method
    """
    enc_label = LabelEncoder()
    enc_onehot = OneHotEncoder()
    # transform categorical labels into integers
    for i in range(X.shape[1]):
        X[:, i] = enc_label.fit_transform(X[:, i])
        # print("labels : ", enc_label.classes_)
    X = X.astype('int')
    # encoding integer labels using one-hot
    X = enc_onehot.fit_transform(X).toarray()
    # print("nb_labels : ", enc_onehot.n_values_)
    return X, enc_onehot.n_values_


def load_letter():
    # load data
    df = pd.read_csv("data/letter/letter-recognition.data.txt",
                     sep=',', header=None)
    # constructing numerical labels
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label = {letter: i for i, letter in enumerate(letters)}
    df.iloc[:, 0] = [label[l] for l in df.iloc[:, 0]]
    # separating source & target
    X_source = df[df[9] < np.median(df[9])]
    X_target = df[df[9] >= np.median(df[9])]
    # separating 5% & 95% of target data, stratified, random
    X_target_095, X_target_005, y_target_095, y_target_005 = train_test_split(
        X_target.iloc[:, 1:],
        X_target.iloc[:, 0],
        test_size=0.05,
        stratify=X_target.iloc[:, 0])
    return [X_source.iloc[:, 1:].values, X_target_005.values, X_target_095.values,
            X_source.iloc[:, 0].values, y_target_005.values, y_target_095.values]


def load_mushroom():
    df = pd.read_csv("data/mushroom/agaricus-lepiota.data.txt", sep=',',
                     header=None)
    # re-labelling
    df.iloc[:, 0][df.iloc[:, 0] == 'p'] = 0
    df.iloc[:, 0][df.iloc[:, 0] == 'e'] = 1
    # separating y from X
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    # transforming categorical labels into numerical
    X, nb_labels = categorical_encoder(X)
    # retrieving index of feature 'stak-shape' with value 't'
    ind_stalk_shape_t = np.sum(nb_labels[:10]) - 1
    ind_s = X[:, ind_stalk_shape_t] == 1
    ind_t = X[:, ind_stalk_shape_t] == 0
    # concatenate X and y
    y = y.reshape((-1, 1))
    X = np.concatenate((y, X), axis=1).astype(int)
    # separating source & target
    X_source = X[ind_s]
    X_target = X[ind_t]
    # selecting only labeled 0
    l_0 = X_target[X_target[:, 0] == 0]
    # selecting only labeled 1
    l_1 = X_target[X_target[:, 0] == 1]
    # 95% of labeled 0
    l_0_095 = l_0[int(0.05 * l_0.shape[0]):, :]
    # 95% of labeled 1
    l_1_095 = l_1[int(0.05 * l_0.shape[0]):, :]
    # concatenate
    X_target_095 = np.concatenate((l_0_095, l_1_095))
    # 5% of labeled 0
    l_0_005 = l_0[:int(0.05 * l_0.shape[0]), :]
    # 5% of labeled 1
    l_1_005 = l_1[:int(0.05 * l_0.shape[0]), :]
    # concatenate
    X_target_005 = np.concatenate((l_0_005, l_1_005))
    # separating y from X
    y_source = X_source[:, 0]
    X_source = X_source[:, 1:]
    y_target_005 = X_target_005[:, 0]
    X_target_005 = X_target_005[:, 1:]
    y_target_095 = X_target_095[:, 0]
    X_target_095 = X_target_095[:, 1:]
    return [X_source, X_target_005, X_target_095,
            y_source, y_target_005, y_target_095]


def load_wine():
    # loading data (source & target already separated)
    df_target = pd.read_csv("data/wine_uminho/winequality-red.csv",
                            sep=';', header=0)
    df_source = pd.read_csv("data/wine_uminho/winequality-white.csv",
                            sep=';', header=0)
    # separating 5% & 95% of target data, stratified, random
    X_target_095, X_target_005, y_target_095, y_target_005 = train_test_split(
            df_target.iloc[:, :-1],
            df_target.iloc[:, -1],
            test_size=0.05,
            stratify=df_target.iloc[:, -1])
    return [df_source.iloc[:, :-1].values, X_target_005.values,
            X_target_095.values, df_source.iloc[:, -1].values,
            y_target_005.values, y_target_095.values]


if __name__ == "__main__":
    print("test")
    # X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_mushroom()
    # X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_wine()
    # X_source, X_target_005, X_target_095, y_source, y_target_005, y_target_095 = load_letter()
