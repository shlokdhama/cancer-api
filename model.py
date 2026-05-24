import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import os

def train_and_save():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    data = pd.read_csv(os.path.join(BASE_DIR, 'data.csv'), index_col=0)
    labels = pd.read_csv(os.path.join(BASE_DIR, 'labels.csv'))

    X = data.to_numpy()
    y = labels['Class'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=200)
    X_train_pca = pca.fit_transform(X_train_scaled)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_pca, y_train)

    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(BASE_DIR, 'pca.pkl'), 'wb') as f:
        pickle.dump(pca, f)
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'wb') as f:
        pickle.dump(model, f)

    print("Model, scaler, and PCA saved.")

def load_model():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'pca.pkl'), 'rb') as f:
        pca = pickle.load(f)
    with open(os.path.join(BASE_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    return scaler, pca, model

if __name__ == "__main__":
    train_and_save()