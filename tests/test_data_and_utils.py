import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from src.data_loader import (
    get_feature_names,
    get_target_names,
    load_iris_as_dataframe,
    get_dataset_info,
    load_iris_data,
)
from src.model import IrisClassifier
from src.utils import plot_confusion_matrix, plot_feature_importance


def test_feature_and_target_names():
    """Les noms de features et de targets doivent correspondre à ceux d'Iris"""
    features = get_feature_names()
    targets = get_target_names()

    assert isinstance(features, list) or isinstance(features, (tuple,))
    assert len(features) == 4
    assert 'sepal length (cm)' in features

    assert isinstance(targets, (list, tuple, np.ndarray))
    assert len(targets) == 3
    assert 'setosa' in targets


def test_load_iris_as_dataframe():
    """Vérifier que le dataset en DataFrame a les colonnes attendues et 150 lignes"""
    df = load_iris_as_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] == 150
    # 4 features + target + species
    assert df.shape[1] == 6
    assert 'species' in df.columns
    # species values doivent correspondre aux target names
    assert set(df['species'].unique()) == set(get_target_names())


def test_get_dataset_info():
    """Vérifier la structure renvoyée par get_dataset_info"""
    info = get_dataset_info()
    assert isinstance(info, dict)
    assert info['n_samples'] == 150
    assert info['n_features'] == 4
    assert info['n_classes'] == 3
    # la somme des effectifs par classe doit être le nombre d'échantillons
    assert sum(info['class_distribution'].values()) == 150


def test_plot_functions_create_files(tmp_path, monkeypatch):
    """Vérifier que les fonctions de tracé sauvegardent bien des fichiers PNG"""
    # changer le répertoire courant vers tmp_path pour capturer les fichiers générés
    monkeypatch.chdir(tmp_path)

    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=0)
    clf = IrisClassifier()
    clf.train(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrix
    plot_confusion_matrix(y_test, y_pred, target_names=get_target_names())
    cm_path = tmp_path / 'confusion_matrix.png'
    assert cm_path.exists()
    assert cm_path.stat().st_size > 0

    # Feature importance
    plot_feature_importance(clf.model, get_feature_names())
    fi_path = tmp_path / 'feature_importance.png'
    assert fi_path.exists()
    assert fi_path.stat().st_size > 0
