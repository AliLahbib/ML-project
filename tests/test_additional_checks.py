import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import numpy as np
from src.data_loader import load_iris_data
from src.model import IrisClassifier


def test_untrained_model_raises():
    """predict, evaluate et save doivent lever ValueError si le modèle n'est pas entraîné"""
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=0)
    clf = IrisClassifier()

    with pytest.raises(ValueError):
        clf.predict(X_test[:5])

    with pytest.raises(ValueError):
        clf.evaluate(X_test, y_test)

    with pytest.raises(ValueError):
        # save_model doit lever si is_trained == False
        clf.save_model('models/should_not_create.pkl')


def test_stratify_preserved():
    """Vérifie que la répartition des classes est préservée par le split (stratify)"""
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=1)

    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)

    # Pour Iris, avec test_size=0.2 on s'attend à 10 échantillons par classe dans le test
    assert len(unique_train) == 3
    assert len(unique_test) == 3
    assert sorted(counts_test.tolist()) == [10, 10, 10]
    assert sorted(counts_train.tolist()) == [40, 40, 40]


def test_save_model_default_path_creates_file(tmp_path, monkeypatch):
    """Sauvegarde via le chemin par défaut doit créer models/iris_classifier.pkl dans le cwd"""
    monkeypatch.chdir(tmp_path)

    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.2, random_state=0)
    clf = IrisClassifier()
    clf.train(X_train, y_train)

    clf.save_model()  # utilise 'models/iris_classifier.pkl' par défaut

    save_path = tmp_path / 'models' / 'iris_classifier.pkl'
    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_evaluate_after_load_consistent(tmp_path):
    """Après sauvegarde et chargement, l'évaluation (accuracy) doit rester la même"""
    X_train, X_test, y_train, y_test = load_iris_data(test_size=0.25, random_state=2)
    clf = IrisClassifier()
    clf.train(X_train, y_train)

    save_path = tmp_path / 'm.pkl'
    clf.save_model(str(save_path))

    new_clf = IrisClassifier()
    new_clf.load_model(str(save_path))

    acc1, _ = clf.evaluate(X_test, y_test)
    acc2, _ = new_clf.evaluate(X_test, y_test)

    assert pytest.approx(acc1, rel=1e-9) == acc2

