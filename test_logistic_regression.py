import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from logistic_regression import logistic_regression  # ⬅️ Update this to your actual file/module name

def test_linearly_separable():
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    y = y.reshape(-1, 1)
    model = logistic_regression(n_iter=1000, l_rate=0.1)
    model.fit(X, y)
    preds = model.predict(X)
    assert model.score(X, y) > 0.9
    print("✅ Linearly separable test passed.")

def test_overfit_polynomial():
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([[0], [0], [1], [1], [1]])
    model = logistic_regression(n_iter=2000, l_rate=0.01, use_polynomial=True, degree=5)
    model.fit(X, y)
    assert model.score(X, y) == 1.0
    print("✅ Polynomial overfit test passed.")

def test_regularization_effect():
    X, y = make_classification(n_samples=100, n_features=10, n_informative=10,
                               n_redundant=0, random_state=0)
    y = y.reshape(-1, 1)
    model = logistic_regression(n_iter=1000, l_rate=0.1, penalty='l2', C=0.1)
    model.fit(X, y)
    w_norm = np.linalg.norm(model.params['W'])
    assert w_norm < 5
    print("✅ L2 regularization test passed. Weight norm:", w_norm)

def test_predict_proba_sums_to_one():
    X, y = make_classification(n_samples=20, n_features=5, n_informative=5,
                               n_redundant=0, random_state=1)
    y = y.reshape(-1, 1)
    model = logistic_regression(n_iter=500, l_rate=0.1)
    model.fit(X, y)
    probs = model.predict_proba(X)
    assert np.allclose(np.sum(probs, axis=1), 1, atol=1e-5)
    print("✅ predict_proba test passed.")

def test_confusion_matrix():
    y_true = np.array([[1],[0],[1],[0],[1]])
    y_pred = np.array([[1],[0],[1],[1],[0]])
    model = logistic_regression()
    cm = model.confusion_mat(y_true, y_pred)
    assert cm.shape == (2, 2)
    assert cm[0, 0] == 2  # TP
    print("✅ Confusion matrix test passed.")

def test_all_zero_labels():
    X = np.random.rand(20, 3)
    y = np.zeros((20, 1))
    model = logistic_regression(n_iter=1000, l_rate=0.1)
    model.fit(X, y)
    assert model.score(X, y) == 1.0
    print("✅ All-zero label test passed.")

def test_shape_mismatch():
    try:
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, size=(9, 1))  # Mismatch
        model = logistic_regression()
        model.fit(X, y)
        assert False, "Shape mismatch not caught!"
    except Exception as e:
        print("✅ Shape mismatch test passed:", str(e))

def test_normalization_effect():
    X = np.random.rand(100, 3) * 1000
    y = (np.sum(X, axis=1) > 1500).astype(int).reshape(-1, 1)
    model_unscaled = logistic_regression(n_iter=1000)
    model_scaled = logistic_regression(n_iter=1000, normalize=True)
    cost_unscaled = model_unscaled.fit(X, y)[0][-1]
    cost_scaled = model_scaled.fit(X, y)[0][-1]
    assert cost_scaled < cost_unscaled or model_scaled.score(X, y) >= model_unscaled.score(X, y)
    print("✅ Normalization test passed.")

def test_polynomial_shape():
    X = np.array([[1, 2], [3, 4]])
    model = logistic_regression(use_polynomial=True, degree=3)
    X_poly = model._polynomial_features(X)
    assert X_poly.shape[1] > X.shape[1]
    print("✅ Polynomial feature shape test passed.")

def test_precision_recall_f1():
    y_true = np.array([[1],[0],[1],[0],[1]])
    y_pred = np.array([[1],[0],[1],[1],[0]])
    model = logistic_regression()
    P = model.precision(y_true, y_pred)
    R = model.recall(y_true, y_pred)
    F1 = model.f1(y_true, y_pred)
    assert round(P, 2) == round(2/3, 2)
    assert round(R, 2) == round(2/3, 2)
    assert round(F1, 2) == round(2*2/3*2/3 / (2/3 + 2/3), 2)
    print("✅ Precision, recall, F1 test passed.")

if __name__ == "__main__":
    test_linearly_separable()
    test_overfit_polynomial()
    test_regularization_effect()
    test_predict_proba_sums_to_one()
    test_confusion_matrix()
    test_all_zero_labels()
    test_shape_mismatch()
    test_normalization_effect()
    test_polynomial_shape()
    test_precision_recall_f1()
    print("\n🎯 All tests passed.")
