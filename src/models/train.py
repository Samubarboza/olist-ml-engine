import os
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def train_models(X_train, y_train, X_test=None, y_test=None, save_models=True, models_dir="models"):
    """
    Entrena todos los modelos obligatorios de churn.
    
    Args:
        X_train, y_train: datos de entrenamiento
        X_test, y_test: datos de test para evaluación opcional
        save_models: si True guarda los modelos en models_dir
        models_dir: carpeta donde guardar los modelos
    
    Returns:
        trained_models: diccionario con modelos entrenados
        results: diccionario con métricas (solo si X_test y y_test se pasan)
    """
    os.makedirs(models_dir, exist_ok=True)
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier(),
        "SVM": SVC(probability=True),
        "KNN": KNeighborsClassifier(),
        "NaiveBayes": GaussianNB()
    }

    trained_models = {}
    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

        if save_models:
            joblib.dump(model, os.path.join(models_dir, f"{name}.pkl"))

        # si hay datos de test, calcular métricas
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred)
            }

    return trained_models, results