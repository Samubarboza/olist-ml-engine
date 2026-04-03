import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, StratifiedKFold
from src.config.settings import RANDOM_STATE, CV_FOLDS, MODELS_DIR


CHURN_MODELS = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "RandomForest": RandomForestClassifier(random_state=RANDOM_STATE),
    "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM": CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=RANDOM_STATE)),
    "KNN": KNeighborsClassifier(),
    "NaiveBayes": GaussianNB()
}


def train_and_validate(X_train, y_train, models=None, cv_folds=None,
                       save_models=True, models_dir=None):
    """
    Entrena modelos de clasificacion y los valida con StratifiedKFold cross-validation.
    Retorna los modelos entrenados (fit en todo el train set) y los resultados de CV.
    """
    if models is None:
        models = CHURN_MODELS
    if cv_folds is None:
        cv_folds = CV_FOLDS
    if models_dir is None:
        models_dir = MODELS_DIR

    os.makedirs(models_dir, exist_ok=True)

    cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    scoring_metrics = ["accuracy", "precision", "recall", "f1"]

    trained_models = {}
    cv_results = {}

    for name, model in models.items():
        print(f"  Entrenando {name}...")

        # cross-validation para estimar rendimiento
        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=cv_strategy,
            scoring=scoring_metrics,
            return_train_score=True
        )

        cv_results[name] = {
            "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
            "cv_accuracy_std": cv_scores["test_accuracy"].std(),
            "cv_precision_mean": cv_scores["test_precision"].mean(),
            "cv_precision_std": cv_scores["test_precision"].std(),
            "cv_recall_mean": cv_scores["test_recall"].mean(),
            "cv_recall_std": cv_scores["test_recall"].std(),
            "cv_f1_mean": cv_scores["test_f1"].mean(),
            "cv_f1_std": cv_scores["test_f1"].std(),
            "train_accuracy_mean": cv_scores["train_accuracy"].mean(),
            "train_f1_mean": cv_scores["train_f1"].mean()
        }

        # entrenar modelo final con todo el set de entrenamiento
        model.fit(X_train, y_train)
        trained_models[name] = model

        if save_models:
            model_path = os.path.join(models_dir, f"{name}.pkl")
            joblib.dump(model, model_path)

    cv_results_df = pd.DataFrame(cv_results).T
    return trained_models, cv_results_df
