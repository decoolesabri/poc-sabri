from numpy.random import multinomial
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_and_evaluate(X_train, X_test, y_train, y_test, model_dir="models"):

    # Model 1: Multinomial Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # Model 2: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    acc_lr = accuracy_score(y_test, y_pred_lr)

    # Beste model kiezen
    if acc_lr >= acc_nb:
        best_model = lr
        best_name = "logistic_regression"
        best_acc = acc_lr
        best_report = classification_report(y_test, y_pred_lr)
    else:
        best_model = nb
        best_name = "multinomial_nb"
        best_acc = acc_nb
        best_report = classification_report(y_test, y_pred_nb)

    # Zorg dat de map bestaat en sla het model op
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{best_name}.joblib")
    joblib.dump(best_model, model_path)

    metrics = {
        "nb_accuracy": acc_nb,
        "lr_accuracy": acc_lr,
        "best_model": best_name,
        "best_accuracy": best_acc,
        "classification_report": best_report,
        "model_path": model_path
    }

    return best_model, metrics