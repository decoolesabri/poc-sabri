# from data_loader import load_and_prepare_data
#
# if __name__ == "__main__":
#     X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data()
#     print("Klaar! Dataset succesvol geladen 🎉")
#     print("Aantal trainingsvoorbeelden:", X_train.shape[0])
#     print("Aantal testvoorbeelden:", X_test.shape[0])

# src/main.py
from data_loader import load_and_prepare_data
from train_model import train_and_evaluate

if __name__ == "__main__":
    print("Bezig met laden van data...")
    X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data()
    print("Klaar! Dataset succesvol geladen 🎉")
    print("Aantal trainingsvoorbeelden:", X_train.shape[0])
    print("Aantal testvoorbeelden:", X_test.shape[0])

    print("\nTraining en evaluatie starten...")
    best_model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    print("\nResultaten:")
    print(" - Naive Bayes accuracy:", metrics["nb_accuracy"])
    print(" - Logistic Regression accuracy:", metrics["lr_accuracy"])
    print(" - Beste model:", metrics["best_model"])
    print(" - Beste accuracy:", metrics["best_accuracy"])
    print("\nClassification report (beste model):\n")
    print(metrics["classification_report"])
    print("\nModel opgeslagen op:", metrics["model_path"])
