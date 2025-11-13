from xml.dom import HierarchyRequestErr

from data_loader import load_and_prepare_data
from train_model import train_and_evaluate
from predict import predict_text

if __name__ == "__main__":
    print("Bezig met laden van data...")
    X_train, X_test, y_train, y_test, vectorizer = load_and_prepare_data()
    print("Klaar, Dataset succesvol geladen")
    print("Aantal trainingsvoorbeelden:", X_train.shape[0])
    print("Aantal testvoorbeelden:", X_test.shape[0])

    print("\nTraining en evaluatie starten...")
    best_model, metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    # Hier kunnen reviews worden toegevoegd voor voorspelling
    new_reviews = [
        "I absolutely loved this movie! Amazing story and acting.",
        "Worst movie ever. I hated it so much.",
    ]

    predictions = predict_text(metrics["model_path"], vectorizer, new_reviews)

    print("\nVoorspellingen voor nieuwe teksten:")
    for review, pred in zip(new_reviews, predictions):
        sentiment = "Positief" if pred == 1 else "Negatief"
        print(f"Review: {review}\nVoorspelling: {sentiment}\n")